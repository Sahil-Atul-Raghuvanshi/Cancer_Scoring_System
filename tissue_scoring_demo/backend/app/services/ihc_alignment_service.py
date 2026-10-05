"""Step 12 - carry step 11's invasive boundary onto the IHC slide.

    report.json        the regions in both slides' coordinates, plus every
                       measurement the gate was decided on
    he_borders.png     the H&E with those regions outlined
    ihc_borders.png    the IHC with the warped regions outlined
    invasive1..3.png   each region cropped from the IHC slide
    he_invasive1..3.png  the same region cropped from the H&E, to compare against
    valis/             VALIS's own working directory, including the registrar
                       pickle that makes a second pass a reload rather than a
                       re-registration

Job-shaped, like steps 2, 8 and 11 and for the same reason: a registration is
minutes of CPU in another process, and a request that waits for it is a
request that times out.

**What crosses is the pixel boundary, and the tile staircase is refused.** Until
step 11 existed this step read step 9's per-class regions and carried the
largest few of them - squares of 224 um windows drawn around tumour. It now
reads `roi_refinement_service.regions()`, which is what BEETLE traced inside the
regions a person ticked on step 10, and it *raises* when that is absent rather
than falling back. The fallback is what the refusal is for: a square ROI and a
pixel ROI produce coherent-looking scores on different denominators, and nothing
downstream of here could tell which one it had been given.

**Nothing here decides that an alignment is good.** The gate can only refuse;
it cannot approve. A registration that passes every threshold is still marked
unconfirmed until a person has looked at the two panels and said so, because
the failure this step has to avoid - a mask on plausible-looking but wrong
tissue - is one that numbers alone have never reliably caught.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.step08_tissue_type_segmentation.classes import SCORED
from app.pipeline.step09_roi_mask.borders import class_regions
from app.pipeline.step09_roi_mask.crops import crop_png, padded_bbox, region_bbox_level0
from app.pipeline.step12_ihc_alignment import overlay
from app.pipeline.step12_ihc_alignment import regions as region_tools
from app.registration import gate, valis_align

# `slide_registration/` is not a package on the backend's path; it is a sibling tree that
# owns the registration decided in advance. Added here rather than installed, because it is
# this project's own code and not a dependency.
_SCRIPTS = Path(__file__).resolve().parents[4] / "slide_registration"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import transform_warp as stored_transforms  # noqa: E402
from app.registration.exceptions import (
    RegistrationError,
    RegistrationRefused,
    RegistrationUnavailable,
)
from app.schemas.ihc_alignment import (
    AlignedRegion,
    AlignmentDiagnostics,
    AlignmentReport,
    AlignmentRun,
)
from app.services.roi_refinement_service import RefinementError, roi_refinement_service
from app.services.roi_selection_service import roi_selection_service
from app.services.roi_service import roi_service
from app.services.tissue_type_service import tissue_type_service
from app.services.upload_service import resolve_ready_path

logger = get_logger(__name__)

#: Longest edge of a region crop, matching step 9's own crops so the two
#: screens show the same tissue at the same size.
CROP_MAX_SIZE = 1200

#: Which regions this step carries across: `"refined"` (step 11's BEETLE pixel boundaries,
#: the default and the only one a score should normally be built on) or `"tiles"` (step
#: 10's coarse squares, skipping step 11 entirely).
#:
#: **An opt-in with a loud name, never a fallback.** The module docstring explains why a
#: silent substitution is forbidden: the two are different denominators and a report built
#: on either looks identical. This exists so a cohort can be scored *both* ways and the
#: difference measured in score units - which is the only way to find out what step 11's
#: refinement is actually worth, and step 11 is by far the most expensive stage in the
#: pipeline. The source travels on every region and is recorded on the report, so the two
#: sets of numbers can always be told apart afterwards.
#:
#: Set by `run_case_scores.py --roi-source tiles`. Nothing sets it from the API.
roi_source: str = "refined"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AlignmentError(ValueError):
    """A client-correctable problem: a missing step, or a slide that is not ready."""


@dataclass
class Job:
    """Live state of one registration, mutated by the worker, read by the API."""

    he_upload_id: str
    ihc_upload_id: str
    state: str = "queued"
    message: str | None = None
    started: float = field(default_factory=time.monotonic)
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    duration: float | None = None
    error: str | None = None


class IhcAlignmentService:
    """Registers a case's two slides and moves the ROI between them."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # --- storage ------------------------------------------------------------

    def key(self, he_upload_id: str, ihc_upload_id: str) -> str:
        return f"{he_upload_id}__{ihc_upload_id}"

    def _dir(self, he_upload_id: str, ihc_upload_id: str) -> Path:
        directory = settings.ihc_alignment_dir / self.key(he_upload_id, ihc_upload_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _path(self, he_upload_id: str, ihc_upload_id: str, name: str) -> Path:
        return self._dir(he_upload_id, ihc_upload_id) / name

    #: How the transform in a stored report was found. A report produced by a different
    #: method is not a cache hit - it is a different answer to the same question.
    REGISTRATION_KIND = "intensity"

    #: Bumped whenever the gate's rules or its measurements change. A stored **refusal**
    #: is only reusable while the rule that produced it still stands.
    #:
    #: This exists because deleting two files was not the fix. CAN_00865 was refused on a
    #: tissue-area ratio measured by the wrong rule; the measurement was corrected, the
    #: case re-run - and the *cached refusal* was served straight back, because it was
    #: produced by the right registration method and the method is all the other guard
    #: checks. Changing a threshold or a measurement has no effect on any pair already
    #: refused unless something invalidates those refusals, and nothing did.
    #:
    #: 1: the original feature-based gate
    #: 2: intensity gate - NMI, folding, tissue ratio, round trip
    #: 3: tissue-area ratio measured by optical density instead of colour saturation
    #: 4: that ratio measured with one cut per case, so both sides are comparable
    #: 5: the two near-blank CD44 pairs allowed past the gate with their failures kept
    #:    as caveats (`GATE_OVERRIDES`), which changes the outcome for those pairs and so
    #:    has to invalidate the refusals version 4 cached for them
    GATE_VERSION = 5

    def _read_report(self, he_upload_id: str, ihc_upload_id: str) -> AlignmentReport | None:
        path = self._path(he_upload_id, ihc_upload_id, "report.json")
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            report = AlignmentReport.model_validate(payload)
        except (OSError, ValueError):
            return None

        # **A report from a superseded registration method is discarded, not reused.**
        #
        # This is not hypothetical tidiness. When step 12 was switched from feature
        # matching to a stored intensity transform, twelve reports from the old method
        # were still on disk - and because a cached report short-circuits the whole step,
        # they were served straight back. Four of them said `ready`, so four pairs were
        # scored against a VALIS transform while the run believed it was using the
        # approved one, and the refusals carried VALIS's wording into a run that no
        # longer uses VALIS. Nothing in the output said which method any row came from.
        #
        # The check is on the diagnostics the report itself carries, so it cannot drift
        # from what actually produced it.
        # Both spellings: the model serialises with camelCase aliases, and reading only
        # the snake_case name is how this guard first came to reject every report it was
        # written to accept - a check that always fails is as useless as one that never
        # does, and costs more to find.
        stored = payload.get("diagnostics") or {}
        kind = stored.get("registrationKind") or stored.get("registration_kind")

        # **Any report, ready or refused, is only as good as the rule and the transform
        # that produced it (P-15).** Ready reports used to be exempt on the reasoning that
        # their transform had not changed - but nothing checked that, and reports on disk
        # carried gate versions None, 3 and 4 and some had no method at all. So both are
        # checked on every read: the gate version, and a fingerprint of everything the
        # warp reads (`transform_warp.fingerprint`). Either one moving is a miss.
        version = stored.get("gateVersion") or stored.get("gate_version") or 0
        if version != self.GATE_VERSION:
            logger.info(
                "re-deciding a stored %s alignment for %s -> %s: it was made by gate "
                "version %s and this is version %s",
                payload.get("state"),
                he_upload_id,
                ihc_upload_id,
                version,
                self.GATE_VERSION,
            )
            return None
        recorded = stored.get("transformSha256") or stored.get("transform_sha256")
        current = stored_transforms.fingerprint(he_upload_id, ihc_upload_id)
        if recorded != current:
            logger.info(
                "re-deciding a stored alignment for %s -> %s: the stored transform has "
                "changed since it was made",
                he_upload_id,
                ihc_upload_id,
            )
            return None
        if kind != self.REGISTRATION_KIND:
            logger.info(
                "discarding a stored alignment for %s -> %s: it was produced by '%s' and "
                "this step now uses '%s'",
                he_upload_id,
                ihc_upload_id,
                kind or "an earlier feature-based method",
                self.REGISTRATION_KIND,
            )
            return None
        return report

    def _write_report(self, report: AlignmentReport) -> None:
        path = self._path(report.he_upload_id, report.ihc_upload_id, "report.json")
        path.write_text(json.dumps(report.model_dump(by_alias=True), indent=2), encoding="utf-8")

    # --- the job ------------------------------------------------------------

    def available(self) -> bool:
        """Whether a registration could run here at all."""
        return valis_align.available()

    def start(
        self, he_upload_id: str, ihc_upload_id: str, *, restart: bool = False
    ) -> AlignmentRun:
        """Queue a registration, or hand back the finished one."""
        if he_upload_id == ihc_upload_id:
            raise AlignmentError(
                "the H&E and IHC slides are the same upload - there is nothing to align"
            )

        resolve_ready_path(upload_id=he_upload_id)
        resolve_ready_path(upload_id=ihc_upload_id)

        if not restart:
            existing = self._read_report(he_upload_id, ihc_upload_id)
            if existing is not None:
                return AlignmentRun(
                    he_upload_id=he_upload_id,
                    ihc_upload_id=ihc_upload_id,
                    state=existing.state,
                    finished_at=existing.generated_at,
                )

        key = self.key(he_upload_id, ihc_upload_id)
        with self._lock:
            job = self._jobs.get(key)
            if job is not None and job.state in {"queued", "running"}:
                return self._as_run(job)
            self._jobs[key] = Job(he_upload_id=he_upload_id, ihc_upload_id=ihc_upload_id)

        return self._as_run(self._jobs[key])

    def state(self, he_upload_id: str, ihc_upload_id: str) -> AlignmentRun:
        key = self.key(he_upload_id, ihc_upload_id)
        job = self._jobs.get(key)
        if job is not None:
            return self._as_run(job)

        report = self._read_report(he_upload_id, ihc_upload_id)
        if report is not None:
            return AlignmentRun(
                he_upload_id=he_upload_id,
                ihc_upload_id=ihc_upload_id,
                state=report.state,
                finished_at=report.generated_at,
            )
        raise AlignmentError("no alignment has been started for this pair")

    def _as_run(self, job: Job) -> AlignmentRun:
        return AlignmentRun(
            he_upload_id=job.he_upload_id,
            ihc_upload_id=job.ihc_upload_id,
            state=job.state,
            message=job.message,
            started_at=job.started_at,
            finished_at=job.finished_at,
            duration=job.duration,
            error=job.error,
        )

    def execute(self, he_upload_id: str, ihc_upload_id: str) -> None:
        """Run the queued registration. Called on a worker thread, never awaited."""
        key = self.key(he_upload_id, ihc_upload_id)
        job = self._jobs.get(key)
        if job is None or job.state != "queued":
            return

        job.state = "running"
        job.message = "registering the two slides"
        started = time.monotonic()

        try:
            report = self._build(he_upload_id, ihc_upload_id, job)
            job.state = report.state
        except RegistrationRefused as refused:
            report = self._refusal(he_upload_id, ihc_upload_id, refused)
            job.state = "refused"
            job.message = report.refusal_reasons[0] if report.refusal_reasons else "refused"
        except (AlignmentError, RegistrationError, RegistrationUnavailable) as failure:
            job.state = "failed"
            job.error = str(failure)
            logger.warning("alignment %s failed: %s", key, failure)
        except Exception as failure:  # noqa: BLE001 - a worker thread must not die silently
            job.state = "failed"
            job.error = f"{type(failure).__name__}: {failure}"
            logger.exception("alignment %s crashed", key)
        finally:
            job.finished_at = _now()
            job.duration = round(time.monotonic() - started, 1)

    # --- the work itself ----------------------------------------------------

    def _build(self, he_upload_id: str, ihc_upload_id: str, job: Job) -> AlignmentReport:
        he_path = resolve_ready_path(upload_id=he_upload_id)
        ihc_path = resolve_ready_path(upload_id=ihc_upload_id)

        # Step 9 must have run: `invasive_mm2` below is measured against its regions,
        # and it is what the coverage figure on this report is a share of. Asked for
        # rather than rebuilt - a step that quietly rebuilt the ROI it is about to move
        # could move a different one than the viewer approved on the previous screen.
        roi_service.report(he_upload_id)

        class_map = tissue_type_service.class_map(he_upload_id)
        invasive_mm2 = sum(
            region.area_mm2 for region in class_regions(class_map).get(SCORED, [])
        )

        selected = self._crossable(he_upload_id)
        reached = (
            sum(region.area_mm2 for region in selected) / invasive_mm2
            if invasive_mm2 > 0
            else 0.0
        )

        he_mpp, ihc_mpp = self._mpp(he_path), self._mpp(ihc_path)
        flat_rings, ring_counts = region_tools.densified_rings(selected, he_mpp=he_mpp)

        job.message = f"carrying {len(flat_rings)} rings across two slides"

        # **The stored transform, not a fresh VALIS registration.** Step 12 used to fit a
        # transform here from matched image features, which fails outright on a section
        # that is nearly unstained: there is nothing for a detector to key on, and three of
        # six cases returned 0 to 13 matched features however they were rotated, whatever
        # detector was used, at any resolution.
        #
        # Registration is now decided beforehand by `slide_registration/`, which fits
        # Mattes mutual information over a similarity transform - no features at all - and
        # stores one transform per pair. `mattes` was chosen over a better-scoring
        # free-form B-spline because a similarity transform has one Jacobian for the whole
        # section and therefore **cannot fold tissue through itself**; 10 of 16 B-spline
        # fits did exactly that, the worst folding 24% of a section while posting the
        # cohort's highest mutual information. See REGISTRATION-PLAN.md section 14.
        warped = stored_transforms.warp_for_pair(he_upload_id, ihc_upload_id, flat_rings)
        warped_rings = warped["rings_level0"]

        diagnostics = {
            # Names which family of checks the gate applies. An intensity-based fit has no
            # keypoints and no keypoint residual, and passing it through checks that read
            # those would be passing it on absent evidence.
            "registration_kind": "intensity",
            "gate_version": self.GATE_VERSION,
            # What the warp read, so a refit transform invalidates this report.
            "transform_sha256": stored_transforms.fingerprint(he_upload_id, ihc_upload_id),
            "method": warped.get("method"),
            "alignment_nmi": warped.get("nmi"),
            "round_trip_median_um": warped.get("round_trip_median_um"),
            "round_trip_max_um": warped.get("round_trip_max_um"),
            "folded_points": warped.get("folded_points", 0),
        }
        diagnostics.update(self._tissue_areas(he_upload_id, ihc_upload_id))

        self._check_gate(he_upload_id, ihc_upload_id, diagnostics)

        crossed = region_tools.regroup(selected, warped_rings, ring_counts, ihc_mpp=ihc_mpp)

        job.message = "rendering the comparison"
        self._render(he_upload_id, ihc_upload_id, crossed, he_path=he_path, ihc_path=ihc_path)

        report = AlignmentReport(
            he_upload_id=he_upload_id,
            ihc_upload_id=ihc_upload_id,
            marker=self._marker_for(ihc_upload_id),
            state="ready",
            generated_at=_now(),
            regions=[
                AlignedRegion(
                    index=region.index,
                    rank=rank,
                    roi_id=region.roi_id,
                    source=region.source,
                    cells=region.cells,
                    area_mm2=region.area_mm2,
                    he_rings=region.he_rings,
                    ihc_rings=region.ihc_rings,
                )
                for rank, region in enumerate(crossed, start=1)
            ],
            diagnostics=AlignmentDiagnostics(
                **{
                    key: value
                    for key, value in diagnostics.items()
                    if key in AlignmentDiagnostics.model_fields
                }
            ),
            area_coverage=round(reached, 4),
            carried_mm2=round(sum(region.area_mm2 for region in crossed), 3),
            invasive_mm2=round(invasive_mm2, 3),
            notes=self._notes(diagnostics, crossed, reached),
        )
        self._write_report(report)
        return report

    def _crossable(self, he_upload_id: str) -> list:
        """Step 11's refined foci, or a refusal saying which step is missing.

        **There is no tile fallback, and that is the point of the method.** Before step
        11 this service chose its own regions off the class map, so it could always
        produce *something*; now the mask it carries is a product of a step that a person
        drove, and quietly substituting the coarse squares when that step has not run
        would put a different denominator under every number after this one - silently,
        and in a report that would look exactly the same.

        `RefinementError` is translated rather than propagated so the message names the
        step to go and run, which is the only thing the caller can act on.
        """
        if roi_source == "tiles":
            # The explicit opt-in. Step 10's ticked candidates, carried as they are.
            selection = roi_selection_service.report(he_upload_id)
            ticked = set(selection.selected)
            candidates = [one for one in selection.candidates if one.roi_id in ticked]
            if not candidates:
                raise AlignmentError(
                    "step 10 has no selected regions on this H&E slide, so there is "
                    "nothing to carry across even with refinement skipped."
                )
            logger.warning(
                "carrying %d COARSE TILE region(s) for %s - step 11 skipped by explicit "
                "request; these numbers are not comparable with refined ones",
                len(candidates),
                he_upload_id,
            )
            return region_tools.tile_regions(candidates)

        try:
            refined = roi_refinement_service.regions(he_upload_id)
        except RefinementError as exc:
            raise AlignmentError(
                "there is no pixel-level invasive mask for this H&E slide yet, and this "
                "step will not carry the coarse tile regions across in its place - they "
                "are a different denominator and nothing downstream could tell. Run "
                f"step 10 and step 11 on it first. ({exc})"
            ) from exc

        crossing = region_tools.selected_regions(refined)
        if not crossing:
            raise AlignmentError(
                "step 11 refined no invasive carcinoma on the H&E slide, so there is no "
                "region to carry across. Either the selected regions held none once "
                "BEETLE looked at them per pixel, or every selected region failed."
            )
        return crossing

    def _refusal(
        self, he_upload_id: str, ihc_upload_id: str, refused: RegistrationRefused
    ) -> AlignmentReport:
        """A refusal is a result, so it is stored like one - with no regions."""
        diagnostics = refused.diagnostics
        report = AlignmentReport(
            he_upload_id=he_upload_id,
            ihc_upload_id=ihc_upload_id,
            marker=self._marker_for(ihc_upload_id),
            state="refused",
            generated_at=_now(),
            regions=[],
            diagnostics=AlignmentDiagnostics(
                **{
                    key: value
                    for key, value in diagnostics.items()
                    if key in AlignmentDiagnostics.model_fields
                }
            ),
            refusal_reasons=refused.reasons,
        )
        self._write_report(report)
        return report

    def _marker_for(self, ihc_upload_id: str) -> str | None:
        """Which antibody this IHC slide carries, from the case sidecar that loaded it.

        Read rather than passed in, so a report on disk says what it is about
        without needing the session that produced it. Absent for a pair that
        was assembled by hand rather than through `POST /cases/load`.
        """
        cases_dir = settings.data_dir / "cases"
        if not cases_dir.is_dir():
            return None
        for path in cases_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.get("ihc_upload_id") == ihc_upload_id:
                return record.get("marker")
        return None

    def _mpp(self, path: Path) -> float | None:
        reader = open_slide(path)
        try:
            return reader.mpp
        finally:
            reader.close()

    #: Pairs allowed through a failed gate, by explicit instruction, with the failure
    #: recorded as a caveat instead of a refusal.
    #:
    #: **Both are the near-blank CD44 sections**, and both were refused outright: CAN_00267
    #: on alignment NMI 1.0060 against a 1.008 floor, CAN_00865 on a tissue-area ratio of
    #: 0.15. The instruction for these two was that registration should proceed and their
    #: near-blank staining should remain a caveat on the resulting scores - a refusal drops
    #: the marker from the CSV entirely, which is not the same thing as a flagged result.
    #:
    #: **Scoped to two pairs on purpose.** Lowering `MIN_ALIGNMENT_NMI` or widening the
    #: ratio bounds would buy these two at the cost of every other pair's protection, and
    #: the gate exists because a mask carried through a bad transform lands on unrelated
    #: cells and nothing downstream can tell. A named pair is auditable; a loosened
    #: threshold is invisible.
    #:
    #: A pair listed here that passes the gate is unaffected - this only ever converts an
    #: existing refusal into a caveat, and never suppresses the reasons.
    GATE_OVERRIDES: dict[tuple[str, str], str] = {
        ("CAN_00267", "A"): "near-blank CD44 section; proceed with the failure as a caveat",
        ("CAN_00865", "A"): "near-blank CD44 section; proceed with the failure as a caveat",
    }

    def _check_gate(
        self, he_upload_id: str, ihc_upload_id: str, diagnostics: dict[str, Any]
    ) -> None:
        """Refuse on a failed gate, unless this pair is explicitly allowed past it.

        The reasons are never discarded. They go into the diagnostics, are logged as a
        warning, and become a caveat on every score the pair produces, because a number
        carried past a failed check is only defensible while the failed check travels
        with it.
        """
        reasons = gate.evaluate(diagnostics)
        if not reasons:
            return

        try:
            pair = stored_transforms._case_for(he_upload_id, ihc_upload_id)
        except Exception:  # noqa: BLE001 - an unidentifiable pair gets no exemption
            pair = None

        note = self.GATE_OVERRIDES.get(pair) if pair else None
        if note is None:
            raise RegistrationRefused(reasons, diagnostics)

        logger.warning(
            "gate OVERRIDDEN for %s/%s (%s): proceeding despite %d failed check(s): %s",
            pair[0],
            pair[1],
            note,
            len(reasons),
            "; ".join(reasons),
        )
        diagnostics["gate_overridden"] = reasons

    def _tissue_areas(self, he_upload_id: str, ihc_upload_id: str) -> dict[str, Any]:
        """How much tissue each section holds, for the gate's ratio test.

        **Measured by optical density, not by colour saturation**, and the difference
        decides whether four of CAN_00865's five markers can be scored at all.

        Step 3's mask keys off colour saturation, which measures how *stained* a slide is
        rather than whether tissue is present. `cohort_characterisation.md` section 2
        measured that rule capturing 4% of the tissue on CAN_00865's CD44 section and 30%
        on CAN_00251's. Used here it compares a strongly-stained H&E against a paler IHC
        section on incompatible terms:

            CAN_00865 H&E vs F     saturation rule   357.2 vs 112.1 mm2   ratio 0.31
                                   optical density    96.5 vs  89.2 mm2   ratio 0.93
                                   published figures   74.3 vs  91.1 mm2   ratio 1.23

        The saturation rule put the H&E at 357 mm2 where the published figure is 74.3 -
        an overestimate of nearly five times - and the gate then refused a registration
        that was perfectly good.

        The optical-density figures come from the registration renders, where both
        sections are measured the same way at the same resolution, and which were checked
        against the cohort's published table before use. Step 3's numbers are still
        reported beside them so the two can be compared and this change can be argued
        with, but the ratio the gate tests is the density one.

        Falls back to step 3 when a case has no render - a wrong-but-present measurement
        is better than none here, because the ratio is the only gate that does not depend
        on the registration having succeeded.
        """
        from app.services.tissue_service import tissue_service

        areas: dict[str, Any] = {}
        #: Why each preferred measurement was not used (P-17). Every fallback below used to
        #: swallow its exception, so a gate decided on step 3's saturation rule - the one
        #: this docstring explains is wrong by up to five times - looked identical to one
        #: decided on the shared optical-density cut. The reasons travel on the report.
        fallbacks: list[str] = []

        # Step 3's figures, kept for comparison rather than for the test.
        try:
            for label, upload_id in (("he", he_upload_id), ("ihc", ihc_upload_id)):
                footprint = tissue_service.footprint(upload_id)
                mm_per_px = (footprint.mpp or 0.0) / 1000.0
                areas[f"{label}_tissue_saturation_mm2"] = round(
                    float(footprint.mask.sum()) * mm_per_px * mm_per_px, 2
                )
        except Exception as exc:  # noqa: BLE001 - one of four gates
            logger.warning("could not measure saturation tissue area: %s", exc)
            fallbacks.append(f"step 3's saturation areas: {type(exc).__name__}: {exc}")

        density = self._density_areas(he_upload_id, ihc_upload_id, fallbacks)
        if density:
            areas.update(density)
        else:
            # No render for this case: fall back to step 3's numbers for the test itself.
            he = areas.get("he_tissue_saturation_mm2")
            ihc = areas.get("ihc_tissue_saturation_mm2")
            if he and ihc:
                areas["he_tissue_mm2"], areas["ihc_tissue_mm2"] = he, ihc
                areas["tissue_area_ratio"] = round(ihc / he, 3)
                areas["tissue_area_source"] = "step3_saturation_fallback"
            else:
                fallbacks.append("no tissue area could be measured, so the ratio test did not run")
        if fallbacks:
            areas["tissue_area_fallbacks"] = fallbacks
        return areas

    def _density_areas(
        self, he_upload_id: str, ihc_upload_id: str, fallbacks: list[str] | None = None
    ) -> dict[str, Any]:
        """Optical-density tissue areas for this pair, measured with **one cut per case**.

        `gate_areas.json` measures every section of a block at the threshold the H&E chose,
        rather than letting each slide pick its own. That distinction is the whole point:
        a ratio between two measurements means nothing unless both were measured the same
        way, and neither earlier rule managed it.

        Step 3 keyed off colour saturation and read CAN_00865's H&E at 357 mm2 against a
        published 74.3. Replacing it with per-slide Otsu fixed the rule but kept the
        incompatibility - Otsu picks a lower cut for an H&E than for a pale IHC section
        (0.190 against 0.251 on CAN_00270), so the H&E was measured generously and the IHC
        conservatively, and two markers that had scored fine were refused.

        With one cut per case, all **30 pairs fall on the same side of the 0.5 threshold as
        the cohort's published figures** - including the two near-blank CD44 sections,
        which refuse under both. Whatever bias the cut carries now falls on both sides of
        the ratio equally, which is the only property the ratio needs.

        Falls back to the per-slide render figures, then to step 3, so a case measured by
        neither still gets a check rather than none. Each fall-through says why in
        `fallbacks` (P-17) - a weaker measurement is acceptable, an unannounced one is not.
        """
        fallbacks = fallbacks if fallbacks is not None else []
        case = marker = None
        try:
            case, marker = stored_transforms._case_for(he_upload_id, ihc_upload_id)
        except Exception as exc:  # noqa: BLE001 - an unknown pair has no case-level measurement
            fallbacks.append(f"no case record for this pair ({exc})")
            return {}

        directory = stored_transforms.common.case_dir(case)

        # Preferred: one cut per case, measured by `gate_areas.py`.
        try:
            payload = json.loads((directory / "gate_areas.json").read_text(encoding="utf-8"))
            areas = payload["areas"]
            he_mm2, ihc_mm2 = float(areas["HE"]), float(areas[marker])
            if he_mm2:
                return {
                    "he_tissue_mm2": round(he_mm2, 2),
                    "ihc_tissue_mm2": round(ihc_mm2, 2),
                    "tissue_area_ratio": round(ihc_mm2 / he_mm2, 3),
                    "tissue_area_source": f"optical_density_shared_cut_{payload.get('cut')}",
                }
        except Exception as exc:  # noqa: BLE001 - fall through to the render figures
            fallbacks.append(f"the shared-cut areas (gate_areas.json): {type(exc).__name__}: {exc}")

        # Fallback: the render's per-slide figures. Worse for a ratio, better than nothing.
        try:
            manifest = json.loads((directory / "render.json").read_text(encoding="utf-8"))
            slides = manifest["slides"]
            he_mm2 = float(slides["HE"]["tissueMm2"])
            ihc_mm2 = float(slides[marker]["tissueMm2"])
        except Exception as exc:  # noqa: BLE001
            fallbacks.append(f"the render's per-slide areas (render.json): {type(exc).__name__}: {exc}")
            return {}
        if not he_mm2:
            fallbacks.append("the render recorded no H&E tissue area")
            return {}
        return {
            "he_tissue_mm2": round(he_mm2, 2),
            "ihc_tissue_mm2": round(ihc_mm2, 2),
            "tissue_area_ratio": round(ihc_mm2 / he_mm2, 3),
            "tissue_area_source": "optical_density_per_slide_otsu",
        }

    def _notes(self, diagnostics: dict, crossed: list, coverage: float = 0.0) -> list[str]:
        notes: list[str] = []

        nmi = diagnostics.get("alignment_nmi")
        base = diagnostics.get("alignment_nmi_unregistered")
        if nmi is not None and base is not None:
            gain = nmi - base
            notes.append(
                f"Structural agreement between the two sections scores {nmi:.3f} after "
                f"registration against {base:.3f} without it"
                + (
                    f" - a gain of {gain:+.3f}."
                    if gain > 0
                    else " - registration did not improve on simply overlaying them, "
                    "which is worth a careful look at the panels."
                )
            )

        clamped = diagnostics.get("clamped_vertices") or 0
        total = diagnostics.get("total_vertices") or 0
        if clamped and total:
            notes.append(
                f"{clamped} of {total} border vertices warped off the edge of the IHC slide "
                f"and were clamped to it - the two sections do not cover quite the same ground."
            )

        carried_mm2 = sum(region.area_mm2 for region in crossed)
        pixel_level = all(region.source == "beetle_pixel" for region in crossed)
        notes.append(
            f"{len(crossed)} invasive focus/foci carried across, {carried_mm2:.1f} mm2. "
            + (
                "These are BEETLE's per-pixel boundaries from step 11, not step 9's tile "
                "squares - every measurement after this one is taken inside them."
                if pixel_level
                else "These are step 9's tile regions."
            )
        )
        notes.append(
            f"That is {coverage:.0%} of the invasive carcinoma step 8 called on the whole "
            "slide. The two are not the same quantity - one is a per-window call and the "
            "other a per-pixel boundary - so the figure is a rough guide to how much "
            "tumour the score will see rather than an exact coverage. OncoStem's "
            "procedure is that the entire slide is scanned and every field averaged, so "
            "tissue left out here is tumour the score never sees."
        )
        notes.append(
            "In-situ disease is not carried: it is excluded from the score by Rule 5, "
            "and step 11 traces BEETLE's invasive class alone."
        )
        return notes

    # --- pictures -----------------------------------------------------------

    def _render(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        crossed: list,
        *,
        he_path: Path,
        ihc_path: Path,
    ) -> None:
        he_reader = open_slide(he_path)
        ihc_reader = open_slide(ihc_path)
        try:
            he_width, he_height = he_reader.dimensions
            ihc_width, ihc_height = ihc_reader.dimensions

            self._path(he_upload_id, ihc_upload_id, "he_borders.png").write_bytes(
                overlay.borders_on_slide_png(
                    he_reader,
                    [region.he_rings for region in crossed],
                    slide_width=he_width,
                    slide_height=he_height,
                )
            )
            self._path(he_upload_id, ihc_upload_id, "ihc_borders.png").write_bytes(
                overlay.borders_on_slide_png(
                    ihc_reader,
                    [region.ihc_rings for region in crossed],
                    slide_width=ihc_width,
                    slide_height=ihc_height,
                )
            )

            pad_px = int(round(settings.roi_crop_pad_um / (ihc_reader.mpp or 0.25)))
            he_pad_px = int(round(settings.roi_crop_pad_um / (he_reader.mpp or 0.25)))

            for rank, region in enumerate(crossed, start=1):
                self._write_crop(
                    he_upload_id,
                    ihc_upload_id,
                    f"invasive{rank}.png",
                    ihc_reader,
                    region.ihc_rings,
                    pad_px=pad_px,
                    slide_width=ihc_width,
                    slide_height=ihc_height,
                )
                self._write_crop(
                    he_upload_id,
                    ihc_upload_id,
                    f"he_invasive{rank}.png",
                    he_reader,
                    region.he_rings,
                    pad_px=he_pad_px,
                    slide_width=he_width,
                    slide_height=he_height,
                )
        finally:
            he_reader.close()
            ihc_reader.close()

    def _write_crop(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        name: str,
        reader: Any,
        rings: list[list[list[float]]],
        *,
        pad_px: int,
        slide_width: int,
        slide_height: int,
    ) -> None:
        if not rings or not rings[0]:
            return

        # `region_bbox_level0` wants something with `.rings`; the outer ring is
        # all it reads, so a tiny stand-in is cheaper than reshaping the region.
        class _Outer:
            def __init__(self, ring: list[list[float]]) -> None:
                self.rings = [[(int(x), int(y)) for x, y in ring]]

        bbox = padded_bbox(
            region_bbox_level0(_Outer(rings[0])), pad_px, slide_width, slide_height
        )
        self._path(he_upload_id, ihc_upload_id, name).write_bytes(
            crop_png(reader, bbox, max_size=CROP_MAX_SIZE)
        )

    # --- reading it back ----------------------------------------------------

    def report(self, he_upload_id: str, ihc_upload_id: str) -> AlignmentReport:
        report = self._read_report(he_upload_id, ihc_upload_id)
        if report is None:
            raise AlignmentError(
                "these two slides have not been aligned yet - run step 10 first"
            )
        return report

    def panel(self, he_upload_id: str, ihc_upload_id: str, name: str) -> bytes:
        if name not in {"he_borders", "ihc_borders"}:
            raise AlignmentError(
                f"unknown panel {name!r}; expected he_borders or ihc_borders"
            )
        path = self._path(he_upload_id, ihc_upload_id, f"{name}.png")
        if not path.is_file():
            raise AlignmentError("no panels for this pair yet - run step 10 first")
        return path.read_bytes()

    def crop(
        self, he_upload_id: str, ihc_upload_id: str, rank: int, *, source: str = "ihc"
    ) -> bytes:
        if source not in {"ihc", "he"}:
            raise AlignmentError(f"unknown source {source!r}; expected ihc or he")
        # Against what this pair actually carried, not against the cap. The cap
        # is an upper bound on a selection that stops when it has covered enough
        # tumour, so a pair with one large region carries one - and validating
        # against the cap would advertise ranks that do not exist.
        carried = len(self.report(he_upload_id, ihc_upload_id).regions)
        if not 1 <= rank <= carried:
            raise AlignmentError(
                f"rank must be between 1 and {carried} for this pair, not {rank}"
            )

        name = f"invasive{rank}.png" if source == "ihc" else f"he_invasive{rank}.png"
        path = self._path(he_upload_id, ihc_upload_id, name)
        if not path.is_file():
            raise AlignmentError(
                f"there is no rank {rank} region for this pair - fewer invasive regions "
                f"were carried across than that"
            )
        return path.read_bytes()

    def confirm(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        *,
        confirmed: bool,
        by: str = "person",
    ) -> AlignmentReport:
        """Record a judgement on the alignment. The blocking gate.

        Only a `ready` alignment can be confirmed: confirming a refusal would
        be a way to talk the gate out of its answer, which is exactly what the
        gate is for.

        `by` is "person" or "machine", and it is the honest half of letting an
        unattended run get past this step at all. The gate exists because
        somebody is meant to look at the two panels and say the regions landed
        on the same tissue; a batch job cannot do that. It can proceed - refusing
        would mean no overnight run could ever produce a score - but what it
        cannot do is leave a record indistinguishable from a human sign-off. So
        the confirmation is stamped, and step 16 turns a machine stamp into a
        caveat printed beside every number that came out of these regions.
        """
        report = self.report(he_upload_id, ihc_upload_id)
        if report.state != "ready" and confirmed:
            raise AlignmentError(
                f"this alignment is {report.state}, not ready - it cannot be confirmed"
            )
        if by not in {"person", "machine"}:
            raise AlignmentError(f"confirmed by {by!r}; expected 'person' or 'machine'")

        report.confirmed = confirmed
        report.confirmed_at = _now() if confirmed else None
        report.confirmed_by = by if confirmed else None
        self._write_report(report)
        return report

    def discard(self, he_upload_id: str, ihc_upload_id: str) -> None:
        """Throw away everything cached for this pair, registration included."""
        import shutil

        directory = settings.ihc_alignment_dir / self.key(he_upload_id, ihc_upload_id)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        self._jobs.pop(self.key(he_upload_id, ihc_upload_id), None)


ihc_alignment_service = IhcAlignmentService()

__all__ = ["AlignmentError", "IhcAlignmentService", "ihc_alignment_service"]
