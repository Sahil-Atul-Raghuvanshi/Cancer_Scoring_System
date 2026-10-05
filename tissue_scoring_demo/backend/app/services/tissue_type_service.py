"""Step 8 - tissue-type segmentation, orchestrated.

A run is a background job and its result is cached on disk, and unlike step 2 the
cache is not an optimisation - it is what makes the step openable. Tens of thousands
of ResNet18 forward passes on a CPU is tens of minutes for one slide, and nobody
opens a screen twice that costs that. So the API starts a job, the client polls, and
the finished class map lands under `data/tissue_type/<upload_id>/`.

Only one run happens at a time, for step 2's reason: letting two slides fight over
the same cores makes both slower than running them in sequence and makes the progress
bar a lie.

**Its input is step 7's index, and it asks step 7's service for it.** Not step 3's
mask and not step 2's artefact map - step 7 has already combined those into an audited
list of tiles, and re-deriving the combination here would be duplicating it rather
than depending on it. The same argument `tissue_service.footprint` and
`calibration_service.white_point` were written for, one step further along.

**Where the pixels come from.** `window_reader` is the one place in this step
that touches a slide, and it is a composition of functions that already exist:
`read_tile` is step 5's, the white point is step 4's service, and the haematoxylin is
`input.haematoxylin_od`, which is `optical_density` plus step 6's `separate` and
nothing else. The guide's one hard rule about the fork is that training and inference
must call the same deconvolution, and this is where that rule is kept or broken for
the model that was trained on it.

What lands on disk per upload:

    report.json     the API payload, exactly as served
    classmap.npz    labels, probabilities and the grid geometry - step 9's input
    map.png         the slide with the class map over it
    flat.png        the class map alone
    scored.png      only the class the score is gated on
    confidence.png  the top-class probability
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.contract import RunCancelled
from app.pipeline.step05_optical_density.tiles import read_tile
from app.pipeline.step08_tissue_type_segmentation.branches import (
    DEFAULT_BRANCH,
    SERVED_BRANCHES,
    SERVED_HEADS,
    ModelBranch,
    parse_branch,
)
from app.pipeline.step08_tissue_type_segmentation import (
    familiarity,
    inference,
    model,
    overlay,
    uncertainty,
)
from app.pipeline.step08_tissue_type_segmentation import input as model_input
from app.pipeline.step08_tissue_type_segmentation import (
    beetle,
    pixel_overlay,
    pixels,
)
from app.pipeline.step08_tissue_type_segmentation.pixels import PixelMap
from app.pipeline.step08_tissue_type_segmentation.classes import (
    CANDIDATE,
    CLASS_COLOURS,
    CLASS_LABELS,
    CLASS_MEANING,
    CLASS_NAMES,
    CLASS_PLAIN,
    DISPLAY_NAMES,
    RIVAL,
    SCORED,
)
from app.pipeline.step08_tissue_type_segmentation.inference import ClassMap, WindowGrid
from app.pipeline.step08_tissue_type_segmentation.uncertainty import UncertaintyParams
from app.schemas.tissue_type import (
    TissueTypeCapability,
    TissueTypeCaveat,
    TissueTypeClass,
    TissueTypeGrid,
    TissueTypeModelInfo,
    TissueTypePaint,
    TissueTypeParams,
    TissueTypeReport,
    TissueTypeRun,
    TissueTypeRefusal,
    TissueTypeUncertainty,
    TissueTypeUncertaintyParams,
)
from app.services.calibration_service import calibration_service
from app.services import tiling_service as tiling_module
from app.services.tiling_service import tiling_service
from app.services.tissue_service import tissue_service
from app.services.upload_service import get_record, resolve_ready_path

logger = get_logger(__name__)

CITATION = (
    "Amgad M, Elfandy H, Hussein H, et al. Structured crowdsourcing enables "
    "convolutional segmentation of histology images. Bioinformatics 35(18):3461-3467 "
    "(2019) - BCSS, the pixel labels behind classes 0 and 2. "
    "Brancati N, Anniciello AM, Pati P, et al. BRACS: A Dataset for BReast Carcinoma "
    "Subtyping in H&E histology images. Database (2022) - the in-situ regions. "
    "Ruifrok AC, Johnston DA. Quantification of histochemical staining by colour "
    "deconvolution. Anal Quant Cytol Histol 23(4):291-299 (2001). "
    "Tellez D, Litjens G, Bandi P, et al. Quantifying the effects of data augmentation "
    "and stain color normalization in convolutional neural networks for computational "
    "pathology. Medical Image Analysis 58:101544 (2019) - why the model reads the "
    "haematoxylin channel rather than RGB."
)

LICENCE_NOTE = (
    "The checkpoint this step serves by default is fitted on BCSS (CC0) plus BRACS "
    "regions of interest labelled by BEETLE's released nnU-Net. BRACS is licensed for "
    "non-commercial research only and BEETLE is CC BY-NC-SA 4.0, so the model and every "
    "number computed from it are RESEARCH ONLY and must not ship. The BCSS-only "
    "checkpoint (invasive_tile_v1_imagenet) is the commercially clean alternative and "
    "cannot separate in-situ from invasive disease, because BCSS contains almost no "
    "in-situ annotation."
)

#: Where each phase sits in the overall progress bar. Building the grid is one pass
#: over step 7's tiles; the forward passes are essentially all of the work.
_PHASE_SPAN = {
    "grid": (0.0, 0.02),
    "classifying": (0.02, 0.94),
    "rendering": (0.94, 1.0),
}

#: Most windows one poll will hand back for painting.
#:
#: A poll two seconds apart on a pass that classifies a few patches a second asks for a
#: few dozen, so this ceiling is never met by a client that has been watching. It is
#: for the one that has not: a reload part-way through, or a viewer opening the screen
#: on a pass already running, both of which ask from zero. Those catch up over a few
#: polls instead of pulling a whole slide's grid into one reply.
_PAINT_CHUNK: int = 8_000


class TissueTypeError(ValueError):
    """A client-correctable problem: no slide, no checkpoint, or a run already going."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


#: In-situ share at or above which step 8 asks for a look (P-17). The measured failure
#: called 45.9% of CAN_00251 in-situ; 0.30 flags that with room to spare, and is a prompt
#: for a person rather than a refusal - a case can genuinely be mostly DCIS.
IN_SITU_FLAG_SHARE = 0.30

#: Share of the run that came back "cannot be determined" - flagged or refused - at or
#: above which the report says so up front (P-17).
UNDETERMINED_FLAG_SHARE = 0.10


def _uncertainty_params() -> UncertaintyParams:
    """The uncertainty layer's settings, as the dataclass the layer takes.

    One place, so `resolve_params` records exactly what `_run` applies. Read every time
    rather than captured at import, because a test that monkeypatches a setting must
    change what the next run does.
    """
    return UncertaintyParams(
        sigma_um=settings.tissue_type_uncertainty_sigma_um,
        area_ref_mm2=settings.tissue_type_uncertainty_area_ref_mm2,
        area_max_mm2=settings.tissue_type_uncertainty_area_max_mm2,
        fill_lo=settings.tissue_type_uncertainty_fill_lo,
        fill_hi=settings.tissue_type_uncertainty_fill_hi,
        min_shape_mm2=settings.tissue_type_uncertainty_min_shape_mm2,
        min_shape_windows=settings.tissue_type_uncertainty_min_shape_windows,
        threshold=settings.tissue_type_uncertainty_threshold,
    )


def _hex(colour: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*colour)


# --- job bookkeeping ---------------------------------------------------------


@dataclass
class Job:
    """Live state of one run. Mutated from the worker thread, read by the API."""

    upload_id: str
    params: dict[str, Any]
    state: str = "queued"
    phase: str | None = None
    message: str | None = None
    progress: float = 0.0
    done: int = 0
    total: int = 0
    started: float = field(default_factory=time.monotonic)
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    duration: float | None = None
    error: str | None = None

    #: Set by `cancel`, read by the worker's progress callback. A plain bool is
    #: enough: one writer, one reader, and a stale read costs one more block.
    cancelled: bool = False

    #: Geometry for painting this pass onto a picture of the slide. Written once, when
    #: the grid exists, and read by every poll after that.
    paint: dict[str, Any] | None = None

    #: Every window classified so far, as flat `row, col, class` triples, in the order
    #: the sweep decided them. The progress screen's feed.
    #:
    #: **A list of ints and no lock.** The worker appends, the API slices, and in
    #: CPython both are single operations under the GIL, so a poll cannot catch a
    #: half-grown list - the same argument `progress` and `cancelled` above are written
    #: under. Flat rather than tuples because the values repeat: a grid's rows, columns
    #: and three class ids are all small integers, which the interpreter interns, so
    #: this is three pointers per window rather than an object each.
    painted: list[int] = field(default_factory=list)

    #: One base64 pixel mask per entry in `painted`, on the BEETLE option only. Appended
    #: in step with it, so `painted_cursor` - which counts windows - indexes this
    #: directly and `painted` in threes.
    #:
    #: **Why the whole run's masks are kept in memory.** Each is `paint_px` squared bytes
    #: (1 kB at the default 32), so a whole-slide pass of a few thousand windows is a
    #: few megabytes - the same order as the flat triples beside it, and the same reason:
    #: a client that reloads mid-pass, or opens the screen on a pass already running,
    #: replays the feed from the beginning rather than seeing a half-painted slide.
    painted_masks: list[str] = field(default_factory=list)


class TissueTypeService:
    """Runs step 8, caches the class map, and reports honestly on what it cannot do."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()

    # --- capability ---------------------------------------------------------

    def capability(self) -> TissueTypeCapability:
        """What step 8 can do right now, and which checkpoint would run.

        **Readiness is asked of the choices step 7 offers, not of the configured
        name.** Which checkpoint runs is step 7's choice, so a report that answered
        "ready: invasive_tile_v3" while every offered choice had no head would be
        describing a model this step will never load. `settings.tissue_type_model`
        survives as the ordering hint for `discover` and as what a caller naming nothing
        directly gets; it is no longer what a run resolves to.

        **And the choice is now a pair.** Both branches share all four geometries by
        construction, so readiness has to be asked per `(branch, field of view)` - eight
        cells, not four. Asking only the four would call the step ready on the strength
        of a haematoxylin head while a viewer sat on the full-colour option with nothing
        published for it.
        """
        state = model.runtime()
        candidates = model.discover()

        # The head at step 7's default choice - the one a viewer who changes nothing
        # will run - and every offered pair that has one at all.
        #
        # `SERVED_HEADS` and not `SERVED_BRANCHES`: BEETLE is now a served branch but it
        # is not a checkpoint of ours, so asking `model_for` about it is the wrong
        # question and it is reported separately below.
        default_head = tiling_module.model_for(
            settings.tiling_field_of_view_um, DEFAULT_BRANCH
        )
        served = {
            (branch, um): tiling_module.model_for(um, branch)
            for branch in SERVED_HEADS
            for um in tiling_module.FIELDS_OF_VIEW
        }
        names = {entry.name for entry in served.values() if entry is not None}

        infos = [self._model_info(entry, selected=entry.name in names) for entry in candidates]

        # BEETLE, listed beside them because step 7 offers it beside them - and listed
        # with its own fields null rather than filled with numbers from somebody else's
        # paper measured on somebody else's split. `available` is a `stat`, so this costs
        # nothing and works with no torch installed.
        beetle_usable, beetle_problem = beetle.available()
        infos.append(self._beetle_info(usable=beetle_usable, problem=beetle_problem))

        selected = default_head.name if default_head else settings.tissue_type_model
        chosen = default_head
        ready = state.usable and (
            any(entry is not None for entry in served.values()) or beetle_usable
        )

        if not state.usable:
            reason = str(state.problem)
        elif chosen is not None:
            reason = (
                f"{chosen.name} on {state.device}: a {chosen.tile_px} px haematoxylin window "
                f"at {chosen.mpp} um/px covering {settings.tiling_field_of_view_um:g} um of "
                "slide, three classes."
            )
        elif ready:
            available = ", ".join(
                f"{um:g} um ({branch.value}: {entry.name})"
                for (branch, um), entry in served.items()
                if entry
            )
            reason = (
                f"no head is published at step 7's default {settings.tiling_field_of_view_um:g} "
                f"um, but one is at {available}. Choose that on step 7."
            )
        else:
            wanted = ", ".join(
                f"{um:g} um" for um in tiling_module.FIELDS_OF_VIEW
            ) + " on either trained branch"
            reason = (
                f"no checkpoint in {model.models_dir()} was fitted at any choice "
                f"step 7 offers ({wanted}), and BEETLE's weights are absent too "
                f"({beetle_problem}). Train a head with tissue_type_model_training - "
                "its publish step writes here, so the trained model and the served model "
                "are the same file."
                + (
                    " Published there, at other fields of view: "
                    + ", ".join(
                        f"{entry.name} ({entry.tile_px * entry.mpp:g} um)"
                        for entry in candidates
                        if entry.usable and entry.tile_px and entry.mpp
                    )
                    + "."
                    if candidates
                    else ""
                )
            )

        return TissueTypeCapability(
            ready=ready,
            reason=reason,
            torch_installed=state.torch_version is not None,
            torch_version=state.torch_version,
            device=state.device,
            device_name=state.device_name,
            threads=state.threads,
            models_root=str(model.models_dir()),
            models=infos,
            default_model=selected,
            licence_track=(chosen.licence_track if chosen else "unknown"),
            licence_note=LICENCE_NOTE,
            citation=CITATION,
        )

    @staticmethod
    def _beetle_info(*, usable: bool, problem: str | None) -> TissueTypeModelInfo:
        """BEETLE as the capability listing reports it.

        **Every field this project measures is left null**, and that is the honest
        entry rather than a sparse one: `init`, `held_out_accuracy`, `dice_invasive` and
        `non_invasive_test_tiles` all mean "measured on our held-out split with our
        labels", and none of them has been. Filling them with the release's published
        figures would put numbers from another dataset's split under column headings a
        reader will compare against our own.

        `tile_px` is null for a different reason: this model has four window sizes rather
        than one - the field of view varies the pixel side and holds the spacing - so
        there is no single geometry to report here. Step 7's picker carries the four.
        """
        return TissueTypeModelInfo(
            name=beetle.MODEL_NAME,
            arch=beetle.ARCH,
            found=usable,
            path=str(beetle.model_zip()),
            # Selected when the viewer commits to it on step 7, which this listing does
            # not know - it describes what is installed, not what a slide chose.
            selected=False,
            mpp=beetle.SPACING,
            licence_track=beetle.LICENCE_TRACK,
            licences=dict(beetle.LICENCES),
            training_source=(
                "BEETLE's own training set - breast whole-slide images annotated for "
                "epithelium, invasive tumour and necrosis. Not this project's data, and "
                "not comparable with a held-out number measured on our split."
            ),
            per_pixel=True,
            classes=list(beetle.PIXEL_CLASSES),
            folds=list(beetle.folds_to_run()) if usable else None,
            folds_available=len(beetle.FOLD_MEMBERS),
            patch_px=beetle.NOMINAL_PATCH_PX,
            citation=beetle.CITATION,
            problem=problem,
        )

    @staticmethod
    def _model_info(candidate: model.Candidate, *, selected: bool) -> TissueTypeModelInfo:
        return TissueTypeModelInfo(
            arch=candidate.arch,
            name=candidate.name,
            found=candidate.usable,
            path=str(candidate.checkpoint),
            selected=selected,
            init=candidate.init,
            tile_px=candidate.tile_px,
            mpp=candidate.mpp,
            standardise=candidate.standardise,
            created=candidate.created,
            sha256=candidate.sha256,
            bytes=candidate.bytes,
            licence_track=candidate.licence_track,
            licences=candidate.licences,
            training_source=candidate.training_source,
            held_out_accuracy=candidate.held_out_accuracy,
            dice_invasive=candidate.dice_invasive,
            dice_non_invasive=candidate.dice_non_invasive,
            non_invasive_test_tiles=candidate.non_invasive_tiles,
            problem=candidate.problem,
        )

    # --- storage ------------------------------------------------------------

    def _dir(self, upload_id: str) -> Path:
        return settings.tissue_type_dir / upload_id

    def _path(self, upload_id: str, name: str) -> Path:
        return self._dir(upload_id) / name

    def _read_json(self, upload_id: str, name: str) -> dict[str, Any] | None:
        path = self._path(upload_id, name)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable step 8 artefact %s", path, exc_info=True)
            return None

    def _write_json(self, upload_id: str, name: str, payload: Any) -> None:
        path = self._path(upload_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def asset(self, upload_id: str, name: str) -> bytes:
        """One cached PNG. Raises when the run has not produced it."""
        path = self._path(upload_id, name)
        if not path.is_file():
            raise TissueTypeError(
                f"no {name} for this slide yet - run step 8 first "
                f"(POST /tissue-type/{upload_id}/run)"
            )
        return path.read_bytes()

    # --- params -------------------------------------------------------------

    def resolve_params(
        self,
        upload_id: str,
        *,
        overlap: float | None = None,
        name: str | None = None,
        branch: str | None = None,
        fov: float | None = None,
    ) -> dict[str, Any]:
        """Settle what a run will actually use, refusing what cannot work.

        The window geometry is **not** a caller's choice: it is read off the manifest,
        because the model's field of view is a property of the checkpoint and serving a
        different one is the same class of error as serving a different channel. Step 7
        reads the same manifest and lays the same square, so what is recorded here as
        `window_px` and `mpp` is also the geometry of the tiles this run is gated by.

        **Which checkpoint runs is step 7's field-of-view choice, inherited.** That is
        the same argument as the overlap below, only sharper: a field of view *is* a
        property of the weights, so "224 um" and "the head fitted at 224 um" are one
        decision and cannot be taken in two places. A caller may still name a checkpoint
        outright - that is a licence decision as much as an accuracy one - and naming one
        whose field of view is not the grid's is the one way the two steps can still come
        apart. Naming nothing inherits what the tiling screen showed and priced.
        """
        capability = self.capability()

        if not capability.torch_installed:
            raise TissueTypeError(capability.reason)

        # Step 7's index is the input, so its own parameters are part of this run's
        # identity: a different tissue threshold is a different set of tiles and
        # therefore a different class map, and a cache that ignored that would serve
        # the old map under the new threshold.
        # Step 7's committed choice, and this is where the two steps are made to agree.
        # `branch` and `fov` are **assertions** when supplied, not overrides: a caller
        # naming a choice that disagrees with what the viewer committed is refused
        # rather than silently run on a different grid, for the same reason
        # `tiling_service.field_of_view` refuses 300 um instead of snapping it.
        selection = tiling_service.selection(upload_id)
        if branch is not None and branch != selection.branch:
            raise TissueTypeError(
                f"step 7's committed choice is {selection.branch} at "
                f"{selection.field_of_view_um:g} um, and this request named {branch}. "
                "Commit the choice on step 7 first - running a different grid from the "
                "one that was priced is how the two steps come apart."
            )
        if fov is not None and abs(float(fov) - selection.field_of_view_um) > 1.0:
            raise TissueTypeError(
                f"step 7's committed field of view is {selection.field_of_view_um:g} um "
                f"and this request named {float(fov):g} um. Commit the choice on step 7 "
                "first."
            )

        # Explicitly, so the grid this run is gated by is the grid step 7 showed and
        # priced. Passing nothing here used to mean the *default* field of view, which
        # is how a caller could name the 448 um head and have step 8 rebuild a 224 um
        # index underneath it.
        #
        # Note `overlap` is deliberately NOT the caller's: `report.params.overlap` is
        # read below as "what step 7 did", recorded separately from this run's own
        # overlap, and collapsing the two would lose a distinction the tests pin.
        report = tiling_service.report(
            upload_id,
            branch=selection.branch,
            fov=selection.field_of_view_um,
            overlap=selection.overlap,
            threshold=selection.tissue_threshold,
        )

        # **The overlap is settled before either branch, because both need it.** Step
        # 7's, not a setting of this step's own: with the two grids sharing a size and a
        # resolution, taking it from anywhere else would put this step on a grid step 7
        # never priced.
        default_overlap = report.params.overlap
        share = default_overlap if overlap is None else float(overlap)
        if not 0.0 <= share < 1.0:
            raise TissueTypeError(
                f"overlap must be at least 0 and less than 1, not {share}; at 1 the "
                "stride would be zero and the grid would never advance"
            )

        common = {
            "overlap": round(share, 4),
            "span": 0,
            "stride": 0,
            "branch": report.params.branch,
            "field_of_view_um": report.params.field_of_view_um,
            "tile_overlap": report.params.overlap,
            "tissue_threshold": report.params.tissue_threshold,
            "tissue_threshold_source": report.params.tissue_threshold_source,
            "qc_gated": report.params.qc_gated,
            "qc_source": report.params.qc_source,
            "min_tissue_share": settings.tissue_type_min_tissue_share,
            # Step 3's mask, by identity. The threshold above does not pin it: the same
            # cut over a mask step 3 has since corrected - P-05's scanner fill, taken
            # out of it - is a different set of windows, and a class map cached before
            # the correction would otherwise keep being served as current.
            "tissue_mask_key": tissue_service.footprint(
                upload_id, threshold=report.params.tissue_threshold
            ).key,
            # Step 7's two gates (P-15). Its grid's overlap and threshold were already in
            # the key; the shares that decide which tiles are kept were not, so tightening
            # either served the class map of the old, larger set of tiles.
            "tile_min_tissue_share": report.params.min_tissue_share,
            "tile_min_clean_share": report.params.min_clean_share,
            "block_windows": settings.tissue_type_block_windows,
            "device": capability.device,
            # The uncertainty layer's own settings. Resolved here, with everything else
            # a run is priced under, and recorded on the report - they decide which
            # tissue comes back purple, and a screen showing a flagged region without
            # saying what bar it cleared is asking to be trusted rather than read.
            #
            # Present on the BEETLE option too, and null there: `common` is shared and
            # the per-pixel branch never derives a layer, so the report says "not
            # applicable" rather than quoting a threshold nothing applied.
            **(
                {}
                if parse_branch(report.params.branch) is ModelBranch.BEETLE
                else {"uncertainty": _uncertainty_params().as_dict()}
            ),
        }

        # --- the BEETLE option ------------------------------------------------
        #
        # Resolved entirely without `model.find`, `model.load_pinned` or a manifest,
        # because there is no checkpoint of ours to find and nothing to verify: the
        # geometry is a division by BEETLE's fixed spacing and the preprocessing is
        # division by 255. What *is* checked, and checked here rather than two hours
        # into the pass, is that the archive is present and is the release this code was
        # written against - `read_archive` re-reads the label codes every time for the
        # reason its docstring gives.
        if parse_branch(report.params.branch) is ModelBranch.BEETLE:
            usable, problem = beetle.available()
            if not usable:
                raise TissueTypeError(problem or "BEETLE's weights are not installed")
            try:
                archive = beetle.read_archive()
                folds = beetle.folds_to_run()
                step = beetle._step(None)
            except beetle.BeetleError as exc:
                raise TissueTypeError(str(exc)) from exc

            window_px = beetle.window_px(report.params.field_of_view_um)
            return {
                **common,
                "model": beetle.MODEL_NAME,
                "model_sha256": None,
                "licence_track": beetle.LICENCE_TRACK,
                "window_px": window_px,
                "window_um": round(window_px * beetle.SPACING, 2),
                "mpp": beetle.SPACING,
                # Null rather than identity values: BEETLE publishes no input contract
                # with these degrees of freedom, so reporting them as "off" would assert
                # one. See `TissueTypeParams`.
                "standardise": None,
                "gamma": None,
                "invert_polarity": None,
                "input_channel": None,
                "per_pixel": True,
                "pixel_classes": list(beetle.PIXEL_CLASSES),
                "folds": list(folds),
                "patch_step": step,
                "patch_px": archive.patch,
                "mask_mpp": settings.tissue_type_beetle_mask_mpp,
                "batch_size": settings.tissue_type_beetle_batch_size,
                "paint_px": settings.tissue_type_beetle_paint_px,
            }

        chosen = name or report.params.model
        if chosen is None:
            # Step 7 laid its grid on a planned geometry because no head is published at
            # that field of view yet. Refusing here rather than falling back to the
            # configured checkpoint: that one was fitted at a *different* field of view,
            # so it would score a grid it never saw and nothing on screen would say so.
            published = [
                f"{entry.name} ({entry.tile_px * entry.mpp:g} um)"
                for entry in model.discover()
                if entry.usable and entry.tile_px and entry.mpp
            ]
            raise TissueTypeError(
                f"no checkpoint fitted at {report.params.field_of_view_um:g} um is "
                f"published in {model.models_dir()}, so step 7's grid has no model to "
                "run on. Publish one there, or choose the other field of view on step 7."
                + (f" Published: {', '.join(published)}." if published else "")
            )

        candidate = model.find(chosen)
        if candidate is None or not candidate.usable:
            published = [entry.name for entry in model.discover()]
            raise TissueTypeError(
                f"no usable checkpoint named {chosen!r} in {model.models_dir()}. "
                + (f"Published there: {published}." if published else capability.reason)
            )

        try:
            pinned = model.load_pinned(chosen)
        except model.ModelError as exc:
            raise TissueTypeError(str(exc)) from exc

        return {
            **common,
            "model": pinned.name,
            "model_sha256": pinned.sha256,
            "licence_track": pinned.licence_track,
            "window_px": pinned.tile_px,
            "window_um": round(pinned.tile_px * pinned.mpp, 2),
            "mpp": pinned.mpp,
            "standardise": pinned.standardise,
            "gamma": pinned.gamma,
            "invert_polarity": pinned.invert_polarity,
            "input_channel": pinned.channel,
            "per_pixel": False,
            "batch_size": settings.tissue_type_batch_size,
            # Which windows the model is allowed to answer for - see `familiarity`.
            # In the key, because a changed cut changes which windows have a class.
            "familiarity": self._gate(pinned).signature,
            # Step 4's white point, by identity (P-15). The trained branches' input is
            # optical density, which is a division by this - so a recalibrated slide
            # is different input and a different class map. Not on the BEETLE option,
            # whose input is the photograph and never sees a white point.
            "white_key": calibration_service.white_point(
                upload_id, threshold=report.params.tissue_threshold
            ).key,
        }

    @staticmethod
    def _gate(pinned: model.Pinned) -> familiarity.Gate:
        return familiarity.Gate(
            reference=pinned.familiarity,
            max_flat_share=settings.tissue_type_max_flat_share,
        )

    @staticmethod
    def _cache_key(params: dict[str, Any]) -> dict[str, Any]:
        """The part of a run's parameters that decides whether the cache still applies.

        Everything that changes the class map and nothing that does not. `span` and
        `stride` are excluded because they are derived from the rest; `device` is
        excluded because a CPU and a GPU are meant to produce the same answer, and if
        they do not that is a bug rather than a reason to re-run.

        **`get`, not `[]`, and that is the whole point of this note.** This runs over
        two dicts: the parameters of the run about to start, and the ones recorded
        beside a cached report - which may have been written by an older version that
        had never heard of a key added since. Indexing would raise on it. Reading a
        missing key as `None` makes the comparison simply fail, which re-runs the pass -
        exactly right, because a cache that predates a setting genuinely does not
        describe what the current settings would produce.
        """
        return {
            key: params.get(key)
            for key in (
                "model",
                "model_sha256",
                # The branch and the field of view, because the same checkpoint name
                # cannot occur under two of them but a *changed* choice must invalidate
                # even when the resolved head happens to be the same file.
                "branch",
                "field_of_view_um",
                "input_channel",
                "window_px",
                "mpp",
                "overlap",
                "standardise",
                "gamma",
                "invert_polarity",
                "tile_overlap",
                "tissue_threshold",
                "min_tissue_share",
                "qc_gated",
                "qc_source",
                "tissue_mask_key",
                "tile_min_tissue_share",
                "tile_min_clean_share",
                "white_key",
                "familiarity",
                # The BEETLE option's own inputs. Every one of them changes the mask:
                # `folds` changes what is averaged, `patch_step` changed a checkerboard
                # into a coherent map, and `mask_mpp` is the resolution the answer is
                # kept at. `paint_px` is deliberately absent - it only sizes the live
                # feed, and a cached pass has rendered panels rather than a paint log.
                "per_pixel",
                "folds",
                "patch_step",
                "patch_px",
                "mask_mpp",
                # The uncertainty layer's settings, as one nested dict. They change
                # which windows come back purple and therefore both the rendered panels
                # and the class table, so a cached pass taken under different ones does
                # not describe what the current settings would produce - which is the
                # whole test this key applies.
                #
                # The cost is real and worth naming: re-deriving the layer needs nothing
                # but `classmap.npz`, so a changed threshold could in principle be a
                # redraw rather than another half hour of forward passes. It is not one
                # yet. If tuning these becomes routine, that fast path - not dropping
                # them from this key - is the fix.
                "uncertainty",
            )
        }

    # --- run ----------------------------------------------------------------

    def discard(self, upload_id: str) -> None:
        """Remove everything this step wrote for a slide - and what reads it.

        **The whole directory**, not just the report: `report.json`, `classmap.npz` and
        all four PNGs. Unlinking only the report - which is what `restart` used to do -
        left `asset()` serving the previous choice's `map.png` and `class_map()` handing
        step 9 the previous choice's labels. A stale answer that looks fresh is worse
        than no answer, because nothing downstream can tell.

        Steps 9, 10 and 11 go too, and the reason strengthens down the chain.
        `roi_service.build` caches on **its own** parameters, so a new class map
        otherwise leaves a region of interest that looks current and was drawn from
        labels that no longer exist. Step 10's candidate ids are *area ranks within a
        class map*, so a fresh class map does not merely invalidate its pictures - it
        re-points every id at different tissue. And step 11's refined boundaries were
        traced inside boxes those ids named, so they would be pixel-accurate answers
        about regions nobody chose.

        Refuses while a pass is running, rather than racing it: the worker writes into
        this directory when it finishes, so deleting underneath it would leave exactly
        the half-state this method exists to prevent.
        """
        with self._lock:
            job = self._jobs.get(upload_id)
            if job is not None and job.state in {"queued", "running"}:
                raise TissueTypeError(
                    "a pass is running on this slide - cancel it before changing step "
                    "7, otherwise the run would write its result into a directory "
                    "being deleted underneath it"
                )
            self._jobs.pop(upload_id, None)

        shutil.rmtree(self._dir(upload_id), ignore_errors=True)

        # Imported here rather than at module scope: each of these services imports
        # this one, so a top-level import would be a cycle.
        from app.services.roi_refinement_service import roi_refinement_service
        from app.services.roi_selection_service import roi_selection_service
        from app.services.roi_service import roi_service

        roi_service.discard(upload_id)
        roi_selection_service.discard(upload_id)
        roi_refinement_service.discard(upload_id)

    def class_map_key(self, upload_id: str) -> str | None:
        """A short, stable name for *which* class map this slide currently holds.

        Steps 10 and 11 both cache things that are only meaningful against one class map
        - a candidate id is an area rank within it, a refined boundary was traced inside
        a box that rank named - and neither can tell by looking that the labels beneath
        them have changed. `discard` above is the first defence and covers every path
        that goes through this service; this is the second, and it covers the rest: a
        class map replaced by a restore, a copied data directory, or a run of the
        headless batch script that wrote one directly.

        Built from the run's own cache key - the model, the branch, the field of view,
        the overlap, every parameter `_cache_key` already decided is what makes two
        passes the same pass - plus the file's own fingerprint, so a re-run at identical
        parameters that nonetheless produced a different map is a different key.

        `None` when there is no class map, which is not an error: the callers use it to
        compare against what they stored, and "there is nothing to compare with" and
        "it does not match" lead to the same rebuild.
        """
        cached = self._read_json(upload_id, "report.json")
        if cached is None:
            return None

        path = self._path(upload_id, "classmap.npz")
        if not path.is_file():
            path = self._path(upload_id, "pixelmap.npz")
        try:
            stat = path.stat()
            fingerprint: tuple[Any, ...] = (path.name, stat.st_size, int(stat.st_mtime))
        except OSError:
            return None

        payload = json.dumps(
            [self._cache_key(cached.get("_key", {})), fingerprint],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]  # noqa: S324

    def _erase_stale(self, upload_id: str, params: dict[str, Any]) -> bool:
        """Throw away a cached pass that describes a different choice. Returns whether.

        The negative case matters as much as the positive one: a *matching* choice must
        still return its cache, because that cache is half an hour of somebody's CPU
        and revisiting the screen has to stay instant.
        """
        cached = self._read_json(upload_id, "report.json")
        if not cached:
            return False
        if self._cache_key(cached.get("_key", {})) == self._cache_key(params):
            return False

        logger.info(
            "tissue_type.discarded_stale",
            extra={
                "upload_id": upload_id,
                "was": cached.get("_key", {}).get("model"),
                "now": params.get("model"),
            },
        )
        self.discard(upload_id)
        return True

    def start(
        self,
        upload_id: str,
        *,
        overlap: float | None = None,
        name: str | None = None,
        branch: str | None = None,
        fov: float | None = None,
    ) -> TissueTypeRun:
        """Queue a run, or hand back the cached one when the parameters already match."""
        resolve_ready_path(upload_id=upload_id)  # raises UploadError when not ready
        params = self.resolve_params(
            upload_id, overlap=overlap, name=name, branch=branch, fov=fov
        )

        cached = self._read_json(upload_id, "report.json")
        if cached and self._cache_key(cached.get("_key", {})) == self._cache_key(params):
            return TissueTypeRun(**cached["run"])

        # The choice moved. Everything on disk describes the old one, so it goes now
        # rather than sitting beside the new numbers until the pass overwrites it.
        self._erase_stale(upload_id, params)

        with self._lock:
            existing = self._jobs.get(upload_id)
            if existing and existing.state in {"queued", "running"}:
                return self._as_run(existing)
            if any(job.state == "running" for job in self._jobs.values()):
                raise TissueTypeError(
                    "a tissue-type run is already in progress for another slide; only one "
                    "runs at a time so neither is starved of cores"
                )
            self._jobs[upload_id] = Job(upload_id=upload_id, params=params)

        return self._as_run(self._jobs[upload_id])

    def cancel(self, upload_id: str) -> TissueTypeRun:
        """Ask a running pass to stop, and report the state that leaves.

        Sets a flag the worker's progress callback checks after every block, so the
        stop lands within a couple of seconds rather than at the end of a slide. What
        it does *not* do is discard anything: this step is the only one in the
        pipeline expensive enough that half a pass is worth something, and a viewer
        who cancels at 80% and immediately restarts should not be punished for it -
        so a restart re-runs from the beginning but a **finished** cached pass is
        never touched by a cancel.

        Cancelling nothing is not an error. A viewer clicking cancel as the last block
        lands has done nothing wrong, and reporting a conflict for it would be
        reporting the race rather than the outcome.
        """
        with self._lock:
            job = self._jobs.get(upload_id)
            if job is not None and job.state in {"queued", "running"}:
                job.cancelled = True
                job.message = "stopping"

        return self.state(upload_id)

    def restart(
        self,
        upload_id: str,
        *,
        overlap: float | None = None,
        name: str | None = None,
        branch: str | None = None,
        fov: float | None = None,
    ) -> TissueTypeRun:
        """Throw the cached pass away and run again.

        The one thing `start` cannot do: it returns the cache when the parameters
        match, which is what makes revisiting the screen instant, and is exactly wrong
        when the viewer is asking for a fresh run. Deleting the report rather than
        adding a `force` flag keeps `start` with one meaning.
        """
        resolve_ready_path(upload_id=upload_id)

        # The whole directory, not `report.json` alone. Removing only the report left
        # `classmap.npz` and the four PNGs behind, so `asset()` kept serving the
        # previous pass's pictures until the new one happened to overwrite them - and
        # if the new pass failed, indefinitely.
        self.discard(upload_id)

        return self.start(
            upload_id, overlap=overlap, name=name, branch=branch, fov=fov
        )

    def state(self, upload_id: str, *, painted_since: int | None = None) -> TissueTypeRun:
        """Current run state: the live job if there is one, else whatever is cached.

        `painted_since` asks for the windows classified after that point in the run's
        paint log, so a screen can draw the pass onto the slide as it happens. Omitted
        means "no cells", which is what every caller that only wants the numbers wants:
        the feed is a whole slide's grid over the life of a pass, and a caller that
        never asks should never carry any of it.
        """
        with self._lock:
            job = self._jobs.get(upload_id)
        if job is not None:
            return self._as_run(job, painted_since=painted_since)

        cached = self._read_json(upload_id, "report.json")
        if cached:
            # A pass read back from disk has no live paint and needs none - it has four
            # rendered panels. Any paint keys an older report happens to carry are
            # dropped rather than replayed as though the run were still going.
            run = dict(cached["run"])
            for key in (
                "paint",
                "paintedCells",
                "paintedCursor",
                "paintedMasks",
                "painted_cells",
                "painted_cursor",
                "painted_masks",
            ):
                run.pop(key, None)
            return TissueTypeRun(**run)
        return TissueTypeRun(upload_id=upload_id, state="idle")

    def _as_run(self, job: Job, *, painted_since: int | None = None) -> TissueTypeRun:
        """The job as the API reports it, with a slice of the paint feed on request.

        The slice is taken with one list operation rather than element by element,
        which is what makes it safe to read while the worker is appending - see `Job`.
        """
        cells: list[int] = []
        masks: list[str] = []
        cursor = 0
        paint: TissueTypePaint | None = None

        if painted_since is not None:
            paint = TissueTypePaint(**job.paint) if job.paint else None
            log = job.painted
            first = max(0, int(painted_since))
            start = first * 3
            cells = log[start : start + _PAINT_CHUNK * 3]
            cursor = first + len(cells) // 3

            # Sliced to exactly the windows `cells` carries, and taken *after* it, so
            # the two cannot disagree even though the worker is appending to both: a
            # mask whose triple has not landed yet is simply not sent, and the next poll
            # picks it up. The other order could send a mask for a window whose row and
            # column the client does not have.
            if job.painted_masks:
                masks = job.painted_masks[first:cursor]

        return TissueTypeRun(
            upload_id=job.upload_id,
            state=job.state,
            phase=job.phase,
            message=job.message,
            progress=round(job.progress, 4),
            done=job.done,
            total=job.total,
            started_at=job.started_at,
            finished_at=job.finished_at,
            duration_seconds=round(job.duration, 1) if job.duration else None,
            error=job.error,
            params=TissueTypeParams(**job.params),
            paint=paint,
            painted_cells=cells,
            painted_masks=masks,
            painted_cursor=cursor,
        )

    def execute(self, upload_id: str) -> None:
        """The background job. Never raises - failures land on the job record."""
        with self._lock:
            job = self._jobs.get(upload_id)
        if job is None or job.state != "queued":
            return

        with self._run_lock:
            job.state = "running"
            job.started = time.monotonic()
            job.started_at = _now()
            try:
                self._run(job)
                job.state = "ready"
                job.phase = None
                job.message = "complete"
                job.progress = 1.0
            except RunCancelled as stopped:
                # A decision, not a fault. Logged at info and reported as its own
                # state, so the screen does not apologise for what the viewer asked
                # for and the log does not bury real errors among cancellations.
                logger.info("step 8 run cancelled for %s: %s", upload_id, stopped)
                job.state = "cancelled"
                job.phase = None
                job.message = str(stopped)
            except Exception as exc:  # noqa: BLE001 - reported to the client verbatim
                logger.exception("step 8 run failed for %s", upload_id)
                job.state = "failed"
                job.error = str(exc)
            finally:
                job.duration = time.monotonic() - job.started
                job.finished_at = _now()

    def _run(self, job: Job) -> None:
        """Build the grid, run the model over every window on it, render and describe.

        **One method with one fork, rather than two methods.** Everything before the
        model runs - step 7's index, step 3's mask, the slide's resolution, the window
        grid and its two gates - is identical on all three branches and is where most of
        the ways this step can be wrong live: a grid built at parameters the run was not
        priced under, or gated against a mask from a different threshold. Splitting
        `_run` in two would duplicate that prologue and let the copies drift, which is a
        worse failure than the `if` below is ugly.

        What forks is only: which model is loaded, which pass runs it, and which renderer
        and describer read the result. `per_pixel` is the fork and it comes from the
        resolved parameters, not from a setting.
        """
        upload_id = job.upload_id
        params = job.params
        per_pixel = bool(params.get("per_pixel"))

        pinned = None
        loaded = None
        if per_pixel:
            try:
                loaded = beetle.load(
                    float(params["field_of_view_um"]),
                    folds=len(params.get("folds") or (0,)),
                )
            except beetle.BeetleError as exc:
                raise TissueTypeError(str(exc)) from exc
            model_name = loaded.name
            window_size, window_mpp = loaded.window_px, loaded.mpp
        else:
            try:
                pinned = model.load_pinned(params["model"])
            except model.ModelError as exc:
                raise TissueTypeError(str(exc)) from exc
            model_name = pinned.name
            window_size, window_mpp = pinned.tile_px, pinned.mpp

        # All four rebuilt at the parameters this run was *priced* under, taken from
        # the job rather than from the defaults. Before this, the worker asked for
        # every one of them bare: the index came back at the configured field of view
        # and the mask and white point at the configured threshold, so the grid the
        # windows were gated against was not the grid `resolve_params` had described.
        selection = tiling_service.selection(upload_id)
        index = tiling_service.tile_index(
            upload_id,
            branch=selection.branch,
            fov=selection.field_of_view_um,
            overlap=selection.overlap,
            threshold=selection.tissue_threshold,
        )
        footprint = tiling_service.report(
            upload_id,
            branch=selection.branch,
            fov=selection.field_of_view_um,
            overlap=selection.overlap,
            threshold=selection.tissue_threshold,
        )
        footprint_mask = tissue_service.footprint(
            upload_id, threshold=params.get("tissue_threshold")
        )
        # **No white point on the BEETLE option, and that is the contract rather than an
        # omission.** Step 4's illumination surface exists to turn intensity into dye
        # concentration, which is what the two ResNet branches' input is built out of.
        # BEETLE's input is the photograph divided by 255; dividing it by a lamp profile
        # first would hand a network fitted on undivided images something it never saw.
        white = (
            None
            if per_pixel
            else calibration_service.white_point(
                upload_id, threshold=params.get("tissue_threshold")
            )
        )
        path = resolve_ready_path(upload_id=upload_id)

        job.phase = "grid"
        job.message = "laying the model's grid over the tissue"

        with open_slide(path) as reader:
            base_mpp = reader.mpp
            if not base_mpp:
                raise TissueTypeError(
                    "this slide records no microns-per-pixel, so there is no physical scale "
                    "to read the model's window at. Supply one on step 1 first."
                )

            grid = inference.build_grid(
                index,
                base_mpp=float(base_mpp),
                slide_size=reader.dimensions,
                # The window's geometry, from whichever model resolved. Both report a
                # side in their own pixels and the resolution those pixels are read at,
                # and `size * mpp` is the field of view either way - which is what lets
                # one grid builder serve a 224 px window at 2 um/px and an 896 px one at
                # 0.5 um/px without knowing which it has.
                size=window_size,
                mpp=window_mpp,
                overlap=float(params["overlap"]),
                max_windows=settings.tissue_type_max_windows,
                # Step 3's mask, so this step can apply a tissue gate of its own on the
                # *window*. Step 7's gate is on its 512 px tile and is set low on
                # purpose; inheriting it alone let the model see windows that were 89%
                # glass, which is where the in-situ rim came from.
                tissue=footprint_mask.mask,
                mask_mpp=footprint_mask.mpp,
                min_tissue_share=float(params["min_tissue_share"]),
            )
            params["span"] = grid.span
            params["stride"] = grid.stride

            job.phase = "classifying"
            job.total = grid.windows

            # What the progress screen needs to paint this pass onto the slide. Set
            # here rather than in `start`, because the grid is what it describes and
            # the grid does not exist until now - a screen polling before this gets a
            # null and says the grid is being laid, which is what is happening.
            #
            # The geometry is the grid's, so it follows whichever model step 7 resolved
            # to: what reaches the client is `span` and `stride` in level-0 pixels, and
            # a 224 px window at 2 um/px and an 896 px one at 0.5 um/px land on the same
            # two numbers. The palette and legend are the running model's own, so a
            # three-class pass and a five-class one paint their own classes without the
            # client holding either list.
            if per_pixel:
                codes = range(len(beetle.PIXEL_CLASSES))
                palette = [_hex(beetle.CLASS_COLOURS[code]) for code in codes]
                legend = [beetle.CLASS_LABELS[code] for code in codes]
            else:
                codes = range(len(CLASS_NAMES))
                palette = [_hex(CLASS_COLOURS[code]) for code in codes]
                legend = [CLASS_LABELS[code] for code in codes]

            job.paint = {
                "cols": grid.cols,
                "rows": grid.rows,
                "span": grid.span,
                "stride": grid.stride,
                "slide_width": grid.slide_width,
                "slide_height": grid.slide_height,
                "colours": palette,
                "labels": legend,
                "per_pixel": per_pixel,
                "mask_px": int(params.get("paint_px") or 0) if per_pixel else 0,
            }

            if per_pixel:
                job.message = (
                    f"{grid.windows:,} windows through BEETLE, "
                    f"{beetle.patches_per_window(grid.field_um, loaded.patch) * len(loaded.nets):,}"
                    " forward passes each"
                )
                result = pixels.segment(
                    grid,
                    loaded,
                    # Step 5's tile read and nothing else - see `beetle_reader`.
                    read_window=self.beetle_reader(reader, base_mpp=float(base_mpp)),
                    mask_mpp=float(params["mask_mpp"]),
                    patch_step=float(params["patch_step"]),
                    batch_size=int(params["batch_size"]),
                    paint_px=int(params["paint_px"]),
                    block_windows=int(params["block_windows"]),
                    progress=self._progress(job),
                    painted=self._pixel_painter(job),
                    should_stop=self._stopper(job),
                )
                params["patches"] = result.patches
            else:
                job.message = f"{grid.windows:,} windows through {model_name}"
                result = inference.classify(
                    grid,
                    net=pinned.net,
                    read_window=self.window_reader(
                        reader,
                        white,
                        base_mpp=float(base_mpp),
                        channel=pinned.channel,
                    ),
                    channel=pinned.channel,
                    standardise=pinned.standardise,
                    gamma=pinned.gamma,
                    invert=pinned.invert_polarity,
                    # The checkpoint's own decision rule. `None` for everything published
                    # before v3, which was validated under a plain argmax; serving a model
                    # that carries a tau without applying it is a different, worse model
                    # and nothing downstream would notice.
                    tau=pinned.tau,
                    block_windows=int(params["block_windows"]),
                    batch_size=int(params["batch_size"]),
                    progress=self._progress(job),
                    painted=self._painter(job),
                    should_stop=self._stopper(job),
                    gate=self._gate(pinned),
                )

                # **The second stage, and it runs here rather than inside `classify`.**
                # Two of its three terms are functions of a window's neighbours and the
                # third of the connected component it lands in, so none of them exists
                # until the last window has come back - which is also why the live paint
                # feed above stays three-class and only the finished report has four.
                #
                # BEETLE never reaches this: it answers per pixel over its own five
                # classes, and `CANDIDATE` and `RIVAL` index ours.
                result = replace(
                    result,
                    uncertainty=uncertainty.derive(
                        result.labels,
                        result.probabilities,
                        # The result's grid, not the one laid above: refused windows
                        # have left `inside`, and the layer must not see them.
                        result.grid.inside,
                        candidate=CANDIDATE,
                        rival=RIVAL,
                        # Microns, not cells - see `uncertainty.derive`. The stride is
                        # in level-0 pixels and `base_mpp` is what makes it a distance,
                        # so a changed overlap moves the sampling and not the
                        # neighbourhood.
                        stride_um=grid.stride * grid.base_mpp,
                        cell_mm2=grid.cell_mm2,
                        params=_uncertainty_params(),
                    ),
                )

        job.phase = "rendering"
        job.message = "drawing the class map"
        thumbnail, _ = calibration_service.thumbnail(upload_id)
        if per_pixel:
            self._render_pixels(upload_id, thumbnail, result)
            self._store_pixels(upload_id, result)
        else:
            self._render(upload_id, thumbnail, result)
            self._store(upload_id, result)

        # **Mark the job finished before describing it.** The report embeds a run block
        # and that block is what `start` and `state` hand back on every later visit, so
        # if it were captured while the job still said "running" then a *completed*
        # pass would report itself as running for ever: the endpoint would never
        # re-queue it (the state is not "queued") and the screen would poll a run that
        # can never change. Everything the run does is done by this point - what is
        # left is writing the file - so recording it as finished here is accurate as
        # well as necessary. `execute` sets the same fields again; that is harmless and
        # keeps the failure path honest.
        job.state = "ready"
        job.phase = None
        job.message = "complete"
        job.progress = 1.0
        job.duration = time.monotonic() - job.started
        job.finished_at = _now()

        describe = self._describe_pixels if per_pixel else self._describe
        report = describe(
            result,
            upload_id=upload_id,
            filename=get_record(upload_id=upload_id).filename,
            params=params,
            pinned=loaded if per_pixel else pinned,
            tiles_kept=footprint.funnel.clean,
            tile_px=footprint.params.tile_size,
            tile_um=footprint.params.tile_um,
            tissue_mm2=footprint.coverage.covered_mm2,
            run=self._as_run(job),
        )

        payload = report.model_dump(by_alias=True)
        payload["_key"] = params
        self._write_json(upload_id, "report.json", payload)

        logger.info(
            "tissue_type.complete",
            extra={
                "upload_id": upload_id,
                "branch": params.get("branch"),
                "per_pixel": per_pixel,
                "windows": result.windows_done if per_pixel else result.classified,
                "tumour_content": round(result.tumour_content, 4),
                "seconds": result.seconds,
            },
        )

    def _progress(self, job: Job):
        """Progress callback for the classifier, mapped onto the phase's own span."""

        def report(done: int, total: int) -> None:
            low, high = _PHASE_SPAN["classifying"]
            fraction = (done / total) if total else 1.0
            job.done = done
            job.total = total
            job.progress = low + (high - low) * min(1.0, fraction)

        return report

    @staticmethod
    def _painter(job: Job):
        """Paint feed for the classifier: every window's class, as it is decided.

        Appends rather than replaces, because a poll asks for what it has not seen and
        not for the state of the whole grid: the screen accumulates, so the same window
        never has to travel twice.
        """

        def append(cells: tuple[tuple[int, int, int], ...]) -> None:
            log = job.painted
            for row, col, label in cells:
                log.append(row)
                log.append(col)
                log.append(label)

        return append

    @staticmethod
    def _pixel_painter(job: Job):
        """Paint feed for the BEETLE pass: each window's pixel mask, as it is decided.

        The same triples the ResNet feed appends, plus the mask beside them. Both are
        appended so a client can draw either - the triple's label is the window's
        dominant class, which is what a legend counts and what the screen falls back to
        if it cannot decode a mask.

        **The two lists are appended in step and never independently.** `painted_cursor`
        counts windows and is used to slice both, so a mask missing for one window would
        shift every mask after it onto the wrong cell - a picture that is wrong rather
        than absent. Appending the mask first would open the same window between the two
        appends for a poll to land in, so the triple goes last: a poll that catches the
        gap sees a shorter `painted` and simply asks again.
        """

        def append(cells: tuple[tuple[int, int, int, str], ...]) -> None:
            for row, col, label, mask in cells:
                job.painted_masks.append(mask)
                job.painted.append(row)
                job.painted.append(col)
                job.painted.append(label)

        return append

    @staticmethod
    def _stopper(job: Job):
        """Whether this job has been asked to stop. Checked once per block."""
        return lambda: job.cancelled

    @classmethod
    def window_reader(
        cls,
        reader: Any,
        white: Any,
        *,
        base_mpp: float,
        channel: str = model_input.CHANNEL_HAEMATOXYLIN,
    ):
        """The reader for whichever input contract the checkpoint declared.

        One factory rather than a branch at the call site, so the decision is taken
        once, next to the composition it selects, and `classify` never has to know
        which of the two it was handed.

        On `rgb_he` there is **no white point and no deconvolution**. Both are
        statements about dye concentration and this branch makes neither: the model
        gets the photograph. Passing the tile through `field_for` anyway would be
        dividing a colour image by a lamp profile before handing it to a network fitted
        on undivided ones - which is precisely the training/serving drift the whole
        input contract exists to refuse.
        """
        if channel == model_input.CHANNEL_RGB_HE:
            return cls.rgb_reader(reader, base_mpp=base_mpp)
        return cls.haematoxylin_reader(reader, white, base_mpp=base_mpp)

    @staticmethod
    def rgb_reader(reader: Any, *, base_mpp: float):
        """Level-0 addresses to sRGB pixels, and nothing else done to them.

        The same `read_tile` as the haematoxylin path - including its rule that any
        resampling happens in intensity space, before the logarithm, with area
        averaging - and then it stops. That is the whole H&E input.
        """

        def read(x: int, y: int, span: int, size: int) -> np.ndarray:
            tile = read_tile(
                reader,
                x=x,
                y=y,
                target_mpp=span * base_mpp / size,
                size=size,
                base_mpp=base_mpp,
            )
            return tile.rgb.astype(np.uint8)

        return read

    @staticmethod
    def beetle_reader(reader: Any, *, base_mpp: float):
        """Level-0 addresses to sRGB pixels at BEETLE's own spacing, and nothing else.

        Identical in body to `rgb_reader` and kept separate on purpose, because the two
        are the same three lines for different reasons and one of them may need to change
        without the other. `rgb_reader` serves the H&E branch, whose *checkpoint* was
        fitted on undivided sRGB and whose manifest says so; this serves a downloaded
        release whose published preprocessing is `RGBTo01Normalization`. A future change
        to either contract must not silently alter the other.

        `read_tile` is step 5's, including its rule that any resampling happens in
        intensity space with area averaging - which matters here more than anywhere else
        in the step, because every window is resampled: the slide is 0.22 um/px and
        BEETLE reads 0.5, so this is always a downsample of about a factor of two, and
        doing it after a logarithm would blur dye concentration rather than light.

        There is no white point and no deconvolution. Both are statements about dye
        concentration and this branch makes neither: the network gets the photograph.
        """

        def read(x: int, y: int, span: int, size: int) -> np.ndarray:
            tile = read_tile(
                reader,
                x=x,
                y=y,
                target_mpp=span * base_mpp / size,
                size=size,
                base_mpp=base_mpp,
            )
            return tile.rgb.astype(np.uint8)

        return read

    @staticmethod
    def haematoxylin_reader(reader: Any, white: Any, *, base_mpp: float):
        """The one function in this step that turns a position into model input pixels.

        Public because gate G7b re-reads windows through it - see
        `scripts/check_tissue_geometry.py`. That check exists to catch a wrong *block
        offset*, so it has to read pixels the same way the run did and differ only in
        the addressing; a second composition over there could fail for its own reasons
        and would prove nothing about the one that matters.

        A composition, not an implementation, and every piece belongs to the step that
        owns it: `read_tile` is step 5's - including its rule that any resampling
        happens in intensity space, before the logarithm, with area averaging -
        `white.field_for` is step 4's, and `haematoxylin_od` is `optical_density` plus
        step 6's `separate`. Nothing here is reimplemented, which is what makes the
        pixels the model is served the same pixels it was fitted on.

        The white point is evaluated **over the block**, not over the slide, because
        step 4 may have justified an illumination surface rather than one flat triple.
        Dividing a block at the edge of the frame by the centre's white point would put
        the lamp's falloff into the model's input.
        """

        def read(x: int, y: int, span: int, size: int) -> np.ndarray:
            tile = read_tile(
                reader,
                x=x,
                y=y,
                target_mpp=span * base_mpp / size,
                size=size,
                base_mpp=base_mpp,
            )
            field = white.field_for(x=tile.x, y=tile.y, size=tile.size, mpp=tile.mpp)
            return model_input.haematoxylin_od(
                tile.rgb.astype(np.float32), field, od_floor=white.od_floor
            )

        return read

    # --- persisting ---------------------------------------------------------

    def _render(self, upload_id: str, thumbnail: np.ndarray, class_map: ClassMap) -> None:
        """Write the four panels. Every class is drawn on `map`; the filter is a query."""
        directory = self._dir(upload_id)
        directory.mkdir(parents=True, exist_ok=True)
        shape = (thumbnail.shape[0], thumbnail.shape[1])

        (directory / "map.png").write_bytes(overlay.map_png(thumbnail, class_map))
        (directory / "flat.png").write_bytes(overlay.flat_png(class_map, shape))
        (directory / "scored.png").write_bytes(overlay.scored_png(thumbnail, class_map))
        (directory / "confidence.png").write_bytes(overlay.confidence_png(class_map, shape))
        (directory / "uncertainty.png").write_bytes(
            overlay.uncertainty_png(class_map, shape)
        )

    def _store(self, upload_id: str, class_map: ClassMap) -> None:
        """Persist the class map itself, for step 9 and for the class filter.

        Probabilities and not only labels: step 10 smooths *probabilities*, because
        averaging argmaxes is a vote and a vote discards the confidence a smoothing
        step needs. The grid geometry travels with them so the arrays can be placed on
        the slide again without re-deriving anything.
        """
        directory = self._dir(upload_id)
        directory.mkdir(parents=True, exist_ok=True)
        grid = class_map.grid

        np.savez_compressed(
            directory / "classmap.npz",
            labels=class_map.labels,
            probabilities=class_map.probabilities,
            inside=grid.inside,
            geometry=np.array(
                [
                    grid.cols,
                    grid.rows,
                    grid.size,
                    grid.stride_px,
                    grid.span,
                    grid.stride,
                    grid.overlap,
                    grid.mpp,
                    grid.base_mpp,
                    grid.slide_width,
                    grid.slide_height,
                ],
                dtype=np.float64,
            ),
            counters=np.array(
                [class_map.batches, class_map.blocks_read, class_map.seconds],
                dtype=np.float64,
            ),
            # Which windows the gate refused and how far each measured. `inside` above
            # is already the gated one; these are the record of why it shrank.
            **(
                {}
                if class_map.refused is None
                else {
                    "refused": class_map.refused,
                    "familiarity_distance": class_map.distance,
                    "familiarity_gate": np.array(
                        class_map.gate.signature if class_map.gate else ""
                    ),
                }
            ),
            # The uncertainty layer, stored rather than re-derived on read. It *is* a
            # pure function of `labels` and `probabilities` above, so recomputing would
            # give the same answer - but only under the same settings, and the settings
            # can move between the run and the read. Storing it is what keeps the class
            # table, the panels on disk and a filtered redraw describing one pass.
            #
            # Three terms and the verdict, not just the verdict: which route flagged a
            # region is the first thing anyone disagreeing with it needs.
            **(
                {}
                if class_map.uncertainty is None
                else {
                    "uncertainty_margin": class_map.uncertainty.margin,
                    "uncertainty_disagreement": class_map.uncertainty.disagreement,
                    "uncertainty_implausibility": class_map.uncertainty.implausibility,
                    "uncertainty_score": class_map.uncertainty.score,
                    "uncertainty_unknown": class_map.uncertainty.unknown,
                    "uncertainty_stats": np.array(
                        [
                            class_map.uncertainty.sigma_cells,
                            class_map.uncertainty.components,
                            class_map.uncertainty.flagged_components,
                            class_map.uncertainty.largest_component_mm2,
                        ],
                        dtype=np.float64,
                    ),
                    "uncertainty_params": np.array(
                        [
                            class_map.uncertainty.params.sigma_um,
                            class_map.uncertainty.params.area_ref_mm2,
                            class_map.uncertainty.params.area_max_mm2,
                            class_map.uncertainty.params.fill_lo,
                            class_map.uncertainty.params.fill_hi,
                            class_map.uncertainty.params.min_shape_mm2,
                            class_map.uncertainty.params.min_shape_windows,
                            class_map.uncertainty.params.threshold,
                        ],
                        dtype=np.float64,
                    ),
                }
            ),
        )

    def _render_pixels(
        self, upload_id: str, thumbnail: np.ndarray, pixel_map: PixelMap
    ) -> None:
        """Write the four panels from the pixel mask. Same four names as the ResNet ones.

        Same names because they answer the same four questions, so a viewer switching
        options on step 7 does not have to learn a new screen and `panel()` needs no
        per-branch routing table. `pixel_overlay` says why they are a separate renderer.
        """
        directory = self._dir(upload_id)
        directory.mkdir(parents=True, exist_ok=True)
        shape = (thumbnail.shape[0], thumbnail.shape[1])

        (directory / "map.png").write_bytes(pixel_overlay.map_png(thumbnail, pixel_map))
        (directory / "flat.png").write_bytes(pixel_overlay.flat_png(pixel_map, shape))
        (directory / "scored.png").write_bytes(
            pixel_overlay.scored_png(thumbnail, pixel_map)
        )
        (directory / "confidence.png").write_bytes(
            pixel_overlay.confidence_png(pixel_map, shape)
        )

    def _store_pixels(self, upload_id: str, pixel_map: PixelMap) -> None:
        """Persist the pixel mask and the per-window summary beside it.

        Written to `pixelmap.npz` rather than `classmap.npz`, and the name matters: they
        hold different things and nothing should be able to load one believing it has the
        other. `class_map` refuses when only this exists, and says which branch ran.

        Compressed, and it compresses very well - a whole-slide mask is mostly the
        `OUTSIDE` sentinel outside the section, and inside it is large flat regions of
        five values. The 50 MB uint8 array for a 28 mm slide lands at a few hundred kB.
        """
        directory = self._dir(upload_id)
        directory.mkdir(parents=True, exist_ok=True)
        grid = pixel_map.grid

        np.savez_compressed(
            directory / "pixelmap.npz",
            mask=pixel_map.mask,
            windows=pixel_map.windows,
            labels=pixel_map.labels,
            inside=grid.inside,
            geometry=np.array(
                [
                    grid.cols,
                    grid.rows,
                    grid.size,
                    grid.stride_px,
                    grid.span,
                    grid.stride,
                    grid.overlap,
                    grid.mpp,
                    grid.base_mpp,
                    grid.slide_width,
                    grid.slide_height,
                    pixel_map.mask_mpp,
                ],
                dtype=np.float64,
            ),
            counters=np.array(
                [
                    pixel_map.windows_done,
                    pixel_map.patches,
                    pixel_map.blocks_read,
                    pixel_map.seconds,
                ],
                dtype=np.float64,
            ),
            # Stored rather than recomputed on load: it is a mean over every pixel the
            # pass produced *before* they were argmaxed into the mask, so the mask alone
            # cannot yield it back.
            class_confidence=np.asarray(pixel_map.class_confidence, dtype=np.float64),
        )

    def pixel_map(self, upload_id: str) -> PixelMap:
        """The cached pixel mask, rehydrated. For the class filter and for step 10 on.

        Read back from disk rather than held in memory, for `class_map`'s reason: a
        runner driving the whole pipeline in one call is not the only caller, and a
        restart must not lose hours of CPU.
        """
        path = self._path(upload_id, "pixelmap.npz")
        if not path.is_file():
            raise TissueTypeError(
                "no per-pixel mask for this slide - either step 8 has not run "
                f"(POST /tissue-type/{upload_id}/run) or it ran on one of the two "
                "trained options, which answer once per window and write a class map "
                "instead. Choose `Run by BEETLE` on step 7 for a per-pixel mask."
            )

        with np.load(path) as stored:
            geometry = stored["geometry"]
            grid = WindowGrid(
                cols=int(geometry[0]),
                rows=int(geometry[1]),
                size=int(geometry[2]),
                stride_px=int(geometry[3]),
                span=int(geometry[4]),
                stride=int(geometry[5]),
                overlap=float(geometry[6]),
                mpp=float(geometry[7]),
                base_mpp=float(geometry[8]),
                slide_width=int(geometry[9]),
                slide_height=int(geometry[10]),
                inside=stored["inside"],
            )
            mask = stored["mask"]
            mask_mpp = float(geometry[11])
            windows = stored["windows"]
            labels = stored["labels"]
            counters = stored["counters"]
            # `.get` rather than `[]`: a pass cached before this figure was measured is
            # still a valid pixel map, and zero reads on screen as "not measured" rather
            # than making the whole cache unreadable.
            confidence = (
                stored["class_confidence"]
                if "class_confidence" in stored
                else np.zeros(len(beetle.PIXEL_CLASSES))
            )

        classes = len(beetle.PIXEL_CLASSES)
        counts = tuple(int((mask == code).sum()) for code in range(classes))
        pixel_mm2 = (mask_mpp / 1000.0) ** 2
        return PixelMap(
            grid=grid,
            mask=mask,
            mask_mpp=mask_mpp,
            windows=windows,
            labels=labels,
            counts=counts,
            areas_mm2=tuple(round(count * pixel_mm2, 4) for count in counts),
            class_confidence=tuple(float(value) for value in confidence),
            windows_done=int(counters[0]),
            patches=int(counters[1]),
            blocks_read=int(counters[2]),
            seconds=float(counters[3]),
        )

    def class_map(self, upload_id: str) -> ClassMap:
        """The cached class map, rehydrated. Step 9's input.

        Read back from disk rather than held in memory, so step 9 does not depend on
        step 8 having run in the same process - which is exactly the case a pipeline
        runner driving the whole sequence in one call does *not* satisfy on a restart.
        """
        path = self._path(upload_id, "classmap.npz")
        if not path.is_file():
            # **Naming the BEETLE case explicitly, because the generic message would be
            # a lie there.** Step 8 may well have run and taken hours; what it produced
            # is a per-pixel mask over BEETLE's five classes, and step 9 is written
            # against a per-window map over our three - `SCORED` indexes that three, and
            # `roi_mask.build` smooths cells. Telling a viewer to "run step 8 first"
            # when they just did would send them round the same hours again.
            if self._path(upload_id, "pixelmap.npz").is_file():
                raise TissueTypeError(
                    "step 8 ran the BEETLE option on this slide, which answers once per "
                    "pixel over its own five classes. Step 9 reads a per-window map over "
                    "this pipeline's three classes, so it cannot use that result yet - "
                    "the per-pixel mask is at `pixel_map()` and step 9 has no adapter "
                    "for it. Re-run step 8 on one of the two trained options to continue "
                    "down the pipeline."
                )
            raise TissueTypeError(
                f"no class map for this slide yet - run step 8 first "
                f"(POST /tissue-type/{upload_id}/run)"
            )

        with np.load(path) as stored:
            geometry = stored["geometry"]
            grid = WindowGrid(
                cols=int(geometry[0]),
                rows=int(geometry[1]),
                size=int(geometry[2]),
                stride_px=int(geometry[3]),
                span=int(geometry[4]),
                stride=int(geometry[5]),
                overlap=float(geometry[6]),
                mpp=float(geometry[7]),
                base_mpp=float(geometry[8]),
                slide_width=int(geometry[9]),
                slide_height=int(geometry[10]),
                inside=stored["inside"],
            )
            labels = stored["labels"]
            probabilities = stored["probabilities"]
            counters = stored["counters"]
            layer = self._stored_layer(stored)
            # Absent on a map written before the gate - read as "not gated".
            refused = stored["refused"] if "refused" in stored.files else None
            distance = (
                stored["familiarity_distance"]
                if "familiarity_distance" in stored.files
                else None
            )

        counts = tuple(int((labels == label).sum()) for label in range(len(CLASS_NAMES)))
        cell = grid.cell_mm2
        return ClassMap(
            grid=grid,
            labels=labels,
            probabilities=probabilities,
            counts=counts,  # type: ignore[arg-type]
            areas_mm2=tuple(round(count * cell, 4) for count in counts),  # type: ignore[arg-type]
            batches=int(counters[0]),
            blocks_read=int(counters[1]),
            seconds=float(counters[2]),
            uncertainty=layer,
            refused=refused,
            distance=distance,
        )

    @staticmethod
    def _stored_layer(stored: Any) -> uncertainty.UncertaintyLayer | None:
        """The uncertainty layer out of an npz, or None on a map written before it.

        **None rather than a re-derivation.** A map from an older run genuinely has no
        layer, and inventing one at read time would attach today's thresholds to a
        report that was written without them and whose panels on disk show three
        colours. `display_labels` falls back to the raw labels, the class table falls
        back to three rows, and the pass reads as what it was.
        """
        if "uncertainty_unknown" not in stored.files:
            return None

        stats = stored["uncertainty_stats"]
        recorded = stored["uncertainty_params"]
        return uncertainty.UncertaintyLayer(
            margin=stored["uncertainty_margin"],
            disagreement=stored["uncertainty_disagreement"],
            implausibility=stored["uncertainty_implausibility"],
            score=stored["uncertainty_score"],
            unknown=stored["uncertainty_unknown"],
            params=UncertaintyParams(
                sigma_um=float(recorded[0]),
                area_ref_mm2=float(recorded[1]),
                area_max_mm2=float(recorded[2]),
                fill_lo=float(recorded[3]),
                fill_hi=float(recorded[4]),
                min_shape_mm2=float(recorded[5]),
                min_shape_windows=int(recorded[6]),
                threshold=float(recorded[7]),
            ),
            sigma_cells=float(stats[0]),
            components=int(stats[1]),
            flagged_components=int(stats[2]),
            largest_component_mm2=float(stats[3]),
        )

    # --- reading ------------------------------------------------------------

    def report(self, upload_id: str) -> TissueTypeReport:
        """The cached report, or a refusal naming the run that has not happened."""
        cached = self._read_json(upload_id, "report.json")
        if cached is None:
            raise TissueTypeError(
                f"step 8 has not run on this slide yet - POST /tissue-type/{upload_id}/run. "
                "It is a job rather than a request because a whole-slide pass is tens of "
                "thousands of forward passes."
            )
        cached.pop("_key", None)
        return TissueTypeReport.model_validate(cached)

    def emitted_classes(self, upload_id: str) -> tuple[str, ...]:
        """The class names the pass on this slide actually emitted, in code order.

        Five when the BEETLE option ran, three otherwise. Read off which artefact is on
        disk, because the class filter is a GET that may arrive long after the pass from
        a client that knows only the panel names - and because it has to be *validated*:
        a filter naming class 4 must be a refusal on a three-class pass rather than a
        silently-ignored token, or a viewer switching a class off would watch nothing
        change and believe the denominator had moved.

        The three-class answer is also what an un-run slide gets, which is right: it is
        what a run would emit unless the viewer picks BEETLE on step 7, and the panel
        request is about to fail for the more useful reason that nothing has run.
        """
        if self._path(upload_id, "pixelmap.npz").is_file():
            return beetle.PIXEL_CLASSES
        return DISPLAY_NAMES

    def panel(
        self, upload_id: str, name: str, *, classes: frozenset[int] | None = None
    ) -> bytes:
        """One panel. Redrawn when a class filter is asked for, served from disk if not.

        The filter is the guide's per-class opacity toggle, and it is applied on the
        server so an unselected class is *absent* from the picture rather than
        recoloured - a viewer switching fat off should watch the tissue leave the map.
        """
        if name not in overlay.PANELS:
            raise TissueTypeError(
                f"unknown panel {name!r}; expected one of {sorted(overlay.PANELS)}"
            )

        # **The one panel the two branches do not share**, refused by name rather than
        # left to fail as a missing file. A per-pixel pass never derives an uncertainty
        # layer - that layer indexes this pipeline's three classes and measures connected
        # components on a window grid, and BEETLE has five classes and no window grid -
        # so `uncertainty.png` was never written and "no such file" would be a confusing
        # way to say a true thing.
        if name == "uncertainty" and self._path(upload_id, "pixelmap.npz").is_file():
            raise TissueTypeError(
                "step 8 ran the BEETLE option on this slide, which answers per pixel "
                "over its own five classes. The uncertainty panel qualifies in-situ "
                "calls made by the two trained options and has nothing to describe "
                f"here; the other panels are {sorted(pixel_overlay.PANELS)}."
            )

        if classes is None or name not in {"map", "flat"}:
            return self.asset(upload_id, f"{name}.png")

        thumbnail, _ = calibration_service.thumbnail(upload_id)
        shape = (thumbnail.shape[0], thumbnail.shape[1])

        # **Which map ran decides which renderer redraws it**, and the answer is read
        # off disk rather than passed in: a filtered panel is a GET that may arrive long
        # after the pass, from a client holding only the four panel names. So the
        # per-pixel mask is looked for first and its presence is the fact.
        if self._path(upload_id, "pixelmap.npz").is_file():
            pixel_map = self.pixel_map(upload_id)
            if name == "flat":
                return pixel_overlay.flat_png(pixel_map, shape, classes=classes)
            return pixel_overlay.map_png(thumbnail, pixel_map, classes=classes)

        class_map = self.class_map(upload_id)
        if name == "flat":
            return overlay.flat_png(class_map, shape, classes=classes)
        return overlay.map_png(thumbnail, class_map, classes=classes)

    # --- describing ---------------------------------------------------------

    def _describe_pixels(
        self,
        pixel_map: PixelMap,
        *,
        upload_id: str,
        filename: str,
        params: dict[str, Any],
        pinned: beetle.Loaded,
        tiles_kept: int,
        tile_px: int,
        tile_um: float,
        tissue_mm2: float,
        run: TissueTypeRun,
    ) -> TissueTypeReport:
        """The report for a BEETLE pass: five classes, counted in pixels.

        `pinned` is a `beetle.Loaded` here rather than a `model.Pinned`. The name is kept
        so `_run` can pick a describer without reshaping its call, and the two are not
        interchangeable anywhere else - which is why this method reads only `.name` and
        `.describe()` off it.
        """
        grid = pixel_map.grid
        height, width = pixel_map.mask.shape

        return TissueTypeReport(
            upload_id=upload_id,
            filename=filename,
            generated_at=_now(),
            params=TissueTypeParams(**params),
            run=run,
            classes=self._pixel_shares(pixel_map),
            grid=TissueTypeGrid(
                cols=grid.cols,
                rows=grid.rows,
                every=grid.every,
                classified=pixel_map.windows_done,
                tiles_kept=tiles_kept,
                tile_px=tile_px,
                tile_um=tile_um,
                min_tissue_share=grid.min_tissue_share,
                gated_out=grid.gated_out,
                blocks_read=pixel_map.blocks_read,
                # A window here is many forward passes rather than a fraction of one, so
                # "batches" would understate the work by two orders of magnitude.
                # `params.patches` is the honest figure and this is the window count.
                batches=pixel_map.windows_done,
                seconds=pixel_map.seconds,
                mask_height=height,
                mask_width=width,
                tissue_pixels=pixel_map.tissue_pixels,
            ),
            tumour_content=round(pixel_map.tumour_content, 4),
            scored_mm2=pixel_map.scored_mm2,
            tissue_mm2=tissue_mm2,
            mean_confidence=round(pixel_map.mean_confidence, 4),
            model=self._beetle_info(usable=True, problem=None).model_copy(
                update={"selected": True, "folds": list(pinned.folds)}
            ),
            caveats=self._pixel_caveats(pixel_map, loaded=pinned),
            notes=self._pixel_notes(pixel_map, loaded=pinned, tile_um=tile_um),
            citation=beetle.CITATION,
        )

    @staticmethod
    def _pixel_shares(pixel_map: PixelMap) -> list[TissueTypeClass]:
        """One entry per BEETLE class, counted in mask pixels.

        `windows` is the number of windows this class *dominated* and `pixels` is what
        `share` is actually computed from. Both are reported because they answer
        different questions and the difference is the branch's point: a window holding
        one small focus of invasion is not an invasive window, but its pixels are
        invasive pixels, and only the second number gets that right.

        **The confidence is measured per pixel, and deriving it from windows gets it
        wrong.** Two bounded runs made that concrete: at 448 um necrosis held 2.9% of the
        tissue and dominated no window, so "mean top probability where this class won"
        reported 0.000 beside 5,801 pixels of it; at 672 um every window was dominated by
        `other`, so three of the five classes had to fall back to their mean asserted
        probability - which lands near the pixel share and reads on screen as a duplicate
        of it. A class can hold a tenth of the tissue without ever being a window's
        plurality. So `PixelMap.class_confidence` accumulates the winning probability
        over exactly the mask pixels each class won, which is defined for every class
        that appears at all and is the question a reader is actually asking.
        """
        shares = pixel_map.shares
        entries: list[TissueTypeClass] = []

        for code, key in enumerate(beetle.PIXEL_CLASSES):
            won = pixel_map.labels == code
            confidence = pixel_map.class_confidence[code]
            entries.append(
                TissueTypeClass(
                    id=code,
                    key=key,
                    label=beetle.CLASS_LABELS[code],
                    blurb=beetle.CLASS_PLAIN[code],
                    meaning=beetle.CLASS_MEANING[code],
                    colour=_hex(beetle.CLASS_COLOURS[code]),
                    scored=code == beetle.SCORED_CODE,
                    windows=int(won.sum()),
                    pixels=pixel_map.counts[code],
                    share=round(shares[code], 4),
                    area_mm2=pixel_map.areas_mm2[code],
                    mean_confidence=round(confidence, 4),
                )
            )
        return entries

    @staticmethod
    def _pixel_caveats(
        pixel_map: PixelMap, *, loaded: beetle.Loaded
    ) -> list[TissueTypeCaveat]:
        """What this option cannot do, from what it actually is.

        Structured rather than prose because the important ones are *facts about the
        provenance of the numbers* - a licence that forbids shipping them, and a model
        never validated on this project's data - and a sentence can be skimmed past in a
        way a labelled severity cannot.
        """
        caveats = [
            TissueTypeCaveat(
                key="licence",
                severity="blocking",
                headline="Research only - these numbers must not be sold",
                detail=(
                    "This model is licensed for non-commercial use, so every figure on "
                    "this screen carries the same restriction. Use one of the "
                    "project-trained models for a commercial result."
                ),
            ),
            TissueTypeCaveat(
                key="not_validated_here",
                severity="warning",
                headline="Accuracy has not been tested on this project's data",
                detail=(
                    "This is another group's model, run as published. It has never been "
                    "tested against this project's own labels, so there is no accuracy "
                    "figure here to compare with the project-trained models."
                ),
            ),
        ]

        if len(loaded.folds) == 1:
            caveats.append(
                TissueTypeCaveat(
                    key="single_fold",
                    severity="note",
                    headline="Only one of five model versions ran",
                    detail=(
                        "This model ships as five versions. Running all five would show "
                        "where they disagree, but would take five times as long. Only "
                        "one ran, so a confident result here is one model being "
                        "confident, not five agreeing."
                    ),
                )
            )

        glass = pixel_map.counts[beetle.GLASS_CODE]
        claimed = glass + pixel_map.tissue_pixels
        if claimed and glass / claimed > 0.25:
            caveats.append(
                TissueTypeCaveat(
                    key="glass_share",
                    severity="note",
                    headline=(
                        f"{glass / claimed:.0%} of what was looked at came back as "
                        "'not tissue'"
                    ),
                    detail=(
                        "That is the model's own background class, not a gap in the map. "
                        "It is left out of every percentage on this screen. A high "
                        "figure means the earlier steps passed through more empty space "
                        "than this model agrees was tissue."
                    ),
                )
            )
        return caveats

    @staticmethod
    def _pixel_notes(
        pixel_map: PixelMap, *, loaded: beetle.Loaded, tile_um: float
    ) -> list[str]:
        """Caveats that belong on screen next to the numbers rather than in a docstring."""
        grid = pixel_map.grid
        patches = beetle.patches_per_window(grid.field_um, loaded.patch) * len(loaded.nets)

        notes = [
            f"Every pixel of the kept tissue was given a class by BEETLE, not one class "
            f"per square. The shares below are shares of {pixel_map.tissue_pixels:,} "
            f"mask pixels at {pixel_map.mask_mpp:g} um each, so an area here is a real "
            f"area rather than a count of grid cells.",
            f"The model was shown {grid.windows:,} windows of {grid.field_um:g} um, one "
            f"at a time, and read each one as {patches} overlapping "
            f"{loaded.patch} px patches at its own {loaded.mpp:g} um/px - "
            f"{pixel_map.patches:,} forward passes in total.",
            "Its answer is kept at a coarser resolution than it was computed at: a whole "
            "section at 0.5 um/px is a gigapixel, so each window's mask was reduced by "
            "area-averaging the class probabilities. Duct shape survives that; a "
            "single-cell detail does not.",
        ]

        if grid.field_um <= 224.0:
            notes.append(
                f"At {grid.field_um:g} um the model cannot see the wall of a large duct - "
                "ducts run 300 to 1500 um across - so solid and comedo in-situ disease "
                "is undecidable at this field of view rather than merely difficult, and "
                "tends to be read as invasive. The wider options exist for that reason."
            )

        return notes

    def _describe(
        self,
        class_map: ClassMap,
        *,
        upload_id: str,
        filename: str,
        params: dict[str, Any],
        pinned: model.Pinned,
        tiles_kept: int,
        tile_px: int,
        tile_um: float,
        tissue_mm2: float,
        run: TissueTypeRun,
    ) -> TissueTypeReport:
        grid = class_map.grid
        candidate = model.find(pinned.name)

        return TissueTypeReport(
            upload_id=upload_id,
            filename=filename,
            generated_at=_now(),
            params=TissueTypeParams(**params),
            run=run,
            classes=self._class_shares(class_map),
            grid=TissueTypeGrid(
                cols=grid.cols,
                rows=grid.rows,
                every=grid.every,
                classified=class_map.classified,
                tiles_kept=tiles_kept,
                tile_px=tile_px,
                tile_um=tile_um,
                min_tissue_share=grid.min_tissue_share,
                gated_out=grid.gated_out,
                blocks_read=class_map.blocks_read,
                batches=class_map.batches,
                seconds=class_map.seconds,
            ),
            # Read off the raw counts, never off `display_labels` - the flag cannot
            # move the score, and this is the line where that would silently stop being
            # true if someone ever "tidied" the two together.
            tumour_content=round(class_map.tumour_content, 4),
            scored_mm2=class_map.scored_mm2,
            tissue_mm2=tissue_mm2,
            mean_confidence=round(class_map.mean_confidence, 4),
            uncertainty=self._uncertainty_block(class_map),
            refused=self._refusal_block(class_map, pinned),
            model=self._model_info(candidate, selected=True)
            if candidate is not None
            else TissueTypeModelInfo(
                name=pinned.name, found=True, selected=True, licence_track=pinned.licence_track
            ),
            caveats=self._caveats(pinned, class_map),
            notes=self._notes(class_map, pinned=pinned, tiles_kept=tiles_kept, tile_um=tile_um),
            citation=CITATION,
        )

    @staticmethod
    def _class_shares(class_map: ClassMap) -> list[TissueTypeClass]:
        """The classes on the map, with their counts, areas and per-class confidence.

        **Tabulated over `display_labels`, so the bars and the picture agree.** A window
        the uncertainty layer flagged is drawn purple and counted on the fourth row, not
        on the in-situ row - a table saying "in-situ 12%" beside a map where most of that
        12% is purple is two answers to one question.

        The denominator does not move. `classified` counts every window the model
        answered for, flagged or not, so the four shares still sum to one and
        `tumourContent` - which is read off the *raw* counts and is what the score is
        gated on - is the same number it would have been without this layer. That is the
        invariant the whole design rests on, and `test_flagging_cannot_move_the_score`
        pins it.

        Three rows on a map with no layer: an older pass, whose panels on disk show three
        colours, describes itself as the three-class pass it was.
        """
        total = max(1, class_map.classified)
        cell = class_map.grid.cell_mm2
        labels = class_map.display_labels
        # Refused windows are drawn purple but were never classified, so they are not
        # counted on any row - they are outside `classified`, and counting them here
        # would make the shares sum past one. The report's `refused` block has them.
        if class_map.refused is not None:
            labels = np.where(class_map.refused == familiarity.ANSWERED, labels, inference.OUTSIDE)
        top = class_map.probabilities.max(axis=-1)

        drawn = DISPLAY_NAMES if class_map.uncertainty is not None else CLASS_NAMES

        rows: list[TissueTypeClass] = []
        for label, key in enumerate(drawn):
            here = labels == label
            windows = int(here.sum())
            rows.append(
                TissueTypeClass(
                    id=label,
                    key=key,
                    label=CLASS_LABELS[label],
                    blurb=CLASS_PLAIN[label],
                    meaning=CLASS_MEANING[label],
                    colour=_hex(CLASS_COLOURS[label]),
                    # Exactly one row is scored, and it is never the fourth: the layer
                    # can only qualify in-situ windows.
                    scored=label == SCORED,
                    windows=windows,
                    share=round(windows / total, 4),
                    area_mm2=round(windows * cell, 4),
                    mean_confidence=round(float(top[here].mean()), 4) if bool(here.any()) else 0.0,
                )
            )
        return rows

    @staticmethod
    def _refusal_block(class_map: ClassMap, pinned: model.Pinned) -> TissueTypeRefusal | None:
        """The gate's verdict as the report carries it, or None on an ungated map."""
        if class_map.refused is None:
            return None
        flat = class_map.refused_count(familiarity.FLAT)
        unfamiliar = class_map.refused_count(familiarity.UNFAMILIAR)
        ran = class_map.classified + flat + unfamiliar
        reference = pinned.familiarity
        return TissueTypeRefusal(
            flat_windows=flat,
            unfamiliar_windows=unfamiliar,
            refused_mm2=round((flat + unfamiliar) * class_map.grid.cell_mm2, 4),
            refused_share=round((flat + unfamiliar) / max(1, ran), 4),
            max_flat_share=settings.tissue_type_max_flat_share,
            distance_threshold=round(reference.threshold, 2) if reference else None,
            distance_calibration=(
                f"the farthest of {reference.held_out:,} tiles from institutions held out "
                f"of training (quantile {reference.quantile:g})"
                if reference
                else None
            ),
        )

    @staticmethod
    def _uncertainty_block(class_map: ClassMap) -> TissueTypeUncertainty | None:
        """The layer as the report carries it, or None where no layer was derived."""
        layer = class_map.uncertainty
        if layer is None:
            return None

        windows = layer.unknown_windows
        return TissueTypeUncertainty(
            params=TissueTypeUncertaintyParams(**layer.params.as_dict()),
            sigma_cells=layer.sigma_cells,
            windows=windows,
            # The same denominator every other share on this report uses.
            share=round(windows / max(1, class_map.classified), 4),
            area_mm2=round(windows * class_map.grid.cell_mm2, 4),
            components=layer.components,
            flagged_components=layer.flagged_components,
            largest_component_mm2=layer.largest_component_mm2,
            mean_score=round(layer.mean_score, 4),
        )

    @staticmethod
    def _caveats(pinned: model.Pinned, class_map: ClassMap) -> list[TissueTypeCaveat]:
        """What this model cannot do, read out of its own manifest.

        Structured rather than prose because the load-bearing one is a *number* - how
        many human-labelled in-situ tiles stood behind the in-situ score - and a
        sentence can be skimmed past in a way a labelled figure cannot.
        """
        caveats: list[TissueTypeCaveat] = []
        manifest = pinned.manifest
        held_out = (manifest.get("metrics") or {}).get("held_out") or {}
        by_source = held_out.get("by_source") or {}
        bcss = by_source.get("bcss") or {}
        borrowed = by_source.get("bracs_dcis") or {}

        if pinned.licence_track == "research-only":
            binding = [
                f"{source} ({terms})"
                for source, terms in sorted(pinned.licences.items())
                if any(
                    marker in terms.lower()
                    for marker in ("non-commercial", "noncommercial", "research only", "nc-sa")
                )
            ]
            caveats.append(
                TissueTypeCaveat(
                    key="licence",
                    severity="blocking",
                    headline="This result cannot be sold",
                    detail=(
                        "The model was trained partly on data licensed for research only, "
                        "so every number on this screen carries the same restriction. "
                        f"What binds: {'; '.join(binding)}. The commercially clean "
                        "alternative cannot tell tumour inside a duct from invasive "
                        "tumour."
                    ),
                )
            )

        # **Absent evidence is said, not skipped (P-17).** This caveat read the per-source
        # breakdown and stayed silent when it was missing - and every served manifest
        # lacks it, so it never fired. The breakdown cannot be invented here; that it is
        # missing is itself the fact a reader needs.
        if not by_source:
            caveats.append(
                TissueTypeCaveat(
                    key="in_situ_ground_truth",
                    severity="warning",
                    headline="How well 'tumour inside a duct' was tested is not recorded",
                    detail=(
                        "This model's record does not say how many of its test tiles for "
                        "'tumour inside a duct' were drawn by a pathologist rather than "
                        "labelled by another model. The line between that class and "
                        "invasive tumour is the one the score depends on, and there is no "
                        "figure here for how far to trust it."
                    ),
                )
            )

        human_tiles = (bcss.get("class_tiles") or {}).get("non_invasive_epithelium")
        if isinstance(human_tiles, int) and human_tiles < 100:
            caveats.append(
                TissueTypeCaveat(
                    key="in_situ_ground_truth",
                    severity="warning",
                    headline=(
                        f"'Tumour inside a duct' was tested on only {human_tiles} "
                        "pathologist-drawn tiles"
                    ),
                    detail=(
                        "Most of the training data for this class came from another "
                        "model's opinion rather than from a pathologist. So the line "
                        "between 'inside a duct' and 'invasive' shows agreement with "
                        "that model, not with a person. The score depends on that line, "
                        "so treat it with care."
                    ),
                )
            )

        if borrowed:
            caveats.append(
                TissueTypeCaveat(
                    key="pseudo_labels",
                    severity="note",
                    headline="Some training labels were produced by another model",
                    detail=(
                        "The 'tumour inside a duct' class was labelled by another model "
                        f"({borrowed.get('labels_drawn_by', 'a model')}) rather than by a "
                        "person, so it can be no more accurate than that model is. The "
                        "two data sources are always reported separately."
                    ),
                )
            )

        # The pipeline runs this step on the case's H&E, so the old "running on a marker
        # slide" caveat was wrong on every slide it was shown on (P-17). The real gap is
        # the laboratory: the training slides are other labs' (P-12).
        caveats.append(
            TissueTypeCaveat(
                key="domain",
                severity="warning",
                headline="Trained on other laboratories' slides, never tested on this one's",
                detail=(
                    "The model learned from public H&E collections (BCSS, BRACS, BACH), "
                    "scanned and stained elsewhere. No slide from this laboratory has been "
                    "labelled for it, so how well it carries over has not been measured."
                ),
            )
        )

        # **The two signatures of the failure that was actually measured (P-17).** The
        # confidence caveat below fires under 0.6, but CAN_00251's false in-situ field -
        # 45.9% of the section - came back at 0.809 confidence, so confidence alone never
        # flagged it. What it did show was a large in-situ share. The bar is a flag for a
        # person, not a verdict: a case can genuinely be mostly DCIS.
        in_situ = class_map.shares[1]
        if in_situ >= IN_SITU_FLAG_SHARE:
            caveats.append(
                TissueTypeCaveat(
                    key="in_situ_share",
                    severity="warning",
                    headline=f"{in_situ:.0%} of the tissue was called 'tumour inside a duct'",
                    detail=(
                        "That is the pattern of the one large failure measured on this "
                        "model: a slide where 46% was called 'inside a duct', confidently "
                        "and wrongly. It can be real, but it is worth a look at the map - "
                        "every patch in this class is left out of the score."
                    ),
                )
            )

        unknown = class_map.uncertainty.unknown_windows if class_map.uncertainty else 0
        refused = class_map.refused_count(familiarity.FLAT) + class_map.refused_count(
            familiarity.UNFAMILIAR
        )
        ran = class_map.classified + refused
        purple = (unknown + refused) / ran if ran else 0.0
        if purple >= UNDETERMINED_FLAG_SHARE:
            caveats.append(
                TissueTypeCaveat(
                    key="undetermined_share",
                    severity="warning",
                    headline=f"{purple:.0%} of the tissue could not be determined",
                    detail=(
                        "These patches are drawn purple: the model's answer was not used, "
                        "either because it did not hold up against its surroundings or "
                        "because the patch looked unlike anything it was trained on. They "
                        "are left out of the score, so a large share of them is tissue the "
                        "score says nothing about."
                    ),
                )
            )

        if class_map.mean_confidence < 0.6:
            caveats.append(
                TissueTypeCaveat(
                    key="confidence",
                    severity="warning",
                    headline=(
                        f"The model was only {class_map.mean_confidence:.0%} confident on "
                        "average"
                    ),
                    detail=(
                        "With three answers to choose from, a pure guess is 33%, so this "
                        "is not far above guessing. Check the confidence map before "
                        "trusting the boundaries - on a slide this uncertain they can "
                        "move a long way."
                    ),
                )
            )

        return caveats

    @staticmethod
    def _notes(
        class_map: ClassMap, *, pinned: model.Pinned, tiles_kept: int, tile_um: float
    ) -> list[str]:
        """The caveats that belong beside the numbers, built from the numbers.

        Plain language by default. The vocabulary a reviewer wants is in `caveats` and
        in the manifest; what belongs on screen is what the picture means.
        """
        grid = class_map.grid
        shares = class_map.shares
        notes: list[str] = []

        notes.append(
            f"Every patch of tissue was shown to a trained model, which answered one "
            f"question about each: what kind of tissue is this? Of "
            f"{class_map.classified:,} patches, {class_map.counts[2]:,} came back as tumour "
            f"that has grown out into the surrounding tissue - {shares[2]:.0%} - covering "
            f"{class_map.scored_mm2:.1f} square millimetres. That is the only tissue the "
            "final score will be measured on."
        )

        notes.append(
            f"The other two answers are both exclusions, for different reasons. "
            f"{shares[0]:.0%} of the tissue is supporting tissue, fat, inflammation or dead "
            f"tissue - none of it is tumour at all. {shares[1]:.0%} is tumour that is still "
            "sitting inside a duct. That second one is not an error and not a small point: "
            "treatment decisions are made on the tumour that has escaped the duct, so "
            "counting the contained tumour would answer a different question about a "
            "different patient."
        )

        flat = class_map.refused_count(familiarity.FLAT)
        unfamiliar = class_map.refused_count(familiarity.UNFAMILIAR)
        if flat:
            notes.append(
                f"{flat:,} patches were not shown to the model's answer at all, because "
                "almost every pixel in them was one identical colour. That is what a "
                "scanner paints where it never took a picture, and no real tissue looks "
                "like that. They are drawn purple and left out of every number above."
            )
        if unfamiliar:
            notes.append(
                f"{unfamiliar:,} patches looked unlike anything the model was trained on - "
                "further out than any slide from the laboratories it was tested on. Its "
                "answer there would be a guess, so it was not used: they are drawn purple "
                "and left out of every number above. Worth a look - an unusual stain, a "
                "fold, or tissue the model has simply never met."
            )

        notes.append(
            "This is also where fat leaves the pipeline, and it is worth saying why it "
            "waited until now. Fat is pale, so the earlier tissue-versus-glass step could "
            "have dropped it on brightness alone - but pale tumour and pale ducts would "
            "have gone with it. Fat is a *kind* of tissue rather than a brightness, so it "
            "is removed by something that knows what fat looks like."
        )

        notes.append(
            f"Each patch the model saw is {grid.field_um:g} microns across - about eleven "
            f"cells - and neighbouring patches overlap by {grid.overlap:.0%}. That size is "
            "not a round number: telling contained tumour from escaped tumour is a question "
            "about *shape*, whether the abnormal cells are still ringed by a duct wall, and "
            "this is the smallest view that holds a whole duct. Zoom in further and you see "
            "the cells but lose the duct; zoom out and the reverse."
        )

        notes.append(
            f"The model was shown the blue stain only - never the brown. Both are on this "
            f"slide, and the brown is the marker being measured, but it is also what makes "
            f"the five markers look nothing like one another: on two of them it floods "
            f"three quarters of the tissue and hides the shapes this step reads. Dropping "
            f"it makes all six slides of a case the same kind of picture, so one model "
            f"serves the whole panel. {grid.windows:,} patches took "
            f"{class_map.seconds / 60:.0f} minutes and {class_map.blocks_read:,} reads off "
            "the slide."
        )

        notes.append(
            f"The patches here are not the same patches as the previous step's. That step "
            f"cut the tissue into {tile_um:g}-micron tiles and kept {tiles_kept:,} of them; "
            f"this step lays its own finer grid inside that region, because "
            f"{grid.field_um:g} microns is the view the model was trained at and showing it "
            "anything else would be showing it something it has never seen. The earlier "
            "step still decides *where* to look - anything it rejected is not looked at."
        )

        if grid.gated_out:
            notes.append(
                f"{grid.gated_out:,} patches were skipped for being mostly empty glass "
                f"- less than {grid.min_tissue_share:.0%} tissue - even though they sat "
                "inside the region the previous step kept. That gate belongs to this "
                "step rather than the last one, and it was added because of what "
                "happens without it: the previous step's squares are more than twice "
                "this size and its own cut-off is deliberately generous, so a patch can "
                "sit inside a kept square and still be nearly all glass. Shown a patch "
                "like that, the model does not answer 'nothing here' - it finds "
                "structure in the little tissue there is and calls it a duct, which drew "
                "a false outline right around the edge of the section."
            )

        layer = class_map.uncertainty
        if layer is not None:
            flagged = layer.unknown_windows
            area = flagged * grid.cell_mm2
            if flagged:
                notes.append(
                    f"{flagged:,} of those patches came back purple - "
                    f"{flagged / max(1, class_map.classified):.1%} of the tissue, "
                    f"{area:.1f} square millimetres. The model called them tumour "
                    "inside a duct and a second check would not stand behind that. It "
                    "asks two things the model cannot ask itself: whether the "
                    "surrounding tissue is being read as escaped tumour, and whether "
                    "the region is even shaped like ducts - a duct system is a bounded, "
                    "branching thing, so a solid sheet of it several millimetres across "
                    "is a shape no duct makes. Purple tissue was already excluded from "
                    "the score for being contained tumour, so nothing here changes the "
                    "number above; it marks where the picture is worth a human look."
                )
            else:
                notes.append(
                    "None of the contained-tumour patches were flagged by the second "
                    "check, which asks whether the surrounding tissue and the shape of "
                    "each region support calling it tumour inside a duct."
                )

            notes.append(
                "That check has one blind spot worth stating plainly: it can only "
                "question tissue the model *did* call contained tumour. Contained "
                "tumour the model missed altogether is coloured as something else, is "
                "never examined, and no amount of purple or its absence says anything "
                "about it. A slide with no purple on it is not a slide that has been "
                "checked."
            )

        notes.append(
            f"The model was {class_map.mean_confidence:.0%} certain on average, and a "
            "three-way choice starts at 33%. Certainty is worth looking at separately, "
            "because the class map draws a barely-decided patch in exactly the same colour "
            "as a certain one. The next step smooths these answers into one region and "
            "leans on the certainty rather than the colours, which is why it is shown here "
            "rather than kept behind the scenes."
        )

        if pinned.licence_track == "research-only":
            notes.append(
                "One thing this screen is not: shippable. The model behind it was trained "
                "partly on images licensed for research only, so these numbers are for "
                "evaluation and not for a report anyone is charged for. Every public "
                "dataset that marks contained tumour as its own category carries that "
                "restriction, which is a fact about the field rather than a shortcut taken "
                "here - and the freely usable alternative cannot tell the two kinds of "
                "tumour apart at all."
            )

        return notes


tissue_type_service = TissueTypeService()

__all__ = ["TissueTypeError", "tissue_type_service"]
