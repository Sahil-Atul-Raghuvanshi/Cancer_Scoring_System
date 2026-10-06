"""Step 11 - segment the nuclei inside the regions step 10 carried across.

    report.json              every number the screen shows
    region{r}/nuclei.json    that region's nuclei, geometry included
    region{r}/f{i}_labels.npz  the instance map for one field, compressed
    region{r}/f{i}_raw.png   one sampled field, as it is
    region{r}/f{i}_input.png the same field with the DAB removed - what the model saw
    region{r}/f{i}_over.png  the outlines drawn back onto the raw field
    compare/{r}_{i}.png      InstanSeg against a naive watershed, same field

Job-shaped like step 10's, for a smaller version of the same reason: a pass is
one to a few minutes of CPU, which is longer than a request should wait even
though it is not longer than a person will.

**Keyed on the pair, not the slide.** Everything here happens in the IHC slide's
coordinates, inside regions that only exist for one (H&E, marker) pair, so a
second marker of the same case is a different result rather than an overwrite -
which is also what makes the cross-marker density check at the bottom possible.

**Detector and denominator (P-03, 6 October 2026).** The nuclei come from
`settings.nuclei_engine` - Cellpose by default, chosen on a benchmark of every
alternative - and the check against the H&E is made per mm2 of *tissue* on both
sides, so a field that lands on glass or scanner fill is not read as missed nuclei.
The H&E reference itself stays on InstanSeg on the H&E photograph, the input P-21
validated it on.

**It refuses to run on an unconfirmed alignment.** Step 10 finishes with
`confirmed = false` and says in its own README that nothing downstream should
measure inside its regions until a person has looked. This is the first step
downstream, so this is where that stops being advice.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.common import stains
from app.common.imaging import optical_density
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.nuclei import cellpose_model
from app.nuclei.model import ModelUnavailable
from app.nuclei.model import load as load_model
from app.pipeline.step13_nuclei_segmentation import overlay
from app.pipeline.step13_nuclei_segmentation.sampling import allocate, fields_for_region
from app.pipeline.step13_nuclei_segmentation.segment import SegmentedField, segment_field
from app.pipeline.step13_nuclei_segmentation.stain_input import (
    StainBasis,
    estimate_basis,
    ruifrok_basis,
    ruifrok_he_basis,
)
from app.schemas.nuclei import (
    ComparisonOut,
    FieldOut,
    NucleiReport,
    NucleiRun,
    NucleusOut,
    RegionNuclei,
    StainReport,
)
from app.services.calibration_service import calibration_service
from app.services.ihc_alignment_service import ihc_alignment_service
from app.services.upload_service import resolve_ready_path

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class NucleiError(ValueError):
    """A client-correctable problem: a missing step, or a model that is not installed."""


@dataclass
class Job:
    """Live state of one pass, mutated by the worker, read by the API."""

    he_upload_id: str
    ihc_upload_id: str
    state: str = "queued"
    message: str | None = None
    progress: float = 0.0
    started: float = field(default_factory=time.monotonic)
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    duration: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class HeReference:
    """The H&E reference: per mm2 of field and per mm2 of tissue, and what it rests on."""

    density: float | None = None
    median_area: float | None = None
    fields: int = 0
    tissue_density: float | None = None
    tissue_share: float | None = None


class NucleiService:
    """Finds the nuclei in one (H&E, IHC) pair's three carried regions."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # --- storage ------------------------------------------------------------

    def key(self, he_upload_id: str, ihc_upload_id: str) -> str:
        return f"{he_upload_id}__{ihc_upload_id}"

    def _dir(self, he_upload_id: str, ihc_upload_id: str) -> Path:
        directory = settings.nuclei_dir / self.key(he_upload_id, ihc_upload_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _path(self, he_upload_id: str, ihc_upload_id: str, *parts: str) -> Path:
        path = self._dir(he_upload_id, ihc_upload_id).joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def artifact(self, he_upload_id: str, ihc_upload_id: str, *parts: str) -> Path:
        """One stored artefact's path, for the routes that serve files directly.

        Public because the router needs it and reaching through `_path` would be
        a private call across a layer boundary. It creates no directories and
        makes no promise the file exists - the caller answers 404 for that.
        """
        return settings.nuclei_dir.joinpath(self.key(he_upload_id, ihc_upload_id), *parts)

    def _read_report(self, he_upload_id: str, ihc_upload_id: str) -> NucleiReport | None:
        path = self._path(he_upload_id, ihc_upload_id, "report.json")
        if not path.is_file():
            return None
        try:
            return NucleiReport.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None

    def _write_report(self, report: NucleiReport) -> None:
        path = self._path(report.he_upload_id, report.ihc_upload_id, "report.json")
        path.write_text(json.dumps(report.model_dump(by_alias=True), indent=2), encoding="utf-8")

    # --- the job ------------------------------------------------------------

    def start(
        self, he_upload_id: str, ihc_upload_id: str, *, restart: bool = False
    ) -> NucleiRun:
        """Queue a pass, or hand back the finished one."""
        resolve_ready_path(upload_id=he_upload_id)
        resolve_ready_path(upload_id=ihc_upload_id)
        alignment = self._require_alignment(he_upload_id, ihc_upload_id)

        if not restart:
            existing = self._read_report(he_upload_id, ihc_upload_id)
            if existing is not None and self._is_current(existing, alignment):
                return NucleiRun(
                    he_upload_id=he_upload_id,
                    ihc_upload_id=ihc_upload_id,
                    state=existing.state,
                    progress=1.0,
                    finished_at=existing.generated_at,
                )
            if existing is not None:
                logger.info(
                    "nuclei %s are not current (alignment %s, now %s; engine %s, now %s); "
                    "re-segmenting rather than reporting them as current",
                    self.key(he_upload_id, ihc_upload_id),
                    existing.alignment_generated_at,
                    alignment.generated_at,
                    existing.engine or "instanseg",
                    settings.nuclei_engine,
                )

        key = self.key(he_upload_id, ihc_upload_id)
        with self._lock:
            job = self._jobs.get(key)
            if job is not None and job.state in {"queued", "running"}:
                return self._as_run(job)
            self._jobs[key] = Job(he_upload_id=he_upload_id, ihc_upload_id=ihc_upload_id)
        return self._as_run(self._jobs[key])

    @staticmethod
    def _is_current(report: NucleiReport, alignment) -> bool:
        """Whether `report` was segmented inside the alignment now on disk, by the
        detector now configured.

        A report with no alignment stamp predates the field and is treated as not
        matching: one wasted re-segmentation is a far smaller cost than measuring the
        wrong regions, and unlike the wrong regions it is visible. A report with no
        engine predates the choice and was InstanSeg; when the configured detector
        differs, its nuclei are a different measurement and are not reused (P-15).
        """
        return (
            report.alignment_generated_at is not None
            and report.alignment_generated_at == alignment.generated_at
            and (report.engine or "instanseg") == settings.nuclei_engine
        )

    def state(self, he_upload_id: str, ihc_upload_id: str) -> NucleiRun:
        job = self._jobs.get(self.key(he_upload_id, ihc_upload_id))
        if job is not None:
            return self._as_run(job)

        report = self._read_report(he_upload_id, ihc_upload_id)
        if report is not None:
            return NucleiRun(
                he_upload_id=he_upload_id,
                ihc_upload_id=ihc_upload_id,
                state=report.state,
                progress=1.0,
                finished_at=report.generated_at,
            )
        raise NucleiError("no nuclei pass has been started for this pair")

    def report(self, he_upload_id: str, ihc_upload_id: str) -> NucleiReport:
        report = self._read_report(he_upload_id, ihc_upload_id)
        if report is None:
            raise NucleiError("this pair has no nuclei result yet")
        return report

    def _as_run(self, job: Job) -> NucleiRun:
        return NucleiRun(
            he_upload_id=job.he_upload_id,
            ihc_upload_id=job.ihc_upload_id,
            state=job.state,
            message=job.message,
            progress=round(job.progress, 3),
            started_at=job.started_at,
            finished_at=job.finished_at,
            duration=job.duration,
            error=job.error,
        )

    def execute(self, he_upload_id: str, ihc_upload_id: str) -> None:
        """Run the queued pass. Called on a worker thread, never awaited."""
        key = self.key(he_upload_id, ihc_upload_id)
        job = self._jobs.get(key)
        if job is None or job.state != "queued":
            return

        job.state = "running"
        job.message = "loading the segmenter"
        started = time.monotonic()

        try:
            report = self._build(he_upload_id, ihc_upload_id, job)
            self._write_report(report)
            job.state = "ready"
            job.progress = 1.0
            job.message = f"{report.counted} nuclei counted"
        except (NucleiError, ModelUnavailable) as failure:
            job.state = "failed"
            job.error = str(failure)
            logger.warning("nuclei %s failed: %s", key, failure)
        except Exception as failure:  # noqa: BLE001 - a worker thread must not die silently
            job.state = "failed"
            job.error = f"{type(failure).__name__}: {failure}"
            logger.exception("nuclei %s crashed", key)
        finally:
            job.finished_at = _now()
            job.duration = round(time.monotonic() - started, 1)

    # --- preconditions ------------------------------------------------------

    def _require_alignment(self, he_upload_id: str, ihc_upload_id: str):
        """Step 10's report, and the two things it has to say before this runs."""
        try:
            alignment = ihc_alignment_service.report(he_upload_id, ihc_upload_id)
        except Exception as exc:  # the alignment service raises its own error type
            raise NucleiError(
                "step 10 has not carried the regions onto this IHC slide yet, so there "
                "is nothing on it to segment inside."
            ) from exc

        if alignment.state != "ready" or not alignment.regions:
            raise NucleiError(
                f"step 10 finished in state '{alignment.state}' with no usable regions. "
                "Nuclei cannot be counted inside a region that was never carried across."
            )

        if not alignment.confirmed:
            raise NucleiError(
                "the alignment has not been confirmed. Step 10 refuses to approve its own "
                "result - a person has to look at the two panels and say the regions landed "
                "on the same tissue - and measuring inside unconfirmed regions is exactly "
                "what that check exists to prevent. Confirm it on step 10, then run this."
            )
        return alignment

    # --- the work itself ----------------------------------------------------

    def _basis(self, reader, white) -> tuple[StainBasis, StainReport]:
        """The basis tiles are un-mixed with, and the diagnostic for the other one.

        Macenko is estimated whether or not it is used, because its drift from
        the published direction is the evidence for the default - see
        `settings.nuclei_macenko_per_slide`. Estimating it costs one thumbnail
        read the caller has already paid for.
        """
        thumbnail = np.asarray(reader.thumbnail_pil(max_size=1024).convert("RGB"), dtype=np.float32)
        density = optical_density(thumbnail, np.asarray(white.rgb, dtype=np.float32))
        estimated = estimate_basis(density.reshape(-1, 3))

        dab_drift: float | None = None
        if estimated.source == "macenko":
            dab_drift = float(
                stains.degrees_between(
                    estimated.matrix[:, 1],
                    np.asarray(stains.REFERENCE_BY_NAME["dab"], dtype=np.float64),
                )
            )

        notes: list[str] = []
        if settings.nuclei_macenko_per_slide:
            chosen = estimated
        else:
            chosen = ruifrok_basis()
            if estimated.source == "macenko":
                notes.append(
                    "Detection un-mixes with Ruifrok's published basis. The per-slide "
                    "Macenko estimate is reported beside it because its haematoxylin arm "
                    "is the check: a large drift means the counterstain is too weak under "
                    "the DAB for the estimate to find it, which is when the fixed basis "
                    "wins. Measured on CD44: 695 nuclei fixed against 581 estimated."
                )

        return chosen, StainReport(
            basis=chosen.source,
            gain=settings.nuclei_haematoxylin_gain,
            macenko_explained=estimated.explained,
            macenko_haematoxylin_drift_deg=estimated.haematoxylin_drift_deg,
            macenko_dab_drift_deg=dab_drift,
            notes=notes,
        )

    def _write_field_images(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        rank: int,
        segmented: SegmentedField,
    ) -> None:
        index = segmented.field.index
        folder = f"region{rank}"
        counted = {n.label for n in segmented.nuclei if n.counted}

        self._path(he_upload_id, ihc_upload_id, folder, f"f{index}_raw.png").write_bytes(
            overlay.field_png(segmented.rgb)
        )
        self._path(he_upload_id, ihc_upload_id, folder, f"f{index}_input.png").write_bytes(
            overlay.field_png(segmented.shown)
        )
        self._path(he_upload_id, ihc_upload_id, folder, f"f{index}_over.png").write_bytes(
            overlay.overlay_png(
                segmented.rgb,
                segmented.labels,
                counted=counted,
                border_px=settings.nuclei_border_margin_px,
            )
        )

        # The instance map itself, not just its outlines.
        #
        # Step 13 grows each nucleus into a membrane ring or a cytoplasm band,
        # which is a raster operation on the labels - a distance transform over
        # the whole field at once, so that neighbouring cells constrain each
        # other. Re-deriving it from the stored polygons would mean rasterising
        # them back and hoping the result matched, and re-running the model
        # would cost the whole pass again. Compressed these are a few tens of
        # kilobytes each: an int32 field of mostly zeros.
        np.savez_compressed(
            self._path(he_upload_id, ihc_upload_id, folder, f"f{index}_labels.npz"),
            labels=segmented.labels.astype(np.int32),
            mpp=np.float32(segmented.mpp),
            x=np.int64(segmented.field.x),
            y=np.int64(segmented.field.y),
            span=np.int64(segmented.field.span),
        )

    def _build(self, he_upload_id: str, ihc_upload_id: str, job: Job) -> NucleiReport:
        alignment = self._require_alignment(he_upload_id, ihc_upload_id)
        engine = settings.nuclei_engine
        if engine == "cellpose":
            model = cellpose_model.load()
        elif engine == "instanseg":
            model = load_model()
        else:
            raise NucleiError(
                f"settings.nuclei_engine is {engine!r}; it must be 'cellpose' or 'instanseg'."
            )

        ihc_path = resolve_ready_path(upload_id=ihc_upload_id)
        reader = open_slide(ihc_path)
        try:
            width, height = reader.dimensions
            base_mpp = reader.mpp
            if not base_mpp:
                raise NucleiError(
                    "this IHC slide records no microns-per-pixel, so there is no physical "
                    "scale to read a field at - and a nucleus count per mm2 would be a "
                    "number with no unit."
                )

            white = calibration_service.white_point(ihc_upload_id)
            basis, stain_report = self._basis(reader, white)

            job.message = "segmenting"
            regions: list[RegionNuclei] = []
            comparisons: list[ComparisonOut] = []

            # The field budget is split between the carried regions in
            # proportion to their area, not given to each of them equally. An
            # equal split makes the measured cells a sample of the regions
            # rather than of the tumour - see `sampling.allocate` - and on this
            # project's own case that put 42% of the cells in a region holding
            # 84% of the invasive area.
            shares = allocate([region.area_mm2 for region in alignment.regions])
            planned = [
                (
                    region,
                    fields_for_region(
                        region.ihc_rings,
                        mpp=base_mpp,
                        slide_width=width,
                        slide_height=height,
                        count=share,
                    ),
                )
                for region, share in zip(alignment.regions, shares, strict=True)
            ]
            total_fields = sum(len(fields) for _, (fields, _) in planned) or 1
            done = 0

            for region, (fields, available) in planned:
                segmented_fields: list[SegmentedField] = []
                for sample in fields:
                    segmented = segment_field(
                        reader,
                        sample,
                        white=white,
                        basis=basis,
                        base_mpp=base_mpp,
                        model_mpp=model.mpp,
                        engine=engine,
                        region_rings=region.ihc_rings,
                    )
                    segmented_fields.append(segmented)
                    self._write_field_images(he_upload_id, ihc_upload_id, region.rank, segmented)

                    done += 1
                    job.progress = done / total_fields
                    job.message = f"region {region.rank}, field {sample.index + 1} of {len(fields)}"

                regions.append(
                    self._summarise(region, segmented_fields, available=available)
                )
                self._write_nuclei(he_upload_id, ihc_upload_id, region.rank, segmented_fields)

                if segmented_fields:
                    # The comparison is for the screen and feeds no count, so it must
                    # not be able to fail the pass - with Cellpose counting, it also
                    # needs InstanSeg, which may not be installed.
                    try:
                        comparisons.append(
                            self._compare(
                                he_upload_id,
                                ihc_upload_id,
                                region.rank,
                                self._busiest(segmented_fields),
                                reader=reader,
                                white=white,
                                basis=basis,
                                base_mpp=base_mpp,
                                model_mpp=model.mpp,
                            )
                        )
                    except Exception:  # noqa: BLE001 - a picture, not a measurement
                        logger.warning("nuclei comparison for region %s skipped", region.rank,
                                       exc_info=True)

            detected = sum(r.detected for r in regions)
            counted = sum(r.counted for r in regions)
            sampled = sum(r.sampled_mm2 for r in regions)
            density = counted / sampled if sampled > 0 else 0.0
            tissue = sum(r.tissue_mm2 or 0.0 for r in regions)
            tissue_density = counted / tissue if tissue > 0 else None

            job.message = "measuring the H&E reference density"
            reference = self._he_reference(he_upload_id, alignment, model_mpp=model.mpp)

            def gap(he: float | None, ihc: float | None) -> float | None:
                return round((he - ihc) / he, 3) if he and he > 0 and ihc is not None else None

            # Per mm2 of tissue on both sides (P-03): the headline. The old per-area
            # figure is kept beside it, so the two can be compared.
            shortfall = gap(reference.tissue_density, tissue_density)
            area_shortfall = gap(reference.density, density)
            tissue_share = tissue / sampled if sampled > 0 else None

            return NucleiReport(
                he_upload_id=he_upload_id,
                ihc_upload_id=ihc_upload_id,
                marker=alignment.marker,
                state="ready",
                generated_at=_now(),
                alignment_generated_at=alignment.generated_at,
                engine=engine,
                model_name=str(model.manifest.get("name")),
                model_version=str(model.manifest.get("version")),
                model_licence=str(model.manifest.get("licence")),
                model_mpp=model.mpp,
                regions=regions,
                stain=stain_report,
                comparison=comparisons,
                detected=detected,
                counted=counted,
                sampled_mm2=round(sampled, 4),
                density_per_mm2=round(density, 1),
                tissue_mm2=round(tissue, 5),
                tissue_share=round(tissue_share, 3) if tissue_share is not None else None,
                density_per_tissue_mm2=(
                    round(tissue_density, 1) if tissue_density is not None else None
                ),
                density_by_marker=self._density_by_marker(
                    he_upload_id, alignment.marker, density
                ),
                he_density_per_mm2=reference.density,
                he_median_area_um2=reference.median_area,
                he_reference_fields=reference.fields,
                he_reference_basis="ruifrok_he",
                he_density_per_tissue_mm2=reference.tissue_density,
                he_tissue_share=reference.tissue_share,
                density_shortfall=shortfall,
                area_shortfall=area_shortfall,
                seconds=round(time.monotonic() - job.started, 1),
                notes=self._notes(
                    regions, shortfall=shortfall, engine=engine, tissue_share=tissue_share
                ),
            )
        finally:
            reader.close()

    # --- summarising --------------------------------------------------------

    @staticmethod
    def _busiest(fields: list[SegmentedField]) -> SegmentedField:
        """The field with the most nuclei, for the touching-nuclei comparison.

        Chosen rather than taken first, and this is the one place in the step
        where picking the densest field is right: the comparison screen exists to
        show what happens where nuclei *touch*, so a sparse field would make the
        two methods agree and demonstrate nothing. It feeds no count.
        """
        return max(fields, key=lambda f: len(f.nuclei))

    def _summarise(
        self, region, fields: list[SegmentedField], *, available: int
    ) -> RegionNuclei:
        out_fields = [
            FieldOut(
                index=f.field.index,
                x=f.field.x,
                y=f.field.y,
                span=f.field.span,
                size=f.field.size,
                mpp=round(f.mpp, 4),
                tissue=round(f.field.tissue, 3),
                detected=len(f.nuclei),
                counted=f.counted,
                counted_mm2=round(f.counted_mm2, 6),
                density_per_mm2=round(f.density_per_mm2, 1),
                tissue_share=round(f.tissue_share, 4),
                density_per_tissue_mm2=round(f.density_per_tissue_mm2, 1),
            )
            for f in fields
        ]

        sampled_mm2 = sum(f.counted_mm2 for f in fields)
        tissue_mm2 = sum(f.tissue_mm2 for f in fields)
        counted = sum(f.counted for f in fields)
        densities = np.array([f.density_per_mm2 for f in fields], dtype=np.float64)
        areas = np.array(
            [n.area_um2 for f in fields for n in f.nuclei if n.counted], dtype=np.float64
        )
        circular = np.array(
            [n.circularity for f in fields for n in f.nuclei if n.counted], dtype=np.float64
        )

        return RegionNuclei(
            rank=region.rank,
            index=region.index,
            area_mm2=region.area_mm2,
            fields=out_fields,
            fields_available=available,
            sampled_mm2=round(sampled_mm2, 5),
            sampled_share=round(sampled_mm2 / region.area_mm2, 5) if region.area_mm2 else 0.0,
            detected=sum(len(f.nuclei) for f in fields),
            counted=counted,
            density_per_mm2=round(counted / sampled_mm2, 1) if sampled_mm2 > 0 else 0.0,
            density_cv=(
                round(float(densities.std() / densities.mean()), 3)
                if densities.size and densities.mean() > 0
                else 0.0
            ),
            tissue_mm2=round(tissue_mm2, 5),
            density_per_tissue_mm2=round(counted / tissue_mm2, 1) if tissue_mm2 > 0 else None,
            median_area_um2=round(float(np.median(areas)), 2) if areas.size else 0.0,
            median_circularity=round(float(np.median(circular)), 3) if circular.size else 0.0,
        )

    def _write_nuclei(
        self, he_upload_id: str, ihc_upload_id: str, rank: int, fields: list[SegmentedField]
    ) -> None:
        """One region's nuclei, kept out of the report on purpose.

        A region's outlines run to a few megabytes of vertices, and the report is
        fetched every time the screen is opened. Splitting them means the summary
        stays small and the geometry is fetched once, per region, when a viewer
        actually looks at one.
        """
        payload = {
            "rank": rank,
            "fields": [
                {
                    "index": f.field.index,
                    "x": f.field.x,
                    "y": f.field.y,
                    "span": f.field.span,
                    "size": f.field.size,
                    "nuclei": [
                        NucleusOut(
                            id=n.label,
                            x=round(n.x, 1),
                            y=round(n.y, 1),
                            area_um2=round(n.area_um2, 2),
                            perimeter_um=round(n.perimeter_um, 2),
                            circularity=round(n.circularity, 3),
                            eccentricity=round(n.eccentricity, 3),
                            haematoxylin=round(n.haematoxylin, 4),
                            rings=[
                                [[round(x, 1), round(y, 1)] for x, y in ring]
                                for ring in n.rings
                            ],
                            counted=n.counted,
                        ).model_dump(by_alias=True)
                        for n in f.nuclei
                    ],
                }
                for f in fields
            ],
        }
        self._path(he_upload_id, ihc_upload_id, f"region{rank}", "nuclei.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def _compare(
        self,
        he_upload_id: str,
        ihc_upload_id: str,
        rank: int,
        reference: SegmentedField,
        *,
        reader,
        white,
        basis,
        base_mpp: float,
        model_mpp: float,
    ) -> ComparisonOut:
        """The same field, several ways, and the picture that makes the point.

        `reference` is the counting detector's field. The InstanSeg counts are its
        two inputs (brown removed, brown left in), run here when InstanSeg is not the
        detector doing the counting.
        """
        watershed = segment_field(
            reader, reference.field, white=white, basis=basis,
            base_mpp=base_mpp, model_mpp=model_mpp, engine="watershed",
        )
        raw = segment_field(
            reader, reference.field, white=white, basis=basis,
            base_mpp=base_mpp, model_mpp=model_mpp, remove_dab=False, engine="instanseg",
        )
        haematoxylin = (
            reference
            if reference.engine == "instanseg"
            else segment_field(
                reader, reference.field, white=white, basis=basis,
                base_mpp=base_mpp, model_mpp=model_mpp, engine="instanseg",
            )
        )

        self._path(he_upload_id, ihc_upload_id, "compare", f"{rank}_watershed.png").write_bytes(
            overlay.side_by_side_png(
                overlay.overlay_array(reference.rgb, reference.labels),
                overlay.overlay_array(watershed.rgb, watershed.labels),
            )
        )
        self._path(he_upload_id, ihc_upload_id, "compare", f"{rank}_rgb.png").write_bytes(
            overlay.overlay_png(raw.rgb, raw.labels)
        )

        return ComparisonOut(
            field_index=reference.field.index,
            region_rank=rank,
            instanseg_haematoxylin=len(haematoxylin.nuclei),
            instanseg_rgb=len(raw.nuclei),
            watershed_haematoxylin=len(watershed.nuclei),
            engine=reference.engine,
            production=len(reference.nuclei),
        )

    def _he_reference(
        self, he_upload_id: str, alignment, *, model_mpp: float
    ) -> HeReference:
        """The same measurement on the H&E slide, inside the same regions.

        The number the IHC density has to be read against. Measured every run
        rather than waiting for a second marker to appear, because otherwise the
        first marker of a case has nothing to compare with - and the check only
        starts working after a failure has already gone unnoticed once.

        **Sampled the way the IHC is, and un-mixed as an H&E (P-21).** It used to take
        six fields of the largest region only, while the IHC count it is compared with
        covers every region with fields allocated by area - so the two densities were
        means over different tissue. It now uses the same regions, the same
        area-proportional allocation and the same field budget, drawn on the H&E side
        of each region. And it never un-mixes the H&E with the H-DAB basis, which read
        every eosin pixel as part haematoxylin: nuclei are found on the H&E's own
        photograph, the segmenter's native input, and the per-nucleus stain values use
        the H&E basis.

        **InstanSeg, whatever the IHC detector (P-03).** The reference is measured with
        the detector and input P-21 validated it on - InstanSeg on the H&E photograph -
        so changing the IHC detector moves only one side of the comparison. And it is
        counted per mm2 of tissue as well as per mm2 of field, like the IHC.
        """
        empty = HeReference()
        if not alignment.regions:
            return empty

        try:
            reader = open_slide(resolve_ready_path(upload_id=he_upload_id))
        except Exception:  # noqa: BLE001 - a missing H&E must not fail the step
            return empty

        try:
            base_mpp = reader.mpp
            if not base_mpp:
                return empty
            width, height = reader.dimensions
            white = calibration_service.white_point(he_upload_id)
            basis = ruifrok_he_basis()

            # The IHC's own allocation, on the H&E side of the same regions.
            shares = allocate([region.area_mm2 for region in alignment.regions])
            counted = 0
            area_mm2 = 0.0
            tissue_mm2 = 0.0
            used = 0
            areas: list[float] = []
            for region, share in zip(alignment.regions, shares, strict=True):
                fields, _ = fields_for_region(
                    region.he_rings,
                    mpp=base_mpp,
                    slide_width=width,
                    slide_height=height,
                    count=share,
                )
                for sample in fields:
                    # The H&E's own photograph, not a redrawn haematoxylin image:
                    # InstanSeg's brightfield model was trained on H&E RGB, so on the
                    # H&E slide there is nothing to separate. Measured on the same six
                    # fields: CAN_00270 1,781 (H-DAB, old) / 440 (H&E basis, redrawn) /
                    # 1,762 (RGB); CAN_00303 9,372 / 4,042 / 4,054. Only RGB is right
                    # on both - the H-DAB render counted eosin as nuclei on one, and
                    # the H&E-basis render was too faint for the model on the other.
                    segmented = segment_field(
                        reader, sample, white=white, basis=basis,
                        base_mpp=base_mpp, model_mpp=model_mpp, remove_dab=False,
                        engine="instanseg", region_rings=region.he_rings,
                    )
                    counted += segmented.counted
                    area_mm2 += segmented.counted_mm2
                    tissue_mm2 += segmented.tissue_mm2
                    used += 1
                    areas.extend(n.area_um2 for n in segmented.nuclei if n.counted)

            if area_mm2 <= 0:
                return HeReference(fields=used)
            median = float(np.median(areas)) if areas else None
            return HeReference(
                density=round(counted / area_mm2, 1),
                median_area=round(median, 2) if median else None,
                fields=used,
                tissue_density=round(counted / tissue_mm2, 1) if tissue_mm2 > 0 else None,
                tissue_share=round(tissue_mm2 / area_mm2, 3),
            )
        except Exception:  # noqa: BLE001 - the reference is a check, not the answer
            logger.warning("H&E reference density could not be measured", exc_info=True)
            return empty
        finally:
            reader.close()

    def _density_by_marker(
        self, he_upload_id: str, marker: str | None, density: float
    ) -> dict[str, float]:
        """This case's other markers' densities, for the guide's free QC check.

        The six slides of a case are serial sections of one block, so their nuclei
        per mm2 should land in the same neighbourhood. A marker 30% below its
        siblings is a segmentation failure rather than biology - and it costs
        nothing to see, because every other marker's report is already on disk.
        """
        out: dict[str, float] = {}
        if marker:
            out[marker] = round(density, 1)

        root = settings.nuclei_dir
        if not root.is_dir():
            return out

        for directory in root.iterdir():
            if not directory.is_dir() or not directory.name.startswith(f"{he_upload_id}__"):
                continue
            path = directory / "report.json"
            if not path.is_file():
                continue
            try:
                other = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            letter = other.get("marker")
            if letter and letter not in out:
                out[letter] = round(float(other.get("densityPerMm2", 0.0)), 1)
        return out

    @staticmethod
    def _notes(
        regions: list[RegionNuclei],
        *,
        shortfall: float | None = None,
        engine: str = "instanseg",
        tissue_share: float | None = None,
    ) -> list[str]:
        notes = [
            "Counts are an estimate from a sample, not a census: each region was "
            "segmented on a few fields spread across it, and every figure is "
            "printed beside the area it was measured over.",
        ]
        if engine == "cellpose":
            notes.append(
                "Nuclei were found by Cellpose's published nuclei model on the field in "
                "grey, chosen on a benchmark of every alternative (P-03). It sees the "
                "field as it is, brown included; the check against the H&E below is what "
                "would show stained cells being found more readily than unstained ones."
            )
        else:
            notes.append(
                "Nuclei were detected on the haematoxylin channel with the DAB removed. "
                "Detecting on raw RGB would let the stain being measured decide where "
                "cells are, which inflates the percentage by a route no later step can see."
            )
        if tissue_share is not None and tissue_share < settings.nuclei_low_tissue_share:
            notes.append(
                f"Only {tissue_share:.0%} of the sampled area is tissue: the fields fell "
                "mostly on glass or scanner fill. That is a question about where the "
                "regions landed on this slide (alignment, region choice), not about "
                "detection, and the densities rest on very little tissue."
            )
        if shortfall is not None and shortfall >= 0.3:
            notes.append(
                f"This slide yields {shortfall:.0%} fewer nuclei per mm2 of tissue than the "
                "case's own H&E inside the same regions. Serial sections of one block hold "
                "the same cells, so that gap is a detection failure rather than biology. On "
                "heavily stained sections the tumour nuclei can show only as pale holes in "
                "the brown, with no counterstain, and no detector tested finds them (P-22). "
                "Which way the missing cells move the percentage has not been measured."
            )

        worst = max((r.density_cv for r in regions), default=0.0)
        if worst > 0.5:
            notes.append(
                f"Field-to-field density varies by {worst:.0%} within a region, so the "
                "region is heterogeneous and this sample size is thin for it. Raise "
                "nuclei_field_budget before reading much into a single figure."
            )
        return notes


nuclei_service = NucleiService()
