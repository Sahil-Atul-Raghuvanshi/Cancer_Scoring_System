"""Step 2 - quality control, orchestrated.

A run is a background job, not a request. On a CPU-only machine GrandQC's
artefact pass is minutes of work: roughly 1,400 forward passes for a 28 mm
slide at 1.5 um/px, of which the tissue detector typically excuses half. So the
API starts a job, the client polls, and the finished report is cached on disk
under `data/qc/<upload_id>/` because the walkthrough asks for it again every
time someone steps back onto step 2.

Only one run happens at a time. The alternative - letting two slides fight over
the same cores - makes both slower than running them in sequence and makes the
progress bar a lie.

What lands on disk per upload:

    report.json     the API payload, exactly as served
    grid.json       per-cell metrics, for client-side heatmaps
    internals.json  the few numbers the report does not carry but the region
                    endpoint needs: mask resolution, slide mpp, baselines
    mask.png        the artefact map, indexed colour, at display resolution
    overlay.png     the slide with artefacts tinted
    flat.png        the class map with no slide under it
    tissue.png      pass 1's output, so the two models stay visibly separate
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.contract import RunCancelled
from app.pipeline.step02_quality_control import classes as qc_classes
from app.pipeline.step02_quality_control import inference, models, overlay
from app.pipeline.step02_quality_control.features import METRIC_KEYS, METRIC_SPECS, metric_spec
from app.schemas.qc import (
    QCCapability,
    QCClassShare,
    QCDownload,
    QCGate,
    QCGrid,
    QCGridCell,
    QCGridMeta,
    QCMetricSummary,
    QCModelInfo,
    QCParams,
    QCRegionExplain,
    QCRegionMetric,
    QCReport,
    QCRun,
    QCTissue,
)
from app.services.upload_service import get_record, resolve_ready_path

logger = get_logger(__name__)

CITATION = (
    "Weng Z. et al. GrandQC: a comprehensive solution to quality control problem in "
    "digital pathology. Nature Communications (2024). "
    "https://doi.org/10.1038/s41467-024-54769-y"
)

LICENCE_NOTE = (
    "GrandQC is distributed under a non-commercial licence. Use of these checkpoints is "
    "subject to the terms of the original GrandQC licence."
)

DOWNLOADS = (
    QCDownload(
        label="GrandQC tissue segmentation model",
        url="https://zenodo.org/records/14507273",
        files=[models.TISSUE_CHECKPOINT],
        target_dir="models/grandqc/td/",
    ),
    QCDownload(
        label="GrandQC artefact segmentation models (5x / 7x / 10x)",
        url="https://zenodo.org/records/14041538",
        files=list(models.ARTEFACT_CHECKPOINTS.values()),
        target_dir="models/grandqc/qc/",
    ),
)


class QCError(ValueError):
    """A client-correctable problem: no slide, no models, or a run already going."""


def _camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part.capitalize() for part in rest)


#: Metric keys as the API spells them. The internals use snake_case; the wire
#: format is camelCase like every other schema in this app.
API_METRIC_KEYS: tuple[str, ...] = tuple(_camel(key) for key in METRIC_KEYS)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _area_mm2(pixels: int, mpp: float | None) -> float | None:
    """Pixel count to square millimetres, or None with no physical scale."""
    if not mpp:
        return None
    return round(pixels * (mpp / 1000.0) ** 2, 4)


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

    #: Set by `cancel`, read by the progress callback. A plain bool is enough: one
    #: writer, one reader, and a stale read costs one more patch.
    cancelled: bool = False


#: Where each phase sits in the overall progress bar. Tissue detection is a
#: handful of forward passes; the artefact pass is nearly all the work.
_PHASE_SPAN = {
    "tissue": (0.0, 0.08),
    "artefacts": (0.08, 0.90),
    "rendering": (0.90, 0.97),
    "summarising": (0.97, 1.0),
}


class QCService:
    """Runs step 2, caches the result, and reports honestly on what is missing."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()

    # --- capability ---------------------------------------------------------

    def capability(self) -> QCCapability:
        """What step 2 can do right now, and what to download if it cannot."""
        runtime = models.runtime()
        found = models.discover_checkpoints()
        features_ok, features_problem = self._features_available()

        model_infos: list[QCModelInfo] = [
            QCModelInfo(
                role="tissue",
                name=models.TISSUE_CHECKPOINT,
                found=found.tissue is not None,
                path=str(found.tissue) if found.tissue else None,
                mpp=settings.qc_tissue_model_mpp,
                magnification="1x",
            )
        ]
        for mpp, name in sorted(models.ARTEFACT_CHECKPOINTS.items()):
            path = found.artefacts.get(mpp)
            model_infos.append(
                QCModelInfo(
                    role="artefact",
                    name=name,
                    found=path is not None,
                    path=str(path) if path else None,
                    mpp=mpp,
                    magnification=models.MPP_LABELS.get(mpp),
                )
            )

        available = sorted(found.artefacts)

        # The artefact model is the irreducible requirement: it is the thing
        # step 2 exists to run. The tissue model is an accelerator with a
        # classical stand-in, so its absence degrades a run rather than
        # blocking one.
        can_run = runtime.usable and features_ok and bool(available)
        ready = can_run and found.tissue is not None

        if ready:
            mode = "full"
            reason = (
                f"GrandQC tissue detection and artefact segmentation on {runtime.device}, "
                "with classical feature maps measured alongside."
            )
        elif can_run:
            mode = "degraded"
            reason = (
                f"{models.TISSUE_CHECKPOINT} is missing, so tissue is found by a "
                "saturation/Otsu threshold instead. The artefact map is still GrandQC's, "
                "but that threshold calls pen marks tissue, which skews the tissue area."
            )
        else:
            mode = "unavailable"
            reason = " ".join(
                part
                for part in (self._describe_missing(runtime, found), features_problem)
                if part
            )

        return QCCapability(
            ready=ready,
            mode=mode,
            reason=reason,
            torch_installed=runtime.torch_version is not None,
            torch_version=runtime.torch_version,
            smp_version=runtime.smp_version,
            device=runtime.device,
            device_name=runtime.device_name,
            features_available=features_ok,
            features_problem=features_problem,
            models_root=str(found.root) if found.root else None,
            searched_paths=[str(path) for path in found.searched],
            models=model_infos,
            available_model_mpps=available,
            default_model_mpp=settings.qc_model_mpp,
            downloads=list(DOWNLOADS),
            citation=CITATION,
            licence_note=LICENCE_NOTE,
        )

    @staticmethod
    def _features_available() -> tuple[bool, str | None]:
        from app.pipeline.step02_quality_control.features import features_available

        return features_available()

    @staticmethod
    def _describe_missing(runtime: models.Runtime, found: models.Checkpoints) -> str:
        if not runtime.usable:
            return f"{runtime.problem}."
        if not found.artefacts:
            return (
                "No GrandQC artefact checkpoint (GrandQC_MPP1.pth, GrandQC_MPP15.pth or "
                "GrandQC_MPP2.pth) was found."
            )
        return "GrandQC is ready."

    # --- storage ------------------------------------------------------------

    def _dir(self, upload_id: str) -> Path:
        return settings.qc_dir / upload_id

    def _path(self, upload_id: str, name: str) -> Path:
        return self._dir(upload_id) / name

    def _read_json(self, upload_id: str, name: str) -> dict[str, Any] | None:
        path = self._path(upload_id, name)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable QC artefact %s", path, exc_info=True)
            return None

    def _write_json(self, upload_id: str, name: str, payload: Any) -> None:
        path = self._path(upload_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def asset(self, upload_id: str, name: str) -> bytes:
        """One cached PNG. Raises QCError when the run has not produced it."""
        path = self._path(upload_id, name)
        if not path.is_file():
            raise QCError(
                f"no {name} for this slide yet - run step 2 first (POST /qc/{upload_id}/run)"
            )
        return path.read_bytes()

    # --- params -------------------------------------------------------------

    def resolve_params(self, *, model_mpp: float | None) -> dict[str, Any]:
        """Settle what a run will actually use, refusing what cannot work.

        The mode is not a caller's choice: it is dictated by which checkpoints
        are on disk. Letting a client ask for `full` when the tissue model is
        absent would only produce a run that lies about how it found tissue.
        """
        capability = self.capability()
        if capability.mode == "unavailable":
            raise QCError(capability.reason)

        chosen = round(model_mpp or settings.qc_model_mpp, 1)
        if chosen not in capability.available_model_mpps:
            raise QCError(
                f"no GrandQC checkpoint for {chosen} um/px on disk. Available: "
                f"{capability.available_model_mpps or 'none'}"
            )

        return {
            "mode": capability.mode,
            "model_mpp": chosen,
            "model_label": models.MPP_LABELS.get(chosen, f"{chosen} um/px"),
            "tissue_model_mpp": settings.qc_tissue_model_mpp,
            "patch_size": settings.qc_patch_size,
            "read_from_level_0": settings.qc_read_from_level_0,
            "collect_features": True,
        }

    # --- run ----------------------------------------------------------------

    def start(self, upload_id: str, *, model_mpp: float | None) -> QCRun:
        """Queue a run, or hand back the cached one when the params already match."""
        resolve_ready_path(upload_id=upload_id)  # raises UploadError when not ready
        params = self.resolve_params(model_mpp=model_mpp)

        cached = self._read_json(upload_id, "report.json")
        if cached and cached.get("params") == QCParams(**params).model_dump(by_alias=True):
            return QCRun(**cached["run"])

        with self._lock:
            existing = self._jobs.get(upload_id)
            if existing and existing.state in {"queued", "running"}:
                return self._as_run(existing)
            if any(job.state == "running" for job in self._jobs.values()):
                raise QCError(
                    "a QC run is already in progress for another slide; only one runs at a "
                    "time so neither is starved of cores"
                )
            self._jobs[upload_id] = Job(upload_id=upload_id, params=params)

        return self._as_run(self._jobs[upload_id])

    def cancel(self, upload_id: str) -> QCRun:
        """Ask a running pass to stop, and report the state that leaves.

        Sets a flag the progress callback checks after every patch, so the stop lands
        in about a second rather than at the end of a slide. Cancelling nothing is not
        an error - a viewer clicking cancel as the last patch lands has done nothing
        wrong, and a conflict there would report the race rather than the outcome.
        """
        with self._lock:
            job = self._jobs.get(upload_id)
            if job is not None and job.state in {"queued", "running"}:
                job.cancelled = True
                job.message = "stopping"

        return self.state(upload_id)

    def restart(self, upload_id: str, *, model_mpp: float | None = None) -> QCRun:
        """Throw the cached run away and run again.

        The one thing `start` cannot do: it returns the cache when the parameters
        match, which is what makes revisiting the screen instant and is exactly wrong
        when the viewer is asking for a fresh run.
        """
        resolve_ready_path(upload_id=upload_id)

        with self._lock:
            job = self._jobs.get(upload_id)
            if job is not None and job.state in {"queued", "running"}:
                raise QCError(
                    "a QC run is already going on this slide - cancel it before "
                    "starting another"
                )
            self._jobs.pop(upload_id, None)

        report = self._path(upload_id, "report.json")
        if report.is_file():
            report.unlink()

        return self.start(upload_id, model_mpp=model_mpp)

    def state(self, upload_id: str) -> QCRun:
        """Current run state: live job if there is one, else whatever is cached."""
        with self._lock:
            job = self._jobs.get(upload_id)
        if job is not None:
            return self._as_run(job)

        cached = self._read_json(upload_id, "report.json")
        if cached:
            return QCRun(**cached["run"])
        return QCRun(upload_id=upload_id, state="idle")

    def _as_run(self, job: Job) -> QCRun:
        return QCRun(
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
            params=QCParams(**job.params),
        )

    def execute(self, upload_id: str) -> None:
        """The background job. Never raises - failures land on the job record."""
        with self._lock:
            job = self._jobs.get(upload_id)
        if job is None or job.state not in {"queued"}:
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
                # A decision, not a fault - its own state, so the screen does not
                # apologise for what the viewer asked for.
                logger.info("QC run cancelled for %s: %s", upload_id, stopped)
                job.state = "cancelled"
                job.phase = None
                job.message = str(stopped)
            except Exception as exc:  # noqa: BLE001 - reported to the client verbatim
                logger.exception("QC run failed for %s", upload_id)
                job.state = "failed"
                job.error = str(exc)
            finally:
                job.duration = time.monotonic() - job.started
                job.finished_at = _now()

    def _progress(self, job: Job) -> inference.Progress:
        def report(phase: str, done: int, total: int) -> None:
            # The one place a cancel is noticed. Raising from here rather than
            # polling a flag at the top of the pass is what makes the stop land
            # promptly: this is already called after every patch.
            if job.cancelled:
                raise RunCancelled(f"stopped after {done:,} of {total:,} patches")

            low, high = _PHASE_SPAN.get(phase, (0.0, 1.0))
            fraction = (done / total) if total else 1.0
            job.phase = phase
            job.done = done
            job.total = total
            job.progress = low + (high - low) * min(1.0, fraction)

        return report

    def _run(self, job: Job) -> None:
        """The actual pipeline: two passes, then rendering, then arithmetic."""
        params = job.params
        path = resolve_ready_path(upload_id=job.upload_id)
        record = get_record(upload_id=job.upload_id)
        progress = self._progress(job)

        runtime = models.runtime()
        found = models.discover_checkpoints()
        device = runtime.device if runtime.usable else "cpu"

        with open_slide(path) as reader:
            base_mpp = reader.mpp
            if not base_mpp:
                raise QCError(
                    "this slide records no microns-per-pixel, so there is no physical scale "
                    "to run the models at. Supply one on step 1 first."
                )

            model_mpp = float(params["model_mpp"])
            patch_size = int(params["patch_size"])

            _, cols, rows = inference.plan_grid(
                reader, base_mpp=base_mpp, model_mpp=model_mpp, patch_size=patch_size
            )
            if cols * rows > settings.qc_max_patches:
                raise QCError(
                    f"this slide needs {cols * rows:,} patches at {model_mpp} um/px, over the "
                    f"{settings.qc_max_patches:,} limit. Run a coarser model (5x = 2.0 um/px) "
                    "or raise QC_MAX_PATCHES."
                )

            # --- pass 1 ---------------------------------------------------
            job.message = "detecting tissue"
            if params["mode"] == "full" and found.tissue is not None:
                tissue_model = models.load_tissue_model(found.tissue, device)
                tissue = inference.detect_tissue(
                    reader,
                    base_mpp=base_mpp,
                    model=tissue_model,
                    target_mpp=float(params["tissue_model_mpp"]),
                    progress=progress,
                )
            else:
                tissue = inference.detect_tissue_fallback(
                    reader, base_mpp=base_mpp, target_mpp=float(params["tissue_model_mpp"])
                )
                progress("tissue", 1, 1)

            # --- pass 2 ---------------------------------------------------
            artefact_path = found.artefact_for(model_mpp)
            if artefact_path is None:
                raise QCError(f"no artefact checkpoint for {model_mpp} um/px")

            job.message = f"segmenting artefacts at {params['model_label']}"
            artefact_model = models.load_artefact_model(artefact_path, device)
            artefacts = inference.segment_artefacts(
                reader,
                base_mpp=base_mpp,
                tissue=tissue,
                model=artefact_model,
                model_mpp=model_mpp,
                patch_size=patch_size,
                from_level_0=bool(params["read_from_level_0"]),
                collect_features=bool(params["collect_features"]),
                progress=progress,
            )

            # --- rendering ------------------------------------------------
            job.phase = "rendering"
            job.message = "rendering overlays"
            progress("rendering", 0, 3)

            directory = self._dir(job.upload_id)
            directory.mkdir(parents=True, exist_ok=True)

            (directory / "overlay.png").write_bytes(
                overlay.artefact_overlay_png(reader, artefacts.mask)
            )
            progress("rendering", 1, 3)
            (directory / "flat.png").write_bytes(overlay.mask_overlay_png(artefacts.mask))
            (directory / "mask.png").write_bytes(overlay.mask_png(artefacts.mask))
            progress("rendering", 2, 3)
            (directory / "tissue.png").write_bytes(
                overlay.tissue_overlay_png(tissue.thumbnail, tissue.mask)
            )
            progress("rendering", 3, 3)

            slide_width, slide_height = reader.dimensions

        # --- arithmetic ---------------------------------------------------
        job.phase = "summarising"
        job.message = "summarising"
        progress("summarising", 0, 1)

        report, internals, grid = self._summarise(
            job=job,
            filename=record.filename,
            base_mpp=base_mpp,
            slide_size=(slide_width, slide_height),
            tissue=tissue,
            artefacts=artefacts,
            device=device,
            runtime=runtime,
            found=found,
        )

        self._write_json(job.upload_id, "report.json", report.model_dump(by_alias=True))
        self._write_json(job.upload_id, "grid.json", grid.model_dump(by_alias=True))
        self._write_json(job.upload_id, "internals.json", internals)
        progress("summarising", 1, 1)

    # --- summarising --------------------------------------------------------

    def _summarise(
        self,
        *,
        job: Job,
        filename: str,
        base_mpp: float,
        slide_size: tuple[int, int],
        tissue: inference.TissueResult,
        artefacts: inference.ArtefactResult,
        device: str,
        runtime: models.Runtime,
        found: models.Checkpoints,
    ) -> tuple[QCReport, dict[str, Any], QCGrid]:
        """Turn masks and cells into the report, all counts exact at model mpp."""
        totals = artefacts.class_totals
        mpp = artefacts.mpp

        tissue_pixels = sum(
            count for class_id, count in totals.items() if class_id in qc_classes.TISSUE_CLASS_IDS
        )
        clean_pixels = totals.get(qc_classes.TISSUE, 0)
        artefact_pixels = tissue_pixels - clean_pixels
        background_pixels = totals.get(qc_classes.BACKGROUND, 0)
        analysed = tissue_pixels + background_pixels

        class_shares: list[QCClassShare] = []
        for item in qc_classes.QC_CLASSES:
            pixels = totals.get(item.id, 0)
            # Background has no share *of tissue* - it is the thing tissue is
            # measured against, so quoting one would be a category error.
            countable = tissue_pixels and item.id != qc_classes.BACKGROUND
            share = (pixels / tissue_pixels) if countable else 0.0
            class_shares.append(
                QCClassShare(
                    id=item.id,
                    key=item.key,
                    label=item.label,
                    blurb=item.blurb,
                    colour=item.hex_colour,
                    is_artefact=item.is_artefact,
                    pixels=pixels,
                    share_of_tissue=round(share, 6),
                    area_mm2=_area_mm2(pixels, mpp),
                    explained_by=_camel(item.explained_by) if item.explained_by else None,
                )
            )

        tissue_summary = QCTissue(
            tissue_pixels=tissue_pixels,
            clean_pixels=clean_pixels,
            artefact_pixels=artefact_pixels,
            background_pixels=background_pixels,
            unanalysed_pixels=artefacts.unanalysed_pixels,
            tissue_area_mm2=_area_mm2(tissue_pixels, mpp),
            clean_area_mm2=_area_mm2(clean_pixels, mpp),
            rejected_area_mm2=_area_mm2(artefact_pixels, mpp),
            rejected_share=round(artefact_pixels / tissue_pixels, 6) if tissue_pixels else 0.0,
            tissue_share_of_slide=round(tissue_pixels / analysed, 6) if analysed else 0.0,
            tissue_source=tissue.source,
        )

        gate = QCGate(
            qc_off_pixels=tissue_pixels,
            qc_on_pixels=clean_pixels,
            qc_off_area_mm2=_area_mm2(tissue_pixels, mpp),
            qc_on_area_mm2=_area_mm2(clean_pixels, mpp),
            removed_area_mm2=_area_mm2(artefact_pixels, mpp),
            removed_share=round(artefact_pixels / tissue_pixels, 6) if tissue_pixels else 0.0,
            note=(
                "This compares the tissue area downstream steps would receive with QC on "
                "against with QC off. It is not a comparison of two IHC scores: steps 3 to "
                "16 are not built, so there is no score yet. What is being shown is the "
                "denominator that score will be divided by."
            ),
        )

        summary = inference.feature_summary(artefacts.cells)
        sample_sizes = inference.patch_counts_by_class(artefacts.cells)

        metrics: list[QCMetricSummary] = []
        baselines: dict[str, float] = {}
        for spec in METRIC_SPECS:
            by_class = summary.get(spec.key, {})
            clean = by_class.get("tissue")
            if clean:
                baselines[spec.key] = clean
            ratios = (
                {key: round(value / clean, 4) for key, value in by_class.items()}
                if clean
                else {}
            )
            metrics.append(
                QCMetricSummary(
                    key=_camel(spec.key),
                    label=spec.label,
                    unit=spec.unit,
                    description=spec.description,
                    low_is_bad=spec.low_is_bad,
                    measured_at_mpp=mpp,
                    clean_mean=round(clean, 6) if clean else None,
                    by_class={key: round(value, 6) for key, value in by_class.items()},
                    ratio_to_clean=ratios,
                    sample_sizes=sample_sizes,
                )
            )

        blocks = max(1, artefacts.blocks_per_patch)
        cell_extent_px = max(1, artefacts.grid_extent_px // blocks)
        measured = sum(1 for cell in artefacts.cells if cell.features is not None)

        grid_meta = QCGridMeta(
            cols=artefacts.cell_cols,
            rows=artefacts.cell_rows,
            patch_cols=artefacts.grid_cols,
            patch_rows=artefacts.grid_rows,
            blocks_per_patch=blocks,
            patch_size=artefacts.patch_size,
            cell_mpp=mpp,
            cell_extent_px=cell_extent_px,
            cell_extent_um=round(cell_extent_px * base_mpp, 1),
            measured_cells=measured,
            skipped_cells=len(artefacts.cells) - measured,
            patches_inferred=artefacts.patches_inferred,
            patches_skipped=artefacts.patches_skipped,
        )

        params = QCParams(**job.params)
        run = self._as_run(job)
        run.state = "ready"
        run.progress = 1.0
        run.phase = None
        run.message = "complete"
        run.finished_at = _now()
        run.duration_seconds = round(time.monotonic() - job.started, 1)

        notes = [
            "GrandQC made every artefact call on this screen. The classical metrics beside "
            "them explain the call; they are not inputs to it.",
            gate.note,
            f"Metrics are measured at {mpp} um/px - the resolution the artefact model runs "
            f"at - in cells of {grid_meta.cell_extent_um:.0f} um. Sharpness is a claim about "
            "a resolution, so click a region to re-measure it at the pipeline's own working "
            "resolution.",
            "Ratios are against this slide's own clean tissue. These metrics are not "
            "comparable between slides, so there is no absolute threshold to quote.",
            f"The mask served for display is reduced to {artefacts.mask_mpp:.0f} um/px. "
            "Every percentage above was counted at full model resolution, not from it.",
        ]
        if tissue.source != "grandqc":
            notes.insert(
                0,
                "Tissue was found by a saturation/Otsu threshold, not by GrandQC's tissue "
                "model. That threshold calls pen marks tissue - download the tissue "
                "checkpoint before trusting the tissue area.",
            )

        model_infos = [
            QCModelInfo(
                role="tissue",
                name=models.TISSUE_CHECKPOINT if tissue.source == "grandqc" else "saturation+Otsu",
                found=tissue.source == "grandqc",
                path=str(found.tissue) if found.tissue and tissue.source == "grandqc" else None,
                mpp=round(tissue.mpp, 3),
                magnification="1x",
                classes=2,
            ),
            QCModelInfo(
                role="artefact",
                name=models.ARTEFACT_CHECKPOINTS.get(round(mpp, 1), "unknown"),
                found=True,
                path=str(found.artefact_for(mpp)) if found.artefact_for(mpp) else None,
                mpp=mpp,
                magnification=models.MPP_LABELS.get(round(mpp, 1)),
                classes=artefacts.model_classes,
            ),
        ]

        report = QCReport(
            upload_id=job.upload_id,
            filename=filename,
            generated_at=_now(),
            params=params,
            run=run,
            classes=class_shares,
            tissue=tissue_summary,
            gate=gate,
            metrics=metrics,
            grid=grid_meta,
            models=model_infos,
            notes=notes,
            citation=CITATION,
        )

        cells = [
            QCGridCell(
                col=cell.col,
                row=cell.row,
                dominant=(
                    item.key if (item := qc_classes.class_by_id(cell.dominant)) else None
                ),
                tissue_share=round(cell.tissue_share, 4),
                metrics=(
                    {_camel(key): round(value, 6) for key, value in cell.features.items()}
                    if cell.features
                    else None
                ),
            )
            for cell in artefacts.cells
        ]

        internals = {
            "base_mpp": base_mpp,
            "slide_width": slide_size[0],
            "slide_height": slide_size[1],
            "mask_mpp": artefacts.mask_mpp,
            "model_mpp": mpp,
            "grid_extent_px": artefacts.grid_extent_px,
            "baselines": baselines,
            "tissue_source": tissue.source,
        }

        return (
            report,
            internals,
            QCGrid(
                upload_id=job.upload_id,
                grid=grid_meta,
                metric_keys=list(API_METRIC_KEYS),
                cells=cells,
            ),
        )

    # --- reading results ----------------------------------------------------

    def report(self, upload_id: str) -> QCReport:
        cached = self._read_json(upload_id, "report.json")
        if cached is None:
            state = self.state(upload_id)
            raise QCError(
                f"no QC report for this slide (run state: {state.state}). "
                f"Start one with POST /qc/{upload_id}/run."
            )
        return QCReport(**cached)

    def artefact_footprint(self, upload_id: str) -> tuple[Any, float, str, str] | None:
        """Step 2's artefact map for a later step to subtract, or None if it never ran.

        Returns `(mask, mask_mpp, tissue_source, generated_at)`. The mask is the
        reduced display mask - which is the right one for this: a consumer is
        masking a thumbnail, not counting pixels, and the counting-versus-drawing
        distinction is documented on `ArtefactResult`.

        Exists so step 3 can ask step 2 for its output through the service that
        owns it, rather than reaching into `data/qc/` and re-deriving what the
        files mean. `generated_at` is returned so a consumer can tell that a
        re-run has invalidated whatever it cached.
        """
        internals = self._read_json(upload_id, "internals.json")
        report = self._read_json(upload_id, "report.json")
        mask = self._load_mask(upload_id)
        if internals is None or report is None or mask is None:
            return None

        return (
            mask,
            float(internals["mask_mpp"]),
            str(internals.get("tissue_source", "unknown")),
            str(report.get("generatedAt") or report.get("generated_at") or ""),
        )

    def grid(self, upload_id: str) -> QCGrid:
        cached = self._read_json(upload_id, "grid.json")
        if cached is None:
            raise QCError("no QC grid for this slide yet - run step 2 first")
        return QCGrid(**cached)

    def heatmap(self, upload_id: str, metric: str) -> bytes:
        """A metric rendered over the patch grid, from the cached cells."""
        internal_key = next((key for key in METRIC_KEYS if _camel(key) == metric), metric)
        if metric_spec(internal_key) is None:
            raise QCError(f"unknown metric {metric!r}; expected one of {list(API_METRIC_KEYS)}")

        grid = self.grid(upload_id)

        @dataclass
        class _Cell:
            col: int
            row: int
            features: dict[str, float] | None

        cells = [
            _Cell(
                col=cell.col,
                row=cell.row,
                features=(
                    {internal_key: cell.metrics[metric]}
                    if cell.metrics and metric in cell.metrics
                    else None
                ),
            )
            for cell in grid.cells
        ]

        png, _ = overlay.feature_heatmap_png(
            cells,
            metric=internal_key,
            grid_cols=grid.grid.cols,
            grid_rows=grid.grid.rows,
        )
        return png

    # --- region explanation -------------------------------------------------

    def _load_mask(self, upload_id: str) -> Any:
        import numpy as np
        from PIL import Image

        path = self._path(upload_id, "mask.png")
        if not path.is_file():
            return None
        with Image.open(path) as image:
            return np.asarray(image.convert("P"))

    def explain_region(
        self, upload_id: str, *, x: int, y: int, size: int, target_mpp: float | None
    ) -> QCRegionExplain:
        """Re-measure one region finely, and say what it is against the baseline."""
        internals = self._read_json(upload_id, "internals.json")
        if internals is None:
            raise QCError("no QC run for this slide yet - run step 2 first")

        path = resolve_ready_path(upload_id=upload_id)
        base_mpp = float(internals["base_mpp"])
        mask = self._load_mask(upload_id)

        with open_slide(path) as reader:
            width, height = reader.dimensions
            size = max(64, min(size, 16384))
            x = max(0, min(x, max(0, width - size)))
            y = max(0, min(y, max(0, height - size)))

            inspection = inference.inspect_region(
                reader,
                base_mpp=base_mpp,
                x=x,
                y=y,
                size_l0=size,
                target_mpp=target_mpp or settings.target_mpp,
                mask=mask,
                mask_mpp=float(internals["mask_mpp"]),
            )

        # The baselines were measured at the model's resolution; this region was
        # measured finer. Ratios across two resolutions would be meaningless, so
        # a baseline is only offered when the scales match.
        comparable = abs(inspection.mpp - float(internals["model_mpp"])) < 1e-3
        baselines: dict[str, float] = internals.get("baselines", {}) if comparable else {}

        metrics: list[QCRegionMetric] = []
        for spec in METRIC_SPECS:
            value = inspection.features.get(spec.key)
            if value is None:
                continue
            clean = baselines.get(spec.key)
            metrics.append(
                QCRegionMetric(
                    key=_camel(spec.key),
                    label=spec.label,
                    unit=spec.unit,
                    description=spec.description,
                    low_is_bad=spec.low_is_bad,
                    value=round(value, 6),
                    clean_mean=round(clean, 6) if clean else None,
                    ratio_to_clean=round(value / clean, 4) if clean else None,
                )
            )

        total = sum(inspection.class_counts.values())
        shares: dict[str, float] = {}
        for class_id, count in inspection.class_counts.items():
            item = qc_classes.class_by_id(class_id)
            if item is not None and total:
                shares[item.key] = round(count / total, 4)

        dominant = qc_classes.class_by_id(inspection.dominant)
        return QCRegionExplain(
            upload_id=upload_id,
            x=x,
            y=y,
            size_px=size,
            measured_at_mpp=round(inspection.mpp, 4),
            dominant=dominant.key if dominant else None,
            class_shares=shares,
            metrics=metrics,
            verdict=self._verdict(dominant, metrics, comparable),
        )

    @staticmethod
    def _verdict(
        dominant: qc_classes.QCClass | None,
        metrics: list[QCRegionMetric],
        comparable: bool,
    ) -> str:
        """One sentence, built from the numbers rather than asserted over them."""
        if dominant is None:
            return "No QC class covers this region."
        if not dominant.is_artefact:
            return (
                f"GrandQC called this {dominant.label.lower()}, so it stays in the analysis."
            )

        expected = _camel(dominant.explained_by) if dominant.explained_by else None
        lead = next((item for item in metrics if item.key == expected), None)

        if lead is None or lead.ratio_to_clean is None:
            scale_note = (
                ""
                if comparable
                else " It was measured finer than the grid, so there is no like-for-like "
                "baseline on this slide to quote a ratio against."
            )
            return (
                f"GrandQC called this {dominant.label.lower()}, so it is excluded from "
                f"analysis.{scale_note}"
            )

        ratio = lead.ratio_to_clean
        comparison = (
            f"{ratio:.0%} of" if ratio < 1 else f"{ratio:.1f}x"
        )
        return (
            f"GrandQC called this {dominant.label.lower()}, so it is excluded. "
            f"{lead.label} here is {comparison} the clean tissue on this same slide, "
            "which is consistent with that call."
        )

    def region_png(self, upload_id: str, *, x: int, y: int, size: int, out: int) -> bytes:
        """The region's own pixels, so the explanation panel has something to show."""
        path = resolve_ready_path(upload_id=upload_id)
        with open_slide(path) as reader:
            width, height = reader.dimensions
            size = max(64, min(size, 16384))
            x = max(0, min(x, max(0, width - size)))
            y = max(0, min(y, max(0, height - size)))
            image = inference.read_patch(
                reader, x=x, y=y, size_l0=size, out=max(64, min(out, 2048))
            )
        return overlay.region_png(image)

    # --- housekeeping -------------------------------------------------------

    def forget(self, upload_id: str) -> None:
        with self._lock:
            self._jobs.pop(upload_id, None)

    def shutdown(self) -> None:
        models.unload_all()


qc_service = QCService()
