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

**Where the pixels come from.** `_read_haematoxylin` is the one place in this step
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

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.contract import RunCancelled
from app.pipeline.step05_optical_density.tiles import read_tile
from app.pipeline.step08_tissue_type_segmentation import inference, model, overlay
from app.pipeline.step08_tissue_type_segmentation import input as model_input
from app.pipeline.step08_tissue_type_segmentation.classes import (
    CLASS_COLOURS,
    CLASS_LABELS,
    CLASS_MEANING,
    CLASS_NAMES,
    CLASS_PLAIN,
    SCORED,
)
from app.pipeline.step08_tissue_type_segmentation.inference import ClassMap, WindowGrid
from app.schemas.tissue_type import (
    TissueTypeCapability,
    TissueTypeCaveat,
    TissueTypeClass,
    TissueTypeGrid,
    TissueTypeModelInfo,
    TissueTypeParams,
    TissueTypeReport,
    TissueTypeRun,
)
from app.services.calibration_service import calibration_service
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


class TissueTypeError(ValueError):
    """A client-correctable problem: no slide, no checkpoint, or a run already going."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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


class TissueTypeService:
    """Runs step 8, caches the class map, and reports honestly on what it cannot do."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()

    # --- capability ---------------------------------------------------------

    def capability(self) -> TissueTypeCapability:
        """What step 8 can do right now, and which checkpoint would run."""
        state = model.runtime()
        candidates = model.discover()
        selected = settings.tissue_type_model

        infos = [self._model_info(entry, selected=entry.name == selected) for entry in candidates]
        chosen = next((entry for entry in candidates if entry.name == selected), None)

        ready = state.usable and chosen is not None and chosen.usable
        if ready and chosen is not None:
            reason = (
                f"{chosen.name} on {state.device}: a {chosen.tile_px} px haematoxylin window "
                f"at {chosen.mpp} um/px, three classes."
            )
        elif not state.usable:
            reason = str(state.problem)
        elif chosen is None:
            reason = (
                f"no checkpoint named {selected!r} in {model.models_dir()}. Train one with "
                "bcss_hchannel_resnet18 - its publish step writes here, so the trained "
                "model and the served model are the same file."
            )
        else:
            reason = f"{selected} cannot be loaded: {chosen.problem}"

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
    def _model_info(candidate: model.Candidate, *, selected: bool) -> TissueTypeModelInfo:
        return TissueTypeModelInfo(
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
    ) -> dict[str, Any]:
        """Settle what a run will actually use, refusing what cannot work.

        The window geometry is **not** a caller's choice: it is read off the manifest,
        because the model's field of view is a property of the checkpoint and serving a
        different one is the same class of error as serving a different channel. Step 7
        reads the same manifest and lays the same square, so what is recorded here as
        `window_px` and `mpp` is also the geometry of the tiles this run is gated by. What a
        caller may move is the overlap, which is a genuine trade - seam quality against
        a compute bill roughly linear in the window count - and which checkpoint runs,
        because that is a licence decision as much as an accuracy one. Naming no
        overlap inherits step 7's, which is what the tiling screen's picker priced.
        """
        capability = self.capability()
        chosen = name or settings.tissue_type_model

        if not capability.torch_installed:
            raise TissueTypeError(capability.reason)

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

        # Step 7's index is the input, so its own parameters are part of this run's
        # identity: a different tissue threshold is a different set of tiles and
        # therefore a different class map, and a cache that ignored that would serve
        # the old map under the new threshold.
        report = tiling_service.report(upload_id)

        # **The overlap is step 7's, not a setting of this step's own**, and with the
        # two grids now sharing a size and a resolution as well, taking it from
        # anywhere else would put this step on a grid step 7 never priced. `report` is
        # built on demand, so this is step 7's own default when nobody has chosen -
        # the value that screen would have used anyway. `settings.tissue_type_overlap`
        # is now only the fallback for a caller that names one explicitly, which is
        # also the only way the two grids can still come apart.
        default = report.params.overlap
        share = default if overlap is None else float(overlap)
        if not 0.0 <= share < 1.0:
            raise TissueTypeError(
                f"overlap must be at least 0 and less than 1, not {share}; at 1 the stride "
                "would be zero and the grid would never advance"
            )

        return {
            "model": pinned.name,
            "model_sha256": pinned.sha256,
            "licence_track": pinned.licence_track,
            "window_px": pinned.tile_px,
            "window_um": round(pinned.tile_px * pinned.mpp, 2),
            "mpp": pinned.mpp,
            "overlap": round(share, 4),
            "standardise": pinned.standardise,
            "gamma": pinned.gamma,
            "invert_polarity": pinned.invert_polarity,
            # Filled in once the grid exists - the level-0 extents depend on the
            # slide's own resolution, which only the reader knows.
            "span": 0,
            "stride": 0,
            "tile_overlap": report.params.overlap,
            "tissue_threshold": report.params.tissue_threshold,
            "tissue_threshold_source": report.params.tissue_threshold_source,
            "qc_gated": report.params.qc_gated,
            "qc_source": report.params.qc_source,
            "min_tissue_share": settings.tissue_type_min_tissue_share,
            "block_windows": settings.tissue_type_block_windows,
            "batch_size": settings.tissue_type_batch_size,
            "device": capability.device,
        }

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
            )
        }

    # --- run ----------------------------------------------------------------

    def start(
        self, upload_id: str, *, overlap: float | None = None, name: str | None = None
    ) -> TissueTypeRun:
        """Queue a run, or hand back the cached one when the parameters already match."""
        resolve_ready_path(upload_id=upload_id)  # raises UploadError when not ready
        params = self.resolve_params(upload_id, overlap=overlap, name=name)

        cached = self._read_json(upload_id, "report.json")
        if cached and self._cache_key(cached.get("_key", {})) == self._cache_key(params):
            return TissueTypeRun(**cached["run"])

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

    def restart(self, upload_id: str, *, overlap: float | None = None, name: str | None = None) -> TissueTypeRun:
        """Throw the cached pass away and run again.

        The one thing `start` cannot do: it returns the cache when the parameters
        match, which is what makes revisiting the screen instant, and is exactly wrong
        when the viewer is asking for a fresh run. Deleting the report rather than
        adding a `force` flag keeps `start` with one meaning.
        """
        resolve_ready_path(upload_id=upload_id)

        with self._lock:
            job = self._jobs.get(upload_id)
            if job is not None and job.state in {"queued", "running"}:
                raise TissueTypeError(
                    "a pass is already running on this slide - cancel it before "
                    "starting another"
                )
            self._jobs.pop(upload_id, None)

        report = self._path(upload_id, "report.json")
        if report.is_file():
            report.unlink()

        return self.start(upload_id, overlap=overlap, name=name)

    def state(self, upload_id: str) -> TissueTypeRun:
        """Current run state: the live job if there is one, else whatever is cached."""
        with self._lock:
            job = self._jobs.get(upload_id)
        if job is not None:
            return self._as_run(job)

        cached = self._read_json(upload_id, "report.json")
        if cached:
            return TissueTypeRun(**cached["run"])
        return TissueTypeRun(upload_id=upload_id, state="idle")

    def _as_run(self, job: Job) -> TissueTypeRun:
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
        """Build the grid, classify every window on it, then render and describe."""
        upload_id = job.upload_id
        params = job.params

        try:
            pinned = model.load_pinned(params["model"])
        except model.ModelError as exc:
            raise TissueTypeError(str(exc)) from exc

        index = tiling_service.tile_index(upload_id)
        footprint = tiling_service.report(upload_id)
        footprint_mask = tissue_service.footprint(upload_id)
        white = calibration_service.white_point(upload_id)
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
                size=pinned.tile_px,
                mpp=pinned.mpp,
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
            job.message = f"{grid.windows:,} windows through {pinned.name}"
            job.total = grid.windows

            class_map = inference.classify(
                grid,
                net=pinned.net,
                read_haematoxylin=self.haematoxylin_reader(
                    reader, white, base_mpp=float(base_mpp)
                ),
                standardise=pinned.standardise,
                gamma=pinned.gamma,
                invert=pinned.invert_polarity,
                block_windows=int(params["block_windows"]),
                batch_size=int(params["batch_size"]),
                progress=self._progress(job),
                should_stop=self._stopper(job),
            )

        job.phase = "rendering"
        job.message = "drawing the class map"
        thumbnail, _ = calibration_service.thumbnail(upload_id)
        self._render(upload_id, thumbnail, class_map)
        self._store(upload_id, class_map)

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

        report = self._describe(
            class_map,
            upload_id=upload_id,
            filename=get_record(upload_id=upload_id).filename,
            params=params,
            pinned=pinned,
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
                "windows": class_map.classified,
                "tumour_content": round(class_map.tumour_content, 4),
                "seconds": class_map.seconds,
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
    def _stopper(job: Job):
        """Whether this job has been asked to stop. Checked once per block."""
        return lambda: job.cancelled

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
        )

    def class_map(self, upload_id: str) -> ClassMap:
        """The cached class map, rehydrated. Step 9's input.

        Read back from disk rather than held in memory, so step 9 does not depend on
        step 8 having run in the same process - which is exactly the case a pipeline
        runner driving the whole sequence in one call does *not* satisfy on a restart.
        """
        path = self._path(upload_id, "classmap.npz")
        if not path.is_file():
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

        if classes is None or name not in {"map", "flat"}:
            return self.asset(upload_id, f"{name}.png")

        class_map = self.class_map(upload_id)
        if name == "flat":
            thumbnail, _ = calibration_service.thumbnail(upload_id)
            return overlay.flat_png(
                class_map, (thumbnail.shape[0], thumbnail.shape[1]), classes=classes
            )

        thumbnail, _ = calibration_service.thumbnail(upload_id)
        return overlay.map_png(thumbnail, class_map, classes=classes)

    # --- describing ---------------------------------------------------------

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
            tumour_content=round(class_map.tumour_content, 4),
            scored_mm2=class_map.scored_mm2,
            tissue_mm2=tissue_mm2,
            mean_confidence=round(class_map.mean_confidence, 4),
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
        """The three classes with their counts, areas and per-class confidence."""
        shares = class_map.shares
        top = class_map.probabilities.max(axis=-1)

        rows: list[TissueTypeClass] = []
        for label, key in enumerate(CLASS_NAMES):
            here = class_map.labels == label
            rows.append(
                TissueTypeClass(
                    id=label,
                    key=key,
                    label=CLASS_LABELS[label],
                    blurb=CLASS_PLAIN[label],
                    meaning=CLASS_MEANING[label],
                    colour=_hex(CLASS_COLOURS[label]),
                    scored=label == SCORED,
                    windows=int(here.sum()),
                    share=round(shares[label], 4),
                    area_mm2=class_map.areas_mm2[label],
                    mean_confidence=round(float(top[here].mean()), 4) if bool(here.any()) else 0.0,
                )
            )
        return rows

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
                    headline="This result cannot be sold or shipped",
                    detail=(
                        "The model was trained partly on data licensed for non-commercial "
                        "research only, so the model and every number on this screen "
                        f"inherit that restriction. What binds: {'; '.join(binding)}. "
                        "The commercially clean alternative is trained on BCSS alone and "
                        "cannot tell in-situ disease from invasive."
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
                        f"The in-situ class was tested against {human_tiles} "
                        "pathologist-drawn tiles"
                    ),
                    detail=(
                        "Every public dataset that annotates ductal carcinoma in situ as its "
                        "own class is non-commercial, and the one that is not - BCSS - is "
                        f"0.129% in-situ by pixel with {human_tiles} such tiles in its "
                        "held-out set. The bulk of this class's training and test data is "
                        "another model's opinion, unreviewed by any pathologist, so the "
                        "in-situ against invasive boundary here measures agreement with that "
                        "model rather than with a person. That boundary is what Rule 5 gates "
                        "the score on, so it is the number to be most careful about."
                    ),
                )
            )

        if borrowed:
            caveats.append(
                TissueTypeCaveat(
                    key="pseudo_labels",
                    severity="note",
                    headline="Part of the training labels were generated by another model",
                    detail=(
                        "The in-situ class comes from BRACS regions of interest segmented by "
                        "BEETLE's released nnU-Net. Its ceiling is that network's own "
                        "external in-situ accuracy, and its labels are recorded in this "
                        f"model's manifest as drawn by {borrowed.get('labels_drawn_by', 'a model')}. "
                        "The two sources are always reported separately and never averaged."
                    ),
                )
            )

        caveats.append(
            TissueTypeCaveat(
                key="domain",
                severity="warning",
                headline="Trained on H&E slides, served on immunostained ones",
                detail=(
                    "Every training tile came from an H&E section; this slide is "
                    "immunostained. The two are made comparable by throwing the brown away "
                    "and keeping only the blue counterstain, which is why the model reads "
                    "the haematoxylin channel rather than colour - but the blue itself "
                    "differs in shape between the two, not just in strength, and the fix "
                    "for that is a training-time one rather than a measured correction on "
                    "this slide. There is no in-domain test set: nobody has drawn class "
                    "boundaries on an immunostained breast section for this project yet."
                ),
            )
        )

        if class_map.mean_confidence < 0.6:
            caveats.append(
                TissueTypeCaveat(
                    key="confidence",
                    severity="warning",
                    headline=(
                        f"The model averaged {class_map.mean_confidence:.0%} certainty on "
                        "this slide"
                    ),
                    detail=(
                        "A three-class choice bottoms out at 33%, so this is not far above "
                        "guessing on average. Read the certainty panel before trusting the "
                        "boundaries: step 10 thresholds probabilities rather than labels, "
                        "and a map this undecided will move a lot when it does."
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
