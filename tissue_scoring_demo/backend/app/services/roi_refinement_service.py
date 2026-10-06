"""Step 11 - BEETLE, region by region, orchestrated.

Job-shaped, like steps 2, 8 and 12: this is minutes of CPU and a request that waits for
it is a request that times out.

What lands on disk per upload:

    report.json             the API payload, exactly as served
    refined.json            every complete region's rings - the authoritative invasive
                            boundary in H&E level-0 pixels, and step 12's input
    coarse.png              the slide with the chosen candidates as tile squares
    refined.png             the slide with what BEETLE drew inside them
    ROI-001/metadata.json   one region's result, standing on its own
    ROI-001/input.png       the padded H&E crop BEETLE was shown
    ROI-001/tile.png        the same crop with step 8's staircase on it
    ROI-001/beetle_mask.png BEETLE's five classes over the box, flat
    ROI-001/beetle_overlay.png  the same crop with the traced boundary on it

**A region is the unit of work, of storage, of failure and of retry.** That one decision
is most of this module. Each region writes its own directory the moment it finishes, so a
screen can show it without waiting for the rest; a region that fails writes its error
there and the pass continues; a restart reads the directories back and re-runs only what
is missing. None of that needs special cases, because none of it is a special case - it
is what storing per region means.

**The combined mask is a derived file, never an accumulator.** `refined.json` is rewritten
from every complete region's metadata each time the pass finishes, rather than appended
to as regions land. An accumulator would have to be correct under cancellation, partial
failure and retry; a recomputation from the regions on disk is correct by construction,
and the regions on disk are the thing that actually survives a restart.
"""

from __future__ import annotations

import json
import shutil
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
from app.pipeline.step11_roi_refinement import contours, overlay
from app.pipeline.step11_roi_refinement import crop as crop_tools
from app.pipeline.step11_roi_refinement import grid as grid_tools
from app.schemas.roi_refinement import (
    RefinedRegionModel,
    RefinementPaint,
    RefinementReport,
    RefinementRun,
    RegionProgress,
)
from app.schemas.roi_selection import RoiCandidateModel
from app.services.roi_selection_service import roi_selection_service
from app.services.tissue_type_service import tissue_type_service
from app.services.upload_service import resolve_ready_path

logger = get_logger(__name__)

#: Grid, in microns, step 9's mask is rasterised on when it is the region source.
STEP9_GRID_UM = 4.0


#: Most windows one poll hands back for painting.
#:
#: A region is minutes and BEETLE answers a window every second or two, so a client
#: polling every two seconds asks for one or two and never meets this. It bounds the
#: pathological case instead: a screen opened after a large region has been running for
#: a while would otherwise ask for its whole backlog of masks in a single response.
_PAINT_CHUNK = 256


def _hex(colour: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*colour)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RefinementError(ValueError):
    """A client-correctable problem: a missing step, or a slide that is not ready."""


@dataclass
class Job:
    """Live state of one slide's pass, mutated by the worker, read by the API."""

    upload_id: str
    #: Which regions this pass is responsible for. A retry names one; a normal run names
    #: everything step 10 ticked.
    roi_ids: tuple[str, ...] = ()
    rebuild: bool = False

    state: str = "queued"
    message: str | None = None
    done: int = 0
    total: int = 0
    windows_done: int = 0
    windows_total: int = 0
    current: str | None = None

    #: One entry per selected region, in the order they will be run, each a dict of the
    #: fields `RegionProgress` carries. Seeded when the pass starts rather than filled
    #: in as regions begin, so every row on screen has a bar from the first poll - a
    #: region that has not started is an empty bar, not a missing one.
    regions: list[dict[str, Any]] = field(default_factory=list)

    #: Geometry for painting the region in progress, or None before the first one
    #: starts. Replaced wholesale when the pass moves to the next region.
    paint: dict[str, Any] | None = None
    #: The region in progress, as flat `row, col, class` triples, and one base64 pixel
    #: mask per window in step with it - so a cursor that counts windows indexes the
    #: masks directly and the triples in threes.
    #:
    #: **Cleared at the start of every region, unlike step 8's whole-slide log.** Step 8
    #: keeps its entire run so that a screen opened late replays the pass from the
    #: beginning onto one picture. Here each region is its own picture, and a viewer who
    #: arrives mid-region wants that region painting - not the six before it, whose
    #: finished overlays are already on disk. It also bounds the memory: one region's
    #: masks rather than a slide's.
    painted: list[int] = field(default_factory=list)
    painted_masks: list[str] = field(default_factory=list)

    started: float = field(default_factory=time.monotonic)
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    duration: float | None = None
    error: str | None = None
    cancelling: bool = False


class RoiRefinementService:
    """Run BEETLE on the chosen regions of one slide, and keep what it drew."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # --- storage ------------------------------------------------------------

    def _dir(self, upload_id: str) -> Path:
        return settings.roi_refinement_dir / upload_id

    def _region_dir(self, upload_id: str, roi_id: str) -> Path:
        return self._dir(upload_id) / roi_id

    def _path(self, upload_id: str, name: str) -> Path:
        return self._dir(upload_id) / name

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable step 11 artefact %s", path, exc_info=True)
            return None

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def discard(self, upload_id: str) -> None:
        """Throw away every refined region for this slide.

        Called when step 8's class map goes: a refined boundary was traced inside a box
        a candidate id named, and on a new class map that id names different tissue - so
        what is on disk is a pixel-accurate answer about a region nobody chose.
        """
        with self._lock:
            self._jobs.pop(upload_id, None)
        shutil.rmtree(self._dir(upload_id), ignore_errors=True)

    # --- reading a stored region --------------------------------------------

    def _stored(self, upload_id: str, roi_id: str) -> dict[str, Any] | None:
        return self._read_json(self._region_dir(upload_id, roi_id) / "metadata.json")

    def _usable(self, stored: dict[str, Any] | None, key: str | None) -> bool:
        """Whether a stored region can be reused instead of re-run.

        Three conditions, and the third is the one that matters. It has to be complete,
        it has to have been produced at the geometry this pass is running at - a boundary
        traced at 4 um/px is not the answer to a request for 1 - and it has to belong to
        the class map that is on disk now. Without the last, a region refined before step
        8 was re-run would be resumed into a report describing a different slide's
        labels, which is precisely the silent failure this step's caching must not have.
        """
        if not stored or stored.get("state") != "complete":
            return False
        if key is not None and stored.get("classMapKey") != key:
            return False
        return (
            float(stored.get("maskMpp") or 0) == float(settings.roi_refinement_mask_mpp)
            and float(stored.get("padUm") or 0) == float(settings.roi_refinement_pad_um)
            and float(stored.get("fieldOfViewUm") or 0)
            == float(settings.roi_refinement_fov_um)
            and float(stored.get("overlap") or 0) == float(settings.roi_refinement_overlap)
            and float(stored.get("groupUm") or 0) == float(settings.roi_refinement_group_um)
            and (stored.get("source") or "beetle") == settings.roi_refinement_source
        )

    # --- the job ------------------------------------------------------------

    def start(
        self,
        upload_id: str,
        *,
        roi_ids: list[str] | None = None,
        rebuild: bool = False,
    ) -> RefinementRun:
        """Queue a pass over the selected regions, or hand back a finished one.

        `roi_ids` restricts the pass, which is how a single failed region is retried
        without disturbing the twelve that succeeded. It is intersected with step 10's
        selection rather than trusted: refining a region nobody ticked would be the one
        thing this step promised not to do.
        """
        resolve_ready_path(upload_id=upload_id)
        selected = roi_selection_service.selected(upload_id)
        if not selected:
            raise RefinementError(
                "no regions are selected. Tick at least one candidate on step 10 - "
                "this step runs BEETLE on what you choose and on nothing else."
            )

        wanted = tuple(selected) if roi_ids is None else tuple(
            one for one in dict.fromkeys(roi_ids) if one in set(selected)
        )
        if roi_ids is not None and not wanted:
            raise RefinementError(
                f"none of {', '.join(roi_ids)} is selected on step 10, so there is "
                "nothing here to run. Tick it there first."
            )

        with self._lock:
            job = self._jobs.get(upload_id)
            if job is not None and job.state in {"queued", "running"}:
                return self._as_run(job)

            key = tissue_type_service.class_map_key(upload_id)
            outstanding = (
                wanted
                if rebuild
                else tuple(
                    one for one in wanted if not self._usable(self._stored(upload_id, one), key)
                )
            )

            if not outstanding:
                # Everything asked for is already on disk at this geometry. The report is
                # still rewritten, because the *selection* may have narrowed since the
                # last pass and the combined mask is a function of it.
                self._finalise(upload_id, selected)
                return RefinementRun(
                    upload_id=upload_id,
                    state="ready",
                    done=len(wanted),
                    total=len(wanted),
                    message="every selected region is already refined",
                    finished_at=_now(),
                )

            self._jobs[upload_id] = Job(
                upload_id=upload_id,
                roi_ids=outstanding,
                rebuild=rebuild,
                total=len(outstanding),
            )
        return self._as_run(self._jobs[upload_id])

    def cancel(self, upload_id: str) -> RefinementRun:
        """Ask a running pass to stop after the region it is on.

        After, not during: a region half-segmented is not a result, and stopping inside
        one would leave a directory that `_usable` would have to learn to distrust. The
        regions already finished keep their directories and are not recomputed on the
        next run, which is the whole value of cancelling a pass like this.
        """
        job = self._jobs.get(upload_id)
        if job is None or job.state not in {"queued", "running"}:
            return self.state(upload_id)
        job.cancelling = True
        job.message = "stopping after this region"
        return self._as_run(job)

    def state(self, upload_id: str, *, painted_since: int | None = None) -> RefinementRun:
        """Where the pass is, and - with `painted_since` - what it has just painted.

        `painted_since` is a position in the *current region's* paint log, not the
        pass's. It resets when the pass moves on, which is why `paint.roi_id` travels
        beside it: a client compares that id with the one its canvas belongs to and
        starts again from zero when they differ, rather than trying to detect a reset
        from a cursor that went backwards.
        """
        job = self._jobs.get(upload_id)
        if job is not None:
            return self._as_run(job, painted_since=painted_since)

        stored = self._read_json(self._path(upload_id, "report.json"))
        if stored is not None:
            report = RefinementReport.model_validate(stored)
            return RefinementRun(
                upload_id=upload_id,
                state=report.state,
                done=report.completed,
                total=report.selected,
                finished_at=report.generated_at,
            )
        raise RefinementError("no refinement has been started for this slide")

    def _as_run(self, job: Job, *, painted_since: int | None = None) -> RefinementRun:
        """The job as the API reports it, with a slice of the paint feed on request.

        The slice is one list operation rather than a loop, which is what makes it safe
        to read while the worker appends to the same lists: a list slice cannot observe
        a half-written element, so the worst case is a window that lands a moment after
        the slice and is picked up by the next poll.
        """
        cells: list[int] = []
        masks: list[str] = []
        cursor = 0
        paint: RefinementPaint | None = None

        if painted_since is not None and job.paint is not None:
            paint = RefinementPaint(**job.paint)
            first = max(0, int(painted_since))
            start = first * 3
            cells = job.painted[start : start + _PAINT_CHUNK * 3]
            cursor = first + len(cells) // 3
            # After the triples and sliced to exactly what they cover, so the two cannot
            # disagree while the worker is appending to both. A mask whose triple has
            # not landed is simply not sent; the other order would send a mask for a
            # window whose position the client does not yet have.
            masks = job.painted_masks[first:cursor]

        return RefinementRun(
            upload_id=job.upload_id,
            state=job.state,
            message=job.message,
            done=job.done,
            total=job.total,
            windows_done=job.windows_done,
            windows_total=job.windows_total,
            current_roi_id=job.current,
            progress=[RegionProgress(**one) for one in job.regions],
            paint=paint,
            painted_cells=cells,
            painted_masks=masks,
            painted_cursor=cursor,
            started_at=job.started_at,
            finished_at=job.finished_at,
            duration=job.duration,
            error=job.error,
        )

    def execute(self, upload_id: str) -> None:
        """Run the queued pass. Called on a worker thread, never awaited."""
        job = self._jobs.get(upload_id)
        if job is None or job.state != "queued":
            return

        job.state = "running"
        job.message = "loading BEETLE"
        started = time.monotonic()

        try:
            self._run(job)
        except RunCancelled as stopped:
            job.state = "cancelled"
            job.message = str(stopped)
        except RefinementError as failure:
            job.state = "failed"
            job.error = str(failure)
            logger.warning("refinement %s failed: %s", upload_id, failure)
        except Exception as failure:  # noqa: BLE001 - a worker thread must not die silently
            job.state = "failed"
            job.error = f"{type(failure).__name__}: {failure}"
            logger.exception("refinement %s crashed", upload_id)
        finally:
            job.finished_at = _now()
            job.duration = round(time.monotonic() - started, 1)
            job.current = None
            # The report is rewritten whatever happened, including after a crash: the
            # regions that did finish are on disk and a report that did not mention them
            # would make a screen offer to recompute work that is already done.
            try:
                self._finalise(upload_id, roi_selection_service.selected(upload_id))
            except Exception:  # noqa: BLE001 - never mask the original failure
                logger.exception("refinement %s could not write its report", upload_id)

    # --- the work itself ----------------------------------------------------

    def _run(self, job: Job) -> None:
        """Load BEETLE once, then take the regions one at a time in rank order."""
        upload_id = job.upload_id
        path = resolve_ready_path(upload_id=upload_id)

        report = roi_selection_service.report(upload_id)
        by_id = {one.roi_id: one for one in report.candidates}
        key = report.class_map_key

        # Rank order, not request order: `ROI-001` is the largest region and a person
        # watching wants the one that matters most to land first. It also makes a pass
        # deterministic, which a set-ordered one would not be.
        queued = sorted(
            (by_id[one] for one in job.roi_ids if one in by_id),
            key=lambda one: one.index,
        )
        job.total = len(queued)
        job.windows_total = sum(one.windows for one in queued)

        # Every region gets its row now, before any of them runs, so the list on screen
        # is complete from the first poll and a region waiting its turn is an empty bar
        # rather than a row that appears later. `windows` is step 9's count for the
        # candidate; `_refine_one` corrects it to the grid's own once the region starts.
        job.regions = [
            {
                "roi_id": one.roi_id,
                "index": one.index,
                "state": "pending",
                "windows_done": 0,
                "windows_total": int(one.windows),
            }
            for one in queued
        ]

        # Imported here rather than at module scope: this pulls in torch, and
        # `app.pipeline.runner` imports every step at start-up.
        from app.pipeline.step08_tissue_type_segmentation import beetle, pixels

        if settings.roi_refinement_source == "step9":
            self._run_from_step9(job, queued, path=path, report=report)
            return

        try:
            loaded = beetle.load(
                settings.roi_refinement_fov_um,
                folds=settings.tissue_type_beetle_folds,
            )
        except beetle.BeetleError as exc:
            raise RefinementError(str(exc)) from exc

        # BEETLE's own palette and legend, sent with the paint rather than written into
        # the client - the same reason step 8 sends its own. A legend on the live paint
        # and a legend on `beetle_mask.png` that could drift apart would eventually say
        # two different things about one colour.
        codes = range(len(beetle.PIXEL_CLASSES))
        palette = [_hex(beetle.CLASS_COLOURS[code]) for code in codes]
        legend = [beetle.CLASS_LABELS[code] for code in codes]

        with open_slide(path) as reader:
            base_mpp = float(reader.mpp or 0.0)
            slide_width, slide_height = reader.dimensions
            grid = grid_tools.build_for(
                upload_id, slide_size=(slide_width, slide_height), base_mpp=base_mpp
            )
            read_window = tissue_type_service.beetle_reader(reader, base_mpp=base_mpp)

            for candidate in queued:
                if job.cancelling:
                    raise RunCancelled(
                        f"stopped after {job.done} of {job.total} regions"
                    )

                job.current = candidate.roi_id
                job.message = f"{candidate.roi_id}: reading the region"
                self._refine_one(
                    job,
                    candidate,
                    row=self._row(job, candidate.roi_id),
                    grid=grid,
                    loaded=loaded,
                    reader=reader,
                    read_window=read_window,
                    base_mpp=base_mpp,
                    slide_width=slide_width,
                    slide_height=slide_height,
                    key=key,
                    segment=pixels.segment,
                    scored_code=beetle.SCORED_CODE,
                    class_names=beetle.PIXEL_CLASSES,
                    palette=palette,
                    legend=legend,
                    # Every selected region ranked above this one, whether or not it is
                    # in this pass: ownership is a property of the selection, so a retry
                    # of one region claims exactly what it claimed the first time.
                    earlier=[
                        one.rings
                        for one in report.candidates
                        if one.roi_id in set(report.selected) and one.index < candidate.index
                    ],
                )
                job.done += 1

        job.state = "ready"
        job.message = "complete"
        job.current = None
        # The feed belongs to a region that is over. Dropping it here is what makes the
        # last region's canvas give way to its finished overlay instead of holding the
        # final frame of the paint beside it.
        job.paint = None
        job.painted = []
        job.painted_masks = []

    def _run_from_step9(self, job: Job, queued: list, *, path: Any, report: Any) -> None:
        """Each selected region's share of step 9's mask, with no BEETLE pass (P-10/P-11).

        **Why this is the default.** Scored against OncoStem's own outlines on the two
        cases they drew cleanly, step 9's mask beat every BEETLE variant: precision /
        recall 97 / 74 % against BEETLE's 99 / 44 % on CAN_00303, and 78 / 82 % against
        63 / 49 % on CAN_00270 - where BEETLE also put 27 % of its area inside the DCIS
        outline against step 9's 12 %. Step 9's mask carries the in-situ carve-out and
        the smoothing, which BEETLE's per-window in-situ calls could not match, and it
        costs nothing to compute. BEETLE stays available as
        `roi_refinement_source = "beetle"`.

        Each region takes the part of step 9's mask inside its own territory - its tile
        outline grown by the pad, less higher-ranked regions' - so no pixel is claimed
        twice and no unselected tumour is swept in.
        """
        from app.services.roi_service import roi_service

        try:
            mask_report = roi_service.report(job.upload_id)
        except Exception as exc:  # noqa: BLE001 - any failure here is "step 9 not built"
            raise RefinementError(
                "step 9 has not built this slide's scoring mask, and step 11 takes each "
                "region's share of it. Run step 9 first."
            ) from exc

        selected = set(report.selected)
        with open_slide(path) as reader:
            base_mpp = float(reader.mpp or 0.0)
            slide_width, slide_height = reader.dimensions
            for candidate in queued:
                if job.cancelling:
                    raise RunCancelled(f"stopped after {job.done} of {job.total} regions")
                job.current = candidate.roi_id
                job.message = f"{candidate.roi_id}: taking its share of step 9's mask"
                self._region_from_step9(
                    job,
                    candidate,
                    row=self._row(job, candidate.roi_id),
                    reader=reader,
                    mask_rings=[region.rings for region in mask_report.regions],
                    base_mpp=base_mpp,
                    slide_width=slide_width,
                    slide_height=slide_height,
                    key=report.class_map_key,
                    earlier=[
                        one.rings
                        for one in report.candidates
                        if one.roi_id in selected and one.index < candidate.index
                    ],
                )
                job.done += 1

        job.state = "ready"
        job.message = "complete"
        job.current = None
        self._finalise(job.upload_id, tuple(report.selected))

    def _region_from_step9(
        self,
        job: Job,
        candidate: RoiCandidateModel,
        *,
        row: dict[str, Any],
        reader: Any,
        mask_rings: list,
        base_mpp: float,
        slide_width: int,
        slide_height: int,
        key: str | None,
        earlier: list,
    ) -> None:
        """One region from step 9's mask: rasterise, keep its territory, trace."""
        from PIL import Image, ImageDraw

        started = time.monotonic()
        directory = self._region_dir(job.upload_id, candidate.roi_id)
        directory.mkdir(parents=True, exist_ok=True)
        box = (
            candidate.x,
            candidate.y,
            candidate.x + candidate.width,
            candidate.y + candidate.height,
        )
        padded = crop_tools.padded_box(
            box,
            pad_um=settings.roi_refinement_pad_um,
            base_mpp=base_mpp,
            slide_width=slide_width,
            slide_height=slide_height,
        )
        # Step 9's rings follow whole step-8 windows, so a few microns per pixel loses
        # nothing and keeps a 20 mm region's canvas small.
        mpp = max(float(settings.roi_refinement_mask_mpp), STEP9_GRID_UM)
        to_grid = base_mpp / mpp
        width = max(1, int(np.ceil((padded[2] - padded[0]) * to_grid)))
        height = max(1, int(np.ceil((padded[3] - padded[1]) * to_grid)))
        record: dict[str, Any] = {
            "roiId": candidate.roi_id,
            "index": candidate.index,
            "slideId": job.upload_id,
            "classMapKey": key,
            "state": "tracing",
            "source": "step9",
            "cropX": padded[0],
            "cropY": padded[1],
            "cropWidth": padded[2] - padded[0],
            "cropHeight": padded[3] - padded[1],
            "padUm": settings.roi_refinement_pad_um,
            "maskMpp": settings.roi_refinement_mask_mpp,
            "fieldOfViewUm": settings.roi_refinement_fov_um,
            "overlap": settings.roi_refinement_overlap,
            "groupUm": settings.roi_refinement_group_um,
            "baseMpp": base_mpp,
            "tileRings": [[list(vertex) for vertex in ring] for ring in candidate.rings],
            "tileAreaMm2": candidate.area_mm2,
            "windows": 0,
        }
        row["state"] = "tracing"
        try:
            self._render_before(job.upload_id, candidate, reader=reader, box=padded)
            inside = np.zeros((height, width), dtype=bool)
            for rings in mask_rings:
                canvas = Image.new("1", (width, height), 0)
                draw = ImageDraw.Draw(canvas)
                for index, ring in enumerate(rings):
                    if len(ring) >= 3:
                        draw.polygon(
                            [((x - padded[0]) * to_grid, (y - padded[1]) * to_grid) for x, y in ring],
                            fill=0 if index else 1,
                        )
                inside |= np.asarray(canvas, dtype=bool)
            claim = crop_tools.territory(
                shape=inside.shape,
                origin=(padded[0], padded[1]),
                mask_mpp=mpp,
                base_mpp=base_mpp,
                own=candidate.rings,
                earlier=earlier,
                pad_um=settings.roi_refinement_pad_um,
            )
            # BEETLE's codes, so the stored panel and the tracer read it like any other:
            # 3 where step 9 scores, 1 ("not tumour") elsewhere in the box.
            mask = np.where(inside & claim, np.uint8(3), np.uint8(1))
            traced = contours.regions(
                mask,
                code=3,
                origin=(padded[0], padded[1]),
                mask_mpp=mpp,
                base_mpp=base_mpp,
                slide_width=slide_width,
                slide_height=slide_height,
                min_component_mm2=settings.roi_refinement_min_component_mm2,
                min_hole_mm2=settings.roi_refinement_min_hole_mm2,
                simplify_um=settings.roi_refinement_simplify_um,
            )
            pieces = [piece.rings for piece in traced]
            piece_areas = [piece.area_mm2 for piece in traced]
            area_mm2 = round(sum(piece_areas), 5)
            self._render_region(
                job.upload_id, candidate, reader=reader, box=padded, pieces=pieces, mask=mask
            )
            record.update(
                state="complete",
                pieces=pieces,
                pieceAreasMm2=piece_areas,
                focusCount=len(traced),
                holes=sum(piece.holes for piece in traced),
                areaMm2=area_mm2,
                keptShare=round(area_mm2 / candidate.area_mm2, 4) if candidate.area_mm2 else 0.0,
                maskWidth=width,
                maskHeight=height,
                classShare={},
                seconds=round(time.monotonic() - started, 2),
                error=None,
            )
            row["state"] = "complete"
        except Exception as failure:  # noqa: BLE001 - one region must not stop the pass
            logger.exception("step 9 region for %s failed", candidate.roi_id)
            record.update(
                state="failed",
                error=f"{type(failure).__name__}: {failure}",
                seconds=round(time.monotonic() - started, 2),
            )
            row["state"] = "failed"
        self._write_json(directory / "metadata.json", record)

    @staticmethod
    def _row(job: Job, roi_id: str) -> dict[str, Any]:
        """The progress row for one region, created if the seeding missed it.

        The fallback is not defensive padding: `_run` seeds from the candidates it
        queued, and a region reaching `_refine_one` that has no row would be a region
        the pass is refining but not reporting. Making one here keeps those two facts
        the same fact.
        """
        for one in job.regions:
            if one["roi_id"] == roi_id:
                return one
        row = {
            "roi_id": roi_id,
            "index": len(job.regions),
            "state": "pending",
            "windows_done": 0,
            "windows_total": 0,
        }
        job.regions.append(row)
        return row

    def _refine_one(
        self,
        job: Job,
        candidate: RoiCandidateModel,
        *,
        row: dict[str, Any],
        grid: Any,
        loaded: Any,
        reader: Any,
        read_window: Any,
        base_mpp: float,
        slide_width: int,
        slide_height: int,
        key: str | None,
        segment: Any,
        scored_code: int,
        class_names: tuple[str, ...],
        palette: list[str],
        legend: list[str],
        earlier: list[list[list[list[float]]]] | None = None,
    ) -> None:
        """One region, end to end, writing its own directory whether or not it worked.

        **A failure here is recorded, not raised.** One region that cannot be segmented -
        because its windows were all gated out, because a read failed, because it sits
        off the edge of the pyramid - is not a reason to abandon the eleven that can be.
        The error is written where the screen can show it beside a retry button, and the
        pass moves on.
        """
        started = time.monotonic()
        directory = self._region_dir(job.upload_id, candidate.roi_id)
        directory.mkdir(parents=True, exist_ok=True)
        box = (
            candidate.x,
            candidate.y,
            candidate.x + candidate.width,
            candidate.y + candidate.height,
        )
        padded = crop_tools.padded_box(
            box,
            pad_um=settings.roi_refinement_pad_um,
            base_mpp=base_mpp,
            slide_width=slide_width,
            slide_height=slide_height,
        )

        record: dict[str, Any] = {
            "roiId": candidate.roi_id,
            "index": candidate.index,
            "slideId": job.upload_id,
            "classMapKey": key,
            "state": "extracting",
            "cropX": padded[0],
            "cropY": padded[1],
            "cropWidth": padded[2] - padded[0],
            "cropHeight": padded[3] - padded[1],
            "padUm": settings.roi_refinement_pad_um,
            "maskMpp": settings.roi_refinement_mask_mpp,
            "fieldOfViewUm": settings.roi_refinement_fov_um,
            "overlap": settings.roi_refinement_overlap,
            "groupUm": settings.roi_refinement_group_um,
            "source": "beetle",
            "baseMpp": base_mpp,
            "tileRings": [[list(vertex) for vertex in ring] for ring in candidate.rings],
            "tileAreaMm2": candidate.area_mm2,
            "windows": candidate.windows,
        }

        row["state"] = "extracting"

        try:
            restricted = crop_tools.restrict(grid, padded)

            # The two panels that do not need the answer, written now. `input.png` is
            # what the live paint lands on and `tile.png` is the "before" beside it, so
            # writing them here is what lets the comparison exist while the region is
            # being segmented rather than only once it is over.
            self._render_before(job.upload_id, candidate, reader=reader, box=padded)

            # The real window count, which is not step 9's: that one counts the
            # candidate's own windows and this counts every window whose core reaches
            # into the *padded* box. Correcting the row and the pass total together
            # keeps the per-region bars and the overall bar measuring the same thing.
            windows = int(restricted.windows)
            job.windows_total += windows - int(row["windows_total"])
            row["windows_total"] = windows
            row["windows_done"] = 0

            # A fresh canvas for this region: geometry first, then the empty log. The
            # order matters to a client that is polling - `paint.roi_id` changing is the
            # signal to start a new canvas, and it must not arrive while the previous
            # region's windows are still in the log behind it.
            job.paint = {
                "roi_id": candidate.roi_id,
                "crop_x": padded[0],
                "crop_y": padded[1],
                "crop_width": padded[2] - padded[0],
                "crop_height": padded[3] - padded[1],
                "span": int(restricted.span),
                "stride": int(restricted.stride),
                "slide_width": int(slide_width),
                "slide_height": int(slide_height),
                "colours": palette,
                "labels": legend,
                "per_pixel": True,
                "mask_px": int(settings.tissue_type_beetle_paint_px),
            }
            job.painted = []
            job.painted_masks = []

            def painted(cells: tuple[tuple[int, int, int, str], ...]) -> None:
                """One window, straight onto the region's canvas.

                Called per window by `pixels.segment`. The masks are appended before the
                triples, so that a poll landing between the two sees a window it has no
                position for - which it drops - rather than a position whose mask is
                missing, which it would draw as a flat colour and never correct.
                """
                for _, _, _, mask in cells:
                    job.painted_masks.append(mask)
                for window_row, window_col, label, _ in cells:
                    job.painted.extend((window_row, window_col, label))

            def advanced(done: int, total: int) -> None:
                """Windows through the network, for this region's bar and the pass's.

                `windows_done` is recomputed from the region's own count rather than
                incremented, because this fires per block while `painted` fires per
                window and an increment in both would count each window twice.
                """
                previous = int(row["windows_done"])
                row["windows_done"] = min(done, int(row["windows_total"]))
                job.windows_done += int(row["windows_done"]) - previous

            job.message = f"{candidate.roi_id}: {restricted.windows:,} windows through BEETLE"
            record["state"] = "segmenting"
            row["state"] = "segmenting"
            result = segment(
                restricted,
                loaded,
                read_window=read_window,
                mask_mpp=settings.roi_refinement_mask_mpp,
                patch_step=settings.tissue_type_beetle_patch_step,
                batch_size=settings.tissue_type_beetle_batch_size,
                paint_px=settings.tissue_type_beetle_paint_px,
                mask_bounds=padded,
                progress=advanced,
                painted=painted,
                should_stop=lambda: job.cancelling,
            )

            # The bar reaches its end when the work does, not when the last block
            # happens to report. `progress` fires per block and the final block can be
            # short, so without this a region can finish at 97%.
            advanced(windows, windows)

            job.message = f"{candidate.roi_id}: tracing the boundary"
            record["state"] = "tracing"
            row["state"] = "tracing"

            # Only this region's own territory is traced (P-10): its tile outline grown by
            # the pad, less what higher-ranked regions own. Outside it is marked as not
            # run, so neither the speck filter nor the grouping can claim it.
            from app.pipeline.step08_tissue_type_segmentation.pixels import OUTSIDE

            claim = crop_tools.territory(
                shape=result.mask.shape,
                origin=result.mask_origin,
                mask_mpp=result.mask_mpp,
                base_mpp=base_mpp,
                own=candidate.rings,
                earlier=list(earlier or []),
                pad_um=settings.roi_refinement_pad_um,
            )
            owned = np.where(claim, result.mask, np.uint8(OUTSIDE))

            traced = contours.regions(
                owned,
                code=scored_code,
                origin=result.mask_origin,
                mask_mpp=result.mask_mpp,
                base_mpp=base_mpp,
                slide_width=slide_width,
                slide_height=slide_height,
                min_component_mm2=settings.roi_refinement_min_component_mm2,
                min_hole_mm2=settings.roi_refinement_min_hole_mm2,
                simplify_um=settings.roi_refinement_simplify_um,
                group_um=settings.roi_refinement_group_um,
                # Never bridged into: glass, windows that did not run, and in-situ
                # disease, which Rule 5 keeps out of the score.
                exclude=_never_grouped(),
            )

            # Grouped, never flattened: each focus keeps its own outer ring, its own
            # holes and its own area. See `RefinedRegionModel.pieces` for what
            # flattening would silently turn a multi-focus region into, and
            # `piece_areas_mm2` for why the areas travel beside them.
            pieces = [piece.rings for piece in traced]
            piece_areas = [piece.area_mm2 for piece in traced]
            area_mm2 = round(sum(piece_areas), 5)

            self._render_region(
                job.upload_id,
                candidate,
                reader=reader,
                box=padded,
                pieces=pieces,
                mask=result.mask,
            )

            record.update(
                state="complete",
                pieces=pieces,
                pieceAreasMm2=piece_areas,
                focusCount=len(traced),
                holes=sum(piece.holes for piece in traced),
                areaMm2=area_mm2,
                keptShare=round(area_mm2 / candidate.area_mm2, 4)
                if candidate.area_mm2
                else 0.0,
                maskWidth=int(result.mask.shape[1]),
                maskHeight=int(result.mask.shape[0]),
                windows=int(restricted.windows),
                classShare=_class_share(result, class_names),
                seconds=round(time.monotonic() - started, 2),
                error=None,
            )
            row["state"] = "complete"
        except RunCancelled:
            # A cancel is not a failure of this region. It leaves no result, and leaving
            # the directory in `pending` is what makes the next pass pick it up again.
            record.update(state="pending", error=None)
            row["state"] = "pending"
            self._write_json(directory / "metadata.json", record)
            raise
        except Exception as failure:  # noqa: BLE001 - one region must not stop the pass
            logger.exception("refinement of %s failed", candidate.roi_id)
            record.update(
                state="failed",
                error=f"{type(failure).__name__}: {failure}",
                seconds=round(time.monotonic() - started, 2),
            )
            row["state"] = "failed"

        self._write_json(directory / "metadata.json", record)

    def _render_before(
        self,
        upload_id: str,
        candidate: RoiCandidateModel,
        *,
        reader: Any,
        box: tuple[int, int, int, int],
    ) -> None:
        """The crop and the coarse square, written before BEETLE runs.

        A second read of the same small box - `_render_region` reads it again at the end
        - and the duplication is the point rather than an oversight. The left and right
        panels of the finished comparison must come from **one** read to be a fair pair,
        so this cannot simply hand its pixels forward; what it produces is the picture
        the live paint needs a minute earlier, and those same bytes are rewritten from
        the matched read when the region is done.

        A failure here is swallowed. Not being able to draw a preview is not a reason to
        abandon a region that BEETLE can still segment: the screen falls back to showing
        the paint on its own, and the panels land at the end as they always did.
        """
        try:
            directory = self._region_dir(upload_id, candidate.roi_id)
            directory.mkdir(parents=True, exist_ok=True)
            panels = overlay.before_pngs(
                reader,
                box,
                tile_rings=candidate.rings,
                max_size=settings.roi_refinement_crop_px,
            )
            for name, payload in panels.items():
                (directory / name).write_bytes(payload)
        except Exception:  # noqa: BLE001 - a preview must not fail a region
            logger.warning(
                "could not pre-render %s; the paint will run without a backdrop",
                candidate.roi_id,
                exc_info=True,
            )

    def _render_region(
        self,
        upload_id: str,
        candidate: RoiCandidateModel,
        *,
        reader: Any,
        box: tuple[int, int, int, int],
        pieces: list[list[list[list[float]]]],
        mask: np.ndarray,
    ) -> None:
        """The four pictures for one region, from one slide read. See `overlay.py`."""
        directory = self._region_dir(upload_id, candidate.roi_id)
        panels = overlay.crop_pngs(
            reader,
            box,
            tile_rings=candidate.rings,
            refined_pieces=pieces,
            max_size=settings.roi_refinement_crop_px,
        )
        for name, payload in panels.items():
            (directory / name).write_bytes(payload)
        (directory / "beetle_mask.png").write_bytes(
            overlay.class_map_png(mask, max_size=settings.roi_refinement_crop_px)
        )

    # --- combining ----------------------------------------------------------

    def _finalise(self, upload_id: str, selected: tuple[str, ...]) -> RefinementReport:
        """Rebuild the combined mask and the report from what is on disk.

        Recomputed from the region directories rather than accumulated during the pass,
        for the reason in this module's docstring: an accumulator would have to be
        correct under cancellation, partial failure and retry, and a recomputation is
        correct by construction.
        """
        selection = roi_selection_service.report(upload_id)
        by_id = {one.roi_id: one for one in selection.candidates}
        key = selection.class_map_key

        regions: list[RefinedRegionModel] = []
        for roi_id in selected:
            candidate = by_id.get(roi_id)
            if candidate is None:
                continue
            stored = self._stored(upload_id, roi_id)
            regions.append(_region_model(roi_id, candidate, stored, key, self._usable))

        complete = [one for one in regions if one.state == "complete"]
        failed = [one for one in regions if one.state == "failed"]

        state: str
        if not regions:
            state = "failed"
        elif len(complete) == len(regions):
            state = "ready"
        elif complete:
            state = "partial"
        else:
            state = "failed"

        self._store_mask(upload_id, complete)
        self._render_slide(upload_id, complete)

        refined_mm2 = round(sum(one.area_mm2 for one in complete), 4)
        tile_mm2 = round(sum(one.tile_area_mm2 for one in complete), 4)
        report = RefinementReport(
            upload_id=upload_id,
            state=state,
            generated_at=_now(),
            slide_width=selection.slide_width,
            slide_height=selection.slide_height,
            model="BEETLE",
            field_of_view_um=settings.roi_refinement_fov_um,
            mask_mpp=settings.roi_refinement_mask_mpp,
            pad_um=settings.roi_refinement_pad_um,
            folds=settings.tissue_type_beetle_folds,
            regions=regions,
            refined_mm2=refined_mm2,
            tile_mm2=tile_mm2,
            kept_share=round(refined_mm2 / tile_mm2, 4) if tile_mm2 else 0.0,
            invasive_mm2=selection.invasive_mm2,
            completed=len(complete),
            failed=len(failed),
            selected=len(regions),
            seconds=round(sum(one.seconds for one in regions), 2),
            windows=sum(one.windows for one in complete),
        )
        report.notes = _notes(report, selection)
        self._write_json(
            self._path(upload_id, "report.json"), report.model_dump(by_alias=True)
        )
        return report

    def _store_mask(self, upload_id: str, complete: list[RefinedRegionModel]) -> None:
        """Write the authoritative boundary: every complete region's rings.

        JSON rather than an array file, because what is stored is a ragged list of
        polygons of different lengths and an array would need a padding convention
        nobody would remember. It is small - a few hundred kilobytes for a busy slide -
        and step 12 reads it once.

        It is a *separate* file from `report.json` even though the report also carries
        the rings, and the separation is the point: this is the pipeline's tumour mask,
        and the one thing step 12 must not be able to do is read a mask out of a
        document that also contains pending rows, failed rows and the coarse boxes they
        came from.
        """
        # **One entry per focus, not per region**, and that is what makes this file the
        # pipeline's tumour mask rather than a description of one. Step 12 warps each
        # entry as a polygon-with-holes and step 13 rasterises it the same way; a region
        # that carried three foci in one entry would come back as the first focus with
        # the other two punched out of it.
        payload = [
            {
                "roiId": one.roi_id,
                "index": one.index,
                "focus": focus,
                "areaMm2": area,
                "rings": rings,
            }
            for one in complete
            for focus, (rings, area) in enumerate(one.focus_areas())
        ]
        self._write_json(self._path(upload_id, "refined.json"), payload)

    def _render_slide(self, upload_id: str, complete: list[RefinedRegionModel]) -> None:
        """The section, twice: the chosen squares, and what BEETLE drew inside them.

        Skipped entirely when nothing completed. A pair of panels showing an empty slide
        would read as "the refinement found no tumour", which is a different claim from
        "the refinement has not produced anything yet".
        """
        if not complete:
            return
        try:
            path = resolve_ready_path(upload_id=upload_id)
        except Exception as exc:  # noqa: BLE001 - panels are not the product
            logger.warning("step 11 could not open the slide to draw its panels: %s", exc)
            return

        with open_slide(path) as reader:
            width, height = reader.dimensions
            self._path(upload_id, "coarse.png").write_bytes(
                overlay.slide_png(
                    reader,
                    [[[list(v) for v in ring] for ring in one.tile_rings] for one in complete],
                    slide_width=width,
                    slide_height=height,
                    colour=overlay.TILE_COLOUR,
                    fill_alpha=overlay.FILL_ALPHA,
                )
            )
            self._path(upload_id, "refined.png").write_bytes(
                overlay.slide_png(
                    reader,
                    [focus for one in complete for focus in one.pieces],
                    slide_width=width,
                    slide_height=height,
                    colour=overlay.REFINED_COLOUR,
                    fill_alpha=overlay.FILL_ALPHA,
                )
            )

    # --- reading it back ----------------------------------------------------

    def report(self, upload_id: str) -> RefinementReport:
        """The cached report. Raises rather than silently running BEETLE."""
        stored = self._read_json(self._path(upload_id, "report.json"))
        if stored is None:
            raise RefinementError(
                f"this slide has not been refined yet - run step 11 first "
                f"(POST /roi-refinement/{upload_id}/run)"
            )
        return RefinementReport.model_validate(stored)

    def regions(self, upload_id: str) -> list[dict[str, Any]]:
        """The authoritative invasive boundary, for step 12.

        One entry per complete region, each with its rings in H&E level-0 pixels. This is
        the method that makes the tile staircase stop being the tumour mask: step 12 asks
        for this, gets pixel-level polygons, and warps those.
        """
        stored = self._read_json(self._path(upload_id, "refined.json"))
        if not stored:
            raise RefinementError(
                "this slide has no refined regions yet. Step 11 is what produces the "
                "pixel-level invasive boundary every step after it measures inside, so "
                f"run it first (POST /roi-refinement/{upload_id}/run)."
            )
        return stored

    def available(self, upload_id: str) -> bool:
        """Whether a refined boundary exists, without raising when it does not."""
        return self._path(upload_id, "refined.json").is_file()

    def asset(self, upload_id: str, roi_id: str, name: str) -> bytes:
        """One region's picture, from cache."""
        allowed = {"input.png", "tile.png", "beetle_mask.png", "beetle_overlay.png"}
        if name not in allowed:
            raise RefinementError(
                f"unknown picture {name!r}; expected one of {sorted(allowed)}"
            )
        path = self._region_dir(upload_id, roi_id) / name
        if not path.is_file():
            raise RefinementError(
                f"no {name} for {roi_id} on this slide - either it has not been refined "
                "yet, or its pass failed before it drew anything."
            )
        return path.read_bytes()

    def panel(self, upload_id: str, name: str) -> bytes:
        """One of the two slide-level panels - `coarse` or `refined`."""
        if name not in {"coarse", "refined"}:
            raise RefinementError(f"unknown panel {name!r}; expected coarse or refined")
        path = self._path(upload_id, f"{name}.png")
        if not path.is_file():
            raise RefinementError(
                "no panels for this slide yet - no selected region has finished refining"
            )
        return path.read_bytes()


def _never_grouped() -> tuple[int, ...]:
    """BEETLE codes step 11's grouping must not bridge into (P-11): glass, in-situ
    disease, and pixels no window ran on. Imported lazily for the reason `_run` gives."""
    from app.pipeline.step08_tissue_type_segmentation import beetle, pixels

    return (
        beetle.GLASS_CODE,
        beetle.PIXEL_CLASSES.index("non_invasive_epithelium"),
        pixels.OUTSIDE,
    )


def _class_share(result: Any, class_names: tuple[str, ...]) -> dict[str, float]:
    """Share of the box's classified pixels each BEETLE class won.

    Over the classified pixels rather than over the whole box: the padded box is a
    rectangle around a region and its corners are routinely off the tissue entirely, so
    a denominator of every pixel would report a smaller tumour share for a region that
    merely sits at an awkward angle.
    """
    total = sum(result.counts)
    if not total:
        return {}
    return {
        name: round(result.counts[code] / total, 4)
        for code, name in enumerate(class_names)
        if code < len(result.counts)
    }


def _region_model(
    roi_id: str,
    candidate: RoiCandidateModel,
    stored: dict[str, Any] | None,
    key: str | None,
    usable: Any,
) -> RefinedRegionModel:
    """One row of the report, from the region's own directory or from nothing.

    A region with no directory is `pending` rather than absent: the screen's list is the
    *selection*, and a row that vanished because it had not run yet would make a pass
    look shorter than it is.
    """
    tile_rings = [[tuple(vertex) for vertex in ring] for ring in candidate.rings]

    if stored is None:
        return RefinedRegionModel(
            roi_id=roi_id,
            index=candidate.index,
            state="pending",
            tile_rings=tile_rings,
            tile_area_mm2=candidate.area_mm2,
            windows=candidate.windows,
        )

    state = stored.get("state", "pending")
    if state == "complete" and not usable(stored, key):
        # Complete, but at a geometry or against a class map this pass is not running
        # at. It is not a result for *this* question, and calling it pending is what
        # sends it back through the queue instead of into the combined mask.
        state = "pending"

    return RefinedRegionModel(
        roi_id=roi_id,
        index=candidate.index,
        state=state,
        error=stored.get("error"),
        crop_x=int(stored.get("cropX") or 0),
        crop_y=int(stored.get("cropY") or 0),
        crop_width=int(stored.get("cropWidth") or 0),
        crop_height=int(stored.get("cropHeight") or 0),
        pad_um=float(stored.get("padUm") or 0.0),
        mask_mpp=float(stored.get("maskMpp") or 0.0),
        base_mpp=float(stored.get("baseMpp") or 0.0),
        mask_width=int(stored.get("maskWidth") or 0),
        mask_height=int(stored.get("maskHeight") or 0),
        tile_rings=tile_rings,
        tile_area_mm2=candidate.area_mm2,
        pieces=stored.get("pieces") or [],
        piece_areas_mm2=[float(one) for one in (stored.get("pieceAreasMm2") or [])],
        focus_count=int(stored.get("focusCount") or 0),
        holes=int(stored.get("holes") or 0),
        area_mm2=float(stored.get("areaMm2") or 0.0),
        kept_share=float(stored.get("keptShare") or 0.0),
        windows=int(stored.get("windows") or candidate.windows),
        seconds=float(stored.get("seconds") or 0.0),
        class_share=stored.get("classShare") or {},
    )


def _notes(report: RefinementReport, selection: Any) -> list[str]:
    """What the refined mask is and is not, written from this slide's own numbers."""
    notes: list[str] = []

    if report.completed:
        notes.append(
            f"{report.completed} region(s) refined. The coarse boxes covered "
            f"{report.tile_mm2:.2f} mm2; BEETLE called {report.refined_mm2:.2f} mm2 of "
            f"that invasive carcinoma - {report.kept_share:.0%}. The rest was stroma, "
            "fat, in-situ disease and glass inside a square drawn around tumour."
        )
        notes.append(
            "From here on the pixel-level boundary is the tumour mask. Step 12 carries "
            "these outlines onto the IHC slide and every measurement after it is taken "
            "inside them - the tile squares are not used again."
        )

    if report.invasive_mm2 > 0 and report.refined_mm2 > 0:
        notes.append(
            f"That is measured against {report.invasive_mm2:.2f} mm2 of invasive "
            "carcinoma step 8 found on the whole slide. The two are not the same "
            "quantity - one is a per-window call and one is a per-pixel boundary - so "
            "their ratio is not a coverage figure and should not be read as one."
        )

    if report.failed:
        notes.append(
            f"{report.failed} region(s) failed and can be retried on their own. The "
            "regions that succeeded are kept and are not recomputed."
        )

    pending = [one for one in report.regions if one.state == "pending"]
    if pending:
        notes.append(
            f"{len(pending)} selected region(s) have not been refined yet, so they are "
            "not in the mask below."
        )

    if not selection.chosen_by_person:
        notes.append(
            "Nobody ticked these regions - they are step 10's default selection. The "
            "score that follows is measured inside them and nowhere else."
        )

    return notes


roi_refinement_service = RoiRefinementService()

__all__ = ["RefinementError", "RoiRefinementService", "roi_refinement_service"]
