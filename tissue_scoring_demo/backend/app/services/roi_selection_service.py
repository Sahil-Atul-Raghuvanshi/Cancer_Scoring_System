"""Step 10 - the candidate review, orchestrated.

**Not a job, like step 9 and for the same reason**: nothing here waits on a model. The
work is one padded slide read per candidate to draw its card - twenty reads of a few
megapixels - so a cold build is seconds and a cached read is milliseconds. A client that
got a job id would poll for ten seconds to learn what one call already knew.

What lands on disk per upload:

    report.json         the API payload, exactly as served
    selection.json      the ticked ids and who ticked them - written separately from
                        the report because it outlives a rebuild of the cards
    ROI-001.png ...     one card per candidate: the padded crop with the tile
                        boundary drawn on it

**The selection is stored apart from the candidates, and that is the one structural
decision in this module.** They change for different reasons: the candidates change when
step 8 changes, the selection changes when a person clicks. Keeping them in one file
would mean either losing a person's answer every time a card was redrawn, or keeping an
answer that refers to regions that no longer exist. Instead the selection records the
class map it was made against, and a selection made against a different one is dropped
with a note rather than silently reinterpreted.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.step08_tissue_type_segmentation.inference import ClassMap
from app.pipeline.step10_roi_selection import candidates as candidate_tools
from app.pipeline.step10_roi_selection.candidates import Candidate, CandidateSet
from app.pipeline.step10_roi_selection.overlay import card_png
from app.pipeline.step11_roi_refinement import crop as crop_tools
from app.pipeline.step11_roi_refinement import grid as grid_tools
from app.schemas.roi_selection import RoiCandidateModel, RoiSelectionReport
from app.services.roi_service import roi_service
from app.services.tissue_type_service import tissue_type_service
from app.services.upload_service import resolve_ready_path

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RoiSelectionError(ValueError):
    """The candidates could not be produced or read for this slide."""


class RoiSelectionService:
    """Offer one slide's coarse invasive regions, and remember which were chosen."""

    # --- storage ------------------------------------------------------------

    def _dir(self, upload_id: str) -> Path:
        return settings.roi_selection_dir / upload_id

    def _path(self, upload_id: str, name: str) -> Path:
        return self._dir(upload_id) / name

    def _read_json(self, upload_id: str, name: str) -> dict[str, Any] | None:
        path = self._path(upload_id, name)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable step 10 artefact %s", path, exc_info=True)
            return None

    def _write_json(self, upload_id: str, name: str, payload: Any) -> None:
        path = self._path(upload_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def discard(self, upload_id: str) -> None:
        """Throw away this slide's candidates *and* its selection.

        Called when step 8's class map goes, for `roi_service.discard`'s reason and one
        of its own: the ids here are ranks within a class map, so a fresh class map does
        not merely invalidate the pictures, it re-points every id at different tissue.
        """
        shutil.rmtree(self._dir(upload_id), ignore_errors=True)

    # --- building -----------------------------------------------------------

    def build(self, upload_id: str, *, rebuild: bool = False) -> RoiSelectionReport:
        """Build the candidate list and its cards, or return the cached one.

        Step 9 is required and is asked for rather than built: the cutoffs this screen
        applies sit on top of step 9's own, and offering a candidate the scored region
        had already discarded would put a region on screen that no later step could use.
        """
        slide_path = resolve_ready_path(upload_id=upload_id)
        # Raises if step 9 has not run. Not rebuilt here - a step that quietly rebuilt
        # the region it is about to offer could offer a different one than the viewer
        # approved on the previous screen.
        roi_service.report(upload_id)

        key = tissue_type_service.class_map_key(upload_id)
        cached = self._read_json(upload_id, "report.json")
        if not rebuild and cached and cached.get("classMapKey") == key:
            return self._with_selection(
                RoiSelectionReport.model_validate(cached), upload_id, key
            )

        class_map = tissue_type_service.class_map(upload_id)
        found = candidate_tools.build(
            class_map,
            slide_id=upload_id,
            min_area_mm2=settings.roi_selection_min_area_mm2,
            max_candidates=settings.roi_selection_max_candidates,
            default_coverage=settings.roi_selection_default_coverage,
        )

        with open_slide(slide_path) as reader:
            base_mpp = float(reader.mpp or 0.0)
            windows = self._price(upload_id, found, reader.dimensions, base_mpp)
            self._render_cards(upload_id, found, reader, class_map, base_mpp)

        report = self._describe(upload_id, found, class_map, windows, key)
        self._write_json(upload_id, "report.json", report.model_dump(by_alias=True))
        return self._with_selection(report, upload_id, key)

    def _price(
        self,
        upload_id: str,
        found: CandidateSet,
        slide_size: tuple[int, int],
        base_mpp: float,
    ) -> dict[str, int]:
        """Forward-pass windows each candidate would cost, before any of it is spent.

        Through the *same* grid builder step 11 runs against, which is the only reason
        this estimate and the eventual run can be trusted to agree. A failure here is
        not fatal: the cost is information on a review screen, and a screen that refused
        to list the regions because it could not price them would be worse than one that
        lists them with no price.
        """
        try:
            grid = grid_tools.build_for(
                upload_id, slide_size=slide_size, base_mpp=base_mpp
            )
        except Exception as exc:  # noqa: BLE001 - a price is not a precondition
            logger.warning("step 10 could not price the candidates: %s", exc)
            return {}

        priced: dict[str, int] = {}
        for one in found.candidates:
            box = crop_tools.padded_box(
                one.bbox,
                pad_um=settings.roi_refinement_pad_um,
                base_mpp=base_mpp,
                slide_width=slide_size[0],
                slide_height=slide_size[1],
            )
            priced[one.roi_id] = crop_tools.window_count(grid, box)
        return priced

    def _render_cards(
        self,
        upload_id: str,
        found: CandidateSet,
        reader: Any,
        class_map: ClassMap,
        base_mpp: float,
    ) -> None:
        """One card per candidate, from the reader the caller already opened."""
        directory = self._dir(upload_id)
        directory.mkdir(parents=True, exist_ok=True)

        pad_px = max(0, round(settings.roi_crop_pad_um / max(base_mpp, 1e-9)))
        width, height = reader.dimensions

        for one in found.candidates:
            (directory / f"{one.roi_id}.png").write_bytes(
                card_png(
                    reader,
                    one.rings,
                    one.bbox,
                    pad_px=pad_px,
                    slide_width=width,
                    slide_height=height,
                    max_size=settings.roi_selection_crop_px,
                )
            )

    # --- the selection ------------------------------------------------------

    def _selection_record(self, upload_id: str, key: str | None) -> dict[str, Any] | None:
        """The stored selection, if it was made against this class map.

        A selection keyed to a different class map is *dropped*, not translated. The ids
        are area ranks; on a class map rebuilt at another threshold `ROI-007` is other
        tissue, and carrying the tick across would be the quietest possible way to refine
        regions nobody chose.
        """
        record = self._read_json(upload_id, "selection.json")
        if record is None:
            return None
        if key is not None and record.get("classMapKey") != key:
            logger.info(
                "step 10 dropped a selection made against a different class map",
                extra={"upload_id": upload_id},
            )
            return None
        return record

    def _with_selection(
        self, report: RoiSelectionReport, upload_id: str, key: str | None
    ) -> RoiSelectionReport:
        """Fill in the ticked ids - the report's one mutable half."""
        offered = {one.roi_id for one in report.candidates}
        record = self._selection_record(upload_id, key)

        if record is None:
            chosen = [one for one in report.default_selection if one in offered]
            report.selected = chosen
            report.chosen_by_person = False
            report.chosen_at = None
        else:
            # Intersected with what is actually offered rather than trusted: a stored id
            # that is no longer a candidate would otherwise be counted in the totals and
            # then quietly do nothing on step 11.
            report.selected = [
                one for one in record.get("selected", []) if one in offered
            ]
            report.chosen_by_person = bool(record.get("chosenByPerson", True))
            report.chosen_at = record.get("chosenAt")

        by_id = {one.roi_id: one for one in report.candidates}
        report.selected_mm2 = round(
            sum(by_id[one].area_mm2 for one in report.selected), 4
        )
        report.selected_windows = sum(by_id[one].windows for one in report.selected)
        report.notes = _notes(report)
        return report

    def select(self, upload_id: str, roi_ids: list[str]) -> RoiSelectionReport:
        """Record which candidates go to BEETLE. Replaces the selection outright.

        Unknown ids are refused rather than ignored. A client that sent a typo and got a
        200 would believe it had selected a region it had not, and would discover that
        only as a missing result at the end of step 11.
        """
        key = tissue_type_service.class_map_key(upload_id)
        report = self.build(upload_id)
        offered = {one.roi_id for one in report.candidates}

        wanted = list(dict.fromkeys(roi_ids))  # de-duplicated, order kept
        unknown = [one for one in wanted if one not in offered]
        if unknown:
            raise RoiSelectionError(
                f"this slide has no candidate {', '.join(unknown)}. It offers "
                f"{len(offered)} region(s), {candidate_tools.roi_id(0)} upward."
            )

        self._write_json(
            upload_id,
            "selection.json",
            {
                "selected": wanted,
                "classMapKey": key,
                "chosenByPerson": True,
                "chosenAt": _now(),
            },
        )
        return self._with_selection(report, upload_id, key)

    def selected(self, upload_id: str) -> tuple[str, ...]:
        """The ticked ids, for step 11. Falls back to the pipeline's own default.

        Never raises for "nobody has chosen". An unattended run has nobody to tick
        anything and still has to refine something, and the honest something is the
        coverage rule's default - which is the selection step 12 would have made for
        itself before this step existed. `chosen_by_person` is what says which happened.
        """
        return tuple(self.build(upload_id).selected)

    # --- reading ------------------------------------------------------------

    def report(self, upload_id: str) -> RoiSelectionReport:
        """The cached report. Raises rather than silently building one."""
        key = tissue_type_service.class_map_key(upload_id)
        cached = self._read_json(upload_id, "report.json")
        if cached is None:
            raise RoiSelectionError(
                f"no candidates for this slide yet - build them first "
                f"(POST /roi-selection/{upload_id}/build)"
            )
        return self._with_selection(
            RoiSelectionReport.model_validate(cached), upload_id, key
        )

    def candidates(self, upload_id: str) -> list[RoiCandidateModel]:
        """Every offered candidate, as the wire model. Step 11's input."""
        return self.report(upload_id).candidates

    def card(self, upload_id: str, roi_id: str) -> bytes:
        """One candidate's card image, from cache."""
        path = self._path(upload_id, f"{roi_id}.png")
        if not path.is_file():
            raise RoiSelectionError(
                f"no card for {roi_id} on this slide - either it is not a candidate "
                f"here, or the candidates have not been built yet "
                f"(POST /roi-selection/{upload_id}/build)"
            )
        return path.read_bytes()

    # --- describing ---------------------------------------------------------

    def _describe(
        self,
        upload_id: str,
        found: CandidateSet,
        class_map: ClassMap,
        windows: dict[str, int],
        key: str | None,
    ) -> RoiSelectionReport:
        grid = class_map.grid
        return RoiSelectionReport(
            upload_id=upload_id,
            generated_at=_now(),
            slide_width=grid.slide_width,
            slide_height=grid.slide_height,
            candidates=[_candidate_model(one, windows) for one in found.candidates],
            selected=list(found.default_selection),
            default_selection=list(found.default_selection),
            chosen_by_person=False,
            class_map_key=key,
            invasive_mm2=found.invasive_mm2,
            offered_mm2=found.offered_mm2,
            selected_mm2=0.0,
            selected_windows=0,
            dropped_small=found.dropped_small,
            dropped_small_mm2=found.dropped_small_mm2,
            dropped_capped=found.dropped_capped,
            dropped_capped_mm2=found.dropped_capped_mm2,
        )


def _candidate_model(one: Candidate, windows: dict[str, int]) -> RoiCandidateModel:
    x0, y0, x1, y1 = one.bbox
    return RoiCandidateModel(
        roi_id=one.roi_id,
        index=one.index,
        x=x0,
        y=y0,
        width=x1 - x0,
        height=y1 - y0,
        candidate_class=one.candidate_class,
        confidence=one.confidence,
        cells=one.cells,
        area_mm2=one.area_mm2,
        rings=[[tuple(vertex) for vertex in ring] for ring in one.rings],
        tile_cells=[tuple(cell) for cell in one.tile_cells],
        tile_ids=list(one.tile_ids),
        windows=windows.get(one.roi_id, 0),
    )


def _notes(report: RoiSelectionReport) -> list[str]:
    """What this selection commits to, written from the numbers rather than fixed.

    The sentence that matters changes with the slide: a selection covering 97% of the
    invasive carcinoma and one covering 40% are different claims about the score that
    follows, and only one of them needs a warning.
    """
    notes: list[str] = [
        "Only the regions ticked here are segmented per pixel. Everything after this "
        "step measures inside what comes back, and nowhere else.",
    ]

    if not report.candidates:
        notes.append(
            "Step 8 found no invasive carcinoma on this slide, so there is nothing to "
            "refine. Either this section holds none, or step 8's threshold is above what "
            "the model reached anywhere on it."
        )
        return notes

    if report.invasive_mm2 > 0:
        share = report.selected_mm2 / report.invasive_mm2
        notes.append(
            f"{len(report.selected)} of {len(report.candidates)} region(s) ticked, "
            f"{report.selected_mm2:.2f} mm2 - {share:.0%} of the "
            f"{report.invasive_mm2:.2f} mm2 of invasive carcinoma step 8 found. "
            "OncoStem's procedure is that the entire slide is scanned and every field "
            "averaged, so what is left out here is tumour the score never sees."
        )

    if report.dropped_small:
        notes.append(
            f"{report.dropped_small} patch(es) below "
            f"{settings.roi_selection_min_area_mm2} mm2 are not offered "
            f"({report.dropped_small_mm2:.2f} mm2). A patch that small is a window or "
            "two of tile, which is too little context for a segmentation network to "
            "decide a boundary from."
        )

    if report.dropped_capped:
        notes.append(
            f"{report.dropped_capped} further patch(es) past the "
            f"{settings.roi_selection_max_candidates}-region cap are not offered "
            f"({report.dropped_capped_mm2:.2f} mm2)."
        )

    if report.selected_windows:
        notes.append(
            f"That is {report.selected_windows:,} windows through BEETLE. The whole "
            "section would be far more - running only on the regions ticked here is "
            "what makes this step affordable."
        )

    if not report.chosen_by_person:
        notes.append(
            "Nobody has chosen yet, so this is the pipeline's own default: enough of "
            f"the largest regions to cover {settings.roi_selection_default_coverage:.0%} "
            "of the offered area."
        )

    return notes


roi_selection_service = RoiSelectionService()

__all__ = ["RoiSelectionError", "RoiSelectionService", "roi_selection_service"]
