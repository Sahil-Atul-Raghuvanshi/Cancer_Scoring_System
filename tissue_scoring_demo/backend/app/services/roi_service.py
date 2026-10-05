"""Step 9 - the ROI mask, orchestrated.

**Not a job, and that is the interesting difference from step 8.** Step 8 is a
background run because tens of thousands of forward passes is tens of minutes. The
region itself takes about a tenth of a second - a Gaussian blur, a closing and a
connected-component pass over a grid a few hundred cells a side - so this service
builds on request, with no queue, no progress and no cancel.

The request is not a tenth of a second, and it is worth being exact about where the
rest goes: on a 127 000 px slide the thumbnail costs about four seconds and the
1600 px panels a few seconds each, so a cold build is roughly twenty and a cached read
is milliseconds. That is a slow request and still not a job - nothing is
waiting on a model, and a client that got a job id would poll for twenty seconds to
learn what one call already knew.

The panels are drawn eagerly rather than on first view for a correctness reason, not a
performance one: a panel rendered later could be drawn from different parameters than
the report it sits next to, and a picture that disagrees with the number under it is
worse than a slow build.

The cache exists for a different reason than step 8's: not because rebuilding is
expensive, but because the ROI is *an input to steps 10 to 15*, and those must all
measure inside the same region. A region silently rebuilt with different parameters
between two later steps would give a coherent-looking score computed on two different
denominators. So the built region is written down, and a rebuild is an explicit act
with explicit parameters.

**Its input is step 8's stored class map, read from disk.** Not held in memory from a
run in this process - `tissue_type_service.class_map` exists precisely so step 9 works
after a restart, and going through it rather than reading the npz here keeps one
definition of how that file is shaped.

What lands on disk per upload:

    report.json     the API payload, exactly as served
    roi.npz         the boolean mask on step 8's grid - step 10's input
    seed.png        the raw invasive windows
    smoothed.png    P(invasive) after the blur
    binary.png      the threshold, before morphology
    region.png      the finished ROI with the protected in-situ drawn
    outline.png     the ROI boundary on the scan
    borders.png     every invasive/DCIS/uncertain patch, bordered - see `borders.py`
    borders_on_slide.png    the same borders on the scan, each class faintly tinted
    dcis1..3.png    the largest DCIS regions, cropped from the slide itself
    invasive1..3.png    likewise, for invasive
    export.geojson  every border-class region, importable into QuPath

The second group is a separate product from the first, built alongside it in the same
request because it is a function of the same `ClassMap` and there is no reason to make
a caller ask twice. See `pipeline/step09_roi_mask/borders.py` for why it does not share
`RoiMask`'s parameters, smoothing or morphology.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide
from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_NAMES, SCORED
from app.pipeline.step08_tissue_type_segmentation.inference import ClassMap
from app.pipeline.step08_tissue_type_segmentation.overlay import MAX_SIZE
from app.pipeline.step08_tissue_type_segmentation.uncertainty import UNCERTAIN
from app.pipeline.step09_roi_mask import mask as roi_mask
from app.pipeline.step09_roi_mask import overlay
from app.pipeline.step09_roi_mask.borders import ClassRegion, class_regions
from app.pipeline.step09_roi_mask.crops import (
    crop_png,
    padded_bbox,
    region_bbox_level0,
    region_crop_with_borders_png,
)
from app.pipeline.step09_roi_mask.mask import IN_SITU, RoiError, RoiMask, RoiParams
from app.pipeline.step09_roi_mask.qupath import to_geojson
from app.schemas.roi import (
    RoiLedger,
    RoiParamsModel,
    RoiRegionModel,
    RoiReport,
)
from app.services.calibration_service import calibration_service
from app.services.tissue_type_service import tissue_type_service
from app.services.upload_service import resolve_ready_path

#: Step 8's class name for each border class, and the order `RoiReport.class_regions`
#: and `class_area_mm2` are keyed in. `uncertain` is not in `CLASS_NAMES` - see
#: `classes.py` - so it is spelled out here rather than looked up.
_CLASS_NAME = {IN_SITU: CLASS_NAMES[IN_SITU], SCORED: CLASS_NAMES[SCORED], UNCERTAIN: "uncertain"}

#: Border classes a top-3 crop is built for. Not `UNCERTAIN`: a "cannot be determined"
#: crop is not something the request asked this screen to rank or picture.
_CROP_CLASSES = (IN_SITU, SCORED)
_CROP_NAME = {IN_SITU: "dcis", SCORED: "invasive"}

#: An on-demand region crop is bigger than a top-3 one on purpose: a viewer who has
#: just clicked a specific region off the borders panel is looking at that one region,
#: not scanning past it, so it is worth the larger read.
_REGION_CROP_MAX_SIZE = MAX_SIZE * 2

logger = get_logger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RoiServiceError(ValueError):
    """The ROI could not be produced or read for this slide."""


class RoiService:
    """Build, cache, read and draw one slide's ROI."""

    # --- storage ------------------------------------------------------------

    def _dir(self, upload_id: str) -> Path:
        return settings.roi_dir / upload_id

    def _path(self, upload_id: str, name: str) -> Path:
        return self._dir(upload_id) / name

    def _read_json(self, upload_id: str, name: str) -> dict[str, Any] | None:
        path = self._path(upload_id, name)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("unreadable step 9 artefact %s", path, exc_info=True)
            return None

    def _write_json(self, upload_id: str, name: str, payload: Any) -> None:
        path = self._path(upload_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def asset(self, upload_id: str, name: str) -> bytes:
        """One cached PNG. Raises when the ROI has not been built."""
        path = self._path(upload_id, name)
        if not path.is_file():
            raise RoiServiceError(
                f"no {name} for this slide yet - build the ROI first "
                f"(POST /roi/{upload_id}/build)"
            )
        return path.read_bytes()

    # --- params -------------------------------------------------------------

    def defaults(self) -> RoiParams:
        """The server's configured defaults, as the dataclass the core takes."""
        return RoiParams(
            sigma=settings.roi_sigma,
            threshold=settings.roi_threshold,
            close_cells=settings.roi_close_cells,
            protect_in_situ=settings.roi_protect_in_situ,
            min_area_mm2=settings.roi_min_area_mm2,
            keep_largest=settings.roi_keep_largest,
        )

    def resolve_params(
        self,
        *,
        sigma: float | None = None,
        threshold: float | None = None,
        close_cells: int | None = None,
        protect_in_situ: float | None = None,
        min_area_mm2: float | None = None,
        keep_largest: int | None = None,
    ) -> RoiParams:
        """Caller's overrides on top of the server defaults, validated once here."""
        base = self.defaults()
        return RoiParams(
            sigma=base.sigma if sigma is None else sigma,
            threshold=base.threshold if threshold is None else threshold,
            close_cells=base.close_cells if close_cells is None else close_cells,
            protect_in_situ=base.protect_in_situ if protect_in_situ is None else protect_in_situ,
            min_area_mm2=base.min_area_mm2 if min_area_mm2 is None else min_area_mm2,
            keep_largest=base.keep_largest if keep_largest is None else keep_largest,
        ).validate()

    # --- building -----------------------------------------------------------

    def discard(self, upload_id: str) -> None:
        """Remove this slide's region, because the class map it was drawn from is gone.

        **Not optional, and not symmetry for its own sake.** `build` decides whether
        its cache still applies by comparing its *own* parameters against the ones
        recorded beside the cached JSON - the threshold, the smoothing, the minimum
        area. It has no way to notice that step 8's labels underneath it have changed,
        so a fresh class map otherwise leaves a region that looks current and was
        traced from labels that no longer exist. Step 8 calls this whenever it throws
        its own answer away.
        """
        shutil.rmtree(self._dir(upload_id), ignore_errors=True)

    def build(
        self, upload_id: str, params: RoiParams | None = None, *, rebuild: bool = False
    ) -> RoiReport:
        """Build the ROI and cache it, or return the cached one.

        A cached report is returned only when it was built with the same parameters.
        Parameters that differ are a different region and therefore a different
        denominator, so they force a rebuild rather than being ignored - the failure
        mode otherwise is a caller passing a closing radius, getting a 200, and
        scoring against somebody else's radius.
        """
        wanted = params or self.defaults()
        wanted.validate()
        # Before anything else, so a slide that does not exist is a 404 rather than
        # step 8's "run me first" - a caller who mistyped an id is not a caller who
        # forgot a step, and telling them the wrong one sends them somewhere useless.
        slide_path = resolve_ready_path(upload_id=upload_id)

        if not rebuild:
            cached = self._read_json(upload_id, "report.json")
            if cached and cached.get("params") == _params_payload(wanted):
                return RoiReport.model_validate(cached)

        class_map = tissue_type_service.class_map(upload_id)
        built = roi_mask.build(class_map, wanted)
        regions = class_regions(class_map)

        self._store(upload_id, built, class_map)
        self._render(upload_id, built, class_map)
        self._render_borders(upload_id, class_map)
        self._render_crops(upload_id, class_map, regions, slide_path)
        self._render_qupath(upload_id, regions)

        report = self._describe(upload_id, built, class_map, regions)
        self._write_json(upload_id, "report.json", report.model_dump(by_alias=True))
        return report

    def report(self, upload_id: str) -> RoiReport:
        """The cached report. Raises rather than silently building one.

        Reading and building are separate verbs because they mean different things to
        a caller: a GET that quietly built a region would let a client change the
        denominator by looking at it.
        """
        cached = self._read_json(upload_id, "report.json")
        if cached is None:
            raise RoiServiceError(
                f"no ROI for this slide yet - build it first (POST /roi/{upload_id}/build)"
            )
        return RoiReport.model_validate(cached)

    def roi(self, upload_id: str) -> np.ndarray:
        """The cached boolean mask on step 8's grid. Step 10's input."""
        path = self._path(upload_id, "roi.npz")
        if not path.is_file():
            raise RoiServiceError(
                f"no ROI mask for this slide yet - build it first (POST /roi/{upload_id}/build)"
            )
        with np.load(path) as stored:
            return stored["mask"].astype(bool)

    def panel(self, upload_id: str, name: str) -> bytes:
        """One panel, from cache. Every panel is written at build time.

        With one backfill: a region built before `borders_on_slide` existed has a
        perfectly current `report.json` and no such file, and the alternative to
        drawing it here is telling a caller to rebuild a denominator that has not
        changed. It is a pure function of the stored class map - the same one the
        cached `borders.png` was drawn from - so the picture it produces is the one
        build time would have produced, and it is written down so this happens once.
        """
        if name not in overlay.PANELS:
            raise RoiServiceError(f"unknown panel {name!r}; expected one of {overlay.PANELS}")

        path = self._path(upload_id, f"{name}.png")
        if name == "borders_on_slide" and not path.is_file():
            if self._read_json(upload_id, "report.json") is None:
                raise RoiServiceError(
                    f"no ROI for this slide yet - build it first (POST /roi/{upload_id}/build)"
                )
            self._render_borders(upload_id, tissue_type_service.class_map(upload_id))

        return self.asset(upload_id, f"{name}.png")

    def top_crop(self, upload_id: str, class_name: str, rank: int) -> bytes:
        """One of the top-3 crops. `rank` is 1-based, matching the URL a viewer reads."""
        if class_name not in _CROP_NAME.values():
            raise RoiServiceError(f"unknown class {class_name!r}; expected one of dcis, invasive")
        if not 1 <= rank <= 3:
            raise RoiServiceError(f"rank must be 1, 2 or 3, not {rank}")
        path = self._path(upload_id, f"{class_name}{rank}.png")
        if not path.is_file():
            if self._read_json(upload_id, "report.json") is None:
                raise RoiServiceError(
                    f"no ROI for this slide yet - build it first (POST /roi/{upload_id}/build)"
                )
            raise RoiServiceError(
                f"this slide has fewer than {rank} {class_name} region(s) - there is no "
                f"rank {rank} crop to show"
            )
        return path.read_bytes()

    def region_crop(self, upload_id: str, class_name: str, index: int) -> bytes:
        """One border-class region, enlarged from the slide, borders drawn on it.

        Built on request rather than at build time, unlike the top-3 crops:
        `borders.class_regions` keeps every component with no area cutoff (see its
        module docstring), so a busy slide can carry dozens of uncertain specks, and
        rendering one of these for each of them at build time would draw crops almost
        nothing asks to see. `index` is 0-based, matching the `index` already on every
        `RoiRegionModel` in `class_regions[class_name]` - the same region a viewer
        clicked on the borders panel, not a fresh rank.
        """
        if class_name not in _CLASS_NAME.values():
            raise RoiServiceError(
                f"unknown class {class_name!r}; expected one of {sorted(_CLASS_NAME.values())}"
            )
        class_id = next(cid for cid, name in _CLASS_NAME.items() if name == class_name)

        slide_path = resolve_ready_path(upload_id=upload_id)
        class_map = tissue_type_service.class_map(upload_id)
        regions = class_regions(class_map)
        found = regions.get(class_id, [])
        if not 0 <= index < len(found):
            raise RoiServiceError(
                f"this slide has no {class_name} region at index {index} - it has "
                f"{len(found)}"
            )

        grid = class_map.grid
        pad_px = max(0, round(settings.roi_crop_pad_um / grid.base_mpp))

        with open_slide(slide_path) as reader:
            return region_crop_with_borders_png(
                reader,
                regions,
                class_id,
                index,
                pad_px=pad_px,
                slide_width=grid.slide_width,
                slide_height=grid.slide_height,
                max_size=_REGION_CROP_MAX_SIZE,
            )

    def qupath(self, upload_id: str) -> bytes:
        """The cached GeoJSON export, exactly as written at build time."""
        return self.asset(upload_id, "export.geojson")

    # --- persisting ---------------------------------------------------------

    def _store(self, upload_id: str, built: RoiMask, class_map: ClassMap) -> None:
        """Persist the mask itself. Bool on step 8's grid, so step 10 needs no geometry.

        The grid is not written again here: it already travels with `classmap.npz`, and
        two copies of one geometry is two things that can disagree about where a window
        is. Step 10 reads both files and gets one answer.
        """
        directory = self._dir(upload_id)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "roi.npz", mask=built.mask)

    def _render(self, upload_id: str, built: RoiMask, class_map: ClassMap) -> None:
        """Write the five panels of the scoring region's own march."""
        directory = self._dir(upload_id)
        directory.mkdir(parents=True, exist_ok=True)

        thumbnail, _ = calibration_service.thumbnail(upload_id)
        shape = (thumbnail.shape[0], thumbnail.shape[1])

        (directory / "seed.png").write_bytes(overlay.seed_png(class_map, built, shape))
        (directory / "smoothed.png").write_bytes(overlay.smoothed_png(class_map, built, shape))
        (directory / "binary.png").write_bytes(overlay.binary_png(class_map, built, shape))
        (directory / "region.png").write_bytes(overlay.region_png(class_map, built, shape))
        (directory / "outline.png").write_bytes(
            overlay.outline_png(thumbnail, class_map, built)
        )

    def _render_borders(self, upload_id: str, class_map: ClassMap) -> None:
        """Write the two border panels - the geometry alone, and the same on the scan.

        One thumbnail read for both: they are the same trace pictured twice, and
        drawing them from two reads is two ways for the pair to disagree.
        """
        directory = self._dir(upload_id)
        thumbnail, _ = calibration_service.thumbnail(upload_id)
        shape = (thumbnail.shape[0], thumbnail.shape[1])
        (directory / "borders.png").write_bytes(overlay.borders_png(class_map, shape))
        (directory / "borders_on_slide.png").write_bytes(
            overlay.borders_on_slide_png(thumbnail, class_map)
        )

    def _render_crops(
        self,
        upload_id: str,
        class_map: ClassMap,
        regions: dict[int, list[ClassRegion]],
        slide_path: Path,
    ) -> None:
        """Crop the top 3 DCIS and invasive regions straight from the slide.

        One reader, opened once, for however many of the up-to-six crops this slide
        actually has - most slides will not have three of each, and a missing rank is
        simply not written rather than padded with something empty (`top_crop` reports
        that distinction to the caller rather than papering over it).
        """
        directory = self._dir(upload_id)
        grid = class_map.grid
        pad_px = max(0, round(settings.roi_crop_pad_um / grid.base_mpp))

        with open_slide(slide_path) as reader:
            for class_id in _CROP_CLASSES:
                name = _CROP_NAME[class_id]
                for region in regions.get(class_id, [])[:3]:
                    bbox = padded_bbox(
                        region_bbox_level0(region), pad_px, grid.slide_width, grid.slide_height
                    )
                    png = crop_png(reader, bbox, max_size=MAX_SIZE)
                    (directory / f"{name}{region.index + 1}.png").write_bytes(png)

    def _render_qupath(self, upload_id: str, regions: dict[int, list[ClassRegion]]) -> None:
        """Write the GeoJSON export every border-class region is available in."""
        payload = to_geojson(regions)
        (self._dir(upload_id) / "export.geojson").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    # --- describing ---------------------------------------------------------

    def _describe(
        self,
        upload_id: str,
        built: RoiMask,
        class_map: ClassMap,
        regions: dict[int, list[ClassRegion]],
    ) -> RoiReport:
        tissue_mm2 = round(class_map.classified * built.cell_mm2, 4)
        return RoiReport(
            upload_id=upload_id,
            generated_at=_now(),
            area_mm2=built.area_mm2,
            cells=built.cells,
            tissue_mm2=tissue_mm2,
            roi_share=round(built.cells / max(1, class_map.classified), 4),
            slide_width=class_map.grid.slide_width,
            slide_height=class_map.grid.slide_height,
            regions=[_region_model(region) for region in built.regions],
            holes=built.holes,
            ledger=RoiLedger(
                seed_mm2=built.seed_mm2,
                seed_components=built.seed_components,
                merged_mm2=round(built.merged_cells * built.cell_mm2, 4),
                protected_mm2=built.protected_mm2,
                protected_cells=built.protected_cells,
                dropped_mm2=built.dropped_mm2,
                dropped_components=built.dropped_components,
            ),
            params=RoiParamsModel(**_params_payload(built.params)),
            notes=_notes(built, class_map),
            class_regions={
                _CLASS_NAME[class_id]: [_region_model(region) for region in found]
                for class_id, found in regions.items()
            },
            class_area_mm2={
                _CLASS_NAME[class_id]: round(sum(region.area_mm2 for region in found), 4)
                for class_id, found in regions.items()
            },
        )


def _region_model(region: Any) -> RoiRegionModel:
    """`RoiRegion` or `ClassRegion` as the wire model - both share this shape."""
    return RoiRegionModel(
        index=region.index,
        cells=region.cells,
        area_mm2=region.area_mm2,
        holes=region.holes,
        rings=[[tuple(vertex) for vertex in ring] for ring in region.rings],
    )


def _params_payload(params: RoiParams) -> dict[str, Any]:
    """The params as the report carries them - the cache key, in one place.

    `build` compares this against the stored report to decide whether the cache
    answers the question that was asked, so it must be exactly what is written.
    """
    return {
        "sigma": params.sigma,
        "threshold": params.threshold,
        "closeCells": params.close_cells,
        "protectInSitu": params.protect_in_situ,
        "minAreaMm2": params.min_area_mm2,
        "keepLargest": params.keep_largest,
    }


def _notes(built: RoiMask, class_map: ClassMap) -> list[str]:
    """What this region does and does not license, in the words the screen shows.

    Written from the numbers rather than fixed text, because the sentence that matters
    changes with the slide: a region built with no in-situ carved out is a different
    claim from one where the carve-out moved a tenth of the area.
    """
    notes: list[str] = [
        "Everything after this step is measured inside this region and nowhere else.",
    ]

    if built.merged_cells > 0:
        notes.append(
            f"Merging nearby tumour added {round(built.merged_cells * built.cell_mm2, 2)} mm2 "
            f"that no window was individually called invasive on, and took "
            f"{built.seed_components} separate patches down to {len(built.regions)}."
        )

    if built.params.protect_in_situ <= 0.0:
        notes.append(
            "In-situ protection is off, so any duct of in-situ carcinoma the merge "
            "enclosed is inside the scored region. Rule 5 says it should not be."
        )
    elif built.protected_cells:
        notes.append(
            f"{built.protected_mm2} mm2 of in-situ carcinoma was inside the merged "
            f"region and has been removed from it - that is what the holes are. "
            f"In-situ disease is not scored."
        )

    if built.dropped_components:
        notes.append(
            f"{built.dropped_components} patches below "
            f"{built.params.min_area_mm2} mm2 were dropped as speckle "
            f"({built.dropped_mm2} mm2)."
        )

    if not built.regions:
        notes.append(
            "No region survived. Either this slide holds no invasive tumour, or the "
            "threshold is above what the model reached anywhere on it."
        )

    return notes


roi_service = RoiService()

__all__ = ["RoiError", "RoiService", "RoiServiceError", "roi_service"]
