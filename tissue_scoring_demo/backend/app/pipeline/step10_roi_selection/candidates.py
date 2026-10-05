"""Step 9's invasive patches, turned into things a person can choose between.

`borders.class_regions` already traces every patch of invasive tile step 8 drew - 13 to
20 of them on a typical section, plus a tail of single-window specks. Until this step
existed they were a picture. Here they become a *list with identity*, because what comes
after them is BEETLE and BEETLE is the most expensive thing in this pipeline per square
millimetre of slide.

Three decisions live here.

--------------------------------------------------------------------------------
1. The id is a rank, and the rank is by area
--------------------------------------------------------------------------------

`ROI-001` is the largest invasive patch on the slide, `ROI-002` the second largest, and
so on. That is `ClassRegion.index + 1` and nothing else, which means the id is a pure
function of the class map: rebuild step 8 with the same parameters and `ROI-007` is the
same tissue. It is **not** stable across a *different* class map, and it must not
pretend to be - a re-run at a different threshold is a different set of regions, and an
id that survived that would let a selection made on one class map be applied to another.
`slide_id` plus `class_map_key` travel with every candidate for exactly that reason.

--------------------------------------------------------------------------------
2. Specks are not offered, and the cutoff is an area rather than a count
--------------------------------------------------------------------------------

`class_regions` keeps every component with no area cutoff, on purpose - it is showing
the model's raw call, speckle included. A review screen is a different job. A single
window of invasive tile is 224 um square; padded for context and handed to BEETLE it is
a couple of forward passes deciding a boundary from almost no evidence, and on a grid of
twenty cards it is a row of noise between the regions somebody is actually choosing
between. So `min_area_mm2` drops them, `max_candidates` caps the tail, and both numbers
are reported so the screen can say what it left out rather than quietly showing fewer.

--------------------------------------------------------------------------------
3. The default selection is the coverage rule, moved one step earlier
--------------------------------------------------------------------------------

Step 12 already decides how much of the invasive area to carry onto the IHC slide: take
regions in rank order until `ihc_alignment_area_coverage` of the area is covered. That
rule was previously invisible - it happened inside the alignment step, after the point
where anyone could see it. Pre-ticking exactly the regions that rule would have taken
makes the default *visible and editable*: a person who does nothing gets the pipeline's
own answer, and a person who disagrees can see which regions it was about to skip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline.step08_tissue_type_segmentation.classes import SCORED
from app.pipeline.step08_tissue_type_segmentation.inference import ClassMap
from app.pipeline.step09_roi_mask.borders import (
    ClassRegion,
    class_region_cells,
    class_regions,
)
from app.pipeline.step09_roi_mask.crops import region_bbox_level0

#: Half-open level-0 box, `(x0, y0, x1, y1)`. The same tuple `crops.Bbox` is.
Bbox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Candidate:
    """One coarse invasive region, with everything needed to reproduce it later.

    **Every field is either identity, geometry or evidence**, and the split matters. The
    identity fields say which tissue this is and on which slide; the geometry fields are
    what step 11 reads pixels from and what puts its answer back on the slide; the
    evidence fields are what a person is choosing on. Nothing here is a result - this
    step measures nothing that step 9 did not already measure.
    """

    # --- identity ---
    roi_id: str
    slide_id: str
    #: Rank within the invasive class, 0-based - `ClassRegion.index`, which is what the
    #: crop endpoints and the QuPath export already key on.
    index: int

    # --- geometry, all in level-0 pixels of the H&E slide ---
    bbox: Bbox
    #: The rings step 9 traced, outer first then holes. Kept so the card can draw the
    #: tile boundary over the slide crop, and so step 11 can show it beside its own.
    rings: tuple[tuple[tuple[int, int], ...], ...]
    #: The windows of step 8's grid this region is made of, as `(row, col)`. The
    #: "tile coordinates" of the region, and the thing that makes the box reproducible
    #: from the class map alone.
    tile_cells: tuple[tuple[int, int], ...]

    # --- evidence ---
    candidate_class: str
    confidence: float
    cells: int
    area_mm2: float

    @property
    def tile_ids(self) -> tuple[str, ...]:
        """`r<row>c<col>` per window. A readable name for a grid position.

        Derived rather than stored: it is `tile_cells` spelled differently, and two
        copies of one list is two things that can disagree about which windows a region
        holds.
        """
        return tuple(f"r{row}c{col}" for row, col in self.tile_cells)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class CandidateSet:
    """Every candidate this slide offers, and an honest account of what it does not.

    `dropped_small` and `dropped_capped` are reported rather than absorbed because the
    screen's claim is "these are the invasive regions", and a screen that silently shows
    18 of 47 is making a different claim than it looks like it is making.
    """

    candidates: tuple[Candidate, ...]
    #: Ids the coverage rule would take. The screen's pre-ticked set.
    default_selection: tuple[str, ...]

    invasive_mm2: float
    offered_mm2: float
    dropped_small: int
    dropped_small_mm2: float
    dropped_capped: int
    dropped_capped_mm2: float

    def by_id(self, roi_id: str) -> Candidate | None:
        return next((one for one in self.candidates if one.roi_id == roi_id), None)


def roi_id(index: int) -> str:
    """`ROI-007` for the eighth-largest region. 1-based, because the screen is."""
    return f"ROI-{index + 1:03d}"


def build(
    class_map: ClassMap,
    *,
    slide_id: str,
    min_area_mm2: float,
    max_candidates: int,
    default_coverage: float,
) -> CandidateSet:
    """Every invasive patch worth offering, largest first, with the default ticked.

    Pure: no slide is opened and no picture is drawn. What comes out is a description of
    regions the caller already has, which is what lets the service cache it as JSON and
    what lets a test build one from a synthetic class map.
    """
    regions = class_regions(class_map)
    ranked: list[ClassRegion] = list(regions.get(SCORED, []))
    cells_of = class_region_cells(class_map, SCORED)

    invasive_mm2 = sum(region.area_mm2 for region in ranked)

    kept: list[tuple[ClassRegion, np.ndarray]] = []
    dropped_small = 0
    dropped_small_mm2 = 0.0
    for region, piece in zip(ranked, cells_of, strict=True):
        if region.area_mm2 < min_area_mm2:
            dropped_small += 1
            dropped_small_mm2 += region.area_mm2
            continue
        kept.append((region, piece))

    capped = kept[: max(0, int(max_candidates))]
    dropped_capped = len(kept) - len(capped)
    dropped_capped_mm2 = sum(region.area_mm2 for region, _ in kept[len(capped) :])

    candidates = tuple(
        Candidate(
            roi_id=roi_id(region.index),
            slide_id=slide_id,
            index=region.index,
            bbox=region_bbox_level0(region),
            rings=region.rings,
            tile_cells=tuple(
                (int(row), int(col)) for row, col in zip(*np.nonzero(piece), strict=True)
            ),
            candidate_class="invasive",
            confidence=region.confidence,
            cells=region.cells,
            area_mm2=region.area_mm2,
        )
        for region, piece in capped
    )

    return CandidateSet(
        candidates=candidates,
        default_selection=default_selection(candidates, default_coverage),
        invasive_mm2=round(invasive_mm2, 4),
        offered_mm2=round(sum(one.area_mm2 for one in candidates), 4),
        dropped_small=dropped_small,
        dropped_small_mm2=round(dropped_small_mm2, 4),
        dropped_capped=dropped_capped,
        dropped_capped_mm2=round(dropped_capped_mm2, 4),
    )


def default_selection(
    candidates: tuple[Candidate, ...], coverage: float
) -> tuple[str, ...]:
    """Enough of the largest candidates to cover `coverage` of the offered area.

    The share is taken over what is *offered*, not over the whole invasive area, and the
    difference is the point: the regions the cutoffs above removed are already gone, and
    measuring a coverage target against tissue nobody can tick would make the default
    selection quietly grow to compensate for a speck it cannot reach.

    The first candidate is always taken. A slide whose every region is small still gets
    its largest one ticked, because "this tumour is finely dispersed" is a property of
    the disease and not a reason to start the review with nothing selected.
    """
    if not candidates:
        return ()

    total = sum(one.area_mm2 for one in candidates)
    wanted = total * max(0.0, min(1.0, coverage))

    chosen: list[str] = []
    covered = 0.0
    for one in candidates:
        chosen.append(one.roi_id)
        covered += one.area_mm2
        if covered >= wanted:
            break
    return tuple(chosen)


__all__ = [
    "Bbox",
    "Candidate",
    "CandidateSet",
    "build",
    "default_selection",
    "roi_id",
]
