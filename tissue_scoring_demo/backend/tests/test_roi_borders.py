"""Tests for step 9's second product - the per-class border regions.

`class_regions` is pure and parameter-free, so like `test_roi_mask.py` this is almost
entirely tested from a hand-built class map with no slide, no model and no disk.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.step08_tissue_type_segmentation.classes import SCORED
from app.pipeline.step08_tissue_type_segmentation.uncertainty import (
    UNCERTAIN,
    UncertaintyLayer,
    UncertaintyParams,
)
from app.pipeline.step09_roi_mask.borders import BORDER_CLASSES, class_regions
from app.pipeline.step09_roi_mask.crops import padded_bbox, region_bbox_level0
from app.pipeline.step09_roi_mask.mask import IN_SITU
from tests.test_roi_mask import CELL, field, make_class_map, paint


def test_stroma_gets_no_regions_even_when_it_covers_the_whole_slide():
    grid = field(10, 10)  # all non_epithelium
    regions = class_regions(make_class_map(grid))

    assert set(regions) == set(BORDER_CLASSES)
    assert all(regions[class_id] == [] for class_id in BORDER_CLASSES)


def test_adjacent_tiles_of_the_same_class_merge_into_one_region():
    grid = field(20, 20)
    paint(grid, slice(4, 12), slice(4, 12), SCORED)
    regions = class_regions(make_class_map(grid))

    assert len(regions[SCORED]) == 1
    assert regions[SCORED][0].cells == 64
    assert regions[IN_SITU] == []


def test_a_single_stray_tile_is_still_its_own_region():
    """Unlike the scored ROI, there is no area cutoff here - the point is to show the
    model's raw call, speckle included."""
    grid = field(12, 12)
    paint(grid, 6, 6, IN_SITU)
    regions = class_regions(make_class_map(grid))

    assert len(regions[IN_SITU]) == 1
    assert regions[IN_SITU][0].cells == 1


def test_regions_are_ranked_largest_first_within_a_class():
    grid = field(30, 30)
    paint(grid, slice(2, 10), slice(2, 10), SCORED)   # 64
    paint(grid, slice(2, 8), slice(14, 20), SCORED)   # 36
    paint(grid, slice(20, 24), slice(20, 24), SCORED)  # 16
    regions = class_regions(make_class_map(grid))

    found = regions[SCORED]
    assert [region.cells for region in found] == [64, 36, 16]
    assert [region.index for region in found] == [0, 1, 2]


def test_uncertain_regions_are_traced_from_the_display_labels_not_the_raw_labels():
    """`display_labels` is where the uncertain overlay lives - `labels` never has it,
    so a region drawn from the raw labels would find nothing to trace."""
    grid = field(16, 16)
    paint(grid, slice(4, 8), slice(4, 8), IN_SITU, value=0.9)  # 16 cells

    class_map = make_class_map(grid)
    unknown = np.zeros((16, 16), dtype=bool)
    unknown[4:8, 4:8] = True
    layer = UncertaintyLayer(
        margin=np.zeros((16, 16), dtype=np.float32),
        disagreement=np.zeros((16, 16), dtype=np.float32),
        implausibility=np.zeros((16, 16), dtype=np.float32),
        score=np.where(unknown, 1.0, 0.0).astype(np.float32),
        unknown=unknown,
        params=UncertaintyParams(),
        sigma_cells=1.0,
        components=1,
        flagged_components=1,
        largest_component_mm2=0.0,
    )
    from dataclasses import replace

    flagged = replace(class_map, uncertainty=layer)
    regions = class_regions(flagged)

    assert regions[IN_SITU] == []
    assert len(regions[UNCERTAIN]) == 1
    assert regions[UNCERTAIN][0].cells == 16


# --- crops ----------------------------------------------------------------------


def test_region_bbox_matches_the_outer_ring_extent():
    grid = field(20, 20)
    paint(grid, slice(4, 8), slice(6, 10), SCORED)
    region = class_regions(make_class_map(grid))[SCORED][0]

    x0, y0, x1, y1 = region_bbox_level0(region)
    assert (x0, y0) == (6 * CELL, 4 * CELL)
    assert (x1, y1) == (10 * CELL, 8 * CELL)


def test_padded_bbox_expands_and_clamps_to_the_slide():
    bbox = (100, 100, 200, 200)
    assert padded_bbox(bbox, 10, slide_width=1000, slide_height=1000) == (90, 90, 210, 210)
    # Padding past the slide edge clamps rather than going negative or past the canvas.
    assert padded_bbox(bbox, 500, slide_width=1000, slide_height=1000) == (0, 0, 700, 700)
