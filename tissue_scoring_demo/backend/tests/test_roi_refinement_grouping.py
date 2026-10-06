"""Step 11's outlines when the tumour arrives as scattered cells (P-11).

BEETLE labels pixels. Where a tumour infiltrates as single cells, the invasive class is a
scatter of nucleus-sized dots, and the speck filter used to delete every one of them -
CAN_00267 kept 0.7 % of its tile area and was scored over 1-73 cells. Grouping joins dots
within a set distance into one area before specks are judged.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.step08_tissue_type_segmentation.pixels import OUTSIDE
from app.pipeline.step11_roi_refinement import contours

STROMA, IN_SITU, INVASIVE, GLASS = 1, 2, 3, 0
EXCLUDE = (GLASS, IN_SITU, OUTSIDE)


def _regions(mask: np.ndarray, gap_um: float) -> list[contours.RefinedRegion]:
    return contours.regions(
        mask, code=INVASIVE, origin=(0, 0), mask_mpp=1.0, base_mpp=0.25,
        slide_width=10**6, slide_height=10**6, min_component_mm2=0.005,
        min_hole_mm2=0.005, simplify_um=2.0, group_um=gap_um, exclude=EXCLUDE,
    )


def _scattered_cells(size: int = 400, spacing: int = 30) -> np.ndarray:
    """Stroma with an 8 um invasive dot every `spacing` um - a dispersed tumour."""
    mask = np.full((size, size), STROMA, dtype=np.uint8)
    for y in range(20, size - 20, spacing):
        for x in range(20, size - 20, spacing):
            mask[y : y + 8, x : x + 8] = INVASIVE
    return mask


def test_scattered_cells_are_deleted_without_grouping():
    """The failure: every cell is a speck, so the region comes back empty."""
    assert _regions(_scattered_cells(), gap_um=0) == []


def test_grouping_turns_scattered_cells_into_one_region():
    regions = _regions(_scattered_cells(), gap_um=40)
    assert len(regions) == 1
    # The area the cells occupy, not the sum of their dots: ~360 um square.
    assert regions[0].area_mm2 > 0.1


def test_dots_further_apart_than_the_gap_stay_apart():
    regions = _regions(_scattered_cells(spacing=120), gap_um=40)
    assert regions == []  # each dot alone is still a speck


def test_grouping_does_not_bridge_into_in_situ_or_glass():
    mask = _scattered_cells()
    mask[150:250, 150:250] = IN_SITU
    mask[:, :10] = GLASS
    grouped = contours.group(mask, code=INVASIVE, exclude=EXCLUDE, mask_mpp=1.0, gap_um=40)
    assert not grouped[150:250, 150:250].any()
    assert not grouped[:, :10].any()


def test_a_solid_tumour_is_left_alone():
    mask = np.full((300, 300), STROMA, dtype=np.uint8)
    mask[50:250, 50:250] = INVASIVE
    plain = contours.group(mask, code=INVASIVE, exclude=EXCLUDE, mask_mpp=1.0, gap_um=0)
    grouped = contours.group(mask, code=INVASIVE, exclude=EXCLUDE, mask_mpp=1.0, gap_um=40)
    # The coarse 5 um grid can move a straight edge by at most one grid cell.
    assert abs(int(grouped.sum()) - int(plain.sum())) <= 4 * 200 * 5


def test_every_original_invasive_pixel_survives_grouping():
    mask = _scattered_cells()
    grouped = contours.group(mask, code=INVASIVE, exclude=EXCLUDE, mask_mpp=1.0, gap_um=40)
    assert grouped[mask == INVASIVE].all()


# --- P-10: each region traces only its own territory ---------------------------

from app.pipeline.step11_roi_refinement import crop as crop_tools  # noqa: E402


def _square(x0, y0, x1, y1):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]


def test_a_region_does_not_claim_tumour_far_outside_its_own_tiles():
    claim = crop_tools.territory(
        shape=(2000, 2000), origin=(0, 0), mask_mpp=1.0, base_mpp=1.0,
        own=_square(0, 0, 500, 500), earlier=[], pad_um=100,
    )
    assert claim[250, 250] and claim[550, 250]  # inside, and within the pad
    assert not claim[1500, 1500]  # in the box, far from the region's tiles


def test_two_neighbouring_regions_never_own_the_same_pixel():
    first = _square(0, 0, 500, 500)
    second = _square(560, 0, 1000, 500)  # 60 um apart: their padded areas overlap
    kwargs = dict(shape=(600, 1200), origin=(0, 0), mask_mpp=1.0, base_mpp=1.0, pad_um=100)
    a = crop_tools.territory(own=first, earlier=[], **kwargs)
    b = crop_tools.territory(own=second, earlier=[first], **kwargs)
    assert not (a & b).any()
    assert b[250, 800]  # the second region still owns its own tiles
