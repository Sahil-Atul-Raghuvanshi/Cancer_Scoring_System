"""What crosses to the IHC slide, and whether it comes back in one piece.

`regions.py` reshapes rings for the warp and puts the warped ones back with the
region they came from. The reshaping is the part worth pinning: rings go out to
the worker as one flat, anonymous list and come back in the same order, so an
off-by-one in the regrouping would silently give a region somebody else's hole.

**What crosses is now step 11's per-pixel boundary**, which `selected_regions`
reads straight off the refinement's own output. `invasive_regions` and
`coverage` are the tile path the step used before that existed; they are still
tested because they still decide what a tile-only run would carry, but nothing
on the pipeline's own path calls them - `ihc_alignment_service` refuses rather
than falling back to them.
"""

from __future__ import annotations

from app.core.config import settings
from app.pipeline.step08_tissue_type_segmentation.classes import SCORED
from app.pipeline.step08_tissue_type_segmentation.uncertainty import UNCERTAIN
from app.pipeline.step09_roi_mask.borders import ClassRegion
from app.pipeline.step09_roi_mask.mask import IN_SITU
from app.pipeline.step12_ihc_alignment.regions import (
    coverage,
    crossable,
    densified_rings,
    invasive_regions,
    regroup,
    selected_regions,
)


def _region(class_id: int, index: int, rings: int = 1, cells: int = 100) -> ClassRegion:
    """A region with `rings` rings: an outer square plus that many holes minus one."""
    square = ((0, 0), (10, 0), (10, 10), (0, 10))
    return ClassRegion(
        class_id=class_id,
        index=index,
        cells=cells,
        area_mm2=round(cells * 0.01, 4),
        rings=tuple(square for _ in range(rings)),
    )


def test_only_invasive_regions_are_carried():
    """DCIS is excluded from the score, so carrying it across would invite it back."""
    regions = {
        SCORED: [_region(SCORED, 0)],
        IN_SITU: [_region(IN_SITU, 0), _region(IN_SITU, 1)],
        UNCERTAIN: [_region(UNCERTAIN, 0)],
    }
    selected = invasive_regions(regions)
    assert [region.class_id for region in selected] == [SCORED]


def test_enough_regions_cross_to_cover_the_tumour():
    """Selection stops on area covered, not on a count.

    Nine equal regions hold a ninth of the tumour each, so reaching 90% of the
    invasive area takes nine of them. A fixed "top three" would have left two
    thirds of the carcinoma unmeasured while reporting a percentage for the
    whole slide - which is the failure this rule exists to prevent, and which it
    did produce: on CAN_00270 the three largest regions are 63% of the invasive
    area.
    """
    regions = {SCORED: [_region(SCORED, index) for index in range(9)]}
    selected = invasive_regions(regions)

    assert len(selected) == 9
    assert coverage(regions, selected) == 1.0


def test_the_selection_stops_once_the_tumour_is_covered():
    """One dominant region is the whole sample; the speckle after it is not needed."""
    regions = {
        SCORED: [
            _region(SCORED, 0, cells=9000),  # 90 mm2
            _region(SCORED, 1, cells=100),
            _region(SCORED, 2, cells=100),
        ]
    }
    selected = invasive_regions(regions)

    assert [region.index for region in selected] == [0]
    assert coverage(regions, selected) > 0.97


def test_regions_too_small_to_carry_a_percentage_are_not_carried():
    """Below the minimum a region holds a handful of fields, so its own
    percentage is 0, 100 or unreadable - and an average of unreadable numbers
    is not more reliable for having more of them."""
    regions = {
        SCORED: [
            _region(SCORED, 0, cells=100),  # 1.0 mm2
            _region(SCORED, 1, cells=5),  # 0.05 mm2, below the floor
            _region(SCORED, 2, cells=5),
        ]
    }
    assert [region.index for region in invasive_regions(regions)] == [0]


def test_a_finely_dispersed_tumour_still_gets_its_largest_region():
    """Every region below the floor must not mean nothing is measured.

    A tumour that is genuinely dispersed is a property of the disease, not a
    reason to refuse to score the case.
    """
    regions = {SCORED: [_region(SCORED, index, cells=3) for index in range(4)]}
    assert len(invasive_regions(regions)) == 1


def test_the_region_cap_is_honoured():
    regions = {SCORED: [_region(SCORED, index) for index in range(60)]}
    assert len(invasive_regions(regions)) <= settings.ihc_alignment_region_count


def test_step_nines_ranking_is_kept():
    """Step 9 already sorted largest first; re-sorting here could disagree with it."""
    regions = {SCORED: [_region(SCORED, index, cells=500 - index) for index in range(5)]}
    assert [region.index for region in invasive_regions(regions)] == [0, 1, 2, 3, 4]


def test_coverage_reports_what_was_actually_reached():
    """The setting is a target; the cap and the size floor can stop it short."""
    regions = {
        SCORED: [_region(SCORED, 0, cells=700), _region(SCORED, 1, cells=300)]
    }
    assert coverage(regions, invasive_regions(regions)) == 1.0
    assert coverage(regions, [regions[SCORED][0]]) == 0.7
    assert coverage(regions, []) == 0.0


def test_a_case_with_no_invasive_tissue_carries_nothing():
    assert invasive_regions({IN_SITU: [_region(IN_SITU, 0)]}) == []


# --- the flatten / regroup round trip ----------------------------------------


def test_every_ring_is_sent_and_the_counts_describe_them():
    selected = crossable([_region(SCORED, 0, rings=3), _region(SCORED, 1, rings=1)])
    flat, counts = densified_rings(selected, he_mpp=0.25)

    assert counts == [3, 1]
    assert len(flat) == 4


def test_regroup_gives_each_region_back_its_own_rings():
    selected = crossable([_region(SCORED, 0, rings=3), _region(SCORED, 1, rings=1)])
    _flat, counts = densified_rings(selected, he_mpp=0.25)

    # Warped rings come back as an anonymous list, tagged here so the test can
    # see where each one landed.
    warped = [[[float(index), float(index)]] * 4 for index in range(4)]
    crossed = regroup(selected, warped, counts, ihc_mpp=0.25)

    assert len(crossed) == 2
    assert len(crossed[0].ihc_rings) == 3
    assert len(crossed[1].ihc_rings) == 1
    # The fourth ring belongs to the second region, not the first.
    assert crossed[1].ihc_rings[0][0] == [3.0, 3.0]


def test_regroup_keeps_the_original_rings_alongside_the_warped_ones():
    """Both are needed: the panels draw one on each slide."""
    selected = crossable([_region(SCORED, 0, rings=1)])
    _flat, counts = densified_rings(selected, he_mpp=0.25)
    crossed = regroup(selected, [[[1.0, 1.0]] * 4], counts, ihc_mpp=0.25)

    assert crossed[0].he_rings == [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]
    assert crossed[0].ihc_rings != crossed[0].he_rings


def test_densifying_actually_adds_vertices_at_slide_scale():
    """A tile-grid ring at 0.2222 um/px must not reach the warp as four corners."""
    selected = crossable(
        [
            ClassRegion(
                class_id=SCORED,
                index=0,
                cells=1,
                area_mm2=1.0,
                # One step-8 window at the default field of view is ~1000 level-0 px.
                rings=(((0, 0), (1000, 0), (1000, 1000), (0, 1000)),),
            )
        ]
    )
    flat, _counts = densified_rings(selected, he_mpp=0.2222)
    assert len(flat[0]) > 20


# --- what actually crosses: step 11's refined foci ---------------------------


def test_refined_foci_cross_one_polygon_at_a_time():
    """Each focus is its own region, holes and all.

    A coarse box routinely refines into several separate foci. Flattened into one
    entry, the first focus would be read as the outer boundary and the rest as holes
    in it - so the tumour would cross as one region with its other foci punched out,
    which rasterises and scores perfectly plausibly and is wrong.
    """
    refined = [
        {"roiId": "ROI-001", "index": 0, "focus": 0, "areaMm2": 0.8,
         "rings": [[[0, 0], [10, 0], [10, 10]], [[2, 2], [4, 2], [4, 4]]]},
        {"roiId": "ROI-001", "index": 0, "focus": 1, "areaMm2": 0.3,
         "rings": [[[50, 50], [60, 50], [60, 60]]]},
    ]
    crossing = selected_regions(refined)

    assert [region.roi_id for region in crossing] == ["ROI-001", "ROI-001"]
    assert [len(region.rings) for region in crossing] == [2, 1]
    assert all(region.source == "beetle_pixel" for region in crossing)


def test_a_refined_focus_reports_no_window_count():
    """`cells` counts step 8's windows, and a pixel boundary is not made of them."""
    refined = [{"roiId": "ROI-002", "index": 1, "areaMm2": 1.5,
                "rings": [[[0, 0], [1, 0], [1, 1]]]}]
    assert selected_regions(refined)[0].cells == 0


def test_nothing_refined_crosses_nothing():
    assert selected_regions([]) == []


# --- how much tumour the selection reaches -----------------------------------


def test_coverage_is_what_the_report_should_publish_not_the_region_count():
    """A count says nothing about how much tumour the score will see.

    Three regions were 63 per cent of CAN_00270's invasive carcinoma, and the
    report said "3 regions carried". The number that decides how much tumour is
    measured has to be the one on the report.
    """
    regions = {
        SCORED: [
            _region(SCORED, 0, cells=2200),
            _region(SCORED, 1, cells=210),
            _region(SCORED, 2, cells=200),
            _region(SCORED, 3, cells=130),
        ]
    }
    selected = invasive_regions(regions)
    reached = coverage(regions, selected)

    # Same count, very different coverage, depending on the sizes - which is the
    # whole reason the count is not the thing to publish.
    assert 0.0 < reached <= 1.0
    assert coverage(regions, regions[SCORED][:1]) < reached


def test_coverage_of_a_selection_that_reaches_everything_is_one():
    regions = {SCORED: [_region(SCORED, 0, cells=100), _region(SCORED, 1, cells=100)]}
    assert coverage(regions, regions[SCORED]) == 1.0
