"""Step 3 tests.

Split the same way step 2's are: the maths and the mask algorithm need nothing
but numpy and scipy, so they are the bulk of this file and run everywhere. The
endpoints are tested for the states that need no slide on disk.

Every assertion here is about a *property* rather than a value - "the two modes
end up on opposite sides", "closing never removes area", "the cutoffs convert
through mpp" - because the numbers depend on the slide and the properties are
what the pipeline actually relies on.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.common.imaging import (
    between_class_variance,
    class_means,
    histogram_256,
    modal_share,
    otsu_threshold,
    saturation_channel,
    triangle_from_histogram,
)
from app.pipeline.step03_tissue_mask import overlay
from app.pipeline.step03_tissue_mask.mask import (
    _area_px,
    _fill_small_holes,
    _radius_px,
    artefact_footprint,
    build_mask,
    choose_threshold,
)

# --- the shared maths --------------------------------------------------------


def test_otsu_separates_two_modes() -> None:
    """The property that matters is the split, not the number.

    With two perfectly separated modes every threshold in [20, 199] scores
    identically, so asserting a particular value would be asserting which one
    argmax happened to reach first. What callers rely on is that thresholding
    with `value > threshold` puts each mode on its own side.
    """
    values = np.concatenate(
        [np.full(1000, 20, dtype=np.uint8), np.full(1000, 200, dtype=np.uint8)]
    )
    threshold = otsu_threshold(values)
    above = values > threshold

    assert not above[:1000].any()
    assert above[1000:].all()


def test_otsu_refuses_thresholds_that_empty_one_side() -> None:
    """A cut with nothing on one side has no between-class variance to speak of."""
    histogram = histogram_256(
        np.concatenate([np.full(100, 10, dtype=np.uint8), np.full(100, 90, dtype=np.uint8)])
    )
    criterion = between_class_variance(histogram)

    # Level 255 puts everything below the cut, so it must never be selectable.
    assert criterion[255] == -1.0
    assert criterion[9] == -1.0  # nothing at or below 9 either
    assert criterion.argmax() < 90


def test_otsu_criterion_peaks_at_the_threshold_it_returns() -> None:
    """The curve the demo draws and the line it draws on it come from one array."""
    values = np.concatenate(
        [np.full(700, 30, dtype=np.uint8), np.full(300, 180, dtype=np.uint8)]
    )
    histogram = histogram_256(values)
    assert int(between_class_variance(histogram).argmax()) == otsu_threshold(values)


def test_empty_histogram_does_not_divide_by_zero() -> None:
    assert otsu_threshold(np.zeros(0, dtype=np.uint8)) == 0
    assert (between_class_variance(np.zeros(256)) == -1.0).all()


def test_saturation_separates_pale_tissue_from_bright_glass() -> None:
    """The reason this step thresholds saturation and not brightness.

    Bright white glass and a pale pink stain can sit at the same brightness. In
    saturation they cannot.
    """
    glass = np.array([[[245, 245, 245]]], dtype=np.uint8)
    pale_tissue = np.array([[[245, 205, 225]]], dtype=np.uint8)

    assert saturation_channel(glass)[0, 0] == 0
    assert saturation_channel(pale_tissue)[0, 0] > 30

    # Same maximum channel, so any brightness-based split puts them together.
    assert glass.max() == pale_tissue.max()


def test_saturation_of_pure_black_is_zero_not_a_division_by_zero() -> None:
    assert saturation_channel(np.zeros((1, 1, 3), dtype=np.uint8))[0, 0] == 0


def test_class_means_land_either_side_of_the_cut() -> None:
    values = np.concatenate(
        [np.full(100, 20, dtype=np.uint8), np.full(100, 200, dtype=np.uint8)]
    )
    below, above = class_means(histogram_256(values), 110)
    assert below == pytest.approx(20)
    assert above == pytest.approx(200)


def test_class_means_report_none_for_an_empty_side() -> None:
    below, above = class_means(histogram_256(np.full(50, 30, dtype=np.uint8)), 255)
    assert below == pytest.approx(30)
    assert above is None


# --- choosing between the two threshold rules --------------------------------
#
# The pipeline guide specifies Otsu. This step overrides that on one histogram
# shape, so these tests pin down both *when* it overrides and *why* - the whole
# justification is that Otsu's assumption is measurably absent.


def _bimodal(counts_low: int = 400_000, counts_high: int = 120_000) -> np.ndarray:
    """Two well-separated humps, each with a real spread - Otsu's home ground."""
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [
            rng.normal(30, 8, counts_low).clip(0, 255),
            rng.normal(160, 20, counts_high).clip(0, 255),
        ]
    ).astype(np.uint8)
    return histogram_256(values)


def _spike_and_tail() -> np.ndarray:
    """One delta plus a monotone tail - a weakly counterstained IHC slide."""
    counts = np.zeros(256)
    counts[0] = 13_000_000
    counts[1:] = (np.arange(1, 256, dtype=float) ** -1.6) * 300_000
    return counts


def test_modal_share_measures_how_spiked_the_histogram_is() -> None:
    level, share = modal_share(_spike_and_tail())
    assert level == 0
    assert share > 0.8

    _, spread = modal_share(_bimodal())
    assert spread < 0.1


def test_otsu_is_used_when_the_histogram_really_has_two_humps() -> None:
    choice = choose_threshold(_bimodal(), spike_share=0.35)

    assert choice.bimodal
    assert choice.rule == "otsu"
    assert choice.value == choice.otsu
    # The valley between N(30,8) and N(160,20) sits well inside this range.
    assert 60 < choice.value < 130


def test_the_triangle_rule_takes_over_on_a_spike_and_a_tail() -> None:
    """The override, and the reason for it: Otsu runs away into the tail.

    Otsu's criterion is a ratio of variances and a spike has none, so with most
    of the mass on one level the criterion is maximised far out in the tail
    rather than at the shoulder where the tail begins.
    """
    choice = choose_threshold(_spike_and_tail(), spike_share=0.35)

    assert not choice.bimodal
    assert choice.rule == "triangle"
    assert choice.value == choice.triangle
    assert choice.triangle < choice.otsu


def test_both_rules_are_always_reported_whichever_one_won() -> None:
    """A threshold never appears without its alternative — on either shape."""
    for histogram in (_bimodal(), _spike_and_tail()):
        choice = choose_threshold(histogram, spike_share=0.35)
        assert 0 <= choice.otsu <= 255
        assert 0 <= choice.triangle <= 255
        assert choice.value in {choice.otsu, choice.triangle}


def test_the_spike_cutoff_is_what_decides_and_nothing_else() -> None:
    """Same histogram, cutoff either side of its modal share, opposite rules.

    A *lower* cutoff is a stricter test of Otsu's assumption, so it is the low
    side that hands over to the triangle rule.
    """
    histogram = _spike_and_tail()
    _, share = modal_share(histogram)

    assert choose_threshold(histogram, spike_share=share - 0.01).rule == "triangle"
    assert choose_threshold(histogram, spike_share=share + 0.01).rule == "otsu"


def test_the_triangle_chord_spans_the_peak_to_the_last_occupied_level() -> None:
    """The report carries the chord so the panel can draw the construction."""
    choice = choose_threshold(_spike_and_tail(), spike_share=0.35)
    assert choice.triangle_from == 0
    assert choice.triangle_to == 255
    assert choice.triangle_from <= choice.triangle <= choice.triangle_to


def test_neither_rule_falls_over_on_a_degenerate_histogram() -> None:
    single = np.zeros(256)
    single[7] = 5000

    assert triangle_from_histogram(single) == 7
    assert triangle_from_histogram(np.zeros(256)) == 0
    assert choose_threshold(np.zeros(256), spike_share=0.35).value == 0


# --- physical conversions ----------------------------------------------------


def test_a_physical_radius_is_the_same_distance_at_any_resolution() -> None:
    """The whole reason the settings are in microns rather than pixels."""
    assert _radius_px(30.0, 2.0) == 15
    assert _radius_px(30.0, 6.0) == 5
    # Half the resolution, half the pixels, same 30 um of tissue.
    assert _radius_px(30.0, 2.0) * 2.0 == pytest.approx(_radius_px(30.0, 6.0) * 6.0, rel=0.1)


def test_a_physical_radius_never_rounds_away_to_nothing() -> None:
    """At 20 um/px a 12 um opening is under a pixel; it must still open by one."""
    assert _radius_px(12.0, 20.0) == 1


def test_an_area_cutoff_converts_through_mpp_both_ways() -> None:
    # 0.02 mm^2 = 20,000 um^2. At 2 um/px one pixel is 4 um^2, so 5,000 px.
    assert _area_px(0.02, 2.0) == 5000
    assert _area_px(0.02, 4.0) == 1250


# --- hole filling ------------------------------------------------------------


def _ring(size: int = 40, hole: int = 10) -> np.ndarray:
    """A solid block with one enclosed square hole in the middle."""
    mask = np.zeros((size, size), dtype=bool)
    mask[5:-5, 5:-5] = True
    centre = size // 2
    half = hole // 2
    mask[centre - half : centre + half, centre - half : centre + half] = False
    return mask


def test_a_small_enclosed_void_is_filled_because_fat_is_tissue() -> None:
    mask = _ring(hole=10)
    filled, area = _fill_small_holes(mask, max_px=200)

    assert area == 100
    assert filled.all() == False  # noqa: E712 - the outer border is still glass
    assert filled[20, 20]
    assert int(np.count_nonzero(filled)) == int(np.count_nonzero(mask)) + 100


def test_a_large_void_is_left_alone_because_it_is_a_real_gap() -> None:
    mask = _ring(hole=20)
    filled, area = _fill_small_holes(mask, max_px=100)

    assert area == 0
    assert np.array_equal(filled, mask)


def test_the_glass_around_the_specimen_is_never_filled() -> None:
    """It is not an enclosed void, it is the outside world, and it touches the border."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[15:25, 15:25] = True

    filled, area = _fill_small_holes(mask, max_px=40 * 40)

    assert area == 0
    assert np.array_equal(filled, mask)


# --- the mask itself ---------------------------------------------------------


def _synthetic_slide(size: int = 200) -> np.ndarray:
    """Glass with one stained block on it, and one saturated speck of dust."""
    rgb = np.full((size, size, 3), 248, dtype=np.uint8)
    rgb[40:160, 40:160] = (150, 90, 160)  # stained tissue
    rgb[10:13, 10:13] = (200, 20, 20)  # a speck, saturated but tiny
    return rgb


def _build(rgb: np.ndarray, *, mpp: float = 2.0, threshold: int | None = None, **kwargs):
    saturation = saturation_channel(rgb)
    defaults = {
        "close_um": 8.0,
        "open_um": 8.0,
        "min_component_mm2": 0.001,
        "fill_hole_max_mm2": 0.5,
        # A synthetic slide is flat colour, so nearly every pixel lands on one
        # of three levels and the spike test would always fire. Pinned above 1.0
        # so these tests exercise Otsu; the rule choice is tested on its own
        # above, against histograms shaped like real ones.
        "spike_share": 1.01,
    }
    return build_mask(
        saturation=saturation,
        considered=kwargs.pop("considered", np.ones(saturation.shape, dtype=bool)),
        mpp=mpp,
        threshold=threshold,
        **{**defaults, **kwargs},
    )


def test_the_mask_finds_the_tissue_and_drops_the_dust() -> None:
    result = _build(_synthetic_slide())

    assert result.mask[100, 100]  # middle of the block
    assert not result.mask[180, 180]  # glass
    assert not result.mask[11, 11]  # the speck, opened away
    assert result.components is not None
    assert result.components.kept == 1


def test_every_cleanup_move_is_recorded_in_order() -> None:
    result = _build(_synthetic_slide())
    assert [stage.key for stage in result.stages] == [
        "threshold",
        "closing",
        "opening",
        "components",
        "fill",
    ]


def test_closing_never_removes_area_and_opening_never_adds_it() -> None:
    """The two pull in opposite directions, which is why both are reported."""
    stages = {stage.key: stage.pixels for stage in _build(_synthetic_slide()).stages}
    assert stages["closing"] >= stages["threshold"]
    assert stages["opening"] <= stages["closing"]


def test_both_automatic_rules_are_reported_even_on_a_manual_run() -> None:
    """The screen must never be able to show a hand-picked cut as an automatic one."""
    result = _build(_synthetic_slide(), threshold=200)

    assert result.choice.value == 200
    assert result.choice.rule == "manual"
    assert result.choice.otsu != 200
    assert 0 <= result.choice.triangle <= 255


def test_a_manual_threshold_is_clamped_into_range() -> None:
    assert _build(_synthetic_slide(), threshold=999).choice.value == 255
    assert _build(_synthetic_slide(), threshold=-5).choice.value == 0


def test_the_histogram_ignores_pixels_quality_control_rejected() -> None:
    """The reason step 2 runs first, as arithmetic rather than as prose.

    A saturated pen mark left in the histogram does not merely add a pen-shaped
    blob to the mask - it pulls Otsu's cut. Subtracting it before counting is
    what stops that, and it is measurable: the two histograms differ by exactly
    the marked pixels.
    """
    rgb = _synthetic_slide()
    rgb[170:190, 20:180] = (60, 20, 200)  # a pen line across the glass

    considered = np.ones(rgb.shape[:2], dtype=bool)
    considered[170:190, 20:180] = False

    ungated = _build(rgb)
    gated = _build(rgb, considered=considered)

    assert ungated.histogram.sum() - gated.histogram.sum() == 20 * 160
    assert gated.artefact_pixels == 20 * 160
    assert ungated.artefact_pixels == 0

    # And the pen line ends up outside the mask only when it was subtracted.
    assert ungated.mask[180, 100]
    assert not gated.mask[180, 100]


def test_a_blank_slide_produces_an_empty_mask_rather_than_an_error() -> None:
    """Otsu still has to return something when there is nothing to split."""
    result = _build(np.full((100, 100, 3), 250, dtype=np.uint8))

    assert result.tissue_pixels == 0
    assert result.components is not None
    assert result.components.found == 0


def test_component_area_cutoff_is_expressed_in_physical_units() -> None:
    """The same cutoff drops the tissue block once it is stated large enough."""
    rgb = _synthetic_slide()
    # The block is 120 x 120 px = 14,400 px; at 2 um/px that is 0.0576 mm^2.
    kept = _build(rgb, min_component_mm2=0.01)
    dropped = _build(rgb, min_component_mm2=0.5)

    assert kept.tissue_pixels > 0
    assert dropped.tissue_pixels == 0


# --- artefact resampling -----------------------------------------------------


def test_artefacts_resample_by_nearest_neighbour_not_by_averaging() -> None:
    """The average of 'pen' and 'clean tissue' is not a class."""
    # Class ids: 1 = tissue, 4 = pen. Half pen, half tissue.
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:, :2] = 4
    mask[:, 2:] = 1

    footprint = artefact_footprint(mask, artefact_ids=(4,), shape=(8, 8))

    assert footprint.shape == (8, 8)
    assert footprint[:, :4].all()
    assert not footprint[:, 4:].any()


# --- the panels --------------------------------------------------------------


def test_every_panel_renders_a_png() -> None:
    rgb = _synthetic_slide()
    result = _build(rgb)

    for payload in (
        overlay.thumbnail_png(rgb),
        overlay.saturation_png(result.saturation),
        overlay.mask_png(result.mask),
        overlay.overlay_png(rgb, result.mask),
    ):
        assert payload.startswith(b"\x89PNG")


def test_the_mask_panel_is_one_bit_per_pixel() -> None:
    """It is the artefact a user downloads and loads beside the slide."""
    from io import BytesIO

    from PIL import Image

    result = _build(_synthetic_slide())
    with Image.open(BytesIO(overlay.mask_png(result.mask))) as image:
        assert image.mode == "1"
        assert image.size == (result.mask.shape[1], result.mask.shape[0])


# --- endpoints ---------------------------------------------------------------


def test_an_unknown_slide_is_reported_missing(client: TestClient) -> None:
    assert client.get("/api/v1/tissue/not-a-real-upload").status_code == 404


def test_an_out_of_range_threshold_is_rejected(client: TestClient) -> None:
    """Validated at the boundary rather than silently clamped by the API."""
    assert client.get("/api/v1/tissue/anything?threshold=300").status_code == 422
    assert client.get("/api/v1/tissue/anything?threshold=-1").status_code == 422


def test_an_unknown_panel_name_is_refused_as_a_bad_request(client: TestClient) -> None:
    """A typo is not a conflict, so it is caught at the boundary, not in the service."""
    assert client.get("/api/v1/tissue/anything/panels/nonsense.png").status_code == 422
    assert client.get("/api/v1/tissue/no-such-slide/panels/mask.png").status_code == 404
