"""Step 4 tests.

Split the way steps 2 and 3's are: the maths needs nothing but numpy and scipy,
so it is the bulk of this file and runs everywhere. The endpoints are tested for
the states that need no slide on disk.

Every assertion is about a *property* rather than a value - "the exclusions only
ever remove", "a synthetic vignette is recovered", "a flat field is not invented"
- because the numbers depend on the slide and the properties are what the
pipeline relies on.

The one place this file does assert a value is `test_a_synthetic_vignette_is_
recovered`, and deliberately: a fitted surface is the strongest claim this step
makes, so the test builds a slide whose illumination field is known by
construction and checks the fit finds it.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.common.imaging import channel_percentile, optical_density
from app.pipeline.step04_white_calibration import calibration as calib
from app.pipeline.step04_white_calibration import overlay
from app.pipeline.step04_white_calibration.calibration import (
    CalibrationError,
    calibrate,
    choose_field,
    detect_fills,
    evaluate_surface,
    fit_surface,
    measure_noise_floor,
    measure_white_point,
    od_difference,
    sample_patches,
    select_glass,
)

# --- the shared primitives ---------------------------------------------------


def test_a_channel_percentile_is_per_channel_and_not_over_luminance() -> None:
    """The whole reason I0 is three numbers: a scanner's illuminant is not neutral."""
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 240
    rgb[..., 1] = 250
    rgb[..., 2] = 230

    assert channel_percentile(rgb, np.ones((10, 10), dtype=bool), 95.0) == (240.0, 250.0, 230.0)


def test_a_channel_percentile_of_nothing_is_zero_rather_than_an_error() -> None:
    """A caller with no glass has a bigger problem, and reports it itself."""
    rgb = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert channel_percentile(rgb, np.zeros((4, 4), dtype=bool), 95.0) == (0.0, 0.0, 0.0)


def test_optical_density_is_zero_at_the_white_point_and_grows_as_light_falls() -> None:
    """Beer-Lambert, and the one fact that makes step 5 possible."""
    white = (200.0, 200.0, 200.0)

    def density(level: float) -> float:
        return float(optical_density(np.full((1, 1, 3), level, dtype=np.float32), white)[0, 0, 0])

    assert density(200.0) == pytest.approx(0.0)
    # A tenth of the light through is exactly 1 OD, by the definition of log10.
    assert density(20.0) == pytest.approx(1.0)
    assert density(2.0) == pytest.approx(2.0)


def test_optical_density_of_black_is_finite_because_the_floor_clips_it() -> None:
    """Without the floor this is an infinity, and infinities propagate."""
    density = optical_density(np.zeros((1, 1, 3), dtype=np.float32), (255.0, 255.0, 255.0))
    assert np.isfinite(density).all()


def test_optical_density_takes_a_field_as_readily_as_a_triple() -> None:
    """One accessor for both, so no caller branches on which one was justified."""
    image = np.full((4, 4, 3), 100, dtype=np.float32)
    field = np.full((4, 4, 3), 200, dtype=np.float32)

    flat = optical_density(image, (200.0, 200.0, 200.0))
    varying = optical_density(image, field)

    assert np.allclose(flat, varying)


def test_optical_density_is_additive_which_is_why_deconvolution_works() -> None:
    """Two stains stacked multiply their transmissions; their densities add.

    This is the single mathematical fact the pipeline guide names as the reason
    optical density is not just a colour trick, and it is worth pinning in a
    test: if this property ever failed, step 6's linear un-mixing would be
    invalid and nothing downstream would notice.
    """
    white = 255.0
    first, second = 0.6, 0.5  # transmission fractions

    stacked = optical_density(np.array([[[white * first * second]]], dtype=np.float32), (white,))
    separate = optical_density(
        np.array([[[white * first]]], dtype=np.float32), (white,)
    ) + optical_density(np.array([[[white * second]]], dtype=np.float32), (white,))

    assert stacked[0, 0, 0] == pytest.approx(separate[0, 0, 0], abs=1e-5)


# --- deciding what counts as glass -------------------------------------------


def _glass_slide(size: int = 400, white: tuple[int, int, int] = (238, 244, 230)) -> np.ndarray:
    """A slide of uniform glass with a darker tissue block on it, plus noise.

    The noise is not decoration. Real glass carries sensor noise, and the fill
    exclusion tells a digital constant from glass by exactly that: a constant has
    no shoulder on its histogram peak and noisy glass does. A noiseless synthetic
    "glass" would be excluded as a fill, correctly.
    """
    rng = np.random.default_rng(0)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    for channel, level in enumerate(white):
        rgb[..., channel] = np.clip(rng.normal(level, 2.0, (size, size)), 0, 255)

    centre = size // 2
    rgb[centre - 60 : centre + 60, centre - 60 : centre + 60] = (110, 70, 130)
    return rgb


def _tissue_block(size: int = 400, half: int = 60) -> np.ndarray:
    """One square of tissue in the middle of the scan."""
    tissue = np.zeros((size, size), dtype=bool)
    centre = size // 2
    tissue[centre - half : centre + half, centre - half : centre + half] = True
    return tissue


def _select(tissue: np.ndarray, considered: np.ndarray | None = None, **kwargs):
    """Select glass over a plain noisy-white image unless the caller supplies one.

    The default image matters: the fill exclusion tests the *pixels*, so a helper
    that fed it a constant array would trip that exclusion on every other test in
    this file. Noisy white is what real glass looks like, which is exactly what
    the fill test is designed not to fire on.
    """
    size = tissue.shape[0]
    rgb = kwargs.pop("rgb", None)
    if rgb is None:
        rgb = _glass_slide(size=size) if size >= 40 else np.full((size, size, 3), 240, np.uint8)

    defaults = {
        "mpp": 2.0,
        "border_um": 40.0,
        "clearance_um": 20.0,
        "fill_min_share": 0.05,
        "fill_max_shoulder": 0.05,
    }
    return select_glass(
        rgb=rgb, tissue=tissue, considered=considered, **{**defaults, **kwargs}
    )


def test_glass_is_the_tissue_mask_inverted_and_then_narrowed_four_more_times() -> None:
    result = _select(_tissue_block())
    assert [step.key for step in result.steps] == [
        "complement",
        "border",
        "artefacts",
        "fill",
        "clearance",
    ]


def test_every_exclusion_only_ever_removes_pixels() -> None:
    """The property the whole ladder rests on: this is a narrowing, not a filter.

    If any move could *add* a pixel, the ladder in the report would stop being
    readable as a sequence of exclusions - and a pixel added after the tissue
    complement would be a pixel inside the tissue.
    """
    result = _select(_tissue_block(), considered=np.ones((400, 400), dtype=bool))
    counts = [step.pixels for step in result.steps]
    assert counts == sorted(counts, reverse=True)


def test_the_border_exclusion_removes_exactly_the_outer_frame() -> None:
    """40 um at 2 um/px is 20 px, so a 400 px scan keeps a 360 px square."""
    result = _select(np.zeros((400, 400), dtype=bool), clearance_um=0.1)
    border = next(step for step in result.steps if step.key == "border")
    assert border.pixels == 360 * 360


def test_a_scan_too_narrow_for_the_border_keeps_it_rather_than_excluding_everything() -> None:
    """A frame wider than the scan would leave nothing to sample at all."""
    tiny = np.zeros((10, 10), dtype=bool)
    result = _select(tiny, border_um=500.0, clearance_um=0.1)

    border = next(step for step in result.steps if step.key == "border")
    assert border.pixels == 100
    assert "Skipped" in border.what


def test_artefacts_are_excluded_from_the_glass_because_pen_ink_is_not_glass() -> None:
    """The exclusion the guide's version misses, and it biases I0 downward."""
    tissue = _tissue_block()
    considered = np.ones(tissue.shape, dtype=bool)
    considered[20:60, 20:380] = False  # a pen line across the glass

    with_qc = _select(tissue, considered=considered)
    without = _select(tissue, considered=None)

    assert with_qc.pixels < without.pixels
    assert not with_qc.glass[40, 200]
    assert without.glass[40, 200]


def test_the_halo_around_the_tissue_is_dropped_even_though_it_is_not_tissue() -> None:
    """The pixels just outside a section are mounting medium and mask error, not glass."""
    tissue = _tissue_block(size=400, half=60)
    result = _select(tissue, clearance_um=40.0)  # 20 px at 2 um/px

    # 5 px outside the tissue edge: not tissue, and still not glass.
    assert not tissue[139, 200]
    assert not result.glass[139, 200]
    # 40 px outside it, well clear of the clearance ring.
    assert result.glass[100, 200]


def test_the_clearance_ring_is_the_same_distance_at_any_resolution() -> None:
    """The whole reason the setting is in microns: 40 um of glass, not 40 pixels.

    Measured as the gap between the tissue edge and the nearest sampled pixel,
    converted back to microns. At 1 um/px that gap is forty pixels wide and at
    4 um/px it is ten, and both are the same forty microns of slide.
    """
    tissue = _tissue_block()  # spans rows 140..259 at any resolution

    def gap_um(mpp: float) -> float:
        glass = _select(tissue, mpp=mpp, clearance_um=40.0, border_um=4.0).glass
        column = glass[:140, 200]
        first = int(np.nonzero(column)[0].max())  # nearest sampled row above the tissue
        return (140 - first) * mpp

    assert gap_um(1.0) == pytest.approx(40.0, abs=mpp_tolerance_um(1.0))
    assert gap_um(4.0) == pytest.approx(40.0, abs=mpp_tolerance_um(4.0))


def mpp_tolerance_um(mpp: float) -> float:
    """One pixel of slack, in microns - a radius has to land on the pixel grid."""
    return mpp



# --- the scanner's background fill -------------------------------------------
#
# The exclusion the demo's own slides forced. Two thirds of an Aperio scan is a
# constant the scanner wrote into the part of the canvas it never imaged, and it
# is darker than the real glass - so left in the sample it drags I0 down and
# reports two tenths of an OD of apparent stain over a region that does not
# exist. These tests pin down both that it is caught and that real glass is not.


def _pure_glass(size: int = 400, white: float = 240.0, noise: float = 2.0) -> np.ndarray:
    """Noisy glass and nothing else - no synthetic tissue block.

    Used by the fill tests. `_glass_slide`'s tissue block is a single flat colour,
    which *is* a digital constant, and the detector correctly says so - on a real
    slide it never reaches the detector because step 3's mask has already removed
    the tissue, but a test that passes an empty mask would put it back.
    """
    values = np.random.default_rng(0).normal(white, noise, (size, size, 3))
    return np.clip(values, 0, 255).astype(np.uint8)


def _padded(size: int = 400, fill: int = 146, white: int = 193) -> np.ndarray:
    """A scan whose outer half is a flat digital fill and whose inner half is glass.

    Modelled on the demo's slides, whose measured values these are: the fill is a
    single exact triple with no shoulder at all, and the glass around 193 with
    real sensor noise.
    """
    rng = np.random.default_rng(1)
    rgb = np.full((size, size, 3), fill, dtype=np.uint8)
    inner = size // 4
    patch = rng.normal(white, 0.7, (size - 2 * inner, size - 2 * inner, 3))
    rgb[inner:-inner, inner:-inner] = np.clip(patch, 0, 255).astype(np.uint8)
    return rgb


def test_a_digital_background_fill_is_found_and_excluded() -> None:
    rgb = _padded()
    result = _select(np.zeros((400, 400), dtype=bool), rgb=rgb)

    assert len(result.fills) == 1
    assert result.fills[0].rgb == (146, 146, 146)
    assert result.fills[0].shoulder < 0.05
    assert not result.glass[10, 10]  # in the fill
    assert result.glass[200, 200]  # in the real glass


def test_the_fill_is_excluded_before_i0_is_measured_and_it_moves_i0() -> None:
    """The point of the exclusion: the fill is darker than glass, so it drags I0 down."""
    rgb = _padded()
    tissue = np.zeros((400, 400), dtype=bool)

    excluded = measure_white_point(
        rgb, _select(tissue, rgb=rgb).glass, percentile=95.0
    )
    kept = measure_white_point(
        rgb,
        _select(tissue, rgb=rgb, fill_min_share=1.01).glass,  # never fires
        percentile=95.0,
    )

    assert excluded.rgb[0] == pytest.approx(194, abs=3)
    assert kept.rgb[0] < excluded.rgb[0]


def test_excluding_the_fill_collapses_the_noise_floor() -> None:
    """Stated in the units step 5 produces, which is the only way it means anything.

    Left in, the fill measures as apparent stain over a region that was never
    imaged - `-log10(146/193)`, about 0.12 OD, which is a third of a real DAB
    signal. Removed, the floor is what empty glass should be: near zero.
    """
    rgb = _padded()
    tissue = np.zeros((400, 400), dtype=bool)

    clean = _select(tissue, rgb=rgb)
    dirty = _select(tissue, rgb=rgb, fill_min_share=1.01)

    clean_floor = measure_noise_floor(
        rgb, clean.glass, measure_white_point(rgb, clean.glass, percentile=95.0).rgb, floor=1.0
    )
    dirty_floor = measure_noise_floor(
        rgb, dirty.glass, measure_white_point(rgb, dirty.glass, percentile=95.0).rgb, floor=1.0
    )

    assert clean_floor.worst < 0.02
    assert dirty_floor.worst > 0.08
    assert dirty_floor.worst == pytest.approx(np.log10(193 / 146), abs=0.02)


def test_real_glass_is_never_mistaken_for_a_fill_however_much_of_the_frame_it_is() -> None:
    """The test that keeps this exclusion safe rather than merely aggressive.

    Glass fills the entire candidate set here, so anything that judged a colour by
    how much of the frame it covers would exclude all of it and leave nothing to
    calibrate from. What saves it is that sensor noise puts colours either side of
    every real measurement, and the fill test requires there to be none.
    """
    rgb = _pure_glass()
    result = _select(np.zeros((400, 400), dtype=bool), rgb=rgb)

    assert result.fills == []
    assert result.pixels > 0


def test_the_shoulder_is_what_decides_and_nothing_else() -> None:
    """Same image, cutoff either side of its measured shoulder, opposite outcomes.

    Low-noise glass, so the share test cannot be what saves it: quantised tightly
    enough, one exact colour holds a tenth of the frame, well over the share
    threshold. The only thing standing between it and being excluded as a fill is
    that colours sit either side of it - which is the whole point.
    """
    glass = np.clip(
        np.random.default_rng(0).normal(240, 0.5, (400, 400, 3)), 0, 255
    ).astype(np.uint8)
    candidate = np.ones((400, 400), dtype=bool)

    lenient = detect_fills(glass, candidate, min_share=0.05, max_shoulder=20.0)
    strict = detect_fills(glass, candidate, min_share=0.05, max_shoulder=0.05)

    # The share test alone would have excluded it: the modal colour clears the bar.
    assert lenient and lenient[0].share >= 0.05
    # And its shoulder is orders of magnitude above a digital constant's.
    assert lenient[0].shoulder > 1.0
    assert strict == []


def test_more_than_one_fill_is_found_because_scans_carry_more_than_one() -> None:
    """A black reader-padding border outside a grey scanner fill is the common pair.

    Also the case that forced `detect_fills` to test the several commonest colours
    per pass rather than only the mode: once the grey fill is gone the real glass
    outnumbers the black border, so a detector that stopped at the mode would have
    left the border in the sample.
    """
    rgb = _padded()
    rgb[:20, :] = 0
    rgb[-20:, :] = 0

    result = _select(np.zeros((400, 400), dtype=bool), rgb=rgb, border_um=2.0)
    found = {fill.rgb for fill in result.fills}

    assert (146, 146, 146) in found
    assert (0, 0, 0) in found
    assert not result.glass[5, 200]  # the black border
    assert not result.glass[30, 200]  # the grey fill
    assert result.glass[200, 200]  # the real glass


def test_a_fill_only_counts_when_all_three_channels_are_constant() -> None:
    """That is what a constant is; requiring it keeps one quantised channel safe.

    A slide compressed hard enough to flatten one channel still has noise in the
    other two, and it is still a measurement. Judging channels independently would
    throw it away.
    """
    rgb = _pure_glass()
    rgb[..., 1] = 200  # green flattened, red and blue still noisy

    result = _select(np.zeros((400, 400), dtype=bool), rgb=rgb)
    assert result.fills == []


# --- the flat white point ----------------------------------------------------


def test_the_white_point_lands_on_the_glass_and_not_on_the_tissue() -> None:
    rgb = _glass_slide(white=(238, 244, 230))
    glass = _select(_tissue_block()).glass

    white = measure_white_point(rgb, glass, percentile=95.0)

    assert white.rgb[0] == pytest.approx(238, abs=5)
    assert white.rgb[1] == pytest.approx(244, abs=5)
    assert white.rgb[2] == pytest.approx(230, abs=5)


def test_the_white_point_is_three_numbers_because_the_illuminant_is_not_neutral() -> None:
    """Collapsing them would bake the scanner's colour cast into every density."""
    white = measure_white_point(
        _glass_slide(white=(250, 240, 210)), _select(_tissue_block()).glass, percentile=95.0
    )
    assert white.cast > 0.1
    assert len(set(round(value) for value in white.rgb)) == 3


def test_a_percentile_ignores_a_hot_pixel_where_the_maximum_would_not() -> None:
    """Why the 95th and not the max: I0 sits in a denominator.

    One specular glint off the coverslip would set the maximum, and an
    overestimated I0 darkens every optical density on the slide.
    """
    rgb = _glass_slide(white=(200, 200, 200))
    # Well inside the sampled glass. Placed there deliberately: a glint in the
    # outermost frame would already have been excluded, and the point of this
    # test is the glint the exclusions cannot catch.
    rgb[200, 50] = (255, 255, 255)
    rgb[201, 51] = (255, 255, 255)

    glass = _select(_tissue_block()).glass
    assert glass[200, 50]

    white = measure_white_point(rgb, glass, percentile=95.0)

    assert white.rgb[0] < 210
    # The maximum, which is what a naive implementation would use, is the glint.
    assert white.ladder[100.0][0] == 255.0


def test_the_percentile_ladder_plateaus_on_clean_glass() -> None:
    """The argument for the 95th: its neighbours agree with it."""
    white = measure_white_point(
        _glass_slide(white=(240, 240, 240)), _select(_tissue_block()).glass, percentile=95.0
    )

    ninety = white.ladder[90.0][0]
    ninety_nine = white.ladder[99.0][0]
    assert abs(ninety_nine - ninety) / ninety < 0.02


def test_a_clipped_channel_is_reported_rather_than_repaired() -> None:
    """The sensor ran out of range before the glass did, and OD is compressed."""
    rgb = np.full((200, 200, 3), 255, dtype=np.uint8)
    rgb[80:120, 80:120] = (100, 100, 100)

    glass = _select(_tissue_block(size=200, half=20)).glass
    white = measure_white_point(rgb, glass, percentile=95.0)

    assert white.saturated
    assert white.clipped[0] > 0.9


def test_clean_glass_below_the_ceiling_is_not_reported_as_clipped() -> None:
    white = measure_white_point(
        _glass_slide(white=(230, 230, 230)), _select(_tissue_block()).glass, percentile=95.0
    )
    assert not white.saturated


def test_no_glass_at_all_is_a_named_failure_and_not_a_division_by_zero() -> None:
    """A slide cropped to its section cannot be calibrated against its own glass."""
    with pytest.raises(CalibrationError, match="no glass left to sample"):
        measure_white_point(
            _glass_slide(), np.zeros((400, 400), dtype=bool), percentile=95.0
        )


# --- the illumination surface ------------------------------------------------


def _vignetted(
    size: int = 400, white: float = 240.0, depth: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Glass under a known quadratic bowl: brightest at the centre, dimmest at the corners.

    `depth` is the fall-off at the corner as a fraction of the centre, so the
    field's true swing is `white * depth` and the fit has a value to be checked
    against rather than merely a shape.
    """
    y, x = np.mgrid[0:size, 0:size]
    nx = 2.0 * x / (size - 1) - 1.0
    ny = 2.0 * y / (size - 1) - 1.0

    field = white * (1.0 - depth * (nx**2 + ny**2) / 2.0)
    rgb = np.repeat(field[..., None], 3, axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8), field


def _patches(rgb: np.ndarray, glass: np.ndarray, **kwargs):
    defaults = {"mpp": 2.0, "patch_um": 100.0, "min_glass_share": 0.35, "percentile": 95.0}
    return sample_patches(rgb, glass, **{**defaults, **kwargs})


def test_a_patch_with_too_little_glass_does_not_vote() -> None:
    """One percentile from a handful of pixels is not a white point."""
    tissue = _tissue_block(size=400, half=120)
    glass = _select(tissue).glass
    patches = _patches(_glass_slide(), glass)

    assert any(patch.used for patch in patches)
    assert any(not patch.used for patch in patches)
    for patch in patches:
        if patch.used:
            assert patch.glass_share >= 0.35
            assert patch.rgb is not None
        else:
            assert patch.rgb is None


def test_a_synthetic_vignette_is_recovered_by_the_fit() -> None:
    """The strongest claim this step makes, checked against a field built by hand.

    Checked against the swing *over the sampled region*, not over the whole
    frame: the border exclusion has already trimmed the outer 10%, so the fit
    never saw the deepest corner of the bowl and cannot be asked to have found it.
    """
    rgb, field = _vignetted(depth=0.15)
    glass = _select(np.zeros((400, 400), dtype=bool)).glass

    surface = fit_surface(_patches(rgb, glass), shape=(400, 400))
    assert surface is not None

    # The fit explains almost all of the patch-to-patch variation...
    assert min(surface.r2) > 0.95
    # ...and its swing matches the bowl it actually got to see.
    sampled_swing = float(field[glass].max() - field[glass].min())
    assert surface.swing[0] == pytest.approx(sampled_swing, rel=0.1)

    # And the *shape* tracks the real field, to within a few levels.
    #
    # Neither the offset nor the residual is an error. Each patch's value is the
    # 95th percentile of the glass *inside that patch*, which sits above the
    # field's value at the patch centre - exactly as the flat I0 sits above the
    # mean of all the glass. So the surface is "the local 95th percentile of
    # glass as a function of position", the spatially-varying counterpart of the
    # flat white point and consistently defined with it. The residual is that
    # same bias varying with the gradient, which is steepest at the corners; it
    # reaches about 4 levels out of 240 here - under 2% - and
    # `test_the_chosen_field_flattens_the_vignette_it_was_fitted_to` measures
    # what it leaves behind after the correction is actually applied.
    #
    # This synthetic field is the *worst* case for that bias, because it carries
    # no sensor noise: with none, a patch's 95th percentile is decided purely by
    # the gradient across the patch. On a real slide the noise dominates the
    # percentile and the bias is smaller.
    recovered = evaluate_surface(surface, (400, 400))[..., 0]
    offset = float(np.mean((recovered - field)[glass]))

    assert np.abs((recovered - field)[glass] - offset).max() < 5.0
    assert 0.0 < offset < 12.0


def test_the_fit_reports_vignetting_as_an_optical_density_error() -> None:
    """The only currency in which "6% of I0" means anything to this pipeline."""
    rgb, _ = _vignetted(depth=0.15)
    glass = _select(np.zeros((400, 400), dtype=bool)).glass
    surface = fit_surface(_patches(rgb, glass), shape=(400, 400))
    assert surface is not None

    # A swing of a fraction f of the level is log10(1/(1-f)) OD, by definition.
    expected = np.log10(1.0 / (1.0 - surface.amplitude))
    assert surface.od_error == pytest.approx(expected, rel=1e-6)


def test_a_flat_field_fits_but_is_not_used_because_its_bend_is_its_own_noise() -> None:
    """The failure mode the SNR test exists to prevent.

    Uniform glass plus sensor noise will still admit a quadratic - least squares
    always returns something. What it must not do is *use* it: a white point that
    curves to follow noise stamps a spurious spatial gradient onto every
    measurement downstream, where nothing later can tell it from biology.
    """
    glass = _select(np.zeros((400, 400), dtype=bool)).glass
    surface = fit_surface(_patches(_glass_slide(white=(240, 240, 240)), glass), shape=(400, 400))

    assert surface is not None  # it fits
    assert surface.snr < 4.0  # and it is noise
    choice = choose_field(
        surface, min_patches=12, required_snr=4.0, leverage_limit=1.0
    )
    assert choice.mode == "flat"


def test_a_real_vignette_clears_the_snr_test_and_is_used() -> None:
    rgb, _ = _vignetted(depth=0.15)
    glass = _select(np.zeros((400, 400), dtype=bool)).glass
    surface = fit_surface(_patches(rgb, glass), shape=(400, 400))
    assert surface is not None

    choice = choose_field(surface, min_patches=12, required_snr=4.0, leverage_limit=1.0)
    assert choice.mode == "surface"
    assert choice.snr >= 4.0


def test_too_few_patches_falls_back_to_flat_however_good_the_fit_looks() -> None:
    """Six coefficients solved from eight points will extrapolate confidently."""
    rgb, _ = _vignetted(depth=0.15)
    glass = _select(np.zeros((400, 400), dtype=bool)).glass
    surface = fit_surface(_patches(rgb, glass), shape=(400, 400))
    assert surface is not None

    choice = choose_field(surface, min_patches=10_000, required_snr=4.0, leverage_limit=1.0)
    assert choice.mode == "flat"
    assert choice.fitted  # reported either way
    assert "usable patches" in choice.reason


def test_no_surface_at_all_is_a_distinct_and_reported_case() -> None:
    """Fewer samples than the quadratic has coefficients: nothing to fit."""
    assert fit_surface([], shape=(100, 100)) is None

    choice = choose_field(None, min_patches=12, required_snr=4.0, leverage_limit=1.0)
    assert choice.mode == "flat"
    assert not choice.fitted


def test_both_answers_are_always_reported_whichever_field_won() -> None:
    """A white point is the most consequential number here, so it never stands alone."""
    for depth in (0.0005, 0.15):
        rgb, _ = _vignetted(depth=depth)
        glass = _select(np.zeros((400, 400), dtype=bool)).glass
        surface = fit_surface(_patches(rgb, glass), shape=(400, 400))
        choice = choose_field(surface, min_patches=12, required_snr=4.0, leverage_limit=1.0)

        assert choice.mode in {"flat", "surface"}
        assert choice.snr_required == 4.0
        assert choice.reason


def test_the_evaluated_field_never_goes_below_one_because_it_is_a_denominator() -> None:
    """A quadratic extrapolates, and a negative I0 would invert every density."""
    steep = calib.Surface(
        coefficients=((0.0, 0.0, 0.0, -900.0, 0.0, -900.0),) * 3,
        r2=(1.0, 1.0, 1.0),
        residual_rms=(0.1, 0.1, 0.1),
        swing=(100.0, 100.0, 100.0),
        mean=(200.0, 200.0, 200.0),
        patches_used=30,
        shape=(64, 64),
        leverage=0.2,
        leverage_max=0.4,
    )
    assert (evaluate_surface(steep, (64, 64)) >= 1.0).all()



# --- is the field supported where it is used? --------------------------------
#
# Glass samples necessarily ring the section, because the middle of the slide is
# the section. Usually the ring encloses the tissue and the field is well pinned
# down across it; when the section runs off the edge of the scan the ring is open
# and the fit is guessing across exactly the part every measurement comes from.
# Leverage is what tells those two apart, and it is measured over the tissue
# rather than over the frame - see `_leverage`.


def test_a_ring_of_glass_around_the_tissue_does_support_the_field() -> None:
    """The common case, and the reason leverage is not measured over the frame.

    The frame's corners are wildly extrapolated by a ring fit and it does not
    matter in the least - they are empty glass, and no density is ever computed
    there. Over the tissue, which the ring encloses, the field is pinned down.
    """
    rgb, _ = _vignetted(depth=0.15)
    tissue = _tissue_block(size=400, half=90)  # enclosed by the glass around it

    surface = fit_surface(
        _patches(rgb, _select(tissue, rgb=rgb).glass),
        shape=(400, 400),
        applies_to=tissue,
    )
    assert surface is not None

    assert surface.leverage < 1.0
    # And the frame really is the pessimistic reading: measured over everything,
    # the same fit looks far worse, because of corners nobody reads.
    over_frame = fit_surface(
        _patches(rgb, _select(tissue, rgb=rgb).glass),
        shape=(400, 400),
        applies_to=np.ones((400, 400), dtype=bool),
    )
    assert over_frame is not None
    assert over_frame.leverage > surface.leverage


def test_a_section_running_off_the_scan_leaves_the_field_unsupported() -> None:
    """The failure the guard exists for: an open ring, and a guess where it counts.

    Tissue filling one whole side of the scan leaves glass only on the other, so a
    quadratic fitted to that glass is extrapolating across the entire section - and
    the section is where every optical density comes from. The flat value is
    uncertain in the same way everywhere, which is the honest form for a guess.
    """
    rgb, _ = _vignetted(depth=0.15)
    tissue = np.zeros((400, 400), dtype=bool)
    tissue[:, 150:] = True  # glass only in the left 150 px

    surface = fit_surface(
        _patches(rgb, _select(tissue, rgb=rgb).glass),
        shape=(400, 400),
        applies_to=tissue,
    )
    assert surface is not None
    assert surface.leverage > 1.0

    choice = choose_field(
        surface, min_patches=12, required_snr=4.0, leverage_limit=1.0
    )
    assert choice.mode == "flat"
    assert "do not enclose the tissue" in choice.reason


def test_the_leverage_cutoff_is_what_decides_and_nothing_else() -> None:
    """Same fit, cutoff either side of its leverage, opposite outcomes."""
    rgb, _ = _vignetted(depth=0.15)
    tissue = np.zeros((400, 400), dtype=bool)
    tissue[:, 150:] = True

    surface = fit_surface(
        _patches(rgb, _select(tissue, rgb=rgb).glass),
        shape=(400, 400),
        applies_to=tissue,
    )
    assert surface is not None

    strict = choose_field(surface, min_patches=12, required_snr=4.0, leverage_limit=1.0)
    lenient = choose_field(
        surface, min_patches=12, required_snr=4.0, leverage_limit=surface.leverage + 1.0
    )

    assert strict.mode == "flat"
    assert lenient.mode == "surface"
    # Reported either way, whichever field won.
    assert strict.leverage == pytest.approx(lenient.leverage)
    assert strict.leverage_limit != lenient.leverage_limit


def test_leverage_over_an_empty_region_is_zero_rather_than_an_error() -> None:
    """A slide with no tissue on it has nowhere the field has to be supported."""
    rgb, _ = _vignetted(depth=0.15)
    glass = _select(np.zeros((400, 400), dtype=bool), rgb=rgb).glass

    surface = fit_surface(
        _patches(rgb, glass), shape=(400, 400), applies_to=np.zeros((400, 400), dtype=bool)
    )
    assert surface is not None
    assert surface.leverage == 0.0
    assert surface.leverage_max == 0.0


# --- the noise floor ---------------------------------------------------------


def test_calibrated_glass_measures_almost_no_stain() -> None:
    """The step's own error bar: what apparent stain the pipeline finds on nothing."""
    rgb = _glass_slide(white=(240, 240, 240))
    glass = _select(_tissue_block()).glass
    white = measure_white_point(rgb, glass, percentile=95.0)

    floor = measure_noise_floor(rgb, glass, white.rgb, floor=1.0)

    assert floor.worst < 0.05
    assert max(floor.median) < 0.02


def test_a_white_point_underestimated_by_tissue_raises_the_noise_floor() -> None:
    """Why the exclusions matter, expressed in the units step 5 produces.

    Sampling a white point from a region that includes tissue pulls I0 down;
    because the glass is then brighter than its own reference, the floor moves -
    which is exactly the symptom a reader should be shown rather than told about.
    """
    rgb = _glass_slide(white=(240, 240, 240))
    glass = _select(_tissue_block()).glass

    honest = measure_white_point(rgb, glass, percentile=95.0)
    too_low = (180.0, 180.0, 180.0)

    assert measure_noise_floor(rgb, glass, honest.rgb, floor=1.0).worst < abs(
        measure_noise_floor(rgb, glass, too_low, floor=1.0).floor[0]
    )


# --- the whole step ----------------------------------------------------------


def _calibrate(rgb: np.ndarray, tissue: np.ndarray, **kwargs):
    defaults = {
        "considered": None,
        "mpp": 2.0,
        "percentile": 95.0,
        "border_um": 40.0,
        "clearance_um": 20.0,
        "fill_min_share": 0.05,
        "fill_max_shoulder": 0.05,
        "patch_um": 100.0,
        "patch_min_glass": 0.35,
        "min_patches": 12,
        "required_snr": 4.0,
        "leverage_limit": 1.0,
        "od_floor": 1.0,
    }
    return calibrate(rgb=rgb, tissue=tissue, **{**defaults, **kwargs})


def test_a_run_on_uniform_glass_chooses_the_flat_white_point() -> None:
    result = _calibrate(_glass_slide(white=(238, 244, 230)), _tissue_block())

    assert result.choice.mode == "flat"
    assert not result.uses_surface
    assert result.field() == result.white.rgb
    assert result.noise.worst < 0.05


def test_a_run_on_a_vignetted_slide_chooses_the_surface() -> None:
    rgb, _ = _vignetted(depth=0.15)
    result = _calibrate(rgb, np.zeros((400, 400), dtype=bool))

    assert result.choice.mode == "surface"
    assert result.uses_surface

    field = result.field()
    assert isinstance(field, np.ndarray)
    assert field.shape == (400, 400, 3)


def test_the_chosen_field_flattens_the_vignette_it_was_fitted_to() -> None:
    """The test that matters, because a fitted surface is a hypothesis until it is applied.

    Divide the slide by the field and the centre-to-corner difference should
    collapse. If it does not, the fit found the wrong bowl.

    Measured over 30 px blocks rather than at single pixels: this is a claim about
    a smooth illumination field, and two individual pixels differ by their own
    sensor noise as well as by the field, which would make the test report the
    noise as a failure to correct.
    """
    rgb, _ = _vignetted(depth=0.15)
    result = _calibrate(rgb, np.zeros((400, 400), dtype=bool))
    assert result.uses_surface

    field = np.asarray(result.field(), dtype=np.float32)
    raw = np.asarray(rgb, dtype=np.float32)[..., 0]
    # Rescaled to the field's own level, so the panel and this test both change
    # only the gradient and not the brightness.
    corrected = raw / field[..., 0] * float(np.mean(field))

    def spread(image: np.ndarray) -> float:
        blocks = [
            float(np.mean(image[row - 15 : row + 15, col - 15 : col + 15]))
            for row, col in ((200, 200), (30, 30), (200, 30), (370, 370))
        ]
        return max(blocks) - min(blocks)

    before, after = spread(raw), spread(corrected)

    assert before > 20.0
    # Most of the vignette goes; what is left is the within-patch percentile bias
    # documented on the recovery test above, and it is a fifth of what it was.
    assert after < before * 0.3


def test_the_noise_floor_is_measured_under_the_field_that_was_actually_chosen() -> None:
    """Quoting a consequence of a calibration that is not the one in force would be a lie."""
    rgb, _ = _vignetted(depth=0.15)
    result = _calibrate(rgb, np.zeros((400, 400), dtype=bool))
    assert result.uses_surface

    under_chosen = measure_noise_floor(rgb, result.glass.glass, result.field(), floor=1.0)
    under_flat = measure_noise_floor(rgb, result.glass.glass, result.white.rgb, floor=1.0)

    assert result.noise.floor == pytest.approx(under_chosen.floor)
    # And the two genuinely differ, so the distinction is not academic.
    assert result.noise.floor != pytest.approx(under_flat.floor)


def test_a_mask_that_does_not_match_the_image_is_refused_rather_than_indexed() -> None:
    """Step 4 decides what glass is by position, so the grids have to be one grid."""
    with pytest.raises(CalibrationError, match="the same one"):
        _calibrate(_glass_slide(size=400), _tissue_block(size=200, half=20))


def test_a_single_channel_image_is_refused() -> None:
    with pytest.raises(CalibrationError, match="RGB"):
        _calibrate(np.zeros((100, 100), dtype=np.uint8), np.zeros((100, 100), dtype=bool))


def test_moving_step_3s_threshold_moves_the_white_point() -> None:
    """Step 3's slider is the control for this step too, and the report says so.

    A mask that claims more of the slide as tissue leaves less glass; a mask that
    claims pale tissue as glass puts tissue into the sample, and tissue is darker
    than glass, so I0 falls.
    """
    rgb = _glass_slide(white=(240, 240, 240))

    tight = _calibrate(rgb, _tissue_block(half=60))
    # A mask that misses the tissue entirely - what an over-high cut produces.
    missed = _calibrate(rgb, np.zeros((400, 400), dtype=bool))

    assert missed.white.rgb[0] <= tight.white.rgb[0]
    assert missed.glass.pixels > tight.glass.pixels


# --- the cross-slide argument ------------------------------------------------


def test_two_white_points_differ_by_a_constant_offset_in_optical_density() -> None:
    """The whole "why per slide" argument, and why a later step cannot absorb it.

    Density is a logarithm of a ratio, so a mismatched I0 does not scale the
    measurements - it *adds* to all of them. On the wrong slide that offset is
    indistinguishable from more stain.
    """
    shift = od_difference((240.0, 240.0, 240.0), (200.0, 200.0, 200.0))
    expected = float(np.log10(240.0 / 200.0))

    assert shift == pytest.approx((expected, expected, expected))
    assert expected > 0.07  # a real offset, not a rounding detail

    # And it really is additive: the same pixel measured under the two white
    # points differs by exactly that offset, whatever the pixel is.
    for level in (40.0, 120.0, 200.0):
        pixel = np.full((1, 1, 3), level, dtype=np.float32)
        difference = optical_density(pixel, (240.0,) * 3) - optical_density(pixel, (200.0,) * 3)
        assert float(difference[0, 0, 0]) == pytest.approx(expected, abs=1e-5)


def test_identical_white_points_cost_nothing() -> None:
    assert od_difference((230.0, 240.0, 220.0), (230.0, 240.0, 220.0)) == pytest.approx(
        (0.0, 0.0, 0.0)
    )


# --- the panels --------------------------------------------------------------


def test_every_panel_renders_a_png() -> None:
    rgb = _glass_slide()
    result = _calibrate(rgb, _tissue_block())

    for payload in (
        overlay.thumbnail_png(rgb),
        overlay.glass_png(rgb, result),
        overlay.field_png(result),
        overlay.corrected_png(rgb, result),
        overlay.swatch_png(result.white.rgb),
    ):
        assert payload.startswith(b"\x89PNG")


def test_the_field_panel_renders_even_when_no_surface_was_fitted() -> None:
    """A reported fact should not look like a server error."""
    result = _calibrate(_glass_slide(), _tissue_block())
    result.surface = None
    assert overlay.field_png(result).startswith(b"\x89PNG")


def test_the_swatch_is_the_white_point_and_nothing_else() -> None:
    """It is the artefact to put next to another slide's, so it must be exact."""
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(overlay.swatch_png((238.4, 244.6, 230.2)))) as image:
        assert image.convert("RGB").getpixel((10, 10)) == (238, 245, 230)


def test_patch_bounds_are_fractions_so_the_browser_needs_no_resolution() -> None:
    result = _calibrate(_glass_slide(), _tissue_block())
    patch = result.patches[0]

    x, y, width, height = overlay.patch_bounds(patch, shape=result.shape)
    for value in (x, y, width, height):
        assert 0.0 <= value <= 1.0


# --- endpoints ---------------------------------------------------------------


def test_an_unknown_slide_is_reported_missing(client: TestClient) -> None:
    assert client.get("/api/v1/calibration/not-a-real-upload").status_code == 404


def test_an_out_of_range_percentile_is_rejected(client: TestClient) -> None:
    """Validated at the boundary rather than silently clamped by the service."""
    assert client.get("/api/v1/calibration/anything?percentile=101").status_code == 422
    assert client.get("/api/v1/calibration/anything?percentile=0").status_code == 422


def test_step_3s_threshold_is_validated_here_too(client: TestClient) -> None:
    assert client.get("/api/v1/calibration/anything?threshold=300").status_code == 422


def test_an_unknown_panel_name_is_refused_as_a_bad_request(client: TestClient) -> None:
    assert client.get("/api/v1/calibration/anything/panels/nonsense.png").status_code == 422
    assert client.get("/api/v1/calibration/no-such-slide/panels/glass.png").status_code == 404


def test_comparing_one_slide_is_refused_because_that_is_the_thing_it_argues_against(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/calibration/compare?uploadIds=only-one")
    assert response.status_code == 409
    assert "at least two" in response.json()["detail"]


def test_the_compare_path_is_not_swallowed_by_the_upload_id_route(
    client: TestClient,
) -> None:
    """Registered first on purpose - otherwise 'compare' is read as a slide id."""
    assert client.get("/api/v1/calibration/compare?uploadIds=a,b").status_code == 404
