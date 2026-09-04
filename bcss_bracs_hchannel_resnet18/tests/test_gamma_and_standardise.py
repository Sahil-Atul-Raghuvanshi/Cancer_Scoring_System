"""The shape term and the tile standardiser - the two things gate G0b forced.

Gate G0b measured the H channel of one of our H&E slides against the IHC section of the
same case and found the density ratio running from 0.06 at p75 to 2.06 at p99: a 37x
spread, where a pure stain-strength difference would be a single constant. Two
explanations were tested and rejected - an H-E rather than H-DAB basis for the H&E slide
(18.8x, no better), and one I0 per slide from glass rather than per tile (36.7x, worse).

What remained is that an IHC counterstain is sparse and punchy where H&E haematoxylin is
broad and mid-toned, and the important consequence is the subject of the first test here:
**no monotone correction can close a gap like that**, so widening `alpha` or adding any
per-slide normalisation was never going to work. `gamma` can, because it is applied to
the value after it is scaled into [0, 1], where a power genuinely reshapes.

Measured coverage of the IHC distribution by the H&E training envelope, 120 tiles each,
percentiles p50-p99:

    alpha only, gamma = 1        1 / 5      <- what the plan had
    alpha + gamma                4 / 5
    alpha + gamma + standardise  5 / 5      <- what it has now
"""

from __future__ import annotations

import numpy as np
import pytest

import hchannel


def denorm(x: np.ndarray) -> np.ndarray:
    """Channel 0 of a model input, back to the [0, 1] the network effectively sees."""
    return x[0] * hchannel.IMAGENET_STD[0] + hchannel.IMAGENET_MEAN[0]


@pytest.fixture
def broad() -> np.ndarray:
    """A broad, mid-toned H channel - what an H&E tile looks like."""
    rng = np.random.default_rng(0)
    return np.clip(rng.gamma(shape=2.0, scale=0.09, size=(64, 64)), 0, 1.4).astype(np.float32)


# --- the argument that rules out every monotone fix ---------------------------


def test_no_monotone_rescale_can_change_a_quantile_ratio(broad):
    """Why `alpha`, and any per-slide normalisation, cannot close G0b's gap.

    Scaling by a positive constant multiplies every quantile by that constant, so the
    RATIO between two quantiles is unchanged. G0b's failure is exactly a ratio
    disagreement, which is why the fix had to be something other than a rescale. This
    test is the plan's reasoning, executed.
    """
    ratio = lambda v: float(np.percentile(v, 75) / np.percentile(v, 99))

    base = ratio(broad)
    for alpha in (0.5, 0.8, 1.25, 2.0):
        assert ratio(broad * alpha) == pytest.approx(base, rel=1e-5)

    # And per-tile standardisation is a rescale too, so it cannot either.
    ref = float(np.percentile(broad, 99.0))
    assert ratio(broad / ref) == pytest.approx(base, rel=1e-5)


def test_gamma_does_change_the_quantile_ratio(broad):
    """`gamma` reshapes, which is the whole reason it exists."""
    ratio = lambda x: float(np.percentile(x, 75) / np.percentile(x, 99))

    flat = ratio(denorm(hchannel.to_model_input(broad, gamma=1.0)))
    sparse = ratio(denorm(hchannel.to_model_input(broad, gamma=3.0)))

    # gamma > 1 pushes mid-tones down relative to the top of the range - the
    # H&E -> IHC direction.
    assert sparse < flat * 0.6


# --- gamma, the details -------------------------------------------------------


def test_gamma_one_is_exactly_the_old_behaviour(broad):
    """The default must be a bit-for-bit no-op, or every cached feature is invalidated."""
    before = hchannel.to_model_input(broad, alpha=0.9, beta=0.01)
    after = hchannel.to_model_input(broad, alpha=0.9, beta=0.01, gamma=1.0)
    assert np.array_equal(before, after)


def test_gamma_is_monotone_and_fixes_both_endpoints(broad):
    """A shape change, not a shift: 0 stays 0 and the top of the range stays there."""
    x = np.array([[0.0, hchannel.OD_CLIP / 2, hchannel.OD_CLIP, 2 * hchannel.OD_CLIP]],
                 dtype=np.float32)
    for gamma in (0.5, 1.0, 2.0, 3.5):
        v = denorm(hchannel.to_model_input(x, gamma=gamma))[0]
        assert v[0] == pytest.approx(0.0, abs=1e-6)
        assert v[2] == pytest.approx(1.0, abs=1e-6)
        assert v[3] == pytest.approx(1.0, abs=1e-6)      # clipped, then powered
        assert np.all(np.diff(v) >= -1e-6)               # still monotone


def test_gamma_applies_after_the_clip(broad):
    """Order matters: on a density it would move the saturation point too.

    Two pixels that both clip must stay equal after gamma. If gamma were applied before
    the clip, the larger would still be larger going in and the clip would flatten them
    afterwards - the same output here, but the mid-range would differ. The direct check
    is that the powered value of the clip point is the clip point.
    """
    x = np.array([[hchannel.OD_CLIP, hchannel.OD_CLIP * 3]], dtype=np.float32)
    v = denorm(hchannel.to_model_input(x, gamma=2.5))[0]
    assert v[0] == pytest.approx(v[1], abs=1e-6)
    assert v[0] == pytest.approx(1.0, abs=1e-6)


# --- standardise --------------------------------------------------------------


def test_standardise_puts_the_tiles_own_p99_at_the_target(broad):
    """Two tiles differing only in stain strength become the same input."""
    faint, strong = broad * 0.5, broad * 1.0

    vf = denorm(hchannel.to_model_input(faint, standardise=True))
    vs = denorm(hchannel.to_model_input(strong, standardise=True))

    assert float(np.percentile(vf, 99)) == pytest.approx(
        hchannel.STANDARDISE_TARGET, abs=0.02)
    assert float(np.percentile(vf, 99)) == pytest.approx(
        float(np.percentile(vs, 99)), abs=0.02)


def test_standardise_leaves_an_unstained_tile_alone():
    """The guard that stops it inventing texture out of sensor noise.

    A tile of bare glass or loose stroma has no stain to normalise against. Dividing by
    its p99 would stretch noise across the whole input range, so below
    STANDARDISE_FLOOR the tile is passed through untouched.
    """
    rng = np.random.default_rng(1)
    blank = (rng.random((64, 64)) * 0.01).astype(np.float32)
    assert float(np.percentile(blank, 99)) < hchannel.STANDARDISE_FLOOR

    off = hchannel.to_model_input(blank, standardise=False)
    on = hchannel.to_model_input(blank, standardise=True)
    assert np.array_equal(off, on)


def test_standardise_off_is_the_default_and_a_no_op(broad):
    before = hchannel.to_model_input(broad)
    after = hchannel.to_model_input(broad, standardise=False)
    assert np.array_equal(before, after)


# --- the contract that stops training and serving drifting apart --------------


def test_descriptor_records_both_serving_values():
    """Whatever inference must reproduce goes in the manifest, as data."""
    d = hchannel.descriptor(tile_px=224, mpp=0.5, invert=False,
                            gamma=1.0, standardise=True)
    assert d["gamma"] == 1.0
    assert d["standardise_tile_p99"] is True
    assert d["standardise_floor"] == hchannel.STANDARDISE_FLOOR
    assert d["standardise_target"] == hchannel.STANDARDISE_TARGET


def test_from_stored_forwards_gamma_and_standardise(broad):
    """The tile store holds raw density, so both are decided at load, not at export."""
    stored = hchannel.quantise(broad)

    direct = hchannel.to_model_input(
        hchannel.dequantise(stored), gamma=2.0, standardise=True)
    viaload = hchannel.from_stored(stored, gamma=2.0, standardise=True)

    assert np.array_equal(direct, viaload)


def test_jitter_sample_returns_three_values_and_respects_its_ranges():
    """The Dataset unpacks three; a two-tuple would be a silent TypeError in a worker."""
    import datasets

    rng = np.random.default_rng(0)
    j = datasets.Jitter()
    for _ in range(200):
        alpha, beta, gamma = j.sample(rng)
        assert j.alpha[0] <= alpha <= j.alpha[1]
        assert j.beta[0] <= beta <= j.beta[1]
        assert j.gamma[0] <= gamma <= j.gamma[1]

    # NO_JITTER must be the exact identity, or `augment=False` is not inference's path.
    alpha, beta, gamma = datasets.NO_JITTER.sample(rng)
    assert (alpha, beta, gamma) == (1.0, 0.0, 1.0)


def test_jitter_gamma_is_log_uniform_so_both_directions_are_equally_likely():
    """gamma is an exponent: 2.0 and 0.5 are the same change in opposite directions."""
    import datasets

    rng = np.random.default_rng(0)
    j = datasets.Jitter(gamma=(0.5, 2.0))
    draws = np.array([j.sample(rng)[2] for _ in range(4000)])

    # Geometric midpoint of (0.5, 2.0) is 1.0, so half the mass should sit either side.
    assert float((draws < 1.0).mean()) == pytest.approx(0.5, abs=0.03)
