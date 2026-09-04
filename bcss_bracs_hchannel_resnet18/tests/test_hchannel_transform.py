"""The input transform: that it is the pipeline's own, and that it is invertible.

Gate G0 in executable form. The failure this guards against is the quietest one in
the plan - a model trained on one definition of "haematoxylin channel" and served
another - and it is invisible in every metric computed inside the training process.
"""

from __future__ import annotations

import numpy as np

import hchannel
from app.common.stains import REFERENCE_BY_NAME, RUIFROK_INVERSE, unmix
from app.common.imaging import optical_density


def synthetic(stain: str, amount: float, size: int = 16) -> np.ndarray:
    """An image of one pure dye at one concentration, built from Beer-Lambert.

    `I = I0 * 10^(-c * v)` for stain direction `v`. Constructed rather than loaded so
    the expected answer is known exactly: deconvolution of this image must return `c`
    for that stain and zero for the other.
    """
    vector = np.asarray(REFERENCE_BY_NAME[stain], dtype=np.float64)
    intensity = 255.0 * np.power(10.0, -amount * vector)
    return np.tile(intensity.astype(np.uint8), (size, size, 1))


def test_uses_the_pipelines_own_deconvolution():
    """`haematoxylin_od` must agree with un-mixing by hand through step 6's basis.

    Not a tautology: it is the assertion that this module adds no arithmetic of its
    own between the pipeline's optical density and the pipeline's stain basis. If
    someone ever "optimises" it with `skimage.color.rgb2hed`, this fails.
    """
    rgb = synthetic("haematoxylin", 0.8)
    white = (255.0, 255.0, 255.0)

    ours = hchannel.haematoxylin_od(rgb, white)

    od = optical_density(rgb, white)
    theirs = unmix(od.reshape(-1, 3), RUIFROK_INVERSE)[:, 0].reshape(od.shape[:2])

    assert np.allclose(ours, theirs, atol=1e-6)


def test_pure_haematoxylin_recovers_its_concentration():
    rgb = synthetic("haematoxylin", 0.8)
    h = hchannel.haematoxylin_od(rgb, (255.0, 255.0, 255.0))
    # 8-bit quantisation of the synthetic image is the only error source here.
    assert abs(float(h.mean()) - 0.8) < 0.02


def test_pure_dab_carries_almost_no_haematoxylin():
    """The brown is what the channel throws away. That is the whole design."""
    rgb = synthetic("dab", 0.8)
    h = hchannel.haematoxylin_od(rgb, (255.0, 255.0, 255.0))
    assert abs(float(h.mean())) < 0.05


def test_model_input_is_three_identical_channels_after_normalisation():
    h = np.full((8, 8), 0.75, dtype=np.float32)
    tensor = hchannel.to_model_input(h)

    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == np.float32
    # Identical *before* normalisation; the ImageNet statistics differ per channel,
    # so the three planes are affine images of one another rather than equal.
    raw = (tensor * np.asarray(hchannel.IMAGENET_STD).reshape(3, 1, 1)) + np.asarray(
        hchannel.IMAGENET_MEAN
    ).reshape(3, 1, 1)
    assert np.allclose(raw[0], raw[1], atol=1e-6)
    assert np.allclose(raw[0], raw[2], atol=1e-6)
    assert abs(float(raw[0].mean()) - 0.75 / hchannel.OD_CLIP) < 1e-5


def test_clip_saturates_at_od_clip():
    dark = np.full((4, 4), 3.0, dtype=np.float32)
    at_clip = np.full((4, 4), hchannel.OD_CLIP, dtype=np.float32)
    assert np.allclose(hchannel.to_model_input(dark), hchannel.to_model_input(at_clip))


def test_storage_round_trip_is_within_one_level():
    h = np.linspace(0.0, hchannel.OD_CLIP, 256, dtype=np.float32).reshape(16, 16)
    back = hchannel.dequantise(hchannel.quantise(h))
    assert np.max(np.abs(back - h)) <= hchannel.OD_CLIP / 255.0 / 2 + 1e-6


def test_from_stored_matches_the_float_path():
    """The training path (stored uint8) and the inference path (live float) agree.

    They must, because the model is fitted through the first and served through the
    second. The tolerance is one quantisation level, which is the only difference the
    tile store introduces.
    """
    h = np.random.default_rng(0).uniform(0.0, hchannel.OD_CLIP, size=(32, 32)).astype(np.float32)
    stored = hchannel.quantise(h)

    from_store = hchannel.from_stored(stored)
    from_float = hchannel.to_model_input(hchannel.dequantise(stored))
    assert np.allclose(from_store, from_float, atol=1e-6)

    # And within quantisation error of the original density.
    assert np.max(np.abs(from_store - hchannel.to_model_input(h))) < 0.05


def test_jitter_scales_before_the_clip():
    """Stain jitter is multiplicative in OD and applied before saturation.

    A tile jittered up must saturate more of its nuclei, which is what a strongly
    counterstained section actually looks like. Applying the jitter after the clip
    would merely brighten an already-flat image and teach the model nothing about
    stain strength.
    """
    h = np.full((4, 4), 1.0, dtype=np.float32)
    strong = hchannel.to_model_input(h, alpha=2.0)
    at_clip = hchannel.to_model_input(np.full((4, 4), hchannel.OD_CLIP, dtype=np.float32))
    assert np.allclose(strong, at_clip)


def test_invert_flips_polarity():
    h = np.full((4, 4), 0.0, dtype=np.float32)
    bright_nuclei = hchannel.to_model_input(h)                # empty tile -> 0
    dark_nuclei = hchannel.to_model_input(h, invert=True)      # empty tile -> 1
    assert float(dark_nuclei.mean()) > float(bright_nuclei.mean())


def test_descriptor_records_everything_inference_needs():
    """The manifest has to pin the whole input contract, not just the tile size."""
    described = hchannel.descriptor(tile_px=224, mpp=0.5, invert=False)
    for key in ("channel", "stain_basis", "od_clip", "tile_px", "mpp", "normalisation",
                "invert_polarity", "replicate_to_3ch"):
        assert key in described
    assert described["od_clip"] == hchannel.OD_CLIP
    assert described["stain_basis"] == "ruifrok_hdab_fixed"


def test_white_point_is_per_channel():
    """Three numbers, not one: a scanner's lamp and coverslip are not neutral."""
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 240, 250, 230
    assert hchannel.white_point(rgb) == (240.0, 250.0, 230.0)
