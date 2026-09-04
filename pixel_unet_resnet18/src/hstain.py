"""The haematoxylin channel: RGB in, the tensor the network sees out.

A standalone reimplementation of the transform `bcss_bracs_hchannel_resnet18/src/hchannel.py`
re-exports from the demo backend. Reimplemented because this folder depends on no code
from approach 1 (see `paths.py`), and that is exactly the thing approach 1's own exporter
warns about:

    "A second exporter is exactly where a stray `skimage.color.rgb2hed` gets introduced,
    and the failure looks like a model problem for a week."

So three things are done about it rather than hoping:

1. **The stain matrix is a hard-coded constant, not a re-derivation.** Approach 1 builds
   Ruifrok's H-DAB basis by completing two reference vectors with their cross product. Two
   implementations of that construction could differ in sign or normalisation and nothing
   would raise. The resolved 3x3 is written out below to full double precision instead, so
   there is no arithmetic to get wrong.
2. **`tests/test_parity.py` asserts bit-for-bit equality** with approach 1 on random
   inputs, importing it in the test only and skipping when it is absent.
3. **The order of operations is stated and is not interchangeable.** Each step is here
   because doing it in a different order is wrong rather than merely different.

--------------------------------------------------------------------------------
The pipeline, and why each step is where it is
--------------------------------------------------------------------------------

    white point   -> the high percentile of each channel. A BCSS region is an extracted
                     rectangle of tissue with no glass in it, so there is nothing to
                     sample an `I0` from and this stands in.
    optical density -> -log10(I / I0), floored so a black pixel gives a large finite
                     density rather than an infinity. The floor clips the OBSERVED
                     intensity and never `I0`.
    deconvolution -> one inverse of a 3x3 and one matrix multiply. Column 0 of the
                     coefficients is haematoxylin.
    quantise      -> to the uint8 the tile store holds: [0, OD_CLIP] onto [0, 255]. The
                     stored artefact is the model's INPUT RANGE, not the raw density.
    to_model_input-> standardise, then scale-and-offset, then clip, then the shape term,
                     then polarity, then ImageNet normalisation.

`gamma` after the clip rather than before is what keeps it a pure shape change: applied to
a density it would also move the saturation point, and then one knob would be doing two
jobs.
"""

from __future__ import annotations

import numpy as np

#: Optical density at which the haematoxylin channel is considered saturated. Above this a
#: nucleus is simply "as dark as nuclei get", and letting the tail run free would hand the
#: whole input range to a handful of pixels.
OD_CLIP: float = 1.5

#: Intensity floor for the Beer-Lambert transform, in intensity units.
OD_FLOOR: float = 1.0

#: Lower bound on a tile's own p99 before `standardise` will rescale it. A tile of bare
#: glass has no stain to normalise against, and dividing by its p99 would stretch sensor
#: noise across the whole input range - inventing texture where the slide has none.
STANDARDISE_FLOOR: float = 0.10

#: Where a standardised tile's own p99 is placed inside [0, 1]. Not 1.0: the tail above
#: p99 needs somewhere to go, and pinning p99 to the top would clip every nucleus.
STANDARDISE_TARGET: float = 0.60

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

#: Ruifrok's 3x3 haematoxylin-DAB matrix, columns being the stains, resolved to double
#: precision from the reference vectors H=(0.65, 0.704, 0.286) and DAB=(0.268, 0.57, 0.776)
#: with the third column their normalised cross product.
#:
#: **Hard-coded, and that is the point.** A per-image estimate rescales itself to whatever
#: it is given, so "0.4 haematoxylin" would mean a different amount of dye on every slide.
#: These are the numbers everyone publishes against, and they are what
#: `skimage.color.rgb2hed` inverts. Verified equal to the demo backend's `RUIFROK_HDAB` by
#: `tests/test_parity.py`.
RUIFROK_HDAB: np.ndarray = np.array(
    [
        [0.6500286018877385, 0.2681475217165787, 0.6362142329617175],
        [0.7040309780445659, 0.5703137588748127, -0.7100267962603203],
        [0.28601258483060493, 0.7764271524330785, 0.3018168291683533],
    ],
    dtype=np.float64,
)

#: Inverted once, here, rather than per call: un-mixing a quarter of a million pixels
#: should invert a 3x3 exactly once.
RUIFROK_INVERSE: np.ndarray = np.linalg.inv(RUIFROK_HDAB)


def white_point(rgb: np.ndarray, percentile: float = 99.0) -> tuple[float, float, float]:
    """`I0` for an image with no glass in it: the high percentile of each channel.

    Taken **once per region**, never per tile. Neighbouring tiles must share a white point
    because they are the same physical section under the same lamp; a per-tile `I0` would
    rescale a tile of pale stroma and a tile of dense tumour against their own brightest
    pixels and erase the difference the model reads.
    """
    flat = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    values = np.percentile(flat, percentile, axis=0)
    return (float(values[0]), float(values[1]), float(values[2]))


def optical_density(
    intensity: np.ndarray,
    white: np.ndarray | tuple[float, float, float],
    *,
    floor: float = OD_FLOOR,
) -> np.ndarray:
    """Beer-Lambert optical density, `-log10(I / I0)`, as float32.

    `floor` clips `I` from below so a black pixel yields a large finite density instead of
    an infinity. It clips the *observed* intensity and never `I0`: a scanner that clipped
    its whites produces a white point pinned at 255, which is a fact to report rather than
    a value to quietly repair.
    """
    observed = np.maximum(np.asarray(intensity, dtype=np.float32), np.float32(floor))
    reference = np.asarray(white, dtype=np.float32)
    return -np.log10(observed / np.maximum(reference, np.float32(floor)))


def haematoxylin_od(
    rgb: np.ndarray,
    white: np.ndarray | tuple[float, float, float],
    *,
    od_floor: float = OD_FLOOR,
) -> np.ndarray:
    """The haematoxylin channel of an HxWx3 uint8 image, in optical density units.

    Unclipped and unscaled - that is `to_model_input`'s job. Not clipped at zero either:
    the system is square, so the coefficients *are* the pixel, and a colour outside the
    cone the two stains span has to be described by a negative amount of one of them.
    """
    od = optical_density(rgb, white, floor=od_floor)
    height, width = od.shape[0], od.shape[1]
    flat = od.reshape(-1, 3).astype(np.float64)
    coefficients = flat @ RUIFROK_INVERSE.T
    return coefficients.reshape(height, width, 3)[..., 0].astype(np.float32)


def quantise(h_od: np.ndarray) -> np.ndarray:
    """Haematoxylin density to the uint8 the tile store holds: `[0, OD_CLIP]` -> `[0, 255]`.

    One level is 0.0059 OD, finer than the 8-bit intensity data the density was computed
    from can resolve, so the quantisation is lossless in every way that matters.
    """
    scaled = np.clip(np.asarray(h_od, dtype=np.float32), 0.0, OD_CLIP) / OD_CLIP
    return np.rint(scaled * 255.0).astype(np.uint8)


def dequantise(stored: np.ndarray) -> np.ndarray:
    """The stored uint8 back to optical density. The inverse of `quantise`."""
    return np.asarray(stored, dtype=np.float32) / 255.0 * OD_CLIP


def to_model_input(
    h_od: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 1.0,
    invert: bool = False,
    standardise: bool = False,
) -> np.ndarray:
    """Haematoxylin density to the 3xHxW float32 the network sees.

    `alpha` and `beta` are the stain-strength jitter and `gamma` is the shape term. They
    are arguments to *this* function rather than a separate augmentation step because
    "HED augmentation" on a single channel is a stain-strength jitter: our input is the H
    channel alone, so there is nothing to recompose, and perturbing three identical copies
    of one number in three directions would produce a colour cast no slide can have.

    `standardise` divides the tile by its own p99 before the clip. It fixes a *scale*
    difference and, being monotone, does nothing for a *shape* difference - which is why
    `gamma` exists as well.

    Order is fixed: standardise, then scale and offset in density units, then the clip,
    then the shape term on the normalised value, then polarity, then ImageNet
    normalisation.
    """
    x = np.asarray(h_od, dtype=np.float32)

    if standardise:
        ref = float(np.percentile(x, 99.0))
        if ref >= STANDARDISE_FLOOR:
            x = x * np.float32(STANDARDISE_TARGET * OD_CLIP / ref)

    x = np.clip(x * np.float32(alpha) + np.float32(beta), 0.0, OD_CLIP) / OD_CLIP

    if gamma != 1.0:
        # x is already in [0, 1], so the power is a shape change and nothing else.
        x = np.power(x, np.float32(gamma), dtype=np.float32)

    if invert:
        x = 1.0 - x

    stack = np.repeat(x[np.newaxis, :, :], 3, axis=0)
    return (stack - _MEAN) / _STD


def from_stored(
    stored: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 1.0,
    invert: bool = False,
    standardise: bool = False,
) -> np.ndarray:
    """A stored tile's uint8 straight to a model input. Dequantise, then transform."""
    return to_model_input(
        dequantise(stored),
        alpha=alpha, beta=beta, gamma=gamma, invert=invert, standardise=standardise,
    )


def descriptor(
    *,
    tile_px: int,
    mpp: float,
    invert: bool,
    gamma: float = 1.0,
    standardise: bool = False,
) -> dict[str, object]:
    """The input contract, as the JSON a checkpoint's manifest carries.

    Inference must reproduce this exactly. The same shape as approach 1's descriptor, so a
    reader comparing the two checkpoints is comparing like with like.
    """
    return {
        "channel": "haematoxylin",
        "stain_basis": "ruifrok_hdab_fixed",
        "od_clip": OD_CLIP,
        "od_floor": OD_FLOOR,
        "tile_px": int(tile_px),
        "mpp": float(mpp),
        "replicate_to_3ch": True,
        "invert_polarity": bool(invert),
        "gamma": float(gamma),
        "standardise_tile_p99": bool(standardise),
        "standardise_floor": STANDARDISE_FLOOR,
        "standardise_target": STANDARDISE_TARGET,
        "normalisation": "imagenet",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }


def resample(
    rgb: np.ndarray, mask: np.ndarray, *, source_mpp: float, target_mpp: float
) -> tuple[np.ndarray, np.ndarray]:
    """Region and mask to the working resolution, each by its correct filter.

    **BOX on the intensities, before any logarithm.** A coarser sensor averages the light
    arriving over a larger area, so averaging transmissions is what a coarser scan
    physically *is*. Averaging densities instead takes the mean of logarithms - the log of
    a geometric mean - which is biased low and is a number no instrument records.

    **NEAREST on the labels. Never BOX, never BILINEAR.** An averaged label is a class
    nobody annotated, and between `tumor` (1) and `stroma` (2) the interpolated value is
    `dcis`-adjacent nonsense.

    The factor is computed from the two resolutions rather than assumed, because a mirror
    that reshipped regions at a different scale would otherwise produce tiles at the wrong
    field of view - and a 224 px tile at 0.25 um/px is 56 um, half the cells and none of
    the architecture.
    """
    from PIL import Image

    if source_mpp <= 0 or target_mpp <= 0:
        raise ValueError("resolutions are microns per pixel and must be positive")

    factor = target_mpp / source_mpp
    if abs(factor - 1.0) < 1e-6:
        return np.asarray(rgb), np.asarray(mask)

    height, width = rgb.shape[:2]
    size = (max(1, int(round(width / factor))), max(1, int(round(height / factor))))

    small_rgb = np.asarray(
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize(
            size, Image.Resampling.BOX
        )
    )
    small_mask = np.asarray(
        Image.fromarray(np.asarray(mask, dtype=np.uint8)).resize(
            size, Image.Resampling.NEAREST
        )
    )
    return small_rgb, small_mask
