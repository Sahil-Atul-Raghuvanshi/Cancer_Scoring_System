"""Classical image-quality metrics - the explanation half of step 2.

These are the measurements HistoQC is built out of: gradient-based sharpness,
Laplacian focus, RMS and Michelson contrast, mean brightness, mean saturation,
and Gabor texture energy. They are computed here directly rather than by
shelling out to HistoQC, because the pipeline only needs the numbers, not
HistoQC's slide-level pass/fail machinery, and a demo should not carry a second
whole-slide framework as a runtime dependency.

What they are for, and what they are *not* for
----------------------------------------------
GrandQC decides. These explain. A blurred region is rejected because the
artefact model said `focus`; the sharpness number is shown next to it so a
viewer can see that the model's call is consistent with something they can
understand. Nothing here feeds back into the decision - if it did, the demo
would be claiming a rigour these thresholds do not have.

Scale matters and is reported
-----------------------------
Sharpness is a statement about a resolution. The same tissue measured at
1.5 um/px and at 0.5 um/px gives different Tenengrad values, and only the finer
one says much about scanner focus. So every result carries the mpp it was
measured at, and comparisons are only ever made between tiles measured at the
same scale - as ratios against the slide's own clean tissue, never against a
constant baked in here.

Dependencies: numpy always; scipy for the derivative filters. Both are cheap
next to the model half, so this layer runs even when the GrandQC checkpoints
are absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

# --- what the UI is told about each metric ----------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """Presentation metadata for one metric."""

    key: str
    label: str
    unit: str
    description: str
    #: True when a *low* value is the suspicious direction. Drives the arrow
    #: the explanation panel draws, nothing else.
    low_is_bad: bool


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="tenengrad",
        label="Sharpness",
        unit="gradient energy",
        description=(
            "Tenengrad: the mean squared Sobel gradient. Edges in focus are steep, so a "
            "sharp field has high gradient energy. The classic focus measure."
        ),
        low_is_bad=True,
    ),
    MetricSpec(
        key="laplacian_variance",
        label="Focus",
        unit="variance",
        description=(
            "Variance of the Laplacian. A second-derivative view of the same question, "
            "more sensitive to fine detail and noisier for it."
        ),
        low_is_bad=True,
    ),
    MetricSpec(
        key="rms_contrast",
        label="Contrast (RMS)",
        unit="std of luminance",
        description=(
            "Standard deviation of luminance. Falls when a region is washed out, rises "
            "where a fold has doubled the optical path."
        ),
        low_is_bad=True,
    ),
    MetricSpec(
        key="michelson_contrast",
        label="Contrast (Michelson)",
        unit="0-1",
        description=(
            "(max - min) / (max + min), taken at the 1st and 99th percentiles so a "
            "single dust speck cannot set the scale."
        ),
        low_is_bad=True,
    ),
    MetricSpec(
        key="brightness",
        label="Brightness",
        unit="0-1",
        description=(
            "Mean luminance. Dark spots and folds pull it down; bubbles and thin tissue "
            "push it up towards glass."
        ),
        low_is_bad=False,
    ),
    MetricSpec(
        key="saturation",
        label="Colour saturation",
        unit="0-1",
        description=(
            "Mean HSV saturation. Pen ink is far more saturated than any stain, which is "
            "exactly why a saturation-based tissue mask mistakes it for tissue."
        ),
        low_is_bad=False,
    ),
    MetricSpec(
        key="texture_energy",
        label="Texture",
        unit="Gabor response",
        description=(
            "Mean response of a four-orientation Gabor bank. High over structured tissue, "
            "near zero over the flat interior of a bubble or a smear of ink."
        ),
        low_is_bad=True,
    ),
)

METRIC_KEYS: tuple[str, ...] = tuple(spec.key for spec in METRIC_SPECS)
_SPEC_BY_KEY = {spec.key: spec for spec in METRIC_SPECS}


def metric_spec(key: str) -> MetricSpec | None:
    return _SPEC_BY_KEY.get(key)


@dataclass(frozen=True)
class TileFeatures:
    """The seven metrics for one tile."""

    tenengrad: float
    laplacian_variance: float
    rms_contrast: float
    michelson_contrast: float
    brightness: float
    saturation: float
    texture_energy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# --- primitives --------------------------------------------------------------


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luminance in 0..1 as float32."""
    array = rgb.astype(np.float32)
    if array.max() > 1.5:  # uint8 input
        array = array / 255.0
    return 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]


def _saturation_channel(rgb: np.ndarray) -> np.ndarray:
    """HSV saturation in 0..1, computed without a colourspace round-trip."""
    array = rgb.astype(np.float32)
    if array.max() > 1.5:
        array = array / 255.0
    high = array.max(axis=-1)
    low = array.min(axis=-1)
    # Guard the achromatic case: where max is 0 the pixel is black and has no
    # defined hue or saturation, so 0 is the right answer rather than a nan.
    return np.divide(high - low, high, out=np.zeros_like(high), where=high > 1e-6)


@lru_cache(maxsize=1)
def _gabor_bank(
    size: int = 21, frequency: float = 0.18, sigma: float = 4.0, gamma: float = 0.6
) -> tuple[np.ndarray, ...]:
    """Four real Gabor kernels at 0, 45, 90 and 135 degrees.

    Each kernel is mean-subtracted so a perfectly flat field responds with
    exactly zero - without that, brightness leaks into what is meant to be a
    texture measurement.
    """
    half = size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)

    kernels = []
    for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        x_rot = x * np.cos(theta) + y * np.sin(theta)
        y_rot = -x * np.sin(theta) + y * np.cos(theta)
        envelope = np.exp(-(x_rot**2 + (gamma * y_rot) ** 2) / (2 * sigma**2))
        kernel = envelope * np.cos(2 * np.pi * frequency * x_rot)
        kernels.append((kernel - kernel.mean()).astype(np.float32))

    return tuple(kernels)


#: Side length the Gabor bank is evaluated at in fast mode.
#:
#: Texture energy is a mean of squared responses, so a few thousand samples is
#: plenty; the cost, on the other hand, is O(pixels x kernel) per orientation
#: and it is the most expensive thing in this module by a wide margin. Fixing
#: the evaluation size keeps the cost per cell constant no matter how large a
#: cell is - which matters, because the whole-slide pass measures thousands of
#: them.
#:
#: The decimation shifts the spatial frequency the bank responds to. That shift
#: is identical for every cell in a run, and texture is only ever reported as a
#: ratio between cells of the same slide, so it cancels.
_TEXTURE_EVAL_SIZE = 64


def _decimate(array: np.ndarray, target: int) -> np.ndarray:
    """Cheap integer-stride decimation, for metrics that do not need full detail."""
    step = max(1, min(array.shape[0], array.shape[1]) // target)
    return array[::step, ::step] if step > 1 else array


# --- the metrics -------------------------------------------------------------


def tile_features(rgb: np.ndarray, *, fast_texture: bool = True) -> TileFeatures:
    """Every metric for one RGB tile.

    `fast_texture` computes the Gabor bank on a decimated copy. Over a whole
    slide that is thousands of convolutions, and texture is a coarse property
    anyway; the explanation panel turns it off for the single region it
    inspects closely.
    """
    from scipy import ndimage

    luminance = _luminance(rgb)

    # Sharpness and focus, both measured at the tile's own resolution:
    # decimating first would measure the blur we introduced, not the scanner's.
    grad_y = ndimage.sobel(luminance, axis=0, mode="reflect")
    grad_x = ndimage.sobel(luminance, axis=1, mode="reflect")
    tenengrad = float(np.mean(grad_x**2 + grad_y**2))

    laplacian = ndimage.laplace(luminance, mode="reflect")
    laplacian_variance = float(np.var(laplacian))

    # Contrast.
    rms_contrast = float(np.std(luminance))
    low, high = np.percentile(luminance, (1.0, 99.0))
    denominator = float(high + low)
    michelson = float((high - low) / denominator) if denominator > 1e-6 else 0.0

    brightness = float(np.mean(luminance))
    saturation = float(np.mean(_saturation_channel(rgb)))

    # Texture. Always via the FFT path: with `fast_texture` off this runs over
    # the region at full size, and a direct 21x21 convolution over a 2048 px
    # region is billions of operations - slow enough to time out a request.
    source = _decimate(luminance, _TEXTURE_EVAL_SIZE) if fast_texture else luminance
    responses = [
        float(np.sqrt(np.mean(_gabor_response(source, kernel) ** 2)))
        for kernel in _gabor_bank()
    ]
    texture_energy = float(np.mean(responses))

    return TileFeatures(
        tenengrad=tenengrad,
        laplacian_variance=laplacian_variance,
        rms_contrast=rms_contrast,
        michelson_contrast=michelson,
        brightness=brightness,
        saturation=saturation,
        texture_energy=texture_energy,
    )


def _gabor_response(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve via FFT, with reflected edges.

    A 21x21 kernel over the whole-patch texture image is the single most
    expensive operation in this module; done directly it costs more than every
    other metric combined. An FFT does the same work an order of magnitude
    faster at these sizes.

    `fftconvolve` has no boundary modes, and zero padding would make the
    mean-subtracted kernel fire along the image border. Padding by hand first
    and taking the `valid` region restores the same edges
    `ndimage.convolve(mode="reflect")` would have produced, and returns an
    array of the original shape.

    Note the padding mode. numpy and ndimage use the same two words for
    opposite conventions: ndimage's "reflect" duplicates the edge sample
    (dcba|abcd), which numpy calls "symmetric"; numpy's "reflect" is
    ndimage's "mirror". Getting this wrong shifts every response by a pixel
    at the border and silently changes the numbers.
    """
    from scipy.signal import fftconvolve

    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode="symmetric")
    return fftconvolve(padded, kernel, mode="valid").astype(np.float32)


def _as_blocks(array: np.ndarray, blocks: int) -> np.ndarray:
    """Reshape HxW into (blocks*blocks, pixels-per-block), row-major by block.

    Block index is `row * blocks + col`, matching the order the caller walks
    its sub-blocks in.
    """
    side = array.shape[0] // blocks
    return (
        array[: blocks * side, : blocks * side]
        .reshape(blocks, side, blocks, side)
        .transpose(0, 2, 1, 3)
        .reshape(blocks * blocks, side * side)
    )


def block_features(rgb: np.ndarray, *, blocks: int) -> list[dict[str, float]]:
    """Metrics for every sub-block of one patch.

    Equivalent to calling `tile_features` on each sub-block, but every filter is
    evaluated once over the whole patch and only the *statistics* are reduced
    per block. That is both faster - one Sobel pass instead of N, and the fixed
    per-call overhead of scipy's filters paid once - and slightly more correct,
    because a block's gradients at its edge are computed from the neighbouring
    tissue rather than from a reflected copy of itself.

    Returns one dict per block, in row-major order.
    """
    from scipy import ndimage

    luminance = _luminance(rgb)
    saturation = _saturation_channel(rgb)

    grad_y = ndimage.sobel(luminance, axis=0, mode="reflect")
    grad_x = ndimage.sobel(luminance, axis=1, mode="reflect")
    gradient_energy = grad_x**2 + grad_y**2
    laplacian = ndimage.laplace(luminance, mode="reflect")

    # Texture: one convolution pass over a decimated copy of the whole patch,
    # then reduced per block like everything else.
    decimated = _decimate(luminance, _TEXTURE_EVAL_SIZE * blocks)
    responses = [_gabor_response(decimated, kernel) ** 2 for kernel in _gabor_bank()]

    flat_luminance = _as_blocks(luminance, blocks)
    flat_gradient = _as_blocks(gradient_energy, blocks)
    flat_laplacian = _as_blocks(laplacian, blocks)
    flat_saturation = _as_blocks(saturation, blocks)
    flat_texture = [_as_blocks(response, blocks) for response in responses]

    low, high = np.percentile(flat_luminance, (1.0, 99.0), axis=1)
    total = high + low
    michelson = np.divide(
        high - low, total, out=np.zeros_like(total), where=np.abs(total) > 1e-6
    )

    tenengrad = flat_gradient.mean(axis=1)
    laplacian_variance = flat_laplacian.var(axis=1)
    rms = flat_luminance.std(axis=1)
    brightness = flat_luminance.mean(axis=1)
    saturation_mean = flat_saturation.mean(axis=1)
    texture = np.mean([np.sqrt(block.mean(axis=1)) for block in flat_texture], axis=0)

    return [
        {
            "tenengrad": float(tenengrad[index]),
            "laplacian_variance": float(laplacian_variance[index]),
            "rms_contrast": float(rms[index]),
            "michelson_contrast": float(michelson[index]),
            "brightness": float(brightness[index]),
            "saturation": float(saturation_mean[index]),
            "texture_energy": float(texture[index]),
        }
        for index in range(blocks * blocks)
    ]


def features_available() -> tuple[bool, str | None]:
    """Whether the classical layer can run, and why not when it cannot."""
    try:
        import scipy  # noqa: F401
    except ImportError as exc:
        return False, f"scipy is not installed ({exc})"
    return True, None
