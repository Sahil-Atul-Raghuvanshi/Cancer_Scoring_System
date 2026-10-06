"""Colour and thresholding primitives shared by more than one step.

Step 3 is built on these. Step 2's tissue-detection fallback is built on the
same two, and that is precisely why they live here rather than in either
package: the fallback is a stand-in *for* step 3's method, so if the two ever
drifted apart the fallback would stop being the thing it claims to be.

`channel_percentile` and `optical_density` are here for the same reason one step
ahead of themselves. Step 4 estimates a white point and then has to say what the
empty glass measures once that white point is divided out - which is the Beer-
Lambert transform step 5 is entirely about. One implementation means the noise
floor step 4 quotes and the density step 5 computes cannot disagree.

Otsu (1979) is stated in terms of a 256-bin histogram, so that is the form
these functions take. Everything downstream - the threshold, the criterion
curve the demo draws, the variance ratio it quotes - is read off the same
histogram, which means the picture on screen and the number beside it cannot
disagree.
"""

from __future__ import annotations

import numpy as np


def saturation_channel(rgb: np.ndarray) -> np.ndarray:
    """HSV saturation of an HxWx3 uint8 image, as uint8.

    Saturation, not brightness, is what separates stained tissue from empty
    glass. Glass is bright and colourless; pale tissue is dim but still
    coloured. The two share brightness and do not share saturation, so a
    brightness threshold has to choose between losing pale tissue and keeping
    bright glass while a saturation threshold has to do neither.

    `S = (max - min) / max` is HSV's own definition. Pure black has no hue to
    be saturated in, so it is defined as 0 rather than left as a division by
    zero.
    """
    scaled = rgb.astype(np.float32) / 255.0
    high = scaled.max(axis=-1)
    low = scaled.min(axis=-1)
    saturation = np.divide(high - low, high, out=np.zeros_like(high), where=high > 1e-6)
    return (saturation * 255.0).astype(np.uint8)



def channel_percentile(
    rgb: np.ndarray, selection: np.ndarray, percentile: float
) -> tuple[float, float, float]:
    """A percentile of each RGB channel over the selected pixels.

    Per channel and not over the luminance, because the three channels of a
    white point are genuinely different numbers: a scanner's lamp and its
    coverslip are not neutral, and the whole reason step 4 stores three values
    rather than one is that collapsing them would bake that cast into every
    optical density downstream.

    Returns zeros for an empty selection rather than raising. A caller with no
    glass left to sample has a bigger problem than a division by zero, and it
    reports that problem itself.
    """
    if not np.any(selection):
        return (0.0, 0.0, 0.0)

    values = rgb[selection]
    result = np.percentile(values.astype(np.float64), percentile, axis=0)
    return (float(result[0]), float(result[1]), float(result[2]))


#: What "optical density" means everywhere in this project (P-21): the scanner's sRGB
#: values used as given, not linearised. See `optical_density`.
OD_CONVENTION = "srgb_encoded"


def optical_density(
    intensity: np.ndarray, white: np.ndarray | tuple[float, float, float], *, floor: float = 1.0
) -> np.ndarray:
    """Beer-Lambert optical density, `-log10(I / I0)`, as float32.

    Here rather than in step 5's package because step 4 needs it before step 5
    exists: a white point is only meaningful if you can say what the *glass*
    measures once you divide by it, and that number - the noise floor every
    downstream threshold has to clear - is this function applied to the glass
    itself. Two copies of a logarithm would be two chances for the floor step 4
    quotes and the density step 5 computes to disagree.

    `floor` clips `I` from below, in intensity units, so a black pixel yields a
    large finite density instead of an infinity. It clips the *observed*
    intensity and never `I0`: a scanner that clipped its whites produces a
    white point pinned at 255, which is a fact to report rather than a value to
    quietly repair.

    `white` may be one triple - a flat white point - or a full HxWx3 field, for
    the fitted surface that also corrects vignetting. Both broadcast.

    **The convention, stated (P-21): `intensity` is the scanner's sRGB-encoded value,
    used as given, not linearised first.** Beer-Lambert holds for linear light, and
    the sRGB curve compresses the dark end, so a density computed this way reads high
    stain lower than a linear one would - which matters most at 2+/3+. It is kept on
    purpose rather than corrected: every optical density in the pipeline is defined
    under it - the step 8 heads were trained on it, the white point and noise floor
    are measured in it, and the cut points are being fitted in it (P-01). Linearising
    here alone would serve every model input it never saw and move every cut. Every
    number this project reports as "OD" is therefore *sRGB-encoded OD*, calibrated
    under that convention; `OD_CONVENTION` names it for any record that needs to.
    """
    observed = np.maximum(np.asarray(intensity, dtype=np.float32), np.float32(floor))
    reference = np.asarray(white, dtype=np.float32)
    return -np.log10(observed / np.maximum(reference, np.float32(floor)))


def histogram_256(values: np.ndarray) -> np.ndarray:
    """Counts per level for a uint8 array, always 256 long."""
    return np.bincount(values.ravel(), minlength=256)[:256].astype(np.float64)


def between_class_variance(histogram: np.ndarray) -> np.ndarray:
    """Otsu's criterion at every threshold, given a 256-bin histogram.

    Entry `t` is the between-class variance obtained by splitting at `t` - that
    is, with `value <= t` on one side and `value > t` on the other. Levels that
    would leave one side empty score -1 so they can never win an argmax.

    Otsu's insight is that minimising within-class variance and maximising this
    are the same problem, and that this form needs only cumulative sums. So the
    whole curve costs one pass, which is what lets the demo draw it rather than
    just assert its peak.
    """
    counts = np.asarray(histogram, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return np.full(256, -1.0)

    levels = np.arange(256, dtype=np.float64)
    weight_low = np.cumsum(counts)
    weight_high = total - weight_low

    sum_low = np.cumsum(counts * levels)
    sum_total = sum_low[-1]

    mean_low = np.divide(sum_low, weight_low, out=np.zeros_like(sum_low), where=weight_low > 0)
    mean_high = np.divide(
        sum_total - sum_low, weight_high, out=np.zeros_like(sum_low), where=weight_high > 0
    )

    variance = weight_low * weight_high * (mean_low - mean_high) ** 2
    variance[(weight_low <= 0) | (weight_high <= 0)] = -1.0
    return variance


def otsu_from_histogram(histogram: np.ndarray) -> int:
    """The threshold that maximises Otsu's criterion, as a level in 0..255."""
    return int(np.argmax(between_class_variance(histogram)))


def otsu_threshold(values: np.ndarray) -> int:
    """Otsu's threshold for a uint8 array. Use with `values > threshold`."""
    return otsu_from_histogram(histogram_256(values))


def modal_share(histogram: np.ndarray) -> tuple[int, float]:
    """The busiest level, and the fraction of all pixels sitting on it.

    This is the test of whether Otsu's assumption holds. Otsu's derivation
    treats the image as two classes each with a spread, and finds the split that
    maximises the variance between them. A single level holding most of the mass
    is not a class with a spread - it is a spike with zero variance - and against
    a long tail the criterion is then maximised far out in that tail rather than
    at any boundary a person would recognise.

    That is not a hypothetical. It is what a weakly counterstained IHC slide
    looks like: the section is so faintly coloured that most of it, and all of
    the background, lands at saturation zero.
    """
    counts = np.asarray(histogram, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0, 0.0
    level = int(np.argmax(counts))
    return level, float(counts[level] / total)


def triangle_from_histogram(histogram: np.ndarray) -> int:
    """Zack's triangle threshold: for a histogram that is one peak and a tail.

    Where Otsu looks for the best split between two humps, this looks for the
    point where the tail departs furthest from a straight line. Draw a chord
    from the top of the peak to the far end of the occupied range; the threshold
    is the level whose bar sits furthest below that chord. On a skewed unimodal
    histogram - a spike of background and a tail of faint tissue - that is the
    shoulder where the peak stops and the tail begins, which is exactly the
    boundary Otsu cannot see.

    It is not a replacement for Otsu. On a genuinely bimodal histogram the chord
    is a poor model and this lands well below the valley, so the two rules suit
    two shapes and the caller has to know which shape it has.

    Zack, Rogers & Latt, J Histochem Cytochem 25(7):741-753 (1977).
    """
    counts = np.asarray(histogram, dtype=np.float64)
    occupied = np.nonzero(counts)[0]
    if occupied.size == 0:
        return 0
    if occupied.size == 1:
        return int(occupied[0])

    peak = int(np.argmax(counts))
    low, high = int(occupied[0]), int(occupied[-1])

    # Take whichever side of the peak is longer: that is the tail.
    if (high - peak) >= (peak - low):
        segment, forwards = counts[peak : high + 1], True
    else:
        segment, forwards = counts[low : peak + 1][::-1], False

    span = len(segment)
    if span < 2:
        return peak

    # Normalise so the chord runs from (0, span-1) to (span-1, 0). Without this
    # the answer would depend on the pixel count, which is not a property of the
    # image's shape.
    tallest = float(segment.max())
    if tallest <= 0:
        return peak

    x = np.arange(span, dtype=np.float64)
    y = segment / tallest * (span - 1)
    y0 = float(y[0])

    # Perpendicular distance from each bar's top to the chord.
    distance = np.abs(y0 * x + (span - 1) * y - y0 * (span - 1)) / np.hypot(y0, span - 1)
    offset = int(np.argmax(distance))

    return peak + offset if forwards else peak - offset


def class_means(histogram: np.ndarray, threshold: int) -> tuple[float | None, float | None]:
    """Mean level either side of a threshold, or None for an empty side.

    The gap between these two is what the threshold bought: on a slide with
    tissue on it they should sit in the two humps of the histogram, not near
    each other.
    """
    counts = np.asarray(histogram, dtype=np.float64)
    levels = np.arange(256, dtype=np.float64)
    cut = max(0, min(255, threshold))

    low, high = counts[: cut + 1], counts[cut + 1 :]
    low_weight, high_weight = low.sum(), high.sum()

    below = float((low * levels[: cut + 1]).sum() / low_weight) if low_weight else None
    above = float((high * levels[cut + 1 :]).sum() / high_weight) if high_weight else None
    return below, above
