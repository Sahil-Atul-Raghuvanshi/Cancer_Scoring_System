"""Stain vectors: the geometry two steps of this pipeline share.

A stain is a *direction* in optical density space. Beer-Lambert says one absorber
at any concentration lies on one ray from the origin - `OD = c * v`, with `v`
fixed and `c` the amount - so everything the pipeline does with stains after step
5 is geometry on unit vectors and non-negative coefficients.

Two steps need that geometry and they need it to agree:

  step 5   finds the two arms of the tile's density cloud and shows they point at
           Ruifrok & Johnston's published vectors. It is looking, not measuring.
  step 6   projects onto those same fixed vectors and keeps the coefficients -
           the haematoxylin channel for the model branch, the DAB channel for the
           measurement branch - and estimates a per-image basis alongside them,
           with `estimate_stain_matrix`, purely to show what using it would cost.

If the two disagreed, the demo would show a reader one pair of arms on step 5's
screen and a different pair driving step 6's arithmetic on the next. So the
geometry lives here once, and both call it. `app.common.imaging.optical_density` is here for the
same reason one step earlier - step 4 needs the transform before step 5 exists -
and this module is its continuation: that one turns colour into density, this one
reads directions out of the result.

Nothing here knows what a slide, a tile or a service is. It takes arrays of
optical density and returns vectors, angles and coefficients.

Ruifrok AC, Johnston DA. Quantification of histochemical staining by colour
deconvolution. Anal Quant Cytol Histol 23(4):291-299 (2001).
Macenko M et al. A method for normalizing histology slides for quantitative
analysis. ISBI 2009:1107-1110. Kept as a reference for the estimator step 5
draws its arms with - the pipeline does not normalise with it.
"""

from __future__ import annotations

import numpy as np

#: Ruifrok & Johnston's published unit optical-density vectors, as used by
#: `skimage.color.rgb2hed`. These are *reference* directions: step 5 draws and
#: names itself against them, and step 6 will un-mix with them.
#:
#: Eosin is carried even on an IHC slide, and deliberately: it is the vector that
#: should *not* line up with anything on a haematoxylin-and-DAB section, so a
#: reader who sees two arms plus one reference sitting in the empty space between
#: them has seen the check work.
REFERENCE_VECTORS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("haematoxylin", (0.650, 0.704, 0.286)),
    ("dab", (0.268, 0.570, 0.776)),
    ("eosin", (0.072, 0.990, 0.105)),
)

#: The same, by name, for callers that want one vector rather than the list.
REFERENCE_BY_NAME: dict[str, tuple[float, float, float]] = dict(REFERENCE_VECTORS)

#: The two stains a haematoxylin-DAB assay actually carries, in the order every
#: stain matrix in this codebase is written in: counterstain first, chromogen
#: second. The order is not cosmetic - a normaliser that recomposed a source's
#: haematoxylin against a target's DAB would return something that looks like a
#: plausible slide and is the wrong way round.
ASSAY_STAINS: tuple[str, str] = ("haematoxylin", "dab")


# --- small vector helpers -----------------------------------------------------


def unit(vector: np.ndarray) -> np.ndarray:
    """`vector` scaled to length one, or left alone if it has no length."""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def degrees_between(left: np.ndarray, right: np.ndarray) -> float:
    """The angle between two vectors, in degrees, sign-insensitive to neither.

    Not folded onto [0, 90]: a stain vector and its negative are not the same
    direction, and an arm that came out pointing backwards is a fault to see
    rather than a sign to drop.
    """
    a = unit(np.asarray(left, dtype=np.float64))
    b = unit(np.asarray(right, dtype=np.float64))
    cosine = float(np.dot(a, b))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def wrap_angles(angles: np.ndarray) -> np.ndarray:
    """Angles onto (-pi, pi]."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


def angular_centre(angles: np.ndarray) -> float:
    """The mean direction of a set of angles, the way a circle requires.

    A plain mean is wrong here and wrong in a way that hides: `atan2` has a seam at
    pi, and a cloud straddling it would report its two arms at opposite ends of the
    range - a wrap-around artefact indistinguishable from a genuinely wide wedge.
    """
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def nearest_reference(vector: np.ndarray) -> tuple[str, float]:
    """Which published vector a direction lands closest to, and how close."""
    name, triple = min(
        REFERENCE_VECTORS,
        key=lambda entry: degrees_between(vector, np.asarray(entry[1], dtype=np.float64)),
    )
    return name, degrees_between(vector, np.asarray(triple, dtype=np.float64))


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The `q`-th percentile of `values`, each pixel counting `weights` times.

    Used for the arms, and the weight is the pixel's own optical density. That is
    not a nicety - it is the only defensible way to read a direction off a stain
    cloud.

    A pixel's angular uncertainty is roughly its noise divided by its length: the
    sensor's noise is much the same at every intensity, so a pixel carrying 0.15 OD
    has several times the angular scatter of one carrying 0.8, while an unweighted
    percentile gives them the same vote. Since the arms *are* the angular extremes,
    an unweighted estimate hands the two most important numbers in step 5 to the
    least reliable pixels in the tile - and it shows: on the demo's slide the
    unweighted 1st and 99th percentiles put the arms 78 degrees apart against 37 for
    Ruifrok's own pair, an overshoot that shrinks steadily as the weight is applied.

    Weighting rather than raising Macenko's alpha, which would do something
    superficially similar. Alpha discards a fixed *share* of pixels regardless of
    whether they were reliable; this discounts each pixel by how reliable it
    actually is, and keeps alpha at the published value so the method is still his.
    """
    return weighted_percentiles(values, weights, (q,))[0]


def weighted_percentiles(
    values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]
) -> tuple[float, ...]:
    """Several weighted percentiles of the same data, from one sort.

    The sort is the whole cost - a quarter of a million floats, and a screen that
    re-reads the arms as the reader moves the tile asks for a stain matrix many
    times over, each needing two percentiles of the same array. Sorting once for
    both halves the work.
    """
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order].astype(np.float64), 0.0)

    total = float(sorted_weights.sum())
    if total <= 0.0:
        return tuple(float(np.percentile(values, q)) for q in quantiles)

    # Mid-point convention: each pixel's cumulative position is taken at the centre
    # of the interval it occupies, so a single pixel does not shift the answer by
    # its whole weight.
    cumulative = np.cumsum(sorted_weights)
    positions = (cumulative - 0.5 * sorted_weights) / total
    return tuple(float(np.interp(q / 100.0, positions, sorted_values)) for q in quantiles)


# --- the plane the cloud lies in ----------------------------------------------


def orient_plane(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pin down the signs of an eigenvector pair, so the geometry is reproducible.

    An eigenvector is only defined up to sign, so without this a scatter would
    mirror itself between runs on the same slide and the arms would swap places -
    and a normaliser would swap which recovered stain it called the counterstain.

    The first axis is oriented into the positive octant: every optical density is
    positive, so the cloud's dominant direction is too, and a first axis pointing
    away from it would put the whole cloud at negative x.

    The second is oriented so that the axis from haematoxylin towards DAB points
    up. That is a choice, and it is the useful one: it means "higher on the plot"
    always reads as "browner", on every slide, so two slides' scatters can be
    compared by eye - and it fixes which of the two arms is the low one.
    """
    if float(np.sum(first)) < 0.0:
        first = -first

    haematoxylin = np.asarray(REFERENCE_BY_NAME["haematoxylin"], dtype=np.float64)
    dab = np.asarray(REFERENCE_BY_NAME["dab"], dtype=np.float64)
    if float(np.dot(second, dab - haematoxylin)) < 0.0:
        second = -second

    return first, second


def principal_plane(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The plane holding most of a density cloud, and how much of it that is.

    Found by eigendecomposition of `OD' OD` and *not* of the covariance. The cloud
    is a pair of rays emanating from the origin, and the origin - "no stain" - is a
    point the plane has to contain; centring the data first would fit a plane
    through the cloud's middle and move the origin off it.

    As a 3x3 Gram matrix this costs one pass over the pixels rather than an SVD of
    a quarter-million-row matrix, and gives the same right singular vectors.

    Returns the two in-plane axes, oriented by `orient_plane`, and the share of the
    cloud's energy the plane holds - near 1 when the cloud really is planar, which
    is what two absorbers produce.
    """
    samples = np.asarray(vectors, dtype=np.float64)
    gram = samples.T @ samples / float(max(1, samples.shape[0]))
    eigenvalues, eigenvectors = np.linalg.eigh(gram)

    first, second = orient_plane(eigenvectors[:, -1], eigenvectors[:, -2])

    total = float(eigenvalues.sum())
    explained = float(eigenvalues[-1] + eigenvalues[-2]) / total if total > 0.0 else 0.0
    return first, second, explained


def arm_offsets(relative: np.ndarray, weights: np.ndarray, *, alpha: float) -> tuple[float, float]:
    """The two angular extremes of a wedge, as offsets from the angle it centres on.

    Macenko's alpha: the extremes of the wedge are the pure stains, but the literal
    extremes are one pixel each, so a robust percentile stands in for them. The
    weight is `weighted_percentile`'s, for the reason given there.
    """
    low, high = weighted_percentiles(relative, weights, (alpha, 100.0 - alpha))
    return low, high


# --- the stain matrix ---------------------------------------------------------


def stain_matrix_from_offsets(
    first: np.ndarray, second: np.ndarray, centre: float, offsets: tuple[float, float]
) -> np.ndarray:
    """Two in-plane angles turned back into two unit vectors in three dimensions.

    Returned as a 3x2 matrix whose *columns* are the stains, which is the shape
    every decomposition below expects: `OD = matrix @ concentrations`.
    """
    columns = [
        unit(np.cos(centre + offset) * first + np.sin(centre + offset) * second)
        for offset in offsets
    ]
    return np.stack(columns, axis=1)


def order_stains(matrix: np.ndarray) -> np.ndarray:
    """Put the counterstain in column 0 and the chromogen in column 1.

    This is the step that keeps a normaliser from swapping the two stains, and it
    has to be done by *identity* rather than by position, because the position an
    arm comes out in depends on the sign convention of an eigenvector.

    Assignment is by nearest published vector, and it is a matching rather than two
    independent lookups: if both columns land nearest the same reference - which
    happens on a tile carrying essentially one stain - independent lookups would
    put the same stain in both slots and the recomposition would be a mixture of a
    thing with itself. So the pair of assignments with the smaller total error wins.
    With two columns and two slots that is the whole of the Hungarian algorithm.
    """
    haematoxylin = np.asarray(REFERENCE_BY_NAME["haematoxylin"], dtype=np.float64)
    dab = np.asarray(REFERENCE_BY_NAME["dab"], dtype=np.float64)

    left, right = matrix[:, 0], matrix[:, 1]

    straight = degrees_between(left, haematoxylin) + degrees_between(right, dab)
    swapped = degrees_between(right, haematoxylin) + degrees_between(left, dab)

    ordered = matrix if straight <= swapped else matrix[:, ::-1]
    return np.ascontiguousarray(ordered)


def concentrations(od_flat: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """How much of each stain every pixel carries: solve `OD = matrix @ c`.

    `od_flat` is Nx3 and the result is NxK, one non-negative coefficient per stain
    per pixel.

    Least squares and then clipped at zero, which is what Macenko's own method and
    every implementation of it does. The clip matters and is not a tidy-up: a
    negative concentration is a negative amount of dye, and the pixels that produce
    one are those lying just outside the wedge - sensor noise, a third absorber, or
    the arm estimate being a degree out. Letting them through would put negative
    numbers into the percentile that sets the output's scale.

    Solved through the pseudo-inverse of a 3xK rather than by `lstsq` on an Nx3, so
    the cost is one small decomposition plus one matrix multiply over the pixels
    instead of a decomposition of a quarter-million-row system.
    """
    pseudo = np.linalg.pinv(np.asarray(matrix, dtype=np.float64))
    return np.maximum(np.asarray(od_flat, dtype=np.float64) @ pseudo.T, 0.0)


def compose(conc: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Concentrations and a stain matrix back into optical density: `OD = c M'`.

    The exact inverse of `concentrations` when the pixel really is a mixture of
    those stains, and the whole reason the pipeline works in density space: this is
    a *sum*, and a sum is what linear algebra can take apart and put back together.
    """
    return np.asarray(conc, dtype=np.float64) @ np.asarray(matrix, dtype=np.float64).T


def estimate_stain_matrix(
    od_flat: np.ndarray,
    *,
    alpha: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Macenko's stain matrix: the two angular extremes of the density cloud.

    `od_flat` must already be the *admitted* pixels - those carrying enough stain to
    have a direction and enough light for that direction to survive 8-bit
    quantisation. Which pixels those are is a judgement about the acquisition, so it
    belongs to the caller and its thresholds travel in the caller's report; this
    function is the geometry only.

    Returns the 3x2 matrix, counterstain first, and the share of the cloud's energy
    the fitted plane holds - the honesty term on the whole procedure, because the
    method assumes two absorbers and that number is how planar the cloud really is.

    This is the estimator step 5 draws as two arms, and it is one function
    precisely so nothing downstream can disagree with the arms on screen about
    where a stain points on the same tile.
    """
    vectors = np.asarray(od_flat, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("a stain matrix is estimated from an Nx3 array of densities")

    first, second, explained = principal_plane(vectors)
    plane = np.stack([first, second], axis=1)

    points = vectors @ plane
    raw = np.arctan2(points[:, 1], points[:, 0])
    centre = angular_centre(raw)
    relative = wrap_angles(raw - centre)

    # The vote, weighted by how much stain the pixel carries. Read off the mean
    # density rather than the vector's length so it is the same "how much stain"
    # the heatmaps and the beta cut use.
    vote = vectors.mean(axis=1) if weights is None else np.asarray(weights, dtype=np.float64)
    offsets = arm_offsets(relative, vote, alpha=alpha)

    return order_stains(stain_matrix_from_offsets(first, second, centre, offsets)), explained


# --- which directions 8-bit data actually supports ----------------------------

#: Natural log of 10, so the quantisation step below is stated once.
_LN10 = float(np.log(10.0))

def direction_is_stable(
    density: np.ndarray, intensity: np.ndarray, *, tolerance_deg: float
) -> np.ndarray:
    """Which pixels' directions survive 8-bit quantisation. See `ANGULAR_TOLERANCE_DEG`.

    One level of intensity uncertainty at `I` is `1/(I ln10)` of optical density, and
    a perturbation of that size across a vector of length `|OD|` turns it by roughly
    `step / |OD|` radians. The test is that angle against `tolerance_deg`, using the
    *weakest* channel because the weakest channel is where the quantisation is
    coarsest and it only takes one bad component to swing a direction.

    `intensity` must already be floor-clipped, so a black pixel gives a large finite
    step rather than a division by zero - and such a pixel fails this test by a wide
    margin, which is the correct answer for a pixel that recorded no light at all.
    """
    step = (1.0 / _LN10) / np.maximum(intensity, 1e-6).min(axis=-1)
    length = np.linalg.norm(density, axis=-1)
    return step <= np.radians(tolerance_deg) * np.maximum(length, 1e-9)


# --- closing a stain pair into something invertible ---------------------------


def complete_basis(pair: np.ndarray) -> np.ndarray:
    """Two stain directions closed into an invertible 3x3, columns being the stains.

    The third column is the cross product of the first two: the direction *no*
    stain occupies, which by construction absorbs whatever the two cannot explain.
    That is what turns an over-determined 3x2 system into a square one, and a
    square system has an exact inverse - so the three coefficients reproduce the
    optical density they came from with nothing left over.

    Step 6 runs this on both of its bases, and that is the point of it being one
    function. The fixed basis and the per-image estimate differ *only* in where the
    first two columns point; everything after this line - the inverse, the three
    channels, the reconstruction check - is identical arithmetic. A comparison
    between two bases is only honest if the machinery either side of the change is
    the same machinery.

    Raises when the two directions are parallel, because two stains that point the
    same way are one stain and there is no plane for a third axis to be normal to.
    """
    matrix = np.asarray(pair, dtype=np.float64)
    if matrix.shape != (3, 2):
        raise ValueError("a stain pair is a 3x2 matrix whose columns are the stains")

    first = unit(matrix[:, 0])
    second = unit(matrix[:, 1])

    residual = np.cross(first, second)
    norm = float(np.linalg.norm(residual))
    if norm <= 1e-9:
        raise ValueError(
            "the two stain directions are parallel, so they span a line rather than a "
            "plane and no third axis can be normal to them - which means this pair "
            "describes one stain and not two"
        )

    return np.stack([first, second, residual / norm], axis=1)


def unmix(od_flat: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """Optical density onto a stain basis: one coefficient per column, exactly.

    `inverse` is the inverse of a 3x3 stain matrix, taken once by the caller
    because a step that un-mixes a quarter of a million pixels should invert a
    3x3 once and not per pixel. Nx3 in, Nx3 out.

    **Not clipped at zero.** The system is square, so the coefficients *are* the
    pixel: clipping one would break the reconstruction that every check on step 6's
    screen depends on. Negative coefficients happen - a pixel whose colour sits
    outside the cone the two stains span has to be described by a negative amount
    of one of them - and step 6 reports how many rather than hiding them.
    """
    return np.asarray(od_flat, dtype=np.float64) @ np.asarray(inverse, dtype=np.float64).T


def recompose(coefficients: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Coefficients and a stain matrix back into optical density: `OD = c M'`.

    The exact inverse of `unmix` for a square basis, and the check step 6 runs to
    show that separating the stains did not lose anything: un-mix, put back
    together, and compare with what went in.
    """
    return np.asarray(coefficients, dtype=np.float64) @ np.asarray(matrix, dtype=np.float64).T


# --- the fixed basis the measurement branch will use --------------------------


def ruifrok_hdab() -> np.ndarray:
    """Ruifrok's 3x3 haematoxylin-DAB matrix, columns being the stains.

    The third column is the cross product of the first two: the direction no stain
    occupies, which by construction absorbs whatever the two stains cannot explain.
    That makes the matrix *invertible*, and therefore the decomposition **exact** -
    any optical density whatsoever is reproduced by its three coefficients, with no
    residual left over.

    Two consumers, and they need the same constant for the same reason.

      step 5   draws its arms against this basis, so a reader can see how far the
               tile's own cloud sits from where the published stains point.
      step 6   will deconvolve with it for real, on both arms of the fork - which
               is what makes the DAB scale and the H channel mean the same thing
               on every slide in a study.

    **Fixed, and that is the whole point.** A per-image estimate rescales itself to
    whatever it is given, so "0.4 DAB" would mean a different amount of stain on
    every slide in a study. These numbers are the same on every slide anyone has
    ever published them against, which is what makes a measurement comparable -
    and it is exactly the property a per-slide estimate destroys, which is why the
    measurement branch never gets one.

    This is what `skimage.color.rgb2hed` inverts.
    """
    haematoxylin = np.asarray(REFERENCE_BY_NAME["haematoxylin"], dtype=np.float64)
    dab = np.asarray(REFERENCE_BY_NAME["dab"], dtype=np.float64)
    # Closed by `complete_basis`, the same function step 6 closes its per-image
    # estimate with, so the published basis and the estimated one are built the
    # same way and differ only in where their first two columns point.
    return complete_basis(np.stack([haematoxylin, dab], axis=1))


#: The matrix itself, built once. Its inverse too, since every caller needs it.
RUIFROK_HDAB: np.ndarray = ruifrok_hdab()
RUIFROK_INVERSE: np.ndarray = np.linalg.inv(RUIFROK_HDAB)

#: Column order of `RUIFROK_HDAB`, so a caller indexes by name rather than by 1.
RUIFROK_CHANNELS: tuple[str, str, str] = ("haematoxylin", "dab", "residual")


def ruifrok_he() -> np.ndarray:
    """Ruifrok's 3x3 haematoxylin-eosin matrix, closed the same way as the H-DAB one.

    **For an H&E slide, never the H-DAB basis (P-21).** Step 11's H&E reference count
    used to un-mix the H&E with the H-DAB matrix, and eosin has no column there: an
    eosin pixel came out as 0.668 haematoxylin, 0.132 DAB and -0.625 residual. Every
    pink pixel added false haematoxylin, so the reference count - the figure the IHC
    shortfall, the DENOMINATOR INCOMPLETE caveat and the typing trust check are all
    measured against - was taken on the wrong picture.
    """
    haematoxylin = np.asarray(REFERENCE_BY_NAME["haematoxylin"], dtype=np.float64)
    eosin = np.asarray(REFERENCE_BY_NAME["eosin"], dtype=np.float64)
    return complete_basis(np.stack([haematoxylin, eosin], axis=1))


RUIFROK_HE: np.ndarray = ruifrok_he()
RUIFROK_HE_INVERSE: np.ndarray = np.linalg.inv(RUIFROK_HE)


def fixed_concentrations(od_flat: np.ndarray) -> np.ndarray:
    """Decompose optical density onto Ruifrok's fixed basis. Exact, and not clipped.

    Nx3 in, Nx3 out: haematoxylin, DAB, and the residual that is neither.

    **Not clipped at zero**, unlike `concentrations`, and the difference matters.
    That function solves an over-determined two-stain system where a negative
    coefficient means "this pixel is outside the wedge" and clipping is the honest
    repair. This one solves a square system exactly, so the coefficients *are* the
    pixel - clipping any of them would mean the three no longer reconstruct what
    they came from, and every use of this function depends on that reconstruction
    being exact.
    """
    return unmix(od_flat, RUIFROK_INVERSE)
