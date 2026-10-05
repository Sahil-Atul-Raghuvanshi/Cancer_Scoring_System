"""Step 5's algorithm: colour turned into how much stain.

The transform itself is one line, and `app.common.imaging.optical_density`
already holds it because step 4 needed it first - a white point is only
meaningful if you can say what the glass measures once you divide by it. So this
module is not about computing `-log10(I / I0)`. It is about the two things that
have to be true for that line to be worth anything, and about saying out loud
where it stops being a measurement.

**The claim.** Beer-Lambert: absorbance is proportional to the concentration of
the absorbing substance. Transmitted intensity is *multiplicative* - two stains
stacked multiply their transmissions - so intensity is not proportional to
anything, and no matrix can un-mix it. The logarithm turns the product into a
sum, and a sum is what linear algebra can invert. That single fact is what makes
step 6 valid, and everything else in this pipeline downstream of step 6 rests on
it.

**The two things this module measures rather than asserts.**

  the arms      In optical density space one stain at any concentration lies on
                one ray from the origin: `OD = c * v`, with `v` fixed and `c` the
                concentration. Two stains therefore make two rays, and a mixed
                slide fills the wedge between them. That picture is the whole of
                colour deconvolution, and it is either there in the data or it is
                not - so `build_cloud` projects the tile's densities onto the
                plane that holds most of them and reports where the two arms sit,
                how far apart they are, and how far each lands from Ruifrok &
                Johnston's published vectors for haematoxylin and DAB.

  the linearity `measure_additivity` tests the claim directly. Take one arm's
                pixels, sort them by how much stain they carry, and ask whether
                the *direction* is the same for the faintest and the darkest. In
                density space it must be, because `c` only scales the ray. In
                intensity space it must not be, because `I = I0 * 10^(-c v)`
                curves. Two numbers, from the reader's own slide, in degrees.

**Where this stops being a measurement.** Three places, all reported by
`measure_limits` rather than quietly repaired:

  the floor     A pixel at intensity zero has infinite density. `optical_density`
                clips the observed intensity from below, so such a pixel gets a
                large finite number that is a *lower bound* and not a reading.
  above I0      A pixel brighter than the white point has negative density -
                negative stain, which does not exist. It means I0 is slightly low
                here, and on a tile of tissue that is worth knowing rather than
                clamping away.
  below beta    A pixel with almost no stain has a density whose *direction* is
                pure noise, because dividing a tiny vector by its tiny length
                amplifies the quantisation. Those pixels are excluded from the
                cloud, exactly as Macenko's method does, and the share excluded
                is reported.

Nothing here is a filter and nothing here is an enhancement. The transform is a
change of units with an exact inverse, `I = I0 * 10^(-OD)`, and `measure_limits`
performs that inverse and reports the residual in intensity levels - which is
the honest way to show that this step threw nothing away.

Ruifrok AC, Johnston DA. Quantification of histochemical staining by colour
deconvolution. Anal Quant Cytol Histol 23(4):291-299 (2001).
Macenko M et al. A method for normalizing histology slides for quantitative
analysis. ISBI 2009:1107-1110.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.common.imaging import optical_density

# The published stain vectors and the vector geometry this step reads its arms
# with all live in `app.common.stains`, because step 6 will deconvolve with the
# same constants. Two copies would be two chances for the arms drawn on this
# screen and the matrix used on the next to disagree about where haematoxylin
# points. Imported into this module's
# namespace rather than referenced through it, because they read as part of the
# step's own vocabulary and every caller of this module already treats them so.
from app.common.stains import (
    REFERENCE_VECTORS,
    angular_centre,
    arm_offsets,
    degrees_between,
    direction_is_stable,
    principal_plane,
    unit,
    wrap_angles,
)

#: Below this mean optical density a pixel is treated as carrying no stain and is
#: kept out of the point cloud. Macenko's beta, at his value.
#:
#: Not a cosmetic cutoff. The cloud is about *direction*, and the direction of a
#: near-zero vector is set by whichever way the last quantisation step rounded.
#: Including those pixels would fill the middle of the wedge with noise that looks
#: exactly like a mixture of the two stains, which is the one thing the picture
#: must not fake.
TRANSPARENT_OD = 0.15

#: Which tail of the angular distribution is taken as an arm. Macenko's alpha.
#: The extremes of the wedge are the pure stains, but the literal extremes are one
#: pixel each, so a robust percentile stands in for them.
ARM_PERCENTILE = 1.0

#: How far quantisation may move a pixel's *direction* before it is kept out of the
#: cloud, in degrees. The second admission test, and it is the mirror image of beta
#: rather than an extra safeguard.
#:
#: Beta excludes pixels with too little signal for a direction. This excludes pixels
#: with too little *light*, and they are a different population entirely - the very
#: dark ones, where a stain has absorbed almost everything in one channel. One 8-bit
#: level at recorded intensity `I` moves that channel's density by `1/(I ln10)`:
#: 0.014 at I=32, 0.054 at I=8, and 0.43 at I=1. A pixel with a channel at 1 has a
#: density whose direction is set by which way the last level rounded, and because
#: such a pixel is also *long* - a large density magnitude - it lands at the extreme
#: of the angular distribution, which is precisely where the arms are read from.
#:
#: Measured on the demo's slide: without this test the pixels beyond the low arm have
#: a median green channel of 1, and the arm they set lands nearest *eosin* on a slide
#: stained with haematoxylin and DAB. With it, the same arm lands on haematoxylin at
#: 11-20 degrees on every tile tried and the wedge narrows from 68 degrees to 63, for
#: 3% of the tile - about a tenth of what would otherwise have been plotted.
#:
#: 1.5 degrees is what 8-bit data can support rather than a preference. At the beta
#: boundary it demands a weakest channel of about 64 levels, and for a dark pixel at
#: 1.5 OD about 11 - both comfortably inside what a working scan records, while the
#: single-digit intensities that caused the problem are excluded. Tightening it to
#: half a degree empties the cloud, which is the honest sign that the limit is the
#: data's and not the method's.
ANGULAR_TOLERANCE_DEG = 1.5

#: Bins in the reported density histogram, and in the angular histogram under the
#: scatter. The angular one is the cloud's shape counted rather than drawn - where
#: the two arms are, and how the tile's pixels distribute between them.
DENSITY_BINS = 64
ANGLE_BINS = 72

#: Concentration bins in the additivity check. Five is enough to see a trend and
#: few enough that each holds a fifth of one arm's pixels.
DRIFT_BINS = 5

#: Share of an arm's angular span, measured from the arm inwards, whose pixels are
#: treated as that stain alone for the additivity check. A quarter: wide enough to
#: hold pixels at every concentration, narrow enough that the far arm's colour is
#: not mixed into the direction being tested.
ARM_CORE_SHARE = 0.25

#: Separation, in degrees, at or above which the cloud is called two-armed. Below
#: it the two reported arms are the two tails of a single lobe, which is what a
#: tile with only a counterstain on it looks like - a fact to report, not to hide.
#: Ruifrok's own haematoxylin and DAB vectors sit 43 degrees apart, so a real pair
#: of arms has room to be much narrower than the references and still be a pair.
TWO_ARM_SEPARATION = 12.0

#: Fewest pixels above beta before a cloud is built at all. Below this the
#: percentiles that define the arms are being read off a handful of points.
MIN_CLOUD_PIXELS = 500


class DensityError(ValueError):
    """The tile cannot be turned into a density - and why, in words."""


# --- where the densities landed ----------------------------------------------


@dataclass(frozen=True)
class DensityStats:
    """The distribution of the tile's densities, per channel and in total.

    `mean` is the average of the three channels, and it is the scalar this whole
    step uses for "how much stain is in this pixel" - the heatmap is drawn from
    it and the transparency test is defined on it. One scalar rather than two so
    the picture on screen and the pixels excluded from the cloud cannot disagree
    about which pixels are faint.
    """

    #: Median, 99th percentile and maximum of each channel's density.
    median: tuple[float, float, float]
    p99: tuple[float, float, float]
    maximum: tuple[float, float, float]

    #: The same three, for the mean-across-channels density.
    mean_median: float
    mean_p99: float
    mean_maximum: float
    mean_minimum: float

    #: Histogram of the mean density, and the range it spans.
    histogram: tuple[int, ...]
    histogram_low: float
    histogram_high: float


def measure_stats(density: np.ndarray) -> DensityStats:
    """Percentiles and a histogram of the tile's densities.

    Percentiles rather than a mean and a standard deviation: a density
    distribution over stained tissue is strongly skewed - most of a tile is
    counterstain and a little of it is a strong positive - and a standard
    deviation would describe a shape this is not.
    """
    flat = density.reshape(-1, 3).astype(np.float64)
    mean_od = flat.mean(axis=1)

    median = np.percentile(flat, 50.0, axis=0)
    p99 = np.percentile(flat, 99.0, axis=0)
    maximum = flat.max(axis=0)

    # The histogram's range is read off the data rather than fixed, because the
    # interesting spread is a few tenths of an OD and a fixed 0-3 axis would put
    # the whole tile in two bins. The bounds travel with the counts so the stretch
    # is stated rather than implied. Trimmed at both ends: one saturated pixel
    # would otherwise set the axis for the other quarter of a million.
    low = min(0.0, float(np.percentile(mean_od, 0.2)))
    high = float(np.percentile(mean_od, 99.8))
    if high - low < 1e-6:
        high = low + 1e-6

    counts, _ = np.histogram(mean_od, bins=DENSITY_BINS, range=(low, high))

    return DensityStats(
        median=(float(median[0]), float(median[1]), float(median[2])),
        p99=(float(p99[0]), float(p99[1]), float(p99[2])),
        maximum=(float(maximum[0]), float(maximum[1]), float(maximum[2])),
        mean_median=float(np.median(mean_od)),
        mean_p99=float(np.percentile(mean_od, 99.0)),
        mean_maximum=float(mean_od.max()),
        mean_minimum=float(mean_od.min()),
        histogram=tuple(int(value) for value in counts),
        histogram_low=low,
        histogram_high=high,
    )


# --- where the transform stops being a measurement ---------------------------


@dataclass(frozen=True)
class Limits:
    """The three places a density is not a reading, and the inverse that proves the rest is.

    Every field here is a share of the tile or an error in intensity levels, so
    all of them can be read as "how much of this picture should I not trust".
    """

    #: Share of pixels where a channel hit the intensity floor. Their densities are
    #: lower bounds, not readings - the sensor recorded no light at all.
    floor_share: float
    #: Share of pixels brighter than I0, whose mean density is therefore negative.
    negative_share: float
    #: The most negative mean density in the tile - how far past I0 the tile goes.
    negative_worst: float
    #: Share of pixels below beta, excluded from the cloud as having no direction.
    transparent_share: float
    beta: float

    #: The exact inverse, `I = I0 * 10^(-OD)`, applied back to the density and
    #: compared with the (floor-clipped) input, in intensity levels. This step is a
    #: change of units and not a filter, and this is the number that says so.
    roundtrip_max: float
    roundtrip_mean: float
    #: Share of pixels the inverse returns to within half a level - i.e. exactly,
    #: at 8-bit precision.
    roundtrip_exact_share: float


def measure_limits(
    rgb: np.ndarray,
    density: np.ndarray,
    white: np.ndarray | tuple[float, float, float],
    *,
    floor: float,
    beta: float = TRANSPARENT_OD,
) -> Limits:
    """Count the pixels where the transform is not a measurement, and invert it.

    The round trip is measured against the *clipped* input, not the raw one, and
    that is the honest comparison: `optical_density` clips before taking the
    logarithm, so a black pixel's density genuinely does not encode the zero it
    came from. That loss is reported separately as `floor_share` rather than
    hidden inside a round-trip residual, because it is a different kind of fact -
    one is the arithmetic's precision and the other is the sensor running out of
    range.
    """
    observed = np.maximum(rgb.astype(np.float32), np.float32(floor))
    reference = np.asarray(white, dtype=np.float32)

    total = int(observed.shape[0] * observed.shape[1])
    if total == 0:
        raise DensityError("an empty tile has no densities to measure")

    mean_od = density.mean(axis=-1)

    restored = reference * np.float32(10.0) ** (-density)
    error = np.abs(restored - observed)

    return Limits(
        floor_share=float(np.mean(np.any(rgb.astype(np.float32) < floor, axis=-1))),
        negative_share=float(np.mean(mean_od < 0.0)),
        negative_worst=float(mean_od.min()),
        transparent_share=float(np.mean(mean_od < beta)),
        beta=beta,
        roundtrip_max=float(error.max()),
        roundtrip_mean=float(error.mean()),
        roundtrip_exact_share=float(np.mean(np.all(error < 0.5, axis=-1))),
    )


# --- the point cloud, and the two arms in it ---------------------------------


@dataclass(frozen=True)
class Arm:
    """One edge of the wedge: a direction the data says a stain lies along."""

    #: Angle in the plot plane, in radians.
    angle: float
    #: The direction in three dimensions, as a unit optical-density vector.
    vector: tuple[float, float, float]
    #: Share of the plotted pixels that sit nearer this arm than the other.
    share: float
    #: Where to draw the arm's tip, in plot coordinates.
    plot: tuple[float, float]
    #: Which reference vector it lands closest to, and how close, in degrees.
    nearest: str
    degrees_from_nearest: float


@dataclass(frozen=True)
class ReferenceProjection:
    """A published stain vector, shown on the same axes as the arms."""

    name: str
    vector: tuple[float, float, float]
    plot: tuple[float, float]
    #: How much of this unit vector points out of the plotted plane. A reference
    #: drawn near an arm but with a large value here is a shadow, not a match -
    #: which is why the number travels with the drawing.
    out_of_plane: float
    #: Degrees from the nearer of the two arms.
    degrees_from_arm: float


@dataclass(frozen=True)
class Cloud:
    """The tile's densities as a point cloud, and the geometry read off it.

    `basis` is the plane the cloud is drawn on: the two directions that hold most
    of its energy. Found by eigendecomposition of `OD' OD` and *not* of the
    covariance - the cloud is a pair of rays emanating from the origin, and the
    origin is "no stain", a point the plane has to contain. Centring the data
    first would fit a plane through the cloud's middle instead, and the origin
    would no longer be at the origin.
    """

    basis: tuple[tuple[float, float, float], tuple[float, float, float]]
    #: Share of the cloud's energy the plane holds. Near 1 means the point cloud
    #: really is planar, which is what two stains produce; well below it means
    #: three or more absorbers, and the flat drawing is a projection to distrust.
    explained: float
    plotted: int
    #: Share of the tile admitted, and the two ways a pixel was turned away.
    admitted_share: float
    #: Below beta: too little stain for a direction.
    faint_share: float
    #: Direction not resolvable at 8 bits: too little light. See `direction_is_stable`.
    unstable_share: float
    tolerance_deg: float

    #: The plotted box, in plane coordinates, equal-scaled on both axes so the
    #: angles on screen are the real angles. Contains the origin by construction.
    x_low: float
    x_high: float
    y_low: float
    y_high: float

    arms: tuple[Arm, ...]
    references: tuple[ReferenceProjection, ...]
    #: Degrees between the two arms, in three dimensions.
    separation: float
    #: True when the separation clears `TWO_ARM_SEPARATION` - i.e. the picture
    #: really does show two stains and not one lobe's two tails.
    two_armed: bool

    #: Counts by angle across the wedge, and the range they span. This is the
    #: cloud's shape counted rather than drawn: where the tile holds regions of
    #: near-pure stain the distribution has a mode at each arm, and where it is all
    #: mixtures it has one broad hump between them.
    angles: tuple[int, ...]
    angle_low: float
    angle_high: float
    #: The angle the wedge is measured about. `point_angles` are relative to this,
    #: because atan2 has a seam at pi and a cloud straddling it would otherwise
    #: report its two arms at opposite ends of the range.
    angle_centre: float

    #: Degrees between Ruifrok's own haematoxylin and DAB vectors - the same
    #: measurement as `separation`, on the published pair. Reported so the wedge's
    #: width can be read against something rather than in the abstract: a wedge much
    #: wider than this is a sign of a third absorber or of the extremes reaching
    #: into noise, and a much narrower one that the two stains overlap on this slide.
    reference_separation: float

    #: The projected points, kept for the renderer. Not serialised.
    points: np.ndarray
    #: Angle of every plotted point, kept so the additivity check partitions the
    #: cloud exactly the way the picture does.
    point_angles: np.ndarray
    #: The density vectors the points came from, in the same order.
    vectors: np.ndarray
    #: Index into the tile of each plotted pixel, so the intensity of the same
    #: pixels can be looked up without recomputing the mask.
    index: np.ndarray


def build_cloud(
    density: np.ndarray,
    intensity: np.ndarray,
    *,
    beta: float = TRANSPARENT_OD,
    alpha: float = ARM_PERCENTILE,
    tolerance_deg: float = ANGULAR_TOLERANCE_DEG,
) -> Cloud | None:
    """Project the tile's densities onto their own plane and find the two arms.

    Two admission tests, and they exclude opposite populations for one reason. A
    direction is only meaningful when the vector is long enough to have one and
    short enough to be resolved: `beta` drops the pixels with too little stain,
    `direction_is_stable` drops the ones so dark that a single 8-bit level would
    swing them. Both matter because the arms are read from the angular *extremes*,
    which is exactly where unreliable pixels end up.

    Returns None when too few pixels survive. That is a real state - a tile of pale
    stroma has no arms to find - and it is reported rather than approximated,
    because a cloud built from two hundred points would produce two confident arms
    that are noise.

    The method is Macenko's, and stops one step short of his. He estimates the
    stain vectors this way *and then measures with them*; this step estimates them
    only to show they are there. Step 6 measures with Ruifrok's fixed vectors
    instead, because a per-image estimate gives every slide its own scale and the
    point of the pipeline is a number two slides can be compared on.
    """
    flat = density.reshape(-1, 3).astype(np.float64)
    observed = intensity.reshape(-1, 3).astype(np.float64)
    mean_od = flat.mean(axis=1)

    admitted = (mean_od >= beta) & direction_is_stable(
        flat, observed, tolerance_deg=tolerance_deg
    )
    keep = np.nonzero(admitted)[0]
    if keep.size < MIN_CLOUD_PIXELS:
        return None

    vectors = flat[keep]

    # `OD' OD` and not a covariance - see the class docstring. `principal_plane`
    # is `app.common.stains`', so the plane the arms are read off is the same
    # geometry step 6 un-mixes against: one function means the arms drawn here
    # and the matrix used there cannot disagree.
    first, second, explained = principal_plane(vectors)
    plane = np.stack([first, second], axis=1)

    points = vectors @ plane
    raw_angles = np.arctan2(points[:, 1], points[:, 0])

    # Each pixel's angular vote, weighted by how much stain it carries - see
    # `weighted_percentile`. Read off the mean density rather than the vector's
    # length so it is the same "how much stain" the heatmap and beta both use.
    weights = vectors.mean(axis=1)

    # Re-centred before the percentiles are taken. atan2 has a seam at pi, and a
    # cloud that happens to straddle it would otherwise report its two arms at
    # opposite ends of the range - a wrap-around artefact indistinguishable from a
    # genuinely wide wedge.
    centre = angular_centre(raw_angles)
    relative = wrap_angles(raw_angles - centre)

    low_offset, high_offset = arm_offsets(relative, weights, alpha=alpha)

    def direction(offset: float) -> np.ndarray:
        angle = centre + offset
        return unit(np.cos(angle) * first + np.sin(angle) * second)

    low_vector = direction(low_offset)
    high_vector = direction(high_offset)
    separation = degrees_between(low_vector, high_vector)

    # Nearer arm decides which stain a pixel is mostly made of, and the split is
    # the midpoint in angle - the same partition the picture shows.
    midpoint = (low_offset + high_offset) / 2.0
    low_share = float(np.mean(relative < midpoint))

    # The plotted box: tight on the data, then expanded to contain the origin and
    # squared up so both axes carry the same scale. Equal scaling is not
    # presentation - an unequal aspect changes every angle on the plot, and the
    # angles are the content.
    x_low = min(0.0, float(np.percentile(points[:, 0], 0.2)))
    x_high = max(0.0, float(np.percentile(points[:, 0], 99.8)))
    y_low = min(0.0, float(np.percentile(points[:, 1], 0.2)))
    y_high = max(0.0, float(np.percentile(points[:, 1], 99.8)))

    radius = max(x_high, abs(x_low), y_high, abs(y_low), 1e-3)
    x_span = max(x_high - x_low, 1e-6)
    y_span = max(y_high - y_low, 1e-6)
    span = max(x_span, y_span) * 1.08

    x_centre, y_centre = (x_high + x_low) / 2.0, (y_high + y_low) / 2.0
    x_low, x_high = x_centre - span / 2.0, x_centre + span / 2.0
    y_low, y_high = y_centre - span / 2.0, y_centre + span / 2.0

    def plot_point(offset: float) -> tuple[float, float]:
        angle = centre + offset
        return (float(radius * np.cos(angle)), float(radius * np.sin(angle)))

    references = _project_references(
        plane, arms=(low_vector, high_vector), radius=radius
    )

    def arm(offset: float, vector: np.ndarray, share: float) -> Arm:
        closest = min(
            REFERENCE_VECTORS,
            key=lambda entry: degrees_between(vector, np.asarray(entry[1], dtype=np.float64)),
        )
        return Arm(
            angle=centre + offset,
            vector=(float(vector[0]), float(vector[1]), float(vector[2])),
            share=share,
            plot=plot_point(offset),
            nearest=closest[0],
            degrees_from_nearest=degrees_between(
                vector, np.asarray(closest[1], dtype=np.float64)
            ),
        )

    angle_counts, _ = np.histogram(
        relative, bins=ANGLE_BINS, range=(low_offset, high_offset)
    )

    return Cloud(
        basis=(
            (float(first[0]), float(first[1]), float(first[2])),
            (float(second[0]), float(second[1]), float(second[2])),
        ),
        explained=explained,
        plotted=int(vectors.shape[0]),
        admitted_share=float(admitted.mean()),
        faint_share=float(np.mean(mean_od < beta)),
        unstable_share=float(
            np.mean(
                (mean_od >= beta)
                & ~direction_is_stable(flat, observed, tolerance_deg=tolerance_deg)
            )
        ),
        tolerance_deg=tolerance_deg,
        x_low=x_low,
        x_high=x_high,
        y_low=y_low,
        y_high=y_high,
        arms=(
            arm(low_offset, low_vector, low_share),
            arm(high_offset, high_vector, 1.0 - low_share),
        ),
        references=references,
        separation=separation,
        two_armed=separation >= TWO_ARM_SEPARATION,
        angles=tuple(int(value) for value in angle_counts),
        angle_low=centre + low_offset,
        angle_high=centre + high_offset,
        angle_centre=centre,
        reference_separation=degrees_between(
            np.asarray(REFERENCE_VECTORS[0][1], dtype=np.float64),
            np.asarray(REFERENCE_VECTORS[1][1], dtype=np.float64),
        ),
        points=points,
        point_angles=relative,
        vectors=vectors,
        index=keep,
    )


def _project_references(
    plane: np.ndarray, *, arms: tuple[np.ndarray, np.ndarray], radius: float
) -> tuple[ReferenceProjection, ...]:
    """Ruifrok's vectors on the cloud's own plane, with the honesty term.

    A unit vector projected onto a plane gets shorter, and how much shorter is
    exactly how much of it points out of the picture. Drawing the shadow without
    that number would let a reference vector look like it lands on an arm when it
    is really some distance behind the page, so `out_of_plane` travels with every
    projection and the arm comparison is done in three dimensions.
    """
    projections: list[ReferenceProjection] = []

    for name, triple in REFERENCE_VECTORS:
        vector = unit(np.asarray(triple, dtype=np.float64))
        flat = vector @ plane
        in_plane = float(np.linalg.norm(flat))
        out_of_plane = float(np.sqrt(max(0.0, 1.0 - in_plane**2)))

        # Drawn at the same length as the arms so the picture compares directions
        # and not magnitudes: a reference vector has no concentration.
        scaled = flat / in_plane * radius if in_plane > 1e-9 else flat

        projections.append(
            ReferenceProjection(
                name=name,
                vector=(float(vector[0]), float(vector[1]), float(vector[2])),
                plot=(float(scaled[0]), float(scaled[1])),
                out_of_plane=out_of_plane,
                degrees_from_arm=min(degrees_between(vector, arm) for arm in arms),
            )
        )

    return tuple(projections)


# --- the Beer-Lambert claim, tested on the reader's own tile ------------------


@dataclass(frozen=True)
class DriftBin:
    """One concentration band of one arm, and where it points in each space."""

    #: Median mean-density of the band - how much stain these pixels carry.
    density: float
    pixels: int
    #: Degrees from the faintest band's direction, in optical density space.
    od_degrees: float
    #: Degrees from the faintest band's direction, in intensity space.
    intensity_degrees: float


@dataclass(frozen=True)
class Additivity:
    """Beer-Lambert, measured rather than asserted.

    One stain at two concentrations is one direction in density space and two
    directions in intensity space. That is the entire reason this step exists, and
    it is a testable statement about the reader's own tile, so it is tested.

    Reported for the arm that holds more of the cloud, because the check needs
    pixels at every concentration and the minor arm may only have them at one.
    Which arm that was is named, so the reader is not left guessing.
    """

    arm: str
    bins: tuple[DriftBin, ...]
    #: Degrees the direction turns between the faintest and darkest band, in
    #: density space. Should be near zero: that is the linearity step 6 needs.
    od_drift: float
    #: The same in intensity space, where `I = I0 * 10^(-c v)` curves. Should be
    #: visibly larger, and how much larger depends on how dark the tile gets.
    intensity_drift: float
    pixels: int


def measure_additivity(
    cloud: Cloud,
    rgb: np.ndarray,
    white: np.ndarray | tuple[float, float, float],
) -> Additivity | None:
    """Test whether one stain keeps one direction as it gets darker.

    Two design decisions here, and both exist to stop the test proving itself.

    **The intensity-space quantity is `I0 - I`, the light the stain removed, not
    `I`.** `I` is dominated by the illumination - all tissue is some shade of
    bright - so the angle between two `I` vectors is small for reasons that have
    nothing to do with stain, and intensity space would look better than it is.
    `I0 - I` is zero where there is no stain and grows with it, which makes it the
    intensity-space counterpart of a density and therefore the fair comparison.

    **The pixels are selected by their direction in *intensity* space, never in
    density space.** This is the part that matters. The check needs one stain's
    pixels rather than the whole wedge of mixtures, so some angular selection has
    to happen - and selecting on density angle would cap how far the density
    direction could then be found to move. The measurement would be partly a
    consequence of the selection, and the conclusion this step wants to reach
    would be baked into its own method.

    So the selection is made in the space the test is trying to discredit. That
    caps the *intensity* drift instead, which biases the whole comparison against
    the conclusion: whatever gap survives is a floor on the real one. The same
    logic as `I0 - I` above, applied to the harder half of the problem.

    Binning is by density magnitude, which is neither space's direction - it is
    "how much stain", the variable the check is a function of - so it favours
    neither.

    A residual confound is left, honestly: darker pixels in a section are also
    likely to be purer, so composition changes a little along the bins. It changes
    identically for both spaces, because both are measured on exactly the same
    pixels in exactly the same bins, so it inflates both numbers and does not
    touch the comparison between them. That is why the report carries the two
    drifts side by side rather than the density one alone.

    Returns None when the selected core has too few pixels to bin.
    """
    if not cloud.arms:
        return None

    flat_rgb = rgb.reshape(-1, 3).astype(np.float64)[cloud.index]
    reference = np.asarray(white, dtype=np.float64)
    if reference.ndim == 3:
        reference = reference.reshape(-1, 3)[cloud.index]

    removed_all = np.maximum(reference - flat_rgb, 0.0)

    # The intensity-space plane and angle, found exactly as `build_cloud` finds the
    # density one - so "the extreme quarter of the wedge" means the same thing in
    # both spaces and only the space differs.
    first, second, _ = principal_plane(removed_all)

    intensity_plane = removed_all @ np.stack([first, second], axis=1)
    raw = np.arctan2(intensity_plane[:, 1], intensity_plane[:, 0])
    centre = angular_centre(raw)
    relative = wrap_angles(raw - centre)

    # Which end of the intensity wedge to take: the one holding more pixels, which
    # is the same rule the density cloud's dominant arm follows.
    midpoint = float(np.median(relative))
    towards_high = float(np.mean(relative > midpoint)) >= 0.5

    if towards_high:
        edge = float(np.percentile(relative, 100.0 * (1.0 - ARM_CORE_SHARE)))
        core = relative >= edge
    else:
        edge = float(np.percentile(relative, 100.0 * ARM_CORE_SHARE))
        core = relative <= edge

    if int(core.sum()) < DRIFT_BINS * 40:
        return None

    vectors = cloud.vectors[core]
    removed = removed_all[core]

    magnitude = vectors.mean(axis=1)
    order = np.argsort(magnitude)
    groups = np.array_split(order, DRIFT_BINS)

    # Named for whichever density arm the selected pixels actually sit nearer, so
    # the label describes the stain rather than the machinery that found it. The
    # comparison is in absolute angle, hence `angle_centre`: `point_angles` are
    # relative to it.
    selected_angle = cloud.angle_centre + float(np.median(cloud.point_angles[core]))
    dominant = min(cloud.arms, key=lambda entry: abs(entry.angle - selected_angle))

    bins: list[DriftBin] = []
    od_first: np.ndarray | None = None
    intensity_first: np.ndarray | None = None

    for group in groups:
        if group.size == 0:
            continue

        od_direction = unit(vectors[group].mean(axis=0))
        intensity_direction = unit(removed[group].mean(axis=0))

        if od_first is None:
            od_first, intensity_first = od_direction, intensity_direction

        assert intensity_first is not None
        bins.append(
            DriftBin(
                density=float(np.median(magnitude[group])),
                pixels=int(group.size),
                od_degrees=degrees_between(od_direction, od_first),
                intensity_degrees=degrees_between(intensity_direction, intensity_first),
            )
        )

    if len(bins) < 2:
        return None

    return Additivity(
        arm=dominant.nearest,
        bins=tuple(bins),
        od_drift=bins[-1].od_degrees,
        intensity_drift=bins[-1].intensity_degrees,
        pixels=int(core.sum()),
    )


# --- the whole step ----------------------------------------------------------


@dataclass
class Density:
    """Everything step 5 produced for one tile."""

    #: The tile as it was read, HxWx3 uint8.
    rgb: np.ndarray
    #: Its optical density, HxWx3 float32. Can be negative - see `Limits`.
    od: np.ndarray
    #: The white point divided out: one triple, or a field of the tile's shape.
    white: np.ndarray | tuple[float, float, float]

    stats: DensityStats
    limits: Limits
    cloud: Cloud | None
    additivity: Additivity | None

    mpp: float
    floor: float
    tolerance_deg: float

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.rgb.shape[0]), int(self.rgb.shape[1]))

    @property
    def mean_od(self) -> np.ndarray:
        """The scalar "how much stain" per pixel: the mean of the three channels."""
        return self.od.mean(axis=-1)


def transform(
    *,
    rgb: np.ndarray,
    white: np.ndarray | tuple[float, float, float],
    mpp: float,
    floor: float,
    beta: float = TRANSPARENT_OD,
    alpha: float = ARM_PERCENTILE,
    tolerance_deg: float = ANGULAR_TOLERANCE_DEG,
) -> Density:
    """Run step 5 end to end on one tile.

    The order matters in one place: the cloud has to be built before the
    additivity check, because the check tests one *arm* and the arms are what the
    cloud found. Everything else is independent and reads off the same density
    array, which is computed exactly once - `optical_density` is cheap but it is
    not free at a quarter of a million pixels, and two copies of it would be two
    chances for the heatmap and the scatter to be pictures of different numbers.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise DensityError("optical density needs an RGB tile, not a single channel")

    reference = np.asarray(white, dtype=np.float32)
    if reference.ndim == 3 and reference.shape[:2] != rgb.shape[:2]:
        raise DensityError(
            f"the white field is {reference.shape[:2]} and the tile is {rgb.shape[:2]}; a "
            "position-dependent I0 has to be evaluated on the tile's own grid"
        )
    if float(np.min(reference)) <= 0.0:
        raise DensityError(
            "the white point has a zero or negative channel, so I / I0 is not defined. "
            "Step 4 found no glass to measure - fix that before asking for a density"
        )

    observed = np.maximum(rgb.astype(np.float32), np.float32(floor))
    density = optical_density(rgb.astype(np.float32), reference, floor=floor)

    cloud = build_cloud(
        density, observed, beta=beta, alpha=alpha, tolerance_deg=tolerance_deg
    )
    additivity = measure_additivity(cloud, rgb, white) if cloud is not None else None

    return Density(
        rgb=rgb,
        od=density,
        white=white,
        stats=measure_stats(density),
        limits=measure_limits(rgb, density, reference, floor=floor, beta=beta),
        tolerance_deg=tolerance_deg,
        cloud=cloud,
        additivity=additivity,
        mpp=mpp,
        floor=floor,
    )
