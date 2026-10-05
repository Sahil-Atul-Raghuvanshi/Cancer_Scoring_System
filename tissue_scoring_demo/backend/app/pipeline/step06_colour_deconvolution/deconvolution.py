"""Step 6's algorithm: one mixed picture into one picture per stain.

Every pixel of an IHC section carries both dyes at once - blue haematoxylin in the
nuclei, brown DAB wherever the antibody bound - and in the scan they arrive added
together. Step 5 put them in the one space where "added together" is literally
true: optical density, where Beer-Lambert makes a mixture a *sum* of one vector
per stain times how much of it is there. A sum is a 3x3 linear system, and a 3x3
linear system has an inverse. That inverse is this step.

**Why this is the fork.** The two channels that come out go to different places
and neither is a display convenience:

  haematoxylin  to the region model (step 8), in place of RGB. The single biggest
                source of colour nuisance between the H&E slide and the five IHC
                slides is the brown, and after this step the brown is *gone* from
                what the model sees - which is what lets one detector work on all
                six rather than one per stain.
  DAB           to the measurement (steps 13-15), on an absolute optical-density
                scale. Rule 3: separate the stains before you threshold, and
                threshold the DAB channel only. A dark blue nucleus under faint
                brown is, in RGB, indistinguishable from a moderately brown one.

**Fixed vectors, and the demo proves why.** The matrix could come from two
places: Ruifrok & Johnston's published constants, or an estimate of this tile's
own two stain directions - Macenko's method, which step 5 already ran to draw its
arms. This module computes **both**, through identical arithmetic, and reports
what each would score. They differ, and that difference is the argument: a
per-image estimate rescales itself to whatever slide it is given, so "0.4 DAB"
would mean a different amount of dye on every slide in a study, and two slides
could not be compared - which is the entire purpose of the pipeline. The
measurement branch therefore gets the fixed basis. The estimate is computed to be
shown, never to be measured with.

**What is honest about this step.** Three things are checked rather than claimed:

  exactness     the basis is square, so un-mixing and putting back together must
                return the input. `measure_basis` does exactly that and reports
                the worst error in optical density units. A separation that lost
                something would not be a change of coordinates.
  negatives     a pixel whose colour sits outside the cone the two stains span
                needs a negative amount of one of them. That cannot be a physical
                concentration, so the share is counted rather than clipped away.
  disentangling before un-mixing, the three optical-density channels are nearly
                the same picture - all of them track how much total dye is there.
                Afterwards the two stain channels should share much less. Both
                correlations are measured, on the reader's own tile.

Nothing here reads a slide, a file or a service. Optical density in, channels and
numbers out.

Ruifrok AC, Johnston DA. Quantification of histochemical staining by colour
deconvolution. Anal Quant Cytol Histol 23(4):291-299 (2001).
Macenko M et al. A method for normalizing histology slides for quantitative
analysis. ISBI 2009:1107-1110 - the estimator, kept for the comparison only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.common.stains import (
    ASSAY_STAINS,
    RUIFROK_CHANNELS,
    RUIFROK_HDAB,
    complete_basis,
    degrees_between,
    estimate_stain_matrix,
    recompose,
    unmix,
)

#: The two bases this step compares, in the order the screen reads them.
FIXED = "fixed"
ESTIMATED = "estimated"
BASES: tuple[str, str] = (FIXED, ESTIMATED)

#: Channel names, in the column order of every matrix here. Ruifrok's own order:
#: counterstain, chromogen, and the direction neither occupies.
CHANNELS: tuple[str, str, str] = RUIFROK_CHANNELS

#: Bins in each channel's reported histogram.
CHANNEL_BINS = 64

#: Fewest pixels with a usable direction before a per-image basis is estimated at
#: all. Macenko's method reads two angular percentiles off a point cloud, and
#: percentiles of a few hundred points are noise wearing a number's clothes. Same
#: figure step 5 requires before it will draw a cloud, for the same reason.
MIN_ESTIMATE_PIXELS = 500

#: Degrees at or beyond which an estimated basis is called a departure from the
#: published one worth remarking on. Not a failure - a slide really can be stained
#: off-nominal - but past this the two bases are measuring on visibly different
#: scales, which is the thing the comparison exists to show.
BASIS_DEPARTURE_DEG = 8.0


class DeconvolutionError(ValueError):
    """The tile cannot be un-mixed - and why, in words."""


# --- one basis's three channels ----------------------------------------------


@dataclass(frozen=True)
class Channels:
    """The three coefficient images one basis produces, as HxW float32 each.

    Kept as separate arrays rather than an HxWx3 stack because they are three
    different quantities that happen to share a shape: two are amounts of a dye
    and the third is what neither dye explains. Stacking them invites a caller to
    treat the triple as a colour, which is the exact confusion this step exists to
    end.
    """

    haematoxylin: np.ndarray
    dab: np.ndarray
    residual: np.ndarray

    def by_name(self, name: str) -> np.ndarray:
        try:
            return {
                "haematoxylin": self.haematoxylin,
                "dab": self.dab,
                "residual": self.residual,
            }[name]
        except KeyError as exc:
            raise DeconvolutionError(
                f"unknown channel {name!r}; this step produces {list(CHANNELS)}"
            ) from exc


def separate(od: np.ndarray, matrix: np.ndarray) -> Channels:
    """Un-mix an HxWx3 optical density tile onto one 3x3 stain basis.

    One inverse of a 3x3 and one matrix multiply over the pixels - the whole of
    colour deconvolution, once the logarithm of step 5 has made the mixture a sum.
    """
    height, width = od.shape[0], od.shape[1]
    flat = od.reshape(-1, 3).astype(np.float64)

    coefficients = unmix(flat, np.linalg.inv(np.asarray(matrix, dtype=np.float64)))
    planes = coefficients.reshape(height, width, 3).astype(np.float32)

    return Channels(
        haematoxylin=planes[..., 0],
        dab=planes[..., 1],
        residual=planes[..., 2],
    )


# --- what one channel looks like ---------------------------------------------


@dataclass(frozen=True)
class ChannelStats:
    """Where one stain channel's values landed, over the stained part of the tile.

    Measured over the stained pixels and not the whole tile, deliberately. Two
    thirds of any tile is faint background, and including it would drag every
    percentile towards zero and make two bases look alike for a reason that has
    nothing to do with either.
    """

    name: str
    #: The column of the basis this channel was projected onto.
    vector: tuple[float, float, float]

    median: float
    p99: float
    maximum: float
    mean: float
    #: Share of stained pixels needing a negative amount of this stain - a colour
    #: outside the cone the two vectors span. Counted, never clipped.
    negative_share: float

    histogram: tuple[int, ...]
    histogram_low: float
    histogram_high: float


def measure_channel(
    values: np.ndarray,
    stained: np.ndarray,
    *,
    name: str,
    vector: np.ndarray,
    low: float,
    high: float,
) -> ChannelStats:
    """Percentiles and a histogram of one channel over the stained pixels.

    `low` and `high` are handed in rather than read off this channel's own data,
    and that is what makes the two bases comparable: a histogram stretched to its
    own extremes would rescale itself for each basis and hide the very shift the
    comparison is about.
    """
    selected = values[stained].astype(np.float64)
    if selected.size == 0:
        raise DeconvolutionError(
            "no pixel of this tile carries enough stain to measure a channel on"
        )

    span = high - low if high - low > 1e-9 else 1e-9
    counts, _ = np.histogram(selected, bins=CHANNEL_BINS, range=(low, low + span))

    return ChannelStats(
        name=name,
        vector=(float(vector[0]), float(vector[1]), float(vector[2])),
        median=float(np.median(selected)),
        p99=float(np.percentile(selected, 99.0)),
        maximum=float(selected.max()),
        mean=float(selected.mean()),
        negative_share=float(np.mean(selected < 0.0)),
        histogram=tuple(int(value) for value in counts),
        histogram_low=float(low),
        histogram_high=float(low + span),
    )


# --- the score this basis would produce --------------------------------------


@dataclass(frozen=True)
class Preview:
    """What the measurement branch would report off this basis, on this tile.

    A preview and not the score. Step 14 sets the real cut-points per antibody
    against clinical guidance, and step 15 aggregates over invasive tumour rather
    than over one tile. What this is for is narrower and worth being exact about:
    it is the *same* arithmetic applied to two bases, so that the difference
    between the two numbers is caused by the basis and by nothing else.

    The cut is an absolute optical density, which is the whole argument. A
    per-slide percentile - "the top 10% of this slide is positive" - would return
    the same number under either basis and under any staining whatsoever, which is
    exactly how a demo can normalise the diagnosis away without noticing.
    """

    #: The absolute DAB optical density a pixel has to clear to count as positive.
    cut: float
    #: Share of the tile's stained pixels that clear it.
    positive_share: float
    #: Mean and 99th percentile of DAB over the stained pixels, in OD units - the
    #: scale the cut is applied against.
    mean_dab: float
    p99_dab: float


def preview_score(dab: np.ndarray, stained: np.ndarray, *, cut: float) -> Preview:
    """The positive share this basis would hand the measurement branch."""
    selected = dab[stained].astype(np.float64)
    if selected.size == 0:
        raise DeconvolutionError("no stained pixel to score")

    return Preview(
        cut=float(cut),
        positive_share=float(np.mean(selected >= cut)),
        mean_dab=float(selected.mean()),
        p99_dab=float(np.percentile(selected, 99.0)),
    )


# --- one whole basis, measured -----------------------------------------------


@dataclass(frozen=True)
class Basis:
    """One stain matrix, the channels it produced, and how well it did."""

    #: "fixed" or "estimated".
    kind: str
    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    channels: tuple[ChannelStats, ChannelStats, ChannelStats]
    preview: Preview

    #: Degrees each of the two stain columns sits from Ruifrok's published vector.
    #: Zero for the fixed basis by construction - it *is* the published pair - and
    #: the honest way to read the estimated one: how far this tile's own stains
    #: have drifted from the constants the measurement is defined against.
    degrees_from_published: tuple[float, float]

    #: Worst and mean error, in optical density, of un-mixing and putting back
    #: together. A square basis makes this float rounding; anything larger would
    #: mean the separation lost something.
    exactness_max: float
    exactness_mean: float

    #: How much of the tile's density the third column had to absorb - what
    #: neither stain explains, as a share of the total. Small on a clean H-DAB
    #: section; large when a third absorber is present or the basis points wrong.
    residual_share: float

    #: How much the two stain channels still share, as absolute Pearson
    #: correlation over the stained pixels. This is what un-mixing is *for*, so it
    #: is measured rather than assumed.
    channel_correlation: float


def measure_basis(
    od: np.ndarray,
    channels: Channels,
    stained: np.ndarray,
    matrix: np.ndarray,
    *,
    kind: str,
    cut: float,
    scale: tuple[float, float, float] | None = None,
) -> Basis:
    """Everything reportable about one basis, on one tile.

    `scale` is the histogram range each channel is counted over, and it is passed
    in so that both bases can be counted over the *fixed* basis's range. Letting
    each stretch to its own would redraw the axis under the comparison and make
    two different answers look like one.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    flat = od.reshape(-1, 3).astype(np.float64)

    if channels.haematoxylin.shape != od.shape[:2]:
        raise DeconvolutionError("the channels do not cover the tile they came from")

    stack = np.stack(
        [channels.haematoxylin, channels.dab, channels.residual], axis=-1
    ).reshape(-1, 3)
    restored = recompose(stack, matrix)
    error = np.abs(restored - flat)

    published = np.asarray(
        [
            np.asarray(RUIFROK_HDAB[:, index], dtype=np.float64)
            for index in range(len(ASSAY_STAINS))
        ]
    )

    # Over the stained pixels for the same reason the channel stats are: the
    # background is neither stain and would make every basis look equally good.
    mask = stained.reshape(-1)
    total = np.abs(flat[mask]).sum(axis=1)
    residual_share = float(
        np.abs(stack[mask][:, 2]).sum() / max(float(total.sum()), 1e-9)
    )

    ranges = scale if scale is not None else channel_range(channels, stained)

    return Basis(
        kind=kind,
        matrix=tuple(
            (float(matrix[0, col]), float(matrix[1, col]), float(matrix[2, col]))
            for col in range(3)
        ),  # type: ignore[arg-type]
        channels=tuple(  # type: ignore[arg-type]
            measure_channel(
                channels.by_name(name),
                stained,
                name=name,
                vector=matrix[:, index],
                low=min(0.0, -0.15 * ranges[index]),
                high=ranges[index],
            )
            for index, name in enumerate(CHANNELS)
        ),
        preview=preview_score(channels.dab, stained, cut=cut),
        degrees_from_published=(
            degrees_between(matrix[:, 0], published[0]),
            degrees_between(matrix[:, 1], published[1]),
        ),
        exactness_max=float(error.max()),
        exactness_mean=float(error.mean()),
        residual_share=residual_share,
        channel_correlation=_correlation(
            channels.haematoxylin[stained], channels.dab[stained]
        ),
    )


def channel_range(channels: Channels, stained: np.ndarray) -> tuple[float, float, float]:
    """The upper end of each channel's histogram, read off the stained pixels.

    The 99.5th percentile rather than the maximum: one saturated nucleus would
    otherwise set the axis for the rest of the tile. A floor keeps the range
    positive on a tile where a channel is essentially empty.

    Public because step 7 draws its sample tile against this scale. A tile shown
    there and the same tile shown on step 6's screen have to be the same picture,
    and they are only the same picture if the stretch is computed by one rule.
    """
    return tuple(  # type: ignore[return-value]
        max(float(np.percentile(channels.by_name(name)[stained], 99.5)), 1e-3)
        for name in CHANNELS
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Absolute Pearson correlation of two flat arrays, or 0 for a constant one."""
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return 0.0

    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(abs(float(a @ b) / denominator))


def mixed_correlation(od: np.ndarray, stained: np.ndarray) -> float:
    """How alike the three optical-density channels are *before* un-mixing.

    Mean absolute correlation of the three pairs, over the stained pixels. It is
    high on any stained tile and that is the point: red, green and blue all rise
    and fall with how much total dye is present, so none of them is a measurement
    of either stain. The same figure for the two stain channels afterwards is what
    says the separation did something.
    """
    flat = od.reshape(-1, 3)[stained.reshape(-1)]
    pairs = ((0, 1), (0, 2), (1, 2))
    return float(
        np.mean([_correlation(flat[:, left], flat[:, right]) for left, right in pairs])
    )


# --- the whole step ----------------------------------------------------------


@dataclass
class Separation:
    """Everything step 6 produced for one tile.

    Both bases are carried, and the estimated one may be None - a tile without
    enough pixels carrying a usable direction has nothing to estimate from, which
    is a state to report rather than to approximate. The fixed basis is never
    None: it does not depend on the tile at all, which is exactly the property the
    measurement branch is built on.
    """

    #: The tile as it was read, and its optical density - step 5's own arrays.
    rgb: np.ndarray
    od: np.ndarray
    #: HxW: pixels carrying at least step 5's beta of stain. Every statistic here
    #: is taken over these, so the background cannot flatter a basis.
    stained: np.ndarray

    fixed_channels: Channels
    estimated_channels: Channels | None

    fixed: Basis
    estimated: Basis | None
    #: Why there is no estimated basis, when there is not.
    estimated_refusal: str | None

    #: The display range both bases' panels are drawn against - the fixed basis's,
    #: so the toggle shows a real change and not a re-stretch.
    scale: tuple[float, float, float]

    #: Correlation among the three OD channels before un-mixing, for comparison
    #: with each basis's `channel_correlation` after it.
    mixed_correlation: float

    cut: float

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.rgb.shape[0]), int(self.rgb.shape[1]))

    def channels_for(self, kind: str) -> Channels:
        """One basis's channel images, by name."""
        if kind == FIXED:
            return self.fixed_channels
        if kind == ESTIMATED:
            if self.estimated_channels is None:
                raise DeconvolutionError(
                    self.estimated_refusal
                    or "no per-image basis could be estimated on this tile"
                )
            return self.estimated_channels
        raise DeconvolutionError(f"unknown basis {kind!r}; expected one of {list(BASES)}")


def deconvolve(
    *,
    od: np.ndarray,
    rgb: np.ndarray,
    stained: np.ndarray,
    admitted: np.ndarray,
    cut: float,
    alpha: float,
) -> Separation:
    """Run step 6 end to end on one tile: both bases, identical arithmetic.

    `stained` is step 5's own "this pixel carries stain" mask and `admitted` its
    own "this pixel has a direction worth estimating from" mask, both carried
    across rather than re-derived. A step that disagreed with the one above it
    about which pixels carry stain would be un-mixing a different tile from the
    one whose arms the reader was just shown.

    The order matters in one place: the fixed basis is measured first, because its
    channel ranges are the scale the estimated basis is then counted and drawn
    against. That is what makes the toggle on screen a comparison rather than two
    separately stretched pictures.
    """
    if od.ndim != 3 or od.shape[2] != 3:
        raise DeconvolutionError("colour deconvolution needs an HxWx3 optical density tile")
    if stained.shape != od.shape[:2]:
        raise DeconvolutionError("the stained mask and the density tile are different sizes")
    if not bool(np.any(stained)):
        raise DeconvolutionError(
            "no pixel of this tile carries enough stain to un-mix. There is nothing "
            "wrong with the arithmetic - a tile of empty glass has no stains in it - "
            "so pick a tile with tissue on it"
        )

    fixed_channels = separate(od, RUIFROK_HDAB)
    scale = channel_range(fixed_channels, stained)
    fixed = measure_basis(
        od, fixed_channels, stained, RUIFROK_HDAB, kind=FIXED, cut=cut, scale=scale
    )

    estimated_channels, estimated, refusal = _estimate(
        od=od, stained=stained, admitted=admitted, cut=cut, alpha=alpha, scale=scale
    )

    return Separation(
        rgb=rgb,
        od=od,
        stained=stained,
        fixed_channels=fixed_channels,
        estimated_channels=estimated_channels,
        fixed=fixed,
        estimated=estimated,
        estimated_refusal=refusal,
        scale=scale,
        mixed_correlation=mixed_correlation(od, stained),
        cut=float(cut),
    )


def _estimate(
    *,
    od: np.ndarray,
    stained: np.ndarray,
    admitted: np.ndarray,
    cut: float,
    alpha: float,
    scale: tuple[float, float, float],
) -> tuple[Channels | None, Basis | None, str | None]:
    """Macenko's per-image basis, computed for the comparison and nothing else.

    Returns `(None, None, reason)` rather than raising when the tile cannot
    support an estimate. The step still has an answer in that case - the fixed
    basis, which is the one that would be used anyway - and turning a missing
    comparison into a failed request would make the demo unable to show a tile
    whose only fault is that Macenko's method does not apply to it.
    """
    flat = od.reshape(-1, 3).astype(np.float64)
    usable = flat[np.asarray(admitted, dtype=bool).reshape(-1)]

    if usable.shape[0] < MIN_ESTIMATE_PIXELS:
        return (
            None,
            None,
            f"only {usable.shape[0]} pixels of this tile have a direction reliable enough "
            f"to estimate a stain vector from, against the {MIN_ESTIMATE_PIXELS} needed. "
            "Macenko's method reads two angular percentiles off a point cloud, and "
            "percentiles of a few hundred points are noise. The fixed basis is unaffected: "
            "it does not depend on the tile at all, which is the property this comparison "
            "exists to demonstrate.",
        )

    try:
        pair, _ = estimate_stain_matrix(usable, alpha=alpha)
        matrix = complete_basis(pair)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return (
            None,
            None,
            f"a per-image basis could not be estimated on this tile: {exc}. The fixed "
            "basis is unaffected - it does not depend on the tile.",
        )

    channels = separate(od, matrix)
    basis = measure_basis(
        od, channels, stained, matrix, kind=ESTIMATED, cut=cut, scale=scale
    )
    return channels, basis, None
