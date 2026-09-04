"""Step 4's algorithm: what does "no stain" measure as, on this slide?

Optical density is defined relative to a reference, `OD = -log10(I / I0)`, so
before step 5 can compute a density at all something has to say what `I0` is.
This module says it, by measuring the one part of the slide that is known to
hold no stain: the empty glass beside the section.

Four moves, in this order:

  1. decide which pixels are glass. That is five exclusions, not one, and each
     one removes a specific way of being wrong - see `select_glass`.
  2. sample a high percentile of each RGB channel over what survives. That is
     the flat white point, and it is what the pipeline guide asks for.
  3. sample the same percentile per *patch* and fit a smooth quadratic surface
     through the patch values. That is the robust variant the guide mentions in
     one line: a white point that varies across the slide, which also corrects
     uneven illumination.
  4. choose between the two from the data, and report both either way.

**Why this cannot be a constant.** The obvious alternative to this whole step is
`I0 = 255`, or a fixed triple measured once from a reference slide. Both are
wrong for the same reason: `I0` is a property of an *acquisition*, not of a
stain. A lamp dims as it ages, a scanner's white balance is set per batch, a
coverslip's mounting medium yellows, and a slide scanned twice on two machines
comes back with two different whites. Every one of those shifts lands in the
denominator of every optical density downstream, and none of it is biology. Two
slides with identical tissue would then produce two different H-scores, and the
difference would be the lamp.

**Why this cannot be a percentile of the whole image, either.** That is the trap
Rule 2 names. Rescaling each slide by its own overall distribution - the usual
"normalise per slide" - also removes real differences in how much stain is
present, which is the quantity being measured. Sampling the *glass* is different
in kind: glass is known a priori to hold zero stain, so what is being removed is
only the illumination, and what survives is an absolute scale on which two
slides can be compared. Calibration preserves the measurement; normalisation
overwrites it.

**Why a percentile rather than the maximum or the mean.** `I0` sits in a
denominator, so an overestimate darkens the whole slide's density and an
underestimate can drive it negative. The maximum is set by a single hot pixel or
one specular glint off the coverslip. The mean is dragged down by any tissue,
dust or precipitate that leaked past step 3's mask. A high percentile is above
the debris and below the outliers, and `WhitePoint.ladder` carries the whole
range so the reader can see the value stop moving rather than take 95 on trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.common.imaging import channel_percentile, optical_density

#: The percentile ladder reported alongside the chosen one. The point of showing
#: it is the plateau: on a slide with real glass on it the value barely moves
#: between the 90th and the 99th, and *that* is the argument for the 95th being
#: a stable choice rather than a tuned one. Where it does not plateau, the glass
#: is contaminated and the report says so.
LADDER: tuple[float, ...] = (50.0, 75.0, 90.0, 95.0, 99.0, 99.9, 100.0)

#: Percentile of the glass used to state the noise floor. The *dimmest* glass,
#: not the brightest: the floor is the largest apparent stain the glass itself
#: produces, so it is read off the tail that looks most like tissue.
FLOOR_PERCENTILE = 5.0

#: Terms of the fitted surface, in normalised slide coordinates. A quadratic and
#: no higher: vignetting is a smooth bowl, and a cubic has enough freedom to
#: start following whichever corner of the slide happened to hold more glass.
SURFACE_TERMS: tuple[str, ...] = ("1", "x", "y", "x^2", "xy", "y^2")


class CalibrationError(ValueError):
    """A precondition step 4 cannot supply for itself."""


# --- deciding what counts as glass -------------------------------------------


@dataclass(frozen=True)
class Exclusion:
    """One move of the glass selection, and what it left behind.

    Recorded per move for the same reason step 3 records its cleanup ladder: the
    interesting fact is not how much glass survived but which exclusion cost
    what. A slide where the border alone removed most of the candidate glass is a
    slide scanned tight to the tissue, and the reader should be able to see that
    rather than infer it.
    """

    key: str
    label: str
    what: str
    #: Physical extent for a distance-based exclusion, else None.
    extent_um: float | None
    pixels: int


@dataclass
class Glass:
    """The pixels step 4 is allowed to measure, and how they were arrived at."""

    glass: np.ndarray
    steps: list[Exclusion] = field(default_factory=list)
    #: Digital background fills found and excluded, most prevalent first.
    fills: list[Fill] = field(default_factory=list)

    @property
    def pixels(self) -> int:
        return int(np.count_nonzero(self.glass))


def _radius_px(extent_um: float, mpp: float) -> int:
    """A physical extent as a pixel radius on this grid, at least 1 px.

    Same helper as step 3's, and deliberately the same rule: every distance in
    this module is stated in microns, so none of them changes meaning on a
    different scanner, and none of them is allowed to round away to zero.
    """
    return max(1, int(round(extent_um / mpp)))


@dataclass(frozen=True)
class Fill:
    """A digital background fill found in the candidate glass, and excluded from it.

    Not glass, and not an artefact either: a scanner writes a constant value into
    the part of the slide canvas it never imaged. On the demo's Aperio scans that
    constant covers two thirds of the frame, and it is *darker* than the real
    glass beside it - so left in the sample it drags I0 down, and left in the
    noise floor it reports `log10(193/146)` = 0.12 OD of apparent stain on a
    region that was never measured at all.

    The test is that an optical measurement cannot have zero variance. Real glass
    carries sensor noise, so the colours immediately around its commonest colour
    hold a substantial fraction of that colour's own count. A digital constant has
    no such neighbours, because nothing wrote anything near it. On the demo's
    scans the fill scores 0.0017 by that measure and the real glass beside it
    0.24 - a 145-fold gap, which is what makes this a test rather than a tuned
    threshold. See `_shoulder`.

    This is the same argument `modal_share` makes for step 3 - a spike has no
    variance - applied to a colour rather than to a threshold rule.
    """

    rgb: tuple[int, int, int]
    #: Share of the candidate glass this fill covered.
    share: float
    #: Neighbouring colours' count over this colour's own. Near zero for a fill.
    shoulder: float
    pixels: int


#: How many of the commonest colours to test per pass. A fill is always among the
#: most prevalent values, but it is not always *the* most prevalent - the real
#: glass can outnumber a border fill - so the mode alone is not enough to look at.
FILL_CANDIDATES = 6


def _shoulder(values: np.ndarray, triple: tuple[int, int, int], exact: int) -> float:
    """How much sits *next to* this colour, over how much sits exactly on it.

    The neighbourhood is the 3x3x3 box of colours within one level in every
    channel, minus the colour itself. That box is the right shape for the question
    because a fill is a constant *triple*: a colour can be a spike in red and a
    hump in green, and only something that is a spike in all three at once is a
    digital constant.

    Near zero means nothing was written near this value, which no sensor can
    manage - photon noise and quantisation both guarantee a spread of at least a
    level. Measured on the demo's Aperio scans: the fill scores 0.0017 and the
    real glass beside it 0.24, a 145-fold gap. Cleaner still on an uncompressed
    scan, where the glass scores 7 and up. The shipped cutoff of 0.05 sits thirty
    times above the fill and five times below the tightest real glass seen, which
    is what makes this a test rather than a tuned threshold.
    """
    if exact <= 0:
        return float("inf")

    near = np.all(
        np.abs(values.astype(np.int16) - np.array(triple, dtype=np.int16)) <= 1, axis=1
    )
    return (int(np.count_nonzero(near)) - exact) / exact


def detect_fills(
    rgb: np.ndarray,
    candidate: np.ndarray,
    *,
    min_share: float,
    max_shoulder: float,
    passes: int = 3,
) -> list[Fill]:
    """Every digital fill in the candidate glass, most prevalent first.

    Iterated rather than run once, because a scan can carry more than one: a black
    padding border outside a grey scanner fill is the common pair. And each pass
    tests the several commonest colours rather than only the mode, because after
    the largest fill is removed the real glass may well outnumber what is left -
    stopping at the mode would leave the smaller fill in the sample.

    Two tests, both of which have to pass. The share test asks whether the colour
    is prevalent enough to matter at all; the shoulder test - see `_shoulder` - is
    the one that decides, and it is the one that keeps real glass safe however
    much of the frame it covers.
    """
    fills: list[Fill] = []
    remaining = candidate.copy()

    for _ in range(passes):
        values = rgb[remaining]
        total = int(values.shape[0])
        if total == 0:
            break

        # Packed to a single integer so `np.unique` counts whole colours rather
        # than channels. A colour is 24 bits, which fits int32 with room over.
        codes = (
            (values[:, 0].astype(np.int32) << 16)
            | (values[:, 1].astype(np.int32) << 8)
            | values[:, 2].astype(np.int32)
        )
        unique, counts = np.unique(codes, return_counts=True)
        order = np.argsort(counts)[::-1]

        found: Fill | None = None
        for index in order[:FILL_CANDIDATES]:
            count = int(counts[index])
            share = count / total
            if share < min_share:
                # Sorted by count, so nothing after this one can qualify either.
                break

            code = int(unique[index])
            triple = ((code >> 16) & 0xFF, (code >> 8) & 0xFF, code & 0xFF)
            shoulder = _shoulder(values, triple, count)

            if shoulder <= max_shoulder:
                found = Fill(rgb=triple, share=share, shoulder=shoulder, pixels=count)
                break

        if found is None:
            break

        fills.append(found)
        remaining = remaining & ~np.all(
            rgb == np.array(found.rgb, dtype=rgb.dtype), axis=-1
        )

    return fills


def select_glass(
    *,
    rgb: np.ndarray,
    tissue: np.ndarray,
    considered: np.ndarray | None,
    mpp: float,
    border_um: float,
    clearance_um: float,
    fill_min_share: float,
    fill_max_shoulder: float,
) -> Glass:
    """Glass is what is left after five exclusions. Each one is a way of being wrong.

    The pipeline guide's version is "invert the tissue mask and drop the
    outermost border". That is the first and second of these. The other two are
    here because the guide's version quietly assumes the only thing on the glass
    is glass, and on a real slide it is not.

      1. **not tissue.** Step 3's mask, inverted. This is the whole basis of the
         step: the tissue is the only part of the slide that might hold stain,
         so everything else is a candidate reference.

      2. **not the outermost frame.** Scanners vignette hardest at the edge of
         the field, the coverslip edge and its meniscus live out there, and a
         slide's own edge is frequently a bright band that is not glass at all.
         A percentile is robust to a few such pixels and not to a whole border.

      3. **not an artefact.** Pen ink is *on* the glass, and so is dust, and so
         is the shadow of a coverslip crack. Step 2 already found them, and
         leaving them in would bias `I0` downward - which compresses every
         optical density downstream, because `I0` is the denominator of the ratio
         the log is taken of: a low `I0` raises `I / I0` and so shrinks
         `-log10` of it. Faint stain and no stain end up measuring closer
         together, which is the same damage a clipped channel does. This is the
         same ordering argument step 3 makes about the histogram, applied to the
         percentile. When step 2 has not run this exclusion is skipped and the
         report says so.

      4. **not the scanner's own background fill.** A whole-slide scanner writes
         a constant value into the part of the canvas it never imaged, and on the
         demo's Aperio scans that constant covers two thirds of the frame. It is
         not glass - nothing was measured there - and it is darker than the real
         glass beside it, so leaving it in drags `I0` down and reports two tenths
         of an OD of apparent stain over a region that was never imaged. See
         `detect_fills` for how it is told apart from glass, which is by the one
         property a digital constant cannot fake.

      5. **not the halo around the tissue.** The pixels immediately outside a
         section are not clean glass: they carry the section's edge, the
         mounting medium that wicked out from under it, and the resampling error
         of the mask boundary itself. All three are darker than glass, and all
         three sit exactly where a naive inversion of the mask puts its brightest
         candidates. Dilating the tissue by a physical clearance and subtracting
         that is what removes them.

    Exclusion 5 also absorbs any one-pixel disagreement between the tissue mask
    and this grid, which is why a nearest-neighbour resample of the mask is
    sufficient upstream: a 60 um clearance is thirty times the error it could
    introduce.
    """
    from scipy import ndimage

    steps: list[Exclusion] = []

    glass = ~tissue.astype(bool)
    steps.append(
        Exclusion(
            key="complement",
            label="Not tissue",
            what=(
                "Step 3's mask, inverted. Tissue is the only part of the slide that can "
                "hold stain, so everything else is a candidate for zero."
            ),
            extent_um=None,
            pixels=int(np.count_nonzero(glass)),
        )
    )

    border_px = _radius_px(border_um, mpp)
    if 2 * border_px < min(glass.shape):
        frame = np.zeros_like(glass)
        frame[border_px:-border_px, border_px:-border_px] = True
        glass = glass & frame
    else:
        # A slide narrower than two borders would be excluded out of existence.
        # Better to keep the frame and say so in the report than to return no
        # glass at all on a small specimen scan.
        border_px = 0
    steps.append(
        Exclusion(
            key="border",
            label="Drop the outer frame",
            what=(
                f"Discards the outermost {border_um:g} um of the scan ({border_px} px "
                "here). Vignetting is worst at the edge of the field, and the coverslip "
                "edge and slide edge both live out there."
                if border_px
                else "Skipped: this scan is not wide enough to give up a frame that size."
            ),
            extent_um=border_um,
            pixels=int(np.count_nonzero(glass)),
        )
    )

    if considered is not None:
        glass = glass & considered.astype(bool)
        what = (
            "Removes what step 2 flagged - pen, dust, coverslip edge. These sit *on* the "
            "glass, and they are darker than it, so leaving them in pulls I0 down and "
            "compresses every optical density downstream - faint stain and no stain "
            "measure closer together than they should."
        )
    else:
        what = (
            "Skipped: step 2 has not run, so nothing is known to be an artefact. Any pen "
            "mark on the glass is in the sample below and is pulling I0 down."
        )
    steps.append(
        Exclusion(
            key="artefacts",
            label="Drop artefacts",
            what=what,
            extent_um=None,
            pixels=int(np.count_nonzero(glass)),
        )
    )

    fills = detect_fills(
        rgb, glass, min_share=fill_min_share, max_shoulder=fill_max_shoulder
    )
    for found in fills:
        glass = glass & ~np.all(rgb == np.array(found.rgb, dtype=rgb.dtype), axis=-1)

    if fills:
        listed = ", ".join(
            f"({item.rgb[0]}, {item.rgb[1]}, {item.rgb[2]}) at {item.share:.1%}"
            for item in fills
        )
        fill_what = (
            f"Removes the scanner's background fill - {listed}. Nothing was imaged there, "
            "and the fill is darker than the real glass, so leaving it in would drag I0 "
            "down. Told apart from glass by having no neighbours: sensor noise puts colours "
            "either side of any real measurement, and a digital constant has none."
        )
    else:
        fill_what = (
            "Skipped: no colour in the candidate glass is a digital constant. Every "
            "prevalent colour has neighbours either side of it, which is what an optical "
            "measurement looks like."
        )

    steps.append(
        Exclusion(
            key="fill",
            label="Drop the background fill",
            what=fill_what,
            extent_um=None,
            pixels=int(np.count_nonzero(glass)),
        )
    )

    clearance_px = _radius_px(clearance_um, mpp)
    square = np.ones((3, 3), dtype=bool)
    halo = ndimage.binary_dilation(
        tissue.astype(bool), structure=square, iterations=clearance_px
    )
    glass = glass & ~halo
    steps.append(
        Exclusion(
            key="clearance",
            label="Stand back from the tissue",
            what=(
                f"Drops the {clearance_um:g} um ({clearance_px} px) ring just outside the "
                "section. Mounting medium, the section's own edge and the mask's boundary "
                "error all live there, and all three are darker than glass."
            ),
            extent_um=clearance_um,
            pixels=int(np.count_nonzero(glass)),
        )
    )

    return Glass(glass=glass, steps=steps, fills=fills)


# --- the flat white point ----------------------------------------------------


@dataclass(frozen=True)
class WhitePoint:
    """I0 as one triple, plus everything needed to judge it.

    `ladder` is not decoration. A percentile is a choice, and the way to show
    that this particular choice is not doing the work is to show that the
    neighbouring ones agree with it. On a slide with real glass the value is flat
    from the 90th to the 99th; where it climbs steeply the sample is contaminated
    and the step should not be trusted, which is a thing the reader can only see
    if they are shown the ladder.
    """

    rgb: tuple[float, float, float]
    percentile: float
    #: percentile -> (R, G, B), for the plateau argument above.
    ladder: dict[float, tuple[float, float, float]]
    #: Share of sampled glass at 255 in each channel - see `clipped`.
    clipped: tuple[float, float, float]
    sampled_pixels: int

    @property
    def cast(self) -> float:
        """Spread of the three channels as a fraction of their mean.

        Zero would mean a neutral illuminant. Real scanners are not neutral, and
        this is the number that says how far from it this one is - which is
        exactly the information a single-channel white point would destroy.
        """
        values = np.asarray(self.rgb, dtype=np.float64)
        mean = float(values.mean())
        return float(np.ptp(values) / mean) if mean > 0 else 0.0

    @property
    def saturated(self) -> bool:
        """Whether the scanner clipped its whites, which caps what OD can resolve.

        A channel pinned at 255 means the sensor ran out of range before the
        glass did, so the true incident intensity is unknown and is being
        underestimated. Optical density is then compressed - a faint stain and no
        stain at all measure closer together than they should - and no amount of
        arithmetic here recovers it. It is a scanning fault, so it is reported
        rather than corrected.
        """
        return any(share > 0.02 for share in self.clipped)


def measure_white_point(
    rgb: np.ndarray, glass: np.ndarray, *, percentile: float
) -> WhitePoint:
    """Sample I0 from the glass, with the percentile ladder beside it."""
    sampled = int(np.count_nonzero(glass))
    if sampled == 0:
        raise CalibrationError(
            "no glass left to sample after the exclusions, so there is nothing to "
            "measure I0 from. Either the tissue mask covers the whole scan - check "
            "step 3's threshold - or the slide was cropped to the section, in which "
            "case this slide cannot be calibrated against its own glass."
        )

    ladder = {
        level: channel_percentile(rgb, glass, level)
        for level in sorted({*LADDER, percentile})
    }

    values = rgb[glass]
    clipped = tuple(
        float(np.count_nonzero(values[:, channel] >= 255) / sampled) for channel in range(3)
    )

    return WhitePoint(
        rgb=ladder[percentile],
        percentile=percentile,
        ladder=ladder,
        clipped=(clipped[0], clipped[1], clipped[2]),
        sampled_pixels=sampled,
    )


# --- the illumination surface ------------------------------------------------


@dataclass(frozen=True)
class Patch:
    """One patch of the slide, and the white point measured inside it.

    Patches are what makes the surface fit robust and what the demo draws. A
    surface fitted to raw pixels would be pulled by every bright speck; fitted to
    one percentile per patch, a speck can only move the patch it is in, and a
    patch with too little glass in it does not vote at all.
    """

    col: int
    row: int
    #: Bounds on the mask grid, for drawing the patch on the thumbnail.
    x: int
    y: int
    width: int
    height: int
    glass_pixels: int
    glass_share: float
    used: bool
    #: The patch's own percentile, or None when it held too little glass.
    rgb: tuple[float, float, float] | None


def sample_patches(
    rgb: np.ndarray,
    glass: np.ndarray,
    *,
    mpp: float,
    patch_um: float,
    min_glass_share: float,
    percentile: float,
) -> list[Patch]:
    """A grid of patch white points, and the ones with too little glass to count.

    Both are returned. The demo marks the used patches in green and the rejected
    ones faintly, because where the glass *is not* is as much a part of the
    picture as where it is - a slide with tissue across its whole width has no
    samples in the middle, and a surface fitted from its edges only is a weaker
    claim than one fitted from all over. That is visible in the panel and stated
    in the report.
    """
    size = max(8, int(round(patch_um / mpp)))
    height, width = glass.shape
    patches: list[Patch] = []

    for row, top in enumerate(range(0, height, size)):
        for col, left in enumerate(range(0, width, size)):
            bottom, right = min(top + size, height), min(left + size, width)
            window = glass[top:bottom, left:right]

            area = window.size
            count = int(np.count_nonzero(window))
            share = count / area if area else 0.0
            used = share >= min_glass_share and count > 0

            patches.append(
                Patch(
                    col=col,
                    row=row,
                    x=left,
                    y=top,
                    width=right - left,
                    height=bottom - top,
                    glass_pixels=count,
                    glass_share=share,
                    used=used,
                    rgb=(
                        channel_percentile(rgb[top:bottom, left:right], window, percentile)
                        if used
                        else None
                    ),
                )
            )

    return patches


def _design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The quadratic design matrix, in the order of `SURFACE_TERMS`."""
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])


def _normalise(
    x: np.ndarray, y: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates onto [-1, 1] in both axes.

    The fit is done in normalised coordinates so the coefficients mean the same
    thing on any slide and the normal equations stay well conditioned - a
    quadratic in raw pixel numbers squares values in the thousands, and the
    resulting matrix is numerically miserable.
    """
    height, width = shape
    return (
        2.0 * x / max(1, width - 1) - 1.0,
        2.0 * y / max(1, height - 1) - 1.0,
    )


def _leverage(
    design: np.ndarray, region: np.ndarray, shape: tuple[int, int]
) -> tuple[float, float]:
    """How far outside its own samples the fit is being asked to reach.

    Returns `(p99, max)` of the hat value `h = d' (X'X)^-1 d` over `region`, where
    `d` is the design row at a point and `X` the design matrix of the patch
    samples. This is the standard measure of extrapolation: `h` is small inside
    the samples' spread, around 1 at their edge, and grows without bound beyond
    it.

    It is measured over `region` - the tissue - and not over the whole frame, and
    that distinction is the point. The glass samples necessarily form a *ring*
    around the section, because the middle of the slide is the section. A quadratic
    fitted to a ring is wildly unconstrained at the frame's corners, and it does
    not matter in the least: the corners are empty glass or never-imaged canvas, and
    no stain will ever be measured there. What matters is whether the ring encloses
    the tissue well enough to pin the field down *over the tissue*, because that is
    where every optical density downstream is computed. Measuring leverage over the
    frame would reject a perfectly sound field for being uncertain about a place
    nobody looks.

    The 99th percentile rather than the maximum, for the same reason: a handful of
    pixels at an awkward corner of the section should not veto a field that is well
    determined across the rest of it, particularly since `evaluate_surface` clamps
    the field from below anyway.
    """
    gram = design.T @ design
    inverse = np.linalg.pinv(gram)

    height, width = shape
    # Subsampled: leverage is a smooth quadratic form, so a sixteenth-scale grid
    # over the region finds its spread without touching seventeen million pixels.
    step = max(1, min(height, width) // 256)
    ys, xs = np.nonzero(region[::step, ::step])
    if ys.size == 0:
        return 0.0, 0.0

    nx = 2.0 * (xs * step) / max(1, width - 1) - 1.0
    ny = 2.0 * (ys * step) / max(1, height - 1) - 1.0
    rows = _design(nx, ny)

    hats = np.einsum("ij,jk,ik->i", rows, inverse, rows)
    return float(np.percentile(hats, 99.0)), float(hats.max())


@dataclass(frozen=True)
class Surface:
    """A least-squares quadratic illumination field, one fit per channel.

    This is the "robust variant" the pipeline guide mentions in a single line:
    instead of one white point for the whole slide, `I0` becomes a smooth
    function of position, so uneven illumination is corrected at the same time as
    the white level. A scanner's field is brightest near the optical axis and
    falls off towards the edge; a quadratic is the lowest order that can express
    that, and low order is the point - the surface must model the lamp and not
    the slide.

    Every diagnostic needed to distrust the fit is here. `residual_rms` says how
    far the patch samples actually sit from it, `swing` says how much it bends
    across the slide, and the ratio between the two is what decides whether the
    bend is signal - see `choose_field`.
    """

    #: 3 channels x 6 terms, in `SURFACE_TERMS` order, for normalised coordinates.
    coefficients: tuple[tuple[float, ...], ...]
    #: Fraction of patch-to-patch variance the fit explains, per channel.
    r2: tuple[float, float, float]
    #: RMS of the residuals in intensity units, per channel.
    residual_rms: tuple[float, float, float]
    #: Peak-to-peak of the fitted field over the slide, in intensity units.
    swing: tuple[float, float, float]
    #: Mean of the fitted field, per channel - the flat value it reduces to.
    mean: tuple[float, float, float]
    patches_used: int
    shape: tuple[int, int]
    #: 99th percentile of extrapolation leverage over the region the field will be
    #: used on. See `_leverage`.
    leverage: float
    #: The worst single value of the same, for reference.
    leverage_max: float

    @property
    def snr(self) -> float:
        """The largest per-channel `swing / residual_rms`.

        The statistic the choice turns on. A surface whose bend is comparable to
        its own scatter is a surface fitted to noise; one whose bend is several
        times its scatter has found something systematic. Reported either way.
        """
        ratios = [
            swing / rms if rms > 1e-9 else 0.0
            for swing, rms in zip(self.swing, self.residual_rms, strict=True)
        ]
        return float(max(ratios)) if ratios else 0.0

    @property
    def amplitude(self) -> float:
        """The bend as a fraction of the level, worst channel. Human-legible vignetting."""
        ratios = [
            swing / mean if mean > 1e-9 else 0.0
            for swing, mean in zip(self.swing, self.mean, strict=True)
        ]
        return float(max(ratios)) if ratios else 0.0

    @property
    def od_error(self) -> float:
        """The optical-density error ignoring this surface would cause, worst channel.

        This is the number that decides whether any of this matters, and it is
        the only honest way to state vignetting in a pipeline whose output is a
        density. A 6% swing in `I0` sounds alarming and is worth
        `log10(1/0.94) = 0.027` OD, which is nothing next to a DAB signal of 0.3
        and everything next to the gap between a 1+ and a 2+ call. Stating the
        percentage alone leaves the reader unable to tell which case they are in.
        """
        ratios = [
            float(np.log10(1.0 / (1.0 - min(0.99, swing / mean)))) if mean > 1e-9 else 0.0
            for swing, mean in zip(self.swing, self.mean, strict=True)
        ]
        return float(max(ratios)) if ratios else 0.0


def fit_surface(
    patches: list[Patch], *, shape: tuple[int, int], applies_to: np.ndarray | None = None
) -> Surface | None:
    """Fit a quadratic through the usable patch white points, or None if too few.

    None rather than a degenerate fit. Six coefficients need six points to be
    determined at all, and a fit that merely interpolates its own samples tells
    the caller nothing about the illumination between them. The caller decides
    what "too few" means - `choose_field` gets the count and the threshold - so
    this returns None only for the case where the algebra itself fails.

    `applies_to` is the region the field will actually be used on - the tissue -
    and it is what the leverage diagnostic is measured over. Omitted, leverage is
    measured over the whole frame, which is the pessimistic reading: the glass
    samples ring the section, so the frame's corners are always extrapolated and
    always irrelevant. See `_leverage`.
    """
    used = [patch for patch in patches if patch.used and patch.rgb is not None]
    if len(used) < len(SURFACE_TERMS):
        return None

    # Patch centres, not corners: the value is a percentile over the whole patch,
    # so the coordinate it belongs at is the middle of it.
    cx = np.array([patch.x + patch.width / 2.0 for patch in used])
    cy = np.array([patch.y + patch.height / 2.0 for patch in used])
    nx, ny = _normalise(cx, cy, shape)
    design = _design(nx, ny)

    # The full-slide grid the swing is measured over, on a coarse lattice: the
    # extremes of a quadratic over a rectangle sit at its corners or along an
    # edge, so a 32-point sweep per axis finds them without evaluating the fit
    # at every one of seventeen million pixels.
    gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, 32), np.linspace(-1.0, 1.0, 32))
    lattice = _design(gx.ravel(), gy.ravel())

    coefficients: list[tuple[float, ...]] = []
    r2: list[float] = []
    rms: list[float] = []
    swing: list[float] = []
    mean: list[float] = []

    for channel in range(3):
        observed = np.array([patch.rgb[channel] for patch in used])  # type: ignore[index]
        beta, *_ = np.linalg.lstsq(design, observed, rcond=None)

        predicted = design @ beta
        residual = observed - predicted
        spread = float(((observed - observed.mean()) ** 2).sum())

        field_values = lattice @ beta
        coefficients.append(tuple(float(value) for value in beta))
        r2.append(float(1.0 - (residual**2).sum() / spread) if spread > 1e-12 else 0.0)
        rms.append(float(np.sqrt((residual**2).mean())))
        swing.append(float(field_values.max() - field_values.min()))
        mean.append(float(field_values.mean()))

    region = applies_to if applies_to is not None else np.ones(shape, dtype=bool)
    leverage, leverage_max = _leverage(design, region, shape)

    return Surface(
        coefficients=tuple(coefficients),
        r2=(r2[0], r2[1], r2[2]),
        residual_rms=(rms[0], rms[1], rms[2]),
        swing=(swing[0], swing[1], swing[2]),
        mean=(mean[0], mean[1], mean[2]),
        patches_used=len(used),
        shape=shape,
        leverage=leverage,
        leverage_max=leverage_max,
    )


def evaluate_surface_on(surface: Surface, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The fitted field over the grid `y` x `x`, in normalised frame coordinates.

    Separate from `evaluate_surface` because not every caller wants the whole
    frame. Step 5 works on one tile at working magnification, and its white point
    is this field evaluated over the small patch of normalised coordinates that
    tile occupies - which is exact rather than approximate, the surface being an
    analytic quadratic in those coordinates. Re-deriving the quadratic there would
    put a second copy of these six terms in the codebase, and the two could then
    disagree about what step 4 decided.

    Evaluated separably rather than by building a six-column design matrix over
    every pixel: at 4,096 px square that matrix would be four hundred megabytes,
    and the terms of a quadratic are products of per-axis vectors anyway.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    height, width = int(y.size), int(x.size)

    ones_x = np.ones_like(x)
    ones_y = np.ones_like(y)

    field_values = np.empty((height, width, 3), dtype=np.float32)
    for channel, beta in enumerate(surface.coefficients):
        c0, cx, cy, cxx, cxy, cyy = (np.float32(value) for value in beta)
        field_values[..., channel] = (
            np.outer(ones_y, c0 * ones_x + cx * x + cxx * x * x)
            + np.outer(cy * y, ones_x)
            + np.outer(cxy * y, x)
            + np.outer(cyy * y * y, ones_x)
        )

    # A fitted quadratic can dip below zero outside the region it was fitted
    # from, and I0 is a denominator. Clamping to 1 keeps the field usable
    # everywhere; where it clamps, the surface was extrapolating and the SNR
    # test has almost certainly already rejected it.
    return np.maximum(field_values, np.float32(1.0))


def normalised_axis(length: int, size: int) -> np.ndarray:
    """`length` evenly spaced coordinates spanning one frame axis of `size` pixels.

    The parameterisation `evaluate_surface_on` expects, in one place, because two
    callers need it and a fit evaluated on a coordinate convention half a pixel
    off the one it was fitted in is wrong in a way nothing would report.
    """
    return 2.0 * np.arange(length, dtype=np.float32) / max(1, size - 1) - 1.0


def evaluate_surface(surface: Surface, shape: tuple[int, int]) -> np.ndarray:
    """The fitted field as an HxWx3 float32 array, for dividing an image by."""
    height, width = shape
    return evaluate_surface_on(
        surface, normalised_axis(width, width), normalised_axis(height, height)
    )


# --- choosing between them ---------------------------------------------------


@dataclass(frozen=True)
class Choice:
    """Which white point is in force, what the alternative said, and why.

    Same contract as step 3's threshold choice, and for the same reason: the step
    is allowed to pick, but not to pick quietly. Both answers, the statistic the
    decision turned on, and the cutoff it was compared against are all reported
    whichever way it went.
    """

    #: "flat" or "surface".
    mode: str
    reason: str
    #: Usable patches, and the fewest that would permit a surface.
    patches_used: int
    min_patches: int
    #: The surface's swing-over-residual, and the cutoff it was tested against.
    snr: float
    snr_required: float
    #: Extrapolation leverage over the tissue, and the cutoff it was tested against.
    leverage: float
    leverage_limit: float
    #: True when a surface was fitted at all, whether or not it was used.
    fitted: bool


def choose_field(
    surface: Surface | None,
    *,
    min_patches: int,
    required_snr: float,
    leverage_limit: float,
) -> Choice:
    """Use the surface only when its bend is demonstrably not its own noise.

    Four ways to fall back to the flat white point, and each is a different
    failure:

      no fit          fewer usable patches than a quadratic has coefficients.
      too few patches enough to solve, not enough to trust. A quadratic through
                      eight samples clustered along one edge will happily
                      extrapolate a bright corner nobody measured.
      low SNR         the fit bends by about as much as it misses its own samples
                      by. Then the bend is the scatter, and a white point that
                      curves to follow noise is worse than a flat one that does
                      not - it adds a position-dependent error to every density
                      while looking more sophisticated.
      extrapolating   the samples do not enclose the tissue well enough to pin the
                      field down over it. Glass samples necessarily ring the
                      section; usually the ring encloses it and the field is well
                      determined inside, but on a section that runs off the edge of
                      the scan the ring is open and the fit is guessing across the
                      part that matters most. `_leverage` is the measure.

    The default cutoff is 4: the surface must swing four times further across the
    slide than it typically misses a patch by. That is a deliberately shy
    threshold, because the cost of the two errors is not symmetric. Missing real
    vignetting leaves a smooth, slide-wide bias that step 5 reports honestly as a
    single white point; inventing vignetting that is not there stamps a spurious
    spatial gradient onto every measurement downstream, where nothing later can
    distinguish it from biology.
    """
    if surface is None:
        return Choice(
            mode="flat",
            reason=(
                f"Fewer than {len(SURFACE_TERMS)} patches held enough glass to sample, "
                "which is fewer than a quadratic has coefficients. There is nothing to "
                "fit a surface through, so one flat white point stands."
            ),
            patches_used=0,
            min_patches=min_patches,
            snr=0.0,
            snr_required=required_snr,
            leverage=0.0,
            leverage_limit=leverage_limit,
            fitted=False,
        )

    if surface.patches_used < min_patches:
        return Choice(
            mode="flat",
            reason=(
                f"{surface.patches_used} usable patches, under the {min_patches} this step "
                "requires before it will let a white point vary across the slide. Six "
                "coefficients can be solved from six points and trusted from far more; "
                "below the bar the fit would mostly be extrapolating into the parts of the "
                "slide that had no glass to sample."
            ),
            patches_used=surface.patches_used,
            min_patches=min_patches,
            snr=surface.snr,
            snr_required=required_snr,
            leverage=surface.leverage,
            leverage_limit=leverage_limit,
            fitted=True,
        )

    if surface.snr < required_snr:
        return Choice(
            mode="flat",
            reason=(
                f"The fitted field swings {surface.snr:.1f}x its own residual scatter, "
                f"under the {required_snr:g}x mark, so its bend is not distinguishable "
                "from the noise in the patch samples. A white point that curves to follow "
                "noise is worse than a flat one that does not, so the flat value is used. "
                f"Ignoring the surface costs at most {surface.od_error:.3f} OD."
            ),
            patches_used=surface.patches_used,
            min_patches=min_patches,
            snr=surface.snr,
            snr_required=required_snr,
            leverage=surface.leverage,
            leverage_limit=leverage_limit,
            fitted=True,
        )

    if surface.leverage > leverage_limit:
        return Choice(
            mode="flat",
            reason=(
                f"The glass samples do not enclose the tissue: over 99% of it the fit is "
                f"reaching {surface.leverage:.1f} times further outside its own samples than "
                f"the {leverage_limit:g} this step allows. That happens when the section runs "
                "off the edge of the scan, so the ring of glass around it is open and the "
                "field is being guessed across exactly the part of the slide every "
                "measurement comes from. The flat value is used instead - it is uncertain in "
                "the same way everywhere, which is the honest form for an unsupported guess."
            ),
            patches_used=surface.patches_used,
            min_patches=min_patches,
            snr=surface.snr,
            snr_required=required_snr,
            leverage=surface.leverage,
            leverage_limit=leverage_limit,
            fitted=True,
        )

    return Choice(
        mode="surface",
        reason=(
            f"The fitted field swings {surface.snr:.1f}x its own residual scatter, over the "
            f"{required_snr:g}x mark, so the slide really is unevenly lit: "
            f"{surface.amplitude:.1%} brightest-to-dimmest, which is "
            f"{surface.od_error:.3f} OD of error a single white point would leave in place. "
            "I0 varies with position from here on."
        ),
        patches_used=surface.patches_used,
        min_patches=min_patches,
        snr=surface.snr,
        snr_required=required_snr,
        leverage=surface.leverage,
        leverage_limit=leverage_limit,
        fitted=True,
    )


# --- what the glass measures as, once calibrated -----------------------------


@dataclass(frozen=True)
class NoiseFloor:
    """What empty glass measures as in optical density, after calibration.

    The point of the whole step, stated as the quantity step 5 produces. If the
    calibration is sound this is near zero and every threshold downstream has
    room above it; if it is not, this number says exactly how much apparent stain
    the pipeline will find on bare glass. No positivity cutoff can sit below it,
    and reporting it here is what stops that from being discovered at step 14.
    """

    #: OD of the dimmest sampled glass, per channel - the worst case.
    floor: tuple[float, float, float]
    #: OD of the median sampled glass, per channel - the typical case.
    median: tuple[float, float, float]
    percentile: float

    @property
    def worst(self) -> float:
        return float(max(self.floor))


def measure_noise_floor(
    rgb: np.ndarray,
    glass: np.ndarray,
    white: np.ndarray | tuple[float, float, float],
    *,
    floor: float,
) -> NoiseFloor:
    """Apply the calibration to the glass it came from and see what is left.

    A round trip on purpose. Every other number in this step describes the
    estimate; this one describes the estimate's consequence, computed through the
    same `optical_density` step 5 will use rather than through a second copy of
    the logarithm.
    """
    density = optical_density(rgb.astype(np.float32), white, floor=floor)
    sampled = density[glass]

    if sampled.size == 0:
        return NoiseFloor(
            floor=(0.0, 0.0, 0.0), median=(0.0, 0.0, 0.0), percentile=FLOOR_PERCENTILE
        )

    # The *low* intensity tail is the *high* density tail, so the floor is the
    # complement of FLOOR_PERCENTILE in density space.
    high = np.percentile(sampled, 100.0 - FLOOR_PERCENTILE, axis=0)
    middle = np.percentile(sampled, 50.0, axis=0)

    return NoiseFloor(
        floor=(float(high[0]), float(high[1]), float(high[2])),
        median=(float(middle[0]), float(middle[1]), float(middle[2])),
        percentile=FLOOR_PERCENTILE,
    )


# --- the whole step ----------------------------------------------------------


@dataclass
class Calibration:
    """Everything step 4 produced for one slide."""

    #: The pixels measured, and the exclusions that produced them.
    glass: Glass
    #: I0 as one triple, sampled over all of `glass`.
    white: WhitePoint
    #: The patch grid - what the demo draws, and what the surface was fitted to.
    patches: list[Patch]
    #: The fitted field, when there was enough glass to fit one.
    surface: Surface | None
    #: Flat or surface, and why.
    choice: Choice
    #: What the glass measures as under the chosen field.
    noise: NoiseFloor
    mpp: float
    shape: tuple[int, int]

    @property
    def uses_surface(self) -> bool:
        return self.choice.mode == "surface" and self.surface is not None

    def field(self) -> np.ndarray | tuple[float, float, float]:
        """The reference step 5 should divide by: a field, or one triple.

        One accessor for both so a caller never has to branch on `choice.mode` -
        `optical_density` broadcasts either shape, and the decision about which
        was justified was already made and reported here.
        """
        if self.uses_surface:
            assert self.surface is not None
            return evaluate_surface(self.surface, self.shape)
        return self.white.rgb


def calibrate(
    *,
    rgb: np.ndarray,
    tissue: np.ndarray,
    considered: np.ndarray | None,
    mpp: float,
    percentile: float,
    border_um: float,
    clearance_um: float,
    fill_min_share: float,
    fill_max_shoulder: float,
    patch_um: float,
    patch_min_glass: float,
    min_patches: int,
    required_snr: float,
    leverage_limit: float,
    od_floor: float,
) -> Calibration:
    """Run step 4 end to end on one slide's thumbnail and tissue mask.

    The order is not interchangeable. The glass has to be decided before
    anything is sampled from it, both samples have to exist before either can be
    chosen between, and the noise floor has to be measured under the field that
    was actually chosen - measuring it under the flat value and then shipping the
    surface would be quoting a consequence of a calibration that is not the one
    in force.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise CalibrationError("white calibration needs an RGB image, not a single channel")
    if tissue.shape != rgb.shape[:2]:
        raise CalibrationError(
            f"the tissue mask is {tissue.shape} and the image is {rgb.shape[:2]}; step 4 "
            "measures glass by position, so the two grids have to be the same one"
        )

    shape = (rgb.shape[0], rgb.shape[1])

    selected = select_glass(
        rgb=rgb,
        tissue=tissue,
        considered=considered,
        mpp=mpp,
        border_um=border_um,
        clearance_um=clearance_um,
        fill_min_share=fill_min_share,
        fill_max_shoulder=fill_max_shoulder,
    )
    white = measure_white_point(rgb, selected.glass, percentile=percentile)
    patches = sample_patches(
        rgb,
        selected.glass,
        mpp=mpp,
        patch_um=patch_um,
        min_glass_share=patch_min_glass,
        percentile=percentile,
    )
    # `applies_to` is the tissue: the field is judged on whether it is pinned down
    # where a density will be measured, not over a frame whose corners nobody reads.
    surface = fit_surface(patches, shape=shape, applies_to=tissue.astype(bool))
    choice = choose_field(
        surface,
        min_patches=min_patches,
        required_snr=required_snr,
        leverage_limit=leverage_limit,
    )

    calibration = Calibration(
        glass=selected,
        white=white,
        patches=patches,
        surface=surface,
        choice=choice,
        noise=NoiseFloor(
            floor=(0.0, 0.0, 0.0), median=(0.0, 0.0, 0.0), percentile=FLOOR_PERCENTILE
        ),
        mpp=mpp,
        shape=shape,
    )
    calibration.noise = measure_noise_floor(
        rgb, selected.glass, calibration.field(), floor=od_floor
    )
    return calibration


def od_difference(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    """`log10(left / right)` per channel: the OD error of using one I0 for the other.

    This is the arithmetic behind "calibration must be per slide". Two slides'
    white points differ by some ratio, and because density is a logarithm of that
    ratio, the difference lands as a *constant additive offset* on every pixel of
    every measurement - not as a scaling that a later normalisation could absorb.
    Applied to the wrong slide, that offset is indistinguishable from more stain.
    """
    a = np.maximum(np.asarray(left, dtype=np.float64), 1e-6)
    b = np.maximum(np.asarray(right, dtype=np.float64), 1e-6)
    shift = np.log10(a / b)
    return (float(shift[0]), float(shift[1]), float(shift[2]))
