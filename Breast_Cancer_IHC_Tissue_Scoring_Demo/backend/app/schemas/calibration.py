"""Schemas for step 4 - white calibration.

Step 4 reports four things, and a reader has to be able to keep them apart:

  the glass       which pixels were measured, and what each exclusion removed.
                  Five exclusions, not one - a naive inversion of the tissue mask
                  includes pen ink, the coverslip edge, the halo around the
                  section and the scanner's own background fill, all of which are
                  darker than glass and all of which bias I0 downward.
  the white point I0 itself, per channel, with the percentile ladder around it
                  so the choice of the 95th can be seen to be stable rather than
                  taken on trust.
  the field       whether I0 was allowed to vary across the slide, the fitted
                  surface's diagnostics either way, and the statistic the
                  decision turned on. Both answers are always present.
  the consequence what empty glass measures as in optical density once the
                  calibration is applied to it. This is the step's own error
                  bar, expressed in the units step 5 will produce.

Nothing here rescales the image, and that distinction is the whole of Rule 2.
Calibration records what "zero stain" means; normalisation rewrites the pixels so
they look like another slide's. The first preserves the measurement and makes it
comparable across slides; the second destroys it. See
`pipeline/step04_white_calibration/calibration.py`.
"""

from enum import Enum

from pydantic import Field

from app.schemas.common import APIModel


class CalibrationPanel(str, Enum):
    """The four panels step 4 is meant to be read as, in that order.

    An enum rather than a free string so a typo is rejected at the API boundary
    with the valid names, instead of reaching the service and coming back as a
    conflict - which is not what a typo is.
    """

    THUMBNAIL = "thumbnail"
    GLASS = "glass"
    FIELD = "field"
    CORRECTED = "corrected"


class CalibrationParams(APIModel):
    """The settings a run actually used, resolved rather than requested."""

    target_mpp: float = Field(description="Resolution asked for, in microns per pixel")
    calibration_mpp: float = Field(
        description=(
            "Resolution achieved. Every physical extent below is divided by this, and it "
            "matches the tissue mask's own resolution so the two grids need no resampling."
        )
    )
    width: int
    height: int
    capped: bool = Field(description="True when the size cap, not the target, set the resolution")

    percentile: float = Field(description="Which percentile of the glass was taken as I0")
    border_um: float = Field(description="Width of the discarded outer frame, in microns")
    fill_min_share: float = Field(
        description="Share of the candidate glass a level must hold to be tested as a fill"
    )
    fill_max_shoulder: float = Field(
        description="Shoulder at or below which a level is a digital constant, not a measurement"
    )
    clearance_um: float = Field(
        description="Width of the discarded ring just outside the tissue, in microns"
    )
    border_px: int
    clearance_px: int
    patch_um: float = Field(description="Side of one sampling patch, in microns")
    patch_px: int
    patch_min_glass: float = Field(
        description="Share of a patch that must be glass before it is sampled"
    )
    min_patches: int = Field(description="Fewest usable patches before a surface is permitted")
    vignette_snr: float = Field(description="Swing-over-residual a surface must clear to be used")
    max_leverage: float = Field(
        description="Extrapolation leverage over the tissue a surface must stay under"
    )
    od_floor: float = Field(
        description="Intensity floor applied to observed pixels, so black is finite, not infinite"
    )

    tissue_threshold: int = Field(description="The step 3 cut this glass set was derived from")
    tissue_threshold_source: str = Field(
        description="'otsu', 'triangle' or 'manual' - which rule set step 3's cut"
    )
    qc_gated: bool = Field(
        description="Whether step 2's artefacts were excluded from the glass as well"
    )
    qc_source: str | None = Field(
        default=None, description="'grandqc' or 'otsu' - which tissue pass step 2 used"
    )


class Channels(APIModel):
    """One value per RGB channel.

    A named triple rather than a bare list, because every quantity in this step
    is per channel and the three numbers are genuinely different: a scanner's
    illuminant is not neutral, and collapsing the three would bake its colour
    cast into every optical density downstream.
    """

    r: float
    g: float
    b: float


class BackgroundFill(APIModel):
    """A digital background fill found in the candidate glass, and excluded from it.

    A scanner writes a constant value into the part of the slide canvas it never
    imaged. It is not glass - nothing was measured there - and it is darker than
    the real glass beside it, so leaving it in the sample drags I0 down and reports
    apparent stain over a region that was never imaged.

    `shoulder` is the evidence, and it is why this is a test and not a guess: real
    glass carries sensor noise, so the colours within one level of its commonest
    colour hold a substantial fraction of that colour's own count. A digital
    constant has no such neighbours. On the demo's scans the two differ by a
    factor of 145.
    """

    rgb: Channels
    hex: str
    share: float = Field(description="Share of the candidate glass this fill covered")
    shoulder: float = Field(
        description=(
            "Colours within one level in every channel, over this colour's own count. Near "
            "zero for a digital constant; 0.24 and up for real glass, whose sensor noise "
            "guarantees it neighbours."
        )
    )
    pixels: int


class CalibrationExclusion(APIModel):
    """One move of the glass selection, and what it left behind."""

    key: str = Field(
        description="'complement', 'border', 'artefacts', 'fill' or 'clearance'"
    )
    label: str
    what: str = Field(description="What the exclusion removes, and why that is not glass")
    extent_um: float | None = Field(
        default=None, description="Physical extent for a distance-based exclusion"
    )
    pixels: int
    area_mm2: float
    delta_pixels: int = Field(description="Change from the previous move; always negative or zero")
    delta_area_mm2: float
    delta_share: float = Field(description="Change as a fraction of the previous selection")


class WhitePointOut(APIModel):
    """I0 as one triple, plus everything needed to distrust it."""

    rgb: Channels = Field(description="I0 - the intensity that corresponds to zero stain")
    hex: str = Field(description="The same value as a CSS colour, for the swatch")
    percentile: float

    ladder: dict[str, Channels] = Field(
        description=(
            "I0 at each of several percentiles, keyed by percentile. The argument for the "
            "95th is that its neighbours agree with it: on clean glass the value is flat "
            "from the 90th to the 99th. Where it climbs steeply the sample is contaminated."
        )
    )
    plateau: float = Field(
        description=(
            "Spread across the 90th to 99th percentiles as a fraction of I0, worst channel. "
            "Small means the choice of percentile is not doing the work."
        )
    )

    clipped: Channels = Field(
        description="Share of sampled glass pinned at 255 in each channel"
    )
    saturated: bool = Field(
        description=(
            "True when a channel clipped. The sensor ran out of range before the glass did, "
            "so I0 is an underestimate and optical density is compressed - a scanning fault, "
            "reported rather than corrected."
        )
    )
    cast: float = Field(
        description=(
            "Spread of the three channels over their mean. Zero would be a neutral "
            "illuminant; real scanners are not, which is why I0 is three numbers."
        )
    )
    sampled_pixels: int
    sampled_area_mm2: float


class CalibrationPatch(APIModel):
    """One sampling patch, its own white point, and whether it voted.

    Bounds are fractions of the slide, not pixels, so the browser can draw the
    grid over a thumbnail of any size without knowing this step's resolution.
    """

    col: int
    row: int
    x: float
    y: float
    width: float
    height: float

    glass_pixels: int
    glass_share: float
    used: bool = Field(description="Whether the patch held enough glass to be sampled")
    rgb: Channels | None = Field(
        default=None, description="The patch's own I0, when it was sampled"
    )
    hex: str | None = Field(default=None, description="The same value as a CSS colour")


class SurfaceOut(APIModel):
    """The fitted illumination field, whether or not it was used.

    Always reported when it could be fitted at all. A white point that varies
    across the slide is a strong claim, so the diagnostics that would refute it -
    how far the samples sit from the fit, how few of them there were - travel
    with it rather than being available on request.
    """

    #: 3 channels x 6 terms, in `terms` order, for coordinates normalised to [-1, 1].
    coefficients: list[list[float]]
    terms: list[str] = Field(description="The quadratic's terms, in coefficient order")

    r2: Channels = Field(description="Share of patch-to-patch variance the fit explains")
    residual_rms: Channels = Field(description="RMS distance from the patch samples, in levels")
    swing: Channels = Field(description="Brightest-to-dimmest of the fitted field, in levels")
    mean: Channels = Field(description="Mean of the fitted field - the flat value it reduces to")

    patches_used: int
    leverage: float = Field(
        description=(
            "99th percentile of the fit's extrapolation leverage over the tissue - how far "
            "outside its own samples it is being asked to reach, where a density will "
            "actually be computed. Around 1 at the edge of the samples' spread and unbounded "
            "beyond it. Measured over the tissue and not the frame on purpose: glass samples "
            "ring the section, so the frame's corners are always extrapolated and no "
            "measurement ever comes from them."
        )
    )
    leverage_max: float = Field(description="The worst single leverage over the tissue")
    snr: float = Field(
        description=(
            "swing / residual_rms, worst channel. The statistic the decision turns on: a "
            "surface that bends by about as much as it misses its own samples by is fitting "
            "noise, not the lamp."
        )
    )
    amplitude: float = Field(
        description="The swing as a fraction of the level, worst channel - vignetting, in percent"
    )
    od_error: float = Field(
        description=(
            "The optical density error a single flat white point would leave in place. The "
            "number that decides whether any of this matters, stated in the units step 5 "
            "produces rather than as a percentage of intensity."
        )
    )


class FieldChoice(APIModel):
    """Flat or varying, what the alternative said, and the statistic behind it."""

    mode: str = Field(description="'flat' - one I0 for the slide - or 'surface'")
    reason: str = Field(description="Why, stated with the numbers it turned on")
    fitted: bool = Field(description="Whether a surface was fitted at all, used or not")
    patches_used: int
    min_patches: int = Field(description="Fewest usable patches before a surface is permitted")
    snr: float
    snr_required: float = Field(description="The cutoff `snr` was compared against")
    leverage: float = Field(description="Extrapolation leverage over the tissue")
    leverage_limit: float = Field(description="The cutoff `leverage` was compared against")


class NoiseFloorOut(APIModel):
    """What empty glass measures as, in optical density, after calibration.

    The step's own error bar, and the reason it is here rather than at step 14:
    no positivity cutoff downstream can sit below this number, and discovering
    that at the scoring stage is discovering it too late.
    """

    floor: Channels = Field(
        description="OD of the dimmest sampled glass - the largest apparent stain on nothing"
    )
    median: Channels = Field(description="OD of the median sampled glass - the typical case")
    percentile: float = Field(description="Which tail of the glass 'floor' was read from")
    worst: float = Field(description="The largest of the three floor values")


class CalibrationReport(APIModel):
    """Everything step 4 produced for one slide."""

    upload_id: str
    filename: str
    generated_at: str

    params: CalibrationParams
    white: WhitePointOut
    exclusions: list[CalibrationExclusion]
    fills: list[BackgroundFill] = Field(
        default_factory=list,
        description="Digital background fills found and excluded, most prevalent first",
    )
    patches: list[CalibrationPatch]
    surface: SurfaceOut | None
    choice: FieldChoice
    noise: NoiseFloorOut

    glass_pixels: int
    glass_area_mm2: float
    glass_share: float = Field(
        description="Sampled glass over the whole scan, after all four exclusions"
    )

    notes: list[str] = Field(
        default_factory=list,
        description="Caveats that belong on screen next to the numbers, not in a docstring",
    )
    citation: str


class CalibrationDifference(APIModel):
    """What using one slide's I0 on another would cost, in optical density.

    The point of the comparison endpoint, and the whole "why per slide" argument
    as one number. Because density is a logarithm of a ratio, a mismatched white
    point lands as a *constant additive offset* on every pixel of every
    measurement - not as a scaling a later normalisation could absorb. Applied to
    the wrong slide, that offset is indistinguishable from more stain.
    """

    upload_id: str
    other_upload_id: str
    od_shift: Channels = Field(description="log10(this I0 / the other's), per channel")
    worst: float = Field(description="The largest absolute shift across the three channels")


class CalibrationSummary(APIModel):
    """One slide's white point, for the side-by-side comparison."""

    upload_id: str
    filename: str
    rgb: Channels
    hex: str
    mode: str = Field(description="'flat' or 'surface' - which field this slide justified")
    saturated: bool
    noise_floor: float = Field(description="Worst-channel OD of the dimmest sampled glass")
    glass_share: float


class CalibrationComparison(APIModel):
    """Two or more slides' white points, and the cost of confusing them.

    This is the demo's per-slide argument in one response: the swatches are
    visibly different colours, and `differences` says what that difference is
    worth in the units the pipeline actually reports.
    """

    slides: list[CalibrationSummary]
    differences: list[CalibrationDifference]
    worst_shift: float = Field(
        description="Largest OD offset between any pair of slides in this comparison"
    )
    note: str
