"""Schemas for step 6 - colour deconvolution.

Step 6 reports one thing twice, and the repetition is the content: the same tile
un-mixed onto Ruifrok & Johnston's published stain vectors, and onto an estimate
of this tile's own. Both come back in one response so a reader can hold the two
side by side without a second request, because the comparison is the argument -
a per-image basis rescales itself to whatever slide it is given, so a measurement
taken against one cannot be compared with the same measurement on the next slide.

  the bases       one entry each for `fixed` and `estimated`, each carrying its
                  matrix, its three channels, the score it would produce, and the
                  checks that say whether the separation was clean.
  the comparison  what changes between them: the angle the estimated vectors sit
                  from the published ones, and how far the score moves.
  the tile        which field of view, inherited from step 5 rather than chosen
                  again - both steps must be looking at the same pixels for the
                  fork to be a fact about the code rather than a drawing.

`Channels` is imported from step 4's schemas, as step 5's are, because a stain
vector is three numbers for exactly the reason a white point is.

See `pipeline/step06_colour_deconvolution/deconvolution.py`.
"""

from enum import Enum

from pydantic import Field

from app.schemas.calibration import Channels
from app.schemas.common import APIModel


class DeconvolutionBasis(str, Enum):
    """Which stain matrix a request wants a picture of.

    An enum rather than a free string so a typo is rejected at the API boundary
    with the valid names, rather than reaching the service and returning as a
    conflict - which is not what a typo is.
    """

    FIXED = "fixed"
    ESTIMATED = "estimated"


class DeconvolutionPanel(str, Enum):
    """The four panels the step is meant to be read as, in that order."""

    TILE = "tile"
    HAEMATOXYLIN = "haematoxylin"
    DAB = "dab"
    RESIDUAL = "residual"


class DeconvolutionParams(APIModel):
    """The settings a run actually used, resolved rather than requested."""

    positive_cut: float = Field(
        description=(
            "The absolute DAB optical density a pixel must clear to count as positive in "
            "the preview score. Absolute and not a per-slide percentile, which is the whole "
            "argument: a percentile would return the same number under either basis and "
            "under any staining at all, which is how a demo normalises the diagnosis away."
        )
    )
    arm_percentile: float = Field(
        description="Macenko's alpha, used only for the estimated basis this step compares against"
    )
    beta: float = Field(
        description=(
            "Step 5's cut for 'this pixel carries stain'. Every statistic here is taken over "
            "those pixels, so a tile's faint background cannot flatter a basis."
        )
    )
    min_estimate_pixels: int = Field(
        description="Fewest pixels with a usable direction before a per-image basis is estimated"
    )
    channel_bins: int


class ChannelOut(APIModel):
    """One stain channel, over the stained part of the tile."""

    name: str = Field(description="'haematoxylin', 'dab' or 'residual'")
    vector: Channels = Field(description="The unit optical-density direction it was projected onto")

    median: float
    p99: float
    maximum: float
    mean: float
    negative_share: float = Field(
        description=(
            "Share of stained pixels needing a negative amount of this stain - a colour "
            "outside the cone the two vectors span. Counted rather than clipped, because a "
            "square basis is exact only if nothing is repaired."
        )
    )

    histogram: list[int]
    histogram_low: float = Field(
        description=(
            "Both bases are counted over the same range - the fixed basis's - so the "
            "comparison is not redrawn under itself."
        )
    )
    histogram_high: float


class PreviewOut(APIModel):
    """What the measurement branch would report off this basis, on this tile.

    A preview and not the score: step 14 sets real cut-points per antibody and
    step 15 aggregates over invasive tumour rather than over one tile. Its purpose
    is narrower - the same arithmetic applied to two bases, so the gap between the
    two numbers is caused by the basis and nothing else.
    """

    cut: float
    positive_share: float = Field(
        description="Share of the tile's stained pixels that clear the cut"
    )
    mean_dab: float
    p99_dab: float


class BasisOut(APIModel):
    """One stain matrix, the channels it produced, and how well it did."""

    kind: str = Field(description="'fixed' or 'estimated'")
    matrix: list[Channels] = Field(
        description=(
            "The three columns: haematoxylin, DAB, and the direction no stain occupies. That "
            "third column is what makes the matrix invertible and therefore the separation "
            "exact."
        )
    )
    channels: list[ChannelOut]
    preview: PreviewOut

    degrees_from_published: list[float] = Field(
        description=(
            "Degrees each stain column sits from Ruifrok's published vector - zero for the "
            "fixed basis by construction, since it is that pair. For the estimated basis it "
            "is how far this tile's own stains have drifted from the constants the "
            "measurement is defined against."
        )
    )

    exactness_max: float = Field(
        description=(
            "Worst error, in optical density, of un-mixing and putting back together. The "
            "basis is square, so this is float rounding; anything larger would mean the "
            "separation lost something."
        )
    )
    exactness_mean: float
    residual_share: float = Field(
        description=(
            "How much of the tile's density the third column had to absorb, as a share of "
            "the total. Small on a clean haematoxylin-DAB section."
        )
    )
    channel_correlation: float = Field(
        description=(
            "How much the two stain channels still share, as absolute Pearson correlation "
            "over the stained pixels. This is what un-mixing is for, so it is measured."
        )
    )


class ComparisonOut(APIModel):
    """What changes when the basis changes - the reason this step uses fixed vectors."""

    #: Degrees between the estimated and published haematoxylin, and DAB.
    haematoxylin_degrees: float
    dab_degrees: float
    #: The two preview scores, and the gap between them in percentage points.
    fixed_positive_share: float
    estimated_positive_share: float
    share_shift: float = Field(
        description="estimated minus fixed, in share units. Signed - it can go either way"
    )
    #: How much the estimated basis rescales the DAB channel: its mean over the
    #: fixed basis's. 1.0 would mean the two agree about how much stain is present.
    dab_scale: float
    #: True when the estimated vectors sit far enough from the published pair that
    #: the two bases are visibly measuring on different scales.
    departed: bool
    departure_deg: float


class DeconvolutionTile(APIModel):
    """The field of view this step un-mixed, inherited from step 5.

    Repeated here rather than left to be fetched from step 5, because a reader
    looking at a haematoxylin channel has to be able to see which tile it is
    without holding two screens in their head - and because the two steps looking
    at the same pixels is the claim, not an implementation detail.
    """

    x: int
    y: int
    size: int
    mpp: float
    tile_um: float
    level: int
    rank: int = Field(description="Where this tile placed in step 5's screening, 1 being best")
    requested: bool = Field(description="True when a position was named rather than scored for")
    candidates_scored: int
    stained_share: float = Field(
        description="Share of the tile carrying at least beta of stain - what the statistics cover"
    )
    admitted_share: float = Field(
        description="Share with a direction reliable enough to estimate a stain vector from"
    )


class ArmAgreementOut(APIModel):
    """How far the published vectors sit from the arms step 5 measured on this tile.

    The bridge between the two screens. Step 5 found two directions in the data
    without being told what either stain looks like; this says how close the
    constants step 6 measures with land to them. It is the evidence that using
    fixed vectors on this slide is not an assumption being imposed.
    """

    stain: str
    #: The arm step 5 reported, and which published vector it landed nearest.
    arm_vector: Channels
    nearest: str
    degrees: float


class DeconvolutionReport(APIModel):
    """Everything step 6 produced for one tile of one slide."""

    upload_id: str
    filename: str
    generated_at: str

    params: DeconvolutionParams
    tile: DeconvolutionTile

    fixed: BasisOut
    estimated: BasisOut | None = Field(
        default=None,
        description=(
            "Null when the tile cannot support a per-image estimate. The step still has an "
            "answer in that case - the fixed basis, which is the one that would be used "
            "anyway - so a missing comparison is not a failed run."
        ),
    )
    estimated_refusal: str | None = Field(
        default=None, description="Why there is no estimated basis, when there is not"
    )
    comparison: ComparisonOut | None = Field(
        default=None, description="Null for the same reason `estimated` is"
    )

    mixed_correlation: float = Field(
        description=(
            "Mean absolute correlation among the three optical-density channels before "
            "un-mixing. High on any stained tile, because red, green and blue all rise and "
            "fall with how much total dye is present - which is why none of them is a "
            "measurement of either stain. Read against each basis's channel_correlation."
        )
    )
    arm_agreement: list[ArmAgreementOut] = Field(
        default_factory=list,
        description="How far the published vectors sit from the arms step 5 measured here",
    )

    notes: list[str] = Field(
        default_factory=list,
        description="Caveats that belong on screen next to the numbers, not in a docstring",
    )
    citation: str
