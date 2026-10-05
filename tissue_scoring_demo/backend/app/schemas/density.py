"""Schemas for step 5 - optical density.

Step 5 reports five things, and a reader has to be able to keep them apart:

  the tile        which field of view was transformed, why that one, and what the
                  alternatives scored. A density is per pixel, so unlike steps 3
                  and 4 this step has to choose a place to stand, and the choice
                  is not neutral - most of a section is counterstain, and a tile
                  of counterstain has one arm rather than two.
  the densities   where they landed, per channel and in total. Percentiles rather
                  than a mean and a spread, because the distribution is skewed by
                  construction: a lot of faint counterstain, a little strong DAB.
  the limits      the three places a density is not a reading - the intensity
                  floor, pixels brighter than I0, and pixels too faint to have a
                  direction - plus the exact inverse applied back, which is what
                  shows the transform threw nothing away.
  the cloud       the densities as a point cloud, the two arms in it, and where
                  Ruifrok & Johnston's published vectors fall on the same axes.
                  This is the picture that makes step 6 obvious rather than
                  magical, and it is either in the data or it is not.
  the additivity  Beer-Lambert, tested on the reader's own tile. One stain keeps
                  one direction in density space as it darkens and does not in
                  intensity space, and both drifts are reported in degrees.

`Channels` is imported from step 4's schemas rather than redefined. It is the
same thing - one value per RGB channel - and the reason step 4 gives for it
being three numbers and not one is exactly the reason a density is three numbers
too.

Nothing here is rewritten, and that is the whole of Rule 2. This step is the
trunk: step 6 deconvolves these densities once and hands the haematoxylin channel
to the model and the DAB channel to the measurement, so both arms leave from the
numbers below. See `pipeline/step05_optical_density/density.py`.
"""

from enum import Enum

from pydantic import Field

from app.schemas.calibration import Channels
from app.schemas.common import APIModel


class DensityPanel(str, Enum):
    """The five panels step 5 is meant to be read as, in that order.

    An enum rather than a free string so a typo is rejected at the API boundary
    with the valid names, instead of reaching the service and coming back as a
    conflict - which is not what a typo is.
    """

    MAP = "map"
    TILE = "tile"
    DENSITY = "density"
    SCATTER = "scatter"
    LIMITS = "limits"


class DensityParams(APIModel):
    """The settings a run actually used, resolved rather than requested."""

    target_mpp: float = Field(description="Working resolution asked for, in microns per pixel")
    tile_mpp: float = Field(
        description=(
            "Resolution the tile was actually read at. Differs from the target only when "
            "the pyramid has no level at or above it."
        )
    )
    tile_size: int = Field(description="Side of the tile in pixels")
    tile_um: float = Field(description="Side of the tile in microns - the physical field of view")
    level: int = Field(description="Pyramid level read. Chosen by microns, never by index")
    downsample: float
    resampled: bool = Field(
        description=(
            "True when the pyramid overshot the target and the read was area-averaged down. "
            "Averaging happens in intensity space, before the logarithm - a coarser sensor "
            "averages transmitted light, and averaging densities instead would take the mean "
            "of logarithms, which is a different and lower number."
        )
    )

    od_floor: float = Field(
        description="Intensity floor applied to observed pixels, so black is finite, not infinite"
    )
    beta: float = Field(
        description=(
            "Mean density below which a pixel is treated as carrying no stain and kept out "
            "of the cloud. Macenko's beta: the direction of a near-zero vector is set by "
            "rounding, not by biology."
        )
    )
    arm_percentile: float = Field(
        description="Which tail of the angular distribution is taken as an arm. Macenko's alpha"
    )
    angular_tolerance_deg: float = Field(
        description=(
            "How far 8-bit quantisation may move a pixel's direction before it is kept out "
            "of the cloud. The mirror image of beta: beta drops pixels with too little "
            "stain to have a direction, this drops the very dark ones, where one intensity "
            "level moves a channel's density by 1/(I ln10) - 0.43 at I=1. Those pixels are "
            "also long, so they land at the angular extremes, which is where the arms are "
            "read from."
        )
    )

    screening_mpp: float = Field(
        description="Resolution of the thumbnail the candidate tiles were scored on"
    )
    screening_block_px: int = Field(description="Side of one candidate block on that thumbnail")
    min_tissue_share: float = Field(
        description="Share of a block that must be tissue before it can be chosen"
    )
    min_stain: float = Field(
        description=(
            "Mean optical density a block must reach to be eligible. A multiple of step 4's "
            "measured noise floor, not a constant: the chooser's mixing term is a ratio of "
            "eigenvalues of a point cloud, and a ratio means nothing until the cloud is "
            "larger than the noise around it. Without this gate a faint patch of false "
            "colour outscores real haematoxylin and DAB, because two wrong colours are "
            "still two colours."
        )
    )
    min_stain_multiple: float = Field(description="The multiple of the noise floor above")

    tissue_threshold: int = Field(description="The step 3 cut this run's white point came from")
    tissue_threshold_source: str = Field(description="'otsu', 'triangle' or 'manual'")
    qc_gated: bool = Field(description="Whether step 2's artefacts were excluded from the search")
    qc_source: str | None = Field(default=None, description="'grandqc' or 'otsu'")


class WhiteUsed(APIModel):
    """Step 4's white point, as this step divided by it.

    Repeated here rather than left to be fetched from step 4, because a density is
    *defined* against this value and a reader looking at one has to be able to see
    which number produced it without holding two screens in their head.
    """

    rgb: Channels = Field(description="I0 - the intensity that corresponds to zero stain")
    hex: str
    mode: str = Field(
        description=(
            "'flat' - one I0 for the slide - or 'surface', in which case the value above is "
            "the field's mean and the tile was divided by the field evaluated over its own "
            "position."
        )
    )
    percentile: float
    saturated: bool = Field(
        description="True when a channel of I0 clipped at 255, so every density here is compressed"
    )
    noise_floor: float = Field(
        description=(
            "Worst-channel density of the dimmest glass step 4 sampled. The floor every "
            "number on this screen has to be read against."
        )
    )


class TileCandidate(APIModel):
    """One block the tile chooser scored, and what it scored.

    Bounds are fractions of the slide, not pixels, so the browser can draw the
    grid over a thumbnail of any size without knowing this step's resolution. Same
    contract as step 4's patch bounds, for the same reason.
    """

    col: int
    row: int
    x: int = Field(description="Level-0 origin - the coordinate that does not move with resolution")
    y: int

    fx: float
    fy: float
    fw: float
    fh: float

    tissue_share: float
    considered_share: float = Field(
        description="Share step 2 left in play. 1.0 when quality control has not run"
    )
    stain: float = Field(description="Mean optical density over the block's tissue")
    mixing: float = Field(
        description=(
            "How far the block's density cloud spreads off its own dominant direction - "
            "sqrt(second eigenvalue / first). One stain is one ray and scores near zero; two "
            "stains open a wedge and score high. The screening test for 'is there more than "
            "one absorber here'."
        )
    )
    score: float = Field(description="stain x mixing - both, because either alone is a bad tile")
    chosen: bool


class TileOut(APIModel):
    """The tile that was actually transformed, and how it was arrived at."""

    x: int
    y: int
    span: int = Field(description="Extent read, in level-0 pixels")
    size: int = Field(description="Side of the tile in its own pixels")
    mpp: float
    level: int
    downsample: float
    resampled: bool

    tissue_share: float
    considered_share: float
    stain: float
    mixing: float
    score: float
    rank: int = Field(description="Position in the scored list, 1 being the best")
    requested: bool = Field(
        description=(
            "True when the caller named a position and the chooser snapped to the nearest "
            "scored block, rather than taking the best."
        )
    )
    candidates_scored: int


class DensityStatsOut(APIModel):
    """Where the densities landed.

    Per channel and in total, because the two answer different questions: the
    channels are what step 6 will un-mix, and the total is what a reader means by
    'how much stain is in this pixel'.
    """

    median: Channels
    p99: Channels
    maximum: Channels

    mean_median: float = Field(description="Median of the mean-across-channels density")
    mean_p99: float
    mean_maximum: float
    mean_minimum: float = Field(
        description="Below zero when some pixel of the tile is brighter than I0"
    )

    histogram: list[int] = Field(description="Counts of the mean density, over the range below")
    histogram_low: float = Field(
        description=(
            "Read off the data rather than fixed at zero: the interesting spread is a few "
            "tenths of a density, and a fixed 0-3 axis would put the whole tile in two bins."
        )
    )
    histogram_high: float


class LimitsOut(APIModel):
    """The three places a density is not a reading, and the inverse that proves the rest is."""

    floor_share: float = Field(
        description=(
            "Share of pixels where a channel recorded no light at all. Their densities are "
            "lower bounds - the true value is larger and unknowable."
        )
    )
    negative_share: float = Field(
        description=(
            "Share of pixels brighter than I0, whose density is therefore negative. Negative "
            "stain does not exist; it means I0 is slightly low here."
        )
    )
    negative_worst: float
    transparent_share: float = Field(
        description="Share below beta, excluded from the cloud as having no reliable direction"
    )
    beta: float

    roundtrip_max: float = Field(
        description=(
            "Worst error, in intensity levels, of applying the exact inverse I = I0 x "
            "10^(-OD) back to the density. This step is a change of units, not a filter, and "
            "this is the number that says so."
        )
    )
    roundtrip_mean: float
    roundtrip_exact_share: float = Field(
        description=(
            "Share of pixels the inverse returns to within half a level - exactly, at 8 bits"
        )
    )


class ArmOut(APIModel):
    """One edge of the wedge: a direction the data says a stain lies along."""

    angle: float = Field(description="Angle in the plotted plane, in radians")
    vector: Channels = Field(description="The direction in three dimensions, as a unit OD vector")
    share: float = Field(description="Share of plotted pixels nearer this arm than the other")
    plot_x: float
    plot_y: float
    nearest: str = Field(description="Which published vector it lands closest to")
    degrees_from_nearest: float = Field(
        description=(
            "How close, in three dimensions and not in the projection. This is the number "
            "that turns 'the arms are the stains' from a claim into a measurement."
        )
    )


class ReferenceOut(APIModel):
    """A published stain vector, shown on the same axes as the arms."""

    name: str
    vector: Channels
    plot_x: float
    plot_y: float
    out_of_plane: float = Field(
        description=(
            "How much of this unit vector points out of the plotted plane. A reference drawn "
            "near an arm but with a large value here is a shadow, not a match - which is why "
            "the number travels with the drawing."
        )
    )
    degrees_from_arm: float = Field(description="Degrees from the nearer of the two arms")


class CloudOut(APIModel):
    """The densities as a point cloud, and the geometry read off it.

    The whole of colour deconvolution is visible here: two rays from the origin,
    the origin being zero stain, and a wedge of mixtures between them. Step 6
    inverts exactly this geometry, so a reader who has seen it does not need to be
    told that un-mixing is possible.
    """

    basis: list[Channels] = Field(
        description=(
            "The two directions the cloud is drawn on - the plane holding most of its "
            "energy. Found without centring the data, because the cloud emanates from the "
            "origin and the origin is a point the plane has to contain."
        )
    )
    explained: float = Field(
        description=(
            "Share of the cloud's energy the plane holds. Near 1 means it really is planar, "
            "which is what two absorbers produce; well below means three or more, and the "
            "flat drawing is a projection to distrust."
        )
    )
    plotted: int
    admitted_share: float = Field(description="Share of the tile that reached the cloud")
    faint_share: float = Field(
        description="Turned away for carrying less than beta - too little stain to have a direction"
    )
    unstable_share: float = Field(
        description=(
            "Turned away for being too dark to resolve - a channel so absorbed that one "
            "8-bit level would swing the pixel's direction past the tolerance."
        )
    )
    tolerance_deg: float

    x_low: float
    x_high: float
    y_low: float
    y_high: float

    arms: list[ArmOut]
    references: list[ReferenceOut]
    separation: float = Field(description="Degrees between the two arms, in three dimensions")
    two_armed: bool = Field(
        description=(
            "False when the two reported arms are the two tails of a single lobe - what a "
            "tile carrying only a counterstain looks like. A fact to report, not to hide."
        )
    )

    angles: list[int] = Field(
        description=(
            "Counts by angle across the wedge - the cloud's shape counted rather than "
            "drawn. Where the tile holds regions of near-pure stain the distribution has a "
            "mode at each arm; where it is all mixtures it has one broad hump between them."
        )
    )
    angle_low: float
    angle_high: float
    angle_centre: float = Field(
        description="The angle the wedge is measured about, for placing the counts on an axis"
    )

    reference_separation: float = Field(
        description=(
            "Degrees between Ruifrok's own haematoxylin and DAB vectors - the same "
            "measurement as `separation`, on the published pair. Reported so the wedge's "
            "width can be read against something: much wider suggests a third absorber or "
            "extremes reaching into noise, much narrower that the two stains overlap here."
        )
    )


class DriftBin(APIModel):
    """One concentration band of one arm, and where it points in each space."""

    density: float = Field(description="Median mean-density of the band")
    pixels: int
    od_degrees: float = Field(description="Degrees from the faintest band, in density space")
    intensity_degrees: float = Field(
        description="Degrees from the faintest band, in intensity space"
    )


class AdditivityOut(APIModel):
    """Beer-Lambert, measured on this tile rather than asserted.

    One stain at two concentrations is one direction in density space and two
    directions in intensity space. That is the single fact that makes colour
    deconvolution valid, and it is a testable statement about the reader's own
    slide, so it is tested.
    """

    arm: str = Field(description="Which arm was used - the one holding more of the cloud")
    bins: list[DriftBin]
    od_drift: float = Field(
        description=(
            "Degrees the direction turns between the faintest and darkest band in density "
            "space. Near zero is the linearity step 6 needs."
        )
    )
    intensity_drift: float = Field(
        description=(
            "The same in intensity space, where I = I0 x 10^(-cv) curves. Larger, and the "
            "gap between the two is the reason for the logarithm."
        )
    )
    pixels: int


class DensityStaining(APIModel):
    """Which dyes this section carries, decided from the point cloud above.

    **On the screen that owns the evidence.** Step 7 gates its H&E branch on this, and
    it quotes this verdict rather than deriving one of its own - so the place a reader
    can see *why* that option is on or off is here, beside the arms it was read from.
    """

    staining: str = Field(
        description=(
            "`he` - two dye directions, one of them eosin. `haematoxylin_dab` - two "
            "directions, neither of them eosin. `single_stain` - one lobe, which is "
            "what a counterstain looks like on its own. `unknown` - not settled by "
            "this tile."
        )
    )
    is_he: bool = Field(description="Whether step 7's H&E branch is offered")
    reason: str = Field(description="One sentence, naming the number that decided")
    eosin_arm_degrees: float | None = Field(
        default=None,
        description=(
            "How far the arm nearest eosin sits from Ruifrok's published eosin vector. "
            "Null when no arm is nearest eosin at all - which is what every "
            "immunostained section in this project's panel measured."
        ),
    )
    tolerance_deg: float | None = Field(
        default=None, description="The angle the value above had to beat"
    )


class DensityReport(APIModel):
    """Everything step 5 produced for one tile of one slide."""

    upload_id: str
    filename: str
    generated_at: str

    params: DensityParams
    white: WhiteUsed
    tile: TileOut
    candidates: list[TileCandidate] = Field(
        description="The blocks that were scored, best first, so the choice can be checked"
    )

    stats: DensityStatsOut
    limits: LimitsOut
    cloud: CloudOut | None = Field(
        default=None,
        description="Null when too few pixels carried any stain for a cloud to mean anything",
    )
    additivity: AdditivityOut | None = Field(
        default=None, description="Null when the dominant arm had too few pixels to bin"
    )

    #: Which dyes this section carries - see `DensityStaining`. Null when this tile
    #: built no point cloud, in which case there is nothing to read a verdict from.
    staining: DensityStaining | None = Field(default=None)

    notes: list[str] = Field(
        default_factory=list,
        description="Caveats that belong on screen next to the numbers, not in a docstring",
    )
    citation: str
