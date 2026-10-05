"""Schemas for step 7 - tiling.

Step 7 reports three things, and the first is the one the step exists for:

  the funnel   how many tiles the grid holds, how many are on tissue, and how
               many of those quality control left alone. Counts and not shares,
               because "412,164 became 24,908" is the compute bill for step 8
               and "94% dropped" is a statistic.
  the grid     the geometry those counts came from - size, stride, overlap, and
               the resolution, each stated in the units it is a claim about.
  the tiles    a sampled slice of the index, with the two shares each tile was
               judged on, so the rule can be checked rather than trusted.

**No tile pixels are in here, and that is the contract.** What this step produces
is addresses - level-0 origin, extent, level - and the pixels a tile resolves to
are step 6's haematoxylin channel, computed on demand through the one
deconvolution function this codebase has. A schema carrying tile images would put
a second, silent answer to "what does the model see" on the wire.

See `pipeline/step07_tiling/index.py`.
"""

from enum import Enum

from pydantic import Field

from app.schemas.common import APIModel


class TilingPanel(str, Enum):
    """The two panels the step is meant to be read as.

    An enum rather than a free string so a typo is rejected at the API boundary
    with the valid names, rather than reaching the service and coming back as a
    conflict - which is not what a typo is.
    """

    GRID = "grid"
    SAMPLE = "sample"


class TilingBranchName(str, Enum):
    """What step 7 can show the model, as the wire spells it.

    An enum for `TilingPanel`'s reason: an unknown branch is a typo and belongs at the
    API boundary with the valid names beside it, not deep in a service where it would
    surface as a conflict.

    `beetle` is here and has no checkpoint. Naming an option the pipeline does not
    implement is the honest way to describe scope - a reader can see that somebody
    else's network is a thing this demo could route to and has not - and the endpoint
    refuses it rather than pretending.
    """

    H_CHANNEL = "h_channel"
    HE = "he"
    BEETLE = "beetle"


class TilingStaining(APIModel):
    """Which dyes step 5 found, and therefore whether the H&E branch is offered.

    Every number that went into the verdict travels with it. A disabled option with no
    explanation is the worst kind of greyed-out button, and `reason` is written to be
    read on screen by somebody who does not know what a stain vector is.
    """

    staining: str = Field(
        description=(
            "`he` - two dye directions, one of them eosin. `haematoxylin_dab` - two "
            "directions, neither of them eosin, i.e. an immunostained section. "
            "`single_stain` - one lobe, which is what a counterstain looks like on its "
            "own. `unknown` - step 5 has not measured this slide, or measured something "
            "it cannot name."
        )
    )
    is_he: bool = Field(
        description="Whether step 7's H&E branch is offered for this slide"
    )
    reason: str = Field(
        description=(
            "One sentence, naming the number that decided. This is rendered verbatim "
            "under a disabled option, so it has to explain rather than announce."
        )
    )
    source: str = Field(
        description="`step5-cloud` when step 5 measured it, `step5-not-run` when it has not"
    )

    two_armed: bool | None = Field(
        default=None,
        description=(
            "Step 5's own test for a second dye: false when the two reported arms are "
            "the two tails of one lobe."
        ),
    )
    separation: float | None = Field(
        default=None, description="Degrees between the two arms"
    )
    reference_separation: float | None = Field(
        default=None,
        description=(
            "The same angle for Ruifrok's published haematoxylin-DAB pair, as the "
            "yardstick. Evidence rather than a criterion - a narrow wedge is worth "
            "noting and is not on its own a reason to refuse."
        ),
    )
    eosin_arm_degrees: float | None = Field(
        default=None,
        description=(
            "How far the arm nearest eosin actually sits from the published eosin "
            "vector. Null when no arm is nearest eosin at all."
        ),
    )
    tolerance_deg: float | None = Field(
        default=None, description="The angle the value above had to beat"
    )
    tile_x: int | None = Field(default=None, description="Which tile step 5 measured")
    tile_y: int | None = Field(default=None)


class TilingBranchOut(APIModel):
    """One of step 7's three options, and whether this slide can take it."""

    id: str
    label: str = Field(description="Plain language, for somebody who is not a pathologist")
    blurb: str = Field(description="One sentence on what the model is actually shown")

    implemented: bool = Field(
        description="False for `beetle`, which is named but not built"
    )
    enabled: bool = Field(
        description=(
            "Whether this slide permits the option. False for `he` on an immunostained "
            "section, because the branch needs a dye that is not there."
        )
    )
    reason: str | None = Field(
        default=None,
        description="Why not, in words, whenever `enabled` or `implemented` is false",
    )

    #: Per branch, because a field of view can have a head on one branch and not the
    #: other - which is exactly the state while the H&E models are being trained.
    fields_of_view: list["TilingFieldOfView"] = Field(default_factory=list)


class TilingSelection(APIModel):
    """The choice a viewer committed on step 7, as it is recorded on disk.

    Small on purpose. This is not the index - step 7 keeps no pixels and no grid on
    disk - it is the four numbers that say which grid to rebuild, so that step 8, step
    9 and the runner all rebuild the same one without being handed it.
    """

    branch: str
    field_of_view_um: float
    overlap: float
    tissue_threshold: int | None = None
    model: str | None = None
    tile_px: int | None = None
    mpp: float | None = None
    chosen_at: str | None = None


class TilingSelectionIn(APIModel):
    """The body of a commit. `overlap` and `threshold` fall back to the defaults."""

    branch: TilingBranchName = TilingBranchName.H_CHANNEL
    fov: float
    overlap: float | None = None
    threshold: int | None = None


class TilingBranches(APIModel):
    """What step 7 offers before any grid exists.

    A separate payload from `TilingReport` because the branch screen comes *before* a
    run: it has to say which options exist and why one is unavailable without reading a
    single slide pixel. Manifests and step 5's cached verdict, and nothing else.
    """

    upload_id: str
    filename: str
    generated_at: str

    staining: TilingStaining
    branches: list[TilingBranchOut] = Field(default_factory=list)
    default_branch: str

    notes: list[str] = Field(default_factory=list)
    citation: str


class TilingParams(APIModel):
    """The settings a run actually used, resolved rather than requested."""

    target_mpp: float = Field(
        description=(
            "Working resolution, in microns per pixel. Step 1's, not this step's - the "
            "working magnification is one decision for the whole pipeline."
        )
    )
    tile_size: int = Field(description="Side of a tile in its own pixels. The model's number")
    tile_um: float = Field(
        description=(
            "Side of a tile in microns - the physical field of view, which is the number "
            "that decides whether gland architecture fits inside one tile. It has to, "
            "because architecture is what separates DCIS from invasive carcinoma at step 8."
        )
    )
    overlap: float = Field(
        description=(
            "Fraction of its own extent a tile shares with its neighbour. A model has no "
            "context past a tile's edge, so its predictions there are its worst; overlap "
            "lets those edges be averaged away rather than stitched into visible seams. It "
            "costs compute quadratically - 25% overlap is 1.8x the tiles, 50% is 4x."
        )
    )
    field_of_view_um: float = Field(
        description=(
            "The field of view this grid was laid at, in microns, and **step 7's one "
            "consequential choice**: it decides which checkpoint step 8 runs, because a "
            "field of view is a property of the weights rather than a setting. Equal to "
            "`tile_um`, and named separately because that one describes the geometry "
            "while this one records the choice that produced it."
        )
    )
    model: str | None = Field(
        default=None,
        description=(
            "The published checkpoint fitted at this field of view, matched on its "
            "manifest's own `tile_px * mpp` rather than on its name. **Null when none is "
            "published yet**, in which case the grid above was laid on the planned "
            "geometry and step 8 cannot run at this field of view - it will say which "
            "file it wants."
        ),
    )
    branch: str = Field(
        default=TilingBranchName.H_CHANNEL.value,
        description=(
            "Which of step 7's options laid this grid. Recorded because it, and not "
            "only the field of view, decides which checkpoint step 8 runs: the two "
            "branches share all four geometries by construction, so `field_of_view_um` "
            "alone no longer names one head."
        ),
    )
    input_channel: str | None = Field(
        default=None,
        description=(
            "What the checkpoint above declares it is shown - `haematoxylin` for a "
            "deconvolved density plane, `rgb_he` for the colour photograph. Read from "
            "the manifest, so it is what will actually be fed rather than what was "
            "asked for. Null when no head is published at this choice."
        ),
    )

    span: int = Field(description="Level-0 pixels one tile covers")
    stride: int = Field(description="Level-0 pixels the grid steps between tiles")

    min_tissue_share: float = Field(
        description=(
            "Share of a tile that must be tissue to be kept. Deliberately far lower than "
            "step 5's, because the two steps ask opposite questions: step 5 needs one tile "
            "it can trust every pixel of, and step 8 needs all the tissue - a tile it never "
            "sees is a region it cannot classify, and a section's edge is exactly where "
            "tiles are half glass."
        )
    )
    min_clean_share: float = Field(
        description=(
            "Share of a tile's *tissue* that step 2 must have left in play. Over the tissue "
            "and not over the tile, because a tile that is half glass would otherwise be "
            "credited for the glass being clean - and glass has nothing on it to be wrong."
        )
    )

    mask_mpp: float = Field(
        description=(
            "Resolution of step 3's mask, which is what the shares were measured on. Coarser "
            "than a tile by design: measuring each tile's own tissue fraction would mean "
            "reading the whole slide at working magnification, which is the exact cost this "
            "step exists to avoid, for a number only ever compared against a threshold."
        )
    )
    share_quantisation: float = Field(
        description=(
            "How coarse that makes each share - one mask pixel as a fraction of a tile. "
            "Reported rather than hidden, because it is the honest precision of the two "
            "gates above."
        )
    )

    tissue_threshold: int = Field(description="The step 3 cut this index was built on")
    tissue_threshold_source: str = Field(description="'otsu', 'triangle' or 'manual'")
    qc_gated: bool = Field(description="Whether step 2's artefacts were excluded")
    qc_source: str | None = Field(default=None, description="'grandqc' or 'otsu'")


class TilingFieldOfView(APIModel):
    """One field of view step 7 offers, and whether a model exists for it.

    Sent for every offered value rather than only the chosen one, because the screen's
    control is a comparison: a reader deciding between 224 and 448 um is weighing a
    square count against a physical scale, and both halves have to be on screen at once.
    `available` is the honest half - an option whose head has not been trained yet is
    shown as unavailable rather than hidden, since hiding it would make the choice look
    smaller than it is.
    """

    um: float = Field(description="Microns of slide across one square")
    tile_px: int = Field(description="Side in the model's own pixels. 224 for both")
    mpp: float = Field(description="Microns per pixel this field of view is read at")
    model: str | None = Field(
        default=None, description="The checkpoint fitted at it, or null if none is"
    )
    available: bool = Field(
        description="Whether a checkpoint exists at this field of view, so step 8 can run"
    )


class TilingFunnel(APIModel):
    """How many tiles survived each gate - the step's headline.

    The whole reason tiling runs after steps 2 and 3 rather than before them. Every
    tile that leaves this step is a forward pass through the region model, which is
    the most expensive thing in the pipeline, so these three numbers are the
    compute budget for everything downstream.
    """

    every: int = Field(description="Tiles in the grid over the whole canvas, glass included")
    on_tissue: int = Field(description="Of those, the ones clearing the tissue gate")
    clean: int = Field(description="Of those, the ones clearing the artefact gate. The output")
    reduction: float = Field(description="How many times fewer tiles reach the model")


class TileOut(APIModel):
    """One tile of the index, addressed rather than stored.

    Bounds are fractions of the slide, not pixels, so a browser can draw the tile
    over a thumbnail of any size without knowing this step's resolution. Same
    contract as steps 4 and 5's patch bounds, for the same reason.
    """

    col: int
    row: int
    x: int = Field(description="Level-0 origin - the coordinate that does not move with resolution")
    y: int
    span: int

    fx: float
    fy: float
    fw: float
    fh: float

    tissue_share: float
    clean_share: float = Field(description="Share of this tile's tissue step 2 left in play")
    kept: bool
    rejected_by: str | None = Field(
        default=None, description="'tissue' or 'clean' - which gate turned it away"
    )


class TilingCoverage(APIModel):
    """What area the kept tiles actually cover.

    Reported beside the tile count because with overlap the two are not
    proportional, and the difference is a thing readers get wrong: `count x area`
    is four times the covered area at 50% overlap, which makes it look as though
    overlapping tiles see more of the slide. They do not - overlap buys reliable
    tile edges at the cost of more passes over the same tissue.
    """

    covered_mm2: float = Field(
        description="Unique slide area the kept tiles cover, counting overlap once"
    )
    tissue_mm2: float = Field(description="Area step 3 called tissue")
    coverage: float = Field(
        description=(
            "Covered over tissue. Above 1 because tiles are squares and a section is not - "
            "a tile clipping the edge brings some glass with it."
        )
    )


class TilingSample(APIModel):
    """The tile the sample panel draws, and where it came from."""

    x: int
    y: int
    span: int
    size: int
    mpp: float
    tissue_share: float


class TilingReport(APIModel):
    """Everything step 7 produced for one slide."""

    upload_id: str
    filename: str
    generated_at: str

    params: TilingParams
    #: Every field of view on offer *for the chosen branch*, chosen or not - see
    #: `TilingFieldOfView`. Scoped to the branch so nothing that reads this field
    #: today changes meaning.
    fields_of_view: list[TilingFieldOfView] = Field(default_factory=list)
    #: Which dyes step 5 found, carried here so the screen can explain a disabled
    #: option without a second request.
    staining: TilingStaining | None = Field(default=None)
    #: All three options with their availability, for the same reason.
    branches: list[TilingBranchOut] = Field(default_factory=list)
    funnel: TilingFunnel
    coverage: TilingCoverage

    cols: int
    rows: int

    tiles: list[TileOut] = Field(
        description=(
            "A sampled slice of the index, evenly spaced through the kept tiles rather than "
            "the first N of them - the grid is generated in row order, so the head of the "
            "list would be the top edge of the section and nothing else."
        )
    )
    listed: int = Field(description="How many tiles the list above holds")

    sample: TilingSample | None = Field(
        default=None, description="Null when no tile survived both gates"
    )

    notes: list[str] = Field(
        default_factory=list,
        description="Caveats that belong on screen next to the numbers, not in a docstring",
    )
    citation: str


TilingBranchOut.model_rebuild()
