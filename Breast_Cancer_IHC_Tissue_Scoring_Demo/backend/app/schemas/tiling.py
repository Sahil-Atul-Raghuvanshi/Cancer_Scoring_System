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
