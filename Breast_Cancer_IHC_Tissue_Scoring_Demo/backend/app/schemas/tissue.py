"""Schemas for step 3 - the tissue mask.

Step 3 reports three things that a reader has to be able to keep apart:

  the decision      one binary mask, and the area it covers.
  the threshold     Otsu's cut, and - when the viewer has moved the slider -
                    the manual cut beside it. Both are always present, so the
                    screen can never show a hand-picked number as if it were
                    the automatic one.
  the cleanup       the mask after each morphological move, in order, with the
                    area each one added or removed. A single before-and-after
                    would hide the fact that closing and opening pull in
                    opposite directions.

Nothing here reports a fat share, and that is deliberate. At this stage the only
question answered is "tissue or glass"; fat is tissue, and it leaves the
analysis at step 8 as a named class. See `pipeline/step03_tissue_mask/mask.py`.
"""

from enum import Enum

from pydantic import Field

from app.schemas.common import APIModel


class TissuePanel(str, Enum):
    """The four panels step 3 is meant to be read as, in that order.

    An enum rather than a free string so an unknown name is rejected at the API
    boundary with the list of valid ones, instead of reaching the service and
    coming back as a conflict - which is not what a typo is.
    """

    THUMBNAIL = "thumbnail"
    SATURATION = "saturation"
    MASK = "mask"
    OVERLAY = "overlay"


class TissueParams(APIModel):
    """The settings a run actually used, resolved rather than requested."""

    target_mpp: float = Field(description="Resolution asked for, in microns per pixel")
    mask_mpp: float = Field(
        description=(
            "Resolution achieved. Coarser than the target on a large slide, because the "
            "longest edge is capped - every physical threshold below is divided by this."
        )
    )
    mask_width: int
    mask_height: int
    capped: bool = Field(description="True when the size cap, not the target, set mask_mpp")

    close_um: float = Field(description="Morphological closing radius, in microns")
    open_um: float = Field(description="Morphological opening radius, in microns")
    close_px: int
    open_px: int
    min_component_mm2: float
    fill_hole_max_mm2: float

    qc_gated: bool = Field(
        description="Whether step 2's artefact map was subtracted before thresholding"
    )
    qc_source: str | None = Field(
        default=None,
        description="'grandqc' or 'otsu' - which tissue pass step 2 used, when it ran",
    )


class TissueThreshold(APIModel):
    """The cut, which rule produced it, and what the other rules said.

    Both automatic rules are always reported, whichever was used and even on a
    manual run. A threshold is the single most consequential number in this step,
    so the screen is never allowed to show one without its alternatives.
    """

    value: int = Field(ge=0, le=255, description="The threshold used, as a saturation level")
    source: str = Field(description="'otsu', 'triangle' or 'manual' - which rule set `value`")

    otsu: int = Field(ge=0, le=255, description="Otsu's answer, computed either way")
    triangle: int = Field(
        ge=0, le=255, description="Zack's triangle answer, computed either way"
    )

    modal_level: int = Field(description="The busiest saturation level")
    modal_share: float = Field(
        description=(
            "That level's share of the histogram. This is the statistic the rule choice "
            "turns on: a level holding most of the mass is a spike with no variance, and "
            "Otsu's criterion is a ratio of variances."
        )
    )
    spike_share: float = Field(description="Share at or above which Otsu is set aside")
    bimodal: bool = Field(
        description="modal_share < spike_share, i.e. Otsu's two-hump assumption holds"
    )

    triangle_from: int = Field(description="Chord start - the modal level - for drawing")
    triangle_to: int = Field(description="Chord end - the last occupied level")

    mean_below: float | None = Field(
        default=None, description="Mean saturation of the pixels called glass"
    )
    mean_above: float | None = Field(
        default=None, description="Mean saturation of the pixels called tissue"
    )
    separation: float | None = Field(
        default=None, description="mean_above - mean_below; the gap the cut bought"
    )
    variance_ratio: float = Field(
        description=(
            "Otsu's criterion at this threshold over its value at Otsu's own threshold. "
            "Meaningful only when `bimodal` is true: on a spike-and-tail histogram Otsu's "
            "criterion is the wrong yardstick, so a low ratio there says the cut disagrees "
            "with Otsu, not that the cut is worse."
        )
    )


class TissueHistogram(APIModel):
    """The saturation histogram the threshold was chosen from.

    Counted over the pixels step 2 left in play, not over the whole image. That
    restriction is the reason QC runs first: pen ink is more saturated than any
    stain, and left in the histogram it drags the cut towards the ink.
    """

    bins: list[int] = Field(description="256 counts, one per saturation level")
    counted_pixels: int
    excluded_pixels: int = Field(description="Pixels step 2 rejected, so never counted here")
    criterion: list[float] = Field(
        description=(
            "Otsu's between-class variance at each of the 256 levels, normalised to a "
            "0-1 peak. Drawing this over the histogram shows why the cut sits where it does."
        )
    )


class TissueStage(APIModel):
    """The mask after one move of the cleanup."""

    key: str = Field(description="'threshold', 'closing', 'opening', 'components' or 'fill'")
    label: str
    what: str = Field(description="What the move does, and why it is that way round")
    extent_um: float | None = Field(
        default=None, description="Physical extent for a morphological move"
    )
    pixels: int
    area_mm2: float
    delta_pixels: int = Field(description="Change from the previous move; negative removes area")
    delta_area_mm2: float
    delta_share: float = Field(description="Change as a fraction of the previous mask")


class TissueComponents(APIModel):
    """What connected-component filtering kept and dropped."""

    found: int
    kept: int
    dropped: int
    kept_pixels: int
    dropped_pixels: int
    dropped_area_mm2: float
    largest_pixels: int
    largest_area_mm2: float
    largest_share: float = Field(description="Largest component over all kept tissue")
    min_area_mm2: float
    min_area_px: int = Field(description="The same cutoff in pixels on this grid, for checking")


class TissueReport(APIModel):
    """Everything step 3 produced for one slide at one threshold."""

    upload_id: str
    filename: str
    generated_at: str

    params: TissueParams
    threshold: TissueThreshold
    histogram: TissueHistogram
    stages: list[TissueStage]
    components: TissueComponents

    tissue_pixels: int
    tissue_area_mm2: float
    slide_area_mm2: float
    tissue_share: float = Field(description="Tissue over the whole slide - glass included")
    glass_share: float = Field(
        description="The complement, and the area downstream steps no longer have to visit"
    )

    holes_filled_pixels: int
    holes_filled_area_mm2: float

    notes: list[str] = Field(
        default_factory=list,
        description="Caveats that belong on screen next to the numbers, not in a docstring",
    )
    citation: str
