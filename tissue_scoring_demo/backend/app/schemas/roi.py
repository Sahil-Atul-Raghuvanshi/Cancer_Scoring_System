"""Schemas for step 9 - the ROI mask.

Step 9 reports three things:

  the region    how much tissue the score is now allowed to be measured on, in mm2
                and as a share, plus one entry per focus with its polygon. The product.
  the ledger    how the region got from step 8's raw windows to that number - what
                the threshold kept, what the close added, what the in-situ carve-out
                took back, what the small-component cutoff dropped. Four separate
                edits to the denominator, itemised, because a clean region that
                arrives without them cannot be checked.
  the params    every knob, echoed back. The same region built at a different closing
                radius is a different claim, and a reader who cannot see the radius
                cannot tell which claim they are looking at.

**The polygons are on the wire; the mask is not.** A boolean grid is step 10's input
and stays server-side, but the rings are what a viewer draws and what a pathologist
edits, and they are small - a few hundred vertices per focus rather than 252 x 252
cells. They are in level-0 slide pixels for the reason `RoiRegion` gives: a client
placing them on the scan should not have to know the window grid exists.

See `pipeline/step09_roi_mask/`.
"""

from enum import Enum

from pydantic import Field

from app.schemas.common import APIModel


class RoiPanel(str, Enum):
    """The panels the step is meant to be read as, in order.

    An enum rather than a free string so a typo is rejected at the API boundary with
    the valid names, rather than reaching the service and coming back as a conflict.
    """

    SEED = "seed"
    SMOOTHED = "smoothed"
    BINARY = "binary"
    REGION = "region"
    OUTLINE = "outline"
    BORDERS = "borders"
    BORDERS_ON_SLIDE = "borders_on_slide"


class RoiTopClass(str, Enum):
    """The two classes a top-3 crop exists for. Not `uncertain` - see `RoiReport`."""

    DCIS = "dcis"
    INVASIVE = "invasive"


class RoiBorderClass(str, Enum):
    """The three classes a selected-region crop exists for.

    Spelled the way `RoiReport.class_regions` keys them, not the way `RoiTopClass`
    does - a viewer clicking a region off the borders panel already has this exact
    string from that dict, uncertain included, so there is no second name to map
    through the way the top-3 crop's `dcis`/`invasive` shorthand needs.
    """

    DCIS = "non_invasive_epithelium"
    INVASIVE = "invasive_epithelium"
    UNCERTAIN = "uncertain"


class RoiParamsModel(APIModel):
    """The knobs the region was built with, echoed back with every report."""

    sigma: float = Field(description="Gaussian blur on P(invasive), in grid cells")
    threshold: float = Field(description="Where the smoothed probability becomes region")
    close_cells: int = Field(description="Closing radius in grid cells - the merge distance")
    protect_in_situ: float = Field(
        description=(
            "In-situ probability at which a window is carved back out of the closed "
            "region. Rule 5: in-situ carcinoma is not scored, and a closing radius "
            "large enough to merge nearby foci is large enough to swallow the ducts "
            "between them. Zero disables the carve-out and gives the plain recipe."
        )
    )
    min_area_mm2: float = Field(description="Components below this are dropped as speckle")
    keep_largest: int | None = Field(
        default=None, description="Keep only the N largest foci; null keeps all that survived"
    )


class RoiRegionModel(APIModel):
    """One focus of the ROI and the rings that draw it."""

    index: int
    cells: int = Field(description="Windows this focus covers")
    area_mm2: float
    holes: int = Field(description="Rings other than the outer one - tissue excluded from within")
    rings: list[list[tuple[int, int]]] = Field(
        description=(
            "Closed rings of (x, y) level-0 slide pixels, outer boundary first. "
            "Rectilinear: the ROI is a union of whole windows and the polygon does not "
            "pretend to sub-window precision."
        )
    )


class RoiLedger(APIModel):
    """What each stage of the recipe did to the denominator.

    Every field is a mm2 the score either gained or lost, and they are reported rather
    than netted off because they are not interchangeable: area added by a closing
    radius is a modelling choice, area removed by the in-situ carve-out is a clinical
    rule, and a single net number would hide both.
    """

    seed_mm2: float = Field(description="Over the threshold, before any morphology")
    seed_components: int = Field(description="Separate foci at that point")
    merged_mm2: float = Field(description="Added by the closing - the merge, priced")
    protected_mm2: float = Field(
        description="In-situ the close annexed and the carve-out took back (Rule 5)"
    )
    protected_cells: int
    dropped_mm2: float = Field(description="Removed as speckle or by keep_largest")
    dropped_components: int


class RoiReport(APIModel):
    """Step 9's payload: where scoring is allowed, and how that was decided."""

    upload_id: str
    generated_at: str

    area_mm2: float = Field(description="The ROI - the denominator every later step measures in")
    cells: int
    tissue_mm2: float = Field(description="Classified tissue from step 8, for the share below")
    roi_share: float = Field(description="ROI as a fraction of the classified tissue")

    slide_width: int = Field(
        description=(
            "The slide's width in level-0 pixels - the frame `regions[].rings` and "
            "`class_regions[][].rings` are in. A client places a region on the borders "
            "panel by dividing its ring coordinates by this and the height below, not "
            "by measuring the picture."
        )
    )
    slide_height: int = Field(description="The slide's height in level-0 pixels. See slide_width.")

    regions: list[RoiRegionModel]
    holes: int = Field(description="Total holes across all foci")

    ledger: RoiLedger
    params: RoiParamsModel

    #: A second, independent product: every connected patch of one tile class, read
    #: straight off step 8's own calls with no smoothing or morphology - unlike
    #: `regions` above, which is the single smoothed, closed, in-situ-protected
    #: scoring region. Keyed by step 8's class name (`non_invasive_epithelium`,
    #: `invasive_epithelium`, `uncertain`) - never `non_epithelium`, which gets no
    #: borders. Each list is sorted largest first, so `class_regions[name][:3]` is
    #: the top 3 the crop and QuPath exports are built from.
    class_regions: dict[str, list[RoiRegionModel]] = Field(default_factory=dict)
    #: Total area per class, in mm2 - the sum `class_regions[name]` would otherwise
    #: make every caller recompute.
    class_area_mm2: dict[str, float] = Field(default_factory=dict)

    notes: list[str] = Field(
        default_factory=list,
        description="What this region does and does not license, in the words the UI shows",
    )
