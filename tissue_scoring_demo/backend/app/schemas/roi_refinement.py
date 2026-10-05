"""Schemas for step 11 - refining each chosen region per pixel.

The shape of this payload follows from one requirement: **a region's result has to be
readable the moment it exists, not when the last region finishes.** So the report is not
a summary written at the end - it is a live list, one entry per selected region, each
carrying its own state, its own numbers and the names of its own pictures. A client polls
it and draws whatever is finished.

That also makes retry and resume expressible rather than special. A region is a row; a
failed row carries its error and can be run again on its own; a finished row is not
recomputed because its row already holds the answer.

**The per-region states are a sequence, not a set.** `pending -> extracting ->
segmenting -> tracing -> complete`, with `failed` reachable from any of them. A screen
can therefore render the stage list as ticks and spinners from the state alone, without
the server also sending a percentage it does not really have - BEETLE reports completion
per window, and a window is seconds, so the honest progress for a region is which stage
it is in plus how many of its windows are done.

See `pipeline/step11_roi_refinement/`.
"""

from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel

RefinementState = Literal["queued", "running", "ready", "partial", "failed", "cancelled"]

RegionState = Literal[
    "pending",
    "extracting",
    "segmenting",
    "tracing",
    "complete",
    "failed",
    "skipped",
]


class RegionProgress(APIModel):
    """One region's own position, for the bar on its own row.

    Separate from `RefinedRegionModel` because it exists *before* that does. A region
    has a window count and a state from the moment the pass is queued; it has rings and
    an area only once it finishes. A screen that wanted to draw a bar per region from
    the report alone could not draw one for a region that had not started.

    `windows_total` is step 9's count for the region until the region starts, and the
    grid's own `restricted.windows` after. Those differ - step 9 counts the candidate's
    windows and step 11 counts the windows whose cores reach the *padded* box - so the
    total can move once, upward, at the moment a region begins. That is honest and it is
    also why `share` clamps: a bar must not be allowed past its own end by a count that
    was provisional.
    """

    roi_id: str
    index: int = 0
    state: RegionState = "pending"
    windows_done: int = 0
    windows_total: int = 0

    @property
    def share(self) -> float:
        """Done of total, clamped to 0..1. What a bar's width is set from."""
        if self.windows_total <= 0:
            return 0.0
        return min(1.0, self.windows_done / self.windows_total)


class RefinementPaint(APIModel):
    """Everything a screen needs to paint one region's segmentation as it happens.

    The same idea as step 8's `TissueTypePaint` and deliberately the same shape, because
    it feeds the same drawing code: BEETLE answers per pixel on both steps, and what
    arrives per window is a little mask of class ids rather than one class. The one
    difference is the frame. Step 8 paints the whole section, so its viewport is the
    slide; step 11 paints one padded box at a time, so the viewport is that box and the
    picture underneath is the region's own crop rather than a thumbnail.

    **The grid coordinates are still the slide's**, and that is not an oversight.
    `crop.restrict` narrows step 8's grid rather than building a new one, so a painted
    window's `(row, col)` indexes the slide-wide grid and its position is `x_of(col)`
    in level-0 pixels. Sending `span`, `stride` and the slide's size alongside the crop
    box lets a client reproduce `x_of` exactly - including its clamp at the last row and
    column - and then project into the box. Renumbering the windows per region here
    would mean reimplementing that clamp in two places.
    """

    roi_id: str = Field(description="The region this feed belongs to; a new one resets")

    #: The padded box being segmented, in level-0 pixels - the canvas the paint lands on
    #: and the extent of the region's `input.png`.
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0

    #: The window grid, in level-0 pixels. `stride` is the cell a window is responsible
    #: for and is what gets painted; `span` is what it was shown, and the two differ
    #: wherever step 7 chose an overlap.
    span: int = 0
    stride: int = 0
    slide_width: int = 0
    slide_height: int = 0

    colours: list[str] = Field(
        default_factory=list,
        description=(
            "Hex colour per BEETLE class, indexed by class id - the same palette "
            "`beetle_mask.png` is drawn in, so the live paint and the finished evidence "
            "panel are one picture rather than two renderings of one answer"
        ),
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Plain-language name per class id, for the legend beside the paint",
    )

    per_pixel: bool = Field(
        default=True,
        description=(
            "Always true on this step - step 11 is BEETLE only - and carried anyway so "
            "one client component serves both steps without a per-step branch"
        ),
    )
    mask_px: int = Field(
        default=0,
        description="Side of each window's mask in `paintedMasks`, in pixels",
    )


class RefinementRun(APIModel):
    """Live state of one slide's refinement pass. Polled while it runs."""

    upload_id: str
    state: RefinementState
    message: str | None = None
    #: Regions finished, of regions selected. The only progress figure that is real at
    #: the level of the whole pass.
    done: int = 0
    total: int = 0
    #: Windows finished, of windows the selection costs. Finer than `done`, and it is
    #: what makes a single large region stop looking stalled.
    windows_done: int = 0
    windows_total: int = 0
    #: The region being worked on right now, so the screen never has to guess.
    current_roi_id: str | None = None

    #: Every selected region's own position, in the order they will be run. What the
    #: bar on each row is drawn from, and it is present from the moment the pass is
    #: queued - so a region that has not started still has a row with an empty bar
    #: rather than no row at all.
    progress: list[RegionProgress] = Field(default_factory=list)

    #: Geometry for painting the region in progress. Null before the first region
    #: starts, and it changes when the pass moves on - `roi_id` is what says which
    #: region the feed belongs to, and a client that sees it change starts a new canvas.
    paint: RefinementPaint | None = None
    #: Windows segmented since `paintedSince`, as flat triples: row, column, dominant
    #: class. Empty unless `paintedSince` was asked for.
    painted_cells: list[int] = Field(default_factory=list)
    #: One base64 pixel mask per window in `painted_cells`, same order. Each decodes to
    #: `paint.mask_px` squared bytes of class ids - the shapes BEETLE found inside that
    #: window, which is what makes this a segmentation landing rather than a grid
    #: filling in.
    painted_masks: list[str] = Field(default_factory=list)
    #: How far into the current region's paint log `painted_cells` reaches. Pass it back
    #: as the next `paintedSince`. It counts windows, so it indexes `painted_masks`
    #: directly and `painted_cells` in threes - and it resets when the region does.
    painted_cursor: int = 0

    started_at: str | None = None
    finished_at: str | None = None
    duration: float | None = None
    error: str | None = None


class RefinedRegionModel(APIModel):
    """One region's before and after, and where it got to.

    `tileRings` and `pieces` are the comparison the step exists to show: the same tissue,
    the coarse staircase and the traced boundary, both in H&E level-0 pixels so a client
    can draw them over each other without a transform of its own.
    """

    roi_id: str
    index: int = Field(description="Step 9's own 0-based rank, largest first")
    state: RegionState
    error: str | None = None

    #: The padded box BEETLE was actually shown, in H&E level-0 pixels. Larger than the
    #: candidate's own box by `roi_refinement_pad_um` on each side; recorded because the
    #: mask is placed back into the slide from this origin and a result that lost it
    #: would be a shape with no position.
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0
    pad_um: float = 0.0
    #: Microns per pixel of the mask that was traced, and of the slide it sits on.
    mask_mpp: float = 0.0
    base_mpp: float = 0.0
    mask_width: int = 0
    mask_height: int = 0

    #: The coarse candidate, as step 9 traced it. The left-hand picture's geometry.
    tile_rings: list[list[tuple[int, int]]] = Field(default_factory=list)
    tile_area_mm2: float = 0.0

    #: What BEETLE drew. One entry per connected focus, and each focus is its own
    #: outer ring followed by its holes - the convention step 9 traces in, step 12
    #: warps in and `sampling.rasterise` fills in.
    #:
    #: **Grouped rather than flattened, and it has to be.** A single coarse box
    #: routinely refines into several separate foci. Flattened into one list of rings,
    #: the first focus would be read as the outer boundary and every other focus as a
    #: hole in it - so the tumour would come back as one region with its other foci
    #: punched out of it, which is both wrong and entirely plausible-looking.
    pieces: list[list[list[list[float]]]] = Field(default_factory=list)
    #: One area per entry in `pieces`, same order, counted from that focus's own mask
    #: pixels. **Not the region's area divided by the count**: step 13 splits its field
    #: budget between regions in proportion to their area, and an equal split would
    #: sample a 0.9 mm2 focus and a 0.01 mm2 one identically - which is the exact
    #: failure `sampling.allocate` exists to prevent, reintroduced one level down.
    piece_areas_mm2: list[float] = Field(default_factory=list)
    focus_count: int = 0
    holes: int = 0
    area_mm2: float = 0.0

    #: `area_mm2 / tile_area_mm2`. Under 1 on essentially every region, and that is the
    #: step working: a square drawn around tumour contains stroma, fat and glass, and
    #: this is how much of the coarse box was not actually invasive carcinoma.
    kept_share: float = 0.0

    windows: int = 0
    seconds: float = 0.0
    #: Share of the box's mask pixels each BEETLE class won, keyed by class name. The
    #: number that shows the refinement is a segmentation and not a rectangle.
    class_share: dict[str, float] = Field(default_factory=dict)

    def focus_areas(self) -> list[tuple[list[list[list[float]]], float]]:
        """Each focus with its own area, paired. The form step 12's input is built from.

        Falls back to an equal share when `piece_areas_mm2` is absent, which is what a
        region refined before the areas were stored looks like. Equal shares are wrong -
        that is the whole reason this field exists - but they are wrong in a way that
        preserves the total, and refusing to read an older result would mean recomputing
        minutes of BEETLE to recover a number already on disk.
        """
        if len(self.piece_areas_mm2) == len(self.pieces):
            return list(zip(self.pieces, self.piece_areas_mm2, strict=True))
        share = round(self.area_mm2 / max(1, len(self.pieces)), 5)
        return [(focus, share) for focus in self.pieces]

    @property
    def rings(self) -> list[list[list[float]]]:
        """Every ring of every focus, flat. For *stroking* an outline and nothing else.

        Safe for a client drawing lines, because a line does not care which ring is a
        hole. Never use it to fill, to rasterise or to warp - that is what `pieces` is
        for, and the distinction is the one this step must not lose.
        """
        return [ring for focus in self.pieces for ring in focus]


class RefinementReport(APIModel):
    """Step 11's result: the pixel-level invasive mask, region by region."""

    upload_id: str
    state: RefinementState
    generated_at: str

    slide_width: int
    slide_height: int

    #: The model that ran, and the geometry it ran at. Echoed for the same reason step 9
    #: echoes its parameters: a boundary traced at 1 um/px and one traced at 4 are
    #: different claims, and a reader who cannot see which cannot tell them apart.
    model: str | None = None
    field_of_view_um: float = 0.0
    mask_mpp: float = 0.0
    pad_um: float = 0.0
    folds: int = 0

    regions: list[RefinedRegionModel] = Field(default_factory=list)

    #: The combined mask, as totals. `refined_mm2` is the union of every complete
    #: region's traced area and is the denominator everything after this step measures
    #: inside.
    refined_mm2: float = 0.0
    tile_mm2: float = Field(
        default=0.0, description="The same regions as coarse boxes, for the comparison"
    )
    kept_share: float = 0.0
    #: All invasive carcinoma step 9 found, including regions nobody selected. What the
    #: refinement covers is `refined_mm2` against this, and the gap is tumour the score
    #: will not see.
    invasive_mm2: float = 0.0

    completed: int = 0
    failed: int = 0
    selected: int = 0

    seconds: float = 0.0
    windows: int = 0

    notes: list[str] = Field(default_factory=list)


class RefinementRequest(APIModel):
    """What to run, and what to leave alone."""

    roi_ids: list[str] | None = Field(
        default=None,
        description=(
            "Restrict this pass to these regions. Null runs every region step 10 has "
            "ticked. Used for retrying one failed region without disturbing the rest."
        ),
    )
    rebuild: bool = Field(
        default=False,
        description=(
            "Recompute regions that already finished. Off by default, which is what "
            "makes a restart cheap: a complete region's answer is already on disk and "
            "re-running it would spend minutes to reproduce it."
        ),
    )
