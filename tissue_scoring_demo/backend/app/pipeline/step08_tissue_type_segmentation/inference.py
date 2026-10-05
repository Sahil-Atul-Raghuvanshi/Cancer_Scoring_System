"""Step 8's algorithm: a class per tile, over the tissue step 7 kept.

This step trains nothing. It loads the checkpoint the build block produced and runs it
forward, which makes the interesting decisions here geometric and arithmetic rather
than statistical.

**This step's grid is step 7's grid.** The region model takes **224 px at the
checkpoint's mpp**, and that is not a preference anything is free to override: it is a
property of the checkpoint, recorded in its manifest, and feeding the model a
differently-scaled field is the same class of failure as feeding it the wrong channel.
So rather than this step laying its own window grid over step 7's coarser tiles, step 7
reads the manifest and lays that square in the first place - see `tiling_service.window`.

The arrangement it replaces was two grids: 512 px tiles here, 224 px windows there,
joined by a centre test. It worked, and it produced a screen that priced 4,952 squares
followed by a screen that made 98,262 passes over the same slide, gated at 0.1 tissue
on one and 0.5 on the other. Nothing about the model required that; it was only that
step 7's tile had been sized for its own display.

What each step still decides:

  step 7 says **where to look** - its two gates are what turn a whole-slide grid into
         a few thousand squares, and every window here inherits that. A window whose
         centre does not land on a kept tile is never read, so quality control and the
         tissue mask still decide the compute bill. It also owns the overlap.
  step 8 says **what the squares mean** - it loads the checkpoint, runs it forward and
         writes a class per square. The geometry it once owned now reaches it through
         step 7, which read it from the same manifest this step loads.

**`build_grid` still exists, and is not now redundant.** With the two grids aligned it
reproduces step 7's own grid and its gate removes nothing. What it covers is the case
where they are not aligned - a caller naming an overlap of its own on this step's
route - and it is also what makes the alignment checkable rather than assumed:
`test_the_two_grids_coincide_when_step_7_lays_the_models_window` pins that the windows
it plans are exactly the tiles step 7 kept.

`build_grid` is the join between them, and `inside` is the only thing it decides.

**The pixels come from the same functions training used.** `read_window` is
supplied by the caller and composes step 5's tile read, step 4's white point and step
6's `separate` - the composition `tiling_service` already performs for its sample
panel. Nothing here recomputes a density or a stain basis; if it did, the model would
be served a different definition of "haematoxylin channel" from the one it was fitted
on, which is the failure `input.py` exists to prevent.

**Blocks, not windows, are what get read.** A window is 224 px and a `read_region` off
the pyramid costs far more than 224 px of pixels, so windows are read in square blocks
of `block_windows` x `block_windows` of them: one read, one deconvolution, up to 64
windows cut out of the result. Blocks holding no kept window are never read at all,
which is what keeps the pass proportional to the tissue rather than to the canvas. The
block is a read amortisation and nothing else - a window's pixels are identical either
way, and `test_a_block_read_gives_the_same_windows_as_reading_each_one` pins that.

**One prediction per window, and no stitching.** Overlapping windows are not several
predictions of one place to be averaged here - that averaging is step 10's first act,
and doing it twice would smooth the map before step 10 could show what raw model
output actually looks like. What leaves this step is exactly what the model said, per
window.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from app.pipeline.contract import RunCancelled
from app.pipeline.step07_tiling.index import TileIndex

from . import input as model_input
from .classes import CLASS_NAMES, SCORED
from .uncertainty import UNCERTAIN, UncertaintyLayer

#: Label written where no window ran. Not a class - the map is defined only over the
#: tissue step 7 kept, and a "background" class would be a fourth output the model
#: does not have and cannot be scored on.
OUTSIDE: int = -1

#: Windows per axis in one read. 8 is 64 windows and about a megapixel per read at the
#: model's own resolution, which is where the per-read overhead stops dominating.
BLOCK_WINDOWS: int = 8


def sweep(rows: int, cols: int, block_windows: int) -> list[tuple[int, int]]:
    """Block origins in the order the pass works through them: a ploughed field.

    Left to right along the first band of blocks, right to left along the next, and so
    on top to bottom. `(row0, col0)` per block, in visiting order.

    **The order does not change a single label.** A block is an independent read and an
    independent set of forward passes, so what this decides is only *when* each part of
    the section is answered - and therefore what someone watching the progress screen
    sees, since that screen paints each block onto the slide as it comes back. Plain
    row-major - every band left to right - was the order before this, and it makes the
    painted edge jump the whole width of the section between bands. Reversing alternate
    bands removes the jump, so the sweep reads as one continuous line of work.

    It is also the marginally kinder order for the slide reader: the next block read is
    next to the last one at a band's turn rather than a slide away, so whatever the
    pyramid has cached is more likely to be the part being asked for again.
    """
    origins: list[tuple[int, int]] = []
    for band, row0 in enumerate(range(0, max(1, rows), block_windows)):
        columns = list(range(0, max(1, cols), block_windows))
        if band % 2:
            columns.reverse()
        origins.extend((row0, col0) for col0 in columns)
    return origins


class SegmentationError(ValueError):
    """This slide cannot be classified - and why, in words."""


@dataclass(frozen=True)
class WindowGrid:
    """Step 8's own grid: where the model's field of view lands, and where it runs.

    Two coordinate systems, and both are needed. The window's **own** pixels are what
    the model sees and what a block is cut up in; **level-0** pixels are how a position
    on the slide is addressed, because that is the only frame of reference that does not
    move when a pyramid level is chosen. Every geometric quantity below says which one
    it is in, and the conversion happens once, here, rather than at each use.
    """

    cols: int
    rows: int

    #: Window side in its own pixels - the model's `tile_px`, from the manifest.
    size: int
    #: Step between windows in their own pixels. The guide's "stride 112" for a 224 px
    #: window at 50% overlap.
    stride_px: int

    #: The same two extents in level-0 pixels.
    span: int
    stride: int

    overlap: float
    #: Resolution a window is read at, in microns per pixel. The manifest's, never a
    #: setting - it is a property of the checkpoint.
    mpp: float
    #: The slide's own resolution, which is what turns a level-0 extent into microns.
    base_mpp: float
    slide_width: int
    slide_height: int

    #: True where a window will actually be classified: its centre lands on a tile
    #: step 7 kept **and** it holds enough tissue of its own. The join between the two
    #: grids, and the only thing `build_grid` decides.
    inside: np.ndarray

    #: Share of a window that must be tissue before the model is shown it.
    min_tissue_share: float = 0.0
    #: Windows whose centre was on kept tissue but which the gate above turned away.
    #: Reported rather than absorbed, because it is the difference between "the model
    #: did not look there" and "there was nothing there to look at".
    gated_out: int = 0

    @property
    def field_um(self) -> float:
        """The window's field of view in microns - what the model was fitted at."""
        return self.size * self.mpp

    @property
    def windows(self) -> int:
        """How many forward passes this grid implies. The step's compute bill."""
        return int(self.inside.sum())

    @property
    def every(self) -> int:
        return self.cols * self.rows

    def x_of(self, col: int) -> int:
        """Level-0 origin of a column, clamped to the canvas.

        The clamp matters at the far edge: without it the last column would ask for
        pixels past the canvas, and the reader would either pad or refuse. A padded
        window is a window of invented tissue.
        """
        return int(min(col * self.stride, max(0, self.slide_width - self.span)))

    def y_of(self, row: int) -> int:
        return int(min(row * self.stride, max(0, self.slide_height - self.span)))

    @property
    def cell_mm2(self) -> float:
        """Slide area one window is *responsible* for, in mm2.

        `stride`, not `span`, and that is the whole point. Windows overlap, so their
        spans do not partition the tissue and summing them would report four times the
        ground at 50% overlap - which would make it look as though overlapping windows
        see more of the slide. They do not; they see the same tissue more often. The
        stride-sized cores do partition it, one per window, so an area built from this
        is a real area and does not move when the overlap does. Step 7 draws the same
        distinction for the same reason.
        """
        return (self.stride * self.base_mpp / 1000.0) ** 2


def build_grid(
    index: TileIndex,
    *,
    base_mpp: float,
    slide_size: tuple[int, int],
    size: int,
    mpp: float,
    overlap: float,
    max_windows: int,
    tissue: np.ndarray | None = None,
    mask_mpp: float | None = None,
    min_tissue_share: float = 0.0,
) -> WindowGrid:
    """Lay the model's own grid over the slide and mark where the model will run.

    Two gates, and the second one exists because of a measurement.

    **The centre test.** A window is a candidate when its **centre** falls inside a kept
    step-7 tile, rather than when it overlaps one at all. Overlap would spill the model
    half a window past the section on every side, while the centre test keeps the run
    inside the region step 7 audited and still covers it to within half a window.

    **The window's own tissue share.** The centre test is not enough, and the reason is
    arithmetic rather than opinion: step 7's tile gate is deliberately low - 10%,
    because "step 8 needs all the tissue" - and its tiles are 512 px against this step's
    224 px. So a window whose centre sits in a kept tile can itself be almost entirely
    glass. Measured on `CAN_00251_26_A`: **the median window classified as in-situ
    epithelium was 11% tissue and 89% glass**, and 69% of them were under half tissue,
    against a median of 100% for the non-epithelium class. The class map's in-situ
    "structures" were a one-window rim tracing the section's edge, and raising this gate
    to 0.5 removed two thirds of them while dropping only 22% of windows.

    That is a false positive with a specific cause - a mostly-empty window standardised
    against its own weak p99 acquires texture the slide does not have - and it belongs
    to step 8 rather than step 7, because step 7's gate is right for step 7's purpose.
    Hence a gate of this step's own, on the window rather than on the tile.

    `tissue` is step 3's mask and `mask_mpp` its resolution. With no mask the gate cannot
    be applied and only the centre test runs, which is the old behaviour and is what the
    geometry tests use.

    `max_windows` is a refusal and not a cap, for step 7's reason: past this the grid
    means either a corrupt slide dimension or an overlap nobody meant to ask for, and
    truncating would hand step 9 an arbitrary part of a section with no record of which
    part.
    """
    if size < 16:
        raise SegmentationError("a window smaller than 16 px is not a field of view")
    if not 0.0 <= overlap < 1.0:
        raise SegmentationError(
            f"overlap must be at least 0 and less than 1, not {overlap}; at 1 the stride "
            "would be zero and the grid would never advance"
        )
    if mpp <= 0 or base_mpp <= 0:
        raise SegmentationError("a resolution of zero microns per pixel is not a resolution")

    width, height = int(slide_size[0]), int(slide_size[1])
    if width <= 0 or height <= 0:
        raise SegmentationError("this slide reports no dimensions, so there is nothing to read")

    # Derived in the window's own pixels first and converted once, so a window's offset
    # inside a block is an exact multiple of `stride_px` rather than a rounded one. A
    # half-pixel drift between training tiles and served tiles is the kind of error
    # that shows up only as a slightly worse model.
    stride_px = max(1, int(round(size * (1.0 - overlap))))
    to_level0 = mpp / base_mpp
    span = max(1, int(round(size * to_level0)))
    stride = max(1, int(round(stride_px * to_level0)))

    # Ceiling division and then a clamp, exactly as step 7 does it, so the last partial
    # window at each edge is still a window. A floor would drop a strip up to one window
    # wide down two sides of every slide, and the invasive front tends to sit at the
    # section's edge.
    cols = max(1, int(math.ceil(max(1, width - span) / stride)) + 1)
    rows = max(1, int(math.ceil(max(1, height - span) / stride)) + 1)

    if cols * rows > max_windows:
        raise SegmentationError(
            f"this grid would hold {cols * rows:,} windows, past the {max_windows:,} this "
            f"step will build. At {mpp:g} um/px with {overlap:.0%} overlap that is "
            f"{cols:,} x {rows:,} over a {width:,} x {height:,} slide. Refusing rather "
            "than truncating - an arbitrary part of a section, with no record of which "
            "part, is worse than no class map at all"
        )

    grid = WindowGrid(
        cols=cols,
        rows=rows,
        size=int(size),
        stride_px=stride_px,
        span=span,
        stride=stride,
        overlap=float(overlap),
        mpp=float(mpp),
        base_mpp=float(base_mpp),
        slide_width=width,
        slide_height=height,
        inside=np.zeros((rows, cols), dtype=bool),
    )

    inside = _mark_inside(index, grid)
    candidates = int(inside.sum())

    if tissue is not None and mask_mpp and min_tissue_share > 0.0:
        # Parenthesised: `&` binds tighter than `>=`, so without them this would
        # compare the *masked share* against the threshold and gate nothing.
        inside = inside & (_tissue_share(grid, tissue, mask_mpp) >= min_tissue_share)
        grid = replace(
            grid,
            min_tissue_share=float(min_tissue_share),
            gated_out=candidates - int(inside.sum()),
        )

    grid = replace(grid, inside=inside)

    if grid.windows == 0:
        raise SegmentationError(
            "no window of the model's own grid both lands on a tile step 7 kept and "
            f"holds at least {min_tissue_share:.0%} tissue, so there is nothing to "
            "classify. That happens when the section is smaller than one "
            f"{grid.field_um:g} um field of view, when step 3's threshold has claimed "
            "almost nothing as tissue, or when this step's own tissue gate is set so "
            "high that nothing clears it"
        )

    return grid


def _tissue_share(grid: WindowGrid, tissue: np.ndarray, mask_mpp: float) -> np.ndarray:
    """What share of each window step 3 called tissue, for the whole grid at once.

    Through a summed-area table rather than a slice per window: a whole-slide grid is a
    quarter of a million cells and a Python loop over them costs more than the forward
    passes this gate exists to save. One cumulative sum, then four lookups per window.

    Measured on step 3's mask rather than on the window's own pixels, for step 7's
    reason: reading every window at working magnification to compute a number that is
    only ever compared against a threshold is the exact cost this gate exists to avoid.
    """
    height, width = tissue.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    np.cumsum(np.cumsum(tissue.astype(np.int64), axis=0), axis=1, out=integral[1:, 1:])

    to_mask = grid.base_mpp / mask_mpp
    columns = np.array([grid.x_of(col) for col in range(grid.cols)], dtype=np.float64)
    lines = np.array([grid.y_of(row) for row in range(grid.rows)], dtype=np.float64)

    def bounds(origins: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        low = np.clip(np.floor(origins * to_mask).astype(np.int64), 0, limit)
        high = np.clip(
            np.ceil((origins + grid.span) * to_mask).astype(np.int64), 0, limit
        )
        return low, np.maximum(high, low + 1).clip(max=limit)

    x0, x1 = bounds(columns, width)
    y0, y1 = bounds(lines, height)

    # (rows, cols) via broadcasting: rows index the first axis, columns the second.
    total = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float64)
    return (total / np.maximum(area, 1.0)).astype(np.float32)


def _mark_inside(index: TileIndex, grid: WindowGrid) -> np.ndarray:
    """Which windows have their centre on a kept step-7 tile.

    **When the two grids coincide, the answer is step 7's own kept mask and the centre
    test is not merely redundant but wrong.** Step 7 lays this model's window, so
    normally every cell here is a cell there; asking "does this centre land in *any*
    kept tile" then admits windows step 7 rejected, because tiles overlap each other
    and a rejected tile's centre sits inside its kept neighbours. Measured on a section
    with an artefact band through it at 50% overlap: 26 windows planned against 20
    tiles kept, and the 6 extra were exactly the tiles step 2 had flagged and step 7
    had dropped for it. Cell-for-cell identity is the correct join, and it also makes
    the QC gate reach this step, which "any kept tile" quietly undid.

    Marked from the kept tiles outward rather than tested per window, because a slide
    has a few thousand of the first and a few hundred thousand of the second. For a
    kept tile covering level-0 `[tx, tx + tile.span)`, the window columns whose centre
    lands inside it are a contiguous integer range on the unclamped line, so each tile
    costs one slice assignment - and the two clamped edge indices are then tested
    directly, because their origin is not `index * stride`.
    """
    if (
        (grid.cols, grid.rows) == (index.cols, index.rows)
        and (grid.span, grid.stride) == (index.span, index.stride)
    ):
        return index.grid_kept.copy()

    inside = np.zeros((grid.rows, grid.cols), dtype=bool)
    half = grid.span / 2.0

    def indices(low: float, high: float, count: int) -> tuple[int, int]:
        """Indices `i` with `low <= i * stride < high`, inclusive at both ends."""
        first = max(0, int(math.ceil(low / grid.stride)))
        final = min(count - 1, int(math.ceil(high / grid.stride)) - 1)
        return first, final

    last_col_x = grid.x_of(grid.cols - 1)
    last_row_y = grid.y_of(grid.rows - 1)

    for tile in index.tiles:
        if not tile.kept:
            continue

        # A window's centre is at `origin + half`, so it lands in `[tx, tx + span_7)`
        # exactly when its origin lands in `[tx - half, tx + span_7 - half)`.
        c0, c1 = indices(tile.x - half, tile.x + tile.span - half, grid.cols)
        r0, r1 = indices(tile.y - half, tile.y + tile.span - half, grid.rows)

        col_hit = c1 >= c0
        row_hit = r1 >= r0
        last_col_hit = tile.x - half <= last_col_x < tile.x + tile.span - half
        last_row_hit = tile.y - half <= last_row_y < tile.y + tile.span - half

        columns = slice(c0, c1 + 1) if col_hit else None
        lines = slice(r0, r1 + 1) if row_hit else None

        if columns is not None and lines is not None:
            inside[lines, columns] = True
        if last_col_hit and lines is not None:
            inside[lines, grid.cols - 1] = True
        if last_row_hit and columns is not None:
            inside[grid.rows - 1, columns] = True
        if last_col_hit and last_row_hit:
            inside[grid.rows - 1, grid.cols - 1] = True

    return inside


# --- the class map -----------------------------------------------------------


@dataclass(frozen=True)
class ClassMap:
    """Step 8's output: one class and one probability vector per window.

    `labels` and `probabilities` are the step's product; everything else is the audit
    trail that makes the product readable - how many passes it took, how long it ran,
    and how the classes divide the tissue.
    """

    grid: WindowGrid

    #: (rows, cols) int8. `OUTSIDE` where no window ran - not a fourth class.
    labels: np.ndarray
    #: (rows, cols, 3) float32 softmax, zero outside. Kept because step 10 stitches
    #: *probabilities* and not labels: averaging argmaxes is a vote, and a vote
    #: discards exactly the confidence a smoothing step needs.
    probabilities: np.ndarray

    #: Windows classified, per class, indexed like `CLASS_NAMES`.
    counts: tuple[int, int, int]
    #: Unique slide area per class, in mm2, from `grid.cell_mm2`.
    areas_mm2: tuple[float, float, float]

    batches: int
    blocks_read: int
    seconds: float

    #: The post-pass in `uncertainty.py`, or None on a map that predates it. Attached
    #: with `dataclasses.replace` after `classify` returns rather than produced by it:
    #: two of its three terms are functions of a window's *neighbours* and the third of
    #: the connected component it lands in, so none of them exists until the last window
    #: has been classified.
    #:
    #: It is deliberately not part of `counts`, `areas_mm2`, `shares` or
    #: `tumour_content`. Those describe what the model said, they are what step 9 and
    #: the score are built from, and a derived flag must not be able to move them.
    uncertainty: UncertaintyLayer | None = None

    @property
    def classified(self) -> int:
        return int(sum(self.counts))

    @property
    def display_labels(self) -> np.ndarray:
        """`labels`, with the flagged in-situ windows recoloured `UNCERTAIN`.

        What the map draws and what the class bars are tabulated over. `labels` itself is
        left alone - it is the model's answer, it is what `classmap.npz` stores and what
        step 9 reads, and the whole safety argument for this layer is that the two arrays
        differ only on windows Rule 5 had already excluded.
        """
        if self.uncertainty is None:
            return self.labels
        return np.where(self.uncertainty.unknown, np.int8(UNCERTAIN), self.labels)

    @property
    def shares(self) -> tuple[float, float, float]:
        total = max(1, self.classified)
        return tuple(count / total for count in self.counts)  # type: ignore[return-value]

    @property
    def tumour_content(self) -> float:
        """Share of the classified tissue the score will actually be measured on.

        The number the guide asks this step to be stratified by, and the one that makes
        the point of the whole step: not "how much tumour" but how much *invasive*
        tumour, with in-situ disease and fat already out of the denominator.
        """
        return self.shares[SCORED]

    @property
    def scored_mm2(self) -> float:
        return self.areas_mm2[SCORED]

    @property
    def mean_confidence(self) -> float:
        """Mean top-class probability over the classified windows.

        Reported because a class map is an argmax and an argmax hides its own margin: a
        window at 0.34/0.33/0.33 and a window at 0.99 are the same colour on the
        overlay and are not the same claim. Step 10 thresholds probabilities for that
        reason, and this is the number that says how much room it has.
        """
        if self.classified == 0:
            return 0.0
        return float(self.probabilities.max(axis=-1)[self.grid.inside].mean())

    def confidence(self) -> np.ndarray:
        """(rows, cols) float32 top-class probability, zero outside. For the panel."""
        top = self.probabilities.max(axis=-1).astype(np.float32)
        return np.where(self.grid.inside, top, 0.0).astype(np.float32)


@dataclass(frozen=True)
class Block:
    """One read: where it sits, and where each of its windows sits inside it.

    Separated from the reading so the geometry can be tested without a slide. The
    offsets are computed from each window's *clamped* level-0 origin rather than from
    its column index, which is what keeps the last column - the one whose origin is not
    `col * stride` - cut from the right pixels.
    """

    x: int
    y: int
    #: Level-0 extent read, and the side it is resampled to in the window's own pixels.
    span: int
    size: int

    #: Per wanted window: its (row, col) on the grid, and its (top, left) offset in the
    #: block's own pixels.
    cells: tuple[tuple[int, int], ...]
    offsets: tuple[tuple[int, int], ...]


def plan_block(grid: WindowGrid, *, row0: int, col0: int, row1: int, col1: int) -> Block | None:
    """Where to read for the windows of one block, and where they land in it.

    Returns None when the block holds no window step 7 kept, which is the case that
    makes this pass cost what the tissue costs rather than what the canvas costs.

    The block is anchored on its own windows' extremes rather than on a round number:
    the first window's origin to the last window's far edge is exactly what has to be
    read, and reading a round extent instead would put the windows at fractional
    offsets inside it. Squared off - `read_window` takes one extent, because step
    5's `read_tile` does - and then nudged back inside the canvas if squaring pushed it
    over the edge.
    """
    wanted = np.argwhere(grid.inside[row0:row1, col0:col1])
    if wanted.size == 0:
        return None

    xs = [grid.x_of(col0 + int(col)) for _, col in wanted]
    ys = [grid.y_of(row0 + int(row)) for row, _ in wanted]

    x, y = min(xs), min(ys)
    extent = max(max(xs) - x, max(ys) - y) + grid.span

    # Squaring can push the block past the far edge; slide it back rather than reading
    # past the canvas, and recompute the offsets against the origin that was used.
    x = max(0, min(x, grid.slide_width - extent))
    y = max(0, min(y, grid.slide_height - extent))
    extent = min(extent, grid.slide_width - x, grid.slide_height - y)

    size = max(grid.size, int(round(extent * grid.base_mpp / grid.mpp)))
    to_window = grid.base_mpp / grid.mpp

    cells: list[tuple[int, int]] = []
    offsets: list[tuple[int, int]] = []
    for (row, col), window_x, window_y in zip(wanted, xs, ys, strict=True):
        left = min(int(round((window_x - x) * to_window)), size - grid.size)
        top = min(int(round((window_y - y) * to_window)), size - grid.size)
        cells.append((row0 + int(row), col0 + int(col)))
        offsets.append((max(0, top), max(0, left)))

    return Block(
        x=int(x),
        y=int(y),
        span=int(extent),
        size=int(size),
        cells=tuple(cells),
        offsets=tuple(offsets),
    )


def _decide(vector: np.ndarray, tau: float | None) -> int:
    """One window's class: argmax, or the invasive threshold the checkpoint carries.

    **Why a threshold and not the argmax.** A three-way argmax lets a badly-conditioned
    in-situ head outvote invasive, and the in-situ head is the badly-conditioned one:
    in-situ carcinoma reaches this project from one or two laboratories only, so the
    class is the least well represented of the three. On `CAN_00251_26_H&E` the v2 model
    called 45.9% of the section in-situ and for 88.5% of those windows the runner-up was
    *non-epithelium* - p(invasive) was last. Thresholding p(invasive) drops those
    windows correctly; no argmax over three classes can.

    So the rule is: invasive when it clears the bar, otherwise the better of the two
    classes that are excluded from the score anyway. It changes only the boundary the
    score is gated on, and leaves the display classes intact.

    `tau=None` restores the argmax, which is what every checkpoint before v3 was
    validated under and must keep being served under.
    """
    if tau is None:
        return int(np.argmax(vector))
    if float(vector[SCORED]) >= tau:
        return SCORED
    # The other two are both excluded from the score, so which of them wins changes
    # nothing downstream - but it is what the class map displays, so it is still the
    # more likely of the two rather than a fixed fallback.
    excluded = [index for index in range(len(vector)) if index != SCORED]
    return int(max(excluded, key=lambda index: vector[index]))


def classify(
    grid: WindowGrid,
    *,
    net: Any,
    read_window: Callable[[int, int, int, int], np.ndarray],
    channel: str = model_input.CHANNEL_HAEMATOXYLIN,
    standardise: bool,
    gamma: float = 1.0,
    invert: bool = False,
    tau: float | None = None,
    block_windows: int = BLOCK_WINDOWS,
    batch_size: int = 32,
    progress: Callable[[int, int], None] | None = None,
    painted: Callable[[tuple[tuple[int, int, int], ...]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> ClassMap:
    """Run the model over every window the grid marked, block by block.

    `read_window(x, y, span, size)` returns the pixels of a square region - level-0
    origin `(x, y)`, level-0 extent `span`, resampled to `size` of its own pixels. The
    caller supplies it so the composition of step 5's read, step 4's white point and
    step 6's `separate` stays in one place, and this module never grows a second answer
    to "what does the model see".

    **`channel` says which of the two contracts the checkpoint declared**, and the two
    are not interchangeable: `haematoxylin` returns HxW float32 optical density and goes
    through the clip, the shape term and the polarity; `rgb_he` returns HxWx3 uint8 sRGB
    and goes through nothing but the ImageNet normalisation. Feeding a colour model
    optical density does not raise - both are arrays of the right rank once replicated -
    it simply scores badly, and the bad score reads as a modelling result. So the value
    comes from the manifest that passed verification, never from a default here.

    **The blocks are visited in a ploughed-field sweep** - see `sweep`. It changes no
    label; it is what makes the progress screen's painting read as one line of work
    crossing the section rather than as a scatter.

    `progress(done, total)` is called after each block, for the job that polls it.
    `should_stop` is checked at the same point and raises `RunCancelled` when it says
    so, which is why a cancel lands within a couple of seconds on a pass that takes
    tens of minutes.

    `painted(cells)` is called after each *batch* with the `(row, col, label)` of every
    window that batch just decided, in the order it decided them. It is the progress
    screen's feed: a step that takes tens of minutes has to show what it is doing, and
    a class per window as it lands is the honest version of that - the same numbers the
    class map is built from, not a mock-up of them.

    It is per batch rather than per block because a block is `block_windows` squared
    forward passes and a batch is `batch_size` of them, so flushing per batch advances
    the painting several times per read instead of once. That leaves the paint feed up
    to one batch ahead of `progress`, which is deliberate: the bar counts finished
    reads, the paint shows finished windows, and neither is worth holding back for the
    other.
    """
    import torch

    labels = np.full((grid.rows, grid.cols), OUTSIDE, dtype=np.int8)
    probabilities = np.zeros((grid.rows, grid.cols, len(CLASS_NAMES)), dtype=np.float32)

    total = grid.windows
    done = 0
    batches = 0
    blocks = 0
    started = time.monotonic()

    for row0, col0 in sweep(grid.rows, grid.cols, block_windows):
        block = plan_block(
            grid,
            row0=row0,
            col0=col0,
            row1=min(grid.rows, row0 + block_windows),
            col1=min(grid.cols, col0 + block_windows),
        )
        if block is None:
            continue

        pixels = read_window(block.x, block.y, block.span, block.size)
        blocks += 1

        if channel == model_input.CHANNEL_RGB_HE:
            tensors = np.stack(
                [
                    model_input.rgb_to_model_input(
                        pixels[top : top + grid.size, left : left + grid.size]
                    )
                    for top, left in block.offsets
                ]
            )
        else:
            tensors = np.stack(
                [
                    model_input.to_model_input(
                        pixels[top : top + grid.size, left : left + grid.size],
                        gamma=gamma,
                        invert=invert,
                        standardise=standardise,
                    )
                    for top, left in block.offsets
                ]
            )

        for start in range(0, len(tensors), batch_size):
            chunk = torch.from_numpy(tensors[start : start + batch_size])
            with torch.inference_mode():
                scores = torch.softmax(net(chunk), dim=1).numpy()
            batches += 1

            decided: list[tuple[int, int, int]] = []
            for offset, vector in enumerate(scores):
                row, col = block.cells[start + offset]
                label = _decide(vector, tau)
                probabilities[row, col] = vector
                labels[row, col] = label
                decided.append((row, col, label))

            if painted is not None:
                painted(tuple(decided))

        done += len(block.cells)
        if progress is not None:
            progress(done, total)
        if should_stop is not None and should_stop():
            # A block boundary is the right granularity: a whole slide is tens of
            # minutes but one block is under two seconds, so a cancel lands
            # promptly without abandoning a batch mid-flight.
            raise RunCancelled(
                f"stopped after {done:,} of {total:,} patches"
            )

    counts = tuple(int((labels == label).sum()) for label in range(len(CLASS_NAMES)))
    cell = grid.cell_mm2

    # One window, one class, one whole cell of area. The boundary error that carries is
    # up to half a window on every edge of every tumour - the honest cost of a model
    # that answers per window, and the reason step 9 smooths before it draws a contour.
    areas = tuple(round(count * cell, 4) for count in counts)

    return ClassMap(
        grid=grid,
        labels=labels,
        probabilities=probabilities,
        counts=counts,  # type: ignore[arg-type]
        areas_mm2=areas,  # type: ignore[arg-type]
        batches=batches,
        blocks_read=blocks,
        seconds=round(time.monotonic() - started, 2),
    )


__all__ = [
    "BLOCK_WINDOWS",
    "OUTSIDE",
    "Block",
    "ClassMap",
    "SegmentationError",
    "WindowGrid",
    "build_grid",
    "classify",
    "plan_block",
    "sweep",
]
