"""The BEETLE pass: one window at a time, one class per pixel, painted as it goes.

This is to the BEETLE branch what `inference.classify` is to the two ResNet branches,
and the difference between them is the whole point of the branch: `classify` runs a
fitted head and writes **one label per window**; this runs a published segmentation
network and writes **one label per pixel**, then reduces the window's pixels to a
probability vector so the report has something to count.

--------------------------------------------------------------------------------
What is shared with the ResNet branches, and what is not
--------------------------------------------------------------------------------

Shared: **geometry, and only geometry.** `WindowGrid`, `build_grid`, `sweep` and
`plan_block` come from `inference` because they answer "where on this slide should
anything look", and that is step 7's answer - the audited tile list, the quality-control
gate and the tissue gate - not a model's. Re-deriving it here would be a second join to
step 7 that could disagree with the first, which is the failure `build_grid`'s docstring
is about. None of it mentions a checkpoint, a class order or an input contract.

Not shared: the network, the class set, the input contract, the checkpoint manifest, the
decision rule and the output type. There is no `model.Pinned` here, no `classes.py`, no
`input.py`, and the map that comes out has BEETLE's five classes in it rather than our
three. `beetle.py`'s docstring says why that separation is enforced rather than merely
observed.

--------------------------------------------------------------------------------
Blocks are what get read; windows are what get segmented
--------------------------------------------------------------------------------

A `read_region` off the pyramid costs far more than a window of pixels, so windows are
read in square blocks of `block_windows` squared of them: one read, one resample, up to
64 windows cut out of the result. Blocks holding no kept window are never read, which is
what keeps the pass proportional to the tissue rather than to the canvas.

**The block is a read amortisation and nothing else - the network still sees one window
at a time.** That is deliberate and it is what makes the four fields of view a real
comparison: running BEETLE over the whole block instead would hand every window the same
generous context and the 112 um option would stop differing from the 672 um one except
at the block's rim. The context a window gets *is* the variable under test here, so it
has to be exactly the window.

--------------------------------------------------------------------------------
Two resolutions, and why the mask is kept at neither of the obvious ones
--------------------------------------------------------------------------------

BEETLE answers at 0.5 um/px. A 28 x 28 mm section at that spacing is 3.2 gigapixels,
which is not an array this step can hold, a file it can write, or a picture a browser can
be handed. So the per-window answer is reduced onto a slide-level canvas at
`mask_mpp` - 4 um/px by default, a 64x area reduction - and that canvas is what is
stored, drawn and handed to step 9.

**The reduction area-averages probabilities and then takes the argmax; it does not
resample labels.** Downsampling a label map by 8x with nearest-neighbour throws away
63 of every 64 pixels and picks the survivor arbitrarily, so a thin duct wall either
vanishes or thickens depending on where the grid falls. Averaging the five probability
planes over each output pixel's footprint and arguing the maximum afterwards is the same
principle the rest of the pipeline already states - step 10 stitches probabilities
because averaging argmaxes is a vote, and a vote discards the margin.

--------------------------------------------------------------------------------
Each window writes its core, not its span
--------------------------------------------------------------------------------

Windows overlap when step 7's overlap is not zero; their `stride`-wide cores do not.
The cores partition the tissue exactly one per window, so writing cores means every
mask pixel is written once, by the window that saw it most centrally, and no area
computed from the mask moves when the overlap does. Writing spans instead would
overprint every neighbour and, at 50% overlap, decide the map by visiting order.
`WindowGrid.cell_mm2` draws the same distinction for the same reason.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from app.pipeline.contract import RunCancelled

from . import beetle
from .inference import BLOCK_WINDOWS, WindowGrid, plan_block, sweep

#: Mask value where no window ran. Not a class - the map is defined only over the tissue
#: step 7 kept, and a sixth "background" value would be confused with BEETLE's own
#: `unannotated`, which is a *learned* claim about a pixel and an entirely different
#: statement from "nothing looked here".
OUTSIDE: int = 255

#: The per-window label used in the window-level summary where no window ran. `-1`
#: rather than 255 because that array is `int8`, and it matches `inference.OUTSIDE` so a
#: reader moving between the two branches meets one sentinel.
OUTSIDE_WINDOW: int = -1


class PixelError(ValueError):
    """This slide cannot be segmented per pixel - and why, in words."""


@dataclass(frozen=True)
class PixelMap:
    """Step 8's output on the BEETLE branch: a pixel mask plus a per-window summary.

    `mask` is the product. Everything else is either the audit trail that makes it
    readable or the per-window reduction the report counts, and both are derived from
    the mask rather than measured separately - so a number on the screen and the picture
    beside it cannot disagree.
    """

    grid: WindowGrid

    #: (H, W) uint8 of BEETLE's five codes at `mask_mpp`, `OUTSIDE` where nothing ran.
    mask: np.ndarray
    mask_mpp: float

    #: (rows, cols, 5) float32 - each window's mean probability over its own core.
    #: Kept for the same reason `ClassMap.probabilities` is: a smoothing step needs the
    #: margin, and an argmax has thrown it away.
    windows: np.ndarray
    #: (rows, cols) int8 argmax of the above, `OUTSIDE_WINDOW` where no window ran.
    labels: np.ndarray

    #: Mask pixels per class, indexed by BEETLE code.
    counts: tuple[int, ...]
    #: Slide area per class in mm2, from the mask's own resolution.
    areas_mm2: tuple[float, ...]

    #: Mean winning probability over the mask pixels each class actually won, indexed by
    #: BEETLE code; 0.0 for a class that won none.
    #:
    #: **Measured per pixel, over exactly the pixels in `mask`.** Deriving it from the
    #: per-window vectors instead was tried and is misleading: at 672 um every window of
    #: a test region was dominated by `other`, so every other class had won no *window*
    #: and had to fall back to its mean asserted probability - which lands near its
    #: pixel share and reads on screen as a duplicate of it. A class can hold 12% of the
    #: tissue without ever being a window's plurality, and this is the number that says
    #: how sure the model was where it drew it.
    class_confidence: tuple[float, ...]

    windows_done: int
    patches: int
    blocks_read: int
    seconds: float

    #: Level-0 pixel the mask's (0, 0) sits on. `(0, 0)` for a slide-wide pass, which is
    #: what every caller before step 11 asks for; step 11 asks for one region's box
    #: instead, and then this is how a mask pixel becomes a slide position.
    #:
    #: **Carried on the map rather than remembered by the caller** because the mask and
    #: its origin are one fact: a mask separated from its origin is a picture of a
    #: tumour that has lost where the tumour is, which is precisely the mistake the
    #: region-wise pass exists not to make.
    mask_origin: tuple[int, int] = (0, 0)

    @property
    def tissue_pixels(self) -> int:
        """Mask pixels BEETLE called tissue - everything but `unannotated`.

        The denominator every share below is taken over. Glass is excluded because a
        slide with more empty space around its section would otherwise report a smaller
        tumour content for that reason alone.
        """
        return int(sum(self.counts[code] for code in beetle.TISSUE_CODES))

    @property
    def shares(self) -> tuple[float, ...]:
        """Share of the tissue each class claimed, indexed by BEETLE code.

        `unannotated`'s own entry is its share of the tissue denominator, which is zero
        by construction; its extent is in `counts` and `areas_mm2` where it belongs.
        """
        total = max(1, self.tissue_pixels)
        return tuple(
            (self.counts[code] / total if code in beetle.TISSUE_CODES else 0.0)
            for code in range(len(self.counts))
        )

    @property
    def tumour_content(self) -> float:
        """Share of the tissue that is invasive carcinoma.

        The number the guide asks this step to be stratified by, and the point of the
        whole step: not "how much tumour" but how much *invasive* tumour, with in-situ
        disease, stroma, fat and necrosis already out of the denominator. Measured over
        pixels rather than windows, which is what a per-pixel model buys - a window
        holding one small focus of invasion counts as that focus and not as a window.
        """
        return self.shares[beetle.SCORED_CODE]

    @property
    def scored_mm2(self) -> float:
        return self.areas_mm2[beetle.SCORED_CODE]

    @property
    def mean_confidence(self) -> float:
        """Mean winning probability over every classified mask pixel.

        Reported because an argmax hides its own margin: a pixel at
        0.21/0.20/0.20/0.20/0.19 and one at 0.99 are the same colour on the map and are
        not the same claim.

        Weighted by pixels rather than by windows, so it describes the picture the
        report is about. A window average would let a window holding four pixels of
        tissue count as much as one holding a duct.
        """
        total = sum(
            count for code, count in enumerate(self.counts) if self.class_confidence[code] > 0
        )
        if total == 0:
            return 0.0
        weighted = sum(
            self.class_confidence[code] * self.counts[code]
            for code in range(len(self.counts))
        )
        return float(weighted / total)

    def confidence(self) -> np.ndarray:
        """(rows, cols) float32 top-class probability, zero outside. For the panel."""
        top = self.windows.max(axis=-1).astype(np.float32)
        return np.where(self.grid.inside, top, 0.0).astype(np.float32)


def mask_shape(
    grid: WindowGrid, mask_mpp: float, bounds: tuple[int, int, int, int] | None = None
) -> tuple[int, int]:
    """The mask's shape at `mask_mpp`, over the whole slide or over one box of it.

    `bounds` is a half-open level-0 box `(x0, y0, x1, y1)`. Without it the mask covers
    the slide, which is what the slide-wide pass wants and what every caller before
    step 11 asks for.

    **With it, the allocation stops being quadratic in the slide.** A 28 x 28 mm section
    at 1 um/px is 784 megapixels - not a canvas this step can hold once, let alone once
    per region - and that is the whole reason the slide-wide pass reduces to 4 um/px. A
    region is not a section: 5 mm2 at 1 um/px is five megapixels, so a pass restricted
    to one region can afford four times the linear precision the slide-wide pass could,
    and `bounds` is what lets it.
    """
    if mask_mpp <= 0:
        raise PixelError("a mask resolution of zero microns per pixel is not a resolution")
    scale = grid.base_mpp / float(mask_mpp)
    if bounds is None:
        return (
            max(1, int(round(grid.slide_height * scale))),
            max(1, int(round(grid.slide_width * scale))),
        )
    x0, y0, x1, y1 = bounds
    return (
        max(1, int(round((y1 - y0) * scale))),
        max(1, int(round((x1 - x0) * scale))),
    )


def reduce_probabilities(probs: np.ndarray, height: int, width: int) -> np.ndarray:
    """Area-average `(C, H, W)` probabilities onto a `(C, height, width)` grid.

    Through PIL's `BOX` filter, which is a true area average, per plane. The alternative
    - slicing out every nth pixel - is nearest-neighbour resampling of a label map by
    the time the argmax is taken, and it makes a duct wall thinner or thicker depending
    on where the sampling grid happens to fall.

    Returns the planes unchanged when no reduction is asked for, so a window already at
    the target size costs nothing.
    """
    planes = np.asarray(probs, dtype=np.float32)
    if planes.shape[1:] == (height, width):
        return planes
    return np.stack(
        [
            np.asarray(
                Image.fromarray(plane, mode="F").resize(
                    (width, height), Image.Resampling.BOX
                ),
                dtype=np.float32,
            )
            for plane in planes
        ]
    )


def window_order(cells: tuple[tuple[int, int], ...], *, reverse: bool) -> list[int]:
    """Indices into `cells` in the order a viewer should watch them fill in.

    A ploughed field at window granularity: down the block's rows, and along each row in
    alternating directions, starting in whichever direction the band of blocks is
    already travelling. `sweep` does this for blocks; without doing it for the windows
    inside them too, the painted edge would cross the block left to right sixty-four
    times while the blocks themselves went right to left, and the sweep would read as a
    scatter inside a march.

    **It changes no label.** Every window is an independent set of forward passes, so
    what this decides is only *when* each part of the section is answered - and
    therefore what someone watching the progress screen sees.
    """
    rows: dict[int, list[int]] = {}
    for index, (row, _) in enumerate(cells):
        rows.setdefault(row, []).append(index)

    order: list[int] = []
    for band, row in enumerate(sorted(rows)):
        indices = sorted(rows[row], key=lambda index: cells[index][1])
        # `reverse` is the band's own direction, flipped every row inside the block so
        # consecutive rows join end to end rather than jumping back across it.
        if bool(band % 2) != reverse:
            indices.reverse()
        order.extend(indices)
    return order


def encode_paint(labels: np.ndarray, side: int) -> str:
    """One window's mask, reduced to `side` squared bytes, base64 for the poll reply.

    Raw class ids rather than a PNG: the client draws them into an `ImageData` directly,
    so a PNG would be an encode on the server and a decode in the browser to move a
    kilobyte. `side` is what the progress canvas can actually show - it draws the whole
    slide into 1200 px, so a window lands on a few dozen of them.
    """
    small = np.asarray(
        Image.fromarray(labels.astype(np.uint8), mode="L").resize(
            (side, side), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )
    return base64.b64encode(small.tobytes()).decode("ascii")


def segment(
    grid: WindowGrid,
    loaded: beetle.Loaded,
    *,
    read_window: Callable[[int, int, int, int], np.ndarray],
    mask_mpp: float,
    patch_step: float | None = None,
    batch_size: int = 4,
    paint_px: int = 32,
    block_windows: int = BLOCK_WINDOWS,
    mask_bounds: tuple[int, int, int, int] | None = None,
    progress: Callable[[int, int], None] | None = None,
    painted: Callable[[tuple[tuple[int, int, int, str], ...]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> PixelMap:
    """Run BEETLE over every window the grid marked, and build the slide's pixel mask.

    `read_window(x, y, span, size)` returns the pixels of a square region - level-0
    origin `(x, y)`, level-0 extent `span`, resampled to `size` of its own pixels. The
    caller supplies it so the composition that turns a slide position into the network's
    input stays in one place. For this branch that composition is short: step 5's tile
    read and nothing else, because BEETLE's input contract is division by 255 - no white
    point, no deconvolution. `grid.mpp` is BEETLE's 0.5, so the block comes back at the
    network's own spacing and the windows cut out of it need no further resampling.

    `progress(done, total)` is called after each block, and `should_stop` is checked at
    the same point - which is why a cancel lands within a window or two of being asked
    for on a pass that takes hours.

    `painted(cells)` is called after **each window** with `(row, col, label, mask)`,
    where `label` is the window's dominant class and `mask` is `encode_paint`'s base64
    of its reduced pixel mask. Per window rather than per batch, because on this branch
    a window is seconds of work rather than milliseconds and it is the unit the viewer
    asked to watch: the tile branches paint a flat colour per window, this paints the
    shapes BEETLE found inside it.

    `mask_bounds` restricts the *canvas*, not the run. Which windows run is the grid's
    `inside` and nothing else; this only says how much of the slide the output array
    covers, so a caller segmenting one region allocates that region rather than the
    section around it. A window whose core falls outside the bounds still runs and still
    lands in the per-window summary; its pixels simply have nowhere to be written, which
    is correct - they are not part of the mask that was asked for.
    """
    classes = len(beetle.PIXEL_CLASSES)
    height, width = mask_shape(grid, mask_mpp, mask_bounds)
    origin_x, origin_y = (0, 0) if mask_bounds is None else (mask_bounds[0], mask_bounds[1])

    mask = np.full((height, width), OUTSIDE, dtype=np.uint8)
    windows = np.zeros((grid.rows, grid.cols, classes), dtype=np.float32)
    labels = np.full((grid.rows, grid.cols), OUTSIDE_WINDOW, dtype=np.int8)

    # Per-class confidence, accumulated over the *reduced* pixels - the ones that end up
    # in `mask` - so the figure describes the picture and not an intermediate. float64
    # because this is a running sum over hundreds of millions of pixels.
    confidence_sum = np.zeros(classes, dtype=np.float64)
    confidence_count = np.zeros(classes, dtype=np.int64)

    total = grid.windows
    done = 0
    patches = 0
    blocks = 0
    started = time.monotonic()

    to_mask = grid.base_mpp / float(mask_mpp)
    # The window's core, in the window's own pixels: `stride_px` wide, centred in the
    # `size`-wide span. See the module docstring for why cores and not spans.
    core_inset_px = (grid.size - grid.stride_px) // 2
    core_px = grid.stride_px
    per_window_patches = beetle.patches_per_window(
        grid.field_um, loaded.patch, patch_step
    ) * len(loaded.nets)

    origins = sweep(grid.rows, grid.cols, block_windows)
    for row0, col0 in origins:
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

        # Which way this band of blocks is travelling, so the windows inside the block
        # continue the plough rather than cutting back across it. `sweep` reverses the
        # columns of alternate bands; this recovers that from the band index.
        band = row0 // max(1, block_windows)
        reverse = bool(band % 2)

        decided: list[tuple[int, int, int, str]] = []
        for index in window_order(block.cells, reverse=reverse):
            row, col = block.cells[index]
            top, left = block.offsets[index]
            window = pixels[top : top + grid.size, left : left + grid.size]

            probs = beetle.predict_window(
                window, loaded, step=patch_step, batch_size=batch_size
            )
            patches += per_window_patches

            # The core, in the window's own pixels and in the mask's.
            core = probs[
                :,
                core_inset_px : core_inset_px + core_px,
                core_inset_px : core_inset_px + core_px,
            ]

            x0 = grid.x_of(col) + int(round(core_inset_px * grid.mpp / grid.base_mpp))
            y0 = grid.y_of(row) + int(round(core_inset_px * grid.mpp / grid.base_mpp))
            core_span = int(round(core_px * grid.mpp / grid.base_mpp))

            # A core that lies wholly outside a restricted canvas would be clamped
            # by the arithmetic below onto a one-pixel sliver of the canvas edge,
            # smearing one window's answer along the border. The window still ran and
            # still counts in the per-window summary; it simply has no pixels here.
            canvas_x1 = origin_x + width / to_mask
            canvas_y1 = origin_y + height / to_mask
            on_canvas = (
                x0 < canvas_x1
                and x0 + core_span > origin_x
                and y0 < canvas_y1
                and y0 + core_span > origin_y
            )

            if on_canvas:
                mx0 = max(0, min(width - 1, int(round((x0 - origin_x) * to_mask))))
                my0 = max(0, min(height - 1, int(round((y0 - origin_y) * to_mask))))
                mx1 = max(
                    mx0 + 1, min(width, int(round((x0 + core_span - origin_x) * to_mask)))
                )
                my1 = max(
                    my0 + 1, min(height, int(round((y0 + core_span - origin_y) * to_mask)))
                )

                reduced = reduce_probabilities(core, my1 - my0, mx1 - mx0)
                reduced_labels = reduced.argmax(axis=0)
                mask[my0:my1, mx0:mx1] = reduced_labels.astype(np.uint8)

                winning = reduced.max(axis=0)
                for code in range(classes):
                    here = reduced_labels == code
                    hits = int(here.sum())
                    if hits:
                        confidence_sum[code] += float(winning[here].sum())
                        confidence_count[code] += hits

            # The window's own summary, over the same core the mask got - so the report
            # counts what the picture shows.
            vector = core.mean(axis=(1, 2))
            windows[row, col] = vector
            label = int(np.argmax(vector))
            labels[row, col] = label

            done += 1
            if painted is not None:
                decided.append(
                    (row, col, label, encode_paint(core.argmax(axis=0), paint_px))
                )
                # Flushed per window rather than per block: a block is up to 64 windows
                # and minutes of work on this branch, and a screen that only advanced
                # once a block would look stalled for all of it.
                painted(tuple(decided))
                decided.clear()

            if should_stop is not None and should_stop():
                raise RunCancelled(
                    f"stopped after {done:,} of {total:,} windows"
                )

        if progress is not None:
            progress(done, total)

    counts = tuple(int((mask == code).sum()) for code in range(classes))
    pixel_mm2 = (float(mask_mpp) / 1000.0) ** 2
    areas = tuple(round(count * pixel_mm2, 4) for count in counts)
    confidence = tuple(
        round(float(confidence_sum[code] / confidence_count[code]), 4)
        if confidence_count[code]
        else 0.0
        for code in range(classes)
    )

    return PixelMap(
        grid=grid,
        mask=mask,
        mask_mpp=float(mask_mpp),
        mask_origin=(int(origin_x), int(origin_y)),
        windows=windows,
        labels=labels,
        counts=counts,
        areas_mm2=areas,
        class_confidence=confidence,
        windows_done=done,
        patches=patches,
        blocks_read=blocks,
        seconds=round(time.monotonic() - started, 2),
    )


__all__ = [
    "OUTSIDE",
    "OUTSIDE_WINDOW",
    "PixelError",
    "PixelMap",
    "encode_paint",
    "mask_shape",
    "reduce_probabilities",
    "segment",
    "window_order",
]
