"""Where BEETLE runs for one candidate: a padded box, and the windows inside it.

This module is the whole performance argument of step 11 expressed as geometry. BEETLE
over a whole section is every window of 800 mm2 of slide; BEETLE over the chosen
candidates is every window of the 40 mm2 that step 8 called invasive. Nothing about the
network changes - the same `pixels.segment`, the same weights, the same 0.5 um/px - and
the only thing this module decides is *which windows are marked*.

--------------------------------------------------------------------------------
The padding is an input decision, not a picture decision
--------------------------------------------------------------------------------

`roi_crop_pad_um` pads a *crop* so a person can see tissue around a lesion. This padding
is different in kind: it pads what the network is *shown*, and it exists because a
segmentation network asked to decide a boundary with nothing beyond it will draw the
boundary at the edge of what it was given. A candidate box is a staircase of 224 um
windows around tumour that does not stop at the staircase, so cropping to the box would
truncate exactly the tissue this step exists to trace.

One window's worth is the smallest padding that works, and the reason is arithmetic:
`build_grid` marks a window when its *centre* lands on kept tissue, so a pixel on the
very edge of the candidate is only covered by a window whose centre is up to half a
window outside it. Padding by a full window means every pixel of the candidate is seen
by at least one window that is looking *at* it rather than past it.

--------------------------------------------------------------------------------
Cores, not spans, decide which windows are marked
--------------------------------------------------------------------------------

`pixels.segment` writes each window's stride-wide **core** and discards the rest of its
span, because cores partition the tissue exactly one per window. So the question "does
this window contribute pixels to my box" is a question about its core, and marking on
span overlap instead would mark a ring of windows around the box whose cores all land
outside it - forward passes, at seconds each, whose output is thrown away.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.pipeline.step08_tissue_type_segmentation.inference import WindowGrid
from app.pipeline.step09_roi_mask.crops import Bbox, padded_bbox


class RefinementGeometryError(ValueError):
    """This candidate cannot be segmented - and why, in words a screen can show."""


def padded_box(
    bbox: Bbox, *, pad_um: float, base_mpp: float, slide_width: int, slide_height: int
) -> Bbox:
    """The candidate's box grown by `pad_um` on each side and clamped to the slide.

    In microns rather than pixels so it survives a change of scanner, and clamped
    because a box that ran off the canvas would either be padded with invented tissue or
    refused by the reader - and a candidate touching the edge of the section is a
    perfectly ordinary candidate.
    """
    pad_px = max(0, int(round(pad_um / max(base_mpp, 1e-9))))
    return padded_bbox(bbox, pad_px, slide_width, slide_height)


def core_bounds(grid: WindowGrid, row: int, col: int) -> Bbox:
    """One window's core in level-0 pixels - the pixels it is responsible for.

    Centred in the window's span: `pixels.segment` insets by `(size - stride_px) / 2` in
    the window's own pixels, and this is that same inset expressed in the slide's. The
    two have to agree, because this decides which windows run and that decides which
    pixels get written.
    """
    inset = int(round((grid.span - grid.stride) / 2.0))
    x0 = grid.x_of(col) + inset
    y0 = grid.y_of(row) + inset
    return (x0, y0, x0 + grid.stride, y0 + grid.stride)


def restrict(grid: WindowGrid, bounds: Bbox) -> WindowGrid:
    """`grid` with only the windows whose core reaches into `bounds` still marked.

    A *narrowing* of the existing grid, never a new one, and that is the safety property
    worth stating: `inside` already carries step 7's audited tile list and step 8's own
    tissue gate, so a window this returns is a window the slide-wide pass would also
    have run. Building a fresh grid over the box instead would quietly re-answer "is
    there tissue here", possibly differently, and step 11 would be segmenting windows
    the pipeline had already decided not to look at.
    """
    inside = np.asarray(grid.inside, dtype=bool)
    if not inside.any():
        raise RefinementGeometryError(
            "step 8's grid has no classified windows on this slide, so there is nothing "
            "to refine"
        )

    x0, y0, x1, y1 = bounds
    rows, cols = inside.shape

    # The core's origin is affine in the row and column index right up to the clamp in
    # `x_of`/`y_of`, which only bites in the last row and column. Building the two
    # coordinate vectors and taking an outer test is what keeps this a handful of
    # microseconds on a 400 x 400 grid rather than a Python loop over 160 000 windows.
    inset = int(round((grid.span - grid.stride) / 2.0))
    core_x = np.array([grid.x_of(col) + inset for col in range(cols)])
    core_y = np.array([grid.y_of(row) + inset for row in range(rows)])

    col_hits = (core_x < x1) & (core_x + grid.stride > x0)
    row_hits = (core_y < y1) & (core_y + grid.stride > y0)

    wanted = inside & row_hits[:, None] & col_hits[None, :]
    if not wanted.any():
        raise RefinementGeometryError(
            "no classified window of step 8's grid has its core inside this region's "
            "box. Either the region is smaller than one window's stride, or step 8's "
            "tissue gate turned away every window it is made of."
        )

    # `gated_out` is deliberately not recomputed: it counts windows step 8's tissue gate
    # turned away over the whole slide, and a per-region copy of that number would read
    # as a per-region measurement it is not.
    return replace(grid, inside=wanted)


#: Grid, in microns, a territory is grown on. Coarse for the reason `contours.GROUP_GRID_UM`
#: gives: a 224 um dilation at the mask's own 1 um/px over a 20 mm region is a
#: half-gigapixel distance transform.
TERRITORY_GRID_UM = 8.0


def territory(
    *,
    shape: tuple[int, int],
    origin: tuple[int, int],
    mask_mpp: float,
    base_mpp: float,
    own: list[list[list[float]]],
    earlier: list[list[list[list[float]]]],
    pad_um: float,
) -> np.ndarray:
    """The mask pixels this region may claim (P-10).

    BEETLE runs over the region's whole padded bounding box, and everything it calls
    invasive in that box used to be traced as this region's - so a slide-spanning
    candidate's box swept in tumour nobody selected, and two neighbouring boxes, which
    overlap by up to twice the pad, traced the same pixels into two regions that were
    then sampled and weighted twice.

    A region's territory is its own tile outline (`own`, level-0 rings) grown by the
    pad, minus the territory of every region ranked above it (`earlier`). Ranks decide
    ties, so each pixel has exactly one owner and the result does not depend on which
    region happened to run first.
    """
    from PIL import Image, ImageDraw
    from scipy import ndimage

    factor = max(1, int(round(TERRITORY_GRID_UM / max(mask_mpp, 1e-9))))
    height, width = shape
    rows, cols = -(-height // factor), -(-width // factor)
    to_grid = base_mpp / (mask_mpp * factor)
    radius = max(1, int(np.ceil(pad_um / (mask_mpp * factor))))

    def grown(rings: list[list[list[float]]]) -> np.ndarray:
        canvas = Image.new("1", (cols, rows), 0)
        draw = ImageDraw.Draw(canvas)
        for index, ring in enumerate(rings):
            if len(ring) >= 3:
                draw.polygon(
                    [((x - origin[0]) * to_grid, (y - origin[1]) * to_grid) for x, y in ring],
                    fill=0 if index else 1,
                )
        cells = np.asarray(canvas, dtype=bool)
        if not cells.any():
            return cells
        return ndimage.distance_transform_edt(~cells) <= radius

    mine = grown(own)
    for rings in earlier:
        mine &= ~grown(rings)
    return np.repeat(np.repeat(mine, factor, axis=0), factor, axis=1)[:height, :width]


def window_count(grid: WindowGrid, bounds: Bbox) -> int:
    """How many forward-pass windows this candidate costs, without building anything.

    Used to price the whole selection before any of it runs, so the screen can say what
    it is about to commit to rather than discovering it a region at a time.
    """
    try:
        return int(restrict(grid, bounds).windows)
    except RefinementGeometryError:
        return 0


__all__ = [
    "TERRITORY_GRID_UM",
    "territory",
    "Bbox",
    "RefinementGeometryError",
    "core_bounds",
    "padded_box",
    "restrict",
    "window_count",
]
