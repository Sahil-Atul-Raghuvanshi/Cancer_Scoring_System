"""Step 9's algorithmic core: a grid of class probabilities to one scorable region.

No model, no training, no slide access. The input is step 8's `ClassMap` and the
output is a boolean grid plus the rings that trace it, so this module is testable
from a hand-built array and is the only place the ROI's shape is decided.

The guide's recipe is smooth, threshold, close, drop small components. Two of those
steps are not what a naive reading suggests, and both are the reason this file is
long:

**Smoothing runs on probabilities, not on labels.** `ClassMap` keeps the full softmax
for exactly this. Averaging argmaxes is a vote, and a vote has already thrown away the
margin - a window at 0.34/0.33/0.33 and a window at 0.99 are the same vote and are not
the same claim. Blurring the invasive probability and thresholding the blur keeps the
uncertain windows uncertain until the last moment.

**Closing must not fill every hole, and this is the trap the step exists to avoid.**
The guide's justification for closing is that "a blood vessel inside a tumour should
not punch a hole in the region", which is right. But a *duct of in-situ carcinoma*
inside a tumour absolutely should punch a hole, because Rule 5 keeps in-situ disease
out of the denominator. Morphology cannot tell those two apart - they are both a
not-invasive island inside an invasive sea - so the class map has to, and
`protect_in_situ` is that: after closing, windows the model calls in-situ with
confidence are carved back out.

That is not hypothetical. On `CAN_00270_26_H&E`, 82 % of the in-situ windows inside
the pathologist's DCIS contour sit within one window of invasive tissue, so closing
reaches straight into them: at radius 3 a plain close swallows 20.3 % of that contour's
in-situ, and at radius 6, 35.0 %, while in-situ elsewhere on the slide - isolated
normal lobules, a median 6.3 windows from anything invasive - stays near 3 %. The
merge is therefore *selective for the in-situ that matters*, which is the worst
possible selectivity. Carving it back costs 1.4 mm2 of a 37.6 mm2 region and returns
the swallowed fraction to 1.6 %.

`protect_in_situ=0.0` disables the carve-out and reproduces the guide's plain recipe,
so the two can be compared rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage

from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_NAMES, SCORED
from app.pipeline.step08_tissue_type_segmentation.inference import ClassMap

#: The class Rule 5 excludes and morphology would otherwise annex. Named here and
#: checked against step 8's order at import for the reason `verify_order` exists: a
#: silent permutation of these two indices produces a plausible ROI that scores the
#: wrong tissue, and nothing downstream could notice.
IN_SITU: int = 1

if CLASS_NAMES[IN_SITU] != "non_invasive_epithelium" or CLASS_NAMES[SCORED] != "invasive_epithelium":
    raise AssertionError(
        f"step 9 reads class {IN_SITU} as in-situ and {SCORED} as invasive, but step 8 "
        f"reports {CLASS_NAMES}. The ROI would be built from the wrong class."
    )


class RoiError(ValueError):
    """The ROI could not be built - bad parameters, or a class map with no tissue."""


@dataclass(frozen=True)
class RoiParams:
    """The knobs, with the defaults this project ships.

    Every spatial quantity is in **grid cells** or **mm2**, never in pixels. A cell is
    one window's stride, so a radius expressed here means the same physical distance
    on any scanner, and `min_area_mm2` survives a change of magnification - which is
    the guide's own instruction for the small-component cutoff.
    """

    #: Gaussian blur applied to P(invasive), in cells. One cell is the smallest blur
    #: that couples a window to its neighbours at all; zero disables smoothing.
    sigma: float = 1.0
    #: Where the smoothed probability becomes region. Deliberately not a per-slide
    #: setting in the UI - a threshold moved per slide is a score moved per slide.
    threshold: float = 0.5
    #: Closing radius in cells. Merges nearby foci and fills small holes.
    close_cells: int = 3
    #: Carve back out any window whose in-situ probability reaches this. Zero disables
    #: the carve-out and gives the guide's plain recipe.
    protect_in_situ: float = 0.5
    #: Components below this are not tumour foci, they are speckle. In mm2 so it
    #: survives a change of scanner.
    min_area_mm2: float = 0.25
    #: Keep only the N largest components, matching how a pathologist circles one or
    #: two foci. `None` keeps every component that cleared `min_area_mm2`.
    keep_largest: int | None = None

    def validate(self) -> RoiParams:
        if not 0.0 <= self.threshold <= 1.0:
            raise RoiError(f"threshold must be in [0, 1], got {self.threshold}")
        if not 0.0 <= self.protect_in_situ <= 1.0:
            raise RoiError(f"protect_in_situ must be in [0, 1], got {self.protect_in_situ}")
        if self.sigma < 0.0:
            raise RoiError(f"sigma must not be negative, got {self.sigma}")
        if self.close_cells < 0:
            raise RoiError(f"close_cells must not be negative, got {self.close_cells}")
        if self.min_area_mm2 < 0.0:
            raise RoiError(f"min_area_mm2 must not be negative, got {self.min_area_mm2}")
        if self.keep_largest is not None and self.keep_largest < 1:
            raise RoiError(f"keep_largest must be at least 1, got {self.keep_largest}")
        return self


@dataclass(frozen=True)
class RoiRegion:
    """One focus of the ROI, and the rings that draw it.

    `rings` are in **level-0 slide pixels**, so a viewer can place them on the scan
    without knowing anything about the window grid. The first ring is the outer
    boundary; the rest are holes, and a hole here is a real statement - it is tissue
    the score must not be measured on, sitting inside tissue it must.
    """

    index: int
    cells: int
    area_mm2: float
    #: Outer ring first, then holes. Each is a closed list of (x, y) level-0 vertices.
    rings: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def holes(self) -> int:
        return max(0, len(self.rings) - 1)


@dataclass(frozen=True)
class RoiMask:
    """Step 9's output: where scoring is allowed, and an account of how it got there.

    The audit fields are not decoration. This step removes tissue from the denominator
    four separate times - by thresholding, by closing, by protecting in-situ and by
    dropping small components - and a region that arrives without saying how much each
    of those moved cannot be checked by the person whose score depends on it.
    """

    #: (rows, cols) bool on step 8's grid. The ROI itself.
    mask: np.ndarray
    regions: tuple[RoiRegion, ...]
    params: RoiParams

    cell_mm2: float

    #: Cells over the threshold, before any morphology. The honest starting point.
    seed_cells: int
    #: Components at that point - what "merge nearby foci" is measured against.
    seed_components: int
    #: Cells added by closing - the merge, priced.
    merged_cells: int
    #: In-situ cells the close annexed and the carve-out took back. The Rule 5 number.
    protected_cells: int
    #: Removed by `min_area_mm2` and `keep_largest`.
    dropped_components: int
    dropped_cells: int

    @property
    def cells(self) -> int:
        return int(self.mask.sum())

    @property
    def area_mm2(self) -> float:
        return round(self.cells * self.cell_mm2, 4)

    @property
    def seed_mm2(self) -> float:
        return round(self.seed_cells * self.cell_mm2, 4)

    @property
    def protected_mm2(self) -> float:
        return round(self.protected_cells * self.cell_mm2, 4)

    @property
    def dropped_mm2(self) -> float:
        return round(self.dropped_cells * self.cell_mm2, 4)

    @property
    def holes(self) -> int:
        return sum(region.holes for region in self.regions)


def _disk(radius: int) -> np.ndarray:
    """A round structuring element. Round, not square, so a merge is isotropic.

    A square element closes 1.41x further along the diagonals than along the axes,
    which on a grid this coarse is most of a window - the merge would depend on which
    way the tissue happened to lie.
    """
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    span = np.arange(-radius, radius + 1)
    ys, xs = np.meshgrid(span, span, indexing="ij")
    return (xs * xs + ys * ys) <= radius * radius


def level0_mapper(grid) -> Callable[[int, int], tuple[int, int]]:
    """A grid vertex to a level-0 pixel, clamped to the canvas.

    The inverse of `overlay.cell_field`, and the clamp is the same defence: the
    last row and column of the grid are pushed inside the slide by `x_of`/`y_of`,
    so a vertex derived from them can otherwise land past the edge of the image a
    viewer is about to draw it on.

    Shared by `build()` (the scored region) and `borders.class_regions()` (the
    per-class regions) - one definition of how a grid vertex becomes a slide pixel.
    """
    offset = (grid.span - grid.stride) / 2.0

    def to_level0(row: int, col: int) -> tuple[int, int]:
        x = int(round(offset + col * grid.stride))
        y = int(round(offset + row * grid.stride))
        return (
            int(np.clip(x, 0, grid.slide_width)),
            int(np.clip(y, 0, grid.slide_height)),
        )

    return to_level0


#: How far a traced border is pulled inside its own region, as a fraction of one grid
#: cell. Zero would be the truthful boundary and is wrong for the picture: two regions
#: of different classes that touch share their cell edge exactly, so both borders land
#: on the same line and a reader cannot tell one region's edge from its neighbour's -
#: invasive against in-situ, in-situ against uncertain, any pair. Insetting both by this
#: leaves a gap of twice it between them.
#:
#: 0.06 of a window is 13 um at the 224 um head - under a tenth of a cell, so the region
#: still reads as the cells it is made of, and about two pixels on a 1600 px panel of a
#: 127 000 px slide, which is the smallest gap that survives being looked at.
#:
#: **It makes every polygon slightly smaller than the region it describes.** That is a
#: drawing convention and nothing measures it: areas come from the cell count
#: (`ClassRegion.area_mm2`), the ROI that gates scoring is `roi.npz`, and neither is
#: traced. Do not use these rings to measure anything.
BORDER_GAP_CELLS: float = 0.06


def _normal(step: tuple[int, int], clockwise: bool) -> tuple[int, int]:
    """The unit normal of an axis-aligned step, on one consistent side of travel."""
    row, col = step
    return (col, -row) if clockwise else (-col, row)


def _ring_area(ring) -> float:
    """Twice the signed shoelace area. Sign is the winding, magnitude the size."""
    total = 0.0
    for index, (row, col) in enumerate(ring):
        next_row, next_col = ring[(index + 1) % len(ring)]
        total += col * next_row - next_col * row
    return total / 2.0


def _inset_ring(ring, delta: float, clockwise: bool):
    """One rectilinear ring pulled `delta` cells toward the material it encloses.

    Every edge here is axis-aligned and one cell long, so the offset is exact rather
    than approximate: a vertex moves along the normals of the two edges meeting at it.
    Collinear neighbours share a normal and must not be counted twice - that is the one
    case worth naming, because double-counting it would inset the straight runs by 2x
    and leave the corners behind, which reads as a ragged border rather than a smaller
    one.
    """
    out = []
    count = len(ring)
    for index in range(count):
        previous = ring[(index - 1) % count]
        current = ring[index]
        following = ring[(index + 1) % count]

        incoming = _normal((current[0] - previous[0], current[1] - previous[1]), clockwise)
        outgoing = _normal((following[0] - current[0], following[1] - current[1]), clockwise)

        if incoming == outgoing:
            shift = incoming
        else:
            shift = (incoming[0] + outgoing[0], incoming[1] + outgoing[1])

        out.append((current[0] + delta * shift[0], current[1] + delta * shift[1]))
    return out


def _inset(rings, delta: float):
    """The outer ring pulled toward the material. Holes are left exactly where they are.

    **Only the outer ring, and that is the whole design rather than a shortcut.** A hole
    in region A is the outer boundary of region B sitting inside it - the same cell
    edges - and B's own outer ring is inset by this function too. So the gap between the
    two already exists once, and growing A's hole as well would only double it while
    risking the one thing that must not happen: a hole that grows into, or through, the
    boundary containing it. JTS rejects the result outright ("Reduction failed"), and
    QuPath then imports nothing at all - measured, on a real export, before this was
    narrowed to the outer ring.

    **Which way is "toward the material" is decided by measurement, not assumption.** The
    winding `trace_rings` produces is a property of how it emits cell edges, so the ring
    is inset both ways and the one that made it *smaller* wins.

    A ring that would collapse is returned unchanged: a region one cell across is still a
    real finding, and a degenerate polygon is worse than a coincident border.
    """
    if delta <= 0 or not rings:
        return rings

    outer = rings[0]

    # **A pinched ring is left alone, and this guard is not optional.** Where two cells
    # of one region meet only at a corner - which 8-connectivity says is one region -
    # the walk passes through that vertex twice. Insetting pushes the two passes apart
    # in different directions and the ring crosses itself; JTS refuses the polygon with
    # "Reduction failed", and because QuPath imports the file as a whole, one such
    # region loses the entire export. Measured on CAN_00303: 6 of 52 rings are pinched
    # and exactly those 6 self-intersect once inset.
    #
    # So those regions keep their true, coincident border. That is the right trade: a
    # border touching its neighbour is a cosmetic flaw on a handful of regions, and no
    # .qpdata at all is not.
    if len(set(outer)) != len(outer):
        return rings

    clockwise = abs(_ring_area(_inset_ring(outer, delta, True))) <= abs(_ring_area(outer))
    shrunk = _inset_ring(outer, delta, clockwise)
    if abs(_ring_area(shrunk)) < 1e-9 or len(shrunk) < 3:
        return rings
    return [shrunk, *rings[1:]]


def trace_rings(
    mask: np.ndarray, to_level0, *, gap_cells: float = 0.0
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Trace one component's boundary as closed rings of level-0 vertices.

    **Rectilinear on purpose.** The ROI is a union of whole grid cells, so its true
    boundary runs along cell edges; a spline through it would look more anatomical and
    would be claiming sub-window precision the class map does not have - 112 um here.
    The polygon is honest about that, and a viewer dragging the boundary in the demo's
    edit mode is then editing something real rather than a curve fitted to a staircase.

    Method: emit each cell's four edges wound clockwise; an edge shared by two cells
    appears twice with opposite direction and cancels; whatever survives is the
    boundary, and chaining it gives the outer ring plus one ring per hole. Exact, with
    no tolerance to tune, and holes come out for free - which matters, because a hole
    is how in-situ disease inside a tumour is represented.

    **The branch rule is not a detail.** Where two cells of a component touch only at a
    corner, that vertex carries two outgoing edges, so the walk has a choice and the
    choice decides the answer: turning leftmost keeps the two cells on one ring, which
    is what 8-connectivity means and is what `ndimage.label` was told to use above;
    turning rightmost would split them into two rings and quietly contradict the
    component count this same function is describing.
    """
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, col in zip(*np.nonzero(mask)):
        corners = (
            ((row, col), (row, col + 1)),
            ((row, col + 1), (row + 1, col + 1)),
            ((row + 1, col + 1), (row + 1, col)),
            ((row + 1, col), (row, col)),
        )
        for start, end in corners:
            twin = edges.get(end)
            if twin and start in twin:  # the neighbour drew this edge; it is interior
                twin.remove(start)
                if not twin:
                    del edges[end]
            else:
                edges.setdefault(start, []).append(end)

    def step(node: tuple[int, int], heading: tuple[int, int]) -> tuple[int, int]:
        """The next vertex, preferring the leftmost turn. See the branch rule above."""
        outgoing = edges[node]
        if len(outgoing) == 1:
            return outgoing.pop()
        row, col = heading
        preference = ((-col, row), (row, col), (col, -row), (-row, -col))  # left, on, right, back
        for want in preference:
            candidate = (node[0] + want[0], node[1] + want[1])
            if candidate in outgoing:
                outgoing.remove(candidate)
                return candidate
        return outgoing.pop()

    rings: list[tuple[tuple[int, int], ...]] = []
    while edges:
        # Start where the walk has no choice. Beginning at a corner-touch vertex would
        # consume one of its two outgoing edges and close the ring on the wrong pass,
        # splitting a single boundary in two - the same contradiction the branch rule
        # above exists to prevent.
        start = next(
            (vertex for vertex, out in edges.items() if len(out) == 1),
            next(iter(edges)),
        )
        ring = [start]
        node = step(start, (0, 0))
        if not edges[start]:
            del edges[start]
        while node != start:
            ring.append(node)
            heading = (node[0] - ring[-2][0], node[1] - ring[-2][1])
            nxt = step(node, heading)
            if not edges[node]:
                del edges[node]
            node = nxt
        rings.append(tuple(ring))

    # Outer ring first, by vertex count: a hole is strictly inside the component it
    # sits in, so it cannot carry more boundary than the ring containing it. Sorted
    # before the inset, because `_inset` reads rings[0] to decide which side the
    # material is on.
    rings.sort(key=len, reverse=True)

    # In grid space, where an edge is one cell long and the arithmetic is exact, then
    # mapped to the slide. Insetting after the mapping would have to undo the clamp and
    # the rounding in `to_level0` to know which way is inward.
    return tuple(
        tuple(to_level0(r, c) for r, c in ring) for ring in _inset(rings, gap_cells)
    )


def build(class_map: ClassMap, params: RoiParams | None = None) -> RoiMask:
    """The ROI for one class map. Pure: no I/O, no slide, no global state."""
    params = (params or RoiParams()).validate()

    grid = class_map.grid
    inside = np.asarray(grid.inside, dtype=bool)
    if not inside.any():
        raise RoiError("this class map has no classified windows - step 8 found no tissue")

    probabilities = np.asarray(class_map.probabilities, dtype=np.float32)
    invasive = probabilities[..., SCORED]

    # Smooth, then threshold. The blur runs over the whole grid and is masked after,
    # not before: masking first would let the zeros outside the tissue bleed inward and
    # shave the ROI at every tissue edge, which is where tumour most often is.
    smoothed = ndimage.gaussian_filter(invasive, params.sigma) if params.sigma > 0 else invasive
    seed = (smoothed >= params.threshold) & inside
    seed_cells = int(seed.sum())
    seed_components = int(ndimage.label(seed, structure=np.ones((3, 3)))[1])

    # Close: merge nearby foci and fill the holes that are not a class of their own.
    if params.close_cells > 0:
        closed = ndimage.binary_closing(seed, structure=_disk(params.close_cells)) & inside
    else:
        closed = seed.copy()
    merged_cells = int(closed.sum()) - seed_cells

    # Carve the in-situ back out. After the close, deliberately: the point is to undo
    # what the close annexed, and a subtraction before it would simply be closed over.
    if params.protect_in_situ > 0.0:
        protect = (probabilities[..., IN_SITU] >= params.protect_in_situ) & inside
        kept = closed & ~protect
        protected_cells = int((closed & protect).sum())
    else:
        kept = closed
        protected_cells = 0

    # Drop speckle, then optionally keep the largest few foci.
    labelled, count = ndimage.label(kept, structure=np.ones((3, 3)))
    sizes = np.bincount(labelled.ravel())[1:] if count else np.zeros(0, dtype=np.int64)
    order = np.argsort(-sizes)

    survivors: list[int] = []
    for position, component in enumerate(order):
        if sizes[component] * grid.cell_mm2 < params.min_area_mm2:
            break  # sorted by size, so everything after this is smaller too
        if params.keep_largest is not None and position >= params.keep_largest:
            break
        survivors.append(int(component) + 1)

    final = np.isin(labelled, survivors) if survivors else np.zeros_like(kept)
    dropped_cells = int(kept.sum()) - int(final.sum())
    dropped_components = int(count) - len(survivors)

    to_level0 = level0_mapper(grid)

    regions: list[RoiRegion] = []
    for index, component in enumerate(survivors):
        piece = labelled == component
        cells = int(piece.sum())
        regions.append(
            RoiRegion(
                index=index,
                cells=cells,
                area_mm2=round(cells * grid.cell_mm2, 4),
                rings=trace_rings(piece, to_level0),
            )
        )

    return RoiMask(
        mask=final,
        regions=tuple(regions),
        params=params,
        cell_mm2=grid.cell_mm2,
        seed_cells=seed_cells,
        seed_components=seed_components,
        merged_cells=merged_cells,
        protected_cells=protected_cells,
        dropped_components=dropped_components,
        dropped_cells=dropped_cells,
    )


__all__ = [
    "BORDER_GAP_CELLS",
    "IN_SITU",
    "RoiError",
    "RoiMask",
    "RoiParams",
    "RoiRegion",
    "build",
    "level0_mapper",
    "trace_rings",
]
