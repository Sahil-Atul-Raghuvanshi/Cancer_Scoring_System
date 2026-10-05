"""Step 9 tests.

`mask.build` is pure - a `ClassMap` in, a region out - so almost everything here is
tested from a hand-built probability grid with no slide, no model and no disk. That is
the point of keeping the core free of I/O.

The assertions are about *properties* rather than values, because the numbers depend on
the slide: closing only ever adds, the carve-out only ever removes, the ledger accounts
for every cell, the polygon encloses exactly the mask it came from. The two places that
do assert a shape are the ones where a plausible-looking wrong answer would survive
every property check - the corner-touch trace, and the Rule 5 carve-out itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_NAMES, SCORED
from app.pipeline.step08_tissue_type_segmentation.inference import ClassMap, WindowGrid
from app.pipeline.step09_roi_mask.mask import IN_SITU, RoiError, RoiParams, build

CELL = 16  # level-0 pixels per cell in the fixtures; small, and not a power of the span


def make_class_map(probabilities: np.ndarray, inside: np.ndarray | None = None) -> ClassMap:
    """A `ClassMap` over a hand-built probability grid.

    `span == stride` here, so a cell's level-0 origin is `col * CELL` and a vertex
    conversion that silently dropped the half-window offset would still pass - which is
    why `test_polygon_vertices_land_on_cell_boundaries` uses a grid where they differ.
    """
    rows, cols, classes = probabilities.shape
    assert classes == len(CLASS_NAMES)
    if inside is None:
        inside = np.ones((rows, cols), dtype=bool)

    grid = WindowGrid(
        cols=cols,
        rows=rows,
        size=CELL,
        stride_px=CELL,
        span=CELL,
        stride=CELL,
        overlap=0.0,
        mpp=1.0,
        base_mpp=1.0,
        slide_width=cols * CELL,
        slide_height=rows * CELL,
        inside=inside,
    )
    labels = np.where(inside, probabilities.argmax(axis=-1), -1).astype(np.int8)
    counts = tuple(int((labels == label).sum()) for label in range(len(CLASS_NAMES)))
    return ClassMap(
        grid=grid,
        labels=labels,
        probabilities=probabilities.astype(np.float32),
        counts=counts,  # type: ignore[arg-type]
        areas_mm2=tuple(count * grid.cell_mm2 for count in counts),  # type: ignore[arg-type]
        batches=1,
        blocks_read=1,
        seconds=0.0,
    )


def field(rows: int, cols: int) -> np.ndarray:
    """An all-non-epithelium probability grid to paint into."""
    out = np.zeros((rows, cols, len(CLASS_NAMES)), dtype=np.float32)
    out[..., 0] = 1.0
    return out


def paint(grid: np.ndarray, rows, cols, label: int, value: float = 0.95) -> None:
    grid[rows, cols, :] = (1.0 - value) / (len(CLASS_NAMES) - 1)
    grid[rows, cols, label] = value


# --- the recipe ---------------------------------------------------------------


def test_a_solid_block_of_invasive_becomes_one_region():
    grid = field(20, 20)
    paint(grid, slice(4, 12), slice(4, 12), SCORED)
    roi = build(make_class_map(grid), RoiParams(sigma=0.0, close_cells=0, min_area_mm2=0.0))

    assert len(roi.regions) == 1
    assert roi.cells == 64
    assert roi.holes == 0


def test_closing_only_ever_adds_and_the_ledger_says_how_much():
    """Two blocks a few cells apart. Closing merges them; the ledger prices the merge."""
    grid = field(20, 20)
    paint(grid, slice(4, 8), slice(3, 7), SCORED)
    paint(grid, slice(4, 8), slice(10, 14), SCORED)
    class_map = make_class_map(grid)

    apart = build(class_map, RoiParams(sigma=0.0, close_cells=0, min_area_mm2=0.0))
    together = build(class_map, RoiParams(sigma=0.0, close_cells=3, min_area_mm2=0.0))

    assert apart.seed_components == 2
    assert len(apart.regions) == 2
    assert len(together.regions) == 1
    assert together.cells > apart.cells
    assert together.merged_cells == together.cells - together.seed_cells > 0
    # Closing is not allowed to remove: every seeded cell survives it.
    assert apart.mask[together.mask].all() or together.mask[apart.mask].all()


def test_in_situ_between_two_foci_is_carved_back_out():
    """The Rule 5 case, and the reason this step is not the guide's recipe verbatim.

    A column of in-situ sits in the gap the closing bridges. With the carve-out off it
    is inside the scored region; with it on, it is not, and the two foci the merge had
    joined are two foci again.
    """
    grid = field(20, 20)
    paint(grid, slice(4, 12), slice(3, 8), SCORED)
    paint(grid, slice(4, 12), slice(11, 16), SCORED)
    paint(grid, slice(4, 12), slice(8, 11), IN_SITU)
    class_map = make_class_map(grid)

    plain = build(class_map, RoiParams(sigma=0.0, close_cells=4, protect_in_situ=0.0, min_area_mm2=0.0))
    guarded = build(class_map, RoiParams(sigma=0.0, close_cells=4, protect_in_situ=0.5, min_area_mm2=0.0))

    strip = np.zeros_like(plain.mask)
    strip[4:12, 8:11] = True

    # The close bridges the interior of the gap but cannot reach around the ends of
    # the strip, which is correct morphology - so the claim is about what it *did*
    # annex, not about the whole strip.
    swallowed = int((plain.mask & strip).sum())
    assert swallowed > 0, "the plain recipe swallows in-situ that lies between two foci"
    assert plain.mask[6:10, 8:11].all(), "and swallows it solidly away from the ends"

    assert not (guarded.mask & strip).any(), "the carve-out must remove all of it"
    assert guarded.protected_cells == swallowed, "it removes exactly what the close annexed"
    assert guarded.cells == plain.cells - swallowed

    # And the merge is undone: what the close joined was in-situ, so with Rule 5
    # applied these are two foci again rather than one. `test_a_hole_becomes_its_own_
    # ring` covers the case where the in-situ is enclosed and a hole is left instead.
    assert len(plain.regions) == 1
    assert len(guarded.regions) == 2


def test_the_carve_out_leaves_confident_invasive_alone():
    """It removes in-situ, never invasive - a carve-out that ate tumour would shrink
    the denominator silently, which is the opposite failure and just as bad."""
    grid = field(16, 16)
    paint(grid, slice(3, 13), slice(3, 13), SCORED, value=0.99)
    roi = build(make_class_map(grid), RoiParams(sigma=0.0, close_cells=2, min_area_mm2=0.0))

    assert roi.protected_cells == 0
    assert roi.cells == 100


def test_small_components_are_dropped_by_physical_area_not_by_cells():
    grid = field(24, 24)
    paint(grid, slice(4, 12), slice(4, 12), SCORED)  # 64 cells
    paint(grid, slice(20, 21), slice(20, 21), SCORED)  # 1 cell of speckle
    class_map = make_class_map(grid)
    cell_mm2 = class_map.grid.cell_mm2

    kept = build(class_map, RoiParams(sigma=0.0, close_cells=0, min_area_mm2=0.0))
    pruned = build(class_map, RoiParams(sigma=0.0, close_cells=0, min_area_mm2=cell_mm2 * 2))

    assert len(kept.regions) == 2
    assert len(pruned.regions) == 1
    assert pruned.dropped_components == 1
    assert pruned.dropped_cells == 1


def test_keep_largest_keeps_the_largest():
    grid = field(30, 30)
    paint(grid, slice(2, 10), slice(2, 10), SCORED)   # 64
    paint(grid, slice(2, 8), slice(14, 20), SCORED)   # 36
    paint(grid, slice(20, 24), slice(20, 24), SCORED)  # 16
    roi = build(
        make_class_map(grid),
        RoiParams(sigma=0.0, close_cells=0, min_area_mm2=0.0, keep_largest=2),
    )

    assert [region.cells for region in roi.regions] == [64, 36]
    assert roi.dropped_components == 1


def test_smoothing_runs_on_probabilities_so_a_lone_uncertain_window_does_not_survive():
    """An isolated window that just clears the threshold is blurred away by its
    neighbours; a solid block is not. That is what smoothing probabilities buys."""
    grid = field(20, 20)
    paint(grid, slice(4, 10), slice(4, 10), SCORED, value=0.9)
    paint(grid, 15, 15, SCORED, value=0.55)

    sharp = build(make_class_map(grid), RoiParams(sigma=0.0, close_cells=0, min_area_mm2=0.0))
    blurred = build(make_class_map(grid), RoiParams(sigma=1.0, close_cells=0, min_area_mm2=0.0))

    assert sharp.mask[15, 15]
    assert not blurred.mask[15, 15]
    assert blurred.mask[6, 6], "the solid block survives the same blur"


# --- geometry -----------------------------------------------------------------


def test_the_polygon_encloses_exactly_the_mask():
    """Ray-cast every cell centre against every ring. The rings are the API's
    version of the mask, and a polygon that disagreed with it would be a region
    the viewer edits and the score does not use."""
    rng = np.random.default_rng(11)
    grid = field(30, 30)
    blob = rng.random((30, 30)) > 0.55
    paint(grid, *np.where(blob), SCORED)
    roi = build(make_class_map(grid), RoiParams(sigma=1.0, close_cells=2, min_area_mm2=0.0))

    rows, cols = roi.mask.shape
    ys, xs = np.mgrid[0:rows, 0:cols]
    px, py = xs + 0.5, ys + 0.5
    enclosed = np.zeros((rows, cols), dtype=bool)
    for region in roi.regions:
        for ring in region.rings:
            vertices = np.array(ring, dtype=float) / CELL
            x1, y1 = vertices[:, 0], vertices[:, 1]
            x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
            for ax, ay, bx, by in zip(x1, y1, x2, y2):
                if ay == by:
                    continue
                enclosed ^= ((ay > py) != (by > py)) & (px < ax + (py - ay) * (bx - ax) / (by - ay))

    assert np.array_equal(enclosed, roi.mask)


def test_two_cells_touching_only_at_a_corner_trace_as_one_ring():
    """8-connectivity says they are one component; the trace must agree.

    A rightmost-turn walk would split them into two rings and quietly contradict the
    component count reported alongside. This is the case that broke the first version.
    """
    grid = field(8, 8)
    paint(grid, [2, 3], [2, 3], SCORED)
    roi = build(make_class_map(grid), RoiParams(sigma=0.0, close_cells=0, min_area_mm2=0.0))

    assert len(roi.regions) == 1
    assert roi.regions[0].cells == 2
    assert len(roi.regions[0].rings) == 1, "one component, one ring"
    assert roi.holes == 0


def test_a_hole_becomes_its_own_ring():
    grid = field(20, 20)
    paint(grid, slice(4, 14), slice(4, 14), SCORED)
    paint(grid, slice(8, 10), slice(8, 10), IN_SITU)
    roi = build(make_class_map(grid), RoiParams(sigma=0.0, close_cells=0, min_area_mm2=0.0))

    assert len(roi.regions) == 1
    assert roi.regions[0].holes == 1
    assert len(roi.regions[0].rings) == 2
    assert len(roi.regions[0].rings[0]) > len(roi.regions[0].rings[1]), "outer ring first"


def test_polygon_vertices_land_on_cell_boundaries_when_windows_overlap():
    """With overlap the cell core is offset by half the difference of span and stride.
    A conversion that used the span, or dropped the offset, lands half a window out."""
    grid = field(10, 10)
    paint(grid, slice(3, 7), slice(3, 7), SCORED)
    class_map = make_class_map(grid)
    wide = WindowGrid(
        **{
            **class_map.grid.__dict__,
            "span": CELL * 2,
            "size": CELL * 2,
        }
    )
    overlapped = ClassMap(**{**class_map.__dict__, "grid": wide})

    roi = build(overlapped, RoiParams(sigma=0.0, close_cells=0, min_area_mm2=0.0))
    offset = (wide.span - wide.stride) / 2.0
    xs = {vertex[0] for vertex in roi.regions[0].rings[0]}

    assert min(xs) == int(round(offset + 3 * CELL))
    assert max(xs) == int(round(offset + 7 * CELL))


# --- the ledger and the guards -------------------------------------------------


def test_the_ledger_accounts_for_every_cell():
    rng = np.random.default_rng(3)
    grid = field(28, 28)
    paint(grid, *np.where(rng.random((28, 28)) > 0.6), SCORED)
    paint(grid, *np.where(rng.random((28, 28)) > 0.85), IN_SITU)
    roi = build(make_class_map(grid), RoiParams(sigma=1.0, close_cells=2, min_area_mm2=0.01))

    assert (
        roi.seed_cells + roi.merged_cells - roi.protected_cells - roi.dropped_cells == roi.cells
    )
    assert roi.cells == sum(region.cells for region in roi.regions)
    assert roi.area_mm2 == pytest.approx(roi.cells * roi.cell_mm2, rel=1e-6)


def test_windows_outside_the_tissue_are_never_claimed():
    """The blur runs over the whole grid, so without the mask the region would bleed
    into glass at every tissue edge."""
    grid = field(20, 20)
    paint(grid, slice(2, 18), slice(2, 18), SCORED)
    inside = np.zeros((20, 20), dtype=bool)
    inside[5:15, 5:15] = True
    roi = build(make_class_map(grid, inside), RoiParams(sigma=2.0, close_cells=3, min_area_mm2=0.0))

    assert not roi.mask[~inside].any()


@pytest.mark.parametrize(
    "params",
    [
        RoiParams(threshold=1.5),
        RoiParams(threshold=-0.1),
        RoiParams(protect_in_situ=2.0),
        RoiParams(sigma=-1.0),
        RoiParams(close_cells=-1),
        RoiParams(min_area_mm2=-0.5),
        RoiParams(keep_largest=0),
    ],
)
def test_impossible_parameters_are_refused(params):
    with pytest.raises(RoiError):
        params.validate()


def test_a_class_map_with_no_tissue_is_refused_rather_than_returning_an_empty_region():
    """An empty region and 'step 8 found nothing' are different answers, and the
    second one is not something a caller should have to infer from a zero."""
    grid = field(10, 10)
    with pytest.raises(RoiError):
        build(make_class_map(grid, np.zeros((10, 10), dtype=bool)))


def test_no_invasive_anywhere_gives_an_empty_region_and_says_so():
    grid = field(12, 12)
    roi = build(make_class_map(grid), RoiParams(sigma=0.0, close_cells=0))

    assert roi.regions == ()
    assert roi.cells == 0
    assert roi.area_mm2 == 0.0
