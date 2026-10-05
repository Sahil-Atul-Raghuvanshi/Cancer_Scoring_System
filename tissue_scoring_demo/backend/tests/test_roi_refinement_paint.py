"""Step 11's live feed: the region being segmented, painting itself onto its own crop.

Three contracts are pinned here, and each of them is one a screen silently depends on.

**Every window is handed over exactly once, with its mask beside it.** The canvas
accumulates rather than redraws, so a window delivered twice is painted twice - visible
at the overlay's opacity - and one delivered never leaves a hole that nothing closes.

**A painted window's `(row, col)` indexes the slide-wide grid, not a per-region one.**
`crop.restrict` narrows step 8's grid rather than building a new one, and the client
reproduces `x_of`'s clamp and then projects into the crop box. If restriction ever
renumbered the windows, the paint would land somewhere else on the region and look
entirely plausible doing it.

**A region's bar cannot run past its own end.** The window total moves once, upward, when
a region starts - step 9 counts the candidate's windows, step 11 counts every window
whose core reaches the padded box - and a share taken before that correction must not
exceed 1.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.step08_tissue_type_segmentation.inference import WindowGrid
from app.pipeline.step11_roi_refinement import crop as crop_tools
from app.schemas.roi_refinement import RegionProgress
from app.services.roi_refinement_service import (
    _PAINT_CHUNK,
    Job,
    roi_refinement_service,
)

CELL = 256


def _grid(cols: int = 40, rows: int = 30, *, overlap: float = 0.0) -> WindowGrid:
    span = CELL if overlap == 0.0 else int(CELL / (1 - overlap))
    return WindowGrid(
        cols=cols,
        rows=rows,
        size=224,
        stride_px=224,
        span=span,
        stride=CELL,
        overlap=overlap,
        mpp=0.5,
        base_mpp=0.25,
        slide_width=cols * CELL,
        slide_height=rows * CELL,
        inside=np.ones((rows, cols), dtype=bool),
    )


def _paint_job(windows: int, *, roi_id: str = "ROI-001") -> Job:
    """A pass part-way through one region, with `windows` windows already painted."""
    job = Job(upload_id="unit", state="running")
    job.current = roi_id
    job.regions = [
        {
            "roi_id": roi_id,
            "index": 0,
            "state": "segmenting",
            "windows_done": windows,
            "windows_total": max(windows, 1),
        },
        {
            "roi_id": "ROI-002",
            "index": 1,
            "state": "pending",
            "windows_done": 0,
            "windows_total": 900,
        },
    ]
    job.paint = {
        "roi_id": roi_id,
        "crop_x": 2_048,
        "crop_y": 1_024,
        "crop_width": 4_096,
        "crop_height": 3_072,
        "span": CELL,
        "stride": CELL,
        "slide_width": 40 * CELL,
        "slide_height": 30 * CELL,
        "colours": ["#f4f4f5", "#facc15", "#3b82f6", "#ef4444", "#a855f7"],
        "labels": ["Background", "Stroma", "In-situ", "Invasive", "Other"],
        "per_pixel": True,
        "mask_px": 32,
    }
    for index in range(windows):
        job.painted.extend((index // 40, index % 40, index % 5))
        job.painted_masks.append(f"mask-{index}")
    return job


def test_a_poll_that_does_not_ask_for_paint_carries_none_of_it():
    """The feed travels only when it is asked for.

    The same poll is what a screen uses for the bars alone, and the bars are cheap: a
    region's masks are kilobytes each and a client that is not painting has no use for
    them.
    """
    run = roi_refinement_service._as_run(_paint_job(64))

    assert run.paint is None
    assert run.painted_cells == []
    assert run.painted_masks == []
    assert run.painted_cursor == 0
    # The per-region rows are not part of the feed and travel regardless - they are what
    # every row's bar is drawn from, including the rows nobody is watching paint.
    assert [one.roi_id for one in run.progress] == ["ROI-001", "ROI-002"]


def test_the_paint_feed_hands_each_window_over_exactly_once():
    """A client polling with the cursor it was given sees every window, once.

    Walked over more than one chunk, because the ceiling is where a feed would be most
    likely to drop or repeat a window.
    """
    windows = _PAINT_CHUNK + 137
    job = _paint_job(windows)

    seen: list[tuple[int, int, int]] = []
    masks: list[str] = []
    cursor = 0
    while True:
        run = roi_refinement_service._as_run(job, painted_since=cursor)
        assert run.paint is not None
        assert run.paint.roi_id == "ROI-001"
        assert run.paint.crop_width == 4_096

        if not run.painted_cells:
            break
        cells = run.painted_cells
        seen.extend(
            (cells[at], cells[at + 1], cells[at + 2]) for at in range(0, len(cells), 3)
        )
        masks.extend(run.painted_masks)
        assert run.painted_cursor > cursor
        cursor = run.painted_cursor

    assert len(seen) == windows
    assert len(masks) == windows
    # Exactly once, and in the order the model answered - the canvas draws the tail of
    # the feed and never looks back, so an out-of-order window would be drawn at the
    # wrong moment and never corrected.
    assert seen == [(index // 40, index % 40, index % 5) for index in range(windows)]
    assert masks == [f"mask-{index}" for index in range(windows)]


def test_a_windows_mask_never_arrives_before_its_position():
    """Masks are sliced to exactly the windows the triples carry, never further.

    The worker appends to both lists, and a poll can land between the two. Sending a
    mask for a window whose row and column the client does not have would leave it with
    a mask it cannot place; the reverse - a window whose mask has not landed - is drawn
    as its dominant class and is harmless.
    """
    job = _paint_job(10)
    # The state a poll can catch: the triples for window 10 have not been appended yet,
    # but its mask has.
    job.painted_masks.append("mask-10")

    run = roi_refinement_service._as_run(job, painted_since=0)

    assert len(run.painted_cells) // 3 == 10
    assert len(run.painted_masks) == 10
    assert "mask-10" not in run.painted_masks


def test_restricting_the_grid_to_a_region_keeps_the_slide_wide_indexing():
    """`(row, col)` means the same thing inside a region as it does over the slide.

    The client paints a window by reproducing `x_of(col)` in level-0 pixels and then
    projecting into the crop box. That only works because restriction narrows which
    windows run and changes nothing about where they are - so this asserts the two grids
    agree on every column and row, not merely on the shape of the array.
    """
    grid = _grid()
    box = (10 * CELL, 6 * CELL, 18 * CELL, 12 * CELL)

    restricted = crop_tools.restrict(grid, box)

    assert (restricted.cols, restricted.rows) == (grid.cols, grid.rows)
    assert (restricted.span, restricted.stride) == (grid.span, grid.stride)
    assert (restricted.slide_width, restricted.slide_height) == (
        grid.slide_width,
        grid.slide_height,
    )
    for col in range(grid.cols):
        assert restricted.x_of(col) == grid.x_of(col)
    for row in range(grid.rows):
        assert restricted.y_of(row) == grid.y_of(row)

    # And the windows it kept are the ones over the box, so a painted window's core
    # lands inside the crop the client is projecting into.
    rows, cols = np.nonzero(np.asarray(restricted.inside))
    assert len(rows) > 0
    for row, col in zip(rows, cols, strict=True):
        assert box[0] - grid.stride < grid.x_of(int(col)) < box[2]
        assert box[1] - grid.stride < grid.y_of(int(row)) < box[3]


def test_a_regions_bar_cannot_run_past_its_own_end():
    """The share clamps, because the total is provisional until the region starts.

    Step 9's count and step 11's differ - the second counts windows reaching the padded
    box - so a poll taken in the moment between a region starting and its total being
    corrected can carry more done than total.
    """
    assert RegionProgress(
        roi_id="ROI-001", windows_done=140, windows_total=120
    ).share == 1.0
    assert RegionProgress(roi_id="ROI-001", windows_done=0, windows_total=0).share == 0.0
    assert RegionProgress(
        roi_id="ROI-001", windows_done=30, windows_total=120
    ).share == 0.25


def test_a_region_that_has_not_started_still_has_a_row():
    """Every selected region is on the list from the first poll, with an empty bar.

    A row that appeared only when its region started would make the list grow as the
    pass ran, and a viewer could not see how many areas there were until the last one
    began.
    """
    run = roi_refinement_service._as_run(_paint_job(8))

    waiting = next(one for one in run.progress if one.roi_id == "ROI-002")
    assert waiting.state == "pending"
    assert waiting.windows_done == 0
    assert waiting.windows_total == 900
    assert waiting.share == 0.0


def test_the_feed_is_asked_for_over_the_wire_by_its_camelcase_name(client):
    """`paintedSince`, not `painted_since`.

    Every other field on this API is camelCase, and a query parameter that quietly did
    not match would not fail: FastAPI would take its default, the response would be a
    valid run with `paint` null, and the screen would poll forever showing an empty
    frame. So the name is pinned here rather than left to the parameter's spelling in
    Python.
    """
    roi_refinement_service._jobs["paint-wire"] = _paint_job(3)
    try:
        asked = client.get(
            "/api/v1/roi-refinement/paint-wire/run", params={"paintedSince": 0}
        ).json()
        ignored = client.get(
            "/api/v1/roi-refinement/paint-wire/run", params={"painted_since": 0}
        ).json()

        assert asked["paint"]["roiId"] == "ROI-001"
        assert asked["paintedCursor"] == 3
        assert len(asked["paintedMasks"]) == 3

        # The snake_case spelling is not the parameter, so it is simply not the feed.
        assert ignored["paint"] is None
        assert ignored["paintedCells"] == []

        resumed = client.get(
            "/api/v1/roi-refinement/paint-wire/run", params={"paintedSince": 2}
        ).json()
        assert resumed["paintedMasks"] == ["mask-2"]
        assert resumed["paintedCursor"] == 3
    finally:
        roi_refinement_service._jobs.pop("paint-wire", None)
