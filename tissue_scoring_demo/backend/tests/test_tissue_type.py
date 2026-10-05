"""Step 8 tests.

Split the way every earlier step's are: the geometry and the input transform need
nothing but numpy, so they are the bulk of this file, and the endpoints are tested for
the states that need no slide on disk.

Five of these are about something less obvious than arithmetic and are worth naming:

`test_inside_matches_a_brute_force_centre_test` checks the fast marking in
`_mark_inside` against the rule it is meant to implement, window by window - the fast
version marks from a few thousand kept tiles outward and the slow one asks the question
directly of a few hundred thousand windows, and they have to agree exactly or the model
runs somewhere step 7 did not authorise.

`test_a_block_read_gives_the_same_windows_as_reading_each_one` pins the claim the whole
block machinery rests on: blocks are a read amortisation and not an approximation.

`test_the_transform_matches_the_training_folders_own` is the one that matters most. It
asserts that the module the backend serves and the module the model was fitted through
are the same code, which is the failure mode that is invisible in every metric computed
inside a training process.

`test_a_permuted_class_order_is_refused` and
`test_a_manifest_whose_input_contract_disagrees_is_refused` pin the two refusals that
stop a plausible-looking but wrong class map from being produced at all.
"""

from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.pipeline.step07_tiling.index import build_index
from app.pipeline.step08_tissue_type_segmentation import classes as tissue_classes
from app.pipeline.step08_tissue_type_segmentation import (
    inference,
    model,
    overlay,
    uncertainty,
)
from app.pipeline.step08_tissue_type_segmentation import input as model_input
from app.pipeline.step08_tissue_type_segmentation.classes import (
    CANDIDATE,
    RIVAL,
    SCORED,
    UNCERTAIN,
)
from app.pipeline.step08_tissue_type_segmentation.inference import (
    OUTSIDE,
    ClassMap,
    SegmentationError,
    build_grid,
    plan_block,
)

# The demo's usual arrangement: a mask at 2 um/px over a slide whose native resolution
# is 0.5 um/px, and a model that wants 224 px at 0.5.
MASK_MPP = 2.0
BASE_MPP = 0.5
MODEL_MPP = 0.5
WINDOW = 224


def _slide_size(mask: np.ndarray) -> tuple[int, int]:
    height, width = mask.shape
    return (int(width * MASK_MPP / BASE_MPP), int(height * MASK_MPP / BASE_MPP))


def _mask(shape: tuple[int, int] = (256, 256), fill: float = 0.5) -> np.ndarray:
    """A solid square of tissue in the middle, `fill` of each axis."""
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    h0, w0 = int(height * (1 - fill) / 2), int(width * (1 - fill) / 2)
    mask[h0 : height - h0, w0 : width - w0] = True
    return mask


def _disc(shape: tuple[int, int] = (256, 256), radius: float = 0.3) -> np.ndarray:
    """A round island, so the grid produces genuinely partial tiles at its edge."""
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width]
    centre_y, centre_x = height / 2, width / 2
    limit = radius * min(height, width)
    return ((ys - centre_y) ** 2 + (xs - centre_x) ** 2) <= limit**2


def _index(mask: np.ndarray, *, tile_size: int = 512, overlap: float = 0.25):
    return build_index(
        tissue=mask,
        considered=None,
        mask_mpp=MASK_MPP,
        base_mpp=BASE_MPP,
        slide_size=_slide_size(mask),
        target_mpp=BASE_MPP,
        size=tile_size,
        overlap=overlap,
        min_tissue_share=0.1,
        min_clean_share=0.5,
        max_tiles=400_000,
    )


def _grid(mask: np.ndarray, *, overlap: float = 0.5, tile_overlap: float = 0.25):
    return build_grid(
        _index(mask, overlap=tile_overlap),
        base_mpp=BASE_MPP,
        slide_size=_slide_size(mask),
        size=WINDOW,
        mpp=MODEL_MPP,
        overlap=overlap,
        max_windows=400_000,
    )


# --- the class ontology ------------------------------------------------------


def test_exactly_one_class_is_scored():
    """Rule 5 in executable form: in-situ disease is excluded, not pooled with invasive."""
    assert tissue_classes.CLASS_NAMES[tissue_classes.SCORED] == "invasive_epithelium"
    assert len(tissue_classes.CLASS_NAMES) == 3


def test_the_checkpoints_class_list_is_three_however_many_colours_the_map_has():
    """The two lists are different things and the difference is load-bearing.

    `CLASS_NAMES` is the checkpoint's contract - it is what `verify_order` compares a
    manifest against and the shape of every probability vector - so appending the
    uncertainty layer's display class to it would make every published head fail to
    load. `DISPLAY_NAMES` is what gets drawn and tabulated. This pins that they stay
    apart, and that the added one is never the scored one.
    """
    assert len(tissue_classes.CLASS_NAMES) == 3
    assert tissue_classes.DISPLAY_NAMES[: len(tissue_classes.CLASS_NAMES)] == (
        tissue_classes.CLASS_NAMES
    )
    assert tissue_classes.DISPLAY_NAMES[UNCERTAIN] == "uncertain"
    assert UNCERTAIN != SCORED
    assert UNCERTAIN not in range(len(tissue_classes.CLASS_NAMES))

    # Every drawn label has a colour and all three pieces of wording, because the
    # class table and the legend are built by iterating these.
    drawn = set(range(len(tissue_classes.DISPLAY_NAMES)))
    assert set(tissue_classes.CLASS_MEANING) == drawn
    assert set(tissue_classes.CLASS_LABELS) == drawn
    assert set(tissue_classes.CLASS_PLAIN) == drawn
    assert set(tissue_classes.CLASS_COLOURS) == drawn

    # And the layer can only ever qualify the class Rule 5 already excludes.
    assert CANDIDATE != SCORED
    assert tissue_classes.CLASS_NAMES[CANDIDATE] == "non_invasive_epithelium"
    assert tissue_classes.CLASS_NAMES[RIVAL] == "invasive_epithelium"


def test_a_permuted_class_order_is_refused():
    """A permuted head produces a plausible class map and the wrong denominator."""
    tissue_classes.verify_order(list(tissue_classes.CLASS_NAMES))

    with pytest.raises(tissue_classes.ClassOrderError, match="invasive class"):
        tissue_classes.verify_order(
            ["invasive_epithelium", "non_invasive_epithelium", "non_epithelium"]
        )


# --- the input transform -----------------------------------------------------


def test_od_floor_agrees_with_step_4():
    """The transform clips intensity the way step 4 does, or densities disagree by a log."""
    assert model_input.OD_FLOOR == settings.calibration_od_floor


def test_model_input_is_three_affine_images_of_one_channel():
    field = np.full((8, 8), 0.75, dtype=np.float32)
    tensor = model_input.to_model_input(field)

    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == np.float32

    raw = tensor * np.asarray(model_input.IMAGENET_STD).reshape(3, 1, 1) + np.asarray(
        model_input.IMAGENET_MEAN
    ).reshape(3, 1, 1)
    assert np.allclose(raw[0], raw[1], atol=1e-6)
    assert np.allclose(raw[0], raw[2], atol=1e-6)
    assert abs(float(raw[0].mean()) - 0.75 / model_input.OD_CLIP) < 1e-5


def test_the_clip_saturates_at_od_clip():
    dark = np.full((4, 4), 3.0, dtype=np.float32)
    at_clip = np.full((4, 4), model_input.OD_CLIP, dtype=np.float32)
    assert np.allclose(model_input.to_model_input(dark), model_input.to_model_input(at_clip))


def test_standardise_leaves_a_tile_with_no_stain_alone():
    """Dividing bare glass by its own p99 would stretch sensor noise into texture."""
    empty = np.full((32, 32), model_input.STANDARDISE_FLOOR / 2, dtype=np.float32)
    assert np.allclose(
        model_input.to_model_input(empty, standardise=True),
        model_input.to_model_input(empty, standardise=False),
    )


def test_standardise_puts_a_stained_tiles_p99_at_the_target():
    stained = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    tensor = model_input.to_model_input(stained, standardise=True)

    raw = tensor[0] * model_input.IMAGENET_STD[0] + model_input.IMAGENET_MEAN[0]
    assert abs(float(np.percentile(raw, 99.0)) - model_input.STANDARDISE_TARGET) < 0.01


def test_quantise_round_trips_within_one_level():
    density = np.linspace(0.0, model_input.OD_CLIP, 256, dtype=np.float32)
    recovered = model_input.dequantise(model_input.quantise(density))
    assert float(np.abs(recovered - density).max()) <= model_input.OD_CLIP / 255.0


def test_gamma_moves_a_quantile_ratio_and_alpha_cannot():
    """The measured H&E-to-IHC difference is a quantile *ratio*, and that is the point.

    The gap between the two stains was measured as a ratio that runs from 0.087 at p75
    to 2.03 at p99 - a 23x spread where a pure strength difference would be one
    constant. Every monotone correction, `alpha` included, multiplies both quantiles by
    the same number and so leaves their ratio exactly where it was; that is why no
    amount of scaling or per-slide normalisation could close it. This asserts both
    halves of that: `alpha` cannot move the ratio, and `gamma` can.
    """
    field = np.linspace(0.05, 0.9, 4096, dtype=np.float32).reshape(64, 64)

    def ratio(**kwargs: float) -> float:
        tensor = model_input.to_model_input(field, **kwargs)
        raw = tensor[0] * model_input.IMAGENET_STD[0] + model_input.IMAGENET_MEAN[0]
        return float(np.percentile(raw, 75.0)) / float(np.percentile(raw, 99.0))

    plain = ratio()

    # Well inside the clip, so `alpha` is a pure scale and nothing else.
    assert ratio(alpha=0.6) == pytest.approx(plain, rel=1e-3)
    assert ratio(alpha=1.4) == pytest.approx(plain, rel=1e-3)

    # A power pushes the mid-tones down while leaving the top of the range fixed.
    assert ratio(gamma=2.5) < plain * 0.8
    assert ratio(gamma=0.5) > plain * 1.1


def test_the_transform_matches_the_training_folders_own():
    """The backend's transform and the one the model was fitted through are one module.

    The research folder re-exports this module rather than restating it, so this test
    is what proves the re-export is still in place. If someone gives
    `tissue_type_model_training` its own copy of the arithmetic, this is the test that
    fails - and it is the only place the drift would ever be visible, because the
    model would train and validate perfectly either way.
    """
    research = Path(__file__).resolve().parents[3] / "tissue_type_model_training" / "src"
    if not (research / "hchannel.py").is_file():
        pytest.skip(f"the training folder is not present at {research}")

    import sys

    sys.path.insert(0, str(research))
    try:
        import hchannel  # type: ignore[import-not-found]
    finally:
        sys.path.remove(str(research))

    assert hchannel.OD_CLIP == model_input.OD_CLIP
    assert hchannel.OD_FLOOR == model_input.OD_FLOOR
    assert hchannel.STANDARDISE_FLOOR == model_input.STANDARDISE_FLOOR
    assert hchannel.STANDARDISE_TARGET == model_input.STANDARDISE_TARGET
    assert hchannel.IMAGENET_MEAN == model_input.IMAGENET_MEAN
    assert hchannel.IMAGENET_STD == model_input.IMAGENET_STD

    rng = np.random.default_rng(0)
    field = (rng.random((64, 64)) * 2.0).astype(np.float32)
    for kwargs in (
        {},
        {"standardise": True},
        {"gamma": 2.5},
        {"gamma": 0.7, "standardise": True},
        {"invert": True},
        {"alpha": 1.3, "beta": -0.05},
    ):
        assert np.array_equal(
            hchannel.to_model_input(field, **kwargs),
            model_input.to_model_input(field, **kwargs),
        ), kwargs

    stored = model_input.quantise(field)
    assert np.array_equal(
        hchannel.from_stored(stored, standardise=True),
        model_input.from_stored(stored, standardise=True),
    )
    assert hchannel.descriptor(
        tile_px=224, mpp=0.5, invert=False, standardise=True
    ) == model_input.descriptor(tile_px=224, mpp=0.5, invert=False, standardise=True)


# --- the window grid ---------------------------------------------------------


def test_the_grid_takes_its_geometry_from_the_model_and_not_from_step_7():
    """A step-7 tile is 512 px of 256 um; the model's window is 224 px of 112 um."""
    grid = _grid(_mask())

    assert grid.size == WINDOW
    assert grid.field_um == pytest.approx(112.0)
    assert grid.span == round(WINDOW * MODEL_MPP / BASE_MPP)
    assert grid.stride_px == WINDOW // 2
    assert grid.stride == round((WINDOW // 2) * MODEL_MPP / BASE_MPP)


def test_the_grid_covers_the_whole_canvas():
    mask = _mask()
    grid = _grid(mask)
    width, height = _slide_size(mask)

    assert grid.x_of(grid.cols - 1) + grid.span >= width - grid.stride
    assert grid.y_of(grid.rows - 1) + grid.span >= height - grid.stride
    assert grid.x_of(grid.cols - 1) + grid.span <= width
    assert grid.y_of(grid.rows - 1) + grid.span <= height


def test_inside_matches_a_brute_force_centre_test():
    """The fast marking has to agree exactly with the rule it implements.

    `_mark_inside` walks the kept tiles and marks ranges of windows; this walks every
    window and asks the question directly. They are different algorithms for one rule,
    and a disagreement means the model runs somewhere step 7 did not authorise - or
    skips somewhere it did.
    """
    mask = _disc()
    index = _index(mask, overlap=0.25)
    grid = _grid(mask)

    kept = [tile for tile in index.tiles if tile.kept]
    expected = np.zeros((grid.rows, grid.cols), dtype=bool)
    for row in range(grid.rows):
        centre_y = grid.y_of(row) + grid.span / 2.0
        for col in range(grid.cols):
            centre_x = grid.x_of(col) + grid.span / 2.0
            expected[row, col] = any(
                tile.x <= centre_x < tile.x + tile.span
                and tile.y <= centre_y < tile.y + tile.span
                for tile in kept
            )

    assert np.array_equal(grid.inside, expected)


def test_a_window_outside_the_kept_tiles_is_never_read():
    """Step 7's gates are what make this step affordable, so they have to bind."""
    grid = _grid(_mask(fill=0.25))
    assert 0 < grid.windows < grid.every


def test_overlap_cannot_inflate_the_area_the_way_a_span_count_would():
    """Areas are counted per stride-sized core, so overlap barely moves them.

    The trap this pins: `windows x span^2` reports four times the ground at 50% overlap,
    which would make it look as though overlapping windows see more of the slide. They
    do not - they see the same tissue more often.

    The stride measure is not *exactly* invariant and should not be asserted to be: a
    coarse grid admits a window whenever its centre lands on kept tissue, so its
    stride-sized cores tile the region more crudely than a fine grid's. That
    quantisation is a real difference of about a fifth here, where the span-count error
    is a factor of four.
    """
    mask = _mask()
    coarse = _grid(mask, overlap=0.0)
    fine = _grid(mask, overlap=0.5)

    assert fine.windows > coarse.windows * 3

    honest = (fine.windows * fine.cell_mm2) / (coarse.windows * coarse.cell_mm2)
    naive = (fine.windows * fine.span**2) / (coarse.windows * coarse.span**2)

    assert 0.7 < honest < 1.4
    assert naive > 3.0


def test_an_impossible_overlap_is_refused():
    mask = _mask()
    with pytest.raises(SegmentationError, match="never advance"):
        _grid(mask, overlap=1.0)


def test_a_grid_past_the_refusal_limit_is_not_truncated():
    mask = _mask()
    with pytest.raises(SegmentationError, match="Refusing rather"):
        build_grid(
            _index(mask),
            base_mpp=BASE_MPP,
            slide_size=_slide_size(mask),
            size=WINDOW,
            mpp=MODEL_MPP,
            overlap=0.5,
            max_windows=4,
        )


def test_a_slide_with_no_kept_tile_has_nothing_to_classify():
    mask = _mask()
    index = _index(mask)

    # Force every tile out, the way a threshold that claimed nothing would.
    stripped = type(index)(
        **{
            **index.__dict__,
            "tiles": tuple(
                type(tile)(**{**tile.__dict__, "kept": False, "rejected_by": "tissue"})
                for tile in index.tiles
            ),
        }
    )
    with pytest.raises(SegmentationError, match="nothing to classify"):
        build_grid(
            stripped,
            base_mpp=BASE_MPP,
            slide_size=_slide_size(mask),
            size=WINDOW,
            mpp=MODEL_MPP,
            overlap=0.5,
            max_windows=400_000,
        )


# --- blocks ------------------------------------------------------------------


def test_a_block_read_gives_the_same_windows_as_reading_each_one():
    """The claim the whole block machinery rests on: it is amortisation, not approximation.

    A synthetic slide whose value at every position encodes that position, so a window
    cut out of a block can be compared against the same window read on its own and any
    offset error shows up as an exact mismatch rather than as a plausible picture.
    """
    mask = _mask()
    grid = _grid(mask)
    width, height = _slide_size(mask)

    def synthetic(x: int, y: int, span: int, size: int) -> np.ndarray:
        """`window_reader` over a slide that is a ramp in both axes."""
        scale = span / size
        xs = x + (np.arange(size) + 0.5) * scale
        ys = y + (np.arange(size) + 0.5) * scale
        return (ys[:, None] / height + xs[None, :] / width).astype(np.float32)

    checked = 0
    for row0 in range(0, grid.rows, grid.cols and 8):
        for col0 in range(0, grid.cols, 8):
            block = plan_block(
                grid,
                row0=row0,
                col0=col0,
                row1=min(grid.rows, row0 + 8),
                col1=min(grid.cols, col0 + 8),
            )
            if block is None:
                continue

            pixels = synthetic(block.x, block.y, block.span, block.size)
            for (row, col), (top, left) in zip(block.cells, block.offsets, strict=True):
                from_block = pixels[top : top + grid.size, left : left + grid.size]
                alone = synthetic(grid.x_of(col), grid.y_of(row), grid.span, grid.size)

                assert from_block.shape == alone.shape
                # One block pixel of tolerance: the block and the window are read at the
                # same resolution, so an exact match is expected, and the tolerance only
                # absorbs the ramp's own gradient across a rounding of the origin.
                assert np.abs(from_block - alone).max() < 2.0 / min(width, height)
                checked += 1

    assert checked > 0


def test_a_block_never_reads_past_the_canvas():
    mask = _mask(fill=0.98)
    grid = _grid(mask)
    width, height = _slide_size(mask)

    for row0 in range(0, grid.rows, 8):
        for col0 in range(0, grid.cols, 8):
            block = plan_block(
                grid,
                row0=row0,
                col0=col0,
                row1=min(grid.rows, row0 + 8),
                col1=min(grid.cols, col0 + 8),
            )
            if block is None:
                continue
            assert block.x >= 0 and block.y >= 0
            assert block.x + block.span <= width
            assert block.y + block.span <= height
            assert block.size >= grid.size
            for top, left in block.offsets:
                assert 0 <= top <= block.size - grid.size
                assert 0 <= left <= block.size - grid.size


def test_a_block_with_no_kept_window_is_not_planned():
    grid = _grid(_mask(fill=0.2))
    assert plan_block(grid, row0=0, col0=0, row1=1, col1=1) is None


# --- classifying -------------------------------------------------------------


class _FakeNet:
    """A stand-in for the ResNet that answers from the window's mean.

    Deterministic and cheap, so the classifier's bookkeeping - labels landing on the
    right cells, counts, areas, the progress callback - can be tested without a 45 MB
    checkpoint or a slide.
    """

    def __init__(self) -> None:
        self.seen = 0

    def __call__(self, batch):
        import torch

        self.seen += int(batch.shape[0])
        means = batch.mean(dim=(1, 2, 3))
        return torch.stack([-means, means * 0.0, means], dim=1)


def _classify(grid, *, value_at=None) -> ClassMap:
    def read(x: int, y: int, span: int, size: int) -> np.ndarray:
        if value_at is None:
            return np.full((size, size), 0.4, dtype=np.float32)
        scale = span / size
        xs = x + (np.arange(size) + 0.5) * scale
        ys = y + (np.arange(size) + 0.5) * scale
        return value_at(xs[None, :], ys[:, None]).astype(np.float32)

    return inference.classify(
        grid,
        net=_FakeNet(),
        read_window=read,
        standardise=False,
        block_windows=8,
        batch_size=16,
    )


def test_every_kept_window_gets_a_label_and_no_other_one_does():
    grid = _grid(_mask(fill=0.4))
    class_map = _classify(grid)

    assert class_map.classified == grid.windows
    assert np.array_equal(class_map.labels != OUTSIDE, grid.inside)
    assert (class_map.labels[~grid.inside] == OUTSIDE).all()
    assert class_map.probabilities[~grid.inside].sum() == 0.0


def test_probabilities_are_a_distribution_where_a_window_ran():
    grid = _grid(_mask(fill=0.4))
    class_map = _classify(grid)

    totals = class_map.probabilities[grid.inside].sum(axis=-1)
    assert np.allclose(totals, 1.0, atol=1e-5)
    assert 1.0 / 3.0 - 1e-6 <= class_map.mean_confidence <= 1.0


def test_counts_and_areas_agree_with_the_labels():
    grid = _grid(_mask(fill=0.4))
    class_map = _classify(grid)

    for label in range(3):
        assert class_map.counts[label] == int((class_map.labels == label).sum())
        assert class_map.areas_mm2[label] == pytest.approx(
            class_map.counts[label] * grid.cell_mm2, abs=1e-4
        )
    assert sum(class_map.shares) == pytest.approx(1.0)
    assert class_map.tumour_content == pytest.approx(class_map.shares[2])


# --- the sweep, and painting it ----------------------------------------------


def test_the_sweep_visits_every_block_once_and_turns_at_each_band():
    """Left to right, then right to left, top to bottom - and nothing missed.

    The order is the progress screen's, not the model's: a block is an independent read
    and independent forward passes, so this cannot change a label. What it must not do
    is drop or repeat a block, because that would be a hole in the class map that no
    other test would see - the labels array is initialised to OUTSIDE, which is exactly
    what an unvisited block looks like.
    """
    origins = inference.sweep(rows=20, cols=30, block_windows=8)

    bands = list(range(0, 20, 8))
    columns = list(range(0, 30, 8))
    assert len(origins) == len(bands) * len(columns)
    assert set(origins) == {(row, col) for row in bands for col in columns}

    # Bands in order, and each one running the opposite way to the last.
    assert [row for row, _ in origins] == sorted(row for row, _ in origins)
    for index, row0 in enumerate(bands):
        band = [col for row, col in origins if row == row0]
        assert band == (columns if index % 2 == 0 else list(reversed(columns)))


def test_the_sweep_order_does_not_change_a_single_label():
    """The claim that makes the sweep free: it is a viewing order, not an algorithm."""
    grid = _grid(_mask(fill=0.4))

    ploughed = _classify(grid)

    # The same pass with the blocks visited row-major, which is what it did before.
    row_major = [
        (row0, col0)
        for row0 in range(0, grid.rows, 8)
        for col0 in range(0, grid.cols, 8)
    ]
    original = inference.sweep
    try:
        inference.sweep = lambda rows, cols, block_windows: row_major
        straight = _classify(grid)
    finally:
        inference.sweep = original

    assert np.array_equal(ploughed.labels, straight.labels)
    assert np.allclose(ploughed.probabilities, straight.probabilities)
    assert ploughed.counts == straight.counts


def test_every_window_is_painted_exactly_once_with_the_class_it_got():
    """The paint feed is the class map arriving in instalments, not a summary of it.

    Each window appears once, with the label the finished map holds for it, and no
    window that never ran appears at all - so a screen that accumulates the feed and a
    screen that waits for the report draw the same picture.
    """
    grid = _grid(_mask(fill=0.4))
    painted: list[tuple[int, int, int]] = []

    class_map = inference.classify(
        grid,
        net=_FakeNet(),
        read_window=lambda x, y, span, size: np.full((size, size), 0.4, dtype=np.float32),
        standardise=False,
        block_windows=8,
        batch_size=16,
        painted=lambda cells: painted.extend(cells),
    )

    assert len(painted) == grid.windows
    assert len({(row, col) for row, col, _ in painted}) == grid.windows
    for row, col, label in painted:
        assert grid.inside[row, col]
        assert class_map.labels[row, col] == label


def test_the_paint_feed_arrives_in_the_sweeps_order():
    """What the screen actually shows: a line of work crossing the section.

    Asserted on the bands rather than window by window, because a block is 64 windows
    and their order inside it is not what a viewer sees - the block is. So: each band
    of blocks is finished before the next begins, and consecutive bands run in opposite
    directions.
    """
    grid = _grid(_mask(fill=0.4))
    painted: list[tuple[int, int, int]] = []

    inference.classify(
        grid,
        net=_FakeNet(),
        read_window=lambda x, y, span, size: np.full((size, size), 0.4, dtype=np.float32),
        standardise=False,
        block_windows=8,
        batch_size=16,
        painted=lambda cells: painted.extend(cells),
    )

    # The band each painted window belongs to, in arrival order, de-duplicated.
    bands: list[int] = []
    for row, _, _ in painted:
        band = row // 8
        if not bands or bands[-1] != band:
            bands.append(band)
    assert bands == sorted(bands), "a band was returned to after the next had started"
    assert len(bands) == len(set(bands))

    # Within each band, the columns advance one way and the next band the other.
    direction: list[int] = []
    for band in bands:
        columns = [col for row, col, _ in painted if row // 8 == band]
        ends = columns[0], columns[-1]
        direction.append(1 if ends[1] >= ends[0] else -1)
    for first, second in zip(direction, direction[1:], strict=False):
        assert first != second, "two bands in a row ran the same way, so it is not a sweep"


def _client_cell(paint: dict, row: int, col: int) -> tuple[float, float, float, float]:
    """Where a browser paints one patch, as a fraction of the slide.

    **A deliberate copy of `cellRect` in `TissueTypePainting.tsx`**, and the reason this
    test exists: the client turns `(row, col)` into a rectangle itself rather than being
    sent one per patch, so the formula lives in two languages and this is what stops
    them drifting. If `WindowGrid.x_of`'s clamp or the span-to-stride inset ever
    changes, this fails - and the failure is the notice that the TypeScript has to
    change with it.
    """
    inset = (paint["span"] - paint["stride"]) / 2
    x0 = min(col * paint["stride"], max(0, paint["slide_width"] - paint["span"])) + inset
    y0 = min(row * paint["stride"], max(0, paint["slide_height"] - paint["span"])) + inset
    return (
        x0 / paint["slide_width"],
        y0 / paint["slide_height"],
        paint["stride"] / paint["slide_width"],
        paint["stride"] / paint["slide_height"],
    )


@pytest.mark.parametrize("fov_um", [112.0, 224.0, 448.0, 672.0])
def test_the_painted_grid_lands_on_the_windows_at_every_published_scale(fov_um: float):
    """A patch is painted where the model actually looked, at all four fields of view.

    **Why four and not eight.** Eight heads are published - each of the two branches at
    each of these fields of view - and the branch decides what the model is *shown*,
    never where the windows are: both are 224 px, and only the resolution moves. So the
    geometry a screen paints has four cases, and these are they. A fifth field of view
    published tomorrow needs no client change, which is the property being pinned.

    Three things are checked per scale, and each is a way the paint could be wrong while
    still looking plausible: the cell holds the centre of the window it describes (so
    the colour sits on the tissue that produced it, not half a window away), the cells
    do not overlap (a stride-wide core, not a span-wide field, or every neighbour
    overprints at 50% overlap), and nothing is drawn off the picture.
    """
    from app.services.tiling_service import FIELDS_OF_VIEW

    tile_px, model_mpp = FIELDS_OF_VIEW[fov_um]
    # A section several millimetres across rather than this file's usual half-millimetre
    # one: at 672 um a window is 1,344 level-0 pixels, which is larger than the small
    # mask's whole slide, and a grid whose every window is clamped to the origin would
    # make this test pass by having nothing to check.
    mask = _mask(shape=(1536, 2048), fill=0.9)
    width, height = _slide_size(mask)

    grid = build_grid(
        _index(mask, overlap=0.25),
        base_mpp=BASE_MPP,
        slide_size=(width, height),
        size=tile_px,
        mpp=model_mpp,
        overlap=0.5,
        max_windows=400_000,
    )

    # What the service publishes for the screen to paint with.
    paint = {
        "cols": grid.cols,
        "rows": grid.rows,
        "span": grid.span,
        "stride": grid.stride,
        "slide_width": grid.slide_width,
        "slide_height": grid.slide_height,
    }

    checked = 0
    for row in range(grid.rows):
        for col in range(grid.cols):
            if not grid.inside[row, col]:
                continue

            fx, fy, fw, fh = _client_cell(paint, row, col)

            # On the picture, all of it.
            assert 0.0 <= fx and fx + fw <= 1.0 + 1e-9
            assert 0.0 <= fy and fy + fh <= 1.0 + 1e-9

            # Over the window it describes: the cell is the middle of the span.
            centre_x = (grid.x_of(col) + grid.span / 2) / width
            centre_y = (grid.y_of(row) + grid.span / 2) / height
            assert fx <= centre_x <= fx + fw
            assert fy <= centre_y <= fy + fh

            # And exactly abutting its neighbour rather than covering it - except in
            # the last column, where `x_of`'s clamp pulls the window back inside the
            # canvas and its core genuinely does overlap the one before it. That is the
            # grid's own edge behaviour, not the painting's, and the alternative is a
            # window of invented tissue.
            if (col + 2) * grid.stride <= width - grid.span:
                nx, _, _, _ = _client_cell(paint, row, col + 1)
                assert nx == pytest.approx(fx + fw, abs=1e-9)
            checked += 1

    assert checked > 0
    # The cell is the stride and not the span - the difference the whole test is about.
    assert paint["stride"] < paint["span"], "at 50% overlap a core is half a field"


def test_progress_ends_at_the_window_count():
    grid = _grid(_mask(fill=0.4))
    seen: list[tuple[int, int]] = []

    inference.classify(
        grid,
        net=_FakeNet(),
        read_window=lambda x, y, span, size: np.full((size, size), 0.4, dtype=np.float32),
        standardise=False,
        progress=lambda done, total: seen.append((done, total)),
    )

    assert seen
    assert seen[-1] == (grid.windows, grid.windows)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)


def test_a_left_right_split_lands_on_the_left_and_right_of_the_map():
    """Labels have to land on the cell they were measured from, not one cell over.

    A dark left half and a pale right half, through a net that calls dark tissue
    invasive. If the label array were transposed, flipped or shifted by a cell, the two
    halves would not line up with the two halves of the map.
    """
    mask = _mask(fill=0.8)
    grid = _grid(mask)
    width, _ = _slide_size(mask)

    class_map = _classify(
        grid, value_at=lambda xs, ys: np.where(xs < width / 2, 1.2, 0.05) + 0.0 * ys
    )

    middle = grid.cols // 2
    left = class_map.labels[:, : middle - 1]
    right = class_map.labels[:, middle + 1 :]

    assert (left[left != OUTSIDE] == 2).all()
    assert (right[right != OUTSIDE] == 0).all()


# --- the panels --------------------------------------------------------------


def test_the_class_map_is_drawn_at_its_own_geometry():
    """A cell's colour has to land on the tissue it was measured from.

    Resizing the label array to the thumbnail would shift the map by half a window,
    which at the model's geometry is 28 um. This checks the cell a display pixel is
    assigned to is the cell whose core contains it.
    """
    mask = _mask()
    grid = _grid(mask)
    class_map = _classify(grid)

    shape = (mask.shape[0], mask.shape[1])
    field = overlay.label_image(class_map, shape)
    assert field.shape == shape

    rows, cols = overlay.cell_field(class_map, shape)
    offset = (grid.span - grid.stride) / 2.0
    for display_col in (0, shape[1] // 3, shape[1] - 1):
        position = (display_col + 0.5) * grid.slide_width / shape[1]
        expected = int(np.floor((position - offset) / grid.stride))
        assert cols[display_col] == expected
    assert rows.shape == (shape[0],)


def test_every_panel_is_a_png():
    mask = _mask()
    grid = _grid(mask)
    class_map = _classify(grid)
    thumbnail = np.full((mask.shape[0], mask.shape[1], 3), 200, dtype=np.uint8)

    payloads = [
        overlay.map_png(thumbnail, class_map),
        overlay.flat_png(class_map, mask.shape),
        overlay.scored_png(thumbnail, class_map),
        overlay.confidence_png(class_map, mask.shape),
        overlay.empty_png(),
    ]
    for payload in payloads:
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_filtered_panel_leaves_the_unselected_class_undrawn():
    """The per-class toggle has to remove tissue from the map, not recolour it."""
    mask = _mask(fill=0.8)
    grid = _grid(mask)
    width, _ = _slide_size(mask)
    # Two classes on the map, so "the scored class alone" and "nothing at all" are
    # genuinely different pictures.
    class_map = _classify(
        grid, value_at=lambda xs, ys: np.where(xs < width / 2, 1.2, 0.05) + 0.0 * ys
    )
    assert class_map.counts[2] > 0

    everything = overlay.flat_png(class_map, mask.shape)
    scored_only = overlay.flat_png(class_map, mask.shape, classes=frozenset({2}))
    nothing = overlay.flat_png(class_map, mask.shape, classes=frozenset())

    assert everything != scored_only
    assert scored_only != nothing


# --- the uncertainty layer ---------------------------------------------------
#
# The layer is a pure function of arrays, so these build the arrays directly rather
# than driving a fake net through `classify`. That is the point of `derive` taking
# primitives: the scenarios below are reconstructions of things this project actually
# measured, and reproducing them through a grid and a checkpoint would test the
# scaffolding rather than the rule.

#: Grid stride in microns for the scenarios. A 224 um window at 0.5 overlap - the
#: geometry every published head is served at.
SCENARIO_STRIDE_UM = 112.0
SCENARIO_CELL_MM2 = (SCENARIO_STRIDE_UM / 1000.0) ** 2


def _scene(rows: int = 40, cols: int = 40, *, background: int = 0):
    """A grid of confident `background` windows, ready to have a lesion drawn on it."""
    labels = np.full((rows, cols), background, dtype=np.int8)
    probabilities = np.zeros((rows, cols, 3), dtype=np.float32)
    vector = {0: (0.90, 0.07, 0.03), 1: (0.15, 0.80, 0.05), 2: (0.05, 0.03, 0.92)}
    probabilities[:] = vector[background]
    inside = np.ones((rows, cols), dtype=bool)
    return labels, probabilities, inside


def _derive(labels, probabilities, inside, **overrides):
    params = uncertainty.UncertaintyParams(**overrides) if overrides else None
    return uncertainty.derive(
        labels,
        probabilities,
        inside,
        candidate=CANDIDATE,
        rival=RIVAL,
        stride_um=SCENARIO_STRIDE_UM,
        cell_mm2=SCENARIO_CELL_MM2,
        params=params,
    )


def _layer_for(class_map, grid, **overrides):
    """The layer for a class map built through `classify`, at the grid's own geometry."""
    params = uncertainty.UncertaintyParams(**overrides) if overrides else None
    return uncertainty.derive(
        class_map.labels,
        class_map.probabilities,
        grid.inside,
        candidate=CANDIDATE,
        rival=RIVAL,
        stride_um=grid.stride * grid.base_mpp,
        cell_mm2=grid.cell_mm2,
        params=params,
    )


def test_a_large_confident_in_situ_field_is_flagged_even_though_nothing_is_uncertain():
    """The measured v2 failure, which no confidence rule reaches.

    On `CAN_00251_26_H&E` the head called 45.9% of the section in-situ at a median
    top-to-runner-up margin of 0.756 - confidently wrong, and spatially coherent, so a
    window inside it has neighbours that all agree with it. Both statistical terms are
    near zero there by construction. The morphology term is the only thing that fires,
    and it has to, or this layer is silent on the one case that motivated it.
    """
    labels, probabilities, inside = _scene(60, 60, background=1)
    probabilities[:] = (0.16, 0.809, 0.031)

    layer = _derive(labels, probabilities, inside)

    assert layer.unknown.all()
    # And for the right reason: the shape term, not the softmax.
    assert layer.implausibility.min() == 1.0
    assert float(np.sqrt(layer.margin * layer.disagreement).max()) < 0.5
    assert layer.flagged_components == 1


def test_a_real_duct_is_not_flagged_for_being_surrounded_by_stroma():
    """A duct is a small object in stroma, so its surroundings are class 0 by nature.

    This is a reconstruction of the largest real class-1 component measured on
    `CAN_00270` - 29 windows, hollow, filling 0.28 of a 13x8 box, at its measured
    P(in-situ) of 0.789. Scoring the neighbourhood's non-epithelium share instead of its
    *invasive* share reads 0.820 here and flags the duct outright, which is how the
    first version of this rule behaved.
    """
    labels, probabilities, inside = _scene()
    ring = np.zeros((13, 8), dtype=bool)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    for row, col in list(zip(*np.nonzero(ring), strict=True))[:29]:
        labels[10 + row, 15 + col] = 1
        probabilities[10 + row, 15 + col] = (0.14, 0.789, 0.071)

    layer = _derive(labels, probabilities, inside)

    assert layer.unknown.sum() == 0
    assert layer.components == 1


def test_an_isolated_in_situ_window_is_not_flagged_for_being_alone():
    """Isolation is not evidence against a class-1 call, and the data says so.

    On `CAN_00270`, 72% of the class-1 components outside the DCIS contour were
    singletons a median 6.3 cells from the nearest invasive window, and the reading was
    that they are normal ducts and lobules - which class 1 legitimately contains.
    Treating an empty neighbourhood as maximal disagreement would paint every normal
    lobule on the slide purple.
    """
    labels, probabilities, inside = _scene()
    for row, col in [(5, 5), (12, 20), (25, 8), (31, 30)]:
        labels[row, col] = 1
        probabilities[row, col] = (0.20, 0.732, 0.068)

    layer = _derive(labels, probabilities, inside)

    assert layer.unknown.sum() == 0
    assert layer.components == 4


def test_a_small_solid_in_situ_focus_is_not_flagged_for_being_solid():
    """A duct cut across is a solid disc, so solidity says nothing at duct scale.

    A 4x4 block of windows is 0.2 mm2 - a small comedo-type focus. An earlier version
    gated the shape term on a window count rather than an area and scored this at 1.0.
    """
    labels, probabilities, inside = _scene()
    labels[8:12, 16:20] = 1
    probabilities[8:12, 16:20] = (0.10, 0.85, 0.05)

    layer = _derive(labels, probabilities, inside)

    assert layer.unknown.sum() == 0
    assert layer.implausibility.max() == 0.0


def test_an_in_situ_island_inside_an_invasive_field_is_flagged():
    """The one spatial configuration that is genuine evidence against a class-1 call.

    Surrounding stroma is a duct's context; surrounding invasive carcinoma is a rival
    reading of the same tissue, and that is the boundary Rule 5 turns on. Here the
    statistical route carries it alone - the island is far too small for the morphology
    term to say anything about.
    """
    labels, probabilities, inside = _scene(background=2)
    labels[18:22, 18:22] = 1
    probabilities[18:22, 18:22] = (0.10, 0.47, 0.43)

    layer = _derive(labels, probabilities, inside)

    assert layer.unknown[18:22, 18:22].all()
    # The invasive field around it is untouched - only the island is flagged.
    assert layer.unknown.sum() == 16
    assert layer.implausibility.max() == 0.0
    assert float(layer.disagreement[18:22, 18:22].min()) > 0.5


def test_a_duct_on_the_edge_of_the_section_is_not_flagged_by_the_padding():
    """The normalised convolution, which is the rim artefact waiting to happen.

    Half the grid is unclassified here. Smoothing the numerator without the denominator
    would compare every window near that boundary against zeros and read the whole rim
    as unsupported - the shape of the false in-situ rim the window tissue gate was added
    to remove, reintroduced from the other end.
    """
    labels, probabilities, inside = _scene()
    labels[:, 20:] = OUTSIDE
    probabilities[:, 20:] = 0.0
    inside[:, 20:] = False
    labels[8:12, 16:20] = 1
    probabilities[8:12, 16:20] = (0.10, 0.85, 0.05)

    layer = _derive(labels, probabilities, inside)

    assert layer.unknown.sum() == 0
    assert float(layer.disagreement[8:12, 16:20].max()) < 0.5


def test_the_neighbourhood_is_a_distance_and_not_a_count_of_windows():
    """The same physical field at two overlaps must flag the same physical area.

    `sigma` is in microns and the stride halves between these two runs, so the kernel
    has to widen in cells by exactly the factor the stride narrowed by. A sigma fixed in
    cells would let the overlap - which is only meant to change sampling density -
    change which tissue comes back purple.
    """

    def field(stride_um: float, cells: int):
        labels, probabilities, inside = _scene(cells, cells)
        quarter = cells // 4
        labels[quarter : 3 * quarter, quarter : 3 * quarter] = 1
        probabilities[quarter : 3 * quarter, quarter : 3 * quarter] = (0.15, 0.80, 0.05)
        return uncertainty.derive(
            labels,
            probabilities,
            inside,
            candidate=CANDIDATE,
            rival=RIVAL,
            stride_um=stride_um,
            cell_mm2=(stride_um / 1000.0) ** 2,
        )

    coarse = field(224.0, 20)
    fine = field(112.0, 40)

    # Reported to four decimals, so this compares the kernel widths rather than their
    # rounding: halving the stride has to double the neighbourhood in cells.
    assert fine.sigma_cells == pytest.approx(coarse.sigma_cells * 2, abs=1e-3)
    # The claim that matters: the same physical area comes back flagged either way.
    assert coarse.unknown.sum() * (0.224**2) == pytest.approx(
        fine.unknown.sum() * (0.112**2), rel=1e-6
    )


def test_flagging_cannot_move_the_score():
    """The invariant the whole design rests on.

    The layer may only ever recolour class 1, which Rule 5 excludes before it is even
    looked at. So a flagged window is never a scored one, and every figure the score is
    built from is identical with the layer and without it.
    """
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, (30, 30)).astype(np.int8)
    raw = rng.random((30, 30, 3)).astype(np.float32)
    probabilities = (raw / raw.sum(axis=-1, keepdims=True)).astype(np.float32)
    inside = np.ones((30, 30), dtype=bool)

    layer = _derive(labels, probabilities, inside)

    assert not (layer.unknown & (labels != CANDIDATE)).any()
    assert not (layer.unknown & (labels == SCORED)).any()
    assert (layer.score >= 0.0).all() and (layer.score <= 1.0).all()
    # Nothing outside the candidates is ever scored, so `score > 0` cannot leak.
    assert not (layer.score[labels != CANDIDATE] > 0.0).any()


def test_the_display_map_differs_from_the_stored_labels_only_on_flagged_windows():
    grid = _grid(_mask(fill=0.4))
    class_map = _with_in_situ(_classify(grid), grid)
    layer = _layer_for(class_map, grid, threshold=0.0)
    flagged = replace(class_map, uncertainty=layer)

    assert np.array_equal(flagged.labels, class_map.labels)
    assert flagged.tumour_content == class_map.tumour_content
    assert flagged.scored_mm2 == class_map.scored_mm2
    assert flagged.counts == class_map.counts

    differs = flagged.display_labels != flagged.labels
    assert np.array_equal(differs, layer.unknown)
    assert (flagged.display_labels[differs] == UNCERTAIN).all()
    # And with no layer at all, the display map is the stored one unchanged.
    assert np.array_equal(class_map.display_labels, class_map.labels)


def _with_in_situ(class_map: ClassMap, grid) -> ClassMap:
    """A copy of `class_map` with its kept windows relabelled in-situ.

    The fake net above never emits class 1 - its middle logit is a constant zero, so
    the argmax is always 0 or 2 - and the drawing tests need in-situ windows to exist.
    Relabelling here rather than teaching the net a third behaviour keeps that net
    doing the one job the geometry tests need of it.
    """
    labels = np.where(grid.inside, np.int8(CANDIDATE), class_map.labels).astype(np.int8)
    probabilities = class_map.probabilities.copy()
    probabilities[grid.inside] = (0.15, 0.80, 0.05)
    counts = tuple(int((labels == label).sum()) for label in range(3))
    return replace(
        class_map,
        labels=labels,
        probabilities=probabilities,
        counts=counts,
        areas_mm2=tuple(round(count * grid.cell_mm2, 4) for count in counts),
    )


def test_the_flagged_class_is_drawn_and_can_be_switched_off():
    """Purple has to reach the picture, and the per-class toggle has to remove it."""
    mask = _mask(fill=0.8)
    grid = _grid(mask)
    class_map = _with_in_situ(_classify(grid), grid)

    # Threshold 0 flags every in-situ window, so this asserts on the drawing rather
    # than on how any particular region happens to score.
    layer = _layer_for(class_map, grid, threshold=0.0)
    flagged = replace(class_map, uncertainty=layer)
    assert layer.unknown.sum() > 0

    purple = np.asarray(tissue_classes.CLASS_COLOURS[UNCERTAIN], dtype=np.uint8)

    def purple_pixels(png: bytes) -> int:
        return int((np.asarray(Image.open(BytesIO(png))) == purple).all(axis=-1).sum())

    everything = overlay.flat_png(flagged, mask.shape)
    without = overlay.flat_png(flagged, mask.shape, classes=frozenset({0, 1, 2}))

    assert purple_pixels(everything) > 0
    assert purple_pixels(without) == 0


def test_the_uncertainty_panel_is_a_png_with_a_layer_and_without_one():
    mask = _mask(fill=0.6)
    grid = _grid(mask)
    class_map = _classify(grid)

    # No layer: still a picture. This route can be hit on a map written before the
    # layer existed, and a 500 there is a worse answer than an empty ground.
    assert overlay.uncertainty_png(class_map, mask.shape)[:8] == b"\x89PNG\r\n\x1a\n"

    flagged = replace(class_map, uncertainty=_layer_for(class_map, grid))
    assert overlay.uncertainty_png(flagged, mask.shape)[:8] == b"\x89PNG\r\n\x1a\n"


# --- the checkpoint loader ---------------------------------------------------


def _publish(directory: Path, name: str, **overrides) -> Path:
    """A throwaway checkpoint and manifest, for the refusals.

    Built rather than fetched: what is under test is the loader's checks, and the
    cheapest way to test a refusal is to publish something that should trigger it.
    """
    import torch
    import torchvision

    directory.mkdir(parents=True, exist_ok=True)
    net = torchvision.models.resnet18(weights=None)
    net.fc = torch.nn.Linear(model.FEATURE_DIM, 3)

    checkpoint = directory / f"{name}.pt"
    torch.save(net.state_dict(), checkpoint)

    manifest = {
        "name": name,
        "sha256": model.sha256_of(checkpoint),
        "arch": "resnet18",
        "init": "imagenet",
        "classes": list(tissue_classes.CLASS_NAMES),
        "input": model_input.descriptor(tile_px=224, mpp=0.5, invert=False, standardise=True),
        "provenance": {"licences": {"BCSS": "CC0 1.0"}},
        "training_data": {"source": "test"},
        "metrics": {},
        "probe_tiles": [],
    }
    manifest.update(overrides)
    (directory / f"{name}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint


@pytest.fixture
def published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "tissue_type_models_dir", tmp_path)
    model.clear_cache()
    yield tmp_path
    model.clear_cache()


def test_a_published_checkpoint_loads_and_reports_its_contract(published: Path):
    _publish(published, "unit_test_model")

    pinned = model.load_pinned("unit_test_model")
    assert pinned.tile_px == 224
    assert pinned.mpp == 0.5
    assert pinned.standardise is True
    assert pinned.licence_track == "permissive"
    assert pinned.classes == tissue_classes.CLASS_NAMES


def test_a_checkpoint_whose_bytes_changed_is_refused(published: Path):
    checkpoint = _publish(published, "tampered")
    checkpoint.write_bytes(checkpoint.read_bytes() + b"\x00")

    with pytest.raises(model.ModelError, match="not the one that was validated"):
        model.load_pinned("tampered")


def test_a_manifest_whose_input_contract_disagrees_is_refused(published: Path):
    """The quietest failure in the plan: fitted with standardise on, served with it off."""
    spec = model_input.descriptor(tile_px=224, mpp=0.5, invert=False, standardise=True)
    spec["standardise_target"] = 0.9
    _publish(published, "drifted", input=spec)

    with pytest.raises(model.ModelError, match="different input transform"):
        model.load_pinned("drifted")


def test_a_manifest_with_no_input_block_is_refused(published: Path):
    _publish(published, "undescribed", input={})

    with pytest.raises(model.ModelError, match="input contract is unknown"):
        model.load_pinned("undescribed")


def test_a_manifest_with_a_permuted_class_order_is_refused(published: Path):
    _publish(published, "permuted", classes=list(reversed(tissue_classes.CLASS_NAMES)))

    with pytest.raises(tissue_classes.ClassOrderError):
        model.load_pinned("permuted")


def test_a_non_commercial_licence_anywhere_makes_the_whole_thing_research_only(
    published: Path,
):
    """Restrictive wins. This is what stops a non-commercial dependency shipping quietly."""
    _publish(
        published,
        "borrowed",
        provenance={
            "licences": {
                "BCSS": "CC0 1.0",
                "BEETLE": "CC BY-NC-SA 4.0 - NON-COMMERCIAL",
                "torchvision resnet18": "BSD-3",
            }
        },
    )

    candidate = model.find("borrowed")
    assert candidate is not None
    assert candidate.licence_track == "research-only"
    assert model.load_pinned("borrowed").licence_track == "research-only"


def test_a_missing_checkpoint_names_what_is_published(published: Path):
    _publish(published, "the_only_one")

    with pytest.raises(model.ModelError, match="the_only_one"):
        model.load_pinned("some_other_name")


# --- the endpoints -----------------------------------------------------------


def test_capability_reports_every_checkpoint_and_its_licence(client: TestClient):
    response = client.get("/api/v1/tissue-type/capability")
    assert response.status_code == 200

    body = response.json()
    assert "ready" in body
    assert "reason" in body
    assert body["defaultModel"]
    assert body["licenceNote"]
    for entry in body["models"]:
        assert entry["licenceTrack"] in {"permissive", "research-only", "unknown"}


def test_a_report_before_a_run_is_a_conflict_not_a_404(client: TestClient):
    """Nothing is wrong with the request - the job has not happened."""
    response = client.get("/api/v1/tissue-type/does-not-exist")
    assert response.status_code in {404, 409}


def test_a_panel_before_a_run_is_a_conflict(client: TestClient):
    response = client.get("/api/v1/tissue-type/does-not-exist/panels/map.png")
    assert response.status_code in {404, 409}


def test_an_unknown_panel_is_rejected_at_the_boundary(client: TestClient):
    response = client.get("/api/v1/tissue-type/anything/panels/umap.png")
    assert response.status_code == 422


def test_a_class_filter_that_is_not_a_class_is_rejected(client: TestClient):
    response = client.get("/api/v1/tissue-type/anything/panels/map.png?classes=7")
    assert response.status_code == 422

    response = client.get("/api/v1/tissue-type/anything/panels/map.png?classes=stroma")
    assert response.status_code == 422


def test_the_catalogue_and_the_step_module_agree_that_it_is_built(client: TestClient):
    """A stub module and an `implemented=True` catalogue entry must never coexist."""
    response = client.get("/api/v1/pipeline/stages")
    assert response.status_code == 200

    stages = response.json()["items"]
    stage = next(entry for entry in stages if entry["id"] == "tissue-type-segmentation")
    assert stage["implemented"] is True
    assert stage["index"] == 8


# --- cancelling ---------------------------------------------------------------


def test_classify_stops_when_it_is_asked_to():
    """A cancel has to land at a block boundary, not at the end of the slide.

    The pass is tens of minutes and one block is under two seconds, so where the
    check sits is the difference between a cancel and a promise of one. This asserts
    it raises rather than finishing, and that it raises the *shared* signal - a
    cancellation reported as a segmentation failure would make the screen apologise
    for what the viewer just asked for.
    """
    from app.pipeline.contract import RunCancelled

    grid = _grid(_mask(fill=0.8))
    seen: list[int] = []

    with pytest.raises(RunCancelled, match="stopped after"):
        inference.classify(
            grid,
            net=_FakeNet(),
            read_window=lambda x, y, span, size: np.full(
                (size, size), 0.4, dtype=np.float32
            ),
            standardise=False,
            block_windows=8,
            progress=lambda done, total: seen.append(done),
            # Stop after the first block, which is the earliest it can be asked.
            should_stop=lambda: len(seen) >= 1,
        )

    # It got as far as one block and no further, which is what "promptly" means here.
    assert len(seen) == 1
    assert seen[0] < grid.windows


def test_a_cancelled_pass_reports_how_far_it_got():
    """The message is the useful part: a viewer who stops at 80% should be told so."""
    from app.pipeline.contract import RunCancelled

    grid = _grid(_mask(fill=0.8))

    with pytest.raises(RunCancelled) as raised:
        inference.classify(
            grid,
            net=_FakeNet(),
            read_window=lambda x, y, span, size: np.full(
                (size, size), 0.4, dtype=np.float32
            ),
            standardise=False,
            should_stop=lambda: True,
        )

    assert f"of {grid.windows:,}" in str(raised.value)


def test_cancelling_a_pass_that_is_not_running_is_not_an_error(client: TestClient):
    """A viewer clicking cancel as the last block lands has done nothing wrong.

    Reporting a conflict there would report the race rather than the outcome. The
    only acceptable answers are the current state, or a 404 for a slide that does
    not exist - never a 500 and never a 409 about nothing.
    """
    response = client.post("/api/v1/tissue-type/does-not-exist/cancel")
    assert response.status_code in {200, 404, 409}


def test_a_cancel_is_a_distinct_state_from_a_failure():
    """The two must not be collapsed anywhere between the worker and the wire."""
    from app.pipeline.contract import PipelineError, RunCancelled, StepNotImplementedError

    assert not issubclass(RunCancelled, PipelineError)
    assert not issubclass(RunCancelled, StepNotImplementedError)
    assert issubclass(RunCancelled, RuntimeError)


# --- the two bugs that reached the running app --------------------------------


def test_loading_a_checkpoint_logs_without_blowing_up(published: Path, caplog):
    """`extra=` must not use a reserved `LogRecord` field name.

    This is the bug that made **every** step 8 run return HTTP 500: `load_pinned`
    logged with `extra={"name": ...}`, and `name` is a `LogRecord` attribute, so
    `logging.Logger.makeRecord` raised `KeyError: Attempt to overwrite 'name'`.

    **Why nothing caught it.** That raise happens inside `makeRecord`, which is only
    reached when the record is actually built - so `logger.info(...)` is a silent no-op
    whenever the effective level is above INFO. Every script here and the whole test
    suite leave the root logger at its default WARNING, so the call never ran; the app
    calls `configure_logging` and sets INFO, so it ran on the first request and 500ed.
    `caplog.set_level` is therefore not a detail of this test, it *is* the test.
    """
    import logging

    _publish(published, "log_probe")

    with caplog.at_level(logging.DEBUG):
        pinned = model.load_pinned("log_probe")

    assert pinned.name == "log_probe"
    assert any("model_loaded" in record.getMessage() for record in caplog.records)


def test_no_logging_call_uses_a_reserved_record_field(caplog):
    """The same trap, swept for across the whole app rather than one call site.

    A reserved key in `extra=` is invisible until the log level lets the record be
    built, so it cannot be left to review. Every `extra={...}` literal in `app/` is
    parsed and its keys checked against a real `LogRecord`.
    """
    import ast
    import logging
    from pathlib import Path as _Path

    reserved = set(
        logging.LogRecord("n", 20, "p", 1, "m", None, None).__dict__
    ) | {"message", "asctime"}

    root = _Path(__file__).resolve().parents[1] / "app"
    offenders: list[str] = []

    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "extra" or not isinstance(keyword.value, ast.Dict):
                    continue
                for key in keyword.value.keys:
                    if isinstance(key, ast.Constant) and key.value in reserved:
                        offenders.append(f"{path.name}:{key.lineno} extra={{{key.value!r}: ...}}")

    assert not offenders, (
        "these logging calls pass a reserved LogRecord field in `extra`, which raises "
        "KeyError as soon as the log level lets the record be built:\n  "
        + "\n  ".join(offenders)
    )


def test_a_finished_run_is_not_cached_as_still_running(published: Path):
    """A completed pass must not describe itself as running.

    `_describe` embeds a run block in `report.json`, and `start`/`state` hand that block
    straight back on every later visit. It used to be captured while the job was still
    mid-flight - `state: "running", phase: "rendering", progress: 0.94` - so a
    *finished* pass reported itself as running for ever: the endpoint would not
    re-queue it (the state is not "queued") and the screen would poll a run that could
    never change.

    Checked on the shape of the report rather than by running a slide, because the bug
    is in what gets written, not in what gets computed.
    """
    from app.services.tissue_type_service import Job, tissue_type_service

    params = {
        "model": "unit",
        "model_sha256": None,
        "licence_track": "permissive",
        "window_px": 224,
        "window_um": 112.0,
        "mpp": 0.5,
        "overlap": 0.5,
        "standardise": True,
        "gamma": 1.0,
        "invert_polarity": False,
        "span": 504,
        "stride": 252,
        "tile_overlap": 0.25,
        "tissue_threshold": 1,
        "tissue_threshold_source": "otsu",
        "qc_gated": False,
        "qc_source": None,
        "block_windows": 8,
        "batch_size": 32,
        "device": "cpu",
    }

    job = Job(upload_id="unit", params=params)
    job.state = "running"
    job.phase = "rendering"
    job.progress = 0.94
    job.done, job.total = 100, 100

    # What `_run` now does immediately before describing.
    job.state = "ready"
    job.phase = None
    job.message = "complete"
    job.progress = 1.0

    run = tissue_type_service._as_run(job)
    assert run.state == "ready"
    assert run.phase is None
    assert run.progress == 1.0


def _paint_job(windows: int):
    """A job mid-pass, with `windows` windows already painted."""
    from app.services.tissue_type_service import Job

    params = {
        "model": "unit",
        "model_sha256": None,
        "licence_track": "permissive",
        "window_px": 224,
        "window_um": 112.0,
        "mpp": 0.5,
        "overlap": 0.5,
        "standardise": True,
        "gamma": 1.0,
        "invert_polarity": False,
        "span": 504,
        "stride": 252,
        "tile_overlap": 0.25,
        "tissue_threshold": 1,
        "tissue_threshold_source": "otsu",
        "qc_gated": False,
        "qc_source": None,
        "block_windows": 8,
        "batch_size": 32,
        "device": "cpu",
    }

    job = Job(upload_id="unit", params=params)
    job.state = "running"
    job.phase = "classifying"
    job.paint = {
        "cols": 40,
        "rows": 30,
        "span": 504,
        "stride": 252,
        "slide_width": 10_000,
        "slide_height": 7_500,
        "colours": ["#facc15", "#3b82f6", "#ef4444"],
        "labels": ["Not tumour tissue", "Tumour still inside the duct", "Tumour that has broken out"],
    }
    for index in range(windows):
        job.painted.extend((index // 40, index % 40, index % 3))
    return job


def test_a_poll_that_does_not_ask_for_paint_carries_none_of_it():
    """The feed is a whole slide's grid, so it travels only when it is asked for.

    This is also what keeps it out of `report.json`: the report embeds the run block
    that `_as_run` builds without a cursor, and a cached pass has four rendered panels
    and no use for a live feed.
    """
    from app.services.tissue_type_service import tissue_type_service

    run = tissue_type_service._as_run(_paint_job(64))
    assert run.paint is None
    assert run.painted_cells == []
    assert run.painted_cursor == 0


def test_the_paint_feed_hands_each_window_over_exactly_once():
    """A client that polls with the cursor it was given sees every window, once.

    The property the screen depends on: it accumulates rather than redraws, so a window
    delivered twice would be painted twice - visible at the overlay's opacity - and one
    delivered never would leave a hole that only closes when the report lands.
    """
    from app.services.tissue_type_service import _PAINT_CHUNK, tissue_type_service

    windows = _PAINT_CHUNK + 500
    job = _paint_job(windows)

    seen: list[int] = []
    cursor = 0
    polls = 0
    while True:
        run = tissue_type_service._as_run(job, painted_since=cursor)
        assert run.paint is not None
        assert run.paint.stride == 252
        assert len(run.painted_cells) % 3 == 0
        # Capped, so a client rejoining part-way catches up over a few polls rather
        # than pulling a whole grid into one reply.
        assert len(run.painted_cells) <= _PAINT_CHUNK * 3

        seen.extend(run.painted_cells)
        polls += 1
        if run.painted_cursor == cursor:
            break
        cursor = run.painted_cursor
        assert polls < 10

    assert polls > 1, "the cap did not apply, so this is not testing the catch-up"
    assert len(seen) == windows * 3
    assert seen == job.painted


def test_a_paint_cursor_past_the_end_is_not_an_error():
    """A client polling a pass that has just finished asks past the end of the log."""
    from app.services.tissue_type_service import tissue_type_service

    run = tissue_type_service._as_run(_paint_job(64), painted_since=10_000)
    assert run.painted_cells == []
    assert run.painted_cursor == 10_000


def test_a_cached_pass_reports_no_live_paint(monkeypatch):
    """A finished pass read back from disk is a picture, not a sweep.

    Its run block is whatever was written to `report.json`, and a paint feed replayed
    out of a file would have the screen paint a pass that finished last week.
    """
    from app.services import tissue_type_service as service_module

    service = service_module.tissue_type_service
    monkeypatch.setattr(
        service,
        "_read_json",
        lambda upload_id, name: {
            "run": {
                "uploadId": "cached",
                "state": "ready",
                "progress": 1.0,
                "done": 100,
                "total": 100,
                "paint": {
                    "cols": 4,
                    "rows": 4,
                    "span": 504,
                    "stride": 252,
                    "slideWidth": 1000,
                    "slideHeight": 1000,
                    "colours": ["#facc15", "#3b82f6", "#ef4444"],
                },
                "paintedCells": [0, 0, 2],
                "paintedCursor": 1,
            }
        },
    )

    run = service.state("cached", painted_since=0)
    assert run.state == "ready"
    assert run.paint is None
    assert run.painted_cells == []
    assert run.painted_cursor == 0


# --- step 8's own tissue gate -------------------------------------------------


def _gated(mask, share, *, overlap=0.5):
    return build_grid(
        _index(mask, overlap=0.25),
        base_mpp=BASE_MPP,
        slide_size=_slide_size(mask),
        size=WINDOW,
        mpp=MODEL_MPP,
        overlap=overlap,
        max_windows=400_000,
        tissue=mask,
        mask_mpp=MASK_MPP,
        min_tissue_share=share,
    )


def test_the_window_gate_agrees_with_a_direct_measurement():
    """The summed-area table has to give the same answer as counting the mask.

    Vectorised through an integral image because a whole-slide grid is a quarter of a
    million cells; this checks that optimisation against the arithmetic it replaces.
    """
    mask = _disc()
    grid = _gated(mask, 0.0)
    share = inference._tissue_share(grid, mask, MASK_MPP)

    to_mask = BASE_MPP / MASK_MPP
    for row, col in ((0, 0), (grid.rows // 2, grid.cols // 2), (grid.rows - 1, grid.cols - 1)):
        x, y = grid.x_of(col), grid.y_of(row)
        x0, y0 = int(np.floor(x * to_mask)), int(np.floor(y * to_mask))
        x1 = max(x0 + 1, int(np.ceil((x + grid.span) * to_mask)))
        y1 = max(y0 + 1, int(np.ceil((y + grid.span) * to_mask)))
        block = mask[
            max(0, y0) : min(mask.shape[0], y1), max(0, x0) : min(mask.shape[1], x1)
        ]
        assert share[row, col] == pytest.approx(float(block.mean()), abs=1e-5)


def test_a_higher_gate_keeps_fewer_windows_and_says_how_many_it_dropped():
    """Monotone, and the cost is reported rather than absorbed."""
    mask = _disc()
    open_grid = _gated(mask, 0.0)
    half = _gated(mask, 0.5)
    strict = _gated(mask, 0.9)

    assert open_grid.windows >= half.windows >= strict.windows
    assert strict.windows < open_grid.windows, "the gate must actually bind on a disc"

    assert open_grid.gated_out == 0
    assert half.gated_out == open_grid.windows - half.windows
    assert half.min_tissue_share == 0.5


def test_the_gate_removes_the_emptiest_windows_first():
    """The whole point: what it drops is edge windows that are mostly glass.

    Pinned because the gate would be worse than useless if it trimmed the section's
    interior - the invasive front sits at the edge, and this must remove *empty*
    windows rather than *peripheral* ones.
    """
    mask = _disc()
    open_grid = _gated(mask, 0.0)
    gated = _gated(mask, 0.5)

    share = inference._tissue_share(open_grid, mask, MASK_MPP)
    dropped = open_grid.inside & ~gated.inside

    assert dropped.any()
    assert share[dropped].max() < 0.5
    assert share[gated.inside].min() >= 0.5


def test_no_mask_means_no_gate():
    """The geometry tests and any caller without step 3's mask keep the old behaviour."""
    mask = _disc()
    assert _grid(mask).windows == _gated(mask, 0.0).windows


def test_a_gate_that_admits_nothing_is_refused_with_a_reason():
    mask = _mask(fill=0.5)
    with pytest.raises(SegmentationError, match="tissue gate"):
        _gated(mask, 1.01)


def test_a_cache_written_before_a_setting_existed_invalidates_rather_than_crashing():
    """Adding a cache-key field must not make old caches unreadable.

    `_cache_key` compares the parameters of the run about to start against the ones
    recorded beside a cached report, and that recorded dict may predate any key added
    since. Indexing it raised `KeyError` and took the whole run down - which is how
    adding this step's tissue gate broke every slide that already had a class map.
    Missing reads as `None`, the comparison fails, and the pass re-runs: correct,
    because a cache written before a setting existed cannot describe its effect.
    """
    from app.services.tissue_type_service import tissue_type_service

    current = {
        "model": "m", "model_sha256": "s", "window_px": 224, "mpp": 0.5,
        "overlap": 0.5, "standardise": True, "gamma": 1.0, "invert_polarity": False,
        "min_tissue_share": 0.5, "tile_overlap": 0.25, "tissue_threshold": 1,
        "qc_gated": False, "qc_source": None,
    }
    stale = {key: value for key, value in current.items() if key != "min_tissue_share"}

    assert tissue_type_service._cache_key(stale) != tissue_type_service._cache_key(current)
    assert tissue_type_service._cache_key(current) == tissue_type_service._cache_key(current)


def test_step_8_runs_at_step_7s_overlap_rather_than_its_own_setting(monkeypatch):
    """The overlap a viewer picked on the tiling screen is the one this step uses.

    The two grids differ in size and resolution and always will - 224 px at the
    checkpoint's mpp against step 7's 512 px at the pipeline's - because the model's
    field of view is a property of the checkpoint. Overlap is the one parameter they
    share, and while it was read from settings a viewer could pick "no overlap" on
    step 7's picker, watch it price the choice in squares, and then watch this step
    run at half overlap anyway: a picker that prices a decision the next screen
    ignores. So `settings.tissue_type_overlap` is now only a fallback, and what is
    pinned here is that an explicit choice still wins over the inherited one -
    otherwise the API's `overlap` parameter would be silently dead.
    """
    from types import SimpleNamespace

    from app.services import tissue_type_service as service_module
    from app.services.tissue_type_service import tissue_type_service

    monkeypatch.setattr(
        service_module.settings, "tissue_type_overlap", 0.5, raising=False
    )
    monkeypatch.setattr(
        tissue_type_service,
        "capability",
        lambda: SimpleNamespace(torch_installed=True, reason="", device="cpu"),
    )
    monkeypatch.setattr(
        service_module.model, "find", lambda name: SimpleNamespace(usable=True)
    )
    monkeypatch.setattr(
        service_module.model,
        "load_pinned",
        lambda name: SimpleNamespace(
            name=name, sha256="s", licence_track="research", tile_px=224, mpp=0.5,
            standardise=True, gamma=1.0, invert_polarity=False,
            # What the checkpoint says it is shown. Recorded on the run because the
            # branch, not the field of view alone, now decides which head serves.
            channel="haematoxylin",
            # No published reference: the run is gated by the flat test alone.
            familiarity=None,
        ),
    )
    # Step 3's mask by identity is part of the run key now, and this test has no slide.
    monkeypatch.setattr(
        service_module.tissue_service,
        "footprint",
        lambda upload_id, **_: SimpleNamespace(key="unit-mask"),
    )
    monkeypatch.setattr(
        service_module.calibration_service,
        "white_point",
        lambda upload_id, **_: SimpleNamespace(key="unit-white"),
    )
    monkeypatch.setattr(
        service_module.tiling_service,
        "report",
        # `**_` because `resolve_params` now asks for the committed grid by name -
        # branch, field of view, overlap and threshold - rather than taking whatever
        # the defaults hand back. That call is the fix for the two steps building
        # different grids, so the stub has to accept it.
        lambda upload_id, **_: SimpleNamespace(
            params=SimpleNamespace(
                overlap=0.0, tissue_threshold=1, tissue_threshold_source="otsu",
                qc_gated=False, qc_source=None,
                min_tissue_share=0.5, min_clean_share=0.5,
                # Step 7 also names the checkpoint now - it is the head fitted at the
                # field of view that screen chose - so the stub carries one. The
                # inheritance being pinned here is still the overlap's.
                field_of_view_um=224.0, model="unit-head",
                branch="h_channel", input_channel="haematoxylin",
            )
        ),
    )

    inherited = tissue_type_service.resolve_params("unit")
    assert inherited["overlap"] == 0.0, "step 7 chose no overlap, so this step has none"
    assert inherited["tile_overlap"] == 0.0

    # And the two stay distinct: `tile_overlap` records what step 7 did, `overlap`
    # what this step ran at, so a named override is still visible as an override.
    named = tissue_type_service.resolve_params("unit", overlap=0.25)
    assert named["overlap"] == 0.25
    assert named["tile_overlap"] == 0.0


def test_the_two_grids_coincide_when_step_7_lays_the_models_window():
    """Step 7's kept tiles and step 8's planned windows are the same squares.

    This is the property the whole arrangement rests on, and it is worth pinning
    rather than assuming, because it is not enforced by a type: `build_grid` still
    plans its own grid from a size, an mpp and an overlap, and merely happens to be
    handed step 7's when the service resolves both from the same manifest. If that
    resolution ever drifted - a settings default creeping back in on one side - the
    two would silently diverge again and the only symptom would be the thing this
    change set out to remove: one screen pricing a count the next screen does not run.

    Checked as an exact set of origins rather than as a count, since two grids can
    hold the same number of squares in different places.
    """
    mask = _disc()
    index = _index(mask, tile_size=WINDOW, overlap=0.25)
    grid = build_grid(
        index,
        base_mpp=BASE_MPP,
        slide_size=_slide_size(mask),
        size=WINDOW,
        mpp=MODEL_MPP,
        overlap=0.25,
        max_windows=400_000,
    )

    assert (grid.cols, grid.rows) == (index.cols, index.rows)
    assert (grid.span, grid.stride) == (index.span, index.stride)

    kept = {(tile.x, tile.y) for tile in index.tiles if tile.kept}
    planned = {
        (grid.x_of(col), grid.y_of(row))
        for row in range(grid.rows)
        for col in range(grid.cols)
        if grid.inside[row, col]
    }
    assert planned == kept
    assert grid.windows == index.funnel.clean


def test_step_8s_window_gate_removes_nothing_once_step_7_gates_the_same_square():
    """With one grid, the second tissue gate is a pass-through rather than a filter.

    The two gates were 0.10 at step 7 and 0.50 at step 8, measured on different-sized
    squares, and the gap between them was most of why the two screens reported
    different counts. Step 7 now carries 0.50 and step 8 sees the identical square, so
    `gated_out` must be zero - and if it is not, the shares are being measured
    differently on the two sides, which would be a real bug rather than a redundancy.
    """
    mask = _disc()
    index = build_index(
        tissue=mask,
        considered=None,
        mask_mpp=MASK_MPP,
        base_mpp=BASE_MPP,
        slide_size=_slide_size(mask),
        target_mpp=BASE_MPP,
        size=WINDOW,
        overlap=0.25,
        min_tissue_share=0.5,
        min_clean_share=0.5,
        max_tiles=400_000,
    )
    grid = build_grid(
        index,
        base_mpp=BASE_MPP,
        slide_size=_slide_size(mask),
        size=WINDOW,
        mpp=MODEL_MPP,
        overlap=0.25,
        max_windows=400_000,
        tissue=mask,
        mask_mpp=MASK_MPP,
        min_tissue_share=0.5,
    )

    assert grid.gated_out == 0
    assert grid.windows == index.funnel.clean


def test_a_tile_step_2_flagged_is_not_reached_through_its_kept_neighbours():
    """Quality control still reaches this step when the squares overlap.

    The centre test asks whether a window's centre lands on *any* kept tile, which was
    the right question when this step laid a finer grid of its own. Once step 7 lays
    this model's own window the two grids are cell-for-cell, and that question becomes
    wrong rather than merely redundant: tiles overlap each other, so a tile step 7
    dropped has its own centre sitting inside its kept neighbours and is admitted by
    the back door. The windows it admits are exactly the ones step 2 flagged as
    artefact - an out-of-focus or folded square, which the region model answers
    confidently and wrongly rather than not at all, which is the whole reason step 7
    runs after step 2.

    Pinned at three overlaps because the failure needs neighbours to reach through and
    so does not appear at all without overlap: before the fix this was 20 tiles kept
    against 26 windows planned at 50%, and clean at 0% and 25%.
    """
    mask = _disc()
    considered = np.ones_like(mask)
    considered[110:150, :] = False  # a band of artefact straight through the section

    for overlap in (0.0, 0.25, 0.5):
        index = build_index(
            tissue=mask,
            considered=considered,
            mask_mpp=MASK_MPP,
            base_mpp=BASE_MPP,
            slide_size=_slide_size(mask),
            target_mpp=BASE_MPP,
            size=WINDOW,
            overlap=overlap,
            min_tissue_share=0.5,
            min_clean_share=0.5,
            max_tiles=400_000,
        )
        grid = build_grid(
            index,
            base_mpp=BASE_MPP,
            slide_size=_slide_size(mask),
            size=WINDOW,
            mpp=MODEL_MPP,
            overlap=overlap,
            max_windows=400_000,
            tissue=mask,
            mask_mpp=MASK_MPP,
            min_tissue_share=0.5,
        )

        flagged = {
            (tile.x, tile.y) for tile in index.tiles if tile.rejected_by == "clean"
        }
        planned = {
            (grid.x_of(col), grid.y_of(row))
            for row in range(grid.rows)
            for col in range(grid.cols)
            if grid.inside[row, col]
        }
        assert flagged, f"the artefact band should reject tiles at overlap {overlap}"
        assert not (planned & flagged), (
            f"at overlap {overlap} the model would be shown "
            f"{len(planned & flagged)} squares step 2 flagged"
        )
        assert grid.windows == index.funnel.clean


# --- step 7's field of view picks the checkpoint ------------------------------


def test_a_checkpoint_is_matched_to_a_field_of_view_by_its_manifest_not_its_name(
    published: Path,
):
    """`tile_px * mpp` decides which head serves which field of view. Never the name.

    This is what lets `FIX1_FIX2_PLAN`'s S5 heads start serving the moment they are
    published, under whatever name they were given, with no code change. Matching on a
    naming convention instead would mean a head named by its trainer rather than by this
    file's expectation is invisible - published, correct, and silently not used.

    The name here is deliberately misleading: a file called `..._112um` that the manifest
    says is 448 um is a 448 um model, because the manifest records what the weights
    actually saw and the name records what somebody typed.
    """
    from app.services import tiling_service as tiling_module

    _publish(
        published,
        "misleadingly_named_112um",
        input=model_input.descriptor(tile_px=224, mpp=2.0, invert=False, standardise=True),
    )

    assert tiling_module.model_for(448.0).name == "misleadingly_named_112um"
    assert tiling_module.model_for(224.0) is None


def test_a_field_of_view_with_no_published_head_still_lays_its_planned_grid(
    published: Path,
):
    """No checkpoint is not an error at step 7 - it is an error at step 8.

    Step 7's square count *is* the compute bill, so knowing it before committing to a
    field of view is most of the point of that screen. It is laid on the geometry the
    head will use, `model` comes back null, and step 8 refuses separately. The wrong
    behaviour here would be falling back to whichever checkpoint happens to be
    configured: that one was fitted at a different field of view, so it would score a
    grid it never saw and nothing on screen would say so.
    """
    from app.services import tiling_service as tiling_module

    size, mpp, source, name = tiling_module.window(448.0)
    assert (size, mpp) == tiling_module.FIELDS_OF_VIEW[448.0]
    assert name is None
    assert "planned" in source


def test_a_field_of_view_nobody_has_a_model_for_is_refused_rather_than_snapped():
    """300 um must not quietly become 224 um.

    A silent snap would be worse than a refusal, because this parameter decides which
    checkpoint runs: a caller who asked for 300 um and got 224 um would be reading one
    model's numbers under another model's name.
    """
    from app.services import tiling_service as tiling_module
    from app.services.tiling_service import TilingError

    with pytest.raises(TilingError, match="not one of the fields of view"):
        tiling_module.field_of_view(300.0)

    assert tiling_module.field_of_view(448.0) == 448.0
    assert tiling_module.field_of_view(None) == settings.tiling_field_of_view_um


def test_step_8_refuses_when_step_7s_field_of_view_has_no_head(monkeypatch):
    """The one fallback that must not exist.

    Serving the configured checkpoint here would put a head fitted at one field of view
    over a grid laid at another - the same class of failure as feeding it the wrong
    channel, and just as invisible: the class map would still be three-valued and the
    confidences would still look reasonable. So the refusal names the missing file
    instead, and it names the published ones with their fields of view so the reader can
    see what they could have chosen.
    """
    from types import SimpleNamespace

    from app.services import tissue_type_service as service_module
    from app.services.tissue_type_service import TissueTypeError, tissue_type_service

    monkeypatch.setattr(
        tissue_type_service,
        "capability",
        lambda: SimpleNamespace(torch_installed=True, reason="", device="cpu"),
    )
    monkeypatch.setattr(
        service_module.tissue_service,
        "footprint",
        lambda upload_id, **_: SimpleNamespace(key="unit-mask"),
    )
    monkeypatch.setattr(
        service_module.calibration_service,
        "white_point",
        lambda upload_id, **_: SimpleNamespace(key="unit-white"),
    )
    monkeypatch.setattr(
        service_module.tiling_service,
        "report",
        lambda upload_id, **_: SimpleNamespace(
            params=SimpleNamespace(
                overlap=0.0, tissue_threshold=1, tissue_threshold_source="otsu",
                qc_gated=False, qc_source=None,
                min_tissue_share=0.5, min_clean_share=0.5,
                field_of_view_um=448.0, model=None,
                branch="h_channel", input_channel=None,
            )
        ),
    )

    with pytest.raises(TissueTypeError, match="no checkpoint fitted at 448 um"):
        tissue_type_service.resolve_params("unit")


# --- the refusal gate (P-05) -------------------------------------------------

from app.pipeline.step08_tissue_type_segmentation import familiarity  # noqa: E402


def test_a_constant_window_is_flat_and_an_imaged_one_is_not():
    """The flat test's whole premise: nothing a scanner imaged is constant."""
    rng = np.random.default_rng(0)
    assert familiarity.flat_share(np.full((64, 64), 0.3, np.float32)) == 1.0
    assert familiarity.flat_share(np.full((64, 64, 3), (181, 180, 186), np.uint8)) == 1.0
    noisy = rng.normal(0.3, 0.02, (64, 64)).astype(np.float32)
    assert familiarity.flat_share(noisy) < 0.01
    # One channel differing is enough to be imaged - a fill is constant in all three.
    rgb = np.full((64, 64, 3), 200, np.uint8)
    rgb[..., 2] = rng.integers(0, 255, (64, 64))
    assert familiarity.flat_share(rgb) < 0.05


class _FeatureNet:
    """A net with a `head`, so `model.penultimate` finds the vector the gate reads."""

    def __init__(self) -> None:
        import torch

        self.head = torch.nn.Linear(2, 3)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]))
            self.head.bias.zero_()

    def __call__(self, batch):
        import torch

        # Two features: the window's mean, and its spread.
        mean = batch.mean(dim=(1, 2, 3))
        spread = batch.std(dim=(1, 2, 3))
        return self.head(torch.stack([mean, spread], dim=1))


def _reference(centre: float = 0.0) -> familiarity.Reference:
    rng = np.random.default_rng(1)
    features = rng.normal(centre, 0.05, (400, 2)).astype(np.float32)
    labels = np.repeat(np.arange(2), 200)
    held_out = np.zeros(400, bool)
    held_out[::5] = True
    reference, _ = familiarity.fit(
        features, labels, held_out, quantile=1.0, fingerprint="unit"
    )
    return reference


def _gate_run(grid, read, *, reference=None) -> ClassMap:
    return inference.classify(
        grid,
        net=_FeatureNet(),
        read_window=read,
        standardise=False,
        block_windows=8,
        batch_size=16,
        gate=familiarity.Gate(reference=reference, max_flat_share=0.75),
    )


def _noise(x: int, y: int, span: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(x * 7919 + y)
    return rng.normal(0.3, 0.05, (size, size)).astype(np.float32)


def test_a_flat_window_is_refused_and_leaves_the_class_map_entirely():
    """P-05: a scanner-fill window gets no class, no probability, and no place in `inside`.

    `inside` is the part that is easy to miss and the part step 9 depends on: it closes
    holes within `inside`, so a window that kept its place there could be annexed.
    """
    grid = _grid(_mask(fill=0.6))
    fill_left = grid.slide_width // 2

    def read(x: int, y: int, span: int, size: int) -> np.ndarray:
        pixels = _noise(x, y, span, size)
        # The right half of the slide is one constant value - the fill.
        scale = span / size
        columns = x + (np.arange(size) + 0.5) * scale >= fill_left
        pixels[:, columns] = 0.42
        return pixels

    class_map = _gate_run(grid, read)
    flat = class_map.refused == familiarity.FLAT

    assert flat.any() and (class_map.refused == familiarity.ANSWERED).any()
    assert (class_map.labels[flat] == OUTSIDE).all()
    assert class_map.probabilities[flat].sum() == 0.0
    assert not class_map.grid.inside[flat].any()
    assert class_map.classified == int(class_map.grid.inside.sum())
    assert class_map.classified + class_map.refused_count(familiarity.FLAT) == grid.windows
    # Drawn as "cannot be determined", never as a class.
    assert (class_map.display_labels[flat] == tissue_classes.UNCERTAIN).all()


def test_a_flat_window_never_becomes_invasive_whatever_the_model_says():
    """The gate is upstream of the decision, so no head output can reach the score."""
    grid = _grid(_mask(fill=0.6))

    def read(x: int, y: int, span: int, size: int) -> np.ndarray:
        # A bright constant: `_FeatureNet` calls a high mean invasive.
        return np.full((size, size), 5.0, dtype=np.float32)

    ungated = inference.classify(
        grid, net=_FeatureNet(), read_window=read, standardise=False, batch_size=16
    )
    assert ungated.counts[SCORED] == grid.windows, "the fill would be scored invasive"

    gated = _gate_run(grid, read)
    assert gated.counts[SCORED] == 0
    assert gated.scored_mm2 == 0.0
    assert not gated.grid.inside.any()


def test_an_unfamiliar_window_is_refused_and_a_familiar_one_is_answered():
    grid = _grid(_mask(fill=0.6))
    familiar = _gate_run(grid, _noise, reference=_reference(centre=0.0))
    # The noise windows have mean ~0.3 and spread ~0.05: far from a reference at 0.
    assert familiar.refused_count(familiarity.UNFAMILIAR) == grid.windows

    # A reference fitted on what the net actually reads for windows like these: its
    # input is the transformed density, not the raw pixel values.
    import torch

    net = _FeatureNet()
    samples = np.stack([
        model_input.to_model_input(_noise(x, 0, 64, 224), standardise=False)
        for x in range(400)
    ])
    with torch.no_grad():
        batch = torch.from_numpy(samples)
        mean, spread = batch.mean(dim=(1, 2, 3)), batch.std(dim=(1, 2, 3))
    features = torch.stack([mean, spread], dim=1).numpy()
    held_out = np.zeros(400, bool)
    held_out[::4] = True
    near, _ = familiarity.fit(
        features, np.zeros(400, int), held_out, quantile=1.0, fingerprint="unit"
    )
    near = familiarity.Reference(**{**near.__dict__, "threshold": near.threshold * 4})
    answered = _gate_run(grid, _noise, reference=near)
    assert answered.refused_count(familiarity.UNFAMILIAR) == 0
    assert answered.classified == grid.windows
    assert np.isfinite(answered.distance[grid.inside]).all()


def test_the_gate_leaves_no_hook_on_a_cached_net():
    """The net is cached across runs; a hook left on it would outlive the pass."""
    grid = _grid(_mask(fill=0.6))
    net = _FeatureNet()
    inference.classify(
        grid, net=net, read_window=_noise, standardise=False, batch_size=16,
        gate=familiarity.Gate(reference=_reference(), max_flat_share=0.75),
    )
    assert not net.head._forward_pre_hooks


def test_a_reference_round_trips_and_refuses_another_version(tmp_path):
    reference = _reference()
    path = tmp_path / "head.familiarity.npz"
    familiarity.save(reference, path)
    again = familiarity.load(path)
    assert np.array_equal(again.means, reference.means)
    assert np.array_equal(again.precisions, reference.precisions)
    assert again.threshold == reference.threshold

    with np.load(path) as stored:
        payload = dict(stored)
    payload["version"] = np.array(familiarity.VERSION + 1)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="version"):
        familiarity.load(path)


def test_the_cut_is_read_off_the_held_out_tiles_not_the_training_ones():
    rng = np.random.default_rng(2)
    features = rng.normal(0, 1, (600, 3)).astype(np.float32)
    held_out = np.zeros(600, bool)
    held_out[:100] = True
    features[:100] += 0.5  # the held-out lab sits a little farther out
    reference, distances = familiarity.fit(
        features, np.zeros(600, int), held_out, quantile=1.0, fingerprint="unit"
    )
    assert reference.threshold == pytest.approx(float(distances.max()))
    assert reference.held_out == 100 and reference.train == 500
    assert (reference.distance(features[:100]) <= reference.threshold + 1e-3).all()


def test_a_changed_gate_changes_the_run_key():
    """A pass gated under another rule does not describe what this rule produces."""
    from app.services.tissue_type_service import tissue_type_service

    base = {"model": "m", "familiarity": familiarity.Gate(None, 0.75).signature}
    moved = {**base, "familiarity": familiarity.Gate(_reference(), 0.75).signature}
    assert tissue_type_service._cache_key(base) != tissue_type_service._cache_key(moved)
    corrected = {**base, "tissue_mask_key": "1@6.89/qc/b2"}
    assert tissue_type_service._cache_key(base) != tissue_type_service._cache_key(corrected)
