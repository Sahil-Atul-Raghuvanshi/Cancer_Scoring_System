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
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.pipeline.step07_tiling.index import build_index
from app.pipeline.step08_tissue_type_segmentation import classes as tissue_classes
from app.pipeline.step08_tissue_type_segmentation import inference, model, overlay
from app.pipeline.step08_tissue_type_segmentation import input as model_input
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
    assert set(tissue_classes.CLASS_MEANING) == {0, 1, 2}
    assert set(tissue_classes.CLASS_LABELS) == {0, 1, 2}
    assert set(tissue_classes.CLASS_PLAIN) == {0, 1, 2}
    assert set(tissue_classes.CLASS_COLOURS) == {0, 1, 2}


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
    `bcss_hchannel_resnet18` its own copy of the arithmetic, this is the test that
    fails - and it is the only place the drift would ever be visible, because the
    model would train and validate perfectly either way.
    """
    research = Path(__file__).resolve().parents[3] / "bcss_hchannel_resnet18" / "src"
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
        """`read_haematoxylin` over a slide that is a ramp in both axes."""
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
        read_haematoxylin=read,
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


def test_progress_ends_at_the_window_count():
    grid = _grid(_mask(fill=0.4))
    seen: list[tuple[int, int]] = []

    inference.classify(
        grid,
        net=_FakeNet(),
        read_haematoxylin=lambda x, y, span, size: np.full((size, size), 0.4, dtype=np.float32),
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
            read_haematoxylin=lambda x, y, span, size: np.full(
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
            read_haematoxylin=lambda x, y, span, size: np.full(
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
        ),
    )
    monkeypatch.setattr(
        service_module.tiling_service,
        "report",
        lambda upload_id: SimpleNamespace(
            params=SimpleNamespace(
                overlap=0.0, tissue_threshold=1, tissue_threshold_source="otsu",
                qc_gated=False, qc_source=None,
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
