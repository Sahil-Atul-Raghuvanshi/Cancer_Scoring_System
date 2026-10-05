"""Step 8's BEETLE option: the refusals, the geometry, and the pass.

Split like `test_tissue_type.py`: almost everything here needs nothing but numpy, and
the handful of tests that need the 1.9 GB archive are skipped when it is absent so a
clone without it still has a green suite.

Five of these are about something less obvious than arithmetic and are worth naming:

`test_a_permuted_label_set_is_refused` is the most important test in this file. The
BEETLE paper and the released weights disagree about which code is invasive and which
is in-situ, and taking the paper's ordering would invert every duct on precisely the
boundary a clinical score is gated on - while producing a class map that looks entirely
plausible. That refusal is the only thing standing between a future release that moves
the codes and a silently inverted result.

`test_a_different_spacing_is_refused` pins the other invisible failure: a network shown
tissue at the wrong physical scale does not fail, it returns a confident wrong answer.

`test_patch_grid_never_leaves_a_join_without_overlap` is the fix for a measured bug.
nnU-Net's own origin rule packs patches flush and can leave two of them meeting edge to
edge; at a full patch step that join was visible as a checkerboard whose squares
disagreed between in-situ and invasive.

`test_cores_partition_the_slide_mask` pins the claim the areas rest on: every mask pixel
is written exactly once, by the window that saw it most centrally, so an area does not
move when the overlap does.

`test_window_order_is_a_serpentine_over_every_cell_once` pins the order the progress
screen paints in - it changes no label, but a scattered sweep is the difference between
a readout and a decoration.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from app.pipeline.step08_tissue_type_segmentation import beetle, pixels
from app.pipeline.step08_tissue_type_segmentation.branches import (
    SERVED_BRANCHES,
    SERVED_HEADS,
    ModelBranch,
)

#: The released archive is 1.9 GB and is not committed. Every test that needs the actual
#: weights is skipped without it rather than failing, which is the same courtesy
#: `test_qc.py` extends to its own downloads.
_HAVE_ARCHIVE, _WHY_NOT = beetle.available()
needs_archive = pytest.mark.skipif(
    not _HAVE_ARCHIVE, reason=f"BEETLE's weights are not installed: {_WHY_NOT}"
)


# --- the refusals ------------------------------------------------------------


def test_the_released_label_set_is_accepted_as_written() -> None:
    """The spelling `dataset.json` actually uses round-trips to `BEETLE_CODES`."""
    released = {
        "unannotated": 0,
        "other": 1,
        "non-invasive epithelium": 2,
        "invasive epithelium": 3,
        "necrosis": 4,
    }
    assert beetle.codes_from_dataset_json(released) == beetle.BEETLE_CODES


def test_the_checkpoints_own_spelling_of_class_zero_is_accepted() -> None:
    """`init_args` calls code 0 `background` where `dataset.json` says `unannotated`.

    Both names mean the same channel, so normalising has to compare meanings rather
    than spellings - otherwise reading the codes from the checkpoint instead of the
    dataset file would be a refusal.
    """
    released = {
        "background": 0,
        "other": 1,
        "non_invasive_epithelium": 2,
        "invasive_epithelium": 3,
        "necrosis": 4,
    }
    assert beetle.codes_from_dataset_json(released) == beetle.BEETLE_CODES


def test_a_permuted_label_set_is_refused() -> None:
    """The BEETLE paper's ordering - invasive at 2 - must not load.

    This is the inversion the whole module is arranged around: with 2 and 3 swapped
    every in-situ duct would be reported as invasive carcinoma and every invasive focus
    as in-situ, on exactly the boundary Rule 5 draws and the score is gated on. It would
    look plausible on screen and be wrong in the one direction nobody questions.
    """
    from_the_paper = {
        "unannotated": 0,
        "other": 1,
        "invasive epithelium": 2,
        "non-invasive epithelium": 3,
        "necrosis": 4,
    }
    with pytest.raises(beetle.BeetleError, match="in-situ duct would be reported"):
        beetle.codes_from_dataset_json(from_the_paper)


def test_an_unknown_class_is_refused_rather_than_guessed() -> None:
    with pytest.raises(beetle.BeetleError, match="no translation"):
        beetle.codes_from_dataset_json({**beetle.BEETLE_CODES, "lymphocytes": 5})


def test_a_different_spacing_is_refused(tmp_path, monkeypatch) -> None:
    """An archive trained at another resolution than this module resamples to is refused.

    Feeding a network the wrong physical scale is the one preprocessing error that looks
    like a bad model rather than a bug, so it has to fail at load and not silently.
    """
    import json
    import zipfile

    plans = {
        "configurations": {
            "2d": {
                "UNet_class_name": "PlainConvUNet",
                "patch_size": [512, 512],
                "pool_op_kernel_sizes": [[1, 1], [2, 2]],
                "conv_kernel_sizes": [[3, 3], [3, 3]],
                "UNet_base_num_features": 32,
                "unet_max_num_features": 512,
                "n_conv_per_stage_encoder": [2, 2],
                "n_conv_per_stage_decoder": [2],
            }
        }
    }
    dataset = {
        "labels": {
            "unannotated": 0,
            "other": 1,
            "non-invasive epithelium": 2,
            "invasive epithelium": 3,
            "necrosis": 4,
        },
        # 0.25 um/px - a 40x scan rather than the 20x the release was fitted at.
        "spacing": 0.25,
    }

    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(beetle.MEMBER_ROOT + "plans.json", json.dumps(plans))
        zf.writestr(beetle.MEMBER_ROOT + "dataset.json", json.dumps(dataset))

    with pytest.raises(beetle.BeetleError, match="wrong physical scale"):
        beetle.read_archive(archive)


def test_a_missing_archive_says_where_it_looked_and_what_to_fetch() -> None:
    """The reason has to be actionable: it is a 1.9 GB download nobody can guess at."""
    usable, problem = beetle.available()
    if usable:
        assert problem is None
        return
    assert "16812932" in problem
    assert str(beetle.model_zip().parent) in problem


def test_a_patch_step_outside_the_unit_interval_is_refused() -> None:
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(beetle.BeetleError, match="patch step"):
            beetle._step(bad)


def test_more_folds_than_the_release_ships_is_refused() -> None:
    with pytest.raises(beetle.BeetleError, match="folds"):
        beetle.folds_to_run(6)
    with pytest.raises(beetle.BeetleError, match="folds"):
        beetle.folds_to_run(0)


# --- the class set -----------------------------------------------------------


def test_the_class_names_are_the_released_codes_in_code_order() -> None:
    """`PIXEL_CLASSES` is indexed by pixel value everywhere, so it must be code order."""
    assert len(beetle.PIXEL_CLASSES) == len(beetle.BEETLE_CODES)
    for name, code in beetle.BEETLE_CODES.items():
        assert beetle.PIXEL_CLASSES[code] == name


def test_the_scored_class_is_invasive_epithelium_and_nothing_else() -> None:
    """Rule 5, as an assertion. Every area and share downstream is this one index."""
    assert beetle.SCORED_CODE == beetle.BEETLE_CODES["invasive_epithelium"]
    assert beetle.PIXEL_CLASSES[beetle.SCORED_CODE] == "invasive_epithelium"


def test_glass_is_outside_the_tissue_denominator_and_everything_else_is_in_it() -> None:
    """A slide with more empty space must not report a smaller tumour content for it."""
    assert beetle.GLASS_CODE == beetle.BEETLE_CODES["unannotated"]
    assert beetle.GLASS_CODE not in beetle.TISSUE_CODES
    assert set(beetle.TISSUE_CODES) == set(range(len(beetle.PIXEL_CLASSES))) - {
        beetle.GLASS_CODE
    }


def test_every_class_has_a_colour_a_label_and_a_meaning() -> None:
    """A legend that can disagree with the pixels it describes eventually will."""
    for code in range(len(beetle.PIXEL_CLASSES)):
        assert code in beetle.CLASS_COLOURS
        assert code in beetle.CLASS_LABELS
        assert code in beetle.CLASS_PLAIN
        assert code in beetle.CLASS_MEANING
    assert len({beetle.CLASS_COLOURS[c] for c in range(5)}) == 5


# --- the input contract ------------------------------------------------------


def test_the_input_is_division_by_255_and_nothing_else() -> None:
    """`RGBTo01Normalization`. Any z-score or ImageNet shift here would be a bug."""
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out = beetle.to_model_input(rgb)

    assert out.shape == (3, 2, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_a_non_three_channel_window_is_refused() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        beetle.to_model_input(np.zeros((4, 4), dtype=np.uint8))


# --- the geometry ------------------------------------------------------------


def test_the_four_fields_of_view_are_the_documented_window_sizes() -> None:
    """The whole geometry of the branch: a constant spacing and four extents.

    Contrast the trained options, where the *pixel* side is held at 224 and the
    resolution moves. Here the resolution is the release's and cannot move, so the pixel
    side does - which is why one download serves all four.
    """
    assert beetle.window_px(112.0) == 224
    assert beetle.window_px(224.0) == 448
    assert beetle.window_px(448.0) == 896
    assert beetle.window_px(672.0) == 1344

    for um in (112.0, 224.0, 448.0, 672.0):
        assert beetle.window_px(um) * beetle.SPACING == pytest.approx(um)


def test_a_field_of_view_too_small_to_be_one_is_refused() -> None:
    with pytest.raises(beetle.BeetleError, match="not a field of view"):
        beetle.window_px(4.0)


def test_a_window_at_or_under_one_patch_needs_a_single_origin() -> None:
    """112 um and 224 um are 224 px and 448 px - both under 512, so both mirror-pad."""
    assert beetle.patch_grid(224, 512, 0.5) == [0]
    assert beetle.patch_grid(448, 512, 0.5) == [0]
    assert beetle.patch_grid(512, 512, 0.5) == [0]


def test_patch_grid_covers_the_whole_extent() -> None:
    """No unsegmented strip at the far edge, at any step or extent."""
    for extent in (513, 640, 896, 1000, 1344, 2048):
        for step in (0.25, 0.5, 0.75, 1.0):
            origins = beetle.patch_grid(extent, 512, step)
            assert origins[0] == 0
            assert origins[-1] == extent - 512
            covered = np.zeros(extent, dtype=bool)
            for origin in origins:
                covered[origin : origin + 512] = True
            assert covered.all(), (extent, step, origins)


def test_patch_grid_never_leaves_a_join_without_overlap() -> None:
    """Consecutive origins are closer together than the patch is wide.

    **The fix for a measured bug.** nnU-Net's own rule - stride from zero, then append
    the flush-right origin - gives 0, 512, 832 for a 1344 px window at a full step, and
    the first two patches share not one pixel. A U-Net is worst at the edge of its
    receptive field, so that join is a visible line, and at a full step it showed up as
    a checkerboard whose squares disagreed between in-situ and invasive: the seam did
    not blur the boundary, it inverted the class across it. Spreading the same number of
    origins evenly costs nothing and leaves every pair overlapping.
    """
    for extent in (513, 640, 896, 1000, 1344, 2048, 3000):
        for step in (0.25, 0.5, 0.75, 1.0):
            origins = beetle.patch_grid(extent, 512, step)
            gaps = np.diff(origins)
            assert (gaps < 512).all(), (extent, step, origins)


def test_patch_grid_keeps_nnunets_own_patch_count_at_every_size_this_branch_uses() -> None:
    """The spreading is a redistribution, not a cost increase.

    For the four window sizes this branch actually reads, at either step, the evenly
    spread origins are the same *number* as nnU-Net's own rule would give - so fixing
    the seam changes where the patches sit and not what the pass costs.
    """
    for um in (112.0, 224.0, 448.0, 672.0):
        extent = beetle.window_px(um)
        for step in (0.5, 1.0):
            stride = int(round(512 * step))
            expected = max(1, int(np.ceil(max(0, extent - 512) / stride)) + 1)
            assert len(beetle.patch_grid(extent, 512, step)) == expected, (um, step)


def test_an_exact_multiple_costs_one_more_patch_to_keep_the_overlap() -> None:
    """The one case even spreading cannot fix, and where the guarantee wins over the cost.

    When `extent - patch` is an exact multiple of the stride the even spread *is* the
    flush packing, so `patch_grid` raises the count instead. It never happens at the
    sizes above, which is what makes the promise free.
    """
    assert beetle.patch_grid(2048, 512, 1.0) == [0, 384, 768, 1152, 1536]


def test_the_evenly_spread_origins_are_the_documented_ones() -> None:
    """The two cases the config comment and the module docstring both quote."""
    # 672 um at a full step: nnU-Net would give 0, 512, 832 with a zero-overlap join.
    assert beetle.patch_grid(1344, 512, 1.0) == [0, 416, 832]
    # 448 um, where the flush rule already overlapped and nothing changes.
    assert beetle.patch_grid(896, 512, 1.0) == [0, 384]


def test_the_patch_count_per_window_is_the_grid_squared() -> None:
    assert beetle.patches_per_window(112.0, 512, 0.5) == 1
    assert beetle.patches_per_window(224.0, 512, 0.5) == 1
    assert beetle.patches_per_window(448.0, 512, 0.5) == 9
    assert beetle.patches_per_window(672.0, 512, 0.5) == 25
    # A full step is four times cheaper, which is why it was tempting.
    assert beetle.patches_per_window(672.0, 512, 1.0) == 9


def test_the_gaussian_weights_never_reach_zero() -> None:
    """A pixel covered only by patch corners must not be divided by nearly nothing."""
    weights = beetle.gaussian_weights(64)
    assert weights.shape == (64, 64)
    assert weights.max() == pytest.approx(1.0)
    assert weights.min() > 0.0


# --- the branch wiring -------------------------------------------------------


def test_beetle_is_a_served_branch_but_not_a_served_head() -> None:
    """It is offered on step 7 and it is not a checkpoint of ours.

    The distinction is load-bearing: a geometry scan over `models/tissue_type/` must
    never be asked about BEETLE, because it would either return nothing - making a
    working option look unavailable - or match one of ours by accident.
    """
    assert ModelBranch.BEETLE in SERVED_BRANCHES
    assert ModelBranch.BEETLE not in SERVED_HEADS


def test_no_checkpoint_of_ours_can_claim_the_beetle_branch() -> None:
    """`Candidate.branch` is derived from a manifest's channel, and BEETLE declares none."""
    from app.pipeline.step08_tissue_type_segmentation import model as region_model

    for candidate in region_model.discover():
        assert candidate.branch is not ModelBranch.BEETLE


def test_model_for_declines_to_answer_about_beetle() -> None:
    from app.services import tiling_service as tiling

    for um in (112.0, 224.0, 448.0, 672.0):
        assert tiling.model_for(um, ModelBranch.BEETLE) is None


def test_step_seven_offers_beetle_at_every_field_of_view_with_its_own_geometry() -> None:
    """One release covers all four, so they are available together or not at all."""
    from app.services import tiling_service as tiling

    offered = tiling._offered_fields_of_view(ModelBranch.BEETLE)
    assert [entry.um for entry in offered] == [112.0, 224.0, 448.0, 672.0]
    for entry in offered:
        assert entry.mpp == beetle.SPACING
        assert entry.tile_px == beetle.window_px(entry.um)
        assert entry.available is _HAVE_ARCHIVE
    assert len({entry.available for entry in offered}) == 1


# --- the pass ----------------------------------------------------------------


def _grid(*, cols: int, rows: int, size: int = 896, overlap: float = 0.0):
    """A window grid with every cell inside, built without a slide or a tile index."""
    from app.pipeline.step08_tissue_type_segmentation.inference import WindowGrid

    stride_px = max(1, int(round(size * (1.0 - overlap))))
    # A base resolution equal to the model's, so level-0 pixels and window pixels
    # coincide and the arithmetic under test is the geometry rather than a rescale.
    return WindowGrid(
        cols=cols,
        rows=rows,
        size=size,
        stride_px=stride_px,
        span=size,
        stride=stride_px,
        overlap=overlap,
        mpp=beetle.SPACING,
        base_mpp=beetle.SPACING,
        slide_width=stride_px * (cols - 1) + size,
        slide_height=stride_px * (rows - 1) + size,
        inside=np.ones((rows, cols), dtype=bool),
    )


def test_window_order_is_a_serpentine_over_every_cell_once() -> None:
    """A ploughed field at window granularity, and a permutation of the block's cells.

    It changes no label - every window is an independent set of forward passes - so what
    it decides is only what someone watching the progress screen sees. A scattered sweep
    inside a marching block reads as a decoration rather than a readout.
    """
    cells = tuple((row, col) for row in range(3) for col in range(4))

    forward = pixels.window_order(cells, reverse=False)
    assert sorted(forward) == list(range(len(cells)))
    assert [cells[index] for index in forward] == [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (1, 2), (1, 1), (1, 0),
        (2, 0), (2, 1), (2, 2), (2, 3),
    ]

    # Reversed, the block starts in the direction the band of blocks is travelling.
    backward = pixels.window_order(cells, reverse=True)
    assert sorted(backward) == list(range(len(cells)))
    assert [cells[index] for index in backward][:4] == [(0, 3), (0, 2), (0, 1), (0, 0)]


def test_window_order_joins_its_rows_end_to_end() -> None:
    """Consecutive rows must meet, or the painted edge jumps the block's width."""
    cells = tuple((row, col) for row in range(4) for col in range(5))
    for reverse in (False, True):
        visited = [cells[index] for index in pixels.window_order(cells, reverse=reverse)]
        for before, after in zip(visited, visited[1:], strict=False):
            # Either the same row one column over, or the next row in the same column.
            assert (before[0] == after[0] and abs(before[1] - after[1]) == 1) or (
                after[0] == before[0] + 1 and before[1] == after[1]
            )


def test_mask_shape_follows_the_slide_and_the_mask_resolution() -> None:
    grid = _grid(cols=4, rows=3, size=896)
    height, width = pixels.mask_shape(grid, 4.0)
    assert width == round(grid.slide_width * grid.base_mpp / 4.0)
    assert height == round(grid.slide_height * grid.base_mpp / 4.0)

    with pytest.raises(pixels.PixelError, match="not a resolution"):
        pixels.mask_shape(grid, 0.0)


def test_reduce_probabilities_area_averages_and_keeps_the_distribution() -> None:
    """Not nearest-neighbour: a thin structure must not vanish or thicken arbitrarily.

    A plane that is half ones and half zeros must reduce to a half everywhere along the
    boundary rather than to whichever pixel the sampling grid happened to land on.
    """
    planes = np.zeros((2, 4, 4), dtype=np.float32)
    planes[0, :2] = 1.0
    planes[1, 2:] = 1.0

    reduced = pixels.reduce_probabilities(planes, 2, 2)
    assert reduced.shape == (2, 2, 2)
    np.testing.assert_allclose(reduced[0], [[1.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(reduced[1], [[0.0, 0.0], [1.0, 1.0]])

    # A 4x1 reduction averages the boundary rather than picking a side.
    halved = pixels.reduce_probabilities(planes, 1, 1)
    assert halved[0, 0, 0] == pytest.approx(0.5)

    # Already the target shape: returned untouched, so a window at size costs nothing.
    assert pixels.reduce_probabilities(planes, 4, 4) is planes


def test_encode_paint_round_trips_to_class_ids() -> None:
    """Raw ids and not a PNG - the client writes them straight into an ImageData."""
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[:32] = beetle.SCORED_CODE
    labels[32:] = beetle.GLASS_CODE

    blob = pixels.encode_paint(labels, 32)
    decoded = np.frombuffer(base64.b64decode(blob), dtype=np.uint8)
    assert decoded.size == 32 * 32

    square = decoded.reshape(32, 32)
    assert (square[:16] == beetle.SCORED_CODE).all()
    assert (square[16:] == beetle.GLASS_CODE).all()


def _fake_beetle(monkeypatch, *, verdict) -> beetle.Loaded:
    """A `Loaded` whose `predict_window` returns a fixed field. No weights, no torch.

    The pass's geometry - which pixels a window owns, what the counts come to, what
    order the paint feed arrives in - is entirely independent of what the network says,
    so it is worth testing without a 1.9 GB download and a minute of CPU per window.
    """
    archive = beetle.Archive(plans={}, dataset={"labels": beetle.BEETLE_CODES,
                                                "spacing": beetle.SPACING},
                             patch=512, path=beetle.model_zip())
    loaded = beetle.Loaded(archive=archive, nets=(), folds=(0,), window_px=896)

    def predict(rgb, model, *, step=None, batch_size=1):
        height, width = np.asarray(rgb).shape[:2]
        field = np.zeros((len(beetle.PIXEL_CLASSES), height, width), dtype=np.float32)
        field[verdict] = 1.0
        return field

    monkeypatch.setattr(beetle, "predict_window", predict)
    return loaded


def test_cores_partition_the_slide_mask(monkeypatch) -> None:
    """Every mask pixel written exactly once, by the window that saw it most centrally.

    The claim every area on the report rests on. Windows overlap when step 7's overlap
    is not zero and their `stride`-wide cores do not, so writing cores means an area
    does not move when the overlap does - and no window overprints its neighbour, which
    at 50% overlap would decide the map by visiting order.
    """
    loaded = _fake_beetle(monkeypatch, verdict=beetle.SCORED_CODE)
    grid = _grid(cols=3, rows=2, size=896, overlap=0.0)

    def read(x, y, span, size):
        return np.zeros((size, size, 3), dtype=np.uint8)

    result = pixels.segment(
        grid, loaded, read_window=read, mask_mpp=4.0, patch_step=1.0, paint_px=8,
        block_windows=8,
    )

    written = result.mask != pixels.OUTSIDE
    # Each window owns a stride-sized core; at zero overlap that is the full window.
    per_window = (grid.stride * grid.base_mpp / 4.0) ** 2
    assert written.sum() == pytest.approx(grid.windows * per_window, rel=0.02)
    assert result.counts[beetle.SCORED_CODE] == written.sum()
    assert result.tumour_content == pytest.approx(1.0)


def test_an_overlapping_grid_does_not_inflate_the_area(monkeypatch) -> None:
    """The same tissue seen more often is not more tissue.

    At 50% overlap there are roughly four times the windows over the same ground. If the
    pass wrote spans instead of cores, the written area would grow with the overlap and
    every mm2 on the report would be a function of a display setting.
    """
    loaded = _fake_beetle(monkeypatch, verdict=beetle.SCORED_CODE)

    def read(x, y, span, size):
        return np.zeros((size, size, 3), dtype=np.uint8)

    for overlap in (0.0, 0.5):
        grid = _grid(cols=4, rows=4, size=896, overlap=overlap)
        result = pixels.segment(
            grid, loaded, read_window=read, mask_mpp=4.0, patch_step=1.0, paint_px=8,
        )
        covered = (result.mask != pixels.OUTSIDE).sum()

        # **Area per window is the core, at every overlap.** That is the invariant, and
        # it is not the same as "the written area is the same": a 50% overlapped grid of
        # the same cell count spans less slide, and its cores leave the outer half-window
        # rim unwritten because no core reaches it. What must not happen is a window
        # claiming its whole span - then four windows over one place would write four
        # times the ground, and every mm2 on the report would move with a display
        # setting. `WindowGrid.cell_mm2` makes the same distinction.
        per_window = (grid.stride * grid.base_mpp / 4.0) ** 2
        assert covered == pytest.approx(grid.windows * per_window, rel=0.02), overlap

        # And the report's own figure agrees with the pixels it was counted from.
        assert result.areas_mm2[beetle.SCORED_CODE] == pytest.approx(
            covered * (4.0 / 1000.0) ** 2, rel=1e-3
        )


def test_the_paint_feed_arrives_once_per_window_in_sweep_order(monkeypatch) -> None:
    """One flush per window, with a decodable mask, and every window exactly once."""
    loaded = _fake_beetle(monkeypatch, verdict=beetle.BEETLE_CODES["necrosis"])
    grid = _grid(cols=3, rows=3, size=896, overlap=0.0)

    seen: list[tuple[int, int, int, str]] = []

    def read(x, y, span, size):
        return np.zeros((size, size, 3), dtype=np.uint8)

    result = pixels.segment(
        grid, loaded, read_window=read, mask_mpp=4.0, patch_step=1.0, paint_px=16,
        painted=lambda cells: seen.extend(cells),
    )

    assert len(seen) == grid.windows == result.windows_done
    assert sorted((row, col) for row, col, _, _ in seen) == sorted(
        (row, col) for row in range(grid.rows) for col in range(grid.cols)
    )
    for _, _, label, blob in seen:
        assert label == beetle.BEETLE_CODES["necrosis"]
        decoded = np.frombuffer(base64.b64decode(blob), dtype=np.uint8)
        assert decoded.size == 16 * 16
        assert (decoded == beetle.BEETLE_CODES["necrosis"]).all()


def test_a_cancel_lands_within_a_window(monkeypatch) -> None:
    """Checked after every window, not every block - a block here is minutes of work."""
    from app.pipeline.contract import RunCancelled

    loaded = _fake_beetle(monkeypatch, verdict=beetle.SCORED_CODE)
    grid = _grid(cols=4, rows=4, size=896, overlap=0.0)

    def read(x, y, span, size):
        return np.zeros((size, size, 3), dtype=np.uint8)

    with pytest.raises(RunCancelled, match="stopped after 1 of"):
        pixels.segment(
            grid, loaded, read_window=read, mask_mpp=4.0, patch_step=1.0, paint_px=8,
            should_stop=lambda: True,
        )


def test_class_confidence_is_measured_over_the_pixels_each_class_won(monkeypatch) -> None:
    """Per pixel, not per window - a class can hold tissue without dominating a window.

    The fix for a measured misreading. At 672 um every window of a test region was
    dominated by `other`, so a window-based figure left three of the five classes
    falling back to their mean asserted probability, which lands near their pixel share
    and reads on screen as a duplicate of it. This is the winning probability over
    exactly the mask pixels the class won.
    """
    loaded = _fake_beetle(monkeypatch, verdict=beetle.SCORED_CODE)
    grid = _grid(cols=2, rows=2, size=896, overlap=0.0)

    def read(x, y, span, size):
        return np.zeros((size, size, 3), dtype=np.uint8)

    result = pixels.segment(
        grid, loaded, read_window=read, mask_mpp=4.0, patch_step=1.0, paint_px=8,
    )

    # The stub asserts its verdict at probability 1.0 everywhere.
    assert result.class_confidence[beetle.SCORED_CODE] == pytest.approx(1.0)
    assert result.mean_confidence == pytest.approx(1.0)
    # A class that won no pixel is 0.0 - "not measured", not "measured as unsure".
    for code in range(len(beetle.PIXEL_CLASSES)):
        if code != beetle.SCORED_CODE:
            assert result.class_confidence[code] == 0.0


def test_a_pixel_map_survives_a_round_trip_through_disk(monkeypatch, tmp_path) -> None:
    """Everything the report reads must come back, including the measured confidence.

    `class_confidence` is a mean over the probabilities *before* they were argmaxed into
    the mask, so the mask alone cannot yield it back - it has to be stored, and a reload
    that quietly returned zeros would make a cached pass report less than a fresh one.
    """
    from app.core.config import settings
    from app.services.tissue_type_service import tissue_type_service

    loaded = _fake_beetle(monkeypatch, verdict=beetle.BEETLE_CODES["non_invasive_epithelium"])
    grid = _grid(cols=2, rows=2, size=896, overlap=0.0)

    def read(x, y, span, size):
        return np.zeros((size, size, 3), dtype=np.uint8)

    fresh = pixels.segment(
        grid, loaded, read_window=read, mask_mpp=4.0, patch_step=1.0, paint_px=8,
    )

    monkeypatch.setattr(settings, "tissue_type_dir", tmp_path)
    tissue_type_service._store_pixels("round-trip", fresh)
    reloaded = tissue_type_service.pixel_map("round-trip")

    assert reloaded.counts == fresh.counts
    assert reloaded.areas_mm2 == fresh.areas_mm2
    assert reloaded.class_confidence == pytest.approx(fresh.class_confidence)
    assert reloaded.mask_mpp == fresh.mask_mpp
    assert reloaded.windows_done == fresh.windows_done
    assert reloaded.patches == fresh.patches
    assert reloaded.tumour_content == pytest.approx(fresh.tumour_content)
    np.testing.assert_array_equal(reloaded.mask, fresh.mask)


def test_glass_is_out_of_the_share_denominator(monkeypatch) -> None:
    """A map that is entirely BEETLE's background class has no tissue to take shares of."""
    loaded = _fake_beetle(monkeypatch, verdict=beetle.GLASS_CODE)
    grid = _grid(cols=2, rows=2, size=896, overlap=0.0)

    def read(x, y, span, size):
        return np.zeros((size, size, 3), dtype=np.uint8)

    result = pixels.segment(
        grid, loaded, read_window=read, mask_mpp=4.0, patch_step=1.0, paint_px=8,
    )
    assert result.counts[beetle.GLASS_CODE] > 0
    assert result.tissue_pixels == 0
    assert result.tumour_content == 0.0
    assert result.areas_mm2[beetle.GLASS_CODE] > 0


# --- the real weights --------------------------------------------------------


@needs_archive
def test_the_released_archive_is_the_one_this_module_was_written_against() -> None:
    archive = beetle.read_archive()
    assert archive.patch == beetle.NOMINAL_PATCH_PX
    assert archive.num_classes == len(beetle.PIXEL_CLASSES)
    assert archive.spacing == beetle.SPACING


@needs_archive
def test_the_rebuilt_network_loads_the_release_with_no_key_mismatch() -> None:
    """A strict key-set comparison. 80% of the weights loading cleanly is not enough.

    A model built from the wrong plans loads most of its parameters and then segments
    noise, and that failure looks exactly like a bad dataset for a week.
    """
    archive = beetle.read_archive()
    net = beetle.load_fold(archive, 0)
    assert sum(p.numel() for p in net.parameters()) > 40_000_000


@needs_archive
def test_the_network_answers_something_structured_rather_than_uniform() -> None:
    """The check a strict weight load cannot make: that the wiring is right.

    A skip connection attached to the wrong stage would load cleanly and segment noise.
    What noise cannot do is produce a spatially structured answer at a confidence well
    above the 1/5 a five-class softmax bottoms out at.
    """
    verdict = beetle.self_check()
    assert verdict["classes"] == len(beetle.PIXEL_CLASSES)
    assert verdict["distinct_labels"] >= 2
    assert verdict["mean_top_probability"] > 0.5


@needs_archive
def test_a_window_smaller_than_one_patch_is_mirror_padded_not_resized() -> None:
    """112 um and 224 um are the ordinary case for this, and both must come back at size.

    The network is fully convolutional and would accept the smaller array, but its
    batch-norm statistics and receptive field were fitted at 512.
    """
    loaded = beetle.load(224.0)
    assert loaded.window_px == 448

    rgb = np.full((448, 448, 3), 200, dtype=np.uint8)
    probs = beetle.predict_window(rgb, loaded, batch_size=1)

    assert probs.shape == (len(beetle.PIXEL_CLASSES), 448, 448)
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-4)


@needs_archive
def test_the_field_of_view_changes_the_window_and_not_the_weights() -> None:
    """One release serves all four, which is why the branch needs no per-scale head."""
    small, large = beetle.load(112.0), beetle.load(672.0)
    assert small.window_px == 224 and large.window_px == 1344
    assert small.mpp == large.mpp == beetle.SPACING
    # The same cached module object, not two loads of it.
    assert small.nets[0] is large.nets[0]
