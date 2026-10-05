"""Step 2 tests.

Split by what they need:

  * the class table, the classical metrics and the mask reduction need nothing
    beyond numpy and scipy, so they run everywhere and are the bulk of this
    file;
  * the endpoints are tested for the states that do not need a model - what
    they say when nothing has been run, and when a checkpoint is missing;
  * anything requiring the GrandQC weights is skipped with a reason rather than
    faked. A mocked segmentation model would test the mock.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.pipeline.step02_quality_control import classes as qc_classes
from app.pipeline.step02_quality_control.features import METRIC_KEYS, block_features, tile_features
from app.pipeline.step02_quality_control.inference import _reduce_patch_mask, plan_grid
from app.pipeline.step02_quality_control.models import ARTEFACT_CHECKPOINTS, discover_checkpoints
from app.services.qc_service import _camel

# --- the class table ---------------------------------------------------------


def test_class_ids_match_grandqc_documented_coding() -> None:
    """The ids are GrandQC's, not ours; getting one wrong mislabels every mask."""
    expected = {
        1: "tissue",
        2: "fold",
        3: "darkspot",
        4: "pen",
        5: "edge",
        6: "focus",
        7: "background",
    }
    assert {item.id: item.key for item in qc_classes.QC_CLASSES} == expected


def test_artefacts_are_classes_two_through_six() -> None:
    assert [item.id for item in qc_classes.ARTEFACT_CLASSES] == [2, 3, 4, 5, 6]


def test_tissue_denominator_excludes_background_and_the_unanalysed_margin() -> None:
    """Glass was never analysable, so it cannot be in a 'share of tissue'."""
    assert qc_classes.BACKGROUND not in qc_classes.TISSUE_CLASS_IDS
    assert qc_classes.UNANALYSED not in qc_classes.TISSUE_CLASS_IDS
    assert set(qc_classes.TISSUE_CLASS_IDS) == {1, 2, 3, 4, 5, 6}


def test_palette_puts_each_colour_at_its_own_class_index() -> None:
    palette = qc_classes.palette_rows()
    for item in qc_classes.QC_CLASSES:
        assert palette[item.id] == item.colour


# --- the classical metrics ---------------------------------------------------


def _checkerboard(size: int = 256, cell: int = 8) -> np.ndarray:
    """A high-frequency pattern: maximal sharpness and contrast."""
    y, x = np.mgrid[0:size, 0:size]
    board = (((x // cell) + (y // cell)) % 2 * 255).astype(np.uint8)
    return np.repeat(board[..., None], 3, axis=2)


def _flat(size: int = 256, value: int = 200) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_every_metric_is_reported() -> None:
    features = tile_features(_checkerboard()).as_dict()
    assert set(features) == set(METRIC_KEYS)
    assert all(np.isfinite(value) for value in features.values())


def test_sharpness_separates_a_sharp_tile_from_a_blurred_copy() -> None:
    """The whole out-of-focus explanation rests on this being true."""
    from scipy import ndimage

    sharp = _checkerboard()
    blurred = ndimage.gaussian_filter(sharp.astype(np.float32), sigma=(3, 3, 0)).astype(np.uint8)

    assert tile_features(sharp).tenengrad > tile_features(blurred).tenengrad * 5
    assert tile_features(sharp).laplacian_variance > tile_features(blurred).laplacian_variance


def test_a_flat_tile_has_no_contrast_and_no_texture() -> None:
    """Gabor kernels are mean-subtracted so a flat field responds with zero."""
    features = tile_features(_flat())
    assert features.rms_contrast == pytest.approx(0.0, abs=1e-6)
    assert features.michelson_contrast == pytest.approx(0.0, abs=1e-6)
    assert features.texture_energy == pytest.approx(0.0, abs=1e-4)


def test_saturation_separates_ink_from_a_grey_field() -> None:
    """This is the metric that explains a pen call, so it has to behave."""
    grey = _flat(value=128)
    ink = np.zeros((256, 256, 3), dtype=np.uint8)
    ink[..., 0] = 200  # strongly red, like a marker

    assert tile_features(grey).saturation == pytest.approx(0.0, abs=1e-6)
    assert tile_features(ink).saturation > 0.9


def test_brightness_orders_dark_below_pale() -> None:
    assert tile_features(_flat(value=30)).brightness < tile_features(_flat(value=230)).brightness


def test_michelson_ignores_a_single_speck() -> None:
    """Percentiles rather than min/max, so one dust pixel cannot set the scale."""
    field = _flat(value=200)
    speckled = field.copy()
    speckled[0, 0] = 0

    assert tile_features(speckled).michelson_contrast == pytest.approx(
        tile_features(field).michelson_contrast, abs=1e-6
    )


# --- block measurement -------------------------------------------------------


def test_block_features_agrees_with_measuring_each_block_alone() -> None:
    """The batched path is an optimisation, so it has to give the same answers.

    Filters run once over the whole patch and are reduced per block, which is
    cheaper and slightly better at block edges. The statistics that do not
    depend on filter boundaries must match a per-block computation exactly.
    """
    rng = np.random.default_rng(7)
    patch = (rng.random((256, 256, 3)) * 255).astype(np.uint8)

    rows = block_features(patch, blocks=2)
    assert len(rows) == 4

    for index, (top, left) in enumerate([(0, 0), (0, 128), (128, 0), (128, 128)]):
        alone = tile_features(patch[top : top + 128, left : left + 128]).as_dict()
        for key in ("brightness", "saturation", "rms_contrast", "michelson_contrast"):
            assert rows[index][key] == pytest.approx(alone[key], rel=1e-6)


def test_block_features_are_row_major() -> None:
    """Block index must be row * blocks + col, or every heatmap is transposed."""
    patch = np.zeros((256, 256, 3), dtype=np.uint8)
    patch[0:128, 128:256] = 255  # top-right block only

    rows = block_features(patch, blocks=2)
    brightness = [row["brightness"] for row in rows]

    assert brightness[1] > 0.9, "index 1 should be row 0, col 1"
    assert max(brightness[0], brightness[2], brightness[3]) < 0.01


def test_gabor_fft_path_matches_a_direct_convolution() -> None:
    """The FFT shortcut must not change the numbers, only the time they take.

    numpy and ndimage give the same two words opposite meanings for padding, so
    this guards the one that is easy to get silently wrong.
    """
    from scipy import ndimage

    from app.pipeline.step02_quality_control.features import (
        _decimate,
        _gabor_bank,
        _gabor_response,
        _luminance,
    )

    rng = np.random.default_rng(3)
    image = _decimate(_luminance((rng.random((256, 256, 3)) * 255).astype(np.uint8)), 256)

    for kernel in _gabor_bank():
        fft = _gabor_response(image, kernel)
        direct = ndimage.convolve(image, kernel, mode="reflect")
        assert np.abs(fft - direct).max() < 1e-3


# --- mask reduction ----------------------------------------------------------


def test_reduction_keeps_a_pen_line_one_pixel_wide() -> None:
    """Nearest-neighbour decimation would erase exactly the finding people look for."""
    patch = np.full((512, 512), qc_classes.TISSUE, dtype=np.uint8)
    patch[:, 100] = qc_classes.PEN

    reduced = _reduce_patch_mask(patch, 64)
    assert reduced.shape == (64, 64)
    assert qc_classes.PEN in np.unique(reduced)


def test_reduction_keeps_the_majority_where_there_is_no_artefact() -> None:
    patch = np.full((512, 512), qc_classes.BACKGROUND, dtype=np.uint8)
    patch[:256] = qc_classes.TISSUE

    reduced = _reduce_patch_mask(patch, 64)
    assert set(np.unique(reduced[:32])) == {qc_classes.TISSUE}
    assert set(np.unique(reduced[32:])) == {qc_classes.BACKGROUND}


def test_reduction_is_a_noop_when_the_cell_size_matches() -> None:
    patch = np.full((64, 64), qc_classes.FOLD, dtype=np.uint8)
    assert np.array_equal(_reduce_patch_mask(patch, 64), patch)


# --- geometry ----------------------------------------------------------------


class _FakeReader:
    """Just enough of a slide for the grid arithmetic."""

    def __init__(self, width: int, height: int) -> None:
        self.dimensions = (width, height)


def test_grid_is_derived_from_mpp_not_from_a_level_index() -> None:
    """A 40x scan and a 20x scan must produce the same physical patch extent."""
    reader = _FakeReader(126976, 126976)

    extent_40x, cols, rows = plan_grid(
        reader, base_mpp=0.2222, model_mpp=1.5, patch_size=512
    )
    extent_20x, _, _ = plan_grid(reader, base_mpp=0.4444, model_mpp=1.5, patch_size=512)

    # 512 model pixels at 1.5 um/px is 768 um of tissue, whatever the scanner.
    assert extent_40x * 0.2222 == pytest.approx(768, rel=0.01)
    assert extent_20x * 0.4444 == pytest.approx(768, rel=0.01)
    assert cols == rows == 126976 // extent_40x


def test_a_coarser_model_needs_fewer_patches() -> None:
    reader = _FakeReader(100000, 100000)
    _, cols_7x, _ = plan_grid(reader, base_mpp=0.25, model_mpp=1.5, patch_size=512)
    _, cols_5x, _ = plan_grid(reader, base_mpp=0.25, model_mpp=2.0, patch_size=512)
    assert cols_5x < cols_7x


# Otsu itself moved to app/common/imaging.py, because step 3 is built on the
# same function - see tests/test_tissue.py for its tests.


# --- naming ------------------------------------------------------------------


def test_metric_keys_are_camel_cased_for_the_wire() -> None:
    assert _camel("laplacian_variance") == "laplacianVariance"
    assert _camel("brightness") == "brightness"
    assert _camel("rms_contrast") == "rmsContrast"


# --- endpoints ---------------------------------------------------------------


def test_capability_answers_even_with_nothing_installed(client: TestClient) -> None:
    """A fresh clone has no weights; that is a state to report, not an error."""
    response = client.get("/api/v1/qc/capability")
    assert response.status_code == 200

    body = response.json()
    assert body["mode"] in {"full", "degraded", "unavailable"}
    assert body["reason"]
    assert body["citation"].startswith("Weng Z.")
    assert len(body["downloads"]) == 2
    assert body["searchedPaths"], "the UI shows where the server looked"


def test_capability_lists_all_four_checkpoints_by_name(client: TestClient) -> None:
    body = client.get("/api/v1/qc/capability").json()
    names = {model["name"] for model in body["models"]}
    assert "Tissue_Detection_MPP10.pth" in names
    assert set(ARTEFACT_CHECKPOINTS.values()) <= names


def test_report_before_any_run_is_a_conflict_not_a_crash(client: TestClient) -> None:
    response = client.get("/api/v1/qc/nope-not-a-slide")
    assert response.status_code == 409
    assert "run" in response.json()["detail"].lower()


def test_run_state_for_an_unrun_slide_is_idle(client: TestClient) -> None:
    body = client.get("/api/v1/qc/nope-not-a-slide/run").json()
    assert body["state"] == "idle"
    assert body["progress"] == 0.0


def test_images_before_a_run_explain_what_to_do(client: TestClient) -> None:
    for name in ("overlay.png", "tissue.png", "classes.png", "mask.png"):
        response = client.get(f"/api/v1/qc/nope-not-a-slide/{name}")
        assert response.status_code == 409
        assert "run step 2" in response.json()["detail"]


def test_starting_a_run_on_an_unknown_upload_is_a_404(client: TestClient) -> None:
    response = client.post("/api/v1/qc/nope-not-a-slide/run")
    # 404 when the upload is missing; 409 when QC itself cannot run here. Both
    # are correct, and which one comes first depends on the machine.
    assert response.status_code in {404, 409}


def test_step_two_is_advertised_as_implemented(client: TestClient) -> None:
    stage = client.get("/api/v1/pipeline/stages/quality-control").json()
    assert stage["implemented"] is True
    assert "GrandQC" in stage["how"]


# The pipeline-wide count of implemented steps is asserted in test_pipeline.py,
# where it belongs - it changes every time a step is built, and a QC test file is
# the wrong place to find out.


# --- the parts that need real weights ---------------------------------------

_checkpoints = discover_checkpoints()

needs_models = pytest.mark.skipif(
    not _checkpoints.complete,
    reason="GrandQC checkpoints not downloaded; see backend/scripts/check_qc_models.py",
)


@needs_models
def test_checkpoints_load_and_emit_the_documented_class_count() -> None:
    """Guards the version skew that pickled models are prone to."""
    from app.pipeline.step02_quality_control.models import load_artefact_model, load_tissue_model

    assert _checkpoints.tissue is not None
    assert load_tissue_model(_checkpoints.tissue, "cpu").classes == 2

    for path in _checkpoints.artefacts.values():
        # GrandQC codes classes 1-7, so the head must have at least 7 channels.
        assert load_artefact_model(path, "cpu").classes >= 7
