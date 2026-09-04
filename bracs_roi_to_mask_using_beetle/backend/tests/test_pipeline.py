"""The pipeline's geometry and its bookkeeping, on synthetic data.

Nothing here loads the 1.9 GB of weights - these are the parts that must be right
*around* the network, and they are the parts a slow test would stop anyone running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bracs_app import compare, config, labels, pipeline, teacher, unet  # noqa: E402

config.install_approach1_path()
import bcss  # noqa: E402
import export  # noqa: E402


# --- geometry -----------------------------------------------------------------


def test_the_last_strip_of_a_region_is_covered():
    """A region 700 px wide must not leave 188 px unsegmented.

    `range(0, extent - patch + 1, step)` stops at 256 for a 700 px extent, covering to
    768 - except 768 > 700, so the real last origin is 188. Without the flush-right
    origin the right-hand strip would be labelled by nothing and silently exported as
    unlabelled tissue.
    """
    starts = teacher._starts(700, 512, 256)
    assert starts[-1] == 700 - 512
    assert starts[-1] + 512 == 700


def test_a_region_smaller_than_a_patch_gets_one_patch():
    assert teacher._starts(300, 512, 256) == [0]


def test_patch_origins_do_not_run_off_the_end():
    for extent in (512, 513, 700, 1024, 1025, 3000):
        for start in teacher._starts(extent, 512, 256):
            assert 0 <= start <= max(0, extent - 512)


def test_gaussian_weights_never_reach_zero():
    """A zero weight at a patch corner divides by zero where only corners overlap."""
    weights = teacher._gaussian_weights(512)
    assert weights.min() > 0
    assert weights.max() == pytest.approx(1.0)
    assert weights[256, 256] == pytest.approx(1.0)


def test_resampling_halves_a_quarter_micron_region():
    """0.25 to 0.5 um/px is a 2x downsample, and the mask must follow the image."""
    rgb = np.random.default_rng(0).integers(0, 255, (400, 600, 3), dtype=np.uint8)
    small, mask = export.resample(
        rgb, np.zeros((400, 600), np.uint8), source_mpp=0.25, target_mpp=0.5
    )
    assert small.shape[:2] == (200, 300) == mask.shape[:2]


# --- the mask file ------------------------------------------------------------


def test_the_mask_round_trips_as_the_same_kind_of_file_bcss_ships(tmp_path):
    """Written, read back, and still the codes that went in.

    `I;16` is the mode PIL gives a single-channel 16-bit PNG, and it is what the BCSS
    release uses. A mask saved as `L` would truncate code 20 fine and be wrong for
    nothing visible - until a code above 255 ever existed.
    """
    codes = labels.to_bcss_codes(np.array([[0, 1, 2], [3, 4, 2]], dtype=np.uint8))
    path = tmp_path / "mask.png"
    Image.fromarray(codes).save(path)

    with Image.open(path) as reopened:
        assert reopened.mode == "I;16"
        back = np.asarray(reopened)

    assert back.dtype == np.uint16
    np.testing.assert_array_equal(back, codes)
    # And approach 1 can read it without knowing where it came from.
    remapped = bcss.remap(back)
    assert remapped[0, 2] == bcss.NON_INVASIVE
    assert remapped[1, 0] == bcss.INVASIVE
    assert remapped[0, 0] == bcss.IGNORE


def test_approach_1s_exporter_accepts_a_teacher_mask(tmp_path):
    """The whole claim of this app, on a synthetic region: our mask goes in unmodified.

    A 700x700 field of in-situ epithelium with a stromal band. At 224 px stride that is
    9 tiles, and the ones inside the epithelium must come out as class 1.
    """
    beetle = np.full((700, 700), labels.BEETLE_CODES["non_invasive_epithelium"], np.uint8)
    beetle[:, 600:] = labels.BEETLE_CODES["other"]

    rgb = np.full((700, 700, 3), 200, np.uint8)
    rgb[::3, ::3] = 60  # something for the white point and the density to work on

    image_path, mask_path = tmp_path / "r.png", tmp_path / "m.png"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(labels.to_bcss_codes(beetle)).save(mask_path)

    region = bcss.Region(
        roi_id="BRACS_9_DCIS_1", slide_id="BRACS_9", institution="BRACS",
        image=image_path, mask=mask_path,
    )
    rows, drops = export.write_region(region, tmp_path / "tiles", source_mpp=0.5)

    assert rows, "no tiles at all means the mask was not read"
    counts = {name: 0 for name in bcss.CLASS_NAMES}
    for row in rows:
        counts[row["label_name"]] += 1
    assert counts["non_invasive_epithelium"] >= 6
    assert set(drops) == {"unusable", "mixed"}

    tile = np.asarray(Image.open(tmp_path / "tiles" / rows[0]["tile_path"]))
    assert tile.shape == (224, 224) and tile.dtype == np.uint8


def test_a_brracs_region_is_never_in_bcss_held_out_institutions():
    """`institution="BRACS"` must not accidentally name a TCGA tissue source site.

    `Region.is_test` reads it, and a two-letter code here would quietly route BRACS
    tiles into approach 1's held-out split - which would put teacher-labelled data into
    the only set that measures the truth.
    """
    region = bcss.Region(
        roi_id="BRACS_1247_DCIS_1", slide_id="BRACS_1247", institution="BRACS",
        image=Path("x.png"), mask=Path("m.png"),
    )
    assert not region.is_test
    assert "BRACS" not in bcss.TEST_INSTITUTIONS


# --- bookkeeping --------------------------------------------------------------


def test_case_id_groups_by_patient_not_by_region():
    """42 of the 665 training regions are patient 1247's, and they must stay together."""
    entry = pipeline.RoiEntry("BRACS_1247_DCIS_42", "train", Path("x"), 10, 10, 1)
    assert entry.case_id == "BRACS_1247"
    assert pipeline.RoiEntry("BRACS_1003697_DCIS_1", "train", Path("x"), 1, 1, 1).case_id == (
        "BRACS_1003697"
    )


def test_verdict_flags_a_dcis_region_with_no_in_situ_found():
    flagged = pipeline.verdict(
        {
            "source": "bracs_dcis",
            "class_area_fraction": {"non_invasive_epithelium": 0.01, "invasive_epithelium": 0.0},
            "tiles": {"by_class": {"non_invasive_epithelium": 0}},
        }
    )
    assert not flagged["usable"]
    assert any("resolution" in note for note in flagged["notes"])


def test_verdict_flags_more_invasive_than_in_situ_in_a_dcis_region():
    """The symptom a swapped label map would produce, called out by name."""
    flagged = pipeline.verdict(
        {
            "source": "bracs_dcis",
            "class_area_fraction": {"non_invasive_epithelium": 0.1, "invasive_epithelium": 0.7},
            "tiles": {"by_class": {"non_invasive_epithelium": 5}},
        }
    )
    assert not flagged["usable"]
    assert any("wrong way round" in note for note in flagged["notes"])


def test_verdict_passes_a_healthy_region():
    assert pipeline.verdict(
        {
            "source": "bracs_dcis",
            "class_area_fraction": {"non_invasive_epithelium": 0.75, "invasive_epithelium": 0.005},
            "tiles": {"by_class": {"non_invasive_epithelium": 38}},
        }
    )["usable"]


def test_verdict_reaches_no_verdict_on_an_uploaded_image():
    """The numbers that flag a BRACS region are not findings about an unknown image.

    Both of the notes are disagreements with BRACS's own annotation: "hardly any in-situ
    epithelium in a region labelled DCIS" is a red flag precisely because two things
    that should agree do not. Nobody has said what an uploaded image contains, so the
    same 1 % is not evidence of anything - and `usable` must be `None` rather than
    `True`, or the UI reads a passed check out of an unperformed one.
    """
    reading = pipeline.verdict(
        {
            "source": "upload",
            "class_area_fraction": {"non_invasive_epithelium": 0.01, "invasive_epithelium": 0.7},
        }
    )
    assert reading["usable"] is None
    assert reading["checked_against_annotation"] is False
    assert reading["notes"] == []
    # The measurements are still there; it is the judgement that is withheld.
    assert reading["non_invasive_area"] == 0.01


def test_uploads_are_labelled_as_uploads_not_as_bracs_regions(tmp_path):
    """`entry_for_upload` must not produce something that later reads as BRACS data.

    `verdict` branches on `manifest["source"]`, and `run_region` derives that from
    `split`. It also must not invent a patient: `RoiEntry.case_id` truncates at the
    second underscore because BRACS encodes the case there, and two uploads sharing a
    prefix are not two regions from one patient.
    """
    path = tmp_path / "some_slide_crop.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)

    entry = pipeline.entry_for_upload(path, "upload_deadbeef1234")
    assert entry.split == "upload"
    assert entry.case_id == "upload_deadbeef1234"
    assert (entry.width, entry.height) == (8, 8)


def test_non_finite_numbers_never_reach_the_json():
    """`json.dumps` emits a bare `NaN`, which no browser will parse."""
    assert compare._finite(float("nan")) is None
    assert compare._finite(float("inf")) is None
    assert compare._finite(1.239, 2) == 1.24


# --- the network --------------------------------------------------------------


def test_the_architecture_is_built_from_the_plans_not_hard_coded():
    """A smaller plans file must produce a correspondingly smaller network."""
    plans = {
        "configurations": {
            "2d": {
                "UNet_class_name": "PlainConvUNet",
                "UNet_base_num_features": 8,
                "unet_max_num_features": 32,
                "n_conv_per_stage_encoder": [2, 2, 2],
                "n_conv_per_stage_decoder": [2, 2],
                "pool_op_kernel_sizes": [[1, 1], [2, 2], [2, 2]],
                "conv_kernel_sizes": [[3, 3], [3, 3], [3, 3]],
            }
        }
    }
    import torch

    model = unet.from_plans(plans, num_classes=5)
    assert model.encoder.output_channels == [8, 16, 32]
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 5, 64, 64)


def test_a_non_plain_architecture_is_refused():
    with pytest.raises(ValueError, match="PlainConvUNet"):
        unet.from_plans(
            {"configurations": {"2d": {"UNet_class_name": "ResidualEncoderUNet"}}}, 5
        )


def test_a_checkpoint_that_does_not_match_is_refused_by_name():
    """Loading 80 % of the weights cleanly is the failure this must not allow."""
    import torch

    plans = {
        "configurations": {
            "2d": {
                "UNet_class_name": "PlainConvUNet",
                "UNet_base_num_features": 8,
                "unet_max_num_features": 16,
                "n_conv_per_stage_encoder": [2, 2],
                "n_conv_per_stage_decoder": [2],
                "pool_op_kernel_sizes": [[1, 1], [2, 2]],
                "conv_kernel_sizes": [[3, 3], [3, 3]],
            }
        }
    }
    model = unet.from_plans(plans, num_classes=5)
    good = {k: v for k, v in model.state_dict().items()}
    unet.load_beetle_weights(model, good)  # the happy path

    del good["encoder.stages.0.0.convs.0.conv.weight"]
    with pytest.raises(RuntimeError, match="absent from the checkpoint"):
        unet.load_beetle_weights(model, good)

    extra = dict(model.state_dict())
    extra["encoder.stages.9.0.convs.0.conv.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="nowhere to go"):
        unet.load_beetle_weights(model, extra)


def test_alias_keys_are_dropped_from_both_sides():
    """The released state dict names every conv block twice; both names are one tensor."""
    assert unet._is_alias("encoder.stages.0.0.convs.0.all_modules.0.weight")
    assert unet._is_alias("decoder.encoder.stages.0.0.convs.0.conv.weight")
    assert not unet._is_alias("encoder.stages.0.0.convs.0.conv.weight")
    assert not unet._is_alias("decoder.seg_layers.6.weight")
