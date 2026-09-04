"""The six-slide comparison: is it comparing two models, or a preprocessing mistake?

None of these need a whole slide or the 1.9 GB teacher. They cover the parts that decide
whether the pictures on that screen mean anything, and the first one is worth more than
all the others together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bracs_app import config, labels, sixslides, student  # noqa: E402

config.install_approach1_path()
import bcss  # noqa: E402
import datasets  # noqa: E402
import hchannel  # noqa: E402


MANIFEST = {
    "input": {
        "tile_px": 224,
        "mpp": 0.5,
        "gamma": 1.0,
        "invert_polarity": False,
        "standardise_tile_p99": True,
    }
}


# --- the one that matters -----------------------------------------------------


def test_the_student_transform_is_bit_for_bit_the_training_transform(tmp_path):
    """Our inference path and `datasets.TileDataset` must produce the same tensor.

    `TileDataset` produced every tensor the head was ever fitted on. If this module's
    transform differs from it by more than float noise, the middle column of the
    comparison is a preprocessing bug being reported as a model's opinion - and nothing
    raises, the picture just comes out wrong.

    This is not a hypothetical. `05_publish.py`'s gate G6 had exactly this fault hours
    ago: it rebuilt its check input without passing `standardise`, the logits moved by
    1.8, and it went unnoticed because a second bug meant the gate was verifying a stale
    checkpoint whose settings happened to match the defaults.
    """
    from PIL import Image

    rng = np.random.default_rng(0)
    stored = rng.integers(0, 256, (224, 224), dtype=np.uint8)
    tiles_dir = tmp_path / "tiles" / "non_epithelium"
    tiles_dir.mkdir(parents=True)
    Image.fromarray(stored, mode="L").save(tiles_dir / "t.png")

    settings = student.input_settings(MANIFEST)
    reference = datasets.TileDataset(
        [{"tile_path": "non_epithelium/t.png", "label": 0, "tile_id": "t"}],
        tmp_path / "tiles",
        augment=False,
        invert=settings["invert"],
        standardise=settings["standardise"],
    )[0][0].numpy()

    ours = student._tile_tensor(hchannel.dequantise(stored), settings)
    np.testing.assert_array_equal(ours, reference)


def test_a_wrong_standardise_setting_would_have_been_caught():
    """The guard is only worth having if the two settings actually differ."""
    rng = np.random.default_rng(1)
    h_od = rng.random((224, 224), dtype=np.float32) * 1.2

    on = student._tile_tensor(h_od, {**student.input_settings(MANIFEST), "standardise": True})
    off = student._tile_tensor(h_od, {**student.input_settings(MANIFEST), "standardise": False})
    assert not np.allclose(on, off), (
        "standardise makes no difference on this input, so the test above would pass "
        "even with the setting ignored - pick an input where it bites"
    )


# --- the transform's settings come from the manifest, never from a default ----


def test_settings_are_read_from_the_manifest():
    assert student.input_settings(MANIFEST) == {
        "gamma": 1.0, "invert": False, "standardise": True, "tile_px": 224, "mpp": 0.5,
    }


def test_a_manifest_missing_its_transform_is_refused_rather_than_guessed():
    """A default here would silently reproduce the G6 failure."""
    with pytest.raises(ValueError, match="does not record"):
        student.input_settings({"input": {"tile_px": 224, "mpp": 0.5}})
    with pytest.raises(ValueError, match="no `input` block"):
        student.input_settings({"classes": ["a", "b", "c"]})


# --- one white point per region, not per tile ---------------------------------


def test_the_white_point_is_taken_once_over_the_whole_region():
    """`export.py`'s rule, and departing from it would rescale every tile separately.

    A region with a bright half and a dark half: with one white point the two halves keep
    their different densities, which is the difference the model reads. Per-tile `I0`
    would normalise each against its own brightest pixels and flatten them together.
    """
    region = np.full((448, 448, 3), 210, np.uint8)
    region[:, 224:] = 120  # the dark half
    region[::4, ::4] = 40  # something for the p99 to find

    white = hchannel.white_point(region, percentile=99.0)
    h_od = hchannel.haematoxylin_od(region, white)
    left, right = h_od[:, :224].mean(), h_od[:, 224:].mean()
    assert right > left * 1.2, "one white point must preserve the two halves' difference"

    # ...and the per-tile alternative does not, which is why it is not used.
    per_tile = [
        hchannel.haematoxylin_od(half, hchannel.white_point(half, 99.0)).mean()
        for half in (region[:, :224], region[:, 224:])
    ]
    assert per_tile[1] < right, "per-tile I0 flattens the contrast the model reads"


# --- the comparison itself ----------------------------------------------------


def test_both_models_speak_the_same_three_classes():
    """The user's 'make sure the classes are same' requirement, enforced in code.

    BEETLE's five codes go through `labels.to_our_classes`; our model's head has three
    outputs. Both must land on the same vocabulary in the same order, or the confusion
    matrix's axes mean different things on each side.
    """
    every_beetle_code = np.arange(5, dtype=np.uint8).reshape(1, 5)
    ours = labels.to_our_classes(every_beetle_code)
    assert set(np.unique(ours).tolist()) <= {0, 1, 2, bcss.IGNORE}
    assert bcss.CLASS_NAMES == (
        "non_epithelium", "non_invasive_epithelium", "invasive_epithelium"
    )
    assert list(sixslides.__dict__)  # module imports cleanly with approach 1 on the path


def test_tile_majority_reduces_pixels_to_one_label_per_tile():
    pixels = np.zeros((448, 448), np.uint8)
    pixels[:224, :224] = bcss.INVASIVE          # top-left all invasive
    pixels[:224, 224:] = bcss.NON_INVASIVE      # top-right all in-situ
    pixels[224:, :224] = bcss.NON_EPITHELIUM
    # bottom-right: 60/40 split, so the majority is what must come out
    pixels[224:, 224:] = bcss.NON_INVASIVE
    pixels[224:, 224 : 224 + 90] = bcss.INVASIVE

    tiles = sixslides.tile_majority(pixels, 224)
    assert tiles.shape == (2, 2)
    assert tiles[0, 0] == bcss.INVASIVE
    assert tiles[0, 1] == bcss.NON_INVASIVE
    assert tiles[1, 0] == bcss.NON_EPITHELIUM
    assert tiles[1, 1] == bcss.NON_INVASIVE


def test_unlabelled_pixels_do_not_win_a_tile():
    """`IGNORE` is the absence of a statement and must never become a class."""
    pixels = np.full((224, 224), bcss.IGNORE, np.uint8)
    pixels[:10, :10] = bcss.NON_INVASIVE
    assert sixslides.tile_majority(pixels, 224)[0, 0] == bcss.NON_INVASIVE


def test_agreement_counts_add_up():
    ours = np.array([[0, 1], [2, 2]], np.uint8)
    theirs = np.array([[0, 1], [1, 2]], np.uint8)

    scores = sixslides.agreement(ours, theirs)
    assert scores["tiles"] == 4
    assert scores["tile_agreement"] == pytest.approx(0.75)

    matrix = np.array(scores["confusion"])
    assert matrix.sum() == ours.size, "every tile must land in exactly one cell"
    # Rows are the teacher: it said in-situ twice, and the student agreed once.
    assert matrix[bcss.NON_INVASIVE].sum() == 2
    assert scores["teacher_in_situ_student_invasive"] == 1
    assert scores["teacher_invasive_student_in_situ"] == 0


def test_agreement_refuses_mismatched_grids():
    with pytest.raises(ValueError, match="disagree"):
        sixslides.agreement(np.zeros((3, 3), np.uint8), np.zeros((3, 4), np.uint8))


def test_perfect_agreement_is_one_and_total_disagreement_is_zero():
    same = np.array([[1, 2], [0, 1]], np.uint8)
    assert sixslides.agreement(same, same)["tile_agreement"] == pytest.approx(1.0)
    assert sixslides.agreement(
        np.zeros((2, 2), np.uint8), np.full((2, 2), 2, np.uint8)
    )["tile_agreement"] == pytest.approx(0.0)


def test_summary_is_weighted_by_tiles_not_by_slide():
    """A 3x3 window must not carry the same weight as a 9x9 one."""
    big = {"agreement": {"tiles": 81, "tile_agreement": 1.0,
                         "teacher_in_situ_student_invasive": 0,
                         "teacher_invasive_student_in_situ": 0}, "seconds": 50.0}
    small = {"agreement": {"tiles": 9, "tile_agreement": 0.0,
                           "teacher_in_situ_student_invasive": 9,
                           "teacher_invasive_student_in_situ": 0}, "seconds": 5.0}

    summary = sixslides.summarise([big, small])
    assert summary["tiles"] == 90
    assert summary["tile_agreement"] == pytest.approx(81 / 90)
    # An unweighted mean would have said 0.5, which the small window does not deserve.
    assert summary["tile_agreement"] > 0.5
    assert summary["teacher_in_situ_student_invasive"] == 9


def test_summary_of_nothing_is_not_a_crash():
    assert sixslides.summarise([]) == {"cases": 0}


# --- the pictures -------------------------------------------------------------


def test_the_disagreement_picture_marks_only_the_tiles_that_differ():
    rgb = np.full((8, 8, 3), 200, np.uint8)
    ours = np.zeros((8, 8), np.uint8)
    theirs = np.zeros((8, 8), np.uint8)
    theirs[:4] = 2

    painted = sixslides.disagreement_picture(rgb, ours, theirs)
    assert not np.array_equal(painted[:4], rgb[:4]), "differing tiles must be marked"
    np.testing.assert_array_equal(painted[4:], rgb[4:])


def test_classify_region_refuses_a_region_too_small_for_one_tile():
    rgb = np.full((100, 100, 3), 200, np.uint8)
    with pytest.raises(ValueError, match="does not hold a single"):
        student.classify_region(rgb, None, MANIFEST)


def test_classify_region_refuses_anything_but_uint8_rgb():
    with pytest.raises(TypeError):
        student.classify_region(np.zeros((448, 448), np.uint8), None, MANIFEST)
    with pytest.raises(TypeError):
        student.classify_region(np.zeros((448, 448, 3), np.float32), None, MANIFEST)


# --- the outputs stay out of the BRACS measurement ----------------------------


def test_six_slide_results_are_not_written_where_bracs_results_are_read():
    """`compare.compare_all` and the training export both walk `RUNS_DIR`.

    A region cut from one of our own H&E slides carries no annotation saying it contains
    in-situ carcinoma. Pooling it into either would corrupt the one measurement this app
    exists to make - the same reason uploads live in their own tree.
    """
    assert config.SIXSLIDES_DIR != config.RUNS_DIR
    assert config.RUNS_DIR not in config.SIXSLIDES_DIR.parents
    assert config.SIXSLIDES_DIR.name == "sixslides"


# --- the pathologist's annotations --------------------------------------------
#
# `qpdata.py` reads QuPath's Java-serialised annotation files. It is the only source of
# human labels on a whole slide in this project, so a silent mis-read here would turn the
# one real evaluation into a fiction - and an EMPTY ground truth would score every model
# as perfect. Every test below guards that direction.


def test_a_non_qpdata_file_is_refused(tmp_path):
    from bracs_app import qpdata

    path = tmp_path / "not.qpdata"
    path.write_bytes(b"GeoJSON{}")
    with pytest.raises(ValueError, match="not a Java-serialised stream"):
        qpdata.parse(path)


def test_polygon_annotations_are_refused_rather_than_half_read(tmp_path):
    """A polygon file must fail loudly: a partial ground truth is worse than none."""
    from bracs_app import qpdata

    path = tmp_path / "poly.qpdata"
    path.write_bytes(b"\xac\xed\x00\x05" + b"junk qupath.lib.roi.PolygonROI more junk")
    with pytest.raises(ValueError, match="PolygonROI"):
        qpdata.parse(path)


def test_a_file_with_no_rectangles_is_refused(tmp_path):
    from bracs_app import qpdata

    path = tmp_path / "empty.qpdata"
    path.write_bytes(b"\xac\xed\x00\x05" + b"nothing useful here")
    with pytest.raises(ValueError, match="no RectangleROI"):
        qpdata.parse(path)


def test_the_real_bracs_annotations_parse_to_the_expected_lesions():
    """The actual file on disk, if it is there. Skipped rather than faked when absent."""
    from bracs_app import config, qpdata

    path = (config.BRACS_WSI_DIR / "annotations" / "test" / "test" / "Group_MT"
            / "Type_IC" / "BRACS_1284.qpdata")
    if not path.exists():
        pytest.skip("BRACS_1284.qpdata is not on this machine")

    found = qpdata.parse(path)
    assert len(found) == 16
    classes = {a.path_class for a in found}
    assert classes == {"DCIS-sure", "Malignant-sure", "UDH-sure"}
    # Every box inside a 22.6 x 20.5 mm slide, and none inverted.
    for a in found:
        assert 0 <= a.x < a.x2 < 90_000 and 0 <= a.y < a.y2 < 82_000
        assert a.width > 0 and a.height > 0


# --- the truth grid -----------------------------------------------------------


def test_unannotated_tissue_is_ignored_not_called_other_tissue():
    """The rule the whole evaluation rests on.

    BRACS boxes lesions only. Scoring unboxed tissue as class 0 would mark both models
    wrong for finding a normal duct that is genuinely there, and would flatter whichever
    model finds LESS epithelium.
    """
    from bracs_app import qpdata, wsi

    boxes = [qpdata.Annotation("DCIS-sure", 0, 0, 448, 448)]
    grid = wsi.truth_grid(boxes, (4, 4), (0, 0), 224, 1.0)

    assert grid[0, 0] == bcss.NON_INVASIVE
    assert grid[3, 3] == bcss.IGNORE, "outside every box must be unlabelled"
    assert (grid == bcss.IGNORE).sum() == 12


def test_udh_is_in_situ_epithelium_and_malignant_is_invasive():
    from bracs_app import wsi

    assert wsi.BRACS_TO_OURS["DCIS-sure"] == bcss.NON_INVASIVE
    assert wsi.BRACS_TO_OURS["UDH-sure"] == bcss.NON_INVASIVE
    assert wsi.BRACS_TO_OURS["Malignant-sure"] == bcss.INVASIVE
    # The two the project turns on must never collapse together.
    assert wsi.BRACS_TO_OURS["DCIS-sure"] != wsi.BRACS_TO_OURS["Malignant-sure"]


def test_an_uncertain_bracs_class_is_left_unlabelled():
    """`-non` classes are the pathologist saying 'probably'. Not ground truth."""
    from bracs_app import qpdata, wsi

    boxes = [qpdata.Annotation("DCIS-non", 0, 0, 448, 448)]
    grid = wsi.truth_grid(boxes, (2, 2), (0, 0), 224, 1.0)
    assert (grid == bcss.IGNORE).all()


def test_a_tile_is_annotated_only_if_its_centre_is_inside_the_box():
    """A looser rule would label the ring of tiles the pathologist drew round."""
    from bracs_app import qpdata, wsi

    # A box covering only the first 100 px: tile 0's centre is at 112, so outside.
    grid = wsi.truth_grid(
        [qpdata.Annotation("DCIS-sure", 0, 0, 100, 100)], (2, 2), (0, 0), 224, 1.0
    )
    assert (grid == bcss.IGNORE).all()

    # A box reaching past 112 catches it.
    grid = wsi.truth_grid(
        [qpdata.Annotation("DCIS-sure", 0, 0, 150, 150)], (2, 2), (0, 0), 224, 1.0
    )
    assert grid[0, 0] == bcss.NON_INVASIVE


def test_invasive_wins_where_two_boxes_overlap():
    """A field containing both is invasive for scoring purposes (guide Rule 5)."""
    from bracs_app import qpdata, wsi

    grid = wsi.truth_grid(
        [qpdata.Annotation("Malignant-sure", 0, 0, 448, 448),
         qpdata.Annotation("DCIS-sure", 0, 0, 448, 448)],
        (2, 2), (0, 0), 224, 1.0,
    )
    assert (grid == bcss.INVASIVE).all()


def test_scoring_uses_annotated_tiles_only():
    from bracs_app import wsi

    truth = np.full((4, 4), bcss.IGNORE, np.uint8)
    truth[0, 0] = bcss.NON_INVASIVE
    truth[0, 1] = bcss.INVASIVE
    ours = np.full((4, 4), bcss.INVASIVE, np.uint8)      # wrong on tile (0,0) only
    beetle = np.full((4, 4), bcss.NON_INVASIVE, np.uint8)  # wrong on tile (0,1) only

    grids = wsi.Grids(ours, beetle, truth, np.ones((4, 4), bool), 224, (0, 0), 0.5)
    scored = wsi.score(grids)

    assert scored["annotated_tiles"] == 2
    assert scored["ours"]["accuracy"] == pytest.approx(0.5)
    assert scored["beetle"]["accuracy"] == pytest.approx(0.5)
    # The 14 unlabelled tiles must not have counted for or against anyone.
    assert sum(sum(row) for row in scored["ours"]["confusion"]) == 2
    assert scored["ours"]["in_situ_called_invasive"] == 1
    assert scored["beetle"]["invasive_called_in_situ"] == 1


def test_scoring_with_nothing_annotated_says_so_rather_than_claiming_perfection():
    from bracs_app import wsi

    truth = np.full((3, 3), bcss.IGNORE, np.uint8)
    grids = wsi.Grids(truth.copy(), truth.copy(), truth, np.ones((3, 3), bool), 224, (0, 0), 0.5)
    scored = wsi.score(grids)
    assert scored["annotated_tiles"] == 0
    assert "note" in scored
    assert "ours" not in scored, "an unscoreable run must not report an accuracy"
