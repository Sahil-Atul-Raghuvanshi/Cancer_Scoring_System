"""What this pipeline must get right on its own, independent of approach 1.

Each test names the failure it prevents. Most of the failures here are silent ones - a
mask transformed differently from its image, an ignore pixel counted as stroma, a stale
shard reused - which all produce a model that trains, scores, and is wrong.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import augment
import classes
import segdata
import segexport
import segreport
import splits


# --- the mask must move with the image ----------------------------------------


def test_the_same_geometry_is_applied_to_image_and_mask():
    """Drawing twice would give the mask a different flip from its image.

    Nothing raises when that happens: both arrays are still 224x224, the loss still
    falls, and the model learns to put boundaries in the wrong place. This is the whole
    reason `sample_geometry` and `apply` are separate functions.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    # A mask that encodes position, so a wrong transform cannot look right by symmetry.
    mask = np.arange(64, dtype=np.uint8).reshape(8, 8)

    for seed in range(16):
        geometry = augment.sample_geometry(np.random.default_rng(seed))
        moved_image = geometry.apply(image)
        moved_mask = geometry.apply(mask)
        # Wherever a pixel of the image went, the mask's pixel went with it.
        for _ in range(5):
            y, x = int(rng.integers(0, 8)), int(rng.integers(0, 8))
            original = mask[y, x]
            positions = np.argwhere(moved_mask == original)
            assert len(positions) == 1
            py, px = positions[0]
            assert moved_image[py, px] == image[y, x]


def test_applying_a_geometry_never_invents_a_label():
    """Flips and 90-degree turns move pixels; other angles would interpolate classes."""
    mask = np.array([[0, 1], [2, classes.IGNORE]], dtype=np.uint8)
    before = set(np.unique(mask).tolist())
    for seed in range(20):
        geometry = augment.sample_geometry(np.random.default_rng(seed))
        assert set(np.unique(geometry.apply(mask)).tolist()) <= before


def test_augmentation_differs_between_epochs():
    """The tile model's per-index seed is epoch-independent; over 30 epochs that would
    flip every tile the same way every time and throw away most of the augmentation."""
    draws = {
        augment.sample_geometry(augment.seed_for(0, 7, epoch))
        for epoch in range(12)
    }
    assert len(draws) > 1, "tile 7 gets the same transform in every epoch"


def test_a_tile_index_is_reproducible_within_an_epoch():
    """Reproducible per (seed, index, epoch), so a run can be repeated exactly."""
    first = augment.sample_geometry(augment.seed_for(0, 3, 5))
    second = augment.sample_geometry(augment.seed_for(0, 3, 5))
    assert first == second


# --- the export ---------------------------------------------------------------


def test_the_spec_fingerprint_changes_with_every_setting_that_changes_a_tile():
    """The fingerprint is what makes the skip safe. A setting it ignores is a setting
    that could silently mix two kinds of tile into one manifest."""
    base = segexport.TileSpec()
    variants = [
        segexport.TileSpec(tile_px=256),
        segexport.TileSpec(mpp=0.25),
        segexport.TileSpec(min_labelled=0.5),
        segexport.TileSpec(stride_px=112),
        segexport.TileSpec(white_percentile=95.0),
    ]
    fingerprints = {base.fingerprint()} | {v.fingerprint() for v in variants}
    assert len(fingerprints) == len(variants) + 1


def test_a_region_is_skipped_only_when_its_shard_matches_and_its_files_exist(
    tmp_path, monkeypatch
):
    """Three ways a skip must be refused, and all three would otherwise be silent."""
    monkeypatch.setattr(segexport.paths, "TILES_DIR", tmp_path)
    monkeypatch.setattr(segexport.paths, "SHARDS_DIR", tmp_path / "shards")
    (tmp_path / "shards").mkdir(parents=True)
    (tmp_path / "images").mkdir()

    spec = segexport.TileSpec()
    tile = tmp_path / "images" / "t.png"
    tile.write_bytes(b"not really a png, but it exists")

    shard = {
        "settings_fingerprint": spec.fingerprint(),
        "rows": [{"image_path": "images/t.png", "mask_path": "images/t.png"}],
    }
    path = tmp_path / "shards" / "R.json"
    path.write_text(json.dumps(shard), encoding="utf-8")
    assert segexport.shard_is_current("R", spec)

    # 1. a different spec
    assert not segexport.shard_is_current("R", segexport.TileSpec(min_labelled=0.9))

    # 2. a missing tile
    tile.unlink()
    assert not segexport.shard_is_current("R", spec)

    # 3. a truncated shard from an interrupted run
    tile.write_bytes(b"back")
    path.write_text("{not json", encoding="utf-8")
    assert not segexport.shard_is_current("R", spec)


def test_no_shard_means_no_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(segexport.paths, "SHARDS_DIR", tmp_path)
    assert not segexport.shard_is_current("never-exported", segexport.TileSpec())


# --- the splits ---------------------------------------------------------------


def _rows(n: int, *, source: str, patient: str, institution: str) -> list[dict]:
    return [
        {
            "tile_id": f"{patient}_{i}", "patient": patient, "institution": institution,
            "source": source, "px_non_epithelium": 100,
            "px_non_invasive_epithelium": 10, "px_invasive_epithelium": 50,
            "px_unlabelled": 5,
        }
        for i in range(n)
    ]


def test_bcss_is_split_by_published_institution():
    rows = (
        _rows(4, source=classes.SOURCE_BCSS, patient="TCGA-A1-0001", institution="A1")
        + _rows(4, source=classes.SOURCE_BCSS, patient="TCGA-OL-0002", institution="OL")
    )
    train, test = splits.split(rows)
    assert {r["institution"] for r in train} == {"A1"}
    assert {r["institution"] for r in test} == {"OL"}


def test_bracs_is_split_by_patient_and_both_sides_are_populated():
    rows = _rows(2, source=classes.SOURCE_BCSS, patient="TCGA-OL-1", institution="OL")
    for case in range(40):
        rows += _rows(
            2, source=classes.SOURCE_DCIS, patient=f"BRACS_{case}",
            institution=classes.DCIS_INSTITUTION,
        )
    train, test = splits.split(rows)
    train_patients = {r["patient"] for r in train if r["source"] == classes.SOURCE_DCIS}
    test_patients = {r["patient"] for r in test if r["source"] == classes.SOURCE_DCIS}
    assert train_patients and test_patients
    assert not (train_patients & test_patients)
    share = len(test_patients) / (len(train_patients) + len(test_patients))
    assert 0.10 < share < 0.45, f"held-out share {share:.2f} is nowhere near the target"


def test_a_patients_side_does_not_move_when_other_patients_are_added():
    """A hash, not a shuffle: today's split must survive tomorrow's bigger export.

    With `random.sample(seed=0)` every assignment changes the moment a case is added, so
    two models both claiming seed 0 would be scored on different tiles.
    """
    small = [f"BRACS_{i}" for i in range(30)]
    large = small + [f"BRACS_{i}" for i in range(30, 200)]
    before = splits.dcis_test_patients(small)
    after = splits.dcis_test_patients(large)
    assert before == {p for p in after if p in set(small)}


def test_a_leak_is_an_assertion_not_a_warning():
    group = _rows(2, source=classes.SOURCE_BCSS, patient="TCGA-A1-1", institution="A1")
    with pytest.raises(AssertionError, match="appears in split"):
        splits.assert_no_leak(group, group)


def test_class_weights_are_per_pixel_not_per_tile():
    """Class 1 is ~14 % of tiles and ~0.15 % of pixels. Weighting by tiles would
    under-weight the rare class by two orders of magnitude, and the model would answer
    'stroma' everywhere while its loss fell nicely."""
    rows = [{
        "px_non_epithelium": 1_000_000,
        "px_non_invasive_epithelium": 1_000,
        "px_invasive_epithelium": 500_000,
        "px_unlabelled": 0,
    }]
    weights = splits.pixel_class_weights(rows)
    assert weights[classes.NON_INVASIVE] > weights[classes.NON_EPITHELIUM] * 100


def test_a_class_with_no_pixels_is_refused():
    rows = [{
        "px_non_epithelium": 10, "px_non_invasive_epithelium": 0,
        "px_invasive_epithelium": 10, "px_unlabelled": 0,
    }]
    with pytest.raises(ValueError, match="no labelled pixels"):
        splits.pixel_class_weights(rows)


# --- the report ---------------------------------------------------------------


def test_ignore_pixels_are_excluded_rather_than_crashing_or_counting():
    """`np.add.at` into a 3x3 with a truth value of 255 is an IndexError; counting them
    as class 0 would mark the model wrong for finding unannotated epithelium."""
    truth = np.full((4, 4), classes.IGNORE, np.uint8)
    truth[0, 0] = classes.INVASIVE
    predicted = np.full((4, 4), classes.NON_EPITHELIUM, np.uint8)
    predicted[0, 0] = classes.INVASIVE

    matrix = segreport.confusion(truth, predicted)
    assert matrix.sum() == 1, "only the one annotated pixel may be counted"
    assert matrix[classes.INVASIVE, classes.INVASIVE] == 1


def test_a_fully_unlabelled_stack_does_not_claim_perfection():
    truth = np.full((2, 4, 4), classes.IGNORE, np.uint8)
    scored = segreport.evaluate(truth, truth.copy())
    assert scored["labelled_pixels"] == 0
    assert scored["accuracy"] is None


def test_dice_is_one_when_the_prediction_is_right_and_zero_when_inverted():
    truth = np.zeros((1, 8, 8), np.uint8)
    truth[0, :4] = classes.INVASIVE
    same = segreport.evaluate(truth, truth.copy())
    assert same["per_class"]["invasive_epithelium"]["dice"] == pytest.approx(1.0)

    flipped = truth.copy()
    flipped[0, :4] = classes.NON_EPITHELIUM
    flipped[0, 4:] = classes.INVASIVE
    worst = segreport.evaluate(truth, flipped)
    assert worst["per_class"]["invasive_epithelium"]["dice"] == pytest.approx(0.0)


def test_boundary_f1_is_the_metric_area_dice_cannot_be():
    """A shape dilated by a few pixels keeps a high Dice and loses its boundary F1.

    That is exactly the error a 224 px tile model makes, and the reason this pipeline
    reports both numbers rather than Dice alone.
    """
    truth = np.zeros((64, 64), np.uint8)
    truth[20:44, 20:44] = classes.INVASIVE

    grown = np.zeros((64, 64), np.uint8)
    grown[14:50, 14:50] = classes.INVASIVE  # 6 px too large on every side

    matrix = segreport.confusion(truth, grown)
    area_dice = segreport.dice(matrix, classes.INVASIVE)
    boundary = segreport.boundary_f1(truth, grown, classes.INVASIVE, tolerance=3)

    assert area_dice > 0.6, "area Dice stays respectable on a modest dilation"
    assert boundary < 0.2, "boundary F1 must collapse - the edge is 6 px out"


def test_boundary_f1_is_one_for_an_exact_match():
    truth = np.zeros((32, 32), np.uint8)
    truth[8:24, 8:24] = classes.NON_INVASIVE
    assert segreport.boundary_f1(
        truth, truth.copy(), classes.NON_INVASIVE, tolerance=3
    ) == pytest.approx(1.0)


def test_the_two_label_sources_are_never_averaged():
    """Nine human pixels must not be diluted by thousands of BEETLE's."""
    truth = np.stack([
        np.full((4, 4), classes.NON_INVASIVE, np.uint8),
        np.full((4, 4), classes.NON_INVASIVE, np.uint8),
    ])
    predicted = np.stack([
        np.full((4, 4), classes.NON_INVASIVE, np.uint8),   # bcss: right
        np.full((4, 4), classes.INVASIVE, np.uint8),        # bracs: wrong
    ])
    scored = segreport.evaluate_by_source(
        truth, predicted, [classes.SOURCE_BCSS, classes.SOURCE_DCIS]
    )
    human = scored["by_source"][classes.SOURCE_BCSS]
    teacher = scored["by_source"][classes.SOURCE_DCIS]
    assert human["per_class"]["non_invasive_epithelium"]["dice"] == pytest.approx(1.0)
    assert teacher["per_class"]["non_invasive_epithelium"]["dice"] == pytest.approx(0.0)
    assert human["labels_drawn_by"] == "pathologists"
    assert "reviewed by nobody" in teacher["labels_drawn_by"]


def test_the_two_cells_that_matter_are_named():
    truth = np.full((1, 2, 2), classes.NON_INVASIVE, np.uint8)
    predicted = np.full((1, 2, 2), classes.INVASIVE, np.uint8)
    scored = segreport.evaluate(truth, predicted)
    assert scored["in_situ_called_invasive"] == 4
    assert scored["invasive_called_in_situ"] == 0


# --- the reporting must not misattribute labels ------------------------------


def test_the_headline_never_claims_pathologists_when_there_are_none():
    """It said "vs pathologists" unconditionally once, on a subset of BEETLE tiles.

    That is the single most misleading thing this report could print: the entire value of
    the `bcss` block is that a person drew those labels, so a teacher-labelled number
    under a human-labelled claim would invert the meaning of the result.
    """
    truth = np.full((1, 4, 4), classes.NON_INVASIVE, np.uint8)
    teacher_only = segreport.evaluate_by_source(
        truth, truth.copy(), [classes.SOURCE_DCIS]
    )
    line = segreport.headline(teacher_only)
    assert "pathologists" not in line
    assert "NO human-labelled tiles" in line

    both = segreport.evaluate_by_source(
        np.concatenate([truth, truth]), np.concatenate([truth, truth]),
        [classes.SOURCE_BCSS, classes.SOURCE_DCIS],
    )
    assert segreport.headline(both).startswith("vs pathologists")


def test_a_validation_limit_spreads_across_the_set_rather_than_taking_a_prefix():
    """A prefix was the wrong sample and it bit once.

    Tile ids sort with `BRACS_...` before `TCGA-...`, so "the first 100 held-out tiles"
    was 100 BEETLE-labelled tiles and not one human-labelled tile - the validation curve
    was measuring agreement with the teacher while appearing to measure accuracy.
    """
    import segtrain

    rows = (
        [{"tile_id": f"BRACS_{i}", "source": classes.SOURCE_DCIS} for i in range(300)]
        + [{"tile_id": f"TCGA_{i}", "source": classes.SOURCE_BCSS} for i in range(100)]
    )
    picked = segtrain._spread(rows, 40)
    assert len(picked) == 40
    sources = {row["source"] for row in picked}
    assert sources == {classes.SOURCE_BCSS, classes.SOURCE_DCIS}, (
        "a spread must reach the human-labelled tail, which a prefix never would"
    )
    # Deterministic, so the epoch-to-epoch curve is comparable.
    assert segtrain._spread(rows, 40) == picked
    # And asking for more than exists is not an error.
    assert segtrain._spread(rows, 10_000) == rows


# --- which source may teach which class ---------------------------------------


def test_bracs_may_only_teach_the_in_situ_class():
    """BRACS regions were selected for being DCIS-heavy, so that is all they attest to.

    Its other two classes are a side effect of that selection and they are not small:
    58.0 M non-epithelium and 16.0 M invasive pixels, all of them BEETLE's predictions,
    against 385 M and 267 M drawn by hand in BCSS. A DCIS-selected, model-labelled
    minority should not get a vote on what invasive carcinoma looks like.
    """
    mask = np.array(
        [
            [classes.NON_EPITHELIUM, classes.NON_INVASIVE],
            [classes.INVASIVE, classes.IGNORE],
        ],
        dtype=np.uint8,
    )

    kept = classes.apply_source_authority(mask, classes.SOURCE_DCIS)
    assert kept[0, 1] == classes.NON_INVASIVE, "in-situ is what BRACS is here for"
    assert kept[0, 0] == classes.IGNORE
    assert kept[1, 0] == classes.IGNORE
    assert kept[1, 1] == classes.IGNORE

    # BCSS is the human-labelled source and keeps everything, untouched.
    assert np.array_equal(classes.apply_source_authority(mask, classes.SOURCE_BCSS), mask)


def test_blanked_pixels_become_ignore_rather_than_a_class():
    """The one thing that must not happen: a blanked pixel silently becoming stroma.

    `IGNORE` already means "no statement" everywhere downstream - the loss skips it, the
    Dice zeroes it, the report excludes it - which is exactly why the rule is expressed
    this way. Folding these into class 0 would teach the model that every duct BEETLE
    saw is surrounded by *confirmed* stroma, and would inflate class 0's support by 58 M
    pixels of somebody's guess.
    """
    mask = np.full((8, 8), classes.INVASIVE, dtype=np.uint8)
    out = classes.apply_source_authority(mask, classes.SOURCE_DCIS)

    assert set(np.unique(out)) == {classes.IGNORE}
    assert classes.NON_EPITHELIUM not in np.unique(out)


def test_the_rule_keeps_the_tile_rather_than_dropping_it():
    """The pixel-level advantage over the tile model, pinned.

    A tile classifier has to discard a whole tile whose label it distrusts. Here a BRACS
    tile holding a duct in stroma keeps teaching in-situ epithelium and only loses the
    stroma pixels - so the rule costs supervision it should not want, and none that it
    should.
    """
    mask = np.full((16, 16), classes.NON_EPITHELIUM, dtype=np.uint8)
    mask[4:12, 4:12] = classes.NON_INVASIVE

    out = classes.apply_source_authority(mask, classes.SOURCE_DCIS)
    assert (out == classes.NON_INVASIVE).sum() == 64, "the duct still supervises"
    assert (out == classes.IGNORE).sum() == 16 * 16 - 64


def test_an_unknown_source_keeps_every_class():
    """Adding a third dataset must fail open, not silently blank all of it."""
    mask = np.array([[0, 1, 2]], dtype=np.uint8)
    assert np.array_equal(classes.apply_source_authority(mask, "some_new_dataset"), mask)


def test_class_weights_count_only_the_pixels_that_will_supervise():
    """Weighting against counts the loss never sees balances against nothing.

    The manifest records what was *annotated*; after the rule, some of that no longer
    supervises. If the weights were computed from the raw counts, class 0 would be
    weighted as though it had 58 M more pixels of support than it does.
    """
    rows = [
        {
            "source": classes.SOURCE_DCIS,
            "px_non_epithelium": 900,
            "px_non_invasive_epithelium": 100,
            "px_invasive_epithelium": 500,
            "px_unlabelled": 0,
        },
        {
            "source": classes.SOURCE_BCSS,
            "px_non_epithelium": 900,
            "px_non_invasive_epithelium": 100,
            "px_invasive_epithelium": 500,
            "px_unlabelled": 0,
        },
    ]

    gated = splits.pixel_class_weights(rows, source_authority=True)
    ungated = splits.pixel_class_weights(rows, source_authority=False)
    assert gated != ungated

    # The invariant, stated as an equivalence rather than as a direction: weighting the
    # real rows under the rule must equal weighting rows that already contain only the
    # pixels the rule keeps. Asserting a *direction* would be guessing - blanking BRACS's
    # other two classes makes in-situ relatively more common, so its weight actually
    # falls, which is easy to get backwards and says nothing either way.
    equivalent = [
        {
            "source": classes.SOURCE_DCIS,
            "px_non_epithelium": 0,
            "px_non_invasive_epithelium": 100,
            "px_invasive_epithelium": 0,
            "px_unlabelled": 0,
        },
        rows[1],
    ]
    assert gated == pytest.approx(
        splits.pixel_class_weights(equivalent, source_authority=False)
    )


def test_the_census_moves_blanked_pixels_into_unlabelled():
    """The census has to sum to the tile area, or it is describing a different dataset."""
    rows = [{
        "source": classes.SOURCE_DCIS,
        "px_non_epithelium": 30,
        "px_non_invasive_epithelium": 50,
        "px_invasive_epithelium": 20,
        "px_unlabelled": 100,
    }]

    census = segdata.pixel_census(rows, source_authority=True)
    assert census["non_invasive_epithelium"] == 50
    assert census["non_epithelium"] == 0
    assert census["invasive_epithelium"] == 0
    assert census["unlabelled"] == 100 + 30 + 20
    assert sum(census.values()) == 200
