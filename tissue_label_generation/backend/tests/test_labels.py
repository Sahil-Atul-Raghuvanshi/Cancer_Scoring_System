"""The tests that would have caught the inverted label map.

Everything here guards one class of failure: a mapping that is wrong in a way that still
produces plausible output. None of these would fail loudly in production - a swapped
code, a merged class and a lost ignore region all yield a mask, tiles, and a number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bracs_app import config, labels  # noqa: E402

config.install_approach1_path()
import bcss  # noqa: E402


def test_release_codes_are_the_ones_the_app_was_written_against():
    """The check that `beetle_teacher_resnet18` deferred, now done against the archive.

    Approach 4a records codes 2 and 3 as invasive and non-invasive respectively, read
    off the paper. The released `dataset.json` has them the other way round. If a future
    release moves them again, this fails rather than inverting the training set.
    """
    verified = labels.codes_from_dataset_json(
        {
            "unannotated": 0,
            "other": 1,
            "non-invasive epithelium": 2,
            "invasive epithelium": 3,
            "necrosis": 4,
        }
    )
    assert verified["non_invasive_epithelium"] == 2
    assert verified["invasive_epithelium"] == 3


def test_the_papers_ordering_is_refused():
    """The specific wrong mapping, refused by name rather than by luck."""
    with pytest.raises(ValueError, match="inverted"):
        labels.codes_from_dataset_json(
            {
                "unannotated": 0,
                "other": 1,
                "invasive epithelium": 2,
                "non-invasive epithelium": 3,
                "necrosis": 4,
            }
        )


def test_background_and_unannotated_are_the_same_channel():
    """`dataset.json` says `unannotated`, the checkpoint's `init_args` says `background`."""
    assert labels.codes_from_dataset_json(
        {
            "background": 0,
            "other": 1,
            "non-invasive epithelium": 2,
            "invasive epithelium": 3,
            "necrosis": 4,
        }
    ) == labels.BEETLE_CODES


def test_an_unknown_class_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="no translation"):
        labels.codes_from_dataset_json({"unannotated": 0, "lobular_carcinoma": 1})


def test_each_beetle_class_lands_where_it_should_in_our_three():
    """The end-to-end statement of intent, through approach 1's own remap.

    This is the assertion that matters: it goes through `to_bcss_codes` and then
    `bcss.remap`, so it fails if either the BCSS code chosen is wrong or approach 1
    later changes what that code means.
    """
    mask = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    ours = labels.to_our_classes(mask)

    assert ours[0, 0] == bcss.IGNORE, "unannotated must carry no label at all"
    assert ours[0, 1] == bcss.NON_EPITHELIUM
    assert ours[0, 2] == bcss.NON_INVASIVE, "code 2 is in-situ epithelium"
    assert ours[1, 0] == bcss.INVASIVE, "code 3 is invasive carcinoma"
    assert ours[1, 1] == bcss.NON_EPITHELIUM, "necrosis is not epithelium of any kind"


def test_in_situ_and_invasive_do_not_collapse():
    """The 5-class BCSS merge, which this project cannot accept, restated for BEETLE.

    Approach 1 has this test on its own mapping. It is repeated here because a second
    label source is a second chance to make the same merge, and the consequence is the
    same: the boundary the region model exists to draw would stop existing.
    """
    in_situ = labels.to_our_classes(np.array([[2]], dtype=np.uint8))[0, 0]
    invasive = labels.to_our_classes(np.array([[3]], dtype=np.uint8))[0, 0]
    assert in_situ != invasive
    assert bcss.GT_CODES[labels.BCSS_TARGETS["non_invasive_epithelium"]] == bcss.GT_CODES["dcis"]
    assert bcss.GT_CODES[labels.BCSS_TARGETS["invasive_epithelium"]] == bcss.GT_CODES["tumor"]


def test_written_masks_use_only_codes_bcss_knows():
    """A code outside the 22 would make `bcss.remap` raise, four screens later."""
    every = np.arange(5, dtype=np.uint8).reshape(1, 5)
    written = labels.to_bcss_codes(every)
    assert written.dtype == np.uint16, "the BCSS release ships uint16 masks"
    assert set(np.unique(written).tolist()) <= set(bcss.GT_CODES.values())


def test_a_code_the_teacher_cannot_emit_is_refused():
    with pytest.raises(ValueError, match="does not use"):
        labels.to_bcss_codes(np.array([[7]], dtype=np.uint8))


def test_a_float_mask_is_refused():
    """An argmax that arrived as float has been through an interpolation somewhere."""
    with pytest.raises(TypeError):
        labels.to_bcss_codes(np.zeros((2, 2), dtype=np.float32))


def test_every_beetle_class_has_a_bcss_target():
    assert set(labels.BCSS_TARGETS) == set(labels.BEETLE_CODES)
    assert set(labels.CHANNEL_NAMES) == set(labels.BEETLE_CODES)


def test_channel_names_are_in_channel_order():
    """`CHANNEL_NAMES[i]` must be the class the network's channel `i` predicts."""
    for index, name in enumerate(labels.CHANNEL_NAMES):
        assert labels.BEETLE_CODES[name] == index


# --- the epithelium-union rule -------------------------------------------------
#
# Fix 1: the ROI consensus decides the lesion, BEETLE decides only where the
# epithelium is. Everything here guards a failure that would still produce a plausible
# mask - a collapse that missed one of the two classes, a re-code that silently moved
# stroma, or a tree collapsing onto a class its consensus cannot vouch for.


def test_epithelium_union_writes_both_carcinoma_classes_as_one_code():
    """The whole of Fix 1, in one assertion.

    BEETLE's codes 2 (non-invasive) and 3 (invasive) must land on the same BCSS code,
    and the three non-epithelium classes must be untouched - a collapse that also
    swallowed `necrosis` or `other` would inflate class 1 with stroma, which is the
    failure the served model already shows from the other direction.
    """
    every = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
    written = labels.to_bcss_codes(every, epithelium_target="dcis")

    assert written[0, 2] == bcss.GT_CODES["dcis"]
    assert written[0, 3] == bcss.GT_CODES["dcis"]
    assert written[0, 0] == bcss.GT_CODES["outside_roi"]
    assert written[0, 1] == bcss.GT_CODES["stroma"]
    assert written[0, 4] == bcss.GT_CODES["necrosis_or_debris"]


def test_epithelium_union_can_point_at_invasive_for_an_ic_region():
    """The mirror. An IC consensus makes BEETLE's in-situ prediction invasive."""
    both = np.array([[2, 3]], dtype=np.uint8)
    ours = labels.to_our_classes(both, epithelium_target="tumor")
    assert ours.tolist() == [[2, 2]]


def test_epithelium_union_is_off_unless_asked_for():
    """The pre-Fix-1 rule must survive verbatim, or `--split-epithelium` is a lie."""
    both = np.array([[2, 3]], dtype=np.uint8)
    assert labels.to_our_classes(both).tolist() == [[1, 2]]


def test_recode_epithelium_equals_collapsing_from_the_teacher_directly():
    """The equivalence `relabel_run` rests on, asserted over every BEETLE code.

    `relabel_run` re-codes a mask already on disk rather than re-running a 25-second
    segmentation, and that shortcut is only safe because `BCSS_TARGETS` is injective on
    BEETLE's five classes. If a future mapping sent two BEETLE classes to one code
    outside the epithelium pair, this fails - and it fails here rather than as a
    training set that quietly relabelled necrosis.
    """
    every = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=np.uint8)

    for non_invasive, epithelium in (
        ("dcis", "dcis"),
        ("dcis", "tumor"),
        ("normal_acinus_or_duct", "normal_acinus_or_duct"),
        ("normal_acinus_or_duct", "tumor"),
    ):
        direct = labels.to_bcss_codes(every, epithelium_target=epithelium)
        two_step = labels.recode_epithelium(
            labels.to_bcss_codes(every, non_invasive_target=non_invasive),
            epithelium_target=epithelium,
            non_invasive_target=non_invasive,
        )
        assert np.array_equal(direct, two_step), (non_invasive, epithelium)


def test_recode_epithelium_refuses_a_mask_it_cannot_have_written():
    """A mask from somewhere else, or one already half re-coded, is not guessed at."""
    stray = np.array([[bcss.GT_CODES["fat"]]], dtype=np.uint16)
    with pytest.raises(ValueError, match="cannot have written"):
        labels.recode_epithelium(stray, epithelium_target="dcis")


def test_recode_epithelium_leaves_a_normal_tree_mask_alone_where_it_should():
    """Codes 13 and 1 both carry epithelium; 2 and 11 do not and must not move."""
    mask = np.array([[
        bcss.GT_CODES["normal_acinus_or_duct"],
        bcss.GT_CODES["tumor"],
        bcss.GT_CODES["stroma"],
        bcss.GT_CODES["necrosis_or_debris"],
    ]], dtype=np.uint16)
    out = labels.recode_epithelium(
        mask,
        epithelium_target="normal_acinus_or_duct",
        non_invasive_target="normal_acinus_or_duct",
    )
    assert out.tolist() == [[
        bcss.GT_CODES["normal_acinus_or_duct"],
        bcss.GT_CODES["normal_acinus_or_duct"],
        bcss.GT_CODES["stroma"],
        bcss.GT_CODES["necrosis_or_debris"],
    ]]


def test_every_tree_collapses_onto_a_class_its_consensus_corroborates():
    """The table invariant, over the whole table.

    `pipeline.run_region` asserts this for the region it is writing; this asserts it for
    every tree at once, so a seventh tree added with a mismatched pair fails in the test
    suite rather than by contributing an empty class. A tree collapsing onto a class
    outside `teaches` would manufacture the unadjudicated label `teaches` exists to keep
    out, and `datasets.by_source_authority` would then drop every one of its tiles in
    silence.
    """
    for name, tree in config.ROI_TREES.items():
        if tree.epithelium_code is None:
            continue
        collapsed = int(bcss.remap(
            np.full((1, 1), bcss.GT_CODES[tree.epithelium_code], dtype=np.uint16)
        )[0, 0])
        assert collapsed in tree.teaches, (
            f"{name} collapses epithelium onto class {collapsed}, outside "
            f"teaches={sorted(tree.teaches)}"
        )


def test_the_carcinoma_trees_do_collapse_and_point_opposite_ways():
    """Guards the one thing a typo here would silently undo: the rule being on at all."""
    assert config.ROI_TREES["dcis"].epithelium_code == "dcis"
    assert config.ROI_TREES["ic"].epithelium_code == "tumor"
    assert config.ROI_TREES["bach_insitu"].epithelium_code == "dcis"
    assert config.ROI_TREES["bach_invasive"].epithelium_code == "tumor"
