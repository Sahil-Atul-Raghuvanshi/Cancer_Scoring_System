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
