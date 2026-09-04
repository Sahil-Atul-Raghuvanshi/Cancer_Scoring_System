"""This folder's transform must equal approach 1's, bit for bit.

**This is the most important test here, and the reason a standalone reimplementation is
defensible at all.** `bcss_bracs_hchannel_resnet18/src/export.py` warns in its own docstring:

    "A second exporter is exactly where a stray `skimage.color.rgb2hed` gets introduced,
    and the failure looks like a model problem for a week."

That warning is correct, and this folder wrote a second one anyway - because it must run
without approach 1 present. The way to have both is to prove the equivalence rather than
assume it: approach 1 is imported **here only**, and skipped when it is absent, so the
runtime code keeps its independence while any machine holding both can check them.

If one of these fails, the two models are not trained on the same input and every
comparison between them is meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
APPROACH1 = WORKSPACE / "bcss_bracs_hchannel_resnet18" / "src"
DEMO_BACKEND = WORKSPACE / "Breast_Cancer_IHC_Tissue_Scoring_Demo" / "backend"

approach1 = pytest.importorskip  # noqa: N816 - readability at the call site below


def _load_approach1():
    """Import approach 1's modules, or skip. Never imported by anything but this file."""
    if not (APPROACH1 / "hchannel.py").is_file():
        pytest.skip("bcss_bracs_hchannel_resnet18 is not on this machine")
    for path in (str(APPROACH1), str(DEMO_BACKEND)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import bcss as a1_bcss
        import export as a1_export
        import hchannel as a1_hchannel
    except Exception as error:  # noqa: BLE001 - a broken approach 1 is not our failure
        pytest.skip(f"approach 1 present but not importable: {error}")
    return a1_bcss, a1_export, a1_hchannel


def _rgb(seed: int = 0, size: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


# --- the transform ------------------------------------------------------------


def test_the_stain_matrix_is_the_same_constant():
    """Hard-coded here, derived there. They must resolve to the same numbers."""
    _load_approach1()
    from app.common.stains import RUIFROK_HDAB

    import hstain

    np.testing.assert_array_equal(hstain.RUIFROK_HDAB, RUIFROK_HDAB)


def test_white_point_is_identical():
    _, _, a1_hchannel = _load_approach1()
    import hstain

    rgb = _rgb()
    assert hstain.white_point(rgb) == a1_hchannel.white_point(rgb)


def test_the_haematoxylin_channel_is_bit_for_bit_identical():
    """The one that would silently poison everything if it drifted."""
    _, _, a1_hchannel = _load_approach1()
    import hstain

    for seed in range(4):
        rgb = _rgb(seed)
        white = hstain.white_point(rgb)
        np.testing.assert_array_equal(
            hstain.haematoxylin_od(rgb, white),
            a1_hchannel.haematoxylin_od(rgb, white),
        )


def test_quantise_and_dequantise_are_identical():
    _, _, a1_hchannel = _load_approach1()
    import hstain

    h_od = np.random.default_rng(1).random((64, 64), dtype=np.float32) * 2.0 - 0.25
    stored = hstain.quantise(h_od)
    np.testing.assert_array_equal(stored, a1_hchannel.quantise(h_od))
    np.testing.assert_array_equal(
        hstain.dequantise(stored), a1_hchannel.dequantise(stored)
    )


def test_the_model_input_is_identical_across_every_setting():
    """Including `standardise`, which is the setting that actually drifted once before."""
    _, _, a1_hchannel = _load_approach1()
    import hstain

    stored = np.random.default_rng(2).integers(0, 256, (64, 64), dtype=np.uint8)
    for standardise in (False, True):
        for gamma in (1.0, 2.5):
            for invert in (False, True):
                np.testing.assert_array_equal(
                    hstain.from_stored(
                        stored, gamma=gamma, invert=invert, standardise=standardise
                    ),
                    a1_hchannel.from_stored(
                        stored, gamma=gamma, invert=invert, standardise=standardise
                    ),
                    err_msg=f"standardise={standardise} gamma={gamma} invert={invert}",
                )


def test_resampling_is_identical():
    """BOX on intensities, NEAREST on labels - and the same rounding of the new size."""
    _, a1_export, _ = _load_approach1()
    import hstain

    rgb = _rgb(3, size=101)
    mask = np.random.default_rng(3).integers(0, 22, (101, 101), dtype=np.uint8)
    for source, target in ((0.25, 0.5), (0.2521, 0.5), (0.5, 0.5)):
        ours = hstain.resample(rgb, mask, source_mpp=source, target_mpp=target)
        theirs = a1_export.resample(rgb, mask, source_mpp=source, target_mpp=target)
        np.testing.assert_array_equal(ours[0], theirs[0])
        np.testing.assert_array_equal(ours[1], theirs[1])


# --- the label mapping --------------------------------------------------------


def test_the_class_mapping_is_identical():
    """Two definitions of class 1 is the failure this project cannot afford."""
    a1_bcss, _, _ = _load_approach1()
    import classes

    assert classes.GT_CODES == a1_bcss.GT_CODES
    assert classes.TO_CLASS == a1_bcss.TO_CLASS
    assert classes.IGNORE_CODES == a1_bcss.IGNORE_CODES
    assert classes.IGNORE == a1_bcss.IGNORE
    assert classes.CLASS_NAMES == a1_bcss.CLASS_NAMES
    assert classes.TEST_INSTITUTIONS == a1_bcss.TEST_INSTITUTIONS
    assert (classes.NON_EPITHELIUM, classes.NON_INVASIVE, classes.INVASIVE) == (
        a1_bcss.NON_EPITHELIUM, a1_bcss.NON_INVASIVE, a1_bcss.INVASIVE
    )


def test_remap_is_identical_on_every_code():
    a1_bcss, _, _ = _load_approach1()
    import classes

    every = np.arange(22, dtype=np.uint8).reshape(1, 22)
    np.testing.assert_array_equal(classes.remap(every), a1_bcss.remap(every))


def test_barcode_parsing_is_identical():
    a1_bcss, _, _ = _load_approach1()
    import classes

    name = "TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png"
    assert classes.parse_bcss(name) == a1_bcss.parse_roi(name)
    assert classes.slide_key(Path(name).stem) == a1_bcss.Region(
        roi_id=Path(name).stem, slide_id="x", institution="A1",
        image=Path("i"), mask=Path("m"),
    ).slide_key


# --- the augmentation ---------------------------------------------------------


def test_the_geometric_draw_matches_approach_ones():
    """Same three draws in the same order, so the two models see the same augmentation.

    It matters only for comparability - but comparability is the whole reason these two
    models can be put beside each other and a difference attributed to the architecture.
    """
    _load_approach1()
    import datasets as a1_datasets

    import augment

    tile = np.arange(16, dtype=np.uint8).reshape(4, 4)
    for index in range(8):
        theirs = a1_datasets.geometric(
            tile.copy(), np.random.default_rng((0, index))
        )
        geometry = augment.sample_geometry(np.random.default_rng((0, index)))
        np.testing.assert_array_equal(geometry.apply(tile.copy()), theirs)


def test_the_jitter_ranges_match():
    _load_approach1()
    import datasets as a1_datasets

    import augment

    ours, theirs = augment.Jitter(), a1_datasets.Jitter()
    assert (ours.alpha, ours.beta, ours.gamma) == (theirs.alpha, theirs.beta, theirs.gamma)
    for index in range(5):
        assert ours.sample(np.random.default_rng(index)) == theirs.sample(
            np.random.default_rng(index)
        )
