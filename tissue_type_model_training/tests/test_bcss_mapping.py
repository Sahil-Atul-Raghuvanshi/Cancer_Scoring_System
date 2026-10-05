"""The label mapping, and the merge trap it exists to make impossible.

These are gate G1 in executable form. The trap they guard against is not
hypothetical: a widely-used five-class version of BCSS folds `dcis` into `tumor`, it
is the default in most tutorials, and adopting it would delete the one distinction
this whole model is being built to draw. A comment saying "use the raw codes" would
not survive contact with a tutorial. A failing test will.
"""

from __future__ import annotations

import numpy as np
import pytest

import bcss


def test_every_code_is_mapped_or_ignored():
    """No code falls through. A fall-through silently discards a tissue type."""
    assigned = set(bcss.TO_CLASS) | set(bcss.IGNORE_CODES)
    assert set(bcss.GT_CODES.values()) == assigned
    assert not set(bcss.TO_CLASS) & set(bcss.IGNORE_CODES)


def test_tumor_and_dcis_are_different_classes():
    """**The merge trap.** Invasive carcinoma and in-situ carcinoma are not one class.

    If this ever passes by accident - someone "simplifying" the mapping to four
    classes, say - the model still trains, still reports a fine accuracy, and has
    stopped being able to answer the only question it was built for.
    """
    assert bcss.TO_CLASS[bcss.GT_CODES["tumor"]] == bcss.INVASIVE
    assert bcss.TO_CLASS[bcss.GT_CODES["dcis"]] == bcss.NON_INVASIVE
    assert bcss.TO_CLASS[bcss.GT_CODES["tumor"]] != bcss.TO_CLASS[bcss.GT_CODES["dcis"]]


def test_normal_ducts_are_not_invasive():
    """A normal duct is epithelium, and it is not cancer. Excluded, like DCIS."""
    assert bcss.TO_CLASS[bcss.GT_CODES["normal_acinus_or_duct"]] == bcss.NON_INVASIVE


def test_angioinvasion_counts_as_invasive():
    """Tumour inside a vessel has left the duct, which is what invasive means here."""
    assert bcss.TO_CLASS[bcss.GT_CODES["angioinvasion"]] == bcss.INVASIVE


def test_fat_and_stroma_collapse_together():
    """Three classes, not eight: everything excluded is excluded for one reason."""
    for name in ("fat", "stroma", "necrosis_or_debris", "lymphocytic_infiltrate"):
        assert bcss.TO_CLASS[bcss.GT_CODES[name]] == bcss.NON_EPITHELIUM


def test_outside_roi_is_ignored_not_a_class():
    """Unannotated background carries zero weight - the authors are explicit.

    Folding code 0 into `non_epithelium` would teach the model that unannotated
    background is stroma, and inflate every accuracy number by the share of the
    dataset that was never annotated at all.
    """
    for name in ("outside_roi", "exclude", "undetermined"):
        assert bcss.GT_CODES[name] in bcss.IGNORE_CODES
        assert bcss.GT_CODES[name] not in bcss.TO_CLASS


def test_remap_produces_ignore_for_unlabelled():
    raw = np.array([[0, 1, 2], [20, 7, 9], [13, 19, 15]], dtype=np.uint8)
    out = bcss.remap(raw)

    assert out[0, 0] == bcss.IGNORE          # outside_roi
    assert out[0, 1] == bcss.INVASIVE        # tumor
    assert out[0, 2] == bcss.NON_EPITHELIUM  # stroma
    assert out[1, 0] == bcss.NON_INVASIVE    # dcis
    assert out[1, 1] == bcss.IGNORE          # exclude
    assert out[1, 2] == bcss.NON_EPITHELIUM  # fat
    assert out[2, 0] == bcss.NON_INVASIVE    # normal_acinus_or_duct
    assert out[2, 1] == bcss.INVASIVE        # angioinvasion
    assert out[2, 2] == bcss.IGNORE          # undetermined


def test_remap_refuses_codes_outside_the_release():
    """A code above 21 means this is not the raw release. Fail here, not three
    notebooks later in a confusion matrix."""
    with pytest.raises(ValueError, match="not in gtruth_codes"):
        bcss.remap(np.array([[1, 200]], dtype=np.uint8))


def test_remap_refuses_a_colour_rendering():
    """Some mirrors ship a colourised mask. Those are pictures of labels, not labels."""
    coloured = np.zeros((2, 2, 3), dtype=np.uint8)
    coloured[..., 0] = 1
    coloured[..., 1] = 2
    with pytest.raises(ValueError, match="colour rendering"):
        bcss.remap(coloured)


def test_barcode_parsing():
    slide, institution = bcss.parse_roi("TCGA-A2-A0YE-DX1_xmin45749_ymin25055_MPP-0.2500.png")
    assert slide == "TCGA-A2-A0YE"
    assert institution == "A2"


def test_held_out_institutions_are_recognised():
    _, institution = bcss.parse_roi("TCGA-OL-A5RW-DX1_xmin1_ymin1_MPP-0.2500.png")
    assert institution in bcss.TEST_INSTITUTIONS


def test_unparseable_filename_is_refused():
    """Grouping drives every split, so a guess here would be an invisible leak."""
    with pytest.raises(ValueError, match="TCGA barcode"):
        bcss.parse_roi("region_0001.png")
