"""The majority vote, including the two boundaries it is easy to get wrong.

Written against hand-built masks rather than real tiles, because the question is
arithmetic and a real tile cannot be made to sit exactly on a threshold.
"""

from __future__ import annotations

import numpy as np

import bcss
from export import DROP_MIXED, DROP_UNUSABLE, vote


def mask_of(shares: dict[int, float], size: int = 100) -> np.ndarray:
    """A `size x size` remapped mask with the requested share of each class.

    Filled row-major, so the shares are exact to one row - which is what lets a test
    sit a tile precisely on 0.70 usable or 0.80 non-epithelium.
    """
    total = size * size
    flat = np.full(total, bcss.IGNORE, dtype=np.uint8)
    cursor = 0
    for value, share in shares.items():
        count = int(round(share * total))
        flat[cursor : cursor + count] = value
        cursor += count
    return flat.reshape(size, size)


def test_mostly_invasive_is_invasive():
    decision = vote(mask_of({bcss.INVASIVE: 0.6, bcss.NON_EPITHELIUM: 0.4}))
    assert decision.label == bcss.INVASIVE
    assert decision.dropped_by is None


def test_mostly_dcis_is_non_invasive():
    decision = vote(mask_of({bcss.NON_INVASIVE: 0.55, bcss.NON_EPITHELIUM: 0.45}))
    assert decision.label == bcss.NON_INVASIVE


def test_stroma_needs_eighty_percent_not_fifty():
    """A tile that is 60% stroma and 40% tumour is a tumour tile with stroma in it.

    Calling it stroma would erode every tumour boundary in the dataset, which is why
    the non-epithelium threshold is 0.80 while the epithelial ones are 0.50.
    """
    decision = vote(mask_of({bcss.NON_EPITHELIUM: 0.6, bcss.INVASIVE: 0.4}))
    assert decision.label is None
    assert decision.dropped_by == DROP_MIXED


def test_clearly_non_epithelium_is_kept():
    decision = vote(mask_of({bcss.NON_EPITHELIUM: 0.85, bcss.INVASIVE: 0.15}))
    assert decision.label == bcss.NON_EPITHELIUM


def test_unusable_tile_is_dropped_before_any_vote():
    """Below 0.70 labelled, the tile is mostly unannotated and has no majority to
    read. Dropped for that reason, not for being mixed."""
    decision = vote(mask_of({bcss.INVASIVE: 0.5}))  # 50% labelled, 50% IGNORE
    assert decision.label is None
    assert decision.dropped_by == DROP_UNUSABLE


def test_fractions_are_over_usable_pixels_not_the_whole_tile():
    """**The subtlety that decides the annotation edges.**

    A tile that is 25% unannotated and 75% tumour is 100% tumour *where it is
    annotated at all*. Measuring the fractions over the whole tile would score it
    0.75 - still above the threshold here, but the same arithmetic drops tiles at the
    edge of an annotation, and the edge of an annotation is where the interesting
    boundaries live.
    """
    decision = vote(mask_of({bcss.INVASIVE: 0.75}))
    assert decision.usable == 0.75
    assert decision.invasive == 1.0
    assert decision.label == bcss.INVASIVE


def test_usable_boundary_is_inclusive():
    """Exactly 0.70 usable is kept: the rule is `usable < 0.70 -> drop`."""
    decision = vote(mask_of({bcss.INVASIVE: 0.70}))
    assert decision.usable == 0.70
    assert decision.label == bcss.INVASIVE


def test_non_epithelium_boundary_is_inclusive():
    """Exactly 0.80 of the usable pixels: kept, by `nonepi_frac >= 0.80`."""
    decision = vote(mask_of({bcss.NON_EPITHELIUM: 0.80, bcss.INVASIVE: 0.20}))
    assert abs(decision.non_epithelium - 0.80) < 1e-9
    assert decision.label == bcss.NON_EPITHELIUM


def test_invasive_wins_ties_against_dcis():
    """Half invasive and half in-situ resolves to invasive, and that is deliberate.

    The vote is ordered, and invasive is tested first. A tile containing both is a
    tile at the invasive front - which is invasive tissue - and under-calling it
    would shrink the scored region at exactly the boundary that matters clinically.
    """
    decision = vote(mask_of({bcss.INVASIVE: 0.5, bcss.NON_INVASIVE: 0.5}))
    assert decision.label == bcss.INVASIVE


def test_fully_unlabelled_tile_does_not_divide_by_zero():
    decision = vote(np.full((32, 32), bcss.IGNORE, dtype=np.uint8))
    assert decision.label is None
    assert decision.usable == 0.0
    assert decision.dropped_by == DROP_UNUSABLE
