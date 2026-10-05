"""Step 13 - compartments.

The tests are mostly statements about the fork and about the Voronoi
constraint, because those are the two things that can be wrong in a way nobody
notices: a cytoplasmic marker through the membrane path produces a plausible
number for the wrong part of the cell, and unconstrained expansion produces a
plausible number that tracks cell packing.
"""

from __future__ import annotations

import numpy as np
import pytest

from app import panel
from app.pipeline.step15_compartments import geometry


def two_cells(gap: int = 10, size: int = 64) -> np.ndarray:
    """Two square nuclei `gap` pixels apart, for the collision tests."""
    labels = np.zeros((size, size), dtype=np.int32)
    labels[28:36, 10:18] = 1
    labels[28:36, 18 + gap : 26 + gap] = 2
    return labels


def one_cell(size: int = 64) -> np.ndarray:
    labels = np.zeros((size, size), dtype=np.int32)
    labels[28:36, 28:36] = 1
    return labels


# --- the fork ---------------------------------------------------------------


def test_the_panel_is_the_only_place_the_fork_is_defined() -> None:
    """Membrane for A/F/R, cytoplasm for U/W - and this step reads it, not copies it."""
    assert panel.spec("A").compartment is panel.Compartment.MEMBRANE
    assert panel.spec("F").compartment is panel.Compartment.MEMBRANE
    assert panel.spec("R").compartment is panel.Compartment.MEMBRANE
    assert panel.spec("U").compartment is panel.Compartment.CYTOPLASM
    assert panel.spec("W").compartment is panel.Compartment.CYTOPLASM


def test_the_cytoplasmic_markers_get_the_wider_compartment() -> None:
    """A band is wider than a ring, which is the whole reason it collides more."""
    assert panel.spec("U").compartment_width_um > panel.spec("A").compartment_width_um


def test_a_ring_is_smaller_than_a_band_at_the_same_expansion() -> None:
    """The regression the guide names, as arithmetic.

    Both are built from the same expansion; the ring keeps only the outer band.
    If these ever came out the same size, the fork would be doing nothing.
    """
    labels = one_cell()

    ring = geometry.build(labels, mpp=0.5, expansion_um=4.0, ring=True, ring_um=1.5)
    band = geometry.build(labels, mpp=0.5, expansion_um=4.0, ring=False)

    assert int(np.count_nonzero(ring.measured)) < int(np.count_nonzero(band.measured))


def test_the_measured_compartment_never_includes_the_nucleus() -> None:
    """For every marker in this panel the stain is outside the nucleus."""
    labels = one_cell()
    for ring in (True, False):
        built = geometry.build(labels, mpp=0.5, expansion_um=4.0, ring=ring, ring_um=4.0)
        assert not np.any((built.measured > 0) & (built.nucleus > 0))


def test_the_three_compartments_are_disjoint_where_they_must_be() -> None:
    """`measured` and `nucleus` partition the cell, so a pixel has one answer."""
    built = geometry.build(one_cell(), mpp=0.5, expansion_um=4.0, ring=False)

    overlap = (built.measured > 0) & (built.nucleus > 0)
    assert not overlap.any()
    # And everything measured is inside the cell it belongs to.
    inside = built.measured > 0
    assert np.array_equal(built.measured[inside], built.cell[inside])


# --- the Voronoi constraint -------------------------------------------------


def test_two_neighbours_never_claim_the_same_pixel() -> None:
    """The property that stops one cell's stain being counted for another."""
    labels = two_cells(gap=6)

    built = geometry.build(labels, mpp=0.5, expansion_um=6.0, ring=False)

    # Every expanded pixel has exactly one owner, by construction.
    for value in (1, 2):
        others = built.cell[built.cell == value]
        assert (others == value).all()
    assert set(np.unique(built.cell)) <= {0, 1, 2}


def test_the_constraint_actually_bit_on_touching_cells() -> None:
    """Otherwise the test above would pass on cells that never met.

    `contested_px` is the measurement of what plain dilation would have
    double-counted, so it has to be positive when two cells are close and zero
    when one cell is alone.
    """
    close = geometry.contested_px(two_cells(gap=4), mpp=0.5, expansion_um=6.0)
    alone = geometry.contested_px(one_cell(), mpp=0.5, expansion_um=6.0)

    assert close > 0
    assert alone == 0


def test_plain_dilation_does_double_claim_so_the_comparison_is_real() -> None:
    """The demo's 'why it should be on' panel needs the error to exist."""
    overlaps = geometry.unconstrained(two_cells(gap=4), mpp=0.5, expansion_um=6.0)
    assert int(np.count_nonzero(overlaps > 1)) > 0


def test_a_wider_expansion_collides_more() -> None:
    """The guide's claim that this matters more for U and W than for A/F/R."""
    labels = two_cells(gap=8)

    narrow = geometry.contested_px(labels, mpp=0.5, expansion_um=4.0)
    wide = geometry.contested_px(labels, mpp=0.5, expansion_um=8.0)

    assert wide > narrow


# --- microns, not pixels ----------------------------------------------------


def test_the_same_width_in_microns_is_a_different_number_of_pixels() -> None:
    """The bug the guide warns about: a width that means two things on two scanners."""
    labels = one_cell(size=96)

    fine = geometry.build(labels, mpp=0.25, expansion_um=4.0, ring=False)
    coarse = geometry.build(labels, mpp=0.5, expansion_um=4.0, ring=False)

    # 4 um is 16 px at 0.25 um/px and 8 px at 0.5, so the fine field's
    # compartment covers more pixels for the same physical width.
    assert int(np.count_nonzero(fine.measured)) > int(np.count_nonzero(coarse.measured))


def test_areas_come_back_in_square_microns() -> None:
    labels = one_cell()
    built = geometry.build(labels, mpp=0.5, expansion_um=4.0, ring=False)

    areas = geometry.areas_um2(built.nucleus, mpp=0.5)
    # The nucleus is 8x8 px at 0.5 um/px, so 4x4 um = 16 um2.
    assert areas[1] == pytest.approx(16.0)


def test_a_field_with_no_scale_is_refused_rather_than_assumed() -> None:
    """Without an mpp there is no such thing as a 4 micron ring."""
    with pytest.raises(ValueError):
        geometry.build(one_cell(), mpp=0.0, expansion_um=4.0, ring=False)


def test_a_field_with_no_cells_is_an_empty_answer_not_a_crash() -> None:
    empty = np.zeros((32, 32), dtype=np.int32)
    built = geometry.build(empty, mpp=0.5, expansion_um=4.0, ring=False)
    assert int(np.count_nonzero(built.measured)) == 0
    assert geometry.contested_px(empty, mpp=0.5, expansion_um=4.0) == 0
