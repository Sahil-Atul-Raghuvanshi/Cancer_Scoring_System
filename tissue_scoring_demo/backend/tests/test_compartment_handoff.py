"""Step 13 supplies step 14, and refuses to let an exploration become a score.

Step 14's declared input is "the region of each tumour cell where the marker is
supposed to be" - which is step 13's output. It used to import step 13's `geometry`
and build those regions again from step 11's labels, so the shapes a person approved
on screen and the shapes a number was measured in were two computations that merely
agreed. These tests pin both halves of the fix: that the arrays really are carried
across, and that a width explored on step 13's slider is *not*.
"""

import numpy as np
import pytest

from app.core.config import settings
from app.pipeline.step15_compartments import geometry
from app.services.compartment_service import compartment_service
from app.services.per_cell_service import per_cell_service

HE, IHC = "_handoff_he", "_handoff_ihc"
MARKER_WIDTH_UM = 4.0  # CD44's ring, from app.panel


@pytest.fixture
def stored_field(tmp_path, monkeypatch):
    """One field of two nuclei, drawn by step 13 at the marker's own width."""
    monkeypatch.setattr(settings, "compartments_dir", tmp_path)

    labels = np.zeros((64, 64), np.int32)
    labels[10:20, 10:20] = 1
    labels[40:50, 40:50] = 2
    built = geometry.build(
        labels, mpp=0.5, expansion_um=MARKER_WIDTH_UM, ring=True, ring_um=MARKER_WIDTH_UM
    )
    compartment_service._write_geometry(
        HE, IHC, 1, 0, built, width_um=MARKER_WIDTH_UM, is_membrane=True, tumour_only=True
    )
    return built


def test_step_14_measures_the_rasters_step_13_drew(stored_field) -> None:
    loaded, reason = per_cell_service._stored_geometry(
        HE, IHC, 1, 0, width_um=MARKER_WIDTH_UM, is_membrane=True, shape=(64, 64)
    )

    assert reason is None
    assert loaded is not None
    # `measured` is the one that matters - it is the mask the DAB mean is taken in.
    assert np.array_equal(loaded.measured, stored_field.measured)
    assert np.array_equal(loaded.cell, stored_field.cell)
    assert np.array_equal(loaded.nucleus, stored_field.nucleus)
    assert loaded.expansion_um == pytest.approx(stored_field.expansion_um)


def test_a_width_explored_on_screen_does_not_become_the_score(stored_field) -> None:
    """Step 13's expansion is adjustable; step 14's is the antibody's own figure.

    Loading whatever happened to be on disk would mean a slider left in the wrong
    place silently moved the deliverable - and the percentage it produced would look
    completely ordinary, which is the worst kind of wrong.
    """
    loaded, reason = per_cell_service._stored_geometry(
        HE, IHC, 1, 0, width_um=6.0, is_membrane=True, shape=(64, 64)
    )

    assert loaded is None
    assert reason is not None and "4 um" in reason and "6 um" in reason


def test_the_other_compartment_kind_is_refused(stored_field) -> None:
    """A ring measured as a band would be a different question about the cell."""
    loaded, reason = per_cell_service._stored_geometry(
        HE, IHC, 1, 0, width_um=MARKER_WIDTH_UM, is_membrane=False, shape=(64, 64)
    )
    assert loaded is None
    assert reason is not None and "compartment kind" in reason


def test_a_shape_mismatch_is_refused(stored_field) -> None:
    """Geometry from another field would measure one cell through another's mask."""
    loaded, reason = per_cell_service._stored_geometry(
        HE, IHC, 1, 0, width_um=MARKER_WIDTH_UM, is_membrane=True, shape=(32, 32)
    )
    assert loaded is None
    assert reason is not None


def test_absent_geometry_is_a_rebuild_and_not_a_complaint(stored_field) -> None:
    """Step 13 simply not having been opened yet is not a disagreement to report.

    The distinction matters on screen: a missing file means "go and run step 13",
    a mismatched one means "step 13 is showing you something else right now".
    """
    loaded, reason = per_cell_service._stored_geometry(
        HE, IHC, 9, 9, width_um=MARKER_WIDTH_UM, is_membrane=True, shape=(64, 64)
    )
    assert loaded is None
    assert reason is None
