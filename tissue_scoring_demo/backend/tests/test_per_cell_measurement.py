"""Step 14's arithmetic, on cells built to have a known answer.

Synthetic rings rather than a slide, because the property under test is
geometric: a cell stained the whole way round should score near 1.0 and a cell
with two stained bins should score near 2/36, whatever the tissue looked like.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.step16_per_cell_measurement.measure import (
    RING_BINS,
    measure_field,
)


def _ring_field(
    size: int = 81,
    inner: float = 12.0,
    outer: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One cell in the middle: its nucleus, its ring, and the angle of each pixel."""
    centre = size // 2
    rows, cols = np.mgrid[0:size, 0:size]
    radius = np.hypot(rows - centre, cols - centre)
    angle = np.arctan2(rows - centre, cols - centre)

    nucleus = np.where(radius <= inner, 1, 0).astype(np.int32)
    ring = np.where((radius > inner) & (radius <= outer), 1, 0).astype(np.int32)
    return nucleus, ring, angle, radius


def _bins_of(angle: np.ndarray) -> np.ndarray:
    return np.floor((angle + np.pi) / (2.0 * np.pi) * RING_BINS).astype(np.int64) % RING_BINS


def test_a_ring_stained_all_the_way_round_scores_one():
    nucleus, ring, _, _ = _ring_field()
    dab = np.where(ring > 0, 0.6, 0.0).astype(np.float32)

    (cell,) = measure_field(
        nucleus, ring, dab, membrane=True, positivity_od=0.15, mpp=0.5,
        region_rank=1, field_index=0,
    )

    assert cell.second == pytest.approx(1.0)
    assert cell.occupied_bins == RING_BINS
    assert cell.intensity_od == pytest.approx(0.6, abs=1e-6)
    assert cell.second_measure == "ring_completeness"


def test_two_stained_bins_score_two_thirty_sixths_not_positive():
    """The speckle case the completeness measure exists to catch.

    Its mean optical density over the whole ring is low, but a cell like this is
    exactly what a max-based or a threshold-fraction measure would call positive.
    """
    nucleus, ring, angle, _ = _ring_field()
    bins = _bins_of(angle)
    speckle = ring.astype(bool) & np.isin(bins, [4, 5])

    dab = np.where(speckle, 0.9, 0.02).astype(np.float32)
    (cell,) = measure_field(
        nucleus, ring, dab, membrane=True, positivity_od=0.15, mpp=0.5,
        region_rank=1, field_index=0,
    )

    assert cell.second == pytest.approx(2 / RING_BINS, abs=1e-6)
    assert cell.occupied_bins == RING_BINS
    # And it fails the completeness cut, which is the point.
    assert cell.second < 0.35


def test_a_crowded_cell_reports_its_missing_bins_rather_than_reading_as_weak():
    """Half the ring taken by a neighbour: complete where it exists, 18 bins occupied."""
    nucleus, ring, angle, _ = _ring_field()
    bins = _bins_of(angle)
    kept = ring.astype(bool) & (bins < RING_BINS // 2)

    measured = np.where(kept, 1, 0).astype(np.int32)
    dab = np.where(kept, 0.6, 0.0).astype(np.float32)

    (cell,) = measure_field(
        nucleus, measured, dab, membrane=True, positivity_od=0.15, mpp=0.5,
        region_rank=1, field_index=0,
    )

    assert cell.occupied_bins == RING_BINS // 2
    assert cell.second == pytest.approx(0.5, abs=0.03)


def test_a_cytoplasmic_marker_gets_a_stained_fraction_and_no_completeness():
    """The fork. A band has no circumference, so there are no bins to report."""
    nucleus, band, angle, _ = _ring_field()
    bins = _bins_of(angle)
    # A quarter of the band stained - as a ring measure this would be 0.25 too,
    # so the test is that the *name* and the absence of bins differ, not the value.
    stained = band.astype(bool) & (bins < RING_BINS // 4)
    dab = np.where(stained, 0.5, 0.01).astype(np.float32)

    (cell,) = measure_field(
        nucleus, band, dab, membrane=False, positivity_od=0.12, mpp=0.5,
        region_rank=1, field_index=0,
    )

    assert cell.second_measure == "stained_fraction"
    assert cell.occupied_bins is None
    assert cell.second == pytest.approx(0.25, abs=0.03)


def test_neighbouring_cells_are_measured_separately():
    """Two cells, one stained and one not, must not average into each other."""
    nucleus = np.zeros((40, 80), dtype=np.int32)
    measured = np.zeros((40, 80), dtype=np.int32)
    nucleus[18:22, 8:12] = 1
    nucleus[18:22, 68:72] = 2
    measured[10:30, 4:24] = 1
    measured[10:30, 56:76] = 2
    measured[nucleus > 0] = 0

    dab = np.zeros((40, 80), dtype=np.float32)
    dab[measured == 1] = 0.8
    dab[measured == 2] = 0.02

    cells = measure_field(
        nucleus, measured, dab, membrane=False, positivity_od=0.15, mpp=0.5,
        region_rank=2, field_index=3,
    )

    by_id = {cell.cell_id: cell for cell in cells}
    assert by_id[1].intensity_od == pytest.approx(0.8, abs=1e-6)
    assert by_id[2].intensity_od == pytest.approx(0.02, abs=1e-6)
    assert by_id[1].second == pytest.approx(1.0)
    assert by_id[2].second == pytest.approx(0.0)
    assert by_id[1].region_rank == 2 and by_id[1].field_index == 3


def test_mismatched_inputs_are_refused_rather_than_broadcast():
    nucleus, ring, _, _ = _ring_field(size=40)
    dab = np.zeros((41, 40), dtype=np.float32)
    with pytest.raises(ValueError, match="same pixels"):
        measure_field(
            nucleus, ring, dab, membrane=True, positivity_od=0.15, mpp=0.5,
            region_rank=1, field_index=0,
        )


def test_area_is_in_square_microns_not_pixels():
    nucleus, ring, _, _ = _ring_field()
    dab = np.full(ring.shape, 0.3, dtype=np.float32)
    (cell,) = measure_field(
        nucleus, ring, dab, membrane=True, positivity_od=0.15, mpp=0.5,
        region_rank=1, field_index=0,
    )
    assert cell.area_um2 == pytest.approx(cell.pixels * 0.25)
