"""P-21: the optical-physics details the review listed.

The H&E reference count is the one that matters - the missing-cells figure, the
DENOMINATOR INCOMPLETE caveat and the typing trust check are all measured against it -
so it is pinned on the un-mixing itself and on how it samples. The other two are small
and pinned on the arithmetic.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.common import imaging, stains
from app.pipeline.step13_nuclei_segmentation.stain_input import (
    HAEMATOXYLIN,
    ruifrok_basis,
    ruifrok_he_basis,
)
from app.pipeline.step16_per_cell_measurement.measure import measure_field
from app.services import nuclei_service as nuclei_module
from app.services.nuclei_service import nuclei_service


def test_eosin_reads_as_haematoxylin_under_the_hdab_basis_and_not_under_the_he_one():
    """The review's figure, reproduced, and the fix: 0.668 false haematoxylin -> none."""
    eosin = np.asarray(stains.REFERENCE_BY_NAME["eosin"], dtype=np.float64)
    wrong = ruifrok_basis().inverse @ eosin
    right = ruifrok_he_basis().inverse @ eosin
    assert wrong[HAEMATOXYLIN] == pytest.approx(0.668, abs=1e-3)
    assert abs(right[HAEMATOXYLIN]) < 1e-6


def test_haematoxylin_is_column_zero_in_both_bases():
    """So every reader of `HAEMATOXYLIN` works on either basis unchanged."""
    haematoxylin = np.asarray(stains.REFERENCE_BY_NAME["haematoxylin"], dtype=np.float64)
    for basis in (ruifrok_basis(), ruifrok_he_basis()):
        coefficients = basis.inverse @ haematoxylin
        assert coefficients[HAEMATOXYLIN] == pytest.approx(1.0, abs=1e-3)


def test_the_he_reference_samples_every_region_with_the_ihc_allocation(monkeypatch):
    """It used to take six fields of the largest region; the IHC covers all of them."""
    regions = [
        SimpleNamespace(rank=1, area_mm2=8.0, he_rings=["big"]),
        SimpleNamespace(rank=2, area_mm2=2.0, he_rings=["small"]),
    ]
    asked: list[tuple[str, int]] = []

    class _Reader:
        mpp = 0.25
        dimensions = (10_000, 10_000)

        def close(self):
            pass

    monkeypatch.setattr(nuclei_module, "open_slide", lambda path: _Reader())
    monkeypatch.setattr(nuclei_module, "resolve_ready_path", lambda upload_id: "he.svs")
    monkeypatch.setattr(nuclei_module.calibration_service, "white_point", lambda upload_id: None)

    def fields(rings, *, mpp, slide_width, slide_height, count):
        asked.append((rings[0], count))
        return [object()] * count, count

    seen_bases = set()

    def segment(reader, sample, *, white, basis, base_mpp, model_mpp, remove_dab=None,
                engine=None, region_rings=None):
        assert region_rings, "the H&E reference counts inside the region too (P-10)"
        seen_bases.add(basis.source)
        assert remove_dab is False, "the H&E is segmented on its own photograph"
        assert engine == "instanseg", "the reference stays on the detector P-21 validated"
        return SimpleNamespace(counted=10, counted_mm2=0.01, tissue_mm2=0.005,
                               nuclei=[SimpleNamespace(counted=True, area_um2=40.0)])

    monkeypatch.setattr(nuclei_module, "fields_for_region", fields)
    monkeypatch.setattr(nuclei_module, "segment_field", segment)

    reference = nuclei_service._he_reference(
        "he", SimpleNamespace(regions=regions), model_mpp=0.5
    )
    density, median, used = reference.density, reference.median_area, reference.fields
    assert reference.tissue_density == pytest.approx(2000.0)
    assert reference.tissue_share == pytest.approx(0.5)

    expected = nuclei_module.allocate([8.0, 2.0])
    assert asked == [("big", expected[0]), ("small", expected[1])]
    assert used == sum(expected)
    assert seen_bases == {"ruifrok_he"}
    assert density == pytest.approx(1000.0)
    assert median == pytest.approx(40.0)


# --- per-cell measurement ------------------------------------------------------------


def _ring_field(dab_value_on_ring: np.ndarray | float):
    """One cell: a nucleus and a ring around it, with chosen DAB on the ring."""
    nucleus = np.zeros((40, 40), np.int32)
    nucleus[15:25, 15:25] = 1
    measured = np.zeros_like(nucleus)
    measured[10:30, 10:30] = 1
    measured[15:25, 15:25] = 0
    dab = np.zeros((40, 40), np.float64)
    dab[measured == 1] = dab_value_on_ring if np.isscalar(dab_value_on_ring) else dab_value_on_ring
    return nucleus, measured, dab


def test_negative_dab_does_not_pull_a_cell_mean_below_zero():
    """Less than no stain is no stain: a ring of +0.2 and -0.2 averages 0.1, not 0."""
    nucleus, measured, dab = _ring_field(0.0)
    ring = np.argwhere(measured == 1)
    for k, (r, c) in enumerate(ring):
        dab[r, c] = 0.2 if k % 2 == 0 else -0.2
    cell = measure_field(nucleus, measured, dab, membrane=False, positivity_od=0.15,
                         mpp=0.5, region_rank=1, field_index=0)[0]
    assert cell.intensity_od == pytest.approx(0.1, abs=0.01)


def test_a_value_exactly_on_the_cut_counts_as_stained_like_the_binning_does():
    nucleus, measured, dab = _ring_field(0.15)
    cell = measure_field(nucleus, measured, dab, membrane=False, positivity_od=0.15,
                         mpp=0.5, region_rank=1, field_index=0)[0]
    assert cell.second == pytest.approx(1.0), "every pixel sits on the cut, so all count"


def test_the_od_convention_is_named():
    assert imaging.OD_CONVENTION == "srgb_encoded"
