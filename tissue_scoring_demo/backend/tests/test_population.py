"""P-04: the score counts every cell in the region; tumour-only is a sensitivity.

The cell typing fails its own check on every slide measured so far, and filtering the
denominator on it put a stain-correlated hole in every score. These pin the change: the
geometry and the measurement follow `settings.score_population`, a stored geometry for
the other population is rebuilt, and a failed typing qualifies only the side figure.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.core.config import settings
from app.pipeline.step15_compartments import geometry
from app.services import score_service as score_module
from app.services.compartment_service import compartment_service
from app.services.nuclei_service import NucleiError
from app.services.per_cell_service import per_cell_service
from app.services.score_service import score_service

HE, IHC = "popHe", "popIhc"


def test_the_default_population_is_every_cell():
    assert settings.score_population == "all"


@pytest.fixture
def stored(tmp_path, monkeypatch):
    """One field drawn by step 15 over every cell."""
    monkeypatch.setattr(settings, "compartments_dir", tmp_path)
    labels = np.zeros((64, 64), np.int32)
    labels[10:20, 10:20] = 1
    labels[40:50, 40:50] = 2
    built = geometry.build(labels, mpp=0.5, expansion_um=4.0, ring=True, ring_um=1.5)
    compartment_service._write_geometry(
        HE, IHC, 1, 0, built, width_um=4.0, is_membrane=True, tumour_only=False,
        shell_um=1.5, nuclei_generated_at="n", typing_stamp="t", field_x=0, field_y=0,
    )


def _read(**overrides):
    kwargs = dict(width_um=4.0, is_membrane=True, shape=(64, 64), tumour_only=False,
                  shell_um=1.5, nuclei_generated_at="n", typing_stamp="t",
                  field_x=0, field_y=0, cell_ids={1, 2})
    kwargs.update(overrides)
    return per_cell_service._stored_geometry(HE, IHC, 1, 0, **kwargs)


def test_an_all_cells_geometry_is_used_for_an_all_cells_measurement(stored):
    loaded, reason = _read()
    assert reason is None and loaded is not None


def test_a_geometry_for_the_other_population_is_rebuilt(stored):
    """Geometry drawn over every cell must not be measured as tumour-only, or back."""
    loaded, reason = _read(tumour_only=True)
    assert loaded is None
    assert reason is not None and "every cell" in reason and "tumour cells only" in reason


def test_the_width_sweep_reads_field_labels_with_their_position(tmp_path, monkeypatch):
    """Regression: P-15 widened `_field_labels` to return the field's position and the
    sweep still unpacked two values, so step 15 crashed on every fresh computation."""
    labels = np.zeros((64, 64), np.int32)
    labels[10:20, 10:20] = 1
    monkeypatch.setattr(
        compartment_service, "_field_labels", lambda *a: (labels, 0.5, 100, 200)
    )
    nuclei = SimpleNamespace(regions=[SimpleNamespace(rank=1, fields=[SimpleNamespace(index=0)])])
    points = compartment_service._sweep(HE, IHC, nuclei, True, False, 0.375)
    assert points, "the sweep should produce its curve"


# --- the score: all cells reported, tumour-only beside it ---------------------------


def _quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "cell_typing_dir", tmp_path / "cell_typing")
    monkeypatch.setattr(
        score_module.nuclei_service, "report",
        lambda *a: (_ for _ in ()).throw(NucleiError("none")),
    )
    monkeypatch.setattr(
        score_module.ihc_alignment_service, "report",
        lambda *a: SimpleNamespace(
            confirmed=True, confirmed_by="person",
            diagnostics=SimpleNamespace(gate_overridden=[], tissue_area_fallbacks=[]),
        ),
    )
    monkeypatch.setattr(score_module.ScoreService, "_regions_chosen_by_machine", staticmethod(lambda he: False))
    monkeypatch.setattr(score_module.ScoreService, "_typing_verdict", staticmethod(lambda he, ihc: (False, "nuclei too small")))


def _result(**overrides):
    base = dict(
        marker="A", marker_name="CD44", percent=20, intensity_raw=0.0, cells=7234,
        cuts_provisional=False, percent_area_weighted=20.0, percent_pooled=20.0,
        percent_plain_mean=20.0, averaging_gap_points=0.0,
        percent_by_partial_rule={"count": 20, "exclude": 20},
        percent_raw=20.0, percent_ci_low=17.0, percent_ci_high=23.0,
        single_field_weight=0.0, unsampled_regions=0, unsampled_area_mm2=0.0,
        total_area_mm2=10.0,
    )
    return SimpleNamespace(**{**base, **overrides})


def _caveats(population, tumour_only=None):
    return score_service._caveats(
        HE, IHC, _result(), SimpleNamespace(crowded_cells=None, cells=7234),
        population=population, tumour_only=tumour_only,
    )


def test_failed_typing_no_longer_refuses_an_all_cells_score(monkeypatch, tmp_path):
    """It decides only the sensitivity figure, so it qualifies that and nothing else."""
    _quiet(monkeypatch, tmp_path)
    caveats = _caveats("all", SimpleNamespace(percent=15, intensity=1.0, cells=1039))
    status, reasons = score_service._status(caveats)
    assert status == "provisional"
    assert "ALL CELLS IN THE REGION, NOT TUMOUR ONLY." in reasons
    assert any(c.startswith("TUMOUR-ONLY FIGURE RESTS ON FAILED TYPING.") for c in caveats)
    population = next(c for c in caveats if c.startswith("ALL CELLS"))
    assert "15 % positive" in population and "1,039 of 7,234" in population and "5 points" in population


def test_failed_typing_still_refuses_a_tumour_only_score(monkeypatch, tmp_path):
    """Back on the tumour-only population, the typing gates the number again - and P-17's
    refusal applies again."""
    _quiet(monkeypatch, tmp_path)
    assert score_service._status(_caveats("tumour"))[0] == "not_a_measurement"
