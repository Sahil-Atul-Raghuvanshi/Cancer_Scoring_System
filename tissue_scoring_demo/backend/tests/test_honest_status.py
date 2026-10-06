"""P-17: a failure must never make a row look cleaner.

Each test sets up one of the review's conditions and checks the row says so - as a
caveat a person reads and as the `status` a filter reads. The rule throughout: when the
evidence is missing, the row is marked down, never left as it was.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services import score_service as score_module
from app.services.nuclei_service import NucleiError
from app.services.score_service import score_service
from app.services.tissue_type_service import tissue_type_service

HE, IHC = "statusHe", "statusIhc"


def _result(**overrides):
    """The fields `_caveats` reads off a scored marker - a clean, well-counted one."""
    base = dict(
        marker="A", marker_name="CD44", percent=60, intensity_raw=0.0, cells=2000,
        cuts_provisional=False, percent_area_weighted=60.0, percent_pooled=60.0,
        percent_plain_mean=60.0, averaging_gap_points=0.0,
        percent_by_partial_rule={"count": 60, "exclude": 60},
        percent_raw=60.0, percent_ci_low=56.0, percent_ci_high=64.0,
        single_field_weight=0.0, unsampled_regions=0, unsampled_area_mm2=0.0,
        total_area_mm2=10.0,
    )
    return SimpleNamespace(**{**base, **overrides})


@pytest.fixture
def quiet(monkeypatch, tmp_path):
    """Every upstream read returns 'nothing to report', so each test adds one condition."""
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
    return tmp_path


def _caveats(result=None):
    measured = SimpleNamespace(crowded_cells=None, cells=2000)
    return score_service._caveats(HE, IHC, result or _result(), measured)


def test_a_clean_row_is_measured(quiet):
    caveats = _caveats()
    assert score_service._status(caveats) == ("measured", [])


def test_an_unreadable_alignment_is_unknown_not_clean(quiet, monkeypatch):
    """It used to be `except Exception: pass` - and the row then read as checked."""
    monkeypatch.setattr(
        score_module.ihc_alignment_service, "report",
        lambda *a: (_ for _ in ()).throw(OSError("report.json is truncated")),
    )
    caveats = _caveats()
    assert any(c.startswith("ALIGNMENT STATUS UNKNOWN.") and "truncated" in c for c in caveats)
    status, reasons = score_service._status(caveats)
    assert status == "not_a_measurement"
    assert "ALIGNMENT STATUS UNKNOWN." in reasons


def test_untrustworthy_typing_is_read_and_refuses(quiet):
    """Step 14 flagged itself on 30/30 slides and nothing read the flag."""
    directory = settings.cell_typing_dir / f"{HE}__{IHC}"
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(
        json.dumps({"trustworthy": False, "trustReason": "median nucleus too small"}),
        encoding="utf-8",
    )
    caveats = _caveats()
    assert any("median nucleus too small" in c for c in caveats)
    assert score_service._status(caveats)[0] == "not_a_measurement"


@pytest.mark.parametrize(
    ("alignment", "heading"),
    [
        (SimpleNamespace(confirmed=True, confirmed_by="machine",
                         diagnostics=SimpleNamespace(gate_overridden=[], tissue_area_fallbacks=[])),
         "ALIGNMENT MACHINE-CONFIRMED."),
        (SimpleNamespace(confirmed=True, confirmed_by="person",
                         diagnostics=SimpleNamespace(gate_overridden=[],
                                                     tissue_area_fallbacks=["gate_areas.json missing"])),
         "ALIGNMENT GATE MEASURED ON A FALLBACK."),
    ],
)
def test_an_unchecked_alignment_is_provisional(quiet, monkeypatch, alignment, heading):
    monkeypatch.setattr(score_module.ihc_alignment_service, "report", lambda *a: alignment)
    status, reasons = score_service._status(_caveats())
    assert status == "provisional" and heading in reasons


def test_a_failed_gate_refuses(quiet, monkeypatch):
    monkeypatch.setattr(
        score_module.ihc_alignment_service, "report",
        lambda *a: SimpleNamespace(
            confirmed=True, confirmed_by="person",
            diagnostics=SimpleNamespace(gate_overridden=["tissue ratio 0.31"], tissue_area_fallbacks=[]),
        ),
    )
    assert score_service._status(_caveats())[0] == "not_a_measurement"


def test_regions_picked_by_the_default_rule_are_provisional(quiet, monkeypatch):
    monkeypatch.setattr(score_module.ScoreService, "_regions_chosen_by_machine", staticmethod(lambda he: True))
    status, reasons = score_service._status(_caveats())
    assert status == "provisional" and "REGIONS CHOSEN BY THE DEFAULT RULE." in reasons


def test_too_few_cells_refuses_and_a_thin_count_is_provisional(quiet):
    assert score_service._status(_caveats(_result(cells=12)))[0] == "not_a_measurement"
    assert score_service._status(_caveats(_result(cells=200)))[0] == "provisional"


def test_the_shortfall_caveat_no_longer_claims_a_direction(quiet, monkeypatch):
    """Whether missed nuclei raise or lower the percentage has not been measured."""
    monkeypatch.setattr(
        score_module.nuclei_service, "report", lambda *a: SimpleNamespace(density_shortfall=0.6)
    )
    shortfall = next(c for c in _caveats() if c.startswith("DENOMINATOR INCOMPLETE."))
    assert "inflates the percentage" not in shortfall
    assert "has not been" in shortfall


def test_stale_wording_is_gone(quiet):
    caveats = " ".join(_caveats(_result(cuts_provisional=True)))
    assert "not on disk" not in caveats


# --- step 8's warnings fire on what actually happened -------------------------------


def _class_map(*, in_situ=0.05, unknown=0, refused=0, confidence=0.9):
    shares = (0.6 - in_situ / 2, in_situ, 0.4 - in_situ / 2)
    return SimpleNamespace(
        shares=shares,
        classified=1000,
        mean_confidence=confidence,
        uncertainty=SimpleNamespace(unknown_windows=unknown) if unknown else None,
        refused_count=lambda reason: refused if reason == 1 else 0,
    )


def _step8(class_map, manifest=None):
    pinned = SimpleNamespace(manifest=manifest or {}, licence_track="commercial", licences={})
    return {c.key: c for c in tissue_type_service._caveats(pinned, class_map)}


def test_the_in_situ_breakdown_being_missing_is_said():
    """Every served manifest lacks `by_source`, so the old caveat never fired."""
    assert "in_situ_ground_truth" in _step8(_class_map())


def test_the_domain_caveat_no_longer_says_marker_slide():
    domain = _step8(_class_map())["domain"]
    assert "marker slide" not in domain.headline + domain.detail


def test_a_large_in_situ_share_is_flagged_even_at_high_confidence():
    """CAN_00251: 45.9% in-situ at 0.809 confidence, which the 0.6 rule never caught."""
    caveats = _step8(_class_map(in_situ=0.459, confidence=0.809))
    assert "in_situ_share" in caveats
    assert "confidence" not in caveats
    assert "in_situ_share" not in _step8(_class_map(in_situ=0.10))


def test_a_large_undetermined_share_is_flagged():
    assert "undetermined_share" in _step8(_class_map(refused=150))
    assert "undetermined_share" not in _step8(_class_map(refused=20))


# --- the CSV carries the status -----------------------------------------------------


def test_the_csv_carries_status_beside_the_number(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3] / "score_all_slides"
    sys.path.insert(0, str(root))
    import pipeline

    monkeypatch.setattr(pipeline, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(pipeline, "CSV_PATH", tmp_path / "scores.csv")
    (tmp_path / "results").mkdir()
    record = {"caseId": "CASE", "markers": [
        {"marker": "A", "state": "scored", "percent": 60, "status": "not_a_measurement",
         "status_reasons": ["NOT A MEASUREMENT."]},
        {"marker": "F", "state": "scored", "percent": 40},  # scored before P-17
    ]}
    (tmp_path / "results" / "CASE.json").write_text(json.dumps(record), encoding="utf-8")
    pipeline.write_csv()

    import csv

    rows = {row["marker"]: row for row in csv.DictReader(open(tmp_path / "scores.csv", encoding="utf-8"))}
    assert rows["A"]["status"] == "not_a_measurement"
    assert rows["A"]["status_reasons"] == "NOT A MEASUREMENT."
    assert rows["F"]["status"] == "unstamped", "an old row must not read as measured"
    header = list(rows["A"].keys())
    assert header.index("status") < header.index("percent")


# --- P-06: too few cells, too wide an interval ---------------------------------


@pytest.mark.parametrize(
    ("overrides", "status", "heading"),
    [
        (dict(cells=150), "not_a_measurement", "NOT A MEASUREMENT."),
        (dict(cells=300), "provisional", "THIN DENOMINATOR."),
        (dict(percent_ci_low=30.0, percent_ci_high=80.0), "not_a_measurement", "INTERVAL TOO WIDE."),
        (dict(percent_ci_low=48.0, percent_ci_high=73.0), "provisional", "WIDE INTERVAL."),
        (dict(single_field_weight=0.6), "provisional", "INTERVAL UNDERSTATED."),
    ],
)
def test_a_thin_or_uncertain_score_says_so(quiet, overrides, status, heading):
    caveats = _caveats(_result(**overrides))
    got, reasons = score_service._status(caveats)
    assert (got, heading in reasons) == (status, True)


def test_unsampled_area_is_reported_without_changing_the_status(quiet):
    caveats = _caveats(_result(unsampled_regions=40, unsampled_area_mm2=1.5))
    assert any(c.startswith("PART OF THE ROI NOT SAMPLED.") for c in caveats)
    assert score_service._status(caveats)[0] == "measured"


# --- P-10: step 11's coverage travels with the score ---------------------------


def _refined(**fields):
    base = dict(state="ready", completed=3, selected=3, failed=0, invasive_mm2=10.0, tile_mm2=9.5)
    return SimpleNamespace(**{**base, **fields})


def test_a_partial_step_11_makes_the_score_provisional(quiet, monkeypatch):
    from app.services.roi_refinement_service import roi_refinement_service

    monkeypatch.setattr(roi_refinement_service, "report", lambda he: _refined(state="partial", completed=2, failed=1))
    caveats = _caveats()
    status, reasons = score_service._status(caveats)
    assert status == "provisional" and "REGIONS MISSING." in reasons


def test_unselected_invasive_area_is_reported(quiet, monkeypatch):
    from app.services.roi_refinement_service import roi_refinement_service

    monkeypatch.setattr(roi_refinement_service, "report", lambda he: _refined(tile_mm2=4.0))
    caveats = _caveats()
    assert any(c.startswith("PART OF THE TUMOUR NOT SELECTED.") and "40%" in c for c in caveats)
