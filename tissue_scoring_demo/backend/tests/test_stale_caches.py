"""P-15: no result is read back unless it was made from the inputs in front of it.

Five places a stale file could be served as current, one test group each. Every check is
made against a file on disk after a round trip, never against the object just built -
a check that cannot fail is how three verification bugs reached this project in one day.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.core import provenance
from app.core.config import settings
from app.services.ihc_alignment_service import ihc_alignment_service
from app.services.ihc_alignment_service import stored_transforms
from app.services.per_cell_service import PerCellError, per_cell_service
from app.services.tissue_type_service import tissue_type_service

HE, IHC = "staleHe", "staleIhc"


# --- 1. per-cell rows: only the regions the current pass measured -------------------


@pytest.fixture
def per_cell_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "per_cell_dir", tmp_path)
    return tmp_path / f"{HE}__{IHC}"


def _region(directory: Path, rank: int, cells: int) -> None:
    (directory / f"region{rank}").mkdir(parents=True, exist_ok=True)
    rows = [{"cellId": index, "regionRank": rank} for index in range(cells)]
    (directory / f"region{rank}" / "cells.json").write_text(
        json.dumps({"rank": rank, "cells": rows}), encoding="utf-8"
    )


def test_rows_ignore_regions_an_earlier_pass_left_behind(per_cell_dir: Path):
    """CAN_00270 tile A/F: 265 stale region folders, 4,013 cells where 932 were real."""
    _region(per_cell_dir, 1, 10)
    _region(per_cell_dir, 2, 5)
    _region(per_cell_dir, 7, 999)  # left by an earlier pass over other regions
    (per_cell_dir / "report.json").write_text(
        json.dumps({"regions": [{"rank": 1}, {"rank": 2}]}), encoding="utf-8"
    )

    rows = per_cell_service.rows(HE, IHC)
    assert len(rows) == 15
    assert {row["regionRank"] for row in rows} == {1, 2}


def test_rows_refuse_without_the_report_rather_than_reading_old_rows(per_cell_dir: Path):
    """A pass that failed half-way leaves rows and no report: that is not a result."""
    _region(per_cell_dir, 1, 10)
    with pytest.raises(PerCellError, match="has not measured"):
        per_cell_service.rows(HE, IHC)


def test_a_listed_region_with_no_rows_is_an_error_not_a_smaller_score(per_cell_dir: Path):
    _region(per_cell_dir, 1, 10)
    (per_cell_dir / "report.json").write_text(
        json.dumps({"regions": [{"rank": 1}, {"rank": 2}]}), encoding="utf-8"
    )
    with pytest.raises(PerCellError, match="region 2"):
        per_cell_service.rows(HE, IHC)


def test_a_new_pass_starts_from_a_clean_directory(per_cell_dir: Path):
    _region(per_cell_dir, 3, 4)
    (per_cell_dir / "report.json").write_text("{}", encoding="utf-8")
    per_cell_service._clear(HE, IHC)
    assert not (per_cell_dir / "region3").exists()
    assert not (per_cell_dir / "report.json").exists()


# --- 3. alignment reports: gate version and transform, ready ones included ----------


def _report(directory: Path, *, state: str, gate: int | None, transform: str | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "heUploadId": HE,
        "ihcUploadId": IHC,
        "state": state,
        "generatedAt": "2026-10-01T00:00:00+00:00",
        "regions": [],
        "diagnostics": {
            "registrationKind": "intensity",
            "gateVersion": gate,
            "transformSha256": transform,
        },
    }
    (directory / "report.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def alignment_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "ihc_alignment_dir", tmp_path)
    monkeypatch.setattr(stored_transforms, "fingerprint", lambda he, ihc: "current-sha")
    return tmp_path / f"{HE}__{IHC}"


def test_a_current_ready_alignment_is_reused(alignment_dir: Path):
    _report(alignment_dir, state="ready", gate=ihc_alignment_service.GATE_VERSION,
            transform="current-sha")
    assert ihc_alignment_service._read_report(HE, IHC) is not None


@pytest.mark.parametrize("gate", [None, 3, 4])
def test_a_ready_alignment_from_an_older_gate_is_redecided(alignment_dir: Path, gate):
    """Reports on disk carried gate versions None, 3 and 4 and were reused as ready."""
    _report(alignment_dir, state="ready", gate=gate, transform="current-sha")
    assert ihc_alignment_service._read_report(HE, IHC) is None


@pytest.mark.parametrize("state", ["ready", "refused"])
def test_an_alignment_whose_transform_changed_is_redecided(alignment_dir: Path, state):
    _report(alignment_dir, state=state, gate=ihc_alignment_service.GATE_VERSION,
            transform="refit-since")
    assert ihc_alignment_service._read_report(HE, IHC) is None


def test_the_transform_fingerprint_follows_the_file(tmp_path: Path, monkeypatch):
    """The real fingerprint, over a real file: change one byte and it moves."""
    case_dir = tmp_path / "CASE"
    (case_dir / "transforms").mkdir(parents=True)
    (case_dir / "transforms" / "A.h5").write_bytes(b"transform-v1")
    (case_dir / "aligned.json").write_text(json.dumps({"slides": {
        "HE": {"level0ToAligned": [[1, 0, 0], [0, 1, 0]]},
        "A": {"level0ToAligned": [[1, 0, 0], [0, 1, 0]]},
    }}), encoding="utf-8")
    (case_dir / "transforms.json").write_text(json.dumps({
        "ok": True, "work_scale": 0.5,
        "saved": {"A": {"ok": True, "file": "A.h5", "method": "mattes"}},
    }), encoding="utf-8")
    monkeypatch.setattr(stored_transforms, "_case_for", lambda he, ihc: ("CASE", "A"))
    monkeypatch.setattr(stored_transforms.common, "case_dir", lambda case: case_dir)

    first = stored_transforms.fingerprint(HE, IHC)
    (case_dir / "transforms" / "A.h5").write_bytes(b"transform-v2")
    second = stored_transforms.fingerprint(HE, IHC)
    assert first and second and first != second


# --- 4. step 8's run key covers step 7's gates and the white point ------------------


@pytest.mark.parametrize(
    "field", ["tile_min_tissue_share", "tile_min_clean_share", "white_key", "tissue_mask_key"]
)
def test_an_upstream_change_invalidates_the_class_map(field):
    base = {"model": "m", field: "before"}
    moved = {**base, field: "after"}
    assert tissue_type_service._cache_key(base) != tissue_type_service._cache_key(moved)


# --- 5. results are stamped, and `done` is re-checked -------------------------------


def test_the_settings_hash_ignores_paths_but_not_settings(monkeypatch):
    before = provenance.config_sha256()
    monkeypatch.setattr(settings, "per_cell_dir", Path("/somewhere/else"))
    assert provenance.config_sha256() == before, "moving the data must not stale results"
    monkeypatch.setattr(settings, "tissue_type_max_flat_share", 0.5)
    assert provenance.config_sha256() != before


def test_an_unstamped_or_changed_result_is_stale():
    current = provenance.code_state()
    assert provenance.stale_fields(None, current) == list(provenance.COMPARED)
    assert provenance.stale_fields(dict(current), current) == []
    assert provenance.stale_fields({**current, "cuts_version": -1}, current) == ["cuts_version"]
    # A dirty tree is recorded, not compared.
    assert provenance.stale_fields({**current, "code_dirty": not current["code_dirty"]}, current) == []


def test_run_all_reruns_a_done_case_scored_with_other_settings(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3] / "score_all_slides"
    sys.path.insert(0, str(root))
    import pipeline
    import run_all

    monkeypatch.setattr(pipeline, "RESULTS", tmp_path)
    current = provenance.code_state()
    record = {"caseId": "CASE", "markers": [{"state": "scored", "provenance": dict(current)}]}
    (tmp_path / "CASE.json").write_text(json.dumps(record), encoding="utf-8")
    assert run_all._stale("CASE") == ""

    record["markers"][0]["provenance"]["config_sha256"] = "an older configuration"
    (tmp_path / "CASE.json").write_text(json.dumps(record), encoding="utf-8")
    assert run_all._stale("CASE") == "config_sha256"

    del record["markers"][0]["provenance"]
    (tmp_path / "CASE.json").write_text(json.dumps(record), encoding="utf-8")
    assert "code_commit" in run_all._stale("CASE")
