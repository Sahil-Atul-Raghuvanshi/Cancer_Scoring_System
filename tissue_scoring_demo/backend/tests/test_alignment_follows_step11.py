"""Step 12 does not reuse a report whose regions step 11 has since replaced (P-10)."""

from __future__ import annotations

import json

from app.core.config import settings
from app.services import ihc_alignment_service as module
from app.services.ihc_alignment_service import ihc_alignment_service

HE, IHC = "keyHe", "keyIhc"


def _setup(monkeypatch, tmp_path, *, stored_key):
    monkeypatch.setattr(settings, "roi_refinement_dir", tmp_path / "roi_refinement")
    monkeypatch.setattr(settings, "ihc_alignment_dir", tmp_path / "ihc_alignment")
    monkeypatch.setattr(module, "resolve_ready_path", lambda upload_id: "x.svs")
    refined = settings.roi_refinement_dir / HE / "refined.json"
    refined.parent.mkdir(parents=True)
    refined.write_text(json.dumps([{"roiId": "ROI-001", "pieces": []}]), encoding="utf-8")
    monkeypatch.setattr(
        ihc_alignment_service, "_read_report",
        lambda he, ihc: type("R", (), {"state": "ready", "regions_key": stored_key,
                                       "generated_at": "then"})(),
    )
    ihc_alignment_service._jobs.pop(ihc_alignment_service.key(HE, IHC), None)


def test_a_report_for_the_current_regions_is_reused(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, stored_key=None)
    current = ihc_alignment_service.regions_key(HE)
    _setup(monkeypatch, tmp_path / "again", stored_key=current)
    run = ihc_alignment_service.start(HE, IHC)
    assert run.state == "ready"


def test_a_report_for_older_regions_is_redone(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, stored_key="0000000000000000")
    run = ihc_alignment_service.start(HE, IHC)
    assert run.state == "queued"
    ihc_alignment_service._jobs.pop(ihc_alignment_service.key(HE, IHC), None)
