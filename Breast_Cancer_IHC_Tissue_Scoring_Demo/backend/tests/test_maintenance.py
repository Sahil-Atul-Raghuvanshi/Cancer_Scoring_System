"""Storage housekeeping tests.

Every path in this module deletes files, so the tests are mostly about what it must
*not* delete. Three are worth naming:

`test_a_sweep_is_a_dry_run_unless_confirmed` pins the default that stops a mistyped
scope destroying an upload; `test_derived_never_touches_the_slide` pins the boundary
between "costs a re-run" and "costs a re-upload"; and
`test_a_cache_whose_slide_is_gone_is_the_only_thing_swept_unattended` pins the rule that
makes the start-up sweep safe to run without asking anyone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services import maintenance_service


def _slide(upload_id: str, *, megabytes: int = 1) -> None:
    """A published slide and its record, big enough that sizes are legible."""
    settings.slides_dir.mkdir(parents=True, exist_ok=True)
    (settings.slides_dir / f"{upload_id}.svs").write_bytes(b"\0" * (megabytes * 1024 * 1024))
    (settings.slides_dir / f"{upload_id}.json").write_text(
        json.dumps({"upload_id": upload_id, "state": "ready"}), encoding="utf-8"
    )


def _cache(upload_id: str, step_dir: Path, *, kilobytes: int = 64) -> Path:
    directory = step_dir / upload_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_bytes(b"\0" * (kilobytes * 1024))
    return directory


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A whole data tree of its own, so a test can never delete real slides."""
    for name, attribute in (
        ("slides", "slides_dir"),
        ("uploads", "uploads_dir"),
        ("qc", "qc_dir"),
        ("tissue", "tissue_dir"),
        ("calibration", "calibration_dir"),
        ("tissue_type", "tissue_type_dir"),
    ):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(settings, attribute, directory)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


# --- measuring ----------------------------------------------------------------


def test_usage_separates_slides_from_caches_and_reports_the_share(store: Path):
    """The share is the number that decides what to do, so it is reported outright.

    A single "cached data" total would suggest clearing caches saves space. On the real
    machine slides are 98.8 % of the tree, and at that ratio they are the only thing
    worth deleting.
    """
    _slide("alpha", megabytes=8)
    _cache("alpha", settings.tissue_dir, kilobytes=64)
    _cache("alpha", settings.tissue_type_dir, kilobytes=128)

    report = maintenance_service.usage()

    assert report.slide_bytes > report.derived_bytes * 10
    assert report.slide_share > 0.9
    assert report.total_bytes == (
        report.slide_bytes + report.derived_bytes + report.staging_bytes
    )
    assert set(report.uploads) == {"alpha"}
    assert report.uploads["alpha"]["derived"].keys() == {
        "tissue-mask",
        "tissue-type-segmentation",
    }


def test_a_cache_whose_slide_is_gone_is_reported_as_orphaned(store: Path):
    _slide("alpha")
    _cache("alpha", settings.tissue_dir)
    _cache("ghost", settings.tissue_dir)

    report = maintenance_service.usage()
    orphans = {item.upload_id for item in report.items if item.orphaned}

    assert orphans == {"ghost"}
    assert report.orphan_bytes > 0


# --- sweeping -----------------------------------------------------------------


def test_a_sweep_is_a_dry_run_unless_confirmed(store: Path):
    """The default that stops a mistyped scope destroying an upload."""
    _slide("alpha", megabytes=4)
    _cache("alpha", settings.tissue_type_dir)

    result = maintenance_service.sweep("slide", upload_id="alpha")

    assert result.dry_run is True
    assert result.freed_bytes > 0
    assert (settings.slides_dir / "alpha.svs").exists(), "a dry run must delete nothing"
    assert (settings.tissue_type_dir / "alpha").exists()


def test_derived_never_touches_the_slide(store: Path):
    """The boundary between 'costs a re-run' and 'costs a re-upload'.

    Half an hour of CPU is a real loss; a gigabyte re-upload is a different kind of loss,
    and a scope that quietly did both would be the worst possible surprise.
    """
    _slide("alpha", megabytes=4)
    _cache("alpha", settings.tissue_dir)
    _cache("alpha", settings.tissue_type_dir)

    maintenance_service.sweep("derived", upload_id="alpha", dry_run=False)

    assert (settings.slides_dir / "alpha.svs").exists()
    assert (settings.slides_dir / "alpha.json").exists()
    assert not (settings.tissue_dir / "alpha").exists()
    assert not (settings.tissue_type_dir / "alpha").exists()


def test_a_cache_whose_slide_is_gone_is_the_only_thing_swept_unattended(store: Path):
    """What makes the start-up sweep safe to run without asking anyone.

    It must remove the unreachable cache and leave every reachable one alone - otherwise
    a restart would silently cost half an hour of step 8 on a slide still in use.
    """
    _slide("alpha")
    _cache("alpha", settings.tissue_type_dir)
    _cache("ghost", settings.tissue_type_dir)

    result = maintenance_service.sweep_orphans_on_start()

    assert not (settings.tissue_type_dir / "ghost").exists()
    assert (settings.tissue_type_dir / "alpha").exists()
    assert (settings.slides_dir / "alpha.svs").exists()
    assert result.freed_bytes > 0


def test_sweeping_one_slide_leaves_the_others_alone(store: Path):
    _slide("alpha", megabytes=2)
    _slide("beta", megabytes=2)
    _cache("alpha", settings.tissue_dir)
    _cache("beta", settings.tissue_dir)

    maintenance_service.sweep("slide", upload_id="alpha", dry_run=False)

    assert not (settings.slides_dir / "alpha.svs").exists()
    assert not (settings.tissue_dir / "alpha").exists()
    assert (settings.slides_dir / "beta.svs").exists()
    assert (settings.tissue_dir / "beta").exists()


def test_all_empties_the_tree_but_keeps_the_directories(store: Path):
    """The directories have to survive, or the next upload fails on a missing path."""
    _slide("alpha", megabytes=2)
    _cache("alpha", settings.qc_dir)

    maintenance_service.sweep("all", dry_run=False)

    assert list(settings.slides_dir.glob("*")) == []
    assert list(settings.qc_dir.glob("*")) == []
    assert settings.slides_dir.is_dir()
    assert settings.qc_dir.is_dir()


def test_a_scope_that_needs_a_slide_refuses_without_one(store: Path):
    for scope in ("derived", "slide"):
        with pytest.raises(maintenance_service.MaintenanceError, match="needs an upload id"):
            maintenance_service.sweep(scope, dry_run=False)


def test_an_unknown_scope_is_refused_rather_than_guessed(store: Path):
    with pytest.raises(maintenance_service.MaintenanceError, match="unknown scope"):
        maintenance_service.sweep("everything", dry_run=False)


def test_retention_reports_stale_slides_but_never_deletes_them(store: Path):
    """`stale_uploads` is a report. Nothing acts on it automatically, at any setting.

    "Old" is a judgement about how a machine is used, and a demo slide somebody returns
    to next month is not waste.
    """
    _slide("alpha")
    assert maintenance_service.stale_uploads(0) == []
    assert maintenance_service.stale_uploads(365) == []

    # Everything is stale under a zero-length window, and it is still only a report.
    assert maintenance_service.stale_uploads(-1) == []
    assert (settings.slides_dir / "alpha.svs").exists()


# --- the endpoints ------------------------------------------------------------


def test_storage_endpoint_reports_the_slide_share(client: TestClient):
    response = client.get("/api/v1/maintenance/storage")
    assert response.status_code == 200

    body = response.json()
    assert "slideShare" in body
    assert 0.0 <= body["slideShare"] <= 1.0
    assert body["totalBytes"] >= body["slideBytes"]


def test_cleanup_defaults_to_a_dry_run(client: TestClient):
    response = client.post("/api/v1/maintenance/cleanup?scope=orphans")
    assert response.status_code == 200
    assert response.json()["dryRun"] is True


def test_cleanup_refuses_a_scope_without_the_slide_it_needs(client: TestClient):
    response = client.post("/api/v1/maintenance/cleanup?scope=derived")
    assert response.status_code == 409


def test_cleanup_rejects_an_unknown_scope_at_the_boundary(client: TestClient):
    response = client.post("/api/v1/maintenance/cleanup?scope=nuke")
    assert response.status_code == 422


def test_the_upload_id_query_parameter_binds_as_camel_case(client: TestClient):
    """The wire name is `uploadId`, like every other query parameter in this API.

    Without the alias the parameter silently does not bind, and the endpoint answers
    "this scope needs an upload id" to a caller who sent one. Pinned because the failure
    reads as a server bug from the client side and as a client bug from the server side.
    """
    snake = client.post("/api/v1/maintenance/cleanup?scope=derived&upload_id=whatever")
    camel = client.post("/api/v1/maintenance/cleanup?scope=derived&uploadId=whatever")

    assert camel.status_code == 200, camel.text
    assert snake.status_code == 409, "the snake_case spelling must not bind"


# --- releasing a slide when the browser goes away ------------------------------


def test_a_disconnect_does_nothing_while_the_feature_is_off(
    store: Path, monkeypatch: pytest.MonkeyPatch
):
    """Off by default, and off means nothing is even scheduled."""
    monkeypatch.setattr(settings, "cleanup_on_disconnect", False)
    assert maintenance_service.release("alpha") is None
    assert maintenance_service.pending_releases() == {}


def test_a_refresh_cancels_the_release_it_just_caused(
    store: Path, monkeypatch: pytest.MonkeyPatch
):
    """The whole safety argument for this feature, in one test.

    `pagehide` fires on a refresh exactly as it does on a close, so the disconnect is a
    guess. What makes the guess survivable is that it only *schedules*: the page that
    reloads claims the slide within a second and the release never runs. Without this,
    pressing F5 would cost half an hour of step 8.
    """
    monkeypatch.setattr(settings, "cleanup_on_disconnect", True)
    monkeypatch.setattr(settings, "cleanup_disconnect_grace_seconds", 60.0)
    _slide("alpha")
    _cache("alpha", settings.tissue_type_dir)

    maintenance_service.release("alpha")
    assert "alpha" in maintenance_service.pending_releases()

    assert maintenance_service.claim("alpha") is True
    assert maintenance_service.pending_releases() == {}

    maintenance_service._run_due_releases()
    assert (settings.tissue_type_dir / "alpha").exists(), "a claimed slide must survive"


def test_an_unclaimed_release_frees_the_caches_but_never_the_slide(
    store: Path, monkeypatch: pytest.MonkeyPatch
):
    """A guess may cost a re-run. It may never cost a re-upload."""
    monkeypatch.setattr(settings, "cleanup_on_disconnect", True)
    monkeypatch.setattr(settings, "cleanup_disconnect_grace_seconds", 0.0)
    _slide("alpha", megabytes=2)
    _cache("alpha", settings.tissue_type_dir)

    maintenance_service.release("alpha")
    assert maintenance_service._run_due_releases() == 1

    assert not (settings.tissue_type_dir / "alpha").exists()
    assert (settings.slides_dir / "alpha.svs").exists(), "a disconnect must not delete a slide"


def test_a_disconnect_cannot_be_configured_to_delete_a_slide(
    store: Path, monkeypatch: pytest.MonkeyPatch
):
    """The blast radius is clamped in code, not left to configuration.

    The trigger is a guess about intent; a setting that let a wrong guess destroy a
    1.5 GB upload would be a footgun no amount of documentation fixes.
    """
    monkeypatch.setattr(settings, "cleanup_on_disconnect", True)
    monkeypatch.setattr(settings, "cleanup_disconnect_grace_seconds", 0.0)
    monkeypatch.setattr(settings, "cleanup_disconnect_scope", "slide")
    _slide("alpha", megabytes=2)
    _cache("alpha", settings.tissue_dir)

    maintenance_service.release("alpha")
    maintenance_service._run_due_releases()

    assert (settings.slides_dir / "alpha.svs").exists()
    assert not (settings.tissue_dir / "alpha").exists()


def test_the_grace_period_is_respected(store: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "cleanup_on_disconnect", True)
    monkeypatch.setattr(settings, "cleanup_disconnect_grace_seconds", 300.0)
    _slide("alpha")
    _cache("alpha", settings.tissue_dir)

    maintenance_service.release("alpha")
    assert maintenance_service._run_due_releases() == 0, "not due yet"
    assert (settings.tissue_dir / "alpha").exists()


def test_release_answers_200_even_when_off(client: TestClient):
    """A beacon cannot read a status code, and a closing page cannot act on an error."""
    response = client.post("/api/v1/maintenance/release?uploadId=whatever")
    assert response.status_code == 200
    assert response.json()["scheduled"] is False


def test_policy_is_cheap_and_says_whether_to_bother(client: TestClient):
    response = client.get("/api/v1/maintenance/policy")
    assert response.status_code == 200

    body = response.json()
    assert body["cleanupOnDisconnect"] is False, "must ship off"
    assert body["scope"] != "slide"
    assert body["graceSeconds"] > 0
