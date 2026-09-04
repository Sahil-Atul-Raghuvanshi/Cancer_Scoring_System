"""Shared pytest fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def _isolated_storage(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point upload storage at a temp directory for the whole test session.

    Without this the suite writes real slides into the project's `data/`
    directory and leaves them there - the tests would slowly fill the disk with
    multi-hundred-megabyte artefacts and could collide with a running dev
    server's uploads.
    """
    root: Path = tmp_path_factory.mktemp("ihc-storage")

    original = (
        settings.data_dir,
        settings.uploads_dir,
        settings.slides_dir,
        settings.qc_dir,
        settings.tissue_dir,
        settings.calibration_dir,
        settings.tissue_type_dir,
    )
    settings.data_dir = root
    settings.uploads_dir = root / "uploads"
    settings.slides_dir = root / "slides"
    # QC caches a report, a grid and four PNGs per slide. Left pointing at the
    # real data directory, the suite would litter it with results for uploads
    # that no longer exist.
    settings.qc_dir = root / "qc"
    # Steps 3 and 4 cache per slide too - a saturation channel and an RGB
    # thumbnail. Left pointing at the real data directory, the suite would
    # litter it with bases for uploads that no longer exist.
    settings.tissue_dir = root / "tissue"
    settings.calibration_dir = root / "calibration"
    # Step 8 caches a report, a class map and four PNGs per slide. Left pointing at
    # the real data directory, the suite would litter it with class maps for uploads
    # that no longer exist.
    settings.tissue_type_dir = root / "tissue_type"
    settings.ensure_dirs()

    yield

    (
        settings.data_dir,
        settings.uploads_dir,
        settings.slides_dir,
        settings.qc_dir,
        settings.tissue_dir,
        settings.calibration_dir,
        settings.tissue_type_dir,
    ) = original


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Return a TestClient bound to the application."""
    return TestClient(app)
