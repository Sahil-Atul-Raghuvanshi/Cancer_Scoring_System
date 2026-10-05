"""Resolves a case folder to its H&E slide and five IHC slides, and loads one.

A case is one directory holding up to six `.svs` files (H&E plus the five
antibody letters, see `app.panel`). `resolve_case` is pure file discovery - it
answers "what is in this folder", nothing more. `load_case` additionally
registers the H&E slide and the chosen marker's slide as local uploads (see
`upload_service.register_local_slide`) and remembers the pairing in a JSON
sidecar under `data/demo/cases/`, the same sidecar-not-database convention
`upload_service` already uses.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.panel import PANEL, infer_from_filename, spec
from app.services import upload_service
from app.services.upload_service import UploadError


class CaseError(ValueError):
    """A client-correctable case-folder problem: bad path, missing slide, bad marker."""


@dataclass
class CaseResolution:
    case_id: str
    case_path: str
    found: dict[str, str]  # letter (or "HE") -> absolute slide path
    missing: list[str]


@dataclass
class CaseSession:
    case_id: str
    marker: str
    case_path: str
    he_upload_id: str
    ihc_upload_id: str
    created_at: str


def _cases_dir() -> Path:
    directory = settings.data_dir / "cases"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _sidecar_path(case_id: str, marker: str) -> Path:
    return _cases_dir() / f"{case_id}_{marker}.json"


def resolve_case(case_path: str) -> CaseResolution:
    """What `case_path` actually holds, matched against the panel's letters."""
    folder = Path(case_path)
    if not folder.is_dir():
        raise CaseError(f"'{case_path}' is not a directory")

    found: dict[str, str] = {}
    for entry in sorted(folder.iterdir()):
        if not entry.is_file() or entry.suffix.lower() not in settings.allowed_slide_ext:
            continue
        letter = infer_from_filename(entry.name)
        # First match wins: two files that both look like the same letter is a
        # folder that should be looked at by a person, not resolved by a guess.
        if letter is None or letter in found:
            continue
        found[letter] = str(entry.resolve())

    missing = sorted(letter for letter in PANEL if letter not in found)
    return CaseResolution(
        case_id=folder.name, case_path=str(folder.resolve()), found=found, missing=missing
    )


def load_case(case_path: str, marker: str) -> CaseSession:
    """Register the H&E slide and one marker's slide from `case_path`.

    Re-loading the same case+marker reuses the earlier upload ids rather than
    writing a fresh sidecar and a fresh pair of records every time - the
    underlying files have not moved, so nothing needs re-registering.
    """
    letter = marker.upper()
    if letter not in PANEL or letter == "HE":
        scorable = sorted(key for key in PANEL if key != "HE")
        raise CaseError(f"'{marker}' is not a scoreable marker; expected one of {scorable}")

    resolution = resolve_case(case_path)
    if "HE" in resolution.missing:
        raise CaseError(f"no H&E slide found in '{resolution.case_path}'")
    if letter in resolution.missing:
        raise CaseError(
            f"no {spec(letter).name} ({letter}) slide found in '{resolution.case_path}'"
        )

    existing = _load_sidecar(resolution.case_id, letter)
    if (
        existing
        and existing.case_path == resolution.case_path
        and upload_service.local_slide_still_valid(existing.he_upload_id)
        and upload_service.local_slide_still_valid(existing.ihc_upload_id)
    ):
        return existing

    try:
        he_upload_id = upload_service.register_local_slide(resolution.found["HE"])
        ihc_upload_id = upload_service.register_local_slide(resolution.found[letter])
    except UploadError as exc:
        raise CaseError(str(exc)) from exc

    session = CaseSession(
        case_id=resolution.case_id,
        marker=letter,
        case_path=resolution.case_path,
        he_upload_id=he_upload_id,
        ihc_upload_id=ihc_upload_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    _sidecar_path(session.case_id, session.marker).write_text(
        json.dumps(asdict(session), indent=2), encoding="utf-8"
    )
    return session


def _load_sidecar(case_id: str, marker: str) -> CaseSession | None:
    path = _sidecar_path(case_id, marker)
    if not path.exists():
        return None
    try:
        return CaseSession(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


__all__ = ["CaseError", "CaseResolution", "CaseSession", "load_case", "resolve_case"]
