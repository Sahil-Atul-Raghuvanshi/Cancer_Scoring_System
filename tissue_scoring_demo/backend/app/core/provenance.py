"""What a result was made with, stamped onto it (P-15).

A score on disk said nothing about how it was produced: not the code, not the settings,
not the cut-point file, not the model that drew its regions, not the transform that
carried them onto the IHC. So a result made last week and one made after a fix looked
identical, and `run_all.py` treated a finished case as finished for ever - a re-run
after any change silently kept the old numbers.

Two parts, kept apart because they are checked differently:

  `code_state()`   the same for every pair: commit, uncommitted changes, settings and
                   cut file. Cheap, and what `run_all.py` compares to decide whether a
                   `done` case is still done.
  `for_pair()`     adds what is specific to one H&E/IHC pair: the step 8 head and the
                   class map it produced, and the registration transform.

A field that cannot be determined is recorded as None rather than guessed, and None is
a mismatch against a known value - an unknown provenance is not evidence of a match.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import settings

#: The repository root - `backend/app/core` is three levels below `tissue_scoring_demo`.
_REPO = Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def _git() -> tuple[str | None, bool | None]:
    """(commit, dirty) of the checked-out tree, or (None, None) without git.

    Cached for the process: the code a process runs does not change while it runs, and
    `--reload` starts a new process when it does.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True,
            timeout=30, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=_REPO,
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit or None, bool(status)


def config_sha256() -> str:
    """A hash of every setting that is not a filesystem location.

    Paths are left out on purpose: they differ between machines and between data
    versions without changing what any step computes, and including them would make
    every result look stale the moment the repository was moved.
    """
    values = {
        key: value
        for key, value in settings.model_dump().items()
        if not isinstance(value, Path)
    }
    payload = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def code_state() -> dict[str, Any]:
    """The pair-independent half of a stamp."""
    from app.scoring import cuts as cut_points

    commit, dirty = _git()
    return {
        "code_commit": commit,
        "code_dirty": dirty,
        "config_sha256": config_sha256(),
        "cuts_version": cut_points.cut_set().version,
    }


def for_pair(he_upload_id: str, ihc_upload_id: str) -> dict[str, Any]:
    """A full stamp for one pair's result."""
    from app.services.tissue_type_service import tissue_type_service

    stamp = code_state()

    tissue_type = tissue_type_service._read_json(he_upload_id, "report.json") or {}
    run = tissue_type.get("_key") or {}
    stamp["tissue_type_model"] = run.get("model")
    stamp["tissue_type_model_sha256"] = run.get("model_sha256")
    stamp["class_map_key"] = tissue_type_service.class_map_key(he_upload_id)

    try:
        from app.services.ihc_alignment_service import stored_transforms

        stamp["transform_sha256"] = stored_transforms.fingerprint(he_upload_id, ihc_upload_id)
    except Exception:  # noqa: BLE001 - an unreadable transform is an unknown, not a crash
        stamp["transform_sha256"] = None
    return stamp


#: The fields `stale_fields` compares. `code_dirty` is recorded but not compared: a
#: result made from a dirty tree is flagged by the field itself, and comparing it would
#: make every result stale the moment a file is opened in an editor.
COMPARED = ("code_commit", "config_sha256", "cuts_version")


def stale_fields(recorded: dict[str, Any] | None, current: dict[str, Any]) -> list[str]:
    """Which compared fields differ - every one of them when nothing was recorded."""
    if not recorded:
        return list(COMPARED)
    return [key for key in COMPARED if recorded.get(key) != current.get(key)]


__all__ = ["COMPARED", "code_state", "config_sha256", "for_pair", "stale_fields"]
