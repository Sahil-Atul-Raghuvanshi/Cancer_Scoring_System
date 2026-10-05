"""Which versioned tree - `v1_data/`, `v2_data/`, ... - the code reads and writes.

Everything that is not code lives under one gitignored folder, `storage/`, at the
workspace root, which keeps two kinds of storage apart:

  storage/
    data/                 SHARED by every version. The inputs nothing regenerates:
                          data/original/ (OncoStem slides, BRACS, BACH, BCSS,
                          annotations, reader scores) and the OncoStem documents.
                          Read-only as far as the pipeline is concerned.

    v<N>_data/            ONE PER VERSION of the pipeline. Everything a version
        version.json      which storage layout it was written with (see LAYOUT)
        VERSION.md        what the version is and how it differs
        data/             produces: demo runtime state, history, registration,
                          scoring runs, training stores, review-report inputs.
        models/           the checkpoints this version scores with
        results/          its deliverables (CSVs, DECISIONS.md, review .docx)
        reports/          write-ups about this version (PIPELINE_PROBLEMS_* ...)
    decrecated_code/      retired code, kept for the record and never run
    ACTIVE_DATA_VERSION   this machine's chosen version

The version asked for is, in order:

    1. the CSS_DATA_VERSION environment variable, if set (e.g. `v2`)
    2. the storage/ACTIVE_DATA_VERSION file - written by the demo's version picker
    3. nothing - in which case the LATEST openable version runs

**Code moves on; old data stays.** A version can be opened only if the code still
reads the layout it was written with (`SUPPORTED_LAYOUTS`). When the version asked
for cannot be opened, nothing is deleted or converted: its folder stays exactly as
it is, and the latest version that *can* be opened runs instead. `resolve()` says
which was asked for, which runs, and why, so the app can tell the viewer.

`models/` is the one per-version folder that is *inherited*: a version without
its own `models/` uses the newest earlier version's, so a v2 that changes only
the scoring does not need a 2 GB copy of unchanged checkpoints.

Every module that needs a path imports this one rather than writing
`WORKSPACE_ROOT / "data"` itself, so there is exactly one place that knows the
layout. See STORAGE_VERSIONING.md for how to start a new version.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent

#: The one gitignored folder that holds every byte that is not code.
STORAGE_ROOT = WORKSPACE_ROOT / "storage"

#: Inputs shared by every version - never written by a pipeline run.
SHARED_DATA_ROOT = STORAGE_ROOT / "data"
ORIGINAL_ROOT = SHARED_DATA_ROOT / "original"

ENV_VAR = "CSS_DATA_VERSION"
SELECTION_FILE = STORAGE_ROOT / "ACTIVE_DATA_VERSION"
MANIFEST_NAME = "version.json"

#: The storage layout this code WRITES. Bump it when a code change means the files
#: an older version wrote can no longer be read - a renamed step folder, a changed
#: record format - and record the change in STORAGE_VERSIONING.md.
LAYOUT = 1

#: The layouts this code can still READ. Drop an old number when support for it is
#: removed; versions written with it are then kept on disk but no longer opened.
SUPPORTED_LAYOUTS = frozenset({1})

_VERSION = re.compile(r"^v([1-9][0-9]*)$")
_FOLDER = re.compile(r"^(v[1-9][0-9]*)_data$")


def _number(version: str) -> int:
    match = _VERSION.match(version)
    if not match:
        raise ValueError(f"{version!r} is not a data version - expected v1, v2, ...")
    return int(match.group(1))


def available() -> list[str]:
    """Every `storage/v<N>_data/` folder, oldest first - openable or not."""
    if not STORAGE_ROOT.is_dir():
        return []
    found = [
        match.group(1)
        for child in STORAGE_ROOT.iterdir()
        if child.is_dir() and (match := _FOLDER.match(child.name))
    ]
    return sorted(found, key=_number)


def version_root(version: str | None = None) -> Path:
    """`<workspace>/storage/v<N>_data/`."""
    return STORAGE_ROOT / f"{version or active()}_data"


# --- can this code open a version? -----------------------------------------


def layout_of(version: str) -> int:
    """The layout recorded in the version's version.json.

    A folder without one was made by hand alongside this code, so it is taken to be
    in the current layout. An unreadable manifest is layout 0, which nothing supports.
    """
    path = version_root(version) / MANIFEST_NAME
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["layout"])
    except FileNotFoundError:
        return LAYOUT
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def why_not_openable(version: str) -> str | None:
    """None when this code can open `version`; otherwise, in plain words, why not."""
    if not version_root(version).is_dir():
        return f"storage/{version}_data does not exist"
    layout = layout_of(version)
    if layout not in SUPPORTED_LAYOUTS:
        if layout > LAYOUT:
            return (
                f"{version} was made by a newer version of the app (storage layout "
                f"{layout}); update the code to open it"
            )
        return (
            f"{version} was made by an older version of the app (storage layout "
            f"{layout}) that this code no longer reads"
        )
    return None


def openable() -> list[str]:
    """The versions this code can open, oldest first."""
    return [v for v in available() if why_not_openable(v) is None]


def latest() -> str | None:
    """The newest version this code can open."""
    versions = openable()
    return versions[-1] if versions else None


# --- which version runs ------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    #: The version that runs. Always set: with no openable version at all it is
    #: the next one, which `ensure_version` creates.
    active: str
    #: What the env var or selection file asked for, if anything.
    requested: str | None
    #: Why `requested` is not the one running, in plain words. None when it is.
    fallback_reason: str | None
    #: True when CSS_DATA_VERSION decided `requested`.
    pinned_by_env: bool


def _requested() -> tuple[str | None, bool]:
    value = os.environ.get(ENV_VAR, "").strip()
    if value:
        return (value if _VERSION.match(value) else None), True
    try:
        text = SELECTION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None, False
    return (text if _VERSION.match(text) else None), False


def resolve() -> Resolution:
    """Which version to run: the one asked for if it opens, else the latest that does."""
    requested, pinned = _requested()
    if requested and why_not_openable(requested) is None:
        return Resolution(requested, requested, None, pinned)
    newest = latest()
    if newest is None:
        existing = available()
        newest = f"v{_number(existing[-1]) + 1}" if existing else "v1"
    reason = None
    if requested:
        reason = (
            f"{why_not_openable(requested)}. Its data is kept, untouched, in "
            f"storage/{requested}_data. Showing {newest}, the latest version."
        )
    return Resolution(newest, requested, reason, pinned)


def active() -> str:
    """The version this process reads and writes."""
    return resolve().active


def pinned_by_env() -> str | None:
    """The version CSS_DATA_VERSION asks for, if it is set."""
    requested, pinned = _requested()
    return requested if pinned else None


def select(version: str) -> None:
    """Make `version` the one every process started from now on runs."""
    reason = why_not_openable(version)
    if reason:
        raise ValueError(reason)
    SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_FILE.write_text(version + "\n", encoding="utf-8")


def ensure_version(version: str, based_on: str | None = None) -> Path:
    """Create `storage/<version>_data/` with its manifest, if it is not there yet."""
    root = version_root(version)
    for sub in ("data", "results", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    manifest = root / MANIFEST_NAME
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                {
                    "version": version,
                    "layout": LAYOUT,
                    "created": date.today().isoformat(),
                    "based_on": based_on,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return root


def create_next() -> str:
    """Start the next version: an empty `v<N+1>_data/` that inherits the models."""
    existing = available()
    version = f"v{_number(existing[-1]) + 1}" if existing else "v1"
    ensure_version(version, based_on=latest())
    return version


# --- paths inside a version --------------------------------------------------


def data_root(version: str | None = None) -> Path:
    """`v<N>_data/data/` - everything this version's runs produce."""
    return version_root(version) / "data"


def results_root(version: str | None = None) -> Path:
    """`v<N>_data/results/` - this version's deliverables."""
    return version_root(version) / "results"


def reports_root(version: str | None = None) -> Path:
    """`v<N>_data/reports/` - write-ups about this version."""
    return version_root(version) / "reports"


def models_root(version: str | None = None) -> Path:
    """This version's `models/`, or the newest earlier version's if it has none.

    Falls back to the version's own (possibly absent) folder when no version
    has one, so callers that create it still create it in the right place.
    """
    version = version or active()
    own = version_root(version) / "models"
    if own.is_dir():
        return own
    for older in sorted(
        (v for v in available() if _number(v) < _number(version)), key=_number, reverse=True
    ):
        candidate = version_root(older) / "models"
        if candidate.is_dir():
            return candidate
    return own


def publish_models_root(version: str | None = None) -> Path:
    """Where a training run may WRITE checkpoints: this version's own `models/`.

    Never the inherited folder - publishing into it would overwrite the models an
    earlier version was scored with. And inheritance is all-or-nothing per version,
    so a version cannot add one checkpoint beside inherited ones: copy the whole
    `models/` across first (STORAGE_VERSIONING.md, "Starting a new version").
    """
    version = version or active()
    own = version_root(version) / "models"
    if not own.is_dir():
        raise FileNotFoundError(
            f"{version}_data has no models/ of its own, so it is using "
            f"{models_root(version)}. Copy that folder to {own} before publishing a "
            "checkpoint, or the earlier version's models would be overwritten."
        )
    return own


if __name__ == "__main__":
    # `python data_versions.py --active` prints just the version, for batch files.
    if "--active" in sys.argv:
        print(active())
        sys.exit(0)
    found = resolve()
    print(f"active    : {found.active}" + ("  (pinned by %s)" % ENV_VAR if found.pinned_by_env else ""))
    if found.fallback_reason:
        print(f"note      : {found.fallback_reason}")
    for version in available():
        reason = why_not_openable(version)
        print(f"  {version:<6} layout {layout_of(version)}  " + (f"NOT OPENABLE: {reason}" if reason else "ok"))
    print(f"shared    : {SHARED_DATA_ROOT}")
    print(f"data      : {data_root()}")
    print(f"models    : {models_root()}")
    print(f"results   : {results_root()}")
