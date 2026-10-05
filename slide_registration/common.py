"""Shared ground for the registration pass: paths, the checkpoint, and saying things.

Imported by every phase. Nothing here does registration; it owns the layout so that the
renderer, the order measurement, the stack registration and the read-only watcher all
agree on where things are without talking to each other.

**The checkpoint is the contract between runs, not a log.** Six cases is longer than one
night, and the machine may be rebooted under it. Every fact the next run needs in order
to avoid repeating work is written the moment it becomes true. Deliberately the same
shape as `score_all_slides/pipeline.py`, which has survived several overnight passes -
including an atomic write, because a truncated checkpoint is worse than none: the next
run reads it, fails to parse it, and starts from the first case.

**Layout is by case, never by pair.** Phase 3 produces one registrar for six slides, so
the old `<he>__<ihc>` directory has nothing to key on. That change also removes a known
hazard: the start-up orphan sweep read pair-keyed directories as orphans and deleted
483 MB on one restart.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402

#: This version's produced data - `v<N>_data/data/`. Inputs stay in the shared `data/original/`.
DATA = data_versions.data_root()
BACKEND = ROOT / "tissue_scoring_demo" / "backend"
PYTHON = BACKEND / ".venv" / "Scripts" / "python.exe"
VALIS_DIR = ROOT / "tissue_scoring_demo" / "valis_service"
VALIS_PYTHON = VALIS_DIR / ".venv" / "Scripts" / "python.exe"
SLIDES = data_versions.ORIGINAL_ROOT / "oncostem_slides"

WORK = DATA / "registration"
STATE = WORK / "_state"
LOGS = WORK / "_logs"

CHECKPOINT = STATE / "checkpoint.json"
RUN_LOG = LOGS / "run.log"
#: The latest deliverables - score and coverage CSVs, DECISIONS.md - kept together.
RESULTS = data_versions.results_root()
SUMMARY_CSV = RESULTS / "registration_summary.csv"

#: The five stained markers. `HE` is the reference section: it is what steps 2-9 run on
#: and the space every ROI is expressed in, so it is never itself a target.
MARKERS = ("A", "F", "R", "U", "W")

#: Marker letter -> what it stains. From the SOP reconstruction, section 3.
PANEL = {
    "A": "CD44",
    "F": "ABCC4",
    "R": "ABCC11",
    "U": "N-cadherin",
    "W": "pan-cadherin",
}


def case_dir(case: str) -> pathlib.Path:
    return WORK / case


def cases() -> list[str]:
    """Every case folder under the slide library, in name order.

    Read from disk rather than listed here, so a case added to the library is picked up
    by the next run without an edit. That is the whole point: this has to work on slides
    nobody has seen yet.
    """
    if not SLIDES.is_dir():
        return []
    return [one.name for one in sorted(SLIDES.iterdir()) if one.is_dir()]


def slides_for(case: str) -> dict[str, pathlib.Path]:
    """Map slide code -> file for one case, discovered from the filenames.

    Codes are recovered from the stem's last underscore- or hyphen-separated field, so
    both `CAN_00865_26-A.svs` and `CAN_00267_26_A.svs` resolve to `A`, and anything
    matching H&E/HE resolves to `HE`. Unknown codes are kept rather than dropped - a new
    panel member should show up as an unregistered slide, not vanish.
    """
    found: dict[str, pathlib.Path] = {}
    directory = SLIDES / case
    if not directory.is_dir():
        return found
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".svs", ".tif", ".tiff", ".ndpi"}:
            continue
        stem = path.stem
        tail = stem.replace("-", "_").split("_")[-1].upper()
        if tail in {"H&E", "HE"} or "H&E" in stem.upper():
            found["HE"] = path
        else:
            found[tail] = path
    return found


def ensure_dirs() -> None:
    for directory in (WORK, STATE, LOGS):
        directory.mkdir(parents=True, exist_ok=True)


# --- saying things -----------------------------------------------------------


def say(message: str, log: pathlib.Path | None = None) -> None:
    """One line, stamped, to the console and to a log if there is one.

    Flushed on every line. The driver runs detached with its output redirected, and a
    buffered stream leaves a person watching the file seeing nothing for however long the
    buffer takes to fill - which on a job this quiet is hours.
    """
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


# --- json that a crash cannot corrupt ----------------------------------------


def write_json(path: pathlib.Path, payload: object) -> None:
    """Write to a temporary file beside the target, then replace - atomic on NTFS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def read_json(path: pathlib.Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- the checkpoint ----------------------------------------------------------


def load_checkpoint() -> dict:
    """The state of the pass, or a blank one.

    A checkpoint that cannot be parsed is treated as absent rather than fatal. The cost
    of being wrong that way is repeating a case; the cost of the other way is an
    overnight run that refuses to start at 2am with nobody there to answer it.
    """
    payload = read_json(CHECKPOINT)
    if not isinstance(payload, dict):
        return {"cases": {}, "started": None}
    payload.setdefault("cases", {})
    return payload


def save_checkpoint(payload: dict) -> None:
    write_json(CHECKPOINT, payload)


def mark(case: str, **fields) -> dict:
    """Record a fact about a case and persist it immediately."""
    payload = load_checkpoint()
    row = payload["cases"].setdefault(case, {})
    row.update(fields)
    row["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_checkpoint(payload)
    return row


def status(case: str) -> dict:
    return load_checkpoint()["cases"].get(case, {})
