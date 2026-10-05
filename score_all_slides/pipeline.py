"""Shared ground for the overnight batch: where things are, and what a result is.

Imported by `run_all.py` (the orchestrator), `run_case.py` (the per-case worker) and
`watch.py` (the read-only status view). Nothing here runs the pipeline; it owns the
paths, the checkpoint and the one CSV, so that three processes which never talk to each
other directly still agree on what has happened.

**The checkpoint is the contract between runs, not a log.** A pass over six cases is
longer than the night it is started in, and the machine may be rebooted under it. So
every fact the next run needs to avoid repeating work is written to `state/checkpoint.json`
the moment it becomes true, and the orchestrator reads it before deciding anything. The
logs are for a person; the checkpoint is for the program.

**The CSV is rebuilt from the per-case JSON records, never appended to.** Appending would
mean a case run twice appears twice, and a run interrupted mid-write leaves a half-row
that a spreadsheet reads as data. Rebuilding is cheap - thirty rows - and it makes the
CSV a pure function of `results/*.json`, so it is correct after a crash rather than
merely usually correct.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import sys
import tempfile
import time

# --- where everything lives --------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402

BACKEND = ROOT / "tissue_scoring_demo" / "backend"
PYTHON = BACKEND / ".venv" / "Scripts" / "python.exe"
SLIDES = data_versions.ORIGINAL_ROOT / "oncostem_slides"

HERE = pathlib.Path(__file__).resolve().parent

#: Run output - checkpoint, per-case records, logs - lives under `v<N>_data/data/` with every
#: other artefact, so this folder holds only code.
OUT = data_versions.data_root() / "score_all_slides"
STATE = OUT / "state"
LOGS = OUT / "logs"
RESULTS = OUT / "results"

CHECKPOINT = STATE / "checkpoint.json"
RUN_LOG = LOGS / "run_all.log"

#: Which region source this pass is scoring on. `refined` is step 11's BEETLE pixel
#: boundaries; `tiles` skips step 11 and carries step 9's coarse squares.
#:
#: **The two modes keep entirely separate state, results and CSV**, and that is the
#: point. They are measured inside different tissue, so interleaving them in one
#: checkpoint would let a case scored one way satisfy the other's "already done" check,
#: and interleaving them in one CSV would produce a file whose rows mean two different
#: things with nothing on the row to say which.
MODE = "refined"

#: The one file that puts every case's every marker together. In `v<N>_data/results/`, beside
#: the other latest deliverables, because that is where a person will look for it.
CSV_PATH = data_versions.results_root() / "oncostem_ai_scores.csv"

#: The five stained markers. `HE` is the reference section and is never scored - it is
#: what steps 2 to 9 run on and what step 10 carries the regions across from.
MARKERS = ("A", "F", "R", "U", "W")


def configure(mode: str) -> None:
    """Point the checkpoint, the per-case records and the CSV at one mode's own files.

    Called once, before anything else, by whichever driver is starting. Everything else
    in this module reads the module-level paths, so there is exactly one place the two
    modes diverge.
    """
    global MODE, STATE, RESULTS, CHECKPOINT, CSV_PATH, RUN_LOG
    MODE = mode
    suffix = "" if mode == "refined" else f"_{mode}"
    STATE = OUT / f"state{suffix}"
    RESULTS = OUT / f"results{suffix}"
    CHECKPOINT = STATE / "checkpoint.json"
    RUN_LOG = LOGS / f"run_all{suffix}.log"
    CSV_PATH = data_versions.results_root() / f"oncostem_ai_scores{suffix}.csv"


def cases() -> list[str]:
    """Every case folder under the slide library, in name order.

    Read from disk rather than listed in this file, so a case added to the library is
    picked up by the next run without an edit here.
    """
    if not SLIDES.is_dir():
        return []
    return [one.name for one in sorted(SLIDES.iterdir()) if one.is_dir()]


def ensure_dirs() -> None:
    for directory in (STATE, LOGS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)


# --- the worker process (P-19) -------------------------------------------------
#
# `run_all.py` starts one `run_case.py` per case, and that worker may start children of
# its own. Two things need to know exactly which processes those are: `keepalive.py`,
# which must judge liveness by *this case's* CPU rather than by any python on the
# machine, and the watchdog, which must stop the whole tree - `Popen.kill()` on Windows
# ends the one process and leaves its children running.


def worker_pid_path(case_id: str) -> pathlib.Path:
    """Where `run_all.py` records the pid of the worker it started for a case."""
    return STATE / f"{case_id}.pid"


def worker_pid(case_id: str) -> int | None:
    try:
        return int(worker_pid_path(case_id).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def process_tree(root: int) -> set[int]:
    """`root` and every process descended from it, from one Toolhelp snapshot.

    ctypes rather than psutil, which is not in the backend's environment. A pid whose
    parent has exited can be reused by Windows, so a descendant is only counted when it
    is reached from `root` itself - never by matching a parent pid that happens to be
    free again.
    """
    import ctypes
    from ctypes import wintypes

    class _Entry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    if snapshot in (None, wintypes.HANDLE(-1).value):
        return {root}

    children: dict[int, list[int]] = {}
    try:
        entry = _Entry()
        entry.dwSize = ctypes.sizeof(_Entry)
        more = kernel32.Process32First(snapshot, ctypes.byref(entry))
        while more:
            children.setdefault(entry.th32ParentProcessID, []).append(entry.th32ProcessID)
            more = kernel32.Process32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    tree, frontier = {root}, [root]
    while frontier:
        for child in children.get(frontier.pop(), []):
            if child not in tree:
                tree.add(child)
                frontier.append(child)
    return tree


def kill_tree(pid: int) -> None:
    """Stop a worker and everything it started. `/T` is the part `Popen.kill` lacks."""
    import subprocess

    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        timeout=120,
        check=False,
    )


# --- saying things -----------------------------------------------------------


def say(message: str, log: pathlib.Path | None = None) -> None:
    """One line, stamped, to the console and to a log if there is one.

    Flushed on every line. The orchestrator runs detached with its output redirected,
    and a buffered stream would leave a person watching the file seeing nothing for
    however long the buffer took to fill - which on a job this quiet is hours.
    """
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


# --- the checkpoint ----------------------------------------------------------


def _write_json(path: pathlib.Path, payload: object) -> None:
    """Write a JSON file the way a crash cannot corrupt.

    To a temporary file beside the target and then a replace, which is atomic on NTFS.
    Writing in place would mean a power cut during the write leaves the checkpoint
    truncated - and a truncated checkpoint is worse than none, because the next run
    would read it, fail to parse it, and start over from the first case.
    """
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


def load_checkpoint() -> dict:
    """The state of the pass, or a blank one.

    A checkpoint that cannot be parsed is treated as absent rather than fatal. The cost
    of being wrong that way is repeating a case; the cost of the other way is an
    overnight run that refuses to start at 2am with nobody there to answer it.
    """
    if CHECKPOINT.is_file():
        try:
            return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Unreadable or unparsable is treated as absent, for the reason above. The
            # two are named rather than caught broadly so that a genuine programming
            # error here still surfaces instead of silently resetting the pass.
            pass
    return {"started": None, "cases": {}}


def save_checkpoint(state: dict) -> None:
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(CHECKPOINT, state)


def case_state(state: dict, case_id: str) -> dict:
    """One case's slot in the checkpoint, created if it is not there yet."""
    return state.setdefault("cases", {}).setdefault(
        case_id,
        {"state": "pending", "attempts": 0, "scored": 0, "failed": 0, "archived": False},
    )


# --- the per-case record and the CSV ----------------------------------------


def result_path(case_id: str) -> pathlib.Path:
    return RESULTS / f"{case_id}.json"


def save_result(case_id: str, payload: dict) -> None:
    _write_json(result_path(case_id), payload)


def load_results() -> list[dict]:
    """Every per-case record on disk, in case order. The CSV's only source."""
    records = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            # A record that cannot be read is skipped rather than fatal: it is one
            # case's worth of rows missing from a CSV that still has the other five,
            # and the run log says what happened to it.
            continue
    return records


#: Provenance fields carried into the CSV. Kept as their own tuple so the row builder
#: reads them out of the nested record rather than off the marker's top level.
PROVENANCE_COLUMNS = (
    "code_commit",
    "code_dirty",
    "config_sha256",
    "cuts_version",
    "tissue_type_model",
    "tissue_type_model_sha256",
    "transform_sha256",
)

#: The CSV's columns, in order. `percent` and `intensity` come first after the
#: identifiers because they are the deliverable - the two numbers OncoStem's readers
#: give - and everything after them is the working that produced them.
COLUMNS = (
    "case_id",
    "marker",
    "marker_name",
    "state",
    "percent",
    "intensity",
    "intensity_label",
    "percent_raw",
    "intensity_raw",
    "percent_pooled",
    "intensity_pooled",
    "compartment",
    "second_measure",
    "cells",
    "positive_cells",
    "h_score",
    "allred_proportion",
    "allred_intensity",
    "allred_total",
    "percent_area_weighted",
    "percent_plain_mean",
    "cuts_provisional",
    "caveats",
    "he_upload_id",
    "ihc_upload_id",
    "seconds",
    "scored_at",
    "error",
    # What the row was made with (P-15), flattened out of each marker's `provenance`.
    # Last, because they are the audit trail rather than the result.
    *PROVENANCE_COLUMNS,
)


def write_csv() -> pathlib.Path:
    """Rebuild the one CSV from every per-case record on disk.

    Rebuilt rather than appended for the reason in this module's docstring, and written
    through the same atomic replace as the checkpoint: a person may well open this file
    while the run is still going, and a half-written CSV opened in Excel is a file that
    looks like it has fewer results than it does.
    """
    # What the records actually say, keyed by the cell they belong to.
    found: dict[tuple[str, str], dict] = {}
    for record in load_results():
        case_id = record.get("caseId", "")
        for marker in record.get("markers", []):
            row = {column: marker.get(column, "") for column in COLUMNS}
            stamp = marker.get("provenance") or {}
            for column in PROVENANCE_COLUMNS:
                value = stamp.get(column)
                row[column] = "" if value is None else value
            row["case_id"] = case_id
            caveats = marker.get("caveats") or []
            row["caveats"] = " | ".join(caveats) if isinstance(caveats, list) else caveats
            found[(case_id, str(marker.get("marker", "")))] = row

    # **Only the rows that carry a measurement.**
    #
    # The file briefly held all thirty cells, scored or not, so that a missing row could
    # never be confused with a lost one. That is the wrong call for this deliverable:
    # this CSV *is* the scores, and a reader opening it should find numbers rather than
    # twenty-one rows explaining numbers that do not exist.
    #
    # Nothing is lost by leaving them out. `results/<case>.json` keeps every attempt with
    # its state and its reason, the checkpoint keeps each case's standing, and the logs
    # keep the detail - so what happened to an unscored pair is still answerable, just
    # not from here.
    rows = [row for row in found.values() if row.get("state") == "scored"]


    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(CSV_PATH.parent), suffix=".tmp")
    try:
        # newline="" is required of every csv writer on Windows; without it every row is
        # followed by a blank one.
        with os.fdopen(handle, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, CSV_PATH)
    except BaseException:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise
    return CSV_PATH
