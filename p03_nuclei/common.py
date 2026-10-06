"""Shared paths, the benchmark, checkpoints and logging for the P-03 overnight run.

P-03: on the IHC slides the nucleus detector misses 35-98% of the nuclei the matching
H&E holds in the same regions, mostly the darkly stained ones. The review gives three
approaches; this run tries each of them on one fixed benchmark so they can be compared:

  1  the detector used as it was trained - InstanSeg on RGB or on optical-density sum,
     plus the classic OD-sum watershed - against today's haematoxylin-only input;
  2  DeepLIIF, which finds nuclei independently of DAB;
  3  an IHC-trained detector - the published LyNSeC IHC model, and a pilot fine-tune of
     Cellpose's nucleus model on LyNSeC's hand-labelled IHC nuclei.

Every detector is scored two ways: density against the H&E reference on our own slides
(P-21's corrected method), and F1 against LyNSeC's labels on held-out images.

Everything writes under `storage/<version>/data/p03_nuclei/`, gitignored, and every unit
of work is checkpointed so the run can be killed and resumed at any point.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import data_versions  # noqa: E402

BACKEND = ROOT / "tissue_scoring_demo" / "backend"
BACKEND_PY = BACKEND / ".venv" / "Scripts" / "python.exe"
PY311 = pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python311" / "python.exe"

OUT = data_versions.data_root() / "p03_nuclei"
STATE = OUT / "state"
LOGS = OUT / "logs"
DOWNLOADS = OUT / "downloads"
FIELDS = OUT / "fields"
LABELS = OUT / "labels"
LYNSEC = OUT / "lynsec"
ENVS = OUT / "envs"
MODELS = OUT / "models"
HISTORY = data_versions.data_root() / "history"

#: The benchmark: (case, marker). Chosen across all five markers and the whole range of
#: measured shortfall (5% to 97%). CAN_00259 and CAN_00865 are left out on purpose: their
#: carried regions still include the scanner-fill rectangle (P-05), so their H&E
#: reference is meaningless until they are re-run.
PAIRS: list[tuple[str, str]] = [
    ("CAN_00251", "A"), ("CAN_00251", "R"),
    ("CAN_00267", "A"), ("CAN_00267", "F"), ("CAN_00267", "W"),
    ("CAN_00270", "A"), ("CAN_00270", "R"),
    ("CAN_00303", "A"), ("CAN_00303", "U"), ("CAN_00303", "W"),
]

#: Fields per pair, sampled reproducibly from the fields production actually used. Enough
#: for a stable density (hundreds of nuclei), few enough that every detector finishes.
FIELD_CAP = 24
SEED = 3

#: Detector names, in report order. `baseline` is what production does today.
DETECTORS = [
    "baseline_h",        # InstanSeg on the haematoxylin-only render (production)
    "instanseg_rgb",     # approach 1: the model's native input
    "instanseg_odsum",   # approach 1: rendered optical-density sum (H + DAB)
    "watershed_odsum",   # approach 1: QuPath-style watershed on OD sum
    "deepliif",          # approach 2
    "lynsec_hovernet",   # approach 3: published IHC-trained model
    "cellpose_nuclei",   # approach 3: zero-shot reference for the fine-tune
    "cellpose_finetuned",  # approach 3: fine-tuned on LyNSeC IHC
]


def ensure_dirs() -> None:
    for directory in (STATE, LOGS, DOWNLOADS, FIELDS, LABELS, LYNSEC, ENVS, MODELS):
        directory.mkdir(parents=True, exist_ok=True)


def pair_id(case: str, marker: str) -> str:
    return f"{case}_{marker}"


def read_json(path: pathlib.Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: pathlib.Path, payload) -> None:
    """Atomic: a crash mid-write leaves the previous file, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=str)
        os.replace(temporary, path)
    except BaseException:
        pathlib.Path(temporary).unlink(missing_ok=True)
        raise


def say(message: str, log: pathlib.Path | None = None) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class Checkpoint:
    """One JSON file of finished units for one stage - a pair, a file, an epoch.

    A stage asks `done(unit)` before doing anything and calls `mark(unit, ...)` the moment
    a unit is finished, so a kill at any point costs at most the unit in progress.
    """

    def __init__(self, stage: str) -> None:
        self.path = STATE / f"{stage}.units.json"
        self.data = read_json(self.path, {}) or {}

    def done(self, unit: str) -> bool:
        return bool((self.data.get(unit) or {}).get("ok"))

    def mark(self, unit: str, **fields) -> None:
        self.data[unit] = {"ok": True, "at": time.strftime("%Y-%m-%d %H:%M:%S"), **fields}
        write_json(self.path, self.data)

    def fail(self, unit: str, error: str) -> None:
        previous = self.data.get(unit) or {}
        self.data[unit] = {
            "ok": False, "error": error[-2000:],
            "attempts": int(previous.get("attempts", 0)) + 1,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(self.path, self.data)


def heartbeat(stage: str, message: str) -> None:
    """Proof of progress for the watcher: a stage that stops calling this is suspect."""
    write_json(STATE / f"{stage}.heartbeat.json", {"at": time.time(), "message": message})
