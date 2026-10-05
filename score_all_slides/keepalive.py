"""Prove a quiet case is working, so the stall watchdog does not kill it for being slow.

    python keepalive.py            run until the pass is over
    python keepalive.py --once     check once and print what it sees

`run_all.py` decides a case has hung when its log has been silent for 90 minutes. That
threshold was sized against step 8, and it is wrong for step 11: a cold per-pixel
refinement on a large slide holds one region for over an hour and prints nothing while
it does, because `pixels.segment` reports per window internally and the headless runner
only logs at stage boundaries. The first case that hit it - CAN_00259, one region, 47
minutes and counting - would have been killed at the 90 minute mark while burning nearly
four cores.

**The fix is to measure the right thing.** A log line is a proxy for liveness; CPU time
is liveness. So this samples the worker's CPU and, when it is genuinely computing, writes
a line to that case's log saying so. The orchestrator sees a fresh heartbeat and leaves
it alone.

**It does not disable the watchdog, and that is the point.** A worker that has truly hung
- deadlocked, waiting on a dead handle, spinning on no CPU - uses no CPU, so no line is
written, the log goes quiet and the 90 minute rule fires exactly as designed. The only
case this rescues is the one the watchdog was never meant to catch: work that is slow.

**The lines it writes are real information, not padding.** Each says how much CPU the
stage is using and how long it has been in it, which is the visibility the silent stretch
was missing. Reading the log in the morning tells you what a stage was doing, not merely
that something touched a file.

Runs beside the pass rather than inside it, so it needs no restart of an orchestrator
that is already hours into its work.
"""

from __future__ import annotations

import argparse
import ctypes
import pathlib
import sys
import time
from ctypes import wintypes

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline

#: How often to sample. Long enough that a short pause between stages does not register
#: as work, short enough to write several heartbeats inside one watchdog window.
SAMPLE_S = 120

#: Fraction of a single core a worker must average across a sample to count as computing.
#:
#: Deliberately low. The question is "is this process doing anything at all", not "is it
#: doing it quickly" - a stage that is disk-bound reading a pyramid legitimately drops to
#: a few percent, and killing it for that would be the same mistake as killing it for
#: being quiet. A hung process sits at zero, which is well clear of this.
BUSY_FRACTION = 0.05

_kernel32 = ctypes.windll.kernel32
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class _FileTime(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]


def _cpu_seconds(pid: int) -> float | None:
    """Kernel + user CPU this process has ever used, or None if it cannot be read.

    `PROCESS_QUERY_LIMITED_INFORMATION` rather than full query access: it is what a
    non-elevated process is granted for its own user's processes, and asking for more
    would make this fail on exactly the machine it is meant to run on.
    """
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    created, exited, kernel, user = _FileTime(), _FileTime(), _FileTime(), _FileTime()
    ok = _kernel32.GetProcessTimes(
        handle,
        ctypes.byref(created),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    )
    _kernel32.CloseHandle(handle)
    if not ok:
        return None
    ticks = ((kernel.high << 32) | kernel.low) + ((user.high << 32) | user.low)
    return ticks / 1e7  # 100-nanosecond intervals


def _worker_pids(case_id: str) -> list[int]:
    """The running case's worker and everything it started - and nothing else (P-19).

    This used to sample every python.exe on the machine and ask whether *any* was busy.
    A hung worker beside one busy, unrelated python - a notebook, the demo backend,
    another script - was then reported as working, kept alive past the silence rule, and
    only stopped at the eight-hour limit. `run_all.py` now records the pid it started, so
    the question is asked of that process tree alone.
    """
    root = pipeline.worker_pid(case_id)
    if root is None:
        return []
    return sorted(pipeline.process_tree(root))


def _running_case() -> str | None:
    """The case the orchestrator says it is working on, or None."""
    for case_id, slot in pipeline.load_checkpoint().get("cases", {}).items():
        if slot.get("state") == "running":
            return case_id
    return None


def _busiest(before: dict[int, float], after: dict[int, float], seconds: float) -> float:
    """The highest per-process CPU share seen across the sample, as a fraction of a core.

    Over the worker's own process tree only - see `_worker_pids`. The maximum rather than
    the sum, because a maximum says something true about a single process, which is what
    "is this stage computing" is about.
    """
    best = 0.0
    for pid, then in before.items():
        now = after.get(pid)
        if now is None:
            continue
        best = max(best, (now - then) / seconds)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="sample once and report")
    args = parser.parse_args()

    pipeline.ensure_dirs()
    log = pipeline.LOGS / "keepalive.log"
    pipeline.say("keepalive watching; a busy case will be kept marked alive", log)

    quiet_since: dict[str, float] = {}

    while True:
        case_id = _running_case()
        if case_id is None:
            # No case is running: either between cases, or the pass is over. Nothing to
            # keep alive either way, and exiting here would mean not being there for the
            # next case - so it waits.
            if args.once:
                print("no case is running")
                return 0
            time.sleep(SAMPLE_S)
            continue

        pids = _worker_pids(case_id)
        if not pids:
            # No recorded worker - an orchestrator from before the pid file, or a case
            # between spawns. Claiming liveness without evidence is the bug this fixes,
            # so nothing is written and the watchdog decides.
            pipeline.say(f"{case_id}: no worker pid recorded - leaving the watchdog to it", log)
            if args.once:
                print(f"{case_id}: no worker pid recorded")
                return 0
            time.sleep(SAMPLE_S)
            continue
        before = {pid: value for pid in pids if (value := _cpu_seconds(pid)) is not None}
        time.sleep(SAMPLE_S)
        after = {pid: value for pid in pids if (value := _cpu_seconds(pid)) is not None}

        share = _busiest(before, after, SAMPLE_S)
        case_log = pipeline.LOGS / f"{case_id}.log"
        silent = (
            time.time() - case_log.stat().st_mtime if case_log.is_file() else 0.0
        )

        if share >= BUSY_FRACTION:
            started = quiet_since.setdefault(case_id, time.time() - silent)
            # Written to the case's own log, which is what the orchestrator reads the
            # modification time of. The content is for a person; the timestamp is what
            # stops the watchdog.
            pipeline.say(
                f"{case_id}: still working - {share * 100:.0f}% of a core, "
                f"{(time.time() - started) / 60:.0f} min in this stage",
                case_log,
            )
        else:
            quiet_since.pop(case_id, None)
            pipeline.say(
                f"{case_id}: idle at {share * 100:.1f}% of a core after "
                f"{silent / 60:.0f} min of silence - leaving the watchdog to it",
                log,
            )

        if args.once:
            print(f"{case_id}: {share * 100:.0f}% of a core, silent {silent / 60:.0f} min")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
