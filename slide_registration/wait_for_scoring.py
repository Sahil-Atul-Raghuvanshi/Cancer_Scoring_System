"""Block until no other scoring pass is running, so a follow-on cannot race the first.

    python wait_for_scoring.py --mode refined

`run_all.py` takes no lock. Two passes over the same mode would interleave their writes to
one checkpoint and one results directory, and the damage would be quiet: each pass reloads
the checkpoint, edits its own case's slot and writes the whole file back, so the last writer
wins and the other pass's completed cases are silently reverted to `pending`. The CSV is
rebuilt from the per-case records, so it would then disagree with what is actually on disk.

**Processes are read through `psutil`, deliberately not through PowerShell.** `sentry.py`
probes with a CIM query, which is right for it because it runs with a console. This script
is started detached by `finish_beetle.cmd`, and a detached, console-less process that spawns
`powershell.exe` hangs - observed here, five minutes with no output and a stuck child, and
`subprocess.run`'s own timeout did not break it out, because killing the child does not
close the pipe the grandchild still holds. `psutil` is already in the backend venv and reads
the process table in-process, so there is no child to hang on.

**One pass shows up as two processes, and that is not a second pass.** The `python.exe` in
a Windows venv's `Scripts` is a stub that re-execs the real interpreter as a child with the
same command line, so every logical process appears twice - same arguments, different pid,
one the parent of the other. Waiting for both is correct and costs nothing, since they end
together; it is only worth knowing so the message is not read as a duplicate run.

Exits 0 when the way is clear. Never kills anything: something else owns the decision to
stop a running pass, and a script that waits should not also be a script that terminates.
"""

from __future__ import annotations

import argparse
import os
import time

import psutil

#: How often to look. The thing being waited for runs for hours; this only has to be
#: prompt enough that the handover does not waste the night.
POLL_S = 60


def _chain_running() -> list[str]:
    """Is the `finish_beetle.cmd` chain still going?

    Waiting on `run_all.py` alone is not enough for a job that has to follow the *whole*
    chain: the chain runs a refined pass, then a tiles pass, then three reports, and there
    are gaps between them where no `run_all.py` exists at all. A waiter watching only for
    that would start during a gap and race the next stage.
    """
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if "finish_beetle.cmd" in command:
                found.append(f"chain pid {process.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _others(mode: str) -> list[str]:
    """Running `run_all.py` processes for this mode, excluding this process."""
    mine = os.getpid()
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.info["pid"] == mine:
                continue
            name = (process.info["name"] or "").lower()
            if not name.startswith("python"):
                continue
            command = " ".join(process.info["cmdline"] or [])
            if "run_all.py" not in command or "Cancer_Scoring_System" not in command:
                continue
            # `--roi-source refined` is also the default, so a bare `run_all.py` is a
            # refined pass too and must count as one.
            if "--roi-source" in command:
                if f"--roi-source {mode}" not in command:
                    continue
            elif mode != "refined":
                continue
            found.append(f"pid {process.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # A process that vanished mid-scan is not a running pass, and one we cannot
            # read is not evidence of one either.
            continue
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="refined", choices=("refined", "tiles"))
    parser.add_argument("--timeout-h", type=float, default=12.0)
    parser.add_argument(
        "--wait-for-chain",
        action="store_true",
        help="also wait for finish_beetle.cmd to finish, not just a scoring pass",
    )
    args = parser.parse_args()

    began = time.monotonic()
    announced = False
    while True:
        busy = _others(args.mode)
        if args.wait_for_chain:
            busy = busy + _chain_running()
        if not busy:
            print(f"[wait_for_scoring] no other {args.mode} pass running; going ahead",
                  flush=True)
            return 0
        if not announced:
            print(f"[wait_for_scoring] waiting for {', '.join(busy)} to finish "
                  f"({args.mode} pass already running)", flush=True)
            announced = True
        if (time.monotonic() - began) / 3600 >= args.timeout_h:
            # Refuse rather than barge in: an overlapping pass corrupts the checkpoint,
            # and a follow-on that never ran is recoverable where that is not.
            print(f"[wait_for_scoring] still busy after {args.timeout_h:g}h; "
                  f"NOT starting a second pass", flush=True)
            return 1
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
