"""Start the P-03 night detached from this session: the supervisor and the watcher.

    python launch.py

Both are started with CREATE_NO_WINDOW in their own process group, so closing the terminal
or ending the assistant session does not stop them (DETACHED_PROCESS is avoided on purpose:
it silently drops a child Python's output). Safe to run twice: the supervisor refuses to
start while its lock exists, and a second watcher only rewrites STATUS.md.
"""

from __future__ import annotations

import subprocess
import sys

import common

FLAGS = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
BREAKAWAY = 0x01000000  # CREATE_BREAKAWAY_FROM_JOB


def start(command: list[str], log_name: str) -> int:
    log = (common.LOGS / log_name).open("a", encoding="utf-8")
    for flags in (FLAGS | BREAKAWAY, FLAGS):
        try:
            return subprocess.Popen(command, cwd=str(common.HERE), stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, creationflags=flags).pid
        except OSError:
            continue  # the job may forbid breakaway; the process group alone still detaches
    raise RuntimeError(f"could not start {command}")


def main() -> int:
    common.ensure_dirs()
    supervisor = start(["cmd.exe", "/c", str(common.HERE / "supervise.cmd")], "supervisor.console.log")
    watcher = start([str(common.BACKEND_PY), "-u", str(common.HERE / "watch.py")], "watch.console.log")
    common.write_json(common.STATE / "launch.json", {"supervisor_pid": supervisor, "watcher_pid": watcher})
    print(f"supervisor pid {supervisor}, watcher pid {watcher}")
    print(f"status: {common.OUT / 'STATUS.md'}")
    print(f"report: {common.OUT / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
