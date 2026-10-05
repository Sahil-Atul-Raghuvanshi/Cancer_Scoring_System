"""Where the overnight pass has got to. Read-only, safe to run while it is going.

    python watch.py           print the state once
    python watch.py --follow  print it again every minute until interrupted

Separate from `run_all.py` because the orchestrator is detached and its console is a
file. This is what a person runs in the morning, or at midnight from another window, to
find out what happened without attaching to anything or reading a four-thousand line log.

It touches nothing. Every number here is read from the checkpoint, the per-case records
and the log files' modification times, so running it cannot disturb a pass in progress -
which matters, because the thing a person most wants to do at 3am is exactly the thing
that would be worst to do carelessly.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline

#: Matches the orchestrator's own patience. A case quieter than this is one the watcher
#: inside `run_all.py` is about to kill, so saying so here explains what is coming.
SILENCE_WARN_S = 90 * 60


def _age(path: pathlib.Path) -> str:
    if not path.is_file():
        return "-"
    seconds = max(0.0, time.time() - path.stat().st_mtime)
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f}h ago"


def report() -> None:
    state = pipeline.load_checkpoint()
    slots = state.get("cases", {})
    every = pipeline.cases()

    print("=" * 74)
    print(f"  overnight scoring - checkpoint last written {state.get('updated', 'never')}")
    print("=" * 74)

    scored_total = failed_total = 0
    for case_id in every:
        slot = slots.get(case_id, {})
        name = slot.get("state", "pending")
        scored = slot.get("scored", 0)
        failed = slot.get("failed", 0)
        scored_total += scored
        failed_total += failed

        line = (
            f"  {case_id:12s} {name:9s} {scored}/{scored + failed or 5} scored"
            f"{'  filed to history' if slot.get('archived') else ''}"
        )
        if slot.get("minutes"):
            line += f"   {slot['minutes']:.0f} min"
        print(line)

        if name == "running":
            log = pipeline.LOGS / f"{case_id}.log"
            silent = time.time() - log.stat().st_mtime if log.is_file() else 0
            print(f"               last log line {_age(log)}")
            if silent > SILENCE_WARN_S:
                print(
                    f"               ** silent {silent / 60:.0f} min - the watcher will "
                    "stop this case **"
                )
            # The last few lines of the case log say which marker and which stage, which
            # is the one thing the checkpoint cannot record while a case is mid-flight.
            if log.is_file():
                tail = log.read_text(encoding="utf-8", errors="replace").splitlines()
                for line in tail[-3:]:
                    print(f"               | {line}")

    print()
    print(f"  markers scored so far : {scored_total}")
    print(f"  markers not scored    : {failed_total}")
    print(f"  scores CSV            : {pipeline.CSV_PATH}  ({_age(pipeline.CSV_PATH)})")
    print(f"  orchestrator log      : {pipeline.RUN_LOG}  ({_age(pipeline.RUN_LOG)})")

    rows = sum(len(one.get("markers", [])) for one in pipeline.load_results())
    print(f"  rows in the CSV       : {rows}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follow", action="store_true", help="repeat every minute")
    args = parser.parse_args()

    while True:
        report()
        if not args.follow:
            return 0
        time.sleep(60)
        print()


if __name__ == "__main__":
    raise SystemExit(main())
