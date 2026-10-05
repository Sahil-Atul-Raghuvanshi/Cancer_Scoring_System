"""Emit one line per campaign state change, and nothing while nothing happens.

    python scripts/10d_watch.py

A reader, never a writer - the driver is the only process that touches the tracker, and
the duplicated stage lines in `PROGRESS.md` are what happens when that stops being
true. This polls `he_campaign_state.json` and prints a line when, and only when,
something changed: a stage started or finished, the overall verdict moved, a failure was
recorded, or a head was published. Exits when the campaign is no longer running, so a
watcher that has gone quiet has either finished or died, and those are distinguishable
by the last line it printed.

**It reports every terminal state, not only success.** A watcher that only announced
publications would stay silent through an aborted pre-flight, a stalled export and a
blown budget, and silence would be indistinguishable from progress.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
STATE = (WORKSPACE / "tissue_type_model_training" / "reports" / "he_campaign"
         / "he_campaign_state.json")
POLL = 20.0
FINISHED = ("DONE", "PARTIAL", "FAILED", "BUDGET_EXCEEDED")


def emit(text: str) -> None:
    print(text, flush=True)


def read() -> dict | None:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # The driver replaces this file atomically, so a failed read is a race with
        # the replace and the next poll will get it. Not worth reporting.
        return None


def main() -> int:
    seen_status: dict[str, str] = {}
    seen_overall = ""
    seen_failures: set[str] = set()
    seen_published: set[str] = set()
    missing_polls = 0

    while True:
        state = read()
        if state is None:
            missing_polls += 1
            if missing_polls == 6:
                emit("WATCH: he_campaign_state.json has been unreadable for 2 minutes")
            time.sleep(POLL)
            continue
        missing_polls = 0

        overall = state["overall"]
        elapsed = state["elapsed_minutes"]
        free = state["free_gb"]

        for stage in state["stages"]:
            was = seen_status.get(stage["id"])
            now_status = stage["status"]
            if was == now_status:
                continue
            seen_status[stage["id"]] = now_status
            if was is None and now_status == "PENDING":
                continue

            if now_status == "RUNNING":
                emit(f"START {stage['id']} (budget {stage['budget_minutes']:.0f}m) "
                     f"[{elapsed:.0f}m in, {free:.1f} GB free]")
            elif now_status == "DONE":
                tiles = f", {stage['tiles']:,} tiles" if stage.get("tiles") else ""
                emit(f"OK    {stage['id']} in {stage['elapsed_minutes']:.1f}m{tiles}")
            elif now_status == "SKIPPED_DONE":
                emit(f"SKIP  {stage['id']} - already on disk and verifies")
            elif now_status in ("FAILED", "TIMED_OUT"):
                first = (stage.get("reason") or "").splitlines()[:1]
                emit(f"FAIL  {stage['id']} ({now_status}) "
                     f"after {stage['elapsed_minutes']:.1f}m: "
                     f"{first[0] if first else 'no reason recorded'}")
            elif now_status == "SKIPPED_UPSTREAM_FAILED":
                emit(f"SKIP  {stage['id']} - an earlier stage of this arm failed")

        for failure in state["failures"]:
            if failure["stage"] not in seen_failures:
                seen_failures.add(failure["stage"])

        for fov, name in sorted(state["published"].items()):
            if fov not in seen_published:
                seen_published.add(fov)
                emit(f"PUBLISHED {name} ({fov} um) - "
                     f"{len(seen_published)} of 4 H&E heads")

        if overall != seen_overall:
            seen_overall = overall
            if overall == "STALLED":
                emit(f"STALLED: the running stage's log has not grown for 20 minutes "
                     f"[{elapsed:.0f}m in]")
            elif overall in FINISHED:
                emit(f"CAMPAIGN {overall} after {elapsed:.0f} min - "
                     f"{len(state['published'])} of 4 published, "
                     f"{len(state['failures'])} failure(s). "
                     f"Read HE_CAMPAIGN_TRACKER.md")

        if overall in FINISHED:
            return 0

        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
