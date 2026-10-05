"""Exit 0 if the night has nothing left to do, 1 if a stage should be retried.

    python night_pending.py

Used by `supervise.cmd` to decide whether to loop. A separate tiny script rather than an
exit code from `night.py` because the two questions are genuinely different: `night.py`
exits 0 when it has finished *this attempt*, which includes attempts where a stage
crashed and should be tried again.

**A stage that has failed repeatedly is treated as finished, not as pending.** Otherwise a
stage that cannot succeed - a missing dependency, a corrupt slide - would spin the
supervisor forever and the night would end with nothing but retries in the log.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import night  # noqa: E402

#: How many times a stage may fail before the supervisor stops retrying it.
MAX_ATTEMPTS = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--peek",
        action="store_true",
        help=(
            "answer without counting an attempt. The supervisor must count, because that "
            "is how a stage that cannot succeed eventually stops being retried; anything "
            "else asking the same question - the sentry polls it every two minutes - must "
            "not, or it would exhaust the retry budget on behalf of a stage that never ran."
        ),
    )
    args = parser.parse_args()

    payload = night.state()
    stages = payload.get("stages", {})

    pending = []
    for name in night.STAGES:
        row = stages.get(name, {})
        state = row.get("state")
        attempts = int(row.get("attempts") or 0)
        if state == "done":
            continue
        if state == "failed" and attempts >= MAX_ATTEMPTS:
            continue
        pending.append(name)

    if not pending:
        print("nothing pending")
        return 0

    nxt = pending[0]
    if args.peek:
        print(f"pending: {', '.join(pending)} (next {nxt})")
        return 1

    # Count this as an attempt for whichever stage is next, so a stage that crashes the
    # supervisor's child every time still runs out of attempts rather than looping.
    row = stages.setdefault(nxt, {})
    row["attempts"] = int(row.get("attempts") or 0) + 1
    common.write_json(night.STATE_FILE, payload)

    print(f"pending: {', '.join(pending)} (next {nxt}, attempt {row['attempts']})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
