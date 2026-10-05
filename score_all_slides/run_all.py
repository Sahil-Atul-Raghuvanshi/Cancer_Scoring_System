"""The overnight pass: every case, one at a time, with a watcher on each one.

    python run_all.py                     every case that is not already done
    python run_all.py --only CAN_00251    just these
    python run_all.py --redo CAN_00270    run these again even if the checkpoint says done
    python run_all.py --deadline 9.5      stop starting new cases after this many hours

Started detached by `run_overnight.cmd` and left alone. It spawns `run_case.py` once per
case, watches that process, and writes the checkpoint after every case so that the next
run picks up where this one stopped.

**Why a watcher rather than a plain `subprocess.run`.** A case is hours of native
inference, and the two ways it goes wrong overnight are both invisible to a blocking
call: it can die without an exit code anybody sees until morning, or - worse - it can
hang, holding the GPU and producing nothing for the remaining eight hours. So the case is
spawned, then polled: the log file's modification time is the heartbeat, and a case whose
log has been silent for longer than any single stage can honestly take is killed and its
slot marked `stalled`. That turns a hung case from "the night was wasted" into "one case
was lost and the other five ran".

**The deadline stops new work, it never interrupts running work.** A case killed
part-way leaves half a registration on disk for the next run to distrust. So when the
deadline passes, the case in progress finishes and no further case is started - which
means the pass can overrun the deadline by up to one case, and that is the intended
behaviour rather than a bug in it.

**Everything is restartable, so the honest plan is that this takes more than one night.**
Six cold cases at the timings in the run logs is comfortably more than ten hours. The
checkpoint, the per-case records and the CSV are all written as work completes, so
running this again tomorrow continues rather than repeats.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline

#: How long a case may run before it is abandoned.
#:
#: Five hours was too tight and it cost a result: CAN_00270 had scored four of its five
#: markers and was working on the fifth when the limit killed it. The estimate behind
#: five hours came from CAN_00251's 77 minutes, which turned out to be the fast case
#: rather than the typical one.
CASE_LIMIT_S = 8 * 60 * 60

#: How long a case's log may be silent before the case is treated as hung.
#:
#: Ninety minutes was the first guess and it was wrong. It was sized against step 8,
#: the stage that was known to be long; the stage that actually goes quiet is step 11,
#: whose cold per-pixel refinement held a single region of CAN_00259 for over an hour
#: while burning four cores and printing nothing, because `pixels.segment` reports per
#: window internally and the headless runner only logs at stage boundaries. The watchdog
#: would have killed work that was going perfectly well.
#:
#: Three hours is past anything observed. `keepalive.py` is the better answer and runs
#: beside the pass - it measures CPU rather than log lines, so it can tell slow from hung
#: where this cannot - but this has to be safe on its own, because a pass whose liveness
#: depends on a second process being up is a pass with two things to go wrong.
SILENCE_LIMIT_S = 3 * 60 * 60

#: How often the watcher looks. The thing being watched moves in minutes, so this is
#: about being responsive to a crash rather than about resolution.
POLL_S = 30


#: Cases in the order they should be attempted, cheapest first, from what previous runs
#: actually measured rather than from an estimate.
#:
#: **Step 11 dominates and is cached per case.** BEETLE runs once on the shared H&E and
#: every later marker reuses it - CAN_00865's first pass through step 11 took 164 minutes
#: (15:00:15 to 17:44:35 on 18 Sep) and its retry took three seconds. So the case order is
#: really an ordering on step-11 cost, which tracks the tile area BEETLE has to examine.
#:
#:   CAN_00270    62 min   completed 5/5 on 18 Sep        1081 tiles
#:   CAN_00251    77 min   completed 5/5 on 17 Sep         306 tiles
#:   CAN_00267   274 min   crashed - but on *alignment*,    74 tiles
#:                         which is fixed; its step 11 is
#:                         the smallest in the cohort
#:   CAN_00303      -      never completed under BEETLE    640 tiles
#:   CAN_00259    ~142 min step 11 only (58 min for ROI-1 3969 tiles
#:                         plus a larger ROI-2)
#:   CAN_00865    164 min  step 11 only, measured         5201 tiles
#:
#: Two measured times invert against tile count - CAN_00270 has three times CAN_00251's
#: tiles and finished fifteen minutes sooner - because the rest of a case (nuclei, typing,
#: per-cell measurement) scales with cell count rather than region area. Where a case has
#: a measured whole-case time that is trusted over the tile proxy; the proxy only orders
#: the cases that have never finished.
#:
#: Cheapest first so a night that gets cut short still returns the most rows, and so a
#: systematic problem shows up in an hour rather than after three.
CASE_ORDER = [
    "CAN_00270",
    "CAN_00251",
    "CAN_00267",
    "CAN_00303",
    "CAN_00259",
    "CAN_00865",
]


def _cheapest_first(names: list[str]) -> list[str]:
    """Known cases in measured-cost order; anything unknown after them, by name.

    An unseen case sorts last rather than first: nothing is known about its cost, and the
    point of the ordering is to bank the cheap certain work before spending hours on the
    expensive uncertain kind.
    """
    rank = {case: index for index, case in enumerate(CASE_ORDER)}
    return sorted(names, key=lambda case: (rank.get(case, len(rank)), case))


def _heartbeat(case_id: str) -> float:
    """Seconds since the case's log was last written, or 0 if it has not been yet.

    The log is the heartbeat because the runner already writes to it after every stage,
    which means liveness costs nothing and cannot drift out of step with the work the
    way a separately-maintained timestamp would.
    """
    log = pipeline.LOGS / f"{case_id}.log"
    if not log.is_file():
        return 0.0
    return max(0.0, time.time() - log.stat().st_mtime)


def run_case(case_id: str) -> tuple[str, str]:
    """Spawn one case and watch it. Returns `(state, detail)`.

    `state` is one of `done`, `partial`, `crashed`, `stalled`, `timeout`.
    """
    log = pipeline.RUN_LOG
    # The mode travels to the child explicitly rather than through the environment, so a
    # log line or a process listing shows which pass a worker belongs to.
    command = [str(pipeline.PYTHON), str(HERE / "run_case.py"), case_id, pipeline.MODE]
    pipeline.say(f"{case_id}: starting", log)

    # cwd is the backend, matching how every other script in this repository is invoked,
    # so anything resolving a path relative to the working directory finds what it
    # expects. Output goes to the case's own log; the runner writes there too, so one
    # file holds both the stage lines and anything a library printed underneath them.
    case_log = pipeline.LOGS / f"{case_id}.log"
    case_log.parent.mkdir(parents=True, exist_ok=True)
    with case_log.open("a", encoding="utf-8", errors="replace") as stream:
        stream.write(f"\n===== spawn {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=str(pipeline.BACKEND),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )

        started = time.monotonic()
        while True:
            code = process.poll()
            if code is not None:
                break

            elapsed = time.monotonic() - started
            if elapsed > CASE_LIMIT_S:
                pipeline.say(
                    f"{case_id}: over the {CASE_LIMIT_S / 3600:.0f}h limit, stopping it",
                    log,
                )
                process.kill()
                process.wait(timeout=120)
                return "timeout", f"killed after {elapsed / 3600:.1f}h"

            silent = _heartbeat(case_id)
            if silent > SILENCE_LIMIT_S:
                pipeline.say(
                    f"{case_id}: silent for {silent / 60:.0f} min, treating as hung",
                    log,
                )
                process.kill()
                process.wait(timeout=120)
                return "stalled", f"no log line for {silent / 60:.0f} min"

            time.sleep(POLL_S)

    # The exit code says whether anything scored; the record says how much. Reading the
    # record rather than trusting the code means a case that scored four markers and
    # failed one is reported as `partial` instead of as success.
    scored = failed = 0
    for record in pipeline.load_results():
        if record.get("caseId") != case_id:
            continue
        for marker in record.get("markers", []):
            if marker.get("state") == "scored":
                scored += 1
            else:
                failed += 1

    if code != 0 and scored == 0:
        return "crashed", f"exit {code}, nothing scored"
    if failed:
        return "partial", f"{scored} scored, {failed} not"
    return "done", f"{scored} scored"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument("--redo", default="", help="case ids to run again regardless")
    parser.add_argument(
        "--deadline",
        type=float,
        default=0.0,
        help="hours after which no further case is started; 0 means no deadline",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="how many times a case may be started across runs before it is given up on",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help=(
            "how many times to sweep the case list. A second sweep picks up cases that "
            "crashed or stalled in the first, which is what 'run until they are all "
            "done' means in practice"
        ),
    )
    parser.add_argument("--roi-source", choices=("refined", "tiles"), default="refined")
    args = parser.parse_args()

    # Before anything reads a path. Both modes are fully separate on disk.
    pipeline.configure(args.roi_source)

    pipeline.ensure_dirs()
    log = pipeline.RUN_LOG
    began = time.monotonic()

    state = pipeline.load_checkpoint()
    state.setdefault("started", time.strftime("%Y-%m-%d %H:%M:%S"))

    every = pipeline.cases()
    only = [one.strip() for one in args.only.split(",") if one.strip()]
    redo = {one.strip() for one in args.redo.split(",") if one.strip()}
    wanted = [one for one in every if not only or one in only]
    wanted = _cheapest_first(wanted)

    pipeline.say("", log)
    pipeline.say("=" * 70, log)
    pipeline.say(f"pass over {len(wanted)} case(s): {', '.join(wanted)}", log)
    pipeline.say(
        f"up to {args.rounds} round(s), {args.attempts} attempt(s) per case", log
    )
    if args.deadline:
        pipeline.say(f"no new case will start after {args.deadline:g}h", log)
    else:
        pipeline.say("no deadline: this runs until every case is done or given up on", log)

    #: Sweeps of the case list. A case that crashed or stalled in round one is retried in
    #: round two, which is the difference between "six cases were attempted" and "six
    #: cases were finished". A round that changes nothing ends the pass - without that
    #: this would spin forever on a case that fails the same way every time.
    for round_number in range(1, max(1, args.rounds) + 1):
        # **`--redo` outranks `done`, which it did not used to.** The flag is documented
        # as "run these again even if the checkpoint says done", and `_sweep` implements
        # exactly that - it resets the slot to pending before running. But this filter
        # required `state != "done"` *before* it ever looked at `redo`, so a finished case
        # could never reach `_sweep` and the flag silently did nothing. It was passed
        # `--redo CAN_00267` to rebuild rows written before a caveat existed, and the round
        # simply reported three cases left with that one absent - no error, no warning.
        remaining = [
            one
            for one in wanted
            if one in redo
            or (
                pipeline.case_state(state, one)["state"] != "done"
                and pipeline.case_state(state, one)["attempts"] < args.attempts
            )
        ]
        if not remaining:
            break

        pipeline.say("", log)
        pipeline.say(
            f"--- round {round_number} of {args.rounds}: "
            f"{len(remaining)} case(s) left ({', '.join(remaining)}) ---",
            log,
        )
        before = _scored_total()
        stop = _sweep(state, remaining, redo, args, began, log)
        redo = set()  # a forced case is forced once, not once per round
        after = _scored_total()

        if stop:
            break
        if after == before:
            pipeline.say(
                f"round {round_number} scored nothing new; the cases still outstanding "
                "fail the same way each time, so the pass stops here rather than "
                "repeating it",
                log,
            )
            break

    pipeline.write_csv()
    pipeline.say("", log)
    pipeline.say("----- where the pass got to -----", log)
    for case_id in every:
        slot = state.get("cases", {}).get(case_id, {})
        pipeline.say(
            f"  {case_id:12s} {slot.get('state', 'pending'):9s} "
            f"{slot.get('scored', 0)} scored, {slot.get('failed', 0)} not"
            f"{'  [filed]' if slot.get('archived') else ''}",
            log,
        )
    pipeline.say(f"scores: {pipeline.CSV_PATH}", log)
    pipeline.say(f"total {(time.monotonic() - began) / 3600:.1f}h", log)
    pipeline.save_checkpoint(state)
    return 0


def _scored_total() -> int:
    """Markers scored across every case on disk. Used to tell a round apart from a loop."""
    return sum(
        1
        for record in pipeline.load_results()
        for marker in record.get("markers", [])
        if marker.get("state") == "scored"
    )


def _sweep(state, wanted, redo, args, began, log) -> bool:
    """One pass over the cases. Returns True if the deadline ended it early."""
    for case_id in wanted:
        slot = pipeline.case_state(state, case_id)

        if case_id in redo:
            slot["state"] = "pending"
            slot["attempts"] = 0
        elif slot["state"] == "done":
            pipeline.say(f"{case_id}: already done, skipping", log)
            continue
        elif slot["attempts"] >= args.attempts and slot["state"] != "pending":
            pipeline.say(
                f"{case_id}: {slot['attempts']} attempts already, skipping "
                f"(last: {slot['state']}). Use --redo to force it.",
                log,
            )
            continue

        hours = (time.monotonic() - began) / 3600
        if args.deadline and hours >= args.deadline:
            pipeline.say(
                f"deadline reached at {hours:.1f}h; {case_id} and the rest are left "
                "for the next run",
                log,
            )
            return True

        slot["attempts"] += 1
        slot["state"] = "running"
        slot["startedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
        pipeline.save_checkpoint(state)

        started = time.monotonic()
        try:
            result, detail = run_case(case_id)
        except Exception as failure:  # noqa: BLE001 - one case must not end the pass
            result, detail = "crashed", f"{type(failure).__name__}: {failure}"

        slot["state"] = result
        slot["detail"] = detail
        slot["minutes"] = round((time.monotonic() - started) / 60, 1)
        slot["finishedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")

        for record in pipeline.load_results():
            if record.get("caseId") == case_id:
                slot["scored"] = sum(
                    1 for one in record.get("markers", []) if one.get("state") == "scored"
                )
                slot["failed"] = sum(
                    1 for one in record.get("markers", []) if one.get("state") != "scored"
                )
                slot["archived"] = bool(record.get("archived"))

        pipeline.save_checkpoint(state)
        pipeline.write_csv()
        pipeline.say(
            f"{case_id}: {result} ({detail}) in {slot['minutes']:.0f} min", log
        )

    return False


if __name__ == "__main__":
    raise SystemExit(main())
