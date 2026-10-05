"""The unattended run: every stage in order, each checkpointed and skippable.

    python night.py                  run every stage that is not already done
    python night.py --from methods   start at this stage
    python night.py --only methods,score_tiles
    python night.py --status         print what is done and exit

One process, one sequence, so the stages cannot fight each other for CPU - which they did
when the registration pass and the method sweep were launched independently and each
halved the other's speed.

**Stages, in order, with the reasoning for the order:**

    methods       every registration method on all 30 pairs. Cheap (minutes) and it is
                  what decides which transform everything downstream uses, so it goes
                  first: no point scoring a cohort with a registration that loses.
    score_tiles   score every case carrying step 9's COARSE TILE regions, skipping step
                  11 entirely. **The fast one, and deliberately first.** Step 11 is the
                  most expensive stage in the pipeline, and if the scores barely move
                  without it then most of the cohort's compute is buying very little.
    score_beetle  the same cohort carrying BEETLE's per-pixel boundaries. Slow, and it is
                  the run the pipeline normally does.
    coverage      the tile-versus-BEETLE area tables, rebuilt from whatever step 11 last
                  produced.
    decide        write v<N>_data/results/DECISIONS.md: what worked, what did not, and
                  what is settled.

Every stage records itself in the checkpoint the moment it finishes, so a machine that
reboots at four in the morning resumes at the next stage rather than starting over.

**Nothing here is destructive.** The two scoring modes write to entirely separate state,
results and CSV files (`oncostem_ai_scores.csv` and `oncostem_ai_scores_tiles.csv`), and
the pre-existing scores were copied to `v<N>_data/data/score_archive/` before any of this began.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

STAGES = ["methods", "score_tiles", "score_beetle", "coverage", "compare", "decide"]

STATE_FILE = common.STATE / "night.json"


def state() -> dict:
    return common.read_json(STATE_FILE) or {"stages": {}}


def mark_stage(name: str, **fields) -> None:
    payload = state()
    row = payload["stages"].setdefault(name, {})
    row.update(fields)
    row["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    common.write_json(STATE_FILE, payload)


def done(name: str) -> bool:
    return state()["stages"].get(name, {}).get("state") == "done"


def _run(command: list[str], cwd: pathlib.Path, log, label: str, timeout: float | None = None) -> int:
    """Run one stage as a child process, streaming nothing, recording everything.

    Output is not piped: the children already write to `run.log` and to their own logs,
    and holding a pipe open for a stage measured in hours is how a buffer fills and a
    child blocks on a write nobody is reading.

    **On timeout the whole tree is killed, not just the child.** Every stage here spawns a
    worker in the other virtual environment - VALIS or SimpleITK - and `Popen.kill()`
    reaches only the process it started. An orphaned worker would hold four cores for the
    rest of the night while the supervisor cheerfully restarted the stage beside it.
    `taskkill /T` walks the tree from this specific pid, so nothing outside this stage is
    touched.
    """
    common.say(f"stage '{label}': {' '.join(str(c) for c in command[-4:])}", log)
    began = time.monotonic()
    process = subprocess.Popen(command, cwd=str(cwd))  # noqa: S603 - fixed command
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        common.say(
            f"stage '{label}' exceeded its wall-clock cap of {timeout / 3600:.1f}h - "
            f"killing pid {process.pid} and its children",
            log,
        )
        subprocess.run(  # noqa: S603 - fixed command, pid from this process
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            text=True,
        )
        try:
            code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            code = -1
    common.say(f"stage '{label}' exit {code} after {(time.monotonic() - began) / 60:.0f} min", log)
    return code


#: How much longer than its own deadline a stage may run before it is killed outright.
#: The deadline a stage receives only stops it *starting* new work - a case already running
#: is always allowed to finish - so the wall-clock cap has to leave room for one long case
#: on top. Without this, a stage that hangs never exits and `supervise.cmd`, which only
#: acts when `night.py` exits, waits for it forever.
HARD_CAP_MULTIPLIER = 1.8


def _cap(deadline_h: float) -> float:
    return deadline_h * 3600 * HARD_CAP_MULTIPLIER


def stage_methods(log, deadline_h: float) -> int:
    return _run(
        [str(common.PYTHON), str(HERE / "methods_run.py"), "--deadline", str(deadline_h)],
        common.BACKEND, log, "methods", timeout=_cap(deadline_h),
    )


def _score(mode: str, log, deadline_h: float) -> int:
    driver = common.ROOT / "score_all_slides" / "run_all.py"
    return _run(
        [str(common.PYTHON), str(driver), "--roi-source", mode, "--deadline", str(deadline_h)],
        common.BACKEND, log, f"score_{mode}", timeout=_cap(deadline_h),
    )


def stage_score_tiles(log, deadline_h: float) -> int:
    return _score("tiles", log, deadline_h)


def stage_score_beetle(log, deadline_h: float) -> int:
    return _score("refined", log, deadline_h)


def stage_coverage(log, deadline_h: float) -> int:
    return _run(
        [str(common.PYTHON), str(HERE / "beetle_coverage.py")],
        common.BACKEND, log, "coverage", timeout=_cap(deadline_h),
    )


def stage_compare(log, deadline_h: float) -> int:
    return _run(
        [str(common.PYTHON), str(HERE / "compare_scores.py")],
        common.BACKEND, log, "compare", timeout=_cap(deadline_h),
    )


def stage_decide(log, deadline_h: float) -> int:
    return _run(
        [str(common.PYTHON), str(HERE / "decide.py")],
        common.BACKEND, log, "decide", timeout=_cap(deadline_h),
    )


RUNNERS = {
    "methods": stage_methods,
    "score_tiles": stage_score_tiles,
    "score_beetle": stage_score_beetle,
    "coverage": stage_coverage,
    "compare": stage_compare,
    "decide": stage_decide,
}

#: Hours each stage may have. The scoring stages get the bulk because they are the only
#: ones measured in hours; the rest are minutes. These are budgets for *starting new
#: work*, not kill timers - a case already running is always allowed to finish.
#: Sized to a 12-15 hour window, which is what is available. The scoring stages take
#: essentially all of it: a case is roughly 80 minutes when everything behaves, six cases
#: is eight hours, and the tiles run should be faster than that because it skips step 11
#: entirely - which is the whole question it exists to answer.
#:
#: These bound *starting new work*. A case already running always finishes, so a stage can
#: overrun its budget by up to one case, and `night.py` applies a separate hard wall-clock
#: cap at 1.8x to catch a stage that has stopped making progress altogether.
BUDGET_H = {
    "methods": 2.0,
    "score_tiles": 5.0,
    "score_beetle": 5.0,
    "coverage": 0.5,
    "compare": 0.25,
    "decide": 0.5,
}


def budget_for(name: str) -> float:
    """Hours this stage may spend starting new work.

    Read from `_state/budgets.json` when that file exists, falling back to `BUDGET_H`.
    Read at the moment the stage starts rather than when the module loads, so the window
    can be changed for a run already in progress - which matters because the window is
    set by how long somebody is going to be asleep, and that is not always known when the
    run is launched.
    """
    override = common.read_json(common.STATE / "budgets.json") or {}
    try:
        return float(override[name])
    except (KeyError, TypeError, ValueError):
        return BUDGET_H.get(name, 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", default="")
    parser.add_argument("--only", default="")
    parser.add_argument("--redo", default="")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    common.ensure_dirs()
    log = common.RUN_LOG

    if args.status:
        payload = state()
        print(f"{'stage':14} {'state':8} {'exit':>5}  updated")
        for name in STAGES:
            row = payload["stages"].get(name, {})
            print(
                f"{name:14} {row.get('state', '-'):8} {str(row.get('exit', '-')):>5}  "
                f"{row.get('updated', '')}"
            )
        return 0

    wanted = [s.strip() for s in args.only.split(",") if s.strip()] or list(STAGES)
    if args.start:
        if args.start not in STAGES:
            print(f"unknown stage {args.start}; known: {STAGES}", file=sys.stderr)
            return 2
        wanted = [s for s in wanted if STAGES.index(s) >= STAGES.index(args.start)]
    redo = {s.strip() for s in args.redo.split(",") if s.strip()}

    common.say("=" * 72, log)
    common.say(f"unattended night starting: stages {wanted}", log)

    for name in wanted:
        if done(name) and name not in redo:
            common.say(f"stage '{name}' already done, skipping", log)
            continue
        common.say("-" * 72, log)
        mark_stage(name, state="running")
        try:
            code = RUNNERS[name](log, budget_for(name))
            mark_stage(name, state="done" if code == 0 else "failed", exit=code)
        except Exception as failure:  # noqa: BLE001 - a stage must not stop the night
            import traceback

            common.say(f"stage '{name}' crashed: {type(failure).__name__}: {failure}", log)
            common.say(traceback.format_exc()[-1500:], log)
            mark_stage(name, state="failed", error=f"{type(failure).__name__}: {failure}")

    common.say("=" * 72, log)
    common.say("unattended night finished", log)
    # The decision record is rebuilt at the end whatever happened, so a night that failed
    # halfway still leaves a readable account of how far it got.
    try:
        import decide

        decide.write()
    except Exception as failure:  # noqa: BLE001
        common.say(f"could not write DECISIONS.md: {failure}", log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
