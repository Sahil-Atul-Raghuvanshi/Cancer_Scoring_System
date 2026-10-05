"""Kill a genuinely hung stage so the supervisor can restart it. Runs beside the night.

    python sentry.py              watch until the night is over
    python sentry.py --once       report what it sees and exit

**The gap this closes.** `supervise.cmd` restarts `night.py` when it exits. A stage that
*hangs* never exits, so the supervisor waits for it forever and a nine-hour unattended run
quietly becomes a nine-hour wait. Nothing else in this pipeline notices: the stage's own
deadline is passed to the child as an argument and only stops it starting new work, and
`subprocess.run` is given no timeout because a legitimate stage can take hours.

**Slow is not hung, and telling them apart is the whole job.** The existing scoring
harness learned this the expensive way: a log-silence watchdog sized against one stage
killed a different stage that was working perfectly, holding one region of CAN_00259 for
forty-seven minutes while burning four cores and printing nothing. A log line is a proxy
for liveness; **CPU time is liveness**. So this requires *both* signals to agree before it
acts:

    the log has not been written for SILENCE_LIMIT
    AND the whole process tree has used almost no CPU over the same period

A stage that is merely slow keeps burning CPU and is left alone. A stage that has
deadlocked, is waiting on a dead handle, or is spinning on nothing uses no CPU, goes quiet,
and gets killed - which is exactly the case the supervisor was built to recover from and
cannot currently reach.

It kills the *worker*, not the orchestrator, so `night.py` sees its child fail, records the
stage as failed, and exits normally - and the supervisor then restarts it with the
checkpoint intact. Killing the orchestrator would lose the stage record.
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

#: How long the run log may be silent before silence counts toward a stall. Generous,
#: because these stages genuinely go quiet: a method sweep logs per case, not per marker,
#: and a scoring case logs at stage boundaries that are tens of minutes apart.
SILENCE_LIMIT_S = 90 * 60

#: Total CPU-seconds the whole tree must accumulate over one poll interval to count as
#: alive. Deliberately tiny - the question is "is anything happening at all", not "is it
#: fast". A single busy thread accrues one CPU-second per wall second.
ALIVE_CPU_S = 2.0

#: How often to look. The thing being watched moves in minutes.
POLL_S = 120

#: Consecutive polls that must agree before anything is killed. A single sample can catch
#: a process between phases - reading a slide off disk uses almost no CPU for a while.
CONFIRMATIONS = 3

#: Consecutive quiet polls tolerated while work is still outstanding. Generous, because
#: a supervisor between attempts is normal and a stage that has just been killed takes a
#: moment to be replaced. Twenty polls is forty minutes.
IDLE_LIMIT = 20

SENTRY_LOG = common.LOGS / "sentry.log"


def _nothing_pending() -> bool:
    """True when `night_pending.py` says every stage is done or permanently failed.

    Shelling out to it rather than importing `night` and reading the file directly, so
    there is exactly one implementation of "is there work left" and the sentry cannot
    drift from what the supervisor believes.
    """
    try:
        done = subprocess.run(  # noqa: S603 - fixed interpreter and script
            [str(common.PYTHON), str(HERE / "night_pending.py"), "--peek"],
            capture_output=True, text=True, timeout=120,
        )
        return done.returncode == 0
    except Exception:  # noqa: BLE001 - if it cannot be asked, assume work remains
        return False


def _tree() -> list[dict]:
    """Every python process belonging to this run, with its CPU time.

    **Matched on the repository path, not on script names.** The first version listed the
    scripts it expected and missed `valis_service/register.py` - the process that actually
    does the work during step 12, while `run_case.py` sits waiting on it. The sentry saw
    an idle parent, read zero CPU, and killed CAN_00251 mid-registration. Anything python
    running out of this repository counts; only the sentry itself is excluded.

    Read through PowerShell's CIM rather than a Python process library so this needs no
    dependency the backend venv might not have.
    """
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*Cancer_Scoring_System*' "
        "-and $_.CommandLine -notlike '*sentry.py*' } | "
        "ForEach-Object { "
        "$p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; "
        "if ($p) { [string]$_.ProcessId + '~' + [string][int]$p.CPU + '~' + "
        "[string]$_.CommandLine } }"
    )
    try:
        out = subprocess.run(  # noqa: S603 - fixed command
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=90,
        ).stdout
    except Exception:  # noqa: BLE001 - a failed probe must not kill the sentry
        return []

    found = []
    for line in out.splitlines():
        parts = line.strip().split("~", 2)
        if len(parts) >= 2 and parts[0].isdigit():
            tail = parts[2] if len(parts) > 2 else ""
            found.append({"pid": int(parts[0]), "cpu": float(parts[1]), "cmd": tail[-60:]})
    return found


def _log_age_s() -> float:
    """Seconds since **any** log this run writes was last touched.

    Not just `run.log`. `night.py` writes that one only at stage boundaries, so during a
    scoring stage - hours long by design - it is silent the whole time and a watchdog
    pointed at it alone sees a stall that is not there. The per-case logs are where the
    liveness actually shows.
    """
    newest = 0.0
    for pattern in (
        common.LOGS / "run.log",
        *(common.DATA / "score_all_slides" / "logs").glob("*.log"),
    ):
        try:
            newest = max(newest, pattern.stat().st_mtime)
        except (OSError, AttributeError):
            continue
    return time.time() - newest if newest else 0.0


def _kill_worker(tree: list[dict]) -> str:
    """Kill the deepest worker, leaving the orchestrator to record the failure.

    The deepest is taken to be the one using the most CPU overall - on a healthy tree that
    is the process doing the work, and on a hung one it is still the process that was.
    `night.py` itself is never a candidate: it has to survive to write the stage record
    the supervisor will read.
    """
    candidates = [p for p in tree if "night.py" not in p["cmd"]]
    if not candidates:
        return "nothing safe to kill"
    victim = max(candidates, key=lambda p: p["cpu"])
    try:
        subprocess.run(  # noqa: S603 - fixed command
            ["taskkill", "/PID", str(victim["pid"]), "/T", "/F"],
            capture_output=True, text=True, timeout=60,
        )
        return f"killed pid {victim['pid']} ({victim['cmd'][:40]})"
    except Exception as failure:  # noqa: BLE001
        return f"could not kill pid {victim['pid']}: {failure}"


def check(previous: dict | None) -> tuple[dict, str]:
    """One observation. Returns the new baseline and a one-line verdict."""
    tree = _tree()
    total_cpu = sum(p["cpu"] for p in tree)
    age = _log_age_s()

    delta = None if previous is None else total_cpu - previous.get("cpu", 0.0)
    now = {"cpu": total_cpu, "at": time.time(), "procs": len(tree)}

    if not tree:
        return now, "no run processes - the night is over or has not started"
    if age < SILENCE_LIMIT_S:
        return now, f"alive: log {age / 60:.0f} min old, {len(tree)} proc, {total_cpu:.0f} CPU-s"
    if delta is None:
        return now, f"first sample while quiet ({age / 60:.0f} min) - waiting to compare"
    if delta >= ALIVE_CPU_S:
        # Quiet but working. This is the case a log-only watchdog gets wrong.
        return now, f"SLOW not hung: log {age / 60:.0f} min old but +{delta:.0f} CPU-s"
    return now, f"STALLED: log {age / 60:.0f} min old and only +{delta:.1f} CPU-s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    common.ensure_dirs()

    previous = None
    strikes = 0
    idle = 0

    while True:
        previous, verdict = check(previous)
        common.say(f"sentry: {verdict}", SENTRY_LOG)

        if args.once:
            return 0

        if verdict.startswith("STALLED"):
            strikes += 1
            common.say(f"sentry: strike {strikes}/{CONFIRMATIONS}", SENTRY_LOG)
            if strikes >= CONFIRMATIONS:
                outcome = _kill_worker(_tree())
                common.say(f"sentry: ACTING - {outcome}", SENTRY_LOG)
                common.say(
                    "sentry: night.py should now record the stage as failed and exit; "
                    "supervise.cmd will restart it from the checkpoint",
                    SENTRY_LOG,
                )
                strikes = 0
                time.sleep(POLL_S * 3)  # let the restart settle before judging again
        else:
            strikes = 0

        if verdict.startswith("no run processes"):
            # **Stand down only when the night is actually finished**, which is a different
            # question from "is anything running right now".
            #
            # The first version stood down after one quiet poll. That was very nearly a
            # coin flip: `supervise.cmd` waits 120s between attempts and this polls every
            # 120s, so an ordinary restart gap looked identical to the end of the run and
            # the sentry would have exited silently, leaving the rest of the night
            # unwatched with nothing to say so.
            #
            # `night_pending.py` is the authoritative answer, so ask it.
            if _nothing_pending():
                common.say("sentry: no work pending either - standing down", SENTRY_LOG)
                return 0
            idle += 1
            common.say(
                f"sentry: nothing running but work is pending - the supervisor is "
                f"probably between attempts (quiet poll {idle}/{IDLE_LIMIT})",
                SENTRY_LOG,
            )
            if idle >= IDLE_LIMIT:
                common.say(
                    "sentry: pending work but nothing has started for a long time - the "
                    "supervisor may be dead. Standing down so this is visible in the log.",
                    SENTRY_LOG,
                )
                return 0
        else:
            idle = 0
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
