"""The watcher: STATUS.md every few minutes, and a guard against a stage that has hung.

    python watch.py           loop until the night reports finished
    python watch.py --once    write STATUS.md once and exit

**Slow is not hung.** A Cellpose fine-tune can run for hours printing nothing while it
burns every core; killing that would be the mistake the scoring harness already paid for
once. So a stage is killed only when two independent signals agree, on three polls in a
row: no heartbeat and no log line for `QUIET_MIN` minutes, **and** its whole process tree
averaging under `IDLE_CPU` of one core over the last poll. The night then records the
failure and retries the stage from its checkpoints.
"""

from __future__ import annotations

import sys
import time

import psutil

import common

sys.path.insert(0, str(common.ROOT / "score_all_slides"))
import pipeline as process_tools  # noqa: E402

POLL_S = 300
QUIET_MIN = 60
IDLE_CPU = 0.03
STRIKES = 3
LOG = common.LOGS / "watch.log"
HEARTBEATS = {
    "fields": "fields", "a1_fields": "a1", "a1_lynsec": "a1", "lynsec_prep": "lynsec_prep",
    "env_cellpose": "env", "env_deepliif": "env", "env_hovernet": "env",
    "a3_cellpose": "a3_cellpose", "a3_hovernet": "a3_hovernet", "a2_deepliif": "a2_deepliif",
    "download": "download",
}


def _tree_cpu(pid: int) -> float:
    total = 0.0
    for p in process_tools.process_tree(pid):
        try:
            times = psutil.Process(p).cpu_times()
            total += times.user + times.system
        except psutil.Error:
            continue
    return total


def _last_activity(stage: str) -> float:
    beat = common.read_json(common.STATE / f"{HEARTBEATS.get(stage, stage)}.heartbeat.json") or {}
    log = common.LOGS / f"{stage}.log"
    latest = max(float(beat.get("at", 0)), log.stat().st_mtime if log.exists() else 0.0)
    if stage == "download":
        # A download's progress is its file growing, not a heartbeat: curl writes no log
        # line while it streams and uses almost no CPU waiting on the network, so without
        # this a healthy 3 GB download read as hung.
        for path in common.DOWNLOADS.glob("*"):
            if path.is_file():
                latest = max(latest, path.stat().st_mtime)
    return latest


def _downloads() -> list[str]:
    lines = []
    for path in sorted(common.DOWNLOADS.glob("*")):
        if path.is_file() and path.suffix in (".zip", ".tar"):
            lines.append(f"- {path.name}: {path.stat().st_size / 1e6:,.0f} MB")
    for ok in sorted(common.DOWNLOADS.glob("*.ok")):
        lines.append(f"- {ok.stem}: verified")
    return lines


def write_status(strikes: dict[str, int]) -> dict:
    night = common.read_json(common.STATE / "night.json", {}) or {}
    lines = ["# P-03 night - status", "", f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')}. "
             f"Night finished: **{night.get('finished', False)}**. Results so far: REPORT.md.", "",
             "| Stage | State | Attempts | Last activity | Note |", "|---|---|---|---|---|"]
    for name, entry in (night.get("stages") or {}).items():
        quiet = (time.time() - _last_activity(name)) / 60
        lines.append(f"| {name} | {entry.get('state')} | {entry.get('attempts', 0)} | "
                     f"{quiet:.0f} min ago | {str(entry.get('note', ''))[:80]} |")
    lines += ["", "## Downloads", "", *(_downloads() or ["- none yet"]), "",
              "## Hang guard", "", f"Strikes: {strikes or 'none'}", ""]
    (common.OUT / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    return night


def main() -> int:
    common.ensure_dirs()
    once = "--once" in sys.argv
    strikes: dict[str, int] = {}
    previous_cpu: dict[str, tuple[float, float]] = {}
    while True:
        night = write_status(strikes)
        if once:
            return 0
        if night.get("finished"):
            common.say("night finished; watcher exiting", LOG)
            return 0
        for name, entry in (night.get("stages") or {}).items():
            if entry.get("state") != "running" or not entry.get("pid"):
                strikes.pop(name, None)
                continue
            pid = int(entry["pid"])
            if not psutil.pid_exists(pid):
                continue
            cpu, now = _tree_cpu(pid), time.time()
            last_cpu, last_at = previous_cpu.get(name, (cpu, now))
            previous_cpu[name] = (cpu, now)
            rate = (cpu - last_cpu) / max(1.0, now - last_at)
            quiet_min = (now - _last_activity(name)) / 60
            if quiet_min >= QUIET_MIN and rate < IDLE_CPU and now - last_at > 1:
                strikes[name] = strikes.get(name, 0) + 1
                common.say(f"{name}: quiet {quiet_min:.0f} min and {rate:.1%} CPU - strike {strikes[name]}", LOG)
                if strikes[name] >= STRIKES:
                    common.say(f"{name}: judged hung - killing process tree {pid}", LOG)
                    process_tools.kill_tree(pid)
                    strikes.pop(name, None)
            else:
                strikes.pop(name, None)
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
