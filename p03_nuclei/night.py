"""The P-03 night: every stage in dependency order, each checkpointed, killable, resumable.

    python night.py           run or resume until every stage is done or has given up

Downloads run in the background beside everything else - the DeepLIIF model alone is 3 GB
on a slow line - and a stage whose inputs are not ready yet waits instead of failing. Each
stage is its own process with a wall-clock cap; on timeout its whole process tree is
killed. A stage gets three attempts. After any stage finishes, the scorer re-runs so
REPORT.md always shows everything finished so far.

State is `state/night.json`; `finished: true` there is what tells the supervisor to stop.
"""

from __future__ import annotations

import subprocess
import sys
import time

import common

sys.path.insert(0, str(common.ROOT / "score_all_slides"))
import pipeline as process_tools  # noqa: E402  - process_tree / kill_tree

LOG = common.LOGS / "night.log"
STATE_FILE = common.STATE / "night.json"
MAX_ATTEMPTS = 3
NOT_READY = 2


def env_py(name: str):
    return common.ENVS / name / "Scripts" / "python.exe"


def ok(name: str) -> bool:
    return (common.DOWNLOADS / f"{name}.ok").exists() or (common.ENVS / f"{name}.ok").exists()


#: name, interpreter, script + args, cap in hours, what must exist first.
STAGES = [
    ("fields", lambda: common.BACKEND_PY, ["stage_fields.py"], 1.0, lambda: True),
    ("a1_fields", lambda: common.BACKEND_PY, ["stage_a1.py", "fields"], 3.0, lambda: done("fields")),
    ("env_cellpose", lambda: common.BACKEND_PY, ["stage_env.py", "cellpose"], 1.5, lambda: True),
    ("env_deepliif", lambda: common.BACKEND_PY, ["stage_env.py", "deepliif"], 1.5, lambda: True),
    ("lynsec_prep", lambda: common.BACKEND_PY, ["stage_lynsec_prep.py"], 1.0, lambda: ok("lynsec_data")),
    ("a1_lynsec", lambda: common.BACKEND_PY, ["stage_a1.py", "lynsec"], 1.0, lambda: done("lynsec_prep")),
    ("a3_cellpose", lambda: env_py("cellpose"), ["stage_a3_cellpose.py"], 6.0,
     lambda: done("env_cellpose") and done("lynsec_prep") and done("fields")),
    ("env_hovernet", lambda: common.BACKEND_PY, ["stage_env.py", "hovernet"], 1.5, lambda: ok("hovernet_code")),
    ("a3_hovernet", lambda: env_py("hovernet"), ["stage_a3_hovernet.py"], 4.0,
     lambda: done("env_hovernet") and ok("lynsec_model") and done("lynsec_prep") and done("fields")
     and done("bcdl_prep")),
    ("a2_deepliif", lambda: env_py("deepliif"), ["stage_a2_deepliif.py"], 5.0,
     lambda: done("env_deepliif") and ok("deepliif_model") and done("lynsec_prep") and done("fields")
     and done("bcdl_prep")),
    # Second run (6 Oct): options 1, 2, 4 and 5 - none needs a pathologist. DeepLIIF and
    # HoVer-Net above are re-opened for it and pick up only the new breast set (their
    # checkpoints skip everything already done), so they wait for it.
    ("pseudo_fields", lambda: common.BACKEND_PY, ["stage_pseudo_fields.py"], 1.5, lambda: done("fields")),
    ("bcdl_prep", lambda: common.BACKEND_PY, ["stage_bcdl_prep.py"], 1.0,
     lambda: ok("bcdl_val") and ok("bcdl_train")),
    ("a1_bcdl", lambda: common.BACKEND_PY, ["stage_a1.py", "bcdl"], 1.0, lambda: done("bcdl_prep")),
    ("renders", lambda: common.BACKEND_PY, ["stage_renders.py"], 1.0, lambda: done("bcdl_prep")),
    ("a4_zero", lambda: env_py("cellpose"), ["stage_a4_cellpose.py", "zero"], 4.0,
     lambda: done("renders") and done("pseudo_fields")),
    # a4_self (self-training on detector agreement) was run once on 6 Oct and is retired:
    # the detectors agree on only 28-40% of nuclei, so their consensus is not ground truth
    # and training on it would copy their blind spots. Training on our slides needs labels.
    ("a4_bc", lambda: env_py("cellpose"), ["stage_a4_cellpose.py", "bc"], 4.5, lambda: done("bcdl_prep")),
    ("visual", lambda: common.BACKEND_PY, ["stage_visual.py"], 0.5, lambda: settled(exclude="visual")),
]


def settled(exclude: str) -> bool:
    """Every other stage has finished one way or the other - the contact sheets go last."""
    stages = state()["stages"]
    return all((stages.get(n) or {}).get("state") in ("done", "gave_up") for n, *_ in STAGES if n != exclude)


def state() -> dict:
    return common.read_json(STATE_FILE, {}) or {"stages": {}}


def save(payload: dict) -> None:
    payload["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    common.write_json(STATE_FILE, payload)


def done(name: str) -> bool:
    return (state()["stages"].get(name) or {}).get("state") == "done"


def run_stage(name, py, args, cap_h) -> int:
    payload = state()
    entry = payload["stages"].setdefault(name, {"attempts": 0})
    entry.update(state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
    log_path = common.LOGS / f"{name}.log"
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {entry['attempts'] + 1} =====\n")
        stream.flush()
        process = subprocess.Popen([str(py), "-u", *args], cwd=str(common.HERE), stdout=stream,
                                   stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        entry["pid"] = process.pid
        save(payload)
        try:
            code = process.wait(timeout=cap_h * 3600)
        except subprocess.TimeoutExpired:
            process_tools.kill_tree(process.pid)
            process.wait(timeout=120)
            code = -9
            common.say(f"{name}: over its {cap_h}h cap - killed", LOG)
    return code


def start_download(payload: dict):
    entry = payload["stages"].setdefault("download", {"attempts": 0})
    if entry.get("state") == "done":
        return None
    stream = (common.LOGS / "download.console.log").open("a", encoding="utf-8")
    process = subprocess.Popen([str(common.BACKEND_PY), "-u", "stage_download.py"], cwd=str(common.HERE),
                               stdout=stream, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    entry.update(state="running", pid=process.pid, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(payload)
    return process


def score() -> None:
    subprocess.run([str(common.BACKEND_PY), "stage_score.py"], cwd=str(common.HERE),
                   stdout=(common.LOGS / "score.console.log").open("a", encoding="utf-8"),
                   stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, check=False)


def main() -> int:
    common.ensure_dirs()
    payload = state()
    payload["finished"] = False
    save(payload)
    common.say("night starting/resuming", LOG)
    download = start_download(payload)

    while True:
        progressed = False
        for name, py, args, cap_h, ready in STAGES:
            payload = state()
            entry = payload["stages"].setdefault(name, {"attempts": 0})
            if entry.get("state") in ("done", "gave_up"):
                continue
            if not ready():
                if entry.get("state") != "waiting":
                    entry["state"] = "waiting"
                    save(payload)
                continue
            common.say(f"{name}: starting", LOG)
            code = run_stage(name, py(), args, cap_h)
            payload = state()
            entry = payload["stages"][name]
            if code == 0:
                entry.update(state="done", finished=time.strftime("%Y-%m-%d %H:%M:%S"), rc=0)
            elif code == NOT_READY:
                entry.update(state="waiting", rc=code)
            else:
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                entry.update(rc=code, state="gave_up" if entry["attempts"] >= MAX_ATTEMPTS else "failed",
                             note=f"exit {code}; see logs/{name}.log")
            save(payload)
            common.say(f"{name}: exit {code} -> {entry['state']}", LOG)
            if code != NOT_READY:
                score()
                progressed = True
                break  # re-evaluate from the top: earlier stages may have become ready

        if download is not None and download.poll() is not None:
            payload = state()
            payload["stages"]["download"].update(
                state="done" if download.returncode == 0 else "failed", rc=download.returncode,
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            save(payload)
            common.say(f"download: exit {download.returncode}", LOG)
            if download.returncode != 0 and payload["stages"]["download"].get("attempts", 0) < MAX_ATTEMPTS:
                payload["stages"]["download"]["attempts"] = payload["stages"]["download"].get("attempts", 0) + 1
                save(payload)
                download = start_download(payload)
            else:
                download = None

        payload = state()
        remaining = [n for n, *_ in STAGES if (payload["stages"].get(n) or {}).get("state") not in ("done", "gave_up")]
        if not remaining and download is None:
            break
        if not progressed:
            if download is None and all(
                (payload["stages"].get(n) or {}).get("state") == "waiting" for n in remaining
            ):
                common.say(f"nothing can run: {remaining} are waiting on inputs that will not arrive", LOG)
                for n in remaining:
                    payload["stages"][n].update(state="gave_up", note="inputs never became available")
                save(payload)
                break
            common.heartbeat("night", f"waiting: {remaining}")
            time.sleep(120)

    score()
    payload = state()
    payload["finished"] = True
    save(payload)
    common.say("night finished", LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
