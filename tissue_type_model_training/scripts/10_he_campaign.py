"""The unattended H&E campaign: four RGB arms, exported, fitted and published.

    python scripts/10_he_campaign.py --go          # run it
    python scripts/10_he_campaign.py --watch       # read-only progress, then exit
    python scripts/10_he_campaign.py --plan        # what it would do, and skip

**One process, one writer, resumable.** Everything is driven from the stage table
below. Each stage records its own start, end, exit code and reason into
`he_campaign_state.json` at the workspace root, and the human-readable
`HE_CAMPAIGN_TRACKER.md` is rewritten in full from that JSON after every transition and
every sixty seconds. Rewritten, never appended: the duplicated stage lines in
`PROGRESS.md` are what two processes appending to one progress file produces, and a
reader who wakes at three in the morning should not have to work out which half is
current.

**Cheapest field of view first.** 672 -> 448 -> 224 -> 112. The 112 um arm is two
thirds of the compute on its own, so ordering it last means a blown budget lands inside
it with three arms already banked, rather than inside the first arm with none.

**One arm's failure never stops another.** An export that dies takes the rest of its
own field of view with it - there is nothing to compute features over - and then the
loop moves to the next. The tracker names what failed and why, with the last twenty
lines of its log.

**A stall is reported, not acted on.** If the running stage's log stops growing for
twenty minutes the tracker says `STALLED`, because `02_export.py` prints per region and
`03_features.py` per twenty batches, so silence that long is genuinely wrong. But only
the *budget* kills a stage: a slow-but-working feature pass is an hour of real work and
throwing it away on a heuristic would be worse than waiting.

**On exceeding the total budget the driver finishes the stage it is in.** Killing a
process inside `torch.save` is how a half-written ninety-megabyte checkpoint gets
published. Then it writes the remaining commands, in order, into the tracker and stops.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
PY = sys.executable

sys.path.append(str(WORKSPACE))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402

#: This stage's exports, under this version's `v<N>_data/data/` tree.
DATA = data_versions.data_root() / "tissue_type_model_training"
#: Where heads are published - the same directory `backend_path.MODELS_DIR` names.
MODELS = data_versions.models_root() / "tissue_type"

#: The campaign writes its state, its tracker and its per-stage logs into one
#: directory. They were at the workspace root, which put three generated files
#: among the hand-written ones; they describe these logs, so they live with them.
LOG_DIR = REPO / "reports" / "he_campaign"
STATE_PATH = LOG_DIR / "he_campaign_state.json"
TRACKER_PATH = LOG_DIR / "HE_CAMPAIGN_TRACKER.md"

sys.path.insert(0, str(REPO / "src"))

#: The four arms, cheapest first, with the mpp that gives each field of view at the
#: fixed 224 px input. Only `mpp` moves - holding the pixel side constant is what
#: leaves the backbone untouched and makes the four comparable.
ARMS: tuple[tuple[int, float], ...] = ((672, 3.0), (448, 2.0), (224, 1.0), (112, 0.5))

#: Every tree, named rather than defaulted. The tree list *is* the dataset, so it
#: belongs in the log where somebody can read what was actually cut.
TREES = ("dcis", "ic", "normal", "bach_insitu", "bach_invasive", "bach_normal")

TOTAL_BUDGET_MIN = 360.0
STALL_MINUTES = 20.0
POLL_SECONDS = 30.0
TRACKER_REFRESH_SECONDS = 60.0

#: Below this the driver stops rather than starting another stage. The pre-flight
#: gate is 12 GB; this is the running one, and it is lower because by then several
#: gigabytes have legitimately been written.
MIN_FREE_GB = 8.0


def now() -> str:
    """Local time with its offset. Nobody should have to guess the zone at 3 a.m."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def free_gb() -> float:
    import shutil

    return shutil.disk_usage(str(WORKSPACE.anchor or "C:\\")).free / 1e9


# --- resume predicates --------------------------------------------------------
#
# The recorded status is primary; these are the fallback for a state file lost
# mid-run, and they *verify* rather than check for presence. A killed
# `savez_compressed` leaves a file of plausible size that `np.load` may open far
# enough to look valid, and treating that as done is the difference between skipping
# a stage and forty minutes of silently wrong features.


def store_dir(fov: int) -> Path:
    return DATA / "he" / f"{fov}um"


def export_done(fov: int, mpp: float) -> bool:
    store = store_dir(fov)
    summary = store / "export_summary.json"
    if not (store / "tiles_manifest.csv").exists() or not summary.exists():
        return False
    try:
        meta = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if float((meta.get("spec") or {}).get("mpp", -1)) != mpp:
        return False
    if str((meta.get("input") or {}).get("channel")) != "rgb_he":
        return False
    return (store / "bcss").is_dir() and (store / "beetle").is_dir()


def paired_done(fov: int, mpp: float) -> bool:
    path = store_dir(fov) / "reports" / "paired.json"
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("ok"))
    except (OSError, json.JSONDecodeError):
        return False


def features_done(fov: int, init: str) -> bool:
    import numpy as np

    import datasets

    store = store_dir(fov)
    cache = store / "features" / f"{init}.npz"
    manifest = store / "tiles_manifest.csv"
    if not cache.exists() or not manifest.exists():
        return False
    try:
        rows = datasets.by_source_authority(datasets.read_manifest(manifest))
        want = datasets.manifest_fingerprint(rows)
        got = str(np.load(cache, allow_pickle=True)["fingerprint"])
    except Exception:
        return False
    return got.startswith(want[:16])


def published_done(fov: int) -> bool:
    import hashlib

    models_dir = MODELS
    name = head_name(fov)
    manifest_path = models_dir / f"{name}.manifest.json"
    checkpoint = models_dir / f"{name}.pt"
    if not manifest_path.exists() or not checkpoint.exists():
        return False
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("sha256")
    except (OSError, json.JSONDecodeError):
        return False
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest() == recorded


def head_name(fov: int) -> str:
    return f"invasive_tile_fov{fov}_he_concat"


def compare_done() -> bool:
    """Only done once the document names **every H&E head that is published**.

    A dry run of `10b` against the haematoxylin arms alone writes a perfectly valid
    file with four empty cells, and treating that as done would skip the stage that
    exists to fill them.

    `all`, not `any`, and the difference cost a re-run: with `any`, a document written
    part-way through the campaign - when three arms had published and the fourth had
    not - satisfied the predicate, so the final stage was skipped and the document
    never gained its last and best-powered cell. A comparison missing the arm with
    five times the held-out tiles of any other is not a comparison worth having.
    Compared against what is *published* rather than against all four, so a campaign
    that legitimately failed an arm still gets its document.
    """
    path = LOG_DIR / "HE_VS_HCHANNEL.md"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    published = [fov for fov, _ in ARMS if published_done(fov)]
    if not published:
        return False
    return all(head_name(fov) in text for fov in published)


# --- the stage table ----------------------------------------------------------


@dataclass
class Stage:
    id: str
    kind: str
    argv: list[str]
    budget_min: float
    done_when: Callable[[], bool]
    fov: int | None = None
    #: `abort` stops the campaign; `arm` skips the rest of this field of view;
    #: `continue` records the failure and carries straight on.
    on_fail: str = "continue"
    status: str = "PENDING"
    started: str | None = None
    ended: str | None = None
    elapsed_min: float = 0.0
    exit_code: int | None = None
    reason: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def log(self) -> Path:
        return LOG_DIR / f"{self.id}.log"


def build_stages() -> list[Stage]:
    stages: list[Stage] = [
        Stage(
            id="A0_preflight",
            kind="preflight",
            argv=[PY, str(REPO / "scripts" / "10a_preflight.py")],
            budget_min=5,
            done_when=lambda: False,     # always re-run; it is cheap and it is the gate
            on_fail="abort",
        ),
        Stage(
            id="S0_pytest",
            kind="tests",
            argv=[PY, "-m", "pytest", "tests", "-q"],
            budget_min=15,
            done_when=lambda: False,
            on_fail="abort",
        ),
    ]

    for fov, mpp in ARMS:
        store = store_dir(fov)
        stages.append(Stage(
            id=f"E1_{fov}_export", kind="export", fov=fov, budget_min=95 if fov == 112 else (55 if fov == 224 else 40),
            argv=[PY, str(REPO / "scripts" / "02_export.py"),
                  "--channel", "he", "--tile-px", "224", "--mpp", str(mpp),
                  "--include", *TREES],
            done_when=(lambda f=fov, m=mpp: export_done(f, m)),
            on_fail="arm",
        ))
        stages.append(Stage(
            id=f"E2_{fov}_split", kind="split", fov=fov, budget_min=15,
            argv=[PY, str(REPO / "scripts" / "07_split_export_by_source.py"),
                  "--dest", str(store), "--remove-source"],
            done_when=(lambda f=fov, m=mpp: export_done(f, m)),
            on_fail="arm",
        ))
        stages.append(Stage(
            id=f"E3_{fov}_paired", kind="pairing", fov=fov, budget_min=3,
            argv=[PY, str(REPO / "scripts" / "10c_assert_paired.py"),
                  "--a", str(DATA / "h_channel" / f"{fov}um"),
                  "--b", str(store)],
            done_when=(lambda f=fov, m=mpp: paired_done(f, m)),
            # A report, not a dependency: if the two stores diverge the head is still
            # worth fitting, it is only the *paired* statistics that become unsound,
            # and the comparison document says so rather than omitting them.
            on_fail="continue",
        ))
        for init in ("imagenet", "simclr"):
            stages.append(Stage(
                id=f"E4_{fov}_{init}", kind="features", fov=fov,
                budget_min=90 if fov == 112 else (30 if fov == 224 else 15),
                argv=[PY, str(REPO / "scripts" / "03_features.py"),
                      "--init", init,
                      "--tiles", str(store),
                      "--features", str(store / "features")],
                done_when=(lambda f=fov, i=init: features_done(f, i)),
                on_fail="arm",
            ))
        stages.append(Stage(
            id=f"E6_{fov}_publish", kind="head", fov=fov, budget_min=15,
            argv=[PY, str(REPO / "scripts" / "08_fit_concat_mlp.py"),
                  "--tiles", str(store),
                  "--features", str(store / "features"),
                  "--publish", head_name(fov)],
            done_when=(lambda f=fov: published_done(f)),
            on_fail="continue",
        ))

    stages.append(Stage(
        id="F1_compare_eight", kind="compare", budget_min=20,
        argv=[PY, str(REPO / "scripts" / "10b_compare_eight.py")],
        done_when=compare_done,
        on_fail="continue",
    ))
    return stages


# --- the tracker --------------------------------------------------------------


def atomic_write(path: Path, text: str) -> None:
    """Whole-file replace, so a reader never sees half a tracker."""
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def state_dict(stages: list[Stage], *, overall: str, current: Stage | None,
               started_iso: str, elapsed_min: float) -> dict:
    return {
        "schema": 1,
        "campaign": "he_vs_hchannel",
        "pid": os.getpid(),
        "started": started_iso,
        "updated": now(),
        "heartbeat": now(),
        "overall": overall,
        "budget_minutes_total": TOTAL_BUDGET_MIN,
        "elapsed_minutes": round(elapsed_min, 1),
        "free_gb": round(free_gb(), 1),
        "current": None if current is None else {
            "stage": current.id,
            "fov": current.fov,
            "started": current.started,
            "budget_minutes": current.budget_min,
            "log": str(current.log.relative_to(WORKSPACE)),
            "log_bytes": current.extra.get("log_bytes", 0),
        },
        "stages": [
            {
                "id": s.id, "kind": s.kind, "fov": s.fov, "status": s.status,
                "started": s.started, "ended": s.ended,
                "elapsed_minutes": round(s.elapsed_min, 1),
                "exit_code": s.exit_code, "reason": s.reason,
                "budget_minutes": s.budget_min,
                "log": str(s.log.relative_to(WORKSPACE)),
                **s.extra,
            }
            for s in stages
        ],
        "published": {
            str(fov): head_name(fov) for fov, _ in ARMS if published_done(fov)
        },
        "failures": [
            {"stage": s.id, "reason": s.reason, "log": str(s.log.relative_to(WORKSPACE))}
            for s in stages if s.status in ("FAILED", "TIMED_OUT")
        ],
    }


def log_tail(path: Path, lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


def render_tracker(state: dict, stages: list[Stage]) -> str:
    published = state["published"]
    current = state["current"]
    out: list[str] = []

    verdict = f"**{state['overall']}** · {len(published)} of 4 H&E arms published"
    if current:
        verdict += f" · `{current['stage']}` running"
    verdict += (f" · {state['elapsed_minutes']:.0f} of {TOTAL_BUDGET_MIN:.0f} min"
                f" · {state['free_gb']:.1f} GB free")

    out.append("# H&E campaign tracker")
    out.append("")
    out.append(verdict)
    out.append("")
    out.append(f"Started {state['started']} · updated {state['updated']} · pid {state['pid']}")
    out.append("")
    if state["overall"] == "STALLED":
        out.append("> **STALLED.** The running stage's log has not grown for "
                   f"{STALL_MINUTES:.0f} minutes. Nothing has been killed - the budget "
                   "does that - but this is worth looking at.")
        out.append("")

    out.append("## Stages")
    out.append("")
    out.append("| stage | fov | status | started | elapsed | tiles | note |")
    out.append("|---|---|---|---|---|---|---|")
    for s in state["stages"]:
        tiles = s.get("tiles", "")
        note = (s.get("reason") or "")[:70]
        started = (s["started"] or "")[11:19]
        out.append(
            f"| `{s['id']}` | {s['fov'] or ''} | {s['status']} | {started} | "
            f"{s['elapsed_minutes']:.1f}m | {tiles} | {note} |"
        )
    out.append("")

    out.append("## Published")
    out.append("")
    out.append("| fov | haematoxylin head | H&E head |")
    out.append("|---|---|---|")
    h_names = {112: "invasive_tile_fov112_fix1_concat", 224: "invasive_tile_fov224_concat",
               448: "invasive_tile_fov448_concat", 672: "invasive_tile_fov672_concat"}
    for fov, _ in sorted(ARMS):
        he = published.get(str(fov), "not published")
        out.append(f"| {fov} µm | `{h_names[fov]}` | `{he}` |")
    out.append("")
    out.append("> `672um` is published for completeness of the 4x2 grid and is **not "
               "interpretable**: 76 held-out tiles of which the in-situ truth row is 11, "
               "and BACH contributes nothing because a 672 µm window needs 1344 source "
               "pixels and BACH regions are 1720x1290. Do not rank it against another "
               "field of view.")
    out.append("")

    if state["failures"]:
        out.append("## Failures")
        out.append("")
        for failure in state["failures"]:
            out.append(f"### `{failure['stage']}`")
            out.append("")
            out.append(f"{failure['reason']}")
            out.append("")
            out.append(f"Log: `{failure['log']}`")
            out.append("")
            tail = log_tail(WORKSPACE / failure["log"])
            if tail:
                out.append("```")
                out.extend(tail)
                out.append("```")
                out.append("")

    out.append("## First thing in the morning")
    out.append("")
    remaining = [s for s in stages if s.status in ("PENDING", "FAILED", "TIMED_OUT")]
    if state["overall"] == "DONE" and not remaining:
        out.append("Open `HE_VS_HCHANNEL.md` - all eight heads are compared there.")
    elif not remaining:
        out.append("Nothing is outstanding. Open `HE_VS_HCHANNEL.md`.")
    else:
        out.append("Re-launching the driver skips every `DONE` stage, so the simplest "
                   "resume is:")
        out.append("")
        out.append("```powershell")
        out.append(r'$PY = ".\tissue_scoring_demo\backend\.venv\Scripts\python.exe"')
        out.append(r"& $PY tissue_type_model_training\scripts\10_he_campaign.py --go")
        out.append("```")
        out.append("")
        out.append("Or the outstanding stages one at a time, in this order:")
        out.append("")
        out.append("```powershell")
        for s in remaining:
            out.append("& $PY " + " ".join(
                part.replace(str(WORKSPACE) + os.sep, "") if isinstance(part, str) else part
                for part in s.argv[1:]
            ))
        out.append("```")
    out.append("")
    return "\n".join(out)


def publish_tracker(stages: list[Stage], *, overall: str, current: Stage | None,
                    started_iso: str, elapsed_min: float) -> None:
    state = state_dict(stages, overall=overall, current=current,
                       started_iso=started_iso, elapsed_min=elapsed_min)
    atomic_write(STATE_PATH, json.dumps(state, indent=2, sort_keys=True))
    atomic_write(TRACKER_PATH, render_tracker(state, stages))


# --- running one stage --------------------------------------------------------


def record_tiles(stage: Stage) -> None:
    """Attach the arm's tile counts to the stage, so the tracker shows the yield."""
    if stage.fov is None:
        return
    summary = store_dir(stage.fov) / "export_summary.json"
    if not summary.exists():
        return
    try:
        meta = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    stage.extra["tiles"] = meta.get("tiles_kept")
    stage.extra["per_class"] = meta.get("per_class")


def run_stage(stage: Stage, stages: list[Stage], *, started_iso: str,
              campaign_start: float) -> str:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if free_gb() < MIN_FREE_GB:
        stage.status = "FAILED"
        stage.reason = (f"only {free_gb():.1f} GB free, under the {MIN_FREE_GB:.0f} GB "
                        "floor. Refusing to start a stage that would fill the volume.")
        return "abort"

    stage.status = "RUNNING"
    stage.started = now()
    stage_start = time.monotonic()
    publish_tracker(stages, overall="RUNNING", current=stage,
                    started_iso=started_iso,
                    elapsed_min=(time.monotonic() - campaign_start) / 60)

    with stage.log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n==== {stage.id} attempt started {now()} ====\n")
        handle.write("     " + " ".join(stage.argv) + "\n\n")
        handle.flush()
        process = subprocess.Popen(
            stage.argv, cwd=str(REPO), stdout=handle,
            stderr=subprocess.STDOUT, text=True,
        )

        last_size = -1
        last_growth = time.monotonic()
        last_tracker = 0.0
        stalled = False

        while True:
            code = process.poll()
            if code is not None:
                break

            elapsed_min = (time.monotonic() - stage_start) / 60
            if elapsed_min > stage.budget_min:
                process.kill()
                process.wait(timeout=60)
                stage.status = "TIMED_OUT"
                stage.elapsed_min = elapsed_min
                stage.ended = now()
                stage.reason = (
                    f"killed after {elapsed_min:.0f} min, over its {stage.budget_min:.0f} "
                    "min budget. Last lines of the log:\n"
                    + "\n".join(log_tail(stage.log, 8))
                )
                return stage.on_fail

            size = stage.log.stat().st_size if stage.log.exists() else 0
            if size != last_size:
                last_size = size
                last_growth = time.monotonic()
                stalled = False
            elif (time.monotonic() - last_growth) / 60 > STALL_MINUTES:
                stalled = True

            stage.extra["log_bytes"] = size
            if time.monotonic() - last_tracker > TRACKER_REFRESH_SECONDS:
                publish_tracker(
                    stages, overall="STALLED" if stalled else "RUNNING", current=stage,
                    started_iso=started_iso,
                    elapsed_min=(time.monotonic() - campaign_start) / 60,
                )
                last_tracker = time.monotonic()

            time.sleep(POLL_SECONDS)

    stage.exit_code = process.returncode
    stage.elapsed_min = (time.monotonic() - stage_start) / 60
    stage.ended = now()
    record_tiles(stage)

    if process.returncode == 0:
        stage.status = "DONE"
        return "ok"

    stage.status = "FAILED"
    stage.reason = (f"exit code {process.returncode}. Last lines of the log:\n"
                    + "\n".join(log_tail(stage.log, 8)))
    return stage.on_fail


# --- the campaign -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--go", action="store_true", help="actually run")
    parser.add_argument("--plan", action="store_true", help="print the stage table and exit")
    parser.add_argument("--watch", action="store_true",
                        help="print the tracker's top every 30 s, read-only, then exit "
                             "when the campaign is no longer RUNNING")
    args = parser.parse_args()

    stages = build_stages()

    if args.plan:
        print(f"{'stage':<22} {'fov':>5} {'budget':>7}  done?")
        for stage in stages:
            try:
                done = stage.done_when()
            except Exception:
                done = False
            print(f"{stage.id:<22} {stage.fov or '':>5} {stage.budget_min:>6.0f}m  {done}")
        print(f"\ntotal budget {TOTAL_BUDGET_MIN:.0f} min · {free_gb():.1f} GB free")
        return 0

    if args.watch:
        while True:
            if not STATE_PATH.exists():
                print("no state file yet")
                return 1
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            current = state.get("current") or {}
            print(f"{state['updated']}  {state['overall']:<16} "
                  f"{current.get('stage', '-'):<22} "
                  f"{state['elapsed_minutes']:.0f}m  "
                  f"{len(state['published'])}/4 published  "
                  f"{state['free_gb']:.1f} GB free", flush=True)
            if state["overall"] not in ("RUNNING", "STALLED"):
                return 0
            time.sleep(POLL_SECONDS)

    if not args.go:
        parser.error("pass --go to run, --plan to preview, or --watch to observe")

    started_iso = now()
    campaign_start = time.monotonic()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    overall = "RUNNING"
    skip_fov: set[int] = set()

    for index, stage in enumerate(stages):
        elapsed_min = (time.monotonic() - campaign_start) / 60

        if stage.fov is not None and stage.fov in skip_fov:
            stage.status = "SKIPPED_UPSTREAM_FAILED"
            stage.reason = f"an earlier stage of the {stage.fov} um arm failed"
            continue

        if elapsed_min > TOTAL_BUDGET_MIN:
            overall = "BUDGET_EXCEEDED"
            for later in stages[index:]:
                if later.status == "PENDING":
                    later.reason = "not started - the total budget was exhausted"
            break

        try:
            if stage.done_when():
                stage.status = "SKIPPED_DONE"
                stage.reason = "its output is already on disk and verifies"
                record_tiles(stage)
                publish_tracker(stages, overall=overall, current=None,
                                started_iso=started_iso, elapsed_min=elapsed_min)
                continue
        except Exception as exc:
            stage.extra["resume_check_error"] = repr(exc)

        outcome = run_stage(stage, stages, started_iso=started_iso,
                            campaign_start=campaign_start)

        if outcome == "abort":
            overall = "FAILED"
            for later in stages[index + 1:]:
                if later.status == "PENDING":
                    later.reason = f"not started - {stage.id} aborted the campaign"
            break
        if outcome == "arm" and stage.fov is not None:
            skip_fov.add(stage.fov)

        publish_tracker(stages, overall=overall, current=None,
                        started_iso=started_iso,
                        elapsed_min=(time.monotonic() - campaign_start) / 60)

    if overall == "RUNNING":
        published = sum(1 for fov, _ in ARMS if published_done(fov))
        failed = [s for s in stages if s.status in ("FAILED", "TIMED_OUT")]
        overall = "DONE" if published == len(ARMS) and not failed else "PARTIAL"

    publish_tracker(stages, overall=overall, current=None, started_iso=started_iso,
                    elapsed_min=(time.monotonic() - campaign_start) / 60)

    print(f"\ncampaign {overall} after "
          f"{(time.monotonic() - campaign_start) / 60:.0f} min")
    print(f"tracker: {TRACKER_PATH}")
    return 0 if overall in ("DONE", "PARTIAL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
