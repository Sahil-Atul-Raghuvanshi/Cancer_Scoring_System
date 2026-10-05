"""Refit each pair's winning method and store the transform for step 12 to use.

    python fit_best.py                  every case
    python fit_best.py --only CAN_00303

The method sweep's job was to *compare*; this one's job is to *keep*. It reads each pair's
winner from `methods.json`, refits that one method, and writes the transform to
`v<N>_data/data/registration/<case>/transforms/<marker>.tfm` together with a manifest recording
which method produced it and at what working scale.

**Why refit rather than save during the sweep.** The sweep ran before transform
persistence existed, and refitting one method per pair costs about a fifth of re-running
all five. It also keeps the stored set honest: exactly one transform per pair, the one the
scoreboard chose, rather than five of which four will never be used.

**Why this is a separate step at all.** The winning method is non-linear on 17 of 30 pairs,
and a B-spline warp cannot be written down as a matrix. Step 12 therefore cannot recompute
it from a few numbers in a JSON file - the transform itself has to exist on disk.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

WORK_DIM = 1024

#: The one method every pair is registered with, overriding whatever the scoreboard
#: preferred.
#:
#: **Chosen for what it cannot do, not only for what it scores.** Ranking the methods by
#: mutual information put `bspline` first, and mutual information is blind to whether a
#: warp is physically possible: measured over 25 pairs, **10 of 16 B-splines folded tissue
#: through itself** - a negative Jacobian determinant - with local stretch reaching 6.1x.
#: CAN_00303 was the worst and had scored *best*: its W marker posted the largest NMI gain
#: of the whole cohort while folding 24% of the section.
#:
#: A similarity transform has one Jacobian for the entire section, so if it is positive
#: once it is positive everywhere. The failure mode is unreachable by construction rather
#: than merely unobserved. It costs about 0.0065 of mean NMI against `bspline`, and part
#: of that gap was bought by deformation the tissue cannot have undergone.
#:
#: Set to None to go back to using each pair's scoreboard winner.
FORCE_METHOD = "mattes"


def fit_case(case: str, log=None) -> dict:
    """Refit and store the winning transform for every marker of one case."""
    directory = common.case_dir(case)
    methods = common.read_json(directory / "methods.json")
    aligned = common.read_json(directory / "aligned.json")
    if not methods or not methods.get("ok") or not aligned:
        raise ValueError(f"{case}: needs a completed method sweep first")

    aligned_dir = directory / "aligned"
    slides = aligned["slides"]
    targets = {}
    for code, entry in methods["slides"].items():
        best = entry.get("best")
        if not best or code not in slides:
            continue
        targets[code] = {
            "path": str((aligned_dir / slides[code]["file"]).resolve()),
            "method": FORCE_METHOD or best,
            "scoreboardWinner": best,
        }

    if not targets:
        raise ValueError(f"{case}: no winning methods recorded")

    request = {
        "mode": "fit_best",
        "reference": str((aligned_dir / slides["HE"]["file"]).resolve()),
        "targets": targets,
        "out_dir": str((directory / "transforms").resolve()),
        "work_dim": WORK_DIM,
    }

    common.say(
        f"{case}: refitting {len(targets)} winner(s) - "
        + ", ".join(f"{c}={t['method']}" for c, t in sorted(targets.items())),
        log,
    )
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        req = pathlib.Path(scratch) / "request.json"
        res = pathlib.Path(scratch) / "response.json"
        req.write_text(json.dumps(request), encoding="utf-8")
        completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
            [str(common.VALIS_PYTHON), str(common.VALIS_DIR / "register_methods.py"),
             str(req), str(res)],
            capture_output=True, text=True, timeout=3600,
        )
        if not res.exists():
            tail = (completed.stderr or completed.stdout or "")[-1200:]
            raise RuntimeError(f"{case}: worker wrote no result (exit {completed.returncode}). {tail}")
        payload = json.loads(res.read_text(encoding="utf-8"))

    payload["case"] = case
    payload["seconds"] = round(time.monotonic() - started, 1)
    # The manifest is what step 12 reads. It names the file, the method and the working
    # scale, so nothing downstream has to guess any of the three.
    common.write_json(directory / "transforms.json", payload)

    ok = sum(1 for v in payload["saved"].values() if v.get("ok"))
    # The worker returns ok=True when it *ran*; whether any transform was actually stored
    # is a different question, and conflating them reported "0 cases failed" while every
    # single pair had thrown.
    payload["ok"] = ok > 0
    payload["stored"] = ok
    common.say(
        f"{case}: stored {ok}/{len(targets)} transform(s) in "
        f"{payload['seconds'] / 60:.1f} min (work scale {payload['work_scale']:.4f})",
        log,
    )
    for code, v in sorted(payload["saved"].items()):
        if not v.get("ok"):
            common.say(f"{case}:   {code} FAILED - {str(v.get('error'))[:120]}", log)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    common.ensure_dirs()
    log = common.RUN_LOG

    names = [c.strip() for c in args.only.split(",") if c.strip()] or common.cases()
    common.say("=" * 72, log)
    common.say(f"storing winning transforms for {len(names)} case(s)", log)

    failures = 0
    for case in names:
        try:
            got = fit_case(case, log)
            if not got.get("ok"):
                common.say(f"{case}: stored NO transforms - every pair failed", log)
                failures += 1
        except Exception as failure:  # noqa: BLE001 - one case must not stop the rest
            common.say(f"{case}: FAILED - {type(failure).__name__}: {failure}", log)
            failures += 1
    common.say(f"transforms stored; {failures} case(s) failed", log)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
