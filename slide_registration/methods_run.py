"""Test every registration method on every pair, in priority order, and pick a winner.

    python methods_run.py                     every case, every method
    python methods_run.py --only CAN_00865
    python methods_run.py --methods outline,mattes
    python methods_run.py --deadline 6

Six cases x five markers = **30 H&E-to-IHC pairs over 36 slides**, each registered by
several methods and scored on one scoreboard.

**The priority order is by likelihood of solving the problem, not by sophistication.**
Every method is tried on every pair regardless - the point is the comparison - but they
run in this order so that a run cut short has still answered the important question:

    1. outline   free. The identity after phases 1-2 have put both sections on one
                 physical scale and one orientation. Already measured to beat VALIS on
                 four of CAN_00865's five markers, so it is the baseline every other
                 method has to beat rather than a fallback of last resort.
    2. mattes    Mattes mutual information over a similarity transform. **The one that
                 solves the actual problem.** On CAN_00865's near-negative CD44 section -
                 the slide VALIS scored 1.0042 on with five matched features - this scores
                 1.0488. It needs no features because it uses every pixel.
    3. affine    the same metric with shear and anisotropic scale. More freedom, and more
                 opportunity to overfit a weak signal, so it is judged not assumed.
    4. bspline   free-form non-rigid on top of the similarity fit. The only method here
                 that can follow the local stretch a section picks up on glass, which is
                 precisely what no amount of outline alignment can correct.
    5. mask      a similarity transform fitted to the tissue masks by distance transform.
                 Geometry only, stain-blind. Kept for diagnosis rather than hope - it has
                 measured worst so far.

    (valis)      the feature-based incumbent, measured separately by `run_overnight.py`
                 and joined into the same table for comparison.

**Fitted on whole tissue, never on the invasive mask.** Two tumour regions do not
constrain a transform - a roughly round region matches many positions equally well - while
a section outline does. The BEETLE invasive mask is only ever *carried* by the transform,
never used to fit it.

**Checkpointed per case.** A case's result is written the moment it completes, and a case
already done is skipped, so this is restartable and a long night can be resumed rather
than repeated.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tissue_scoring_demo" / "backend"))

import common  # noqa: E402

#: Methods in priority order. See the module docstring for why this order.
PRIORITY = ["outline", "mattes", "affine", "bspline", "mask"]

#: Longest edge the intensity methods work at. 1024 over a tissue-cropped render at
#: 8 um/px is about 16 um per working pixel on the largest section here.
WORK_DIM = 1024

#: How long one case's whole method sweep may take before it is abandoned. Generous:
#: five methods over five markers, with the non-rigid one the slow part.
CASE_TIMEOUT_S = 60 * 60

METHODS_CSV = common.RESULTS / "registration_methods.csv"


def ensure_inputs(case: str, mpp: float, log) -> None:
    """Make sure this case has renders, a measured order, and aligned images.

    Self-sufficient rather than depending on the VALIS pass having been here first, so
    this can run on a case nothing has touched.
    """
    import align
    import order
    import render

    directory = common.case_dir(case)
    if not (directory / "render.json").is_file():
        render.render_case(case, mpp, log=log)
    if not (directory / "order.json").is_file():
        order.measure_case(case, log=log)
    if not (directory / "aligned.json").is_file():
        align.write_aligned(case, log=log)


def run_case(case: str, methods: list[str], mpp: float, timeout: float, log) -> dict:
    """Every method on every marker of one case."""
    ensure_inputs(case, mpp, log)

    directory = common.case_dir(case)
    aligned = common.read_json(directory / "aligned.json")
    slides = aligned["slides"]
    if "HE" not in slides:
        raise ValueError(f"{case}: no H&E among {sorted(slides)}")

    aligned_dir = directory / "aligned"
    request = {
        "reference": str((aligned_dir / slides["HE"]["file"]).resolve()),
        "targets": {
            code: str((aligned_dir / entry["file"]).resolve())
            for code, entry in slides.items()
            if code != "HE"
        },
        "work_dim": WORK_DIM,
        "methods": methods,
    }

    python = common.VALIS_PYTHON
    script = common.VALIS_DIR / "register_methods.py"
    if not python.exists():
        raise RuntimeError(f"no VALIS environment at {python}")

    common.say(
        f"{case}: {len(request['targets'])} marker(s) x {len(methods)} method(s) "
        f"({', '.join(methods)})",
        log,
    )
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as scratch:
        request_path = pathlib.Path(scratch) / "request.json"
        response_path = pathlib.Path(scratch) / "response.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        try:
            completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
                [str(python), str(script), str(request_path), str(response_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            payload = {"ok": False, "error": f"method sweep exceeded {timeout:.0f}s"}
            common.write_json(directory / "methods.json", payload)
            return payload

        if not response_path.exists():
            tail = (completed.stderr or completed.stdout or "")[-1500:]
            payload = {
                "ok": False,
                "error": f"worker wrote no result (exit {completed.returncode}). {tail}",
            }
            common.write_json(directory / "methods.json", payload)
            return payload
        payload = json.loads(response_path.read_text(encoding="utf-8"))

    payload["case"] = case
    payload["seconds"] = round(time.monotonic() - started, 1)
    common.write_json(directory / "methods.json", payload)

    if payload.get("ok"):
        for code, entry in sorted((payload.get("slides") or {}).items()):
            scores = {
                name: attempt.get("nmi")
                for name, attempt in entry["methods"].items()
                if attempt.get("ok")
            }
            common.say(
                f"{case}: {code} best={entry['best']} ({entry['best_nmi']})  "
                + "  ".join(f"{k}={v}" for k, v in scores.items()),
                log,
            )
    else:
        common.say(f"{case}: method sweep FAILED - {str(payload.get('error'))[:200]}", log)
    return payload


def summarise(log) -> None:
    """One row per pair per method, plus VALIS joined in for comparison.

    Rebuilt from what is on disk every time rather than appended to, so it is correct
    after an interruption rather than merely usually correct.
    """
    rows = []
    for case in sorted(common.cases()):
        directory = common.case_dir(case)
        methods = common.read_json(directory / "methods.json") or {}
        valis = common.read_json(directory / "registration.json") or {}
        valis_slides = valis.get("slides") or {}

        for code, entry in sorted((methods.get("slides") or {}).items()):
            outline = entry.get("outline_nmi")
            for name, attempt in entry["methods"].items():
                rows.append(
                    {
                        "case": case,
                        "marker": code,
                        "stain": common.PANEL.get(code, ""),
                        "method": name,
                        "ok": attempt.get("ok", False),
                        "nmi": attempt.get("nmi"),
                        "outlineNmi": outline,
                        # The number that matters: did this method beat doing nothing?
                        "gainOverOutline": (
                            round(attempt["nmi"] - outline, 4)
                            if attempt.get("nmi") is not None and outline is not None
                            else None
                        ),
                        "onSlide": attempt.get("onSlide") or attempt.get("on_slide"),
                        "seconds": attempt.get("seconds"),
                        "isBest": name == entry.get("best"),
                        "error": attempt.get("error", ""),
                    }
                )
            # VALIS as one more row per pair, on the same scoreboard.
            if code in valis_slides:
                rows.append(
                    {
                        "case": case,
                        "marker": code,
                        "stain": common.PANEL.get(code, ""),
                        "method": "valis",
                        "ok": bool(valis.get("ok")),
                        "nmi": valis_slides[code].get("alignment_nmi"),
                        "outlineNmi": outline,
                        "gainOverOutline": (
                            round(valis_slides[code]["alignment_nmi"] - outline, 4)
                            if valis_slides[code].get("alignment_nmi") is not None
                            and outline is not None
                            else None
                        ),
                        "onSlide": valis_slides[code].get("probe_on_slide"),
                        "seconds": valis.get("seconds"),
                        "isBest": False,
                        "error": "",
                    }
                )

    if not rows:
        return
    with METHODS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common.say(f"methods table: {len(rows)} row(s) -> {METHODS_CSV}", log)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    parser.add_argument("--redo", default="")
    parser.add_argument("--methods", default=",".join(PRIORITY))
    parser.add_argument("--mpp", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=CASE_TIMEOUT_S)
    parser.add_argument("--deadline", type=float, default=0.0, help="hours; 0 means none")
    args = parser.parse_args()

    common.ensure_dirs()
    log = common.RUN_LOG
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    # Keep the caller's set but impose the priority order, so an interrupted run has
    # always done the cheap decisive ones first.
    methods = [m for m in PRIORITY if m in methods] + [m for m in methods if m not in PRIORITY]
    redo = {c.strip() for c in args.redo.split(",") if c.strip()}

    names = [c.strip() for c in args.only.split(",") if c.strip()] or common.cases()

    common.say("=" * 72, log)
    common.say(f"method comparison: {len(names)} case(s), methods in priority order {methods}", log)

    started = time.monotonic()
    for case in names:
        if args.deadline and (time.monotonic() - started) > args.deadline * 3600:
            common.say(f"deadline passed, not starting {case}", log)
            break
        existing = common.read_json(common.case_dir(case) / "methods.json")
        if existing and existing.get("ok") and case not in redo:
            common.say(f"{case}: method sweep already done, skipping", log)
            continue
        common.say("-" * 72, log)
        common.mark(case, methodsPhase="running")
        try:
            payload = run_case(case, methods, args.mpp, args.timeout, log)
            common.mark(
                case,
                methodsPhase="done" if payload.get("ok") else "failed",
                methodsError=None if payload.get("ok") else str(payload.get("error"))[:300],
            )
        except Exception as failure:  # noqa: BLE001 - one case must not stop the sweep
            import traceback

            common.say(f"{case}: FAILED - {type(failure).__name__}: {failure}", log)
            common.say(traceback.format_exc()[-1500:], log)
            common.mark(case, methodsPhase="failed", methodsError=f"{type(failure).__name__}: {failure}")
        summarise(log)

    common.say("=" * 72, log)
    common.say("method comparison finished", log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
