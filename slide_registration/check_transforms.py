"""Is each stored transform anatomically possible? Checks all 30 pairs and judges them.

    python check_transforms.py
    python check_transforms.py --only CAN_00303

**This is the check the method sweep does not do.** The sweep picked a winner per pair on
mutual information, and mutual information is blind to whether a warp is physically
possible: a free-form B-spline can fold tissue through itself, tear it, or stretch a region
threefold, and each of those can *raise* MI by lining up intensity statistics. The result
scores well and is anatomically nonsense.

That matters more here than in most registration problems, because the transform's job is
to carry an invasive-tumour boundary and the reported percentages are measured **inside**
that boundary. An impossible warp puts the mask on the wrong cells and nothing downstream
can tell - every number still looks healthy.

### What is measured, and what each threshold is for

| measure | limit | why |
|---|---|---|
| folded fraction | must be **0** | a negative Jacobian determinant is tissue folded through itself: topologically impossible, not merely unlikely |
| Jacobian 1st-99th pct | inside `JACOBIAN_BAND` | local area change. Serial sections stretch a few per cent being floated onto glass; tens of per cent is the transform inventing anatomy |
| non-rigid excess | under `EXCESS_LIMIT_UM` | how far the free-form part moves tissue *beyond* the rigid fit it started from. This is the only part free to invent, so it is judged on its own |

A pair that fails is not silently dropped - it is reported with its numbers, because the
decision to fall back to a rigid transform for that pair changes which tissue its score is
measured inside, and that is not a decision a script should take on its own.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

#: Local area change a serial section can plausibly show. A section is a few microns thick,
#: floated on water and picked up on glass; it stretches, but it does not double in area in
#: one place and halve in another. Outside this band the transform is describing something
#: that did not happen to the tissue.
JACOBIAN_BAND = (0.70, 1.45)

#: How far the free-form part may move tissue beyond the rigid fit, in microns. Sections of
#: one block are cut microns apart; the real local disagreement between them is tens of
#: microns, not hundreds.
EXCESS_LIMIT_UM = 400.0

WORK_DIM = 1024


def check_pair(case: str, marker: str, log=None) -> dict:
    directory = common.case_dir(case)
    manifest = common.read_json(directory / "transforms.json") or {}
    saved = (manifest.get("saved") or {}).get(marker)
    if not saved or not saved.get("ok"):
        return {"ok": False, "error": "no stored transform"}

    render = common.read_json(directory / "render.json") or {}
    he = (render.get("slides") or {}).get("HE") or {}
    aligned = common.read_json(directory / "aligned.json") or {}
    work_scale = float(manifest["work_scale"])

    request = {
        "transform": str((directory / "transforms" / saved["file"]).resolve()),
        # The H&E's own tissue, which is where the boundary being carried lives.
        "mask": str((directory / "render" / he["maskFile"]).resolve()),
        "work_dim": WORK_DIM,
        # One working pixel is this many microns.
        "mpp_at_work": float(aligned["targetMpp"]) / work_scale,
    }
    with tempfile.TemporaryDirectory() as scratch:
        req = pathlib.Path(scratch) / "request.json"
        res = pathlib.Path(scratch) / "response.json"
        req.write_text(json.dumps(request), encoding="utf-8")
        completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
            [str(common.VALIS_PYTHON), str(common.VALIS_DIR / "check_plausible.py"),
             str(req), str(res)],
            capture_output=True, text=True, timeout=900,
        )
        if not res.exists():
            return {"ok": False, "error": f"worker wrote nothing (exit {completed.returncode})"}
        payload = json.loads(res.read_text(encoding="utf-8"))

    payload["method"] = saved.get("method")
    payload["nmi"] = saved.get("nmi")
    return payload


def verdict(result: dict) -> tuple[str, list[str]]:
    """Pass, or the reasons it does not."""
    if not result.get("ok"):
        return "ERROR", [str(result.get("error"))[:80]]

    problems = []
    if result.get("folded_points", 0) > 0:
        problems.append(
            f"FOLDED at {result['folded_points']} point(s) "
            f"({result['folded_fraction']:.2%}) - tissue through itself"
        )
    lo, hi = JACOBIAN_BAND
    if result.get("jacobian_p01", 1.0) < lo:
        problems.append(f"compresses to {result['jacobian_p01']:.2f}x")
    if result.get("jacobian_p99", 1.0) > hi:
        problems.append(f"stretches to {result['jacobian_p99']:.2f}x")
    excess = result.get("nonrigid_excess_p95_um")
    if excess is not None and excess > EXCESS_LIMIT_UM:
        problems.append(f"free-form part moves tissue {excess:.0f} um beyond the rigid fit")
    return ("pass" if not problems else "FAIL"), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    common.ensure_dirs()

    names = [c.strip() for c in args.only.split(",") if c.strip()] or common.cases()

    print(f"{'case':12} {'mk':3} {'method':9} {'jac p01':>8} {'jac p99':>8} "
          f"{'folded':>7} {'disp um':>8} {'excess':>7}  verdict")
    print("-" * 104)
    rows = []
    for case in names:
        manifest = common.read_json(common.case_dir(case) / "transforms.json")
        if not manifest or not manifest.get("ok"):
            print(f"{case:12} (no stored transforms)")
            continue
        for marker in sorted(manifest.get("saved", {})):
            result = check_pair(case, marker)
            call, problems = verdict(result)
            rows.append({"case": case, "marker": marker, "verdict": call,
                         "problems": problems, **{k: v for k, v in result.items()
                                                  if k != "traceback"}})
            print(
                f"{case:12} {marker:3} {str(result.get('method','-')):9} "
                f"{str(result.get('jacobian_p01','-')):>8} {str(result.get('jacobian_p99','-')):>8} "
                f"{str(result.get('folded_points','-')):>7} "
                f"{str(result.get('displacement_median_um','-')):>8} "
                f"{str(result.get('nonrigid_excess_p95_um','-')):>7}  "
                f"{call}" + (f" - {'; '.join(problems)}" if problems else "")
            )

    common.write_json(common.STATE / "transform_checks.json", {"rows": rows})
    failed = [r for r in rows if r["verdict"] != "pass"]
    print("-" * 104)
    print(f"{len(rows) - len(failed)}/{len(rows)} plausible, {len(failed)} not")
    if failed:
        print()
        print("Pairs that need a decision (fall back to the rigid fit, or accept):")
        for r in failed:
            print(f"  {r['case']}/{r['marker']} ({r.get('method')}): {'; '.join(r['problems'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
