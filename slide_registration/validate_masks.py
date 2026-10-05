"""Check Phase 1's tissue masks against the cohort's own published measurements.

    python validate_masks.py [--mpp 8.0] [--render]

The registration is only ever as good as the tissue it is given, and the tissue mask is the
one part of this pipeline with an independent ground truth to check against: the cohort
characterisation measured every slide with its own optical-density rule at 7.11 um/px and
published the table. If a mask here disagrees with that by more than a little, the mask is
wrong - and a wrong mask is invisible downstream, because a registration on the wrong
tissue still returns a transform.

This exists because a first version of `render.py` failed exactly that way: an optical
density floor set to clear scanner padding, when padding had already been removed by
flatness, clamped the threshold up past the tissue on pale sections. CAN_00865's near-blank
CD44 slide measured 0.05 mm2 against a published 17.5, and CAN_00267's pale H&E measured
4.6 against 163.5 - while every number on CAN_00303 looked perfect. One case agreeing
proves nothing; this checks all of them.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tissue_scoring_demo" / "backend"))

import common  # noqa: E402

#: Tissue area in mm2 per slide, from `data/oncostem_docs/client/OncoStem/analysis/
#: cohort_characterisation.md` section 3, measured with an optical-density rule at
#: 7.11 um/px. Independent of anything in this directory, which is what makes it a test
#: rather than a restatement.
PUBLISHED = {
    "CAN_00251": {"HE": 166.3, "A": 95.0, "F": 176.8, "R": 133.6, "U": 203.4, "W": 129.1},
    "CAN_00259": {"HE": 315.4, "A": 261.1, "F": 368.5, "R": 295.6, "U": 338.5, "W": 315.5},
    "CAN_00267": {"HE": 163.5, "A": 39.3, "F": 160.0, "R": 144.6, "U": 293.1, "W": 114.0},
    "CAN_00270": {"HE": 189.8, "A": 101.2, "F": 228.3, "R": 177.1, "U": 247.4, "W": 123.0},
    "CAN_00303": {"HE": 35.8, "A": 31.6, "F": 35.9, "R": 33.4, "U": 43.6, "W": 31.5},
    "CAN_00865": {"HE": 74.3, "A": 17.5, "F": 91.1, "R": 84.8, "U": 96.1, "W": 84.8},
}

#: How far a mask may sit from the published figure before it is called wrong. Generous
#: on purpose: the two measurements use different resolutions (8 vs 7.11 um/px), different
#: cleanup and different island rules, so exact agreement is not expected and would be
#: suspicious. What is being caught here is an order-of-magnitude miss, not a percent.
TOLERANCE = 0.35


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpp", type=float, default=8.0)
    parser.add_argument("--render", action="store_true", help="re-render before checking")
    args = parser.parse_args()
    common.ensure_dirs()

    if args.render:
        import render

        for case in sorted(PUBLISHED):
            try:
                render.render_case(case, args.mpp)
            except Exception as failure:  # noqa: BLE001 - report and keep going
                common.say(f"{case}: render failed - {type(failure).__name__}: {failure}")

    print(f"\n{'case':12} {'slide':5} {'measured':>9} {'published':>10} {'ratio':>7}  verdict")
    print("-" * 62)
    failures = 0
    checked = 0
    for case in sorted(PUBLISHED):
        manifest = common.read_json(common.case_dir(case) / "render.json")
        if manifest is None:
            print(f"{case:12} (not rendered)")
            continue
        for code, expected in PUBLISHED[case].items():
            entry = manifest["slides"].get(code)
            if entry is None:
                print(f"{case:12} {code:5} {'-':>9} {expected:>10.1f}      -  MISSING")
                failures += 1
                continue
            measured = entry["tissueMm2"]
            ratio = measured / expected if expected else 0.0
            ok = abs(ratio - 1.0) <= TOLERANCE
            checked += 1
            if not ok:
                failures += 1
            print(
                f"{case:12} {code:5} {measured:>9.1f} {expected:>10.1f} {ratio:>7.2f}  "
                f"{'ok' if ok else 'WRONG'}"
            )

    print("-" * 62)
    print(f"{checked - failures}/{checked} within {TOLERANCE:.0%}, {failures} outside")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
