"""Both reported numbers against the pathologists', side by side.

    python scripts/compare_with_readers.py CAN_00270

`verify_case_scores.py` checks that the reported pair re-derives from the stored
rows - that the arithmetic is right. This checks something different and harder:
whether the answer is *close to what four pathologists said*, on both numbers,
with the readers' own disagreement printed beside it so "close" has a scale.

**The readers' spread is the scale, not a perfect answer.** There is no true
percent positive for a slide. Four trained readers produce four numbers, and the
width of that disagreement is the resolution of the measurement. A system inside
that width has done what can be done; one reported against an imagined correct
value is being measured against a fiction. OncoStem's own re-review rule puts the
width at +/-10 absolute points (SOP 8.3, 8.4), and no reading in their 120
exceeds it.

Intensity gets no pass band, deliberately. Reader disagreement on intensity
reaches 0.625 on a 0-2 scale, which no 10 per cent rule would tolerate, and
whether the rule applies to intensity at all is open question Q4.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app import panel  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.services.case_score_service import case_score_service  # noqa: E402
from app.services.validation_service import (  # noqa: E402
    _normalise_case,
    _same_case,
    validation_service,
)

TOLERANCE = 10.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    args = parser.parse_args()

    readers = validation_service.readers()
    pairs = case_score_service.pairs(args.case_id)
    ours_key = _normalise_case(args.case_id)

    rows = []
    for letter in panel.SCORED_MARKERS:
        found = pairs.get(letter)
        if not found:
            continue
        he, ihc = found
        path = settings.scores_dir / f"{he}__{ihc}" / "report.json"
        if not path.is_file():
            continue
        score = json.loads(path.read_text(encoding="utf-8"))["score"]

        cases = readers.get(letter, {})
        match = next((key for key in cases if _same_case(ours_key, key)), None)
        readings = cases.get(match, []) if match else []
        if not readings:
            continue

        percents = [r[0] for r in readings]
        intensities = [r[1] for r in readings if r[1] is not None]
        rows.append(
            {
                "letter": letter,
                "name": panel.spec(letter).name,
                "our_percent": score["percent"],
                "their_percent": float(np.mean(percents)),
                "percent_spread": max(percents) - min(percents),
                "our_intensity": score["intensity"],
                "their_intensity": float(np.mean(intensities)) if intensities else None,
                "intensity_spread": (
                    max(intensities) - min(intensities) if intensities else None
                ),
                "cells": score["cells"],
            }
        )

    if not rows:
        print("nothing scored for this case has a matching reader row")
        return 1

    print(f"{args.case_id}: both numbers against the four readers' consensus")
    print()
    print("PERCENT POSITIVE")
    print(
        f"{'mk':>3} {'marker':<14} {'ours':>6} {'readers':>8} {'diff':>7} "
        f"{'their spread':>13} {'within +/-10':>13}"
    )
    for row in rows:
        diff = row["our_percent"] - row["their_percent"]
        ok = "yes" if abs(diff) <= TOLERANCE else "NO"
        print(
            f"{row['letter']:>3} {row['name']:<14} {row['our_percent']:>5}% "
            f"{row['their_percent']:>7.1f}% {diff:>+7.1f} {row['percent_spread']:>12.1f} "
            f"{ok:>13}"
        )

    errors = [row["our_percent"] - row["their_percent"] for row in rows]
    inside = sum(1 for e in errors if abs(e) <= TOLERANCE)
    print()
    print(
        f"    mean absolute error {np.mean(np.abs(errors)):.1f} points   "
        f"bias {np.mean(errors):+.1f} points   "
        f"{inside} of {len(errors)} inside the human spread"
    )

    print()
    print("INTENSITY (0-2 scale; no pass band - see Q4)")
    print(
        f"{'mk':>3} {'marker':<14} {'ours':>6} {'readers':>8} {'diff':>7} "
        f"{'their spread':>13}"
    )
    gaps = []
    for row in rows:
        if row["their_intensity"] is None:
            print(f"{row['letter']:>3} {row['name']:<14} {row['our_intensity']:>6} "
                  f"{'no column':>8}")
            continue
        diff = row["our_intensity"] - row["their_intensity"]
        gaps.append(diff)
        print(
            f"{row['letter']:>3} {row['name']:<14} {row['our_intensity']:>6} "
            f"{row['their_intensity']:>8.3f} {diff:>+7.3f} "
            f"{row['intensity_spread']:>12.3f}"
        )
    if gaps:
        print()
        print(
            f"    mean absolute error {np.mean(np.abs(gaps)):.3f}   "
            f"bias {np.mean(gaps):+.3f}   "
            f"readers disagree among themselves by up to "
            f"{max(r['intensity_spread'] for r in rows if r['intensity_spread'] is not None):.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
