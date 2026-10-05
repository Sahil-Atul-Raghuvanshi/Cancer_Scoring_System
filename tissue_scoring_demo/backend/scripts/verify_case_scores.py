"""Re-derive a case's scores from the stored rows, without the scoring code.

    python scripts/verify_case_scores.py CAN_00270

Step 16 is a separate module with no image processing in it precisely so that a
disputed number can be re-derived with a calculator. This script is that
calculator. It reads the same two things a reviewer would - step 14's
`cells.json` rows and the cut-point file - and recomputes the pair with plain
Python arithmetic, importing nothing from `step18_aggregate`.

**It is not a unit test and it is not a second implementation to be kept in
sync.** A unit test checks the code against itself; this checks the *reported
numbers* against the *stored evidence*, which is the thing a reader of the
workbook actually wants to know. If it ever disagrees with the report, one of
them is wrong and the disagreement is the finding.

The only shared input is `app.scoring.cuts`, because the thresholds genuinely
are the same data - re-typing them here would test nothing except whether the
copy was current.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import panel  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.scoring import cuts as cut_points  # noqa: E402
from app.services.case_score_service import case_score_service  # noqa: E402


def round5(value: float) -> int:
    """Nearest 5, halves up. Written out rather than imported, on purpose."""
    return int(math.floor(value / 5 + 0.5) * 5)


def band_of(od: float, table) -> float:
    for ceiling, value in table:
        if ceiling is None or od < ceiling:
            return float(value)
    return float(table[-1][1])


def verify(case_id: str) -> int:
    pairs = case_score_service.pairs(case_id)
    marker_cuts = cut_points.cut_set()
    failures = 0

    print(f"case {case_id}: re-deriving {len(pairs)} marker(s) from the stored rows\n")

    for letter in panel.SCORED_MARKERS:
        found = pairs.get(letter)
        if not found:
            continue
        he, ihc = found
        spec = panel.spec(letter)
        cuts = marker_cuts.for_marker(letter)

        # --- read step 14's rows, exactly as a reviewer would ---------------
        rows: list[dict] = []
        directory = settings.per_cell_dir / f"{he}__{ihc}"
        for path in sorted(directory.glob("region*/cells.json")):
            rows.extend(json.loads(path.read_text(encoding="utf-8")).get("cells", []))
        if not rows:
            print(f"{letter}: no stored rows; nothing to verify")
            continue

        # --- read the region areas from step 14's own report ----------------
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        areas = {int(r["rank"]): float(r["areaMm2"]) for r in report["regions"]}

        # --- recompute, by hand ---------------------------------------------
        by_region: dict[int, list[dict]] = {}
        for row in rows:
            by_region.setdefault(int(row["regionRank"]), []).append(row)

        region_percent: dict[int, float] = {}
        region_intensity: dict[int, float] = {}
        for rank, cells in by_region.items():
            positive = [
                cell
                for cell in cells
                if float(cell["intensityOd"]) >= cuts.od[0]
                and float(cell["second"]) >= cuts.second_min
            ]
            region_percent[rank] = 100.0 * len(positive) / len(cells)
            region_intensity[rank] = (
                sum(float(cell["intensityOd"]) for cell in positive) / len(positive)
                if positive
                else 0.0
            )

        weight = sum(areas.get(rank, 0.0) for rank in by_region)
        percent_raw = (
            sum(region_percent[rank] * areas.get(rank, 0.0) for rank in by_region) / weight
            if weight > 0
            else 100.0
            * sum(
                1
                for row in rows
                if float(row["intensityOd"]) >= cuts.od[0]
                and float(row["second"]) >= cuts.second_min
            )
            / len(rows)
        )

        stained = [rank for rank in by_region if region_intensity[rank] > 0]
        stained_weight = sum(areas.get(rank, 0.0) for rank in stained)
        intensity_raw = (
            sum(region_intensity[rank] * areas.get(rank, 0.0) for rank in stained)
            / stained_weight
            if stained_weight > 0
            else 0.0
        )

        mine = (round5(percent_raw), band_of(intensity_raw, cuts.od_to_band))

        # --- what the pipeline reported -------------------------------------
        stored = json.loads(
            (settings.scores_dir / f"{he}__{ihc}" / "report.json").read_text(
                encoding="utf-8"
            )
        )["score"]
        theirs = (int(stored["percent"]), float(stored["intensity"]))

        agree = mine == theirs
        failures += 0 if agree else 1
        mark = "OK " if agree else "MISMATCH"
        print(
            f"{mark} {letter} {spec.full_name:22s} "
            f"recomputed {mine[0]:>3d} % / {mine[1]:<4g}   "
            f"reported {theirs[0]:>3d} % / {theirs[1]:<4g}   "
            f"({len(rows):,} rows, raw {percent_raw:.2f} % / {intensity_raw:.4f} OD)"
        )

        # A percentage outside the spread OncoStem has actually seen is not an
        # arithmetic error, but it is the thing worth knowing about the result.
        low, high = spec.expected_percent
        if high > 0 and not (low <= theirs[0] <= high):
            print(
                f"     ^ outside the {low}-{high} % range OncoStem has reported for "
                f"{spec.name}. The arithmetic is right; the cut points are unfitted."
            )

    print()
    if failures:
        print(f"{failures} marker(s) disagree with the stored report.")
    else:
        print("Every reported pair re-derives from the stored rows.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    return verify(parser.parse_args().case_id)


if __name__ == "__main__":
    raise SystemExit(main())
