"""What BEETLE's refinement is worth, in score units.

    python compare_scores.py

The two scoring runs differ in **one thing only**: which regions step 12 carries. Both use
the same slides, the same `mattes` registration, the same nuclei segmentation and the same
cut points. So the difference between them is what step 11 buys - which is the question the
whole two-pass exercise exists to answer, and which no amount of area statistics can settle.

Area coverage says BEETLE keeps 13.4% of what the tile grid claims. That is a fact about
*area*, and it does not translate to a fact about *scores*: a percentage measured inside a
region is not proportional to that region's size. If most of the discarded area holds few
cells, or cells that stain like the kept ones, the score barely moves. If the discarded
area is full of weakly-stained stroma, it moves a lot. Only running both tells you which.

**Read the per-marker differences, not the mean.** A mean across markers with different
biology and different cell counts is a number without a referent.
"""

from __future__ import annotations

import csv
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

TILES = common.RESULTS / "oncostem_ai_scores_tiles.csv"
BEETLE = common.RESULTS / "oncostem_ai_scores.csv"

#: Rows measured on fewer cells than this are reported with a warning. At 100 cells one
#: cell is worth a full percentage point, so a score quoted to the nearest percent implies
#: precision the measurement does not have.
THIN_DENOMINATOR = 400


def load(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    out = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("state") != "scored":
                continue
            out[(row["case_id"], row["marker"])] = row
    return out


def main() -> int:
    tiles, beetle = load(TILES), load(BEETLE)
    print(f"tiles rows: {len(tiles)}   BEETLE rows: {len(beetle)}")

    shared = sorted(set(tiles) & set(beetle))
    if not shared:
        print("\nNo pair has been scored both ways yet - nothing to compare.")
        print("The tiles pass runs first; the comparison fills in as BEETLE catches up.")
        return 0

    print(f"\n=== {len(shared)} pair(s) scored both ways ===")
    print(f"{'case':12} {'mk':3} {'no BEETLE':>10} {'with BEETLE':>12} {'diff':>7} "
          f"{'cells tiles':>12} {'cells BEETLE':>13}")
    print("-" * 78)

    diffs = []
    for key in shared:
        a, b = tiles[key], beetle[key]
        try:
            d = float(b["percent"]) - float(a["percent"])
            diffs.append(d)
            shown = f"{d:+.0f}"
        except (TypeError, ValueError):
            shown = "-"
        flag = ""
        for row in (a, b):
            try:
                if int(row["cells"]) < THIN_DENOMINATOR:
                    flag = "  <- thin denominator"
            except (TypeError, ValueError):
                pass
        print(f"{key[0]:12} {key[1]:3} {a['percent']+'%':>10} {b['percent']+'%':>12} "
              f"{shown:>7} {a['cells']:>12} {b['cells']:>13}{flag}")

    if diffs:
        moved = [d for d in diffs if abs(d) >= 5]
        print("-" * 78)
        print(f"median difference   {st.median(diffs):+.1f} points")
        print(f"largest             {max(diffs, key=abs):+.0f} points")
        print(f"pairs moving >=5pts {len(moved)} of {len(diffs)}")
        print(f"pairs unchanged     {sum(1 for d in diffs if d == 0)} of {len(diffs)}")
        print()
        print("A positive difference means BEETLE's refinement raised the score - the tile")
        print("region was diluting it with tissue that is not invasive carcinoma.")

    only_tiles = sorted(set(tiles) - set(beetle))
    only_beetle = sorted(set(beetle) - set(tiles))
    if only_tiles or only_beetle:
        print()
        if only_tiles:
            print(f"scored only without BEETLE ({len(only_tiles)}): "
                  + ", ".join(f"{c}/{m}" for c, m in only_tiles[:8]))
        if only_beetle:
            print(f"scored only with BEETLE ({len(only_beetle)}): "
                  + ", ".join(f"{c}/{m}" for c, m in only_beetle[:8]))
        print("A pair scored one way and not the other is itself a result: it says the")
        print("region source changed whether the pair could be measured at all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
