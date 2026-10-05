"""Is a region unstained because of the biology, or because it is not tumour?

    python scripts/check_region_consistency.py CAN_00270

OncoStem's SOP, clause 2.6: *"Areas unstained across **all** markers are treated
as non-representative and counted out."*

I previously recorded that this was not implementable because each marker's IHC
slide is sampled at its own coordinates, so there is no field-level
correspondence between markers. **That was wrong at the level that matters.**
The regions are traced once on the H&E and warped onto each marker's slide, so
*region rank N is the same piece of anatomy on all five slides*. The
correspondence exists one level up from where I looked for it.

This script prints the grid that decides whether clause 2.6 is worth
implementing: every carried region against every marker, as percent positive.

**What to look for.** A region that is near-zero on one marker and strongly
positive on another is real tumour with genuinely variable staining, and
excluding it would be discarding a measurement. A region that is near-zero on
*all five* is what clause 2.6 describes - and on this pipeline it is also what a
step 8 false positive looks like, and what a region that the registration
dropped onto stroma looks like. All three want the same treatment, which is why
the clause exists.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app import panel  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.services.case_score_service import case_score_service  # noqa: E402

#: Percent positive at or below which a region counts as unstained for a marker.
#: Not zero: a handful of cells over the cut in a region of thirty is noise
#: rather than staining, and clause 2.6 is about areas that carry no signal.
UNSTAINED_PERCENT = 10.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--unstained", type=float, default=UNSTAINED_PERCENT)
    args = parser.parse_args()

    pairs = case_score_service.pairs(args.case_id)
    per_marker: dict[str, dict[int, dict]] = {}
    areas: dict[int, float] = {}

    for letter, (he, ihc) in pairs.items():
        path = settings.scores_dir / f"{he}__{ihc}" / "report.json"
        if not path.is_file():
            continue
        score = json.loads(path.read_text(encoding="utf-8"))["score"]
        per_marker[letter] = {}
        for region in score["regions"]:
            per_marker[letter][region["rank"]] = region
            areas[region["rank"]] = region["areaMm2"]

    if not per_marker:
        print("no scored marker for this case")
        return 1

    letters = [letter for letter in panel.SCORED_MARKERS if letter in per_marker]
    ranks = sorted(areas)

    print(f"{args.case_id}: percent positive by region and marker")
    print(f"a region counts as unstained for a marker at or below {args.unstained:g}%")
    print()
    header = f"{'rank':>4} {'area mm2':>9} {'cells':>6}  " + "  ".join(
        f"{letter:>6}" for letter in letters
    )
    print(header)
    print("-" * len(header))

    dead: list[int] = []
    for rank in ranks:
        cells = max(
            (per_marker[letter].get(rank, {}).get("cells", 0) for letter in letters),
            default=0,
        )
        values = []
        unstained_everywhere = True
        for letter in letters:
            region = per_marker[letter].get(rank)
            if region is None:
                values.append("    -")
                unstained_everywhere = False
                continue
            percent = region["percentRaw"]
            values.append(f"{percent:>6.1f}")
            if percent > args.unstained:
                unstained_everywhere = False
        flag = "  <- unstained on every marker" if unstained_everywhere else ""
        if unstained_everywhere:
            dead.append(rank)
        print(f"{rank:>4} {areas[rank]:>9.2f} {cells:>6}  " + "  ".join(values) + flag)

    print()
    total_area = sum(areas.values())
    dead_area = sum(areas[rank] for rank in dead)
    print(
        f"{len(dead)} of {len(ranks)} regions are unstained on every marker, "
        f"{dead_area:.2f} of {total_area:.2f} mm2 ({dead_area / total_area:.0%} of the "
        "carried area)."
    )
    if dead:
        print(f"  ranks: {dead}")

    print()
    print("What each marker would report with those regions counted out (SOP 2.6):")
    print()
    print(f"{'mk':>3} {'as scored':>10} {'2.6 applied':>12} {'change':>8}")
    for letter in letters:
        kept = [
            region
            for rank, region in per_marker[letter].items()
            if rank not in dead and region["areaMm2"] > 0
        ]
        every = list(per_marker[letter].values())

        def weighted(regions: list[dict]) -> float:
            total = sum(r["areaMm2"] for r in regions)
            if total <= 0:
                return 0.0
            return sum(r["percentRaw"] * r["areaMm2"] for r in regions) / total

        before, after = weighted(every), weighted(kept)
        print(f"{letter:>3} {before:>9.1f}% {after:>11.1f}% {after - before:>+7.1f}")

    print()
    print(
        "This is a diagnostic. Nothing is excluded by the pipeline on the strength of\n"
        "it - clause 2.6 says 'unstained across all markers', and whether a 10% floor\n"
        "is what OncoStem mean by unstained is a question for them, not for this script."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
