"""Assert two tile stores of one field of view describe the same squares.

    python scripts/10c_assert_paired.py --a ../v1_data/data/tissue_type_model_training/h_channel/224um --b ../v1_data/data/tissue_type_model_training/he/224um

**Why this is an assertion and not a comment.** `export.export_region` computes the
grid and the vote from the mask alone: the white point, the deconvolution and the
quantisation all happen after the vote is taken. So one spec and one region list should
produce the same `tile_id`s with the same labels no matter which variant stored the
pixels. Every paired statistic in the eight-model comparison rests on that - McNemar on
the in-situ rows is a test on *matched pairs*, and matched means the same square of the
same section.

Nothing else in the pipeline would notice it failing. Two stores with different tile
sets still train, still score, still produce a confusion matrix; the comparison between
them would simply be answering a question nobody asked. So it is checked, once per arm,
before the features are computed - and it is checked as an exact sorted-list comparison
rather than a count, because two stores can hold the same number of different tiles.

This is also the reason the two stores are cut in two independent passes rather than
dual-written from one loop over one region. Run separately, the equality below is a real
assertion about the exporter. Produced by one loop it would be a tautology.

**The most likely real cause of a mismatch is that a region tree has grown.** The `ic`
tree records 115 deferred regions; if BEETLE has been re-run over any of them since the
haematoxylin arms were cut, the tree has more regions now and the two exports genuinely
saw different data. `10a_preflight.py` checks the per-tree counts before the export
starts, so that case is caught in seconds rather than after half an hour of cutting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets  # noqa: E402

#: Columns the vote itself produced. If any of these differ for a shared tile id, the
#: two exports did not see the same mask and the pairing is worthless even though the
#: ids line up.
VOTE_COLUMNS = (
    "label",
    "label_name",
    "source",
    "slide_id",
    "institution",
    "x",
    "y",
    "usable",
    "invasive_frac",
    "non_invasive_frac",
    "non_epithelium_frac",
)

#: Geometry fields that must agree between the two export summaries. Not the whole
#: spec: `white_percentile` is meaningless on the RGB store, so requiring it to match
#: would be requiring a number nobody used.
SPEC_FIELDS = ("tile_px", "mpp", "stride", "tile_um")


def load(store: Path) -> tuple[dict[str, dict[str, str]], dict]:
    manifest = store / "tiles_manifest.csv"
    summary = store / "export_summary.json"
    for path in (manifest, summary):
        if not path.exists():
            raise SystemExit(f"{path} is missing - {store} is not a finished export")

    rows = datasets.read_manifest(manifest)
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        tile_id = str(row["tile_id"])
        if tile_id in by_id:
            raise SystemExit(
                f"{manifest} lists {tile_id} twice. A tile id is the join key between "
                "the two stores, so a duplicate makes the pairing ambiguous."
            )
        by_id[tile_id] = row

    return by_id, json.loads(summary.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, required=True,
                        help="the haematoxylin store, e.g. data/h_channel/224um")
    parser.add_argument("--b", type=Path, required=True,
                        help="the H&E store, e.g. data/he/224um")
    parser.add_argument("--expect-a", default="haematoxylin")
    parser.add_argument("--expect-b", default="rgb_he")
    args = parser.parse_args()

    a_rows, a_summary = load(args.a)
    b_rows, b_summary = load(args.b)

    problems: list[str] = []

    # --- the channels are what they claim ---------------------------------
    for store, summary, expected in ((args.a, a_summary, args.expect_a),
                                     (args.b, b_summary, args.expect_b)):
        found = str((summary.get("input") or {}).get("channel", "haematoxylin"))
        if found != expected:
            problems.append(f"{store} stores {found!r}, expected {expected!r}")

    # --- one geometry ------------------------------------------------------
    a_spec = a_summary.get("spec") or {}
    b_spec = b_summary.get("spec") or {}
    for field in SPEC_FIELDS:
        if a_spec.get(field) != b_spec.get(field):
            problems.append(
                f"spec.{field}: {args.a} has {a_spec.get(field)!r}, "
                f"{args.b} has {b_spec.get(field)!r}"
            )

    # --- the same tiles ----------------------------------------------------
    a_ids = sorted(a_rows)
    b_ids = sorted(b_rows)
    if a_ids != b_ids:
        only_a = sorted(set(a_ids) - set(b_ids))
        only_b = sorted(set(b_ids) - set(a_ids))
        problems.append(
            f"the tile sets differ: {len(a_ids):,} in {args.a.name} and "
            f"{len(b_ids):,} in {args.b.name}; {len(only_a):,} only in the first "
            f"({only_a[:10]}) and {len(only_b):,} only in the second ({only_b[:10]})"
        )

    # --- the same vote on each shared tile ---------------------------------
    shared = sorted(set(a_ids) & set(b_ids))
    disagreements: list[str] = []
    for tile_id in shared:
        row_a, row_b = a_rows[tile_id], b_rows[tile_id]
        for column in VOTE_COLUMNS:
            if row_a.get(column) != row_b.get(column):
                disagreements.append(
                    f"{tile_id}.{column}: {row_a.get(column)!r} vs {row_b.get(column)!r}"
                )
    if disagreements:
        problems.append(
            f"{len(disagreements):,} vote fields disagree on shared tiles, e.g.\n    "
            + "\n    ".join(disagreements[:10])
        )

    result = {
        "ok": not problems,
        "a": str(args.a),
        "b": str(args.b),
        "tiles_a": len(a_ids),
        "tiles_b": len(b_ids),
        "shared": len(shared),
        "spec": {field: a_spec.get(field) for field in SPEC_FIELDS},
        "problems": problems,
    }

    reports = args.b / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "paired.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"\n== pairing: {args.a.name} vs {args.b.name} ==")
    print(f"  tiles      : {len(a_ids):,} / {len(b_ids):,}, shared {len(shared):,}")
    print(f"  geometry   : {a_spec.get('tile_px')} px @ {a_spec.get('mpp')} um/px "
          f"= {a_spec.get('tile_um')} um")

    if problems:
        print("\n  NOT PAIRED:")
        for problem in problems:
            print(f"    - {problem}")
        print("\n  Every paired statistic between these two stores would be comparing "
              "\n  squares that are not the same squares. Refusing to record a pass.")
        return 1

    print(f"  PAIRED - identical tile ids and identical votes on all {len(shared):,}.")
    print("  A per-tile McNemar between heads fitted on these two stores is sound.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
