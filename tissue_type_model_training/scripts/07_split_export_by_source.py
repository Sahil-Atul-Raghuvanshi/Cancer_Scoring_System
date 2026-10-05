"""Re-file one tile export into `<dest>/{bcss,beetle,bach}/`, by which laboratory drew it.

`02_export.py` writes every tile of every source into one tree, keyed on the `source`
column, because that is what the training scripts read and what `report.by_source` keeps
apart. This script takes that tree and lays it out by **provenance** instead, which is
the shape a person wants when the question is "what did each dataset contribute at this
field of view" rather than "fit a head".

    data/tiles/                              data/224_um/
      non_epithelium/*.png                     bcss/non_epithelium/*.png
      non_invasive_epithelium/*.png     ->     beetle/non_invasive_epithelium/*.png
      invasive_epithelium/*.png                bach/invasive_epithelium/*.png
      tiles_manifest.csv                       tiles_manifest.csv   (all three, re-pathed)
      export_summary.json                      export_summary.json  (+ a per-group block)

**Three groups.** `bcss` is the only one whose labels a person drew - 151 TCGA regions,
pixel-annotated, CC0. `beetle` is the BRACS trees, where BEETLE's nnU-Net placed the
epithelium and a three-pathologist ROI consensus named the lesion. `bach` is the same
arrangement on a second laboratory's images, and it is split out from `beetle` rather
than pooled with it because being a different laboratory is the entire reason those
tiles exist - pooling them would hide the one contrast they were added to provide.

**The files move rather than copy.** A tile is 224x224 of uint8 and the export runs to
tens of thousands of them; writing a second copy would double the disk for a layout
change. The source tree is left holding nothing but its manifests, and `--remove-source`
deletes it.

**Both manifests are kept, and that is deliberate.** Each group gets its own
`tiles_manifest.csv` so it can be read alone, and the field-of-view directory keeps the
combined one with every `tile_path` re-written to include its group - so `03_features.py`
can still be pointed at `<dest>` and see one export, which a layout that only had three
separate manifests would have quietly broken.

    python scripts/07_split_export_by_source.py --dest data/224_um --remove-source
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backend_path  # noqa: F401,E402
import bcss  # noqa: E402


def group_for(source: str) -> str:
    """Which provenance group a `source` value belongs to.

    Derived from the source string rather than listed tree by tree, so a seventh tree
    lands in the right group the day it is added - and an unrecognised source raises
    rather than being filed under a default, because a tile whose provenance nobody can
    state is the one thing this layout exists to make impossible.
    """
    if source == "bcss":
        return "bcss"
    if source.startswith("bach"):
        return "bach"
    if source.startswith("bracs"):
        return "beetle"
    raise ValueError(
        f"source {source!r} belongs to no known group. Add it to `group_for` rather "
        "than letting it default - a tile filed under the wrong laboratory is a "
        "provenance claim this project reports on."
    )


def split(tiles_dir: Path, dest: Path) -> dict:
    """Move every tile into its group's subtree and write the four manifests."""
    manifest_path = tiles_dir / "tiles_manifest.csv"
    summary_path = tiles_dir / "export_summary.json"
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path} - run 02_export.py first")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("the manifest is empty")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[group_for(row["source"])].append(row)

    dest.mkdir(parents=True, exist_ok=True)
    combined: list[dict] = []
    group_blocks: dict[str, dict] = {}

    for group in sorted(by_group):
        group_rows = by_group[group]
        group_dir = dest / group
        own_rows: list[dict] = []

        for row in group_rows:
            relative = Path(row["tile_path"])
            target = group_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source_file = tiles_dir / relative
            if source_file.exists():
                shutil.move(str(source_file), str(target))
            elif not target.exists():
                raise SystemExit(
                    f"{source_file} is missing and {target} does not exist either - "
                    "the manifest and the tiles on disk disagree, so this export is "
                    "not safe to re-file."
                )
            # Unchanged, so the group manifest stands alone against its own directory.
            own_rows.append(row)
            # And prefixed, for the combined manifest one level up.
            combined.append({**row, "tile_path": str(Path(group) / relative)})

        fields = list(own_rows[0].keys())
        with (group_dir / "tiles_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(own_rows)

        counts = Counter(row["label_name"] for row in own_rows)
        block = {
            "group": group,
            "tiles": len(own_rows),
            "per_class": {name: counts.get(name, 0) for name in bcss.CLASS_NAMES},
            "sources": sorted({row["source"] for row in own_rows}),
            "regions": len({row["roi_id"] for row in own_rows}),
            "slides": len({row["slide_id"] for row in own_rows}),
            "institutions": sorted({row["institution"] for row in own_rows}),
            # Carried from the parent export so a group directory read on its own still
            # says what geometry produced it. A tile whose field of view has to be
            # inferred from a folder name is a tile that will be pooled with the wrong
            # ones eventually.
            "spec": summary.get("spec"),
            "input": summary.get("input"),
            "labels_are": (
                "human pixel annotation (BCSS, CC0)"
                if group == "bcss"
                else "BEETLE's nnU-Net located the epithelium; a pathologist ROI "
                     "consensus named the lesion. Model-assisted, research only."
            ),
        }
        group_blocks[group] = block
        (group_dir / "export_summary.json").write_text(
            json.dumps(block, indent=2), encoding="utf-8"
        )
        per_class = "  ".join(
            f"{name}={block['per_class'][name]}" for name in bcss.CLASS_NAMES
        )
        print(f"  {group:8} {len(own_rows):>6} tiles  {per_class}", file=sys.stderr)

    fields = list(combined[0].keys())
    with (dest / "tiles_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)

    out = {**summary, "grouped_by": "provenance", "by_group": group_blocks}
    (dest / "export_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tiles", type=Path, default=backend_path.TILES_DIR,
        help="the export to re-file. Defaults to 02_export.py's own output directory.",
    )
    parser.add_argument(
        "--dest", type=Path, required=True,
        help="where the grouped tree goes, e.g. data/224_um",
    )
    parser.add_argument(
        "--remove-source", action="store_true",
        help="delete the now-empty source tree afterwards",
    )
    args = parser.parse_args()

    out = split(args.tiles, args.dest)
    spec = out["spec"]
    print(f"\n  {out['tiles_kept']:,} tiles -> {args.dest}", file=sys.stderr)
    print(
        f"  field of view {spec['tile_um']:.0f} um "
        f"({spec['tile_px']} px at {spec['mpp']} um/px)",
        file=sys.stderr,
    )

    if args.remove_source:
        shutil.rmtree(args.tiles, ignore_errors=True)
        print(f"  removed {args.tiles}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
