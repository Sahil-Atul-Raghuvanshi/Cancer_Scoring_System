"""Cut `(image, mask)` tile pairs from the shared regions. Safe to re-run at any time.

    python scripts/01_export_seg_tiles.py                # all 270 regions, ~20 min
    python scripts/01_export_seg_tiles.py --limit 6      # a smoke run first
    python scripts/01_export_seg_tiles.py --bcss-only    # the permissive half alone
    python scripts/01_export_seg_tiles.py --force        # ignore the cache, redo everything

**Re-running is the normal case, not the exception.** Every region writes a shard as its
last act, and a region whose shard matches the current settings and whose tiles are all on
disk is skipped. So an interrupted export resumes, a partial export can be widened, and
running this twice in a row is a no-op that takes a second.

Changing any geometry setting changes the spec fingerprint, which invalidates every shard -
correctly, because tiles cut at a different resolution or floor are not the tiles the new
spec describes. The gate at the end refuses to hand on a manifest that mixes the two.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classes  # noqa: E402
import paths  # noqa: E402
import segexport  # noqa: E402
import splits  # noqa: E402


def gate(manifest: dict) -> int:
    """Is this dataset trainable, and is it honest about what is in it?"""
    rows = manifest["rows"]
    print("\n== G1: is the tile set trainable, and honest? ==")
    print(f"  tiles      : {manifest['tiles']:,} from {manifest['regions']} regions")
    print(f"  patients   : {manifest['patients']}")

    total_px = sum(manifest["pixels_per_class"].values()) + manifest["pixels_unlabelled"]
    print(f"  pixels     : {total_px / 1e6:,.1f} M")
    for name, count in manifest["pixels_per_class"].items():
        print(f"    {name:<26} {count / 1e6:8.2f} M  {100 * count / total_px:6.3f}%")
    print(f"    {'unlabelled':<26} {manifest['pixels_unlabelled'] / 1e6:8.2f} M  "
          f"{100 * manifest['pixels_unlabelled'] / total_px:6.3f}%")

    print("\n  by label source:")
    for source, block in sorted(manifest["by_source"].items()):
        drawn = manifest["labels_drawn_by"][source]
        print(f"    {source:<12} {block['tiles']:>6,} tiles, labels by {drawn}")
        for name in classes.CLASS_NAMES:
            print(f"    {'':<12}   {name:<26} {block[name] / 1e6:7.2f} M px")

    # Every class must exist as PIXELS, which is a different question from every class
    # existing as tiles - and it is the question a segmentation loss actually asks.
    absent = [n for n, c in manifest["pixels_per_class"].items() if c == 0]
    if absent:
        raise AssertionError(
            f"these classes have no labelled pixels at all: {absent}. That is a mapping "
            "fault or a floor set too high, not a small dataset."
        )

    # The split, and the number that makes this pipeline worth building: how much
    # PATHOLOGIST-drawn class 1 sits on the held-out side.
    train, test = splits.split(rows)
    print(f"\n  split      : {len(train):,} train / {len(test):,} test tiles")
    print(f"  weights    : {[round(w, 3) for w in splits.pixel_class_weights(train)]}")

    held_human_c1 = sum(
        int(row["px_non_invasive_epithelium"]) for row in test
        if row["source"] == classes.SOURCE_BCSS
    )
    print(f"\n  held-out class-1 pixels drawn by a PATHOLOGIST: {held_human_c1 / 1e6:.3f} M")
    if held_human_c1 == 0:
        print("  ! none, so DCIS boundaries cannot be scored against a human on this")
        print("  ! split. That is the situation the tile model was stuck in.")
    else:
        print("  This is the evaluation set. It is thin, and it is real.")

    splits.assert_no_leak(train, test)
    print("  G1 OK - no patient crosses the split, every class has pixels")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="export only the first N regions, for a smoke run")
    parser.add_argument("--bcss-only", action="store_true",
                        help="skip the BEETLE-labelled BRACS regions. The result is "
                             "CC0/BSD-3 rather than CC BY-NC-SA, at the cost of almost "
                             "all the class-1 signal.")
    parser.add_argument("--force", action="store_true",
                        help="re-cut every region even if its shard is current")
    parser.add_argument("--tile-px", type=int, default=224)
    parser.add_argument("--mpp", type=float, default=0.5)
    parser.add_argument("--min-labelled", type=float, default=0.20)
    parser.add_argument("--check-only", action="store_true",
                        help="skip the export; rebuild the manifest and run the gate")
    args = parser.parse_args()

    paths.ensure_dirs()
    paths.require_inputs()

    spec = segexport.TileSpec(
        tile_px=args.tile_px, mpp=args.mpp, min_labelled=args.min_labelled
    )
    print(f"== export: {spec.tile_px}px at {spec.mpp} um/px "
          f"({spec.tile_um:.0f} um field), floor {spec.min_labelled:.0%} ==")
    print(f"   settings fingerprint {spec.fingerprint()}")
    print(f"   reading  {paths.SHARED_DATA}")
    print(f"   writing  {paths.TILES_DIR}")

    if not args.check_only:
        regions = segexport.find_regions(include_dcis=not args.bcss_only)
        if args.limit:
            regions = regions[: args.limit]
        print(f"   {len(regions)} regions\n")

        started = time.time()
        redone = 0
        for _, _, was_redone in segexport.export_all(
            regions, spec=spec, force=args.force, progress=print
        ):
            redone += int(was_redone)
        print(f"\n   {redone} region(s) cut, {len(regions) - redone} reused, "
              f"{(time.time() - started) / 60:.1f} min")

    manifest = segexport.rebuild_manifest(spec)
    if manifest["stale_shards_ignored"]:
        print(f"\n   NOTE {manifest['stale_shards_ignored']} shard(s) from a different "
              "settings fingerprint were ignored, not mixed in.")
    print(f"   manifest: {paths.TILES_DIR / 'manifest.json'}")

    code = gate(manifest)
    print("\nNext: python scripts/02_train_seg.py --smoke")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
