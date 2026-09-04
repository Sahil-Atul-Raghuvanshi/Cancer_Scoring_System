"""Fit the segmentation network. Safe to re-run: it resumes from its own last epoch.

    python scripts/02_train_seg.py --smoke        # 2 epochs on 200 tiles, ~10 min
    python scripts/02_train_seg.py                # the real run
    python scripts/02_train_seg.py --epochs 20    # a different recipe = a different run

**Re-running is how this is meant to be used.** A checkpoint is written after every epoch,
and a later invocation with the same recipe and the same tile set picks up where it
stopped. Once the epochs are done, running it again re-scores and rewrites the report
without training - which is also how you regenerate the report after changing only
`segreport.py`.

Changing the recipe changes the fingerprint, so it starts a fresh run into a differently
named checkpoint rather than continuing someone else's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classes  # noqa: E402
import paths  # noqa: E402
import segexport  # noqa: E402
import segtrain  # noqa: E402


def report_path(recipe: segtrain.Recipe) -> Path:
    return paths.REPORTS_DIR / f"02_train_{recipe.init}_{recipe.fingerprint()}.json"


def print_scored(scored: dict) -> None:
    """The held-out numbers, with the two label sources kept apart."""
    for source in sorted(scored["by_source"]):
        block = scored["by_source"][source]
        print(f"\n  === {source} ({block['tiles']:,} tiles, labels by "
              f"{block['labels_drawn_by']}) ===")
        print(f"      {block['labelled_pixels']:,} labelled pixels, "
              f"accuracy {block['accuracy']}")
        print(f"      {'class':<26}{'Dice':>8}{'IoU':>8}{'recall':>8}"
              f"{'bF1':>8}{'pixels':>12}")
        for name, entry in block["per_class"].items():
            print(f"      {name:<26}{str(entry['dice']):>8}{str(entry['iou']):>8}"
                  f"{str(entry['recall']):>8}{str(entry['boundary_f1']):>8}"
                  f"{entry['pixels']:>12,}")
        print(f"      in-situ called invasive {block['in_situ_called_invasive']:,} | "
              f"invasive called in-situ {block['invasive_called_in_situ']:,}")

    human = scored["by_source"].get(classes.SOURCE_BCSS)
    if human:
        print("\n  The `bcss` block is the one that answers 'is the model right'.")
        print("  The `bracs_dcis` block answers 'does it agree with BEETLE', which is a")
        print("  different and weaker question - those labels were reviewed by nobody.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=("imagenet", "simclr"), default="imagenet")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tiles-per-epoch", type=int, default=None,
                        help="draw only N tiles per epoch. The model still sees every "
                             "tile over the run, and each epoch stays short enough that "
                             "the validation curve is readable while it runs.")
    parser.add_argument("--no-standardise", dest="standardise", action="store_false",
                        help="train without per-tile p99 standardisation. Off by default "
                             "is wrong for the served domain - gate G0b moved percentile "
                             "coverage from 1/6 to 6/6 with it on.")
    parser.add_argument(
        "--all-sources", dest="source_authority", action="store_false",
        help="let every source teach every class - the old behaviour, in which BRACS "
             "supplied 58.0 M non-epithelium and 16.0 M invasive pixels predicted by "
             "BEETLE inside DCIS-selected ROIs. Kept so the two can be compared.")
    parser.add_argument("--val-limit", type=int, default=400,
                        help="tiles scored between epochs. The full set is scored once at "
                             "the end regardless.")
    parser.add_argument("--heartbeat", type=int, default=5,
                        help="print a progress line every N batches. 0 turns it off, "
                             "which is right for a log nobody watches and wrong for a "
                             "window somebody does.")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs on 200 tiles. Proves the loss falls before an "
                             "overnight run is committed to.")
    args = parser.parse_args()

    paths.ensure_dirs()
    rows = segexport.read_manifest()

    recipe = segtrain.Recipe(
        epochs=2 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        tiles_per_epoch=200 if args.smoke else args.tiles_per_epoch,
        init=args.init,
        standardise=args.standardise,
        source_authority=args.source_authority,
    )

    print(f"== train: {recipe.init}, {recipe.epochs} epochs, batch {recipe.batch_size} ==")
    print(f"   recipe fingerprint {recipe.fingerprint()}")
    print(f"   {len(rows):,} tiles in the manifest")
    if recipe.source_authority:
        print(f"   source authority: {dict(sorted((k, sorted(v)) for k, v in classes.SOURCE_CLASSES.items()))}")
        print("     BRACS teaches in-situ only; its non-epithelium and invasive pixels "
              "are BEETLE's predictions inside DCIS-selected regions and are blanked")
    else:
        print("   source authority OFF - every source teaches every class")
    print(f"   checkpoint {segtrain.checkpoint_path(recipe)}")
    if args.smoke:
        print("   SMOKE RUN - the numbers below mean nothing except 'it learns'")

    started = time.time()
    result = segtrain.train(
        rows, recipe=recipe,
        val_limit=50 if args.smoke else args.val_limit,
        # A smoke run must not pay for a full held-out scoring: 4,537 tiles is minutes of
        # forward passes plus a per-tile boundary F1, and the point here is only that the
        # loss falls.
        final_limit=100 if args.smoke else None,
        heartbeat=args.heartbeat,
    )
    net = result.pop("net")

    print(f"\n== held out: {result['tiles']['test']:,} tiles from "
          f"{result['patients']['test']} patients ==")
    print_scored(result["held_out"])
    print(f"\n  {result['headline']}")
    print(f"  {(time.time() - started) / 60:.1f} min")

    path = report_path(recipe)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  report: {path}")

    if args.smoke:
        losses = result["history"]["train_loss"]
        if len(losses) >= 2 and losses[-1] >= losses[0]:
            print("\n  SMOKE FAILED - the loss did not fall. Do not start the real run.")
            return 1
        print("\n  smoke OK - the loss falls. Next: python scripts/02_train_seg.py")
    else:
        print("\nNext: python scripts/03_publish_seg.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
