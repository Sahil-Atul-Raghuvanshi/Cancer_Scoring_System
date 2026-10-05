"""Gate G4: fit the three-class head and report honestly.

    python scripts/04_train.py --init imagenet          # approach 1
    python scripts/04_train.py --init simclr            # approach 3
    python scripts/04_train.py --compare                # both, with the paired test

Minutes of compute and the whole argument of the project. What is fitted is a
`Linear(512, 3)` on cached features; what is decided is whether this pipeline can tell
invasive carcinoma from carcinoma still inside a duct.

Three rules, each of which has sunk a pathology model before:

  class weights    inversely proportional to frequency, and demonstrated by ablation
                   rather than asserted.
  grouped splits   GroupKFold by slide for selection, then one report on the held-out
                   institutions. Never a random tile split.
  the one cell     invasive <-> non-invasive, reported on its own and never folded into
                   a macro average. That number is the project.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

import backend_path  # noqa: F401
import bcss
import datasets
import report
import train

TARGET_DICE = 0.75


def load_cache(init: str, *, invert: bool, standardise: bool,
               fingerprint: str,
               features_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    suffix = f"{'_inverted' if invert else ''}{'_std' if standardise else ''}"
    path = (features_dir or backend_path.FEATURES_DIR) / f"{init}{suffix}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing. Run: python scripts/03_features.py --init {init}"
            + (" --invert" if invert else "")
            + (" --standardise" if standardise else "")
        )

    cache = np.load(path, allow_pickle=False)
    # The check that prevents the quietest failure in the plan: features paired with
    # the wrong manifest train to about chance and look like a model that will not learn.
    if str(cache["fingerprint"]) != fingerprint:
        raise ValueError(
            f"{path.name} was built from a different tile manifest. Re-run "
            f"scripts/03_features.py --init {init} against the current export."
        )
    return cache["features"], cache["labels"]


def print_fit(fit: train.Fit) -> None:
    scored = fit.held_out
    print(f"\n=== {fit.init} : held-out institutions "
          f"({' '.join(sorted(bcss.TEST_INSTITUTIONS))}) ===")
    print(report.format_matrix(scored.matrix))
    print()
    print("  " + scored.headline())
    print(f"  accuracy {scored.accuracy:.3f} | macro F1 {scored.macro_f1:.3f}")
    for name, value in scored.per_class_dice.items():
        print(f"    dice {name:<26} {value:.3f}")
    print(f"  recall: invasive {scored.invasive_recall:.3f} | "
          f"in-situ {scored.non_invasive_recall:.3f}")

    if scored.by_source:
        # Two class-1 populations with two different meanings, so two lines. The BCSS
        # line is the only one where a person drew the boundary; the BRACS line is the
        # head's agreement with the teacher that labelled it. Averaging them would let
        # thousands of teacher labels drown out the nine that are real.
        print()
        print("  by label source:")
        for source, block in sorted(scored.by_source.items()):
            drawn = block["labels_drawn_by"]
            n1 = block["class_tiles"]["non_invasive_epithelium"]
            recall = block["non_invasive_recall"]
            shown = "n/a" if recall is None else f"{recall:.3f}"
            print(f"    {source:<12} {block['tiles']:>6,} tiles, labels by {drawn}")
            print(f"    {'':<12} accuracy {block['accuracy']:.3f} | "
                  f"in-situ recall {shown} (n={n1})")
    print(f"  GroupKFold on the training institutions: "
          f"{fit.fold_mean:.3f} +/- {fit.fold_sd:.3f}  "
          f"{[round(f, 3) for f in fit.folds]}")

    print(f"\n  {'tumour content':<16}{'tiles':>9}{'accuracy':>11}{'dice inv':>11}")
    for row in scored.by_tumour_content:
        accuracy = "-" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
        dice = "-" if row["dice_invasive"] is None else f"{row['dice_invasive']:.3f}"
        print(f"  {row['bin']:<16}{row['tiles']:>9,}{accuracy:>11}{dice:>11}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=("imagenet", "simclr"), default="imagenet")
    parser.add_argument("--compare", action="store_true",
                        help="fit both initialisations and run the paired comparison")
    parser.add_argument("--invert", action="store_true", help="ablation A2")
    parser.add_argument("--standardise", action="store_true",
                        help="use the per-tile-p99 feature cache. Costs ~0.001 "
                             "accuracy on held-out BCSS and buys G0b's 4/6 -> 6/6 "
                             "coverage of the served IHC distribution.")
    parser.add_argument(
        "--all-sources", action="store_true",
        help="let every source contribute every class - the old behaviour, in which "
             "BRACS supplied 270 invasive and 662 non-epithelium tiles guessed by "
             "BEETLE inside DCIS-selected ROIs. Kept so the two can be compared.")
    parser.add_argument(
        "--tiles", type=Path, default=None,
        help="the tile export to read, e.g. data/224_um. Defaults to data/tiles. "
             "Must be the same export `03_features.py` was pointed at - the manifest "
             "fingerprint is checked against the cache, so a mismatch fails rather "
             "than fitting a head on vectors whose rows mean something else.")
    parser.add_argument(
        "--features", type=Path, default=None,
        help="the feature cache to read. Defaults to data/features.")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    backend_path.ensure_dirs()

    tiles_dir = args.tiles or backend_path.TILES_DIR
    features_dir = args.features or backend_path.FEATURES_DIR
    print(f"tiles    : {tiles_dir}")
    print(f"features : {features_dir}")

    rows = datasets.read_manifest(tiles_dir / "tiles_manifest.csv")
    if not args.all_sources:
        before = len(rows)
        rows = datasets.by_source_authority(rows)
        dropped = before - len(rows)
        print(f"source authority: dropped {dropped:,} tiles a source is not "
              f"authoritative for ({datasets.SOURCE_CLASSES})")
    # The fingerprint is over the *filtered* rows, so running 03 and 04 with
    # different `--all-sources` settings is caught by the cache check rather than
    # silently fitting a head on features whose rows mean something else.
    fingerprint = datasets.manifest_fingerprint(rows)
    summary = json.loads((tiles_dir / "export_summary.json").read_text())
    content = {row["roi_id"]: row["invasive_share"] for row in summary["per_region"]}

    train_rows, test_rows = datasets.institution_split(rows)
    print(f"{len(rows):,} tiles | train {len(train_rows):,} "
          f"({len({r['slide_id'] for r in train_rows})} slides) / "
          f"test {len(test_rows):,} ({len({r['slide_id'] for r in test_rows})} slides)")
    weights = datasets.class_weights(train_rows)
    print(f"class weights: "
          f"{dict(zip(bcss.CLASS_NAMES, [round(float(w), 3) for w in weights]))}")

    inits = ("imagenet", "simclr") if args.compare else (args.init,)
    fits: dict[str, train.Fit] = {}

    for init in inits:
        features, labels = load_cache(init, invert=args.invert,
                                      standardise=args.standardise,
                                      fingerprint=fingerprint,
                                      features_dir=features_dir)

        def on_fold(number, scored, tiles, init=init):
            print(f"  [{init}] fold {number}: dice invasive vs in-situ "
                  f"{scored.dice_invasive:.3f} ({tiles:,} tiles)", flush=True)

        print(f"\n== fitting {init} ==")
        fit = train.evaluate(init, features, labels, rows, tumour_content=content,
                             folds=args.folds, on_fold=on_fold)
        fits[init] = fit
        print_fit(fit)

        payload = train.as_json(fit)

        naive = train.ablate_class_weights(features, labels, rows)
        payload["ablation_no_class_weights"] = {
            "accuracy": naive.accuracy,
            "dice_invasive_vs_non_invasive": naive.dice_invasive,
            "confusion": naive.matrix.tolist(),
        }
        print(f"\n  ablation, no class weights: accuracy {naive.accuracy:.3f} "
              f"(vs {fit.held_out.accuracy:.3f}), dice invasive "
              f"{naive.dice_invasive:.3f} (vs {fit.held_out.dice_invasive:.3f})")
        if naive.accuracy >= fit.held_out.accuracy:
            print("  ^ the unweighted model has the HIGHER accuracy and the worse "
                  "matrix. That is the point being made about accuracy.")

        # _std in the name, or the two variants overwrite each other's report and the
        # ablation that justified the choice becomes unreproducible.
        suffix = (f"{init}{'_inverted' if args.invert else ''}"
                  f"{'_std' if args.standardise else ''}")
        out = backend_path.REPORTS_DIR / f"04_report_{suffix}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        torch.save(fit.head.state_dict(), backend_path.REPORTS_DIR / f"04_head_{suffix}.pt")
        print(f"  wrote {out.name} and 04_head_{suffix}.pt")

    if args.compare:
        base, cand = fits["imagenet"], fits["simclr"]
        print("\n" + "=" * 76)
        print("== G4b: the paired comparison ==")
        print(f"{'':<44}{'imagenet':>10}{'simclr':>10}{'delta':>11}")

        def line(name: str, left: float, right: float) -> None:
            print(f"{name:<44}{left:>10.3f}{right:>10.3f}{right - left:>+11.3f}")

        line("dice invasive vs non-invasive (held out)",
             base.held_out.dice_invasive, cand.held_out.dice_invasive)
        line("macro F1", base.held_out.macro_f1, cand.held_out.macro_f1)
        line("accuracy", base.held_out.accuracy, cand.held_out.accuracy)
        line("recall, invasive",
             base.held_out.invasive_recall, cand.held_out.invasive_recall)
        line("recall, in-situ",
             base.held_out.non_invasive_recall, cand.held_out.non_invasive_recall)
        line("GroupKFold mean", base.fold_mean, cand.fold_mean)
        line("GroupKFold sd", base.fold_sd, cand.fold_sd)

        paired = report.paired_comparison(base.held_out.by_slide, cand.held_out.by_slide)
        print(f"\n  {paired.headline()}")
        if paired.p_value is not None:
            print(f"  Wilcoxon signed-rank p = {paired.p_value:.4f}")

        if paired.significant:
            winner = "simclr" if paired.mean_difference > 0 else "imagenet"
            print(f"\n  VERDICT: {winner} wins; the interval excludes zero.")
        else:
            winner = "imagenet"
            print("\n  VERDICT: inside the noise. Ship the simpler one (imagenet) and")
            print("  write that down - the fold spread above is the same size as the gap.")

        payload = {
            "winner": winner,
            "paired": {
                "slides": paired.slides,
                "mean_difference": paired.mean_difference,
                "ci": [paired.ci_low, paired.ci_high],
                "wins": paired.wins, "losses": paired.losses, "ties": paired.ties,
                "p_value": paired.p_value, "significant": paired.significant,
            },
            "arms": {init: train.as_json(fit) for init, fit in fits.items()},
        }
        out = backend_path.REPORTS_DIR / "04_ab_imagenet_vs_simclr.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  wrote {out.name}")
        print(f"\nNext: python scripts/05_publish.py --init {winner}")
        return 0

    held = fits[args.init].held_out
    dice = held.dice_invasive

    # **A metric needs ground truth to be a metric.** Measured on the real download:
    # the six held-out institutions contain ZERO class-1 tiles - all 9 in the whole
    # dataset are in training institutions (AR 7, A2 2), because BCSS carries `dcis` in
    # one region of 150 and that region is in AR.
    #
    # With an empty in-situ row, "dice invasive vs non-invasive" degenerates into
    # invasive-vs-non-epithelium, which is a different and much easier question. An
    # earlier version of this gate reported 0.951 and said OK - a real number attached
    # to the wrong name, which is worse than no number at all.
    import numpy as np  # noqa: PLC0415
    support = int(np.asarray(held.matrix)[bcss.NON_INVASIVE].sum())

    print(f"\n== G4: dice invasive vs non-invasive {dice:.3f} "
          f"(target >= {TARGET_DICE}) ==")

    if support == 0:
        print("  G4 NOT MEASURABLE - and this is a finding, not a failure.")
        print()
        print("  The held-out institutions contain ZERO non-invasive tiles, so the pair")
        print("  this project exists to separate has no ground truth to be scored on.")
        print(f"  The {dice:.3f} above is invasive vs NON-EPITHELIUM wearing the wrong")
        print("  name, and it must not be quoted as the headline number.")
        print()
        print("  What IS measurable here, and should be reported instead:")
        names = bcss.CLASS_NAMES
        print(f"    dice {names[bcss.NON_EPITHELIUM]:<24} "
              f"{held.per_class_dice[names[bcss.NON_EPITHELIUM]]:.3f}")
        print(f"    dice {names[bcss.INVASIVE]:<24} "
              f"{held.per_class_dice[names[bcss.INVASIVE]]:.3f}")
        print("  i.e. a working epithelium detector, with the invasive/in-situ boundary")
        print("  untested. Say exactly that in the report.")
        print()
        print("  The fix is not a bigger model or a longer fit - no backbone recovers a")
        print("  class the held-out data does not contain. It is a better label source:")
        print("  see approach-4a-training-plan.md (BEETLE teacher), or a few hundred")
        print("  DCIS tiles clicked on our own slides in QuPath, which beats both.")
    elif support < 30:
        print(f"  G4 WEAK - only {support} non-invasive tiles in the held-out set.")
        print("  Report this Dice with its tile count and a confidence interval, or do")
        print("  not report it. A cell resting on tens of tiles is a curiosity.")
    elif dice >= TARGET_DICE:
        print("  G4 OK - publish it with scripts/05_publish.py")
    elif dice >= 0.60:
        print("  Short of target. The next lever is unfreezing layer4 (notebook 05),")
        print("  ~1 h on CPU measured - and only worth it if the gain beats the fold spread.")
    else:
        print("  Well short. Do NOT reach for a bigger model: at this level the usual")
        print("  cause is a label or alignment fault. Re-check G2's tile panel first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
