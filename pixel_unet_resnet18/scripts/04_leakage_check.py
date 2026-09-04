"""Is "in-situ" separable from "this is a BRACS image"? The pixel-level leakage check.

    python scripts/04_leakage_check.py               # data confound + encoder probe
    python scripts/04_leakage_check.py --tiles 3000  # a bigger sample for the probe

**Why this exists, and why it is worse here than for the tile model.** BRACS supplies
**98.8 %** of this export's in-situ pixels - 89.5 M against BCSS's 1.09 M. So `source` is
very nearly a synonym for the class this whole project is built to find, and a network
can score well on in-situ epithelium by recognising *which dataset it is looking at* -
the scanner, the stain era, the crop - and never learn duct architecture at all. It would
pass the held-out institution split, pass the patient-grouped folds, and then collapse on
a slide from neither dataset, which is exactly the slide it is meant to score.

`classes.SOURCE_CLASSES` makes this **worse on purpose**: blanking BRACS's non-epithelium
and invasive pixels is the right call on label provenance, and it removes the only pixels
that were teaching the model that a BRACS image can be something other than in-situ. The
two changes have to be reported together, so this script is the companion to that rule
rather than an optional extra.

**What it measures.**

1. *The confound itself*, straight from the manifest - no model, no compute. How much of
   each class each source supplies, before and after the authority rule.
2. *Whether the encoder can tell the datasets apart*, from pooled features over a sample
   of tiles. High separability is not proof of cheating - two datasets genuinely do look
   different, and a network that sees tissue will see that too. What would be damning is
   in-situ separability collapsing once the source signal is removed.

**How to read a result.** Read it as an upper bound on how much the in-situ number can be
trusted, not as a verdict on the model. Nothing here can make that number trustworthy on
an immunostained slide; only in-domain tiles labelled by a pathologist can.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classes  # noqa: E402
import paths  # noqa: E402
import segdata  # noqa: E402
import segexport  # noqa: E402
import segnet  # noqa: E402

TICK = "  ok  "
WARN = " warn "
CROSS = " FAIL "

#: Source separability above this and the datasets are trivially distinguishable.
LOUD = 0.90

#: Below this, the projection has genuinely removed the source signal and the in-situ
#: result that follows it means something. Above, the test is inconclusive - which is a
#: real outcome and must be reported as one rather than rounded to a pass.
REMOVED = 0.65


def line(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def _probe(features: np.ndarray, target: np.ndarray, *, seed: int = 0) -> float:
    """Balanced accuracy of a logistic probe, held out on a random half.

    A random split rather than a patient-grouped one, deliberately: the question is
    whether the information is *present* in the features at all, which is the most
    generous reading - and therefore the right one for a test that passes by finding
    nothing.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    cut = len(order) // 2
    train, test = order[:cut], order[cut:]

    if len(np.unique(target[train])) < 2 or len(np.unique(target[test])) < 2:
        return float("nan")

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(features[train], target[train])
    return float(balanced_accuracy_score(target[test], model.predict(features[test])))


def _confound(rows: list[dict]) -> None:
    """The data-level confound, from the manifest. No model, no compute."""
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = totals.setdefault(row["source"], {name: 0 for name in classes.CLASS_NAMES})
        for name in classes.CLASS_NAMES:
            bucket[name] += int(row[f"px_{name}"])

    line("      ", "")
    line("      ", "annotated pixels per class, by source:")
    header = f"{'source':12s}" + "".join(f"{n[:16]:>20s}" for n in classes.CLASS_NAMES)
    line("      ", header)
    for source in sorted(totals):
        counts = totals[source]
        row_text = f"{source:12s}" + "".join(
            f"{counts[n]:>20,}" for n in classes.CLASS_NAMES
        )
        line("      ", row_text)

    line("      ", "")
    for index, name in enumerate(classes.CLASS_NAMES):
        total = sum(counts[name] for counts in totals.values())
        borrowed = totals.get(classes.SOURCE_DCIS, {}).get(name, 0)
        if not total:
            continue
        share = borrowed / total
        authoritative = index in classes.SOURCE_CLASSES.get(classes.SOURCE_DCIS, frozenset())
        note = "kept" if authoritative else "BLANKED by the authority rule"
        line(
            WARN if (authoritative and share > 0.95) else "      ",
            f"{name:26s} BRACS supplies {share:6.1%}  ({borrowed:,} px) - {note}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default="imagenet")
    parser.add_argument("--tiles", type=int, default=1500,
                        help="tiles sampled for the encoder probe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-standardise", dest="standardise", action="store_false")
    arguments = parser.parse_args()

    print("=" * 74)
    print("Dataset-as-label leakage check (pixel)")
    print("=" * 74)

    rows = segexport.read_manifest()
    line(TICK, f"{len(rows):,} tiles in the manifest")
    _confound(rows)

    # --- the encoder probe ----------------------------------------------
    print("-" * 74)
    import torch

    rng = np.random.default_rng(arguments.seed)
    pick = rng.choice(len(rows), size=min(arguments.tiles, len(rows)), replace=False)
    sampled = [rows[int(i)] for i in pick]

    dataset = segdata.SegTileDataset(
        sampled,
        paths.TILES_DIR,
        augment_data=False,
        standardise=arguments.standardise,
        # The masks are what the *rule* leaves, because the question is about the model
        # that will actually be fitted.
        source_authority=True,
    )

    net = segnet.ResNet18UNet(arguments.init).eval()
    line("      ", f"probing {len(sampled):,} tiles through the {arguments.init} encoder")

    features = np.zeros((len(sampled), 512), dtype=np.float32)
    with torch.inference_mode():
        for index in range(len(sampled)):
            image, _ = dataset[index]
            x = net.stem(image[None])
            x = net.layer1(net.pool(x))
            x = net.layer2(x)
            x = net.layer3(x)
            x = net.layer4(x)
            features[index] = x.mean(dim=(2, 3)).numpy()[0]
            if (index + 1) % 250 == 0:
                print(f"    {index + 1}/{len(sampled)}", flush=True)

    is_borrowed = np.array(
        [1 if row["source"] == classes.SOURCE_DCIS else 0 for row in sampled]
    )
    # "Does this tile contain in-situ epithelium at all", which is the tile-level shadow
    # of the pixel question and the only binary a linear probe can be asked here.
    has_in_situ = np.array(
        [1 if int(row["px_non_invasive_epithelium"]) > 0 else 0 for row in sampled]
    )

    source_acc = _probe(features, is_borrowed)
    in_situ_acc = _probe(features, has_in_situ)

    line(
        WARN if source_acc > LOUD else TICK,
        f"source is predictable from pooled encoder features at {source_acc:.3f} "
        "(0.5 is chance)",
    )
    line("      ", f"'contains in-situ' is predictable at {in_situ_acc:.3f}")

    # --- does in-situ survive losing the source direction? --------------
    from sklearn.linear_model import LogisticRegression

    direction_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    direction_model.fit(features, is_borrowed)
    direction = direction_model.coef_[0]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    projected = features - np.outer(features @ direction, direction)

    after = _probe(projected, has_in_situ)
    residual = _probe(projected, is_borrowed)

    print("-" * 74)
    line("      ", f"in-situ separability, features as they are : {in_situ_acc:.3f}")
    line("      ", f"in-situ separability, source direction gone: {after:.3f}")
    line("      ", f"source separability after removal          : {residual:.3f}")

    # The removal has to have worked before its result means anything. Projecting out one
    # direction only helps if that direction *was* the source signal; if source is still
    # readable afterwards then "in-situ survived" says nothing, because the feature the
    # model could be leaning on is still sitting there.
    if residual > REMOVED:
        line(
            WARN,
            f"INCONCLUSIVE - the projection did not remove the source signal "
            f"({source_acc:.3f} -> {residual:.3f}, chance is 0.5), so 'in-situ survived "
            "it' proves nothing. The dataset fingerprint is spread across many "
            "directions, not one.",
        )
        verdict = None
    else:
        verdict = after >= in_situ_acc - 0.05
        line(
            TICK if verdict else CROSS,
            (
                "in-situ survives with the source signal genuinely removed"
                if verdict
                else "in-situ collapses once the source signal is removed - it is "
                "measuring which dataset a tile came from"
            ),
        )

    print("-" * 74)
    line(
        "      ",
        "However this lands, the in-situ number is not trustworthy on an immunostained "
        "slide. What it measures is agreement with BEETLE on Aperio crops; only "
        "in-domain tiles labelled by a pathologist answer the question this model is "
        "actually asked.",
    )
    return 0 if verdict is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
