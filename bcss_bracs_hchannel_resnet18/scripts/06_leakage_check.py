"""Can the frozen features tell you which *dataset* a tile came from?

    python scripts/06_leakage_check.py --init imagenet --standardise

**Why this has to be run, and run alongside the source-authority rule rather than
instead of it.** Class 1 is 1,843 tiles from BRACS against **9** from BCSS. So `source`
is very nearly a synonym for "class 1", and a head fitted on these features can score
well on in-situ epithelium by recognising *which dataset it is looking at* - the
scanner, the stain era, the crop's edges - and never learn duct architecture at all.
That model would pass GroupKFold, pass the institution split, and then collapse on a
slide from neither dataset, which is exactly the slide it is built to score.

`approach-1-plus-bracs.md` names this as rule 3 and asks for precisely this test. It is
also the reason `datasets.SOURCE_CLASSES` carries a warning: making BRACS the only
source of class 1 is the right call on label quality *and* makes this confound worse, so
the two have to be reported together.

**What it does.** Fits a throwaway logistic probe on the same frozen features to predict
`source`, and reports how well it does against the base rate. Then the number that
actually matters: the same probe restricted to a **single class**, where source and
label are not confounded by construction, and a class-1 probe fitted on features with
the source direction projected out - if in-situ recall survives that, the model is using
something other than the dataset's fingerprint.

**How to read it.** High source accuracy on its own is not proof of cheating - two
datasets *do* look different, and a model that sees tissue will also see that. What
would be damning is class-1 recall collapsing once the source direction is removed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backend_path  # noqa: E402,F401 - imported for the sys.path side effect
import datasets  # noqa: E402

TICK = "  ok  "
WARN = " warn "
CROSS = " FAIL "


def line(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def _probe(features: np.ndarray, target: np.ndarray, *, seed: int = 0) -> float:
    """Balanced accuracy of a logistic probe, held out on a random half.

    A random split rather than a grouped one, deliberately: this is asking "is the
    information *present* in the features", which is the most generous possible reading
    and therefore the right one for a test whose passing condition is "it is not".
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


def _source_direction(features: np.ndarray, source: np.ndarray) -> np.ndarray:
    """The single direction that best separates the two datasets, unit length."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(features, source)
    weight = model.coef_[0]
    return weight / max(float(np.linalg.norm(weight)), 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default="imagenet")
    parser.add_argument("--standardise", action="store_true")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--all-sources", action="store_true",
                        help="check the unfiltered manifest instead")
    arguments = parser.parse_args()

    print("=" * 74)
    print("Dataset-as-label leakage check")
    print("=" * 74)

    rows = datasets.read_manifest(backend_path.TILES_DIR / "tiles_manifest.csv")
    if not arguments.all_sources:
        rows = datasets.by_source_authority(rows)

    suffix = ("_inv" if arguments.invert else "") + ("_std" if arguments.standardise else "")
    path = backend_path.FEATURES_DIR / f"{arguments.init}{suffix}.npz"
    if not path.exists():
        line(CROSS, f"{path.name} is missing - run scripts/03_features.py first")
        return 1

    cache = np.load(path, allow_pickle=False)
    if str(cache["fingerprint"]) != datasets.manifest_fingerprint(rows):
        line(CROSS, f"{path.name} was built from a different tile set - re-run 03")
        return 1

    features = cache["features"]
    labels = cache["labels"]
    source = np.array([row["source"] for row in rows])
    is_borrowed = (source == "bracs_dcis").astype(int)

    line(TICK, f"{len(rows):,} tiles, {features.shape[1]} features")
    for name in sorted(set(source)):
        held = source == name
        counts = {int(c): int((labels[held] == c).sum()) for c in sorted(set(labels))}
        line("      ", f"{name:12s} {held.sum():6,} tiles  per class {counts}")

    # --- 1. is the dataset identifiable at all? -------------------------
    overall = _probe(features, is_borrowed)
    line("      ", "")
    line(
        WARN if overall > 0.9 else TICK,
        f"source is predictable from the features at {overall:.3f} balanced accuracy "
        "(0.5 would be chance)",
    )

    # --- 2. the confound, quantified ------------------------------------
    class_one = labels == 1
    borrowed_share = float(is_borrowed[class_one].mean()) if class_one.any() else float("nan")
    line(
        WARN if borrowed_share > 0.95 else TICK,
        f"class 1 is {borrowed_share:.1%} borrowed - so 'is this BRACS' and 'is this "
        "in-situ' are nearly the same question",
    )

    # --- 3. within one class, where source and label are not confounded --
    print("-" * 74)
    for target_class in sorted(set(labels)):
        held = labels == target_class
        if len(set(is_borrowed[held])) < 2:
            line(
                "      ",
                f"class {target_class}: one source only ({sorted(set(source[held]))}), "
                "nothing to separate",
            )
            continue
        within = _probe(features[held], is_borrowed[held])
        line("      ", f"class {target_class}: source still predictable at {within:.3f}")

    # --- 4. does class 1 survive losing the source direction? -----------
    print("-" * 74)
    direction = _source_direction(features, is_borrowed)
    projected = features - np.outer(features @ direction, direction)

    before = _probe(features, (labels == 1).astype(int))
    after = _probe(projected, (labels == 1).astype(int))
    residual = _probe(projected, is_borrowed)

    line("      ", f"in-situ separability, features as they are : {before:.3f}")
    line("      ", f"in-situ separability, source direction gone: {after:.3f}")
    line("      ", f"source separability after removal          : {residual:.3f}")

    # **The removal has to have worked before its result means anything.** Projecting
    # out one direction only helps if that direction *was* the source signal; if source
    # is still readable afterwards, then "in-situ survived" says nothing - the feature
    # the head could be cheating with is still sitting there. Measured here: source
    # separability barely moves (0.904 -> 0.883), so the dataset fingerprint is spread
    # across many directions rather than concentrated in one, and this test cannot
    # clear the model. Reporting that as a pass would be worse than not running it.
    removed = residual <= 0.65
    survived = after >= before - 0.05

    if not removed:
        line(
            WARN,
            f"INCONCLUSIVE - the projection did not actually remove the source signal "
            f"({overall:.3f} -> {residual:.3f}, chance is 0.5), so 'in-situ survived it' "
            "proves nothing. The dataset fingerprint is spread across many directions, "
            "not one.",
        )
        verdict = None
    else:
        verdict = survived
        line(
            TICK if survived else CROSS,
            (
                "in-situ survives with the source signal genuinely removed, so it is "
                "not only the dataset fingerprint"
                if survived
                else "in-situ collapses once the source signal is removed - the class-1 "
                "number is measuring which dataset a tile came from"
            ),
        )

    print("-" * 74)
    line(
        "      ",
        "Neither outcome makes the in-situ number trustworthy on an IHC slide. What it "
        "measures either way is agreement with BEETLE on Aperio crops; the only thing "
        "that settles the real question is in-domain tiles clicked by a pathologist.",
    )
    # Inconclusive is not a pass, but it is not a failure to report either.
    return 0 if verdict is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
