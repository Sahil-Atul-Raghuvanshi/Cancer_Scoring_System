"""Score two published heads on ONE held-out set. The controlled test of the label rule.

**Why nothing before this was a fair comparison.** Every number in
`RESULTS_224_VS_448.md` compares a head fitted under Fix 1 against the served v3 on
*different* held-out sets - and the Fix 1 sets deliberately contain the solid and comedo
DCIS that the old rule **deleted**. So a Fix 1 head scoring worse might mean the rule
hurt, or might mean it is being examined on harder material that v3 never had to face.
Those are opposite conclusions and the numbers cannot separate them.

This can, because two things happen to be true:

  * v3 and `invasive_tile_fov112_fix1_concat` are the **same architecture** at the
    **same geometry** - `concat_resnet18_mlp`, 224 px at 0.5 um/px - and their manifests
    declare the same input transform down to `standardise_tile_p99` and the OD clip. So
    both heads consume the identical 1024-vector, and the cached features serve both.
  * `datasets.institution_split` is **deterministic on patient and institution**, so the
    same patients are held out of both exports. v3 never trained on the tiles below.

That makes the only difference between the two heads the data they were fitted on, which
is exactly the label rule. Same test set, same features, same architecture: whatever
separates them is Fix 1.

    python scripts/09_head_to_head.py --tiles ../v1_data/data/tissue_type_model_training/h_channel/112um --features ../v1_data/data/tissue_type_model_training/h_channel/112um/features \\
        --heads invasive_tile_v3_concat_bach invasive_tile_fov112_fix1_concat
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backend_path  # noqa: F401,E402
import bcss  # noqa: E402
import datasets  # noqa: E402

HIDDEN = 256
NON_EPITHELIUM, IN_SITU, INVASIVE = 0, 1, 2


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def load_head(name: str, in_dim: int) -> torch.nn.Sequential:
    """One published checkpoint's MLP head, without its two frozen bodies.

    The bodies are 90 MB of weights this script has no use for - the features they would
    compute are already cached - so only the `head.*` tensors are lifted out. Their
    shapes are asserted rather than assumed: a head of a different width would load
    partially under `strict=False` and score like a randomly initialised one.
    """
    path = backend_path.MODELS_DIR / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"{path} is missing")
    state = torch.load(path, map_location="cpu", weights_only=True)
    head_state = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
    if not head_state:
        raise SystemExit(f"{name} carries no `head.*` tensors - is it a concat model?")

    head = torch.nn.Sequential(
        torch.nn.Linear(in_dim, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.0),          # eval only; dropout would add noise to a score
        torch.nn.Linear(HIDDEN, len(bcss.CLASS_NAMES)),
    )
    expected = head["0"].weight.shape if isinstance(head, dict) else None  # noqa: F841
    if tuple(head_state["0.weight"].shape) != (HIDDEN, in_dim):
        raise SystemExit(
            f"{name}'s head is {tuple(head_state['0.weight'].shape)} and these features "
            f"are {in_dim}-dimensional. Refusing to score a head on vectors it was not "
            "fitted for."
        )
    head.load_state_dict(head_state, strict=True)
    return head.eval()


def score(truth: np.ndarray, predicted: np.ndarray) -> dict:
    n = len(bcss.CLASS_NAMES)
    confusion = [[int(((truth == t) & (predicted == p)).sum()) for p in range(n)]
                 for t in range(n)]
    totals = [sum(r) for r in confusion]
    recalls = [confusion[i][i] / totals[i] if totals[i] else float("nan") for i in range(n)]
    tp = confusion[INVASIVE][INVASIVE]
    fp = confusion[IN_SITU][INVASIVE]
    fn = confusion[INVASIVE][IN_SITU]
    return {
        "insitu_called_invasive": confusion[IN_SITU][INVASIVE] / totals[IN_SITU],
        "insitu_errors": confusion[IN_SITU][INVASIVE],
        "insitu_total": totals[IN_SITU],
        "recall_insitu": recalls[IN_SITU],
        "recall_invasive": recalls[INVASIVE],
        "recall_stroma": recalls[NON_EPITHELIUM],
        "min_recall": min(recalls),
        "macro_recall": sum(recalls) / n,
        "dice_invasive_vs_non_invasive": 2 * tp / (2 * tp + fp + fn) if (2*tp+fp+fn) else float("nan"),
        "confusion": confusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = datasets.by_source_authority(
        datasets.read_manifest(args.tiles / "tiles_manifest.csv"))
    fingerprint = datasets.manifest_fingerprint(rows)

    halves = []
    for init in ("imagenet", "simclr"):
        cache = np.load(args.features / f"{init}_std.npz", allow_pickle=True)
        if not str(cache["fingerprint"]).startswith(fingerprint[:16]):
            raise SystemExit(f"{init}_std.npz was computed over different tiles")
        halves.append(cache["features"])
    features = np.concatenate(halves, axis=1)
    labels = np.array([int(r["label"]) for r in rows])

    _, test_rows = datasets.institution_split(rows)
    index = {id(r): i for i, r in enumerate(rows)}
    test_idx = np.array([index[id(r)] for r in test_rows])
    x = torch.from_numpy(np.ascontiguousarray(features[test_idx], dtype=np.float32))
    truth = labels[test_idx]

    print(f"{len(rows):,} tiles, fingerprint {fingerprint[:16]}")
    print(f"ONE held-out set: {len(test_idx):,} tiles, "
          f"{int((truth == IN_SITU).sum()):,} of them in-situ")
    print(f"held-out slides: {len({r['slide_id'] for r in test_rows})}\n")

    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    for name in args.heads:
        head = load_head(name, features.shape[1])
        with torch.no_grad():
            predicted = head(x).numpy().argmax(axis=1)
        predictions[name] = predicted
        results[name] = score(truth, predicted)

    width = max(len(n) for n in args.heads)
    print(f"{'head':<{width}}  insitu->inv          recall_insitu  recall_inv  dice")
    for name in args.heads:
        r = results[name]
        lo, hi = wilson(r["insitu_errors"], r["insitu_total"])
        print(f"{name:<{width}}  {r['insitu_errors']:>5}/{r['insitu_total']:<5} "
              f"{r['insitu_called_invasive']:6.1%} [{lo:.1%},{hi:.1%}]  "
              f"{r['recall_insitu']:>13.3f}  {r['recall_invasive']:>10.3f}  "
              f"{r['dice_invasive_vs_non_invasive']:.3f}")

    # McNemar on the in-situ rows: the two heads see the *same* tiles, so what matters is
    # the tiles they disagree on, not their marginal rates. A paired test is the correct
    # one here and an unpaired proportion test would overstate the uncertainty.
    if len(args.heads) == 2:
        a, b = args.heads
        insitu = truth == IN_SITU
        wrong_a = (predictions[a] == INVASIVE) & insitu
        wrong_b = (predictions[b] == INVASIVE) & insitu
        only_a = int((wrong_a & ~wrong_b).sum())
        only_b = int((wrong_b & ~wrong_a).sum())
        print(f"\nMcNemar on the {int(insitu.sum()):,} in-situ tiles (paired):")
        print(f"  {a} wrong alone: {only_a}")
        print(f"  {b} wrong alone: {only_b}")
        if only_a + only_b:
            chi = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b)
            p = math.erfc(math.sqrt(chi / 2))
            better = b if only_a > only_b else a
            print(f"  chi2 = {chi:.1f}, p = {p:.2g}  ->  "
                  f"{'no separation' if p > 0.05 else better + ' is better'}")

    out = args.out or (args.tiles / "reports" / "09_head_to_head.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tiles_dir": str(args.tiles),
        "manifest_fingerprint": fingerprint,
        "held_out_tiles": int(len(test_idx)),
        "note": "one held-out set, one feature space, one architecture - the heads "
                "differ only in the data they were fitted on, i.e. the label rule",
        "heads": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
