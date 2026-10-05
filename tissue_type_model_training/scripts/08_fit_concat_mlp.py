"""Fit the concat+MLP head - the served architecture - on one export's two caches.

**Why this script exists.** The published `invasive_tile_v3_concat_bach` is two frozen
ResNet18 bodies concatenated into an MLP, and `reports/best_head_concat_mlp_bach.json`
records its recipe exactly. But the code that fitted it is not in this repository any
more, and `04_train.py` can only fit a `Linear(512,3)`. So comparing a new field of view
against the served model was impossible: the only heads obtainable were a weaker family,
and a weaker family scoring lower says nothing about the geometry.

Everything here is read off that recorded recipe rather than chosen:

    architecture   Linear(1024,256) -> ReLU -> Dropout(0.2) -> Linear(256,3)
    features       imagenet_std (512) ++ simclr_std (512), both frozen
    class weights  balanced, `datasets.class_weights`
    decision rule  invasive when p(invasive) >= tau, else the better of the other two
    tau            0.40

**Both caches must describe the same tiles**, and that is checked rather than assumed:
`03_features.py` stamps each cache with the manifest fingerprint it was computed over, so
a cache from a different export - or from the same export before a re-cut - is refused
instead of being concatenated into a 1024-vector whose two halves describe different
pixels. That failure would produce a plausible model and a meaningless number.

    python scripts/08_fit_concat_mlp.py --tiles ../v1_data/data/tissue_type_model_training/h_channel/224um --features ../v1_data/data/tissue_type_model_training/h_channel/224um/features
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backend_path  # noqa: F401,E402
import bcss  # noqa: E402
import datasets  # noqa: E402
import train as train_lib  # noqa: E402

#: The head, exactly as `models.ConcatResNet18MLP` builds it. Restated here rather than
#: imported because that class also constructs two ResNet18 bodies, which this script has
#: no use for - it fits on cached vectors. `models.CONCAT_ARCH` is the string a published
#: checkpoint carries, and `test_concat_head_matches_published_arch` compares the two.
HIDDEN = 256
DROPOUT = 0.2

#: From the recorded recipe. Not a tuned value here - it is the served rule, and a head
#: selected under a different one would not be the same model.
TAU = 0.40


def build_head(in_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Dropout(DROPOUT),
        torch.nn.Linear(HIDDEN, len(bcss.CLASS_NAMES)),
    )


def load_pair(features_dir: Path, fingerprint: str) -> tuple[np.ndarray, dict]:
    """The two caches, concatenated, with the fingerprint checked on both.

    Returns the 1024-vectors and what the caches say about themselves, because the
    published manifest has to state the transform the head was *actually* fitted
    under rather than the one this script used to assume.

    Two suffixes are looked for, in order. `_std.npz` is the per-tile-standardised
    cache every haematoxylin arm was fitted on. Plain `.npz` is what an `rgb_he`
    store produces, because standardising a photograph by its own p99 density is not
    a thing that can be done - the RGB input transform has no parameters at all. The
    two halves must agree on which they are: concatenating a standardised half with
    an unstandardised one would build vectors whose two halves describe the same
    pixels under different transforms, and nothing downstream would notice.
    """
    halves = []
    suffixes: set[str] = set()
    standardised: set[bool] = set()
    channels: set[str] = set()

    for init in ("imagenet", "simclr"):
        for suffix in ("_std", ""):
            path = features_dir / f"{init}{suffix}.npz"
            if path.exists():
                break
        else:
            raise SystemExit(
                f"neither {init}_std.npz nor {init}.npz is in {features_dir}. Run:\n"
                f"  python scripts/03_features.py --init {init} "
                f"--tiles <export> --features {features_dir}\n"
                "(adding --standardise on a haematoxylin store, and not on an rgb_he "
                "one, which has no such transform)"
            )

        cache = np.load(path, allow_pickle=True)
        suffixes.add(suffix)
        standardised.add(bool(cache["standardise"]))
        channels.add(str(cache["channel"]) if "channel" in cache.files else "haematoxylin")
        got = str(cache["fingerprint"])
        if not got.startswith(fingerprint[:16]):
            raise SystemExit(
                f"{path.name} was computed over a different set of tiles: its "
                f"fingerprint is {got[:16]} and this manifest's is {fingerprint[:16]}. "
                "Concatenating them would build 1024-vectors whose two halves describe "
                "different pixels. Recompute the cache against this export."
            )
        halves.append(cache["features"])

    if len(suffixes) != 1 or len(standardised) != 1 or len(channels) != 1:
        raise SystemExit(
            f"the two caches in {features_dir} disagree about what they are: "
            f"suffixes {sorted(suffixes)}, standardise {sorted(standardised)}, "
            f"channel {sorted(channels)}. Both halves of a concatenated vector have "
            "to have been computed under one transform. Recompute them together."
        )

    meta = {
        "suffix": suffixes.pop(),
        "standardise": standardised.pop(),
        "channel": channels.pop(),
    }
    print(f"features: {meta['channel']}, standardise={meta['standardise']} "
          f"(cache suffix {meta['suffix'] or 'none'!r})")
    return np.concatenate(halves, axis=1), meta


def decide(probabilities: np.ndarray, tau: float | None) -> np.ndarray:
    """The served decision rule. `tau=None` is a plain argmax.

    A three-way argmax lets a badly-conditioned in-situ head outvote invasive, which is
    the failure the served checkpoint's manifest documents at length. Both are reported
    below, because the two answer different questions: argmax says how well the classes
    separate, tau says what the pipeline would actually do.
    """
    if tau is None:
        return probabilities.argmax(axis=1)
    scored = probabilities[:, bcss.INVASIVE]
    others = [i for i in range(probabilities.shape[1]) if i != bcss.INVASIVE]
    fallback = np.array(others)[probabilities[:, others].argmax(axis=1)]
    return np.where(scored >= tau, bcss.INVASIVE, fallback)


def score(truth: np.ndarray, predicted: np.ndarray) -> dict:
    n = len(bcss.CLASS_NAMES)
    confusion = [[int(((truth == t) & (predicted == p)).sum()) for p in range(n)]
                 for t in range(n)]
    totals = [sum(row) for row in confusion]
    recalls = [confusion[i][i] / totals[i] if totals[i] else float("nan")
               for i in range(n)]
    # Dice between the two carcinoma classes, which is the number this project exists
    # for: how cleanly invasive is separated from in-situ, ignoring stroma entirely.
    tp = confusion[bcss.INVASIVE][bcss.INVASIVE]
    fp = confusion[bcss.NON_INVASIVE][bcss.INVASIVE]
    fn = confusion[bcss.INVASIVE][bcss.NON_INVASIVE]
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {
        "tiles": int(len(truth)),
        "recall_stroma": recalls[bcss.NON_EPITHELIUM],
        "recall_insitu": recalls[bcss.NON_INVASIVE],
        "recall_invasive": recalls[bcss.INVASIVE],
        "min_recall": min(recalls),
        "macro_recall": sum(recalls) / n,
        "accuracy": float((truth == predicted).mean()),
        # The headline: the share of held-out in-situ tiles called invasive. This is the
        # failure the whole exercise is about, so it is named rather than left to be
        # read out of a confusion matrix.
        "dcis_called_invasive": (
            confusion[bcss.NON_INVASIVE][bcss.INVASIVE] / totals[bcss.NON_INVASIVE]
            if totals[bcss.NON_INVASIVE] else float("nan")
        ),
        "dice_invasive_vs_non_invasive": dice,
        "confusion": confusion,
        "confusion_axes": {"rows": "truth", "cols": "prediction",
                           "order": list(bcss.CLASS_NAMES)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=train_lib.EPOCHS)
    parser.add_argument("--lr", type=float, default=train_lib.LR)
    parser.add_argument("--weight-decay", type=float, default=train_lib.WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=train_lib.SEED)
    parser.add_argument("--out", type=Path, default=None,
                        help="where the report goes. Defaults to <tiles>/reports/.")
    parser.add_argument(
        "--publish", metavar="NAME", default=None,
        help="also write a servable checkpoint to models/tissue_type/ under this name. "
             "**This puts the head into service**: step 7 offers a field of view as "
             "soon as a checkpoint whose `tile_px * mpp` matches it is discoverable, "
             "and step 8 then runs it - so publishing is a deployment, not a save.")
    args = parser.parse_args()

    rows = datasets.read_manifest(args.tiles / "tiles_manifest.csv")
    before = len(rows)
    rows = datasets.by_source_authority(rows)
    print(f"source authority: dropped {before - len(rows):,} tiles")
    fingerprint = datasets.manifest_fingerprint(rows)
    print(f"{len(rows):,} tiles, fingerprint {fingerprint[:16]}")

    features, cache_meta = load_pair(args.features, fingerprint)

    # The store and its feature caches must be the same channel. They are written by
    # two different scripts an hour apart, so this is a real way to get it wrong.
    store_channel = datasets.channel_of(args.tiles)
    if store_channel != cache_meta["channel"]:
        raise SystemExit(
            f"{args.tiles} stores {store_channel!r} tiles but the caches in "
            f"{args.features} were computed over {cache_meta['channel']!r} ones. "
            "Recompute the features against this store."
        )
    labels = np.array([int(r["label"]) for r in rows])
    print(f"concatenated features: {features.shape}")

    train_rows, test_rows = datasets.institution_split(rows)
    index = {id(r): i for i, r in enumerate(rows)}
    train_idx = np.array([index[id(r)] for r in train_rows])
    test_idx = np.array([index[id(r)] for r in test_rows])
    print(f"train {len(train_idx):,} / held out {len(test_idx):,}")

    weights = datasets.class_weights(train_rows)
    print(f"class weights: {[round(float(w), 3) for w in weights]}")

    torch.manual_seed(args.seed)
    head = build_head(features.shape[1])
    optimiser = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    x = torch.from_numpy(np.ascontiguousarray(features[train_idx], dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(labels[train_idx]))

    head.train()
    for epoch in range(args.epochs):
        optimiser.zero_grad()
        loss = loss_fn(head(x), y)
        loss.backward()
        optimiser.step()
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"  epoch {epoch + 1}/{args.epochs}  loss {float(loss):.4f}")

    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(
            np.ascontiguousarray(features[test_idx], dtype=np.float32)))
        probabilities = torch.softmax(logits, dim=1).numpy()

    truth = labels[test_idx]
    report = {
        "architecture": f"Linear({features.shape[1]},{HIDDEN}) -> ReLU -> "
                        f"Dropout({DROPOUT}) -> Linear({HIDDEN},{len(bcss.CLASS_NAMES)})",
        "features": "imagenet_std (512) ++ simclr_std (512), both frozen",
        "tiles_dir": str(args.tiles),
        "manifest_fingerprint": fingerprint,
        "channel": cache_meta["channel"],
        "standardise": cache_meta["standardise"],
        "feature_cache_suffix": cache_meta["suffix"],
        "recipe": {"epochs": args.epochs, "lr": args.lr,
                   "weight_decay": args.weight_decay, "seed": args.seed,
                   "class_weights": "balanced (datasets.class_weights)"},
        "train_tiles": int(len(train_idx)),
        "held_out_argmax": score(truth, decide(probabilities, None)),
        "held_out_tau": {**score(truth, decide(probabilities, TAU)), "tau": TAU},
        "licence": "research only - BRACS non-commercial, BEETLE CC BY-NC-SA, BACH ND",
    }

    out_dir = args.out or (args.tiles / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "08_concat_mlp.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    torch.save(head.state_dict(), out_dir / "08_concat_mlp.pt")

    for name, block in (("argmax", report["held_out_argmax"]),
                        (f"tau={TAU}", report["held_out_tau"])):
        print(f"\n== held out, {name} ==")
        for key in ("dcis_called_invasive", "recall_insitu", "recall_invasive",
                    "recall_stroma", "min_recall", "macro_recall",
                    "dice_invasive_vs_non_invasive"):
            print(f"  {key:32} {block[key]:.4f}")
    print(f"\nwrote {out_dir / '08_concat_mlp.json'}")

    if args.publish:
        publish(args, rows, test_rows, index, features, head, report, cache_meta)
    return 0


def publish(args, rows, test_rows, index, features, head, report,
            cache_meta) -> None:
    """Assemble the servable checkpoint: two real bodies, the fitted head, a manifest.

    **The head here was fitted on cached features, and the served net computes its own.**
    Those are two different code paths over the same weights, and if they disagree the
    published model is not the model these metrics describe. Nothing downstream would
    say so: it would load, run, and score a few points worse for reasons nobody could
    find. So the two are compared directly on a handful of tiles before anything is
    written, which is a stronger check than G6 - G6 replays the published net against
    itself and cannot see a body wired in the wrong order.

    Body order is the specific hazard. `ConcatResNet18MLP.forward` concatenates
    `[body_a, body_b]` and `build` puts imagenet in `body_a`; `load_pair` above stacks
    imagenet then simclr. Those two facts have to agree, and the check below is what
    holds them together.
    """
    import models

    summary = json.loads((args.tiles / "export_summary.json").read_text(encoding="utf-8"))
    spec = summary["spec"]

    net = models.ConcatResNet18MLP.build(weights_dir=backend_path.PRETRAINED_DIR)
    net.head.load_state_dict(head.state_dict())
    net.eval()

    # A stratified handful, so the probe covers all three classes. Twenty tiles of
    # stroma would reproduce perfectly and prove almost nothing.
    rng = np.random.default_rng(0)
    picked: list[dict] = []
    for cls in range(len(bcss.CLASS_NAMES)):
        pool = [row for row in test_rows if int(row["label"]) == cls]
        if not pool:
            continue
        for i in rng.choice(len(pool), size=min(7, len(pool)), replace=False):
            picked.append(pool[int(i)])

    # The probe replays stored tiles through the *published* transform. Hard-coding
    # `standardise=True` here was safe only while every store was haematoxylin: on an
    # rgb_he store it would assert a transform that never ran and G6 would compare the
    # checkpoint against something else. Read it from the cache instead.
    probe_set = datasets.TileDataset(picked, args.tiles, augment=False,
                                     invert=False,
                                     standardise=cache_meta["standardise"],
                                     channel=cache_meta["channel"])
    with torch.inference_mode():
        batch = torch.stack([probe_set[i][0] for i in range(len(probe_set))])
        served = net(batch).numpy()

    # The same tiles the other way round: this run's head over this run's cached
    # features. Agreement proves the bodies, their order and the input transform.
    with torch.no_grad():
        rows_idx = [index[id(row)] for row in picked]
        fitted = head(torch.from_numpy(
            np.ascontiguousarray(features[rows_idx], dtype=np.float32))).numpy()

    worst = float(np.max(np.abs(served - fitted)))
    print(f"\nserved-vs-fitted agreement on {len(picked)} probe tiles: "
          f"worst |difference| {worst:.2e}")
    if worst > 1e-2:
        raise SystemExit(
            f"the checkpoint about to be published disagrees with the head that was "
            f"measured: worst logit difference {worst:.3e} over {len(picked)} tiles. "
            "The likely cause is the two bodies being concatenated in a different "
            "order from the feature caches, or a different input transform. Refusing "
            "to publish a model these metrics do not describe."
        )

    checkpoint, manifest_path = models.publish(
        net,
        name=args.publish,
        models_dir=backend_path.publish_dir(),
        init="imagenet1k_v1 + simclr_ciga_tenpercent",
        invert_polarity=False,
        standardise=cache_meta["standardise"],
        channel=cache_meta["channel"],
        tile_px=int(spec["tile_px"]),
        mpp=float(spec["mpp"]),
        metrics={
            "held_out": report["held_out_argmax"],
            "held_out_tau": report["held_out_tau"],
            "decision_rule": {
                "rule": "invasive when p(invasive) >= tau, else argmax over the other two",
                "tau": TAU,
                "why": "carried from the served v3's recipe, so the two are comparable. "
                       "Both argmax and tau numbers are recorded above because they "
                       "answer different questions.",
            },
            "sample_size_warning": (
                f"the held-out in-situ row is "
                f"{sum(report['held_out_argmax']['confusion'][bcss.NON_INVASIVE])} "
                "tiles. Read `dcis_called_invasive` with that in mind - see "
                "RESULTS_224_VS_448.md for the intervals."
            ),
        },
        training_data={
            "tiles_dir": str(args.tiles),
            "field_of_view_um": float(spec["tile_um"]),
            "tiles": summary.get("tiles_kept"),
            "per_class": summary.get("per_class"),
            "by_group": {k: v.get("tiles") for k, v in (summary.get("by_group") or {}).items()},
            "label_rule": (
                "Fix 1 - both of BEETLE's epithelium classes written as one code, "
                "chosen by the three-pathologist ROI consensus. A region the teacher "
                "misreads is relabelled, not deleted."
            ),
            "manifest_fingerprint": report["manifest_fingerprint"],
            "train_tiles": report["train_tiles"],
        },
        provenance={
            "fitted_by": "scripts/08_fit_concat_mlp.py",
            "recipe_from": "reports/best_head_concat_mlp_bach.json",
            "served_vs_fitted_worst_logit_delta": worst,
            "licence": "RESEARCH ONLY - BRACS non-commercial, BEETLE CC BY-NC-SA, "
                       "BACH CC BY-NC-ND. Do not ship.",
        },
        probe_tiles=[
            {"tile_id": row["tile_id"], "label": int(row["label"]),
             "logits": [round(float(v), 6) for v in served[i]]}
            for i, row in enumerate(picked)
        ],
    )
    print(f"published {checkpoint.name} and {manifest_path.name}")
    print(f"  field of view {spec['tile_um']:.0f} um "
          f"({spec['tile_px']} px at {spec['mpp']} um/px)")
    print("  step 7 will now offer this field of view and step 8 will run it.")


if __name__ == "__main__":
    raise SystemExit(main())
