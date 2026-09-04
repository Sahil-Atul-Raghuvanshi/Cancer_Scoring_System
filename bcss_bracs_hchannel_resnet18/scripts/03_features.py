"""Gate G3: run the frozen body over every tile once, and cache the 512-vectors.

    python scripts/03_features.py --init imagenet
    python scripts/03_features.py --init simclr        # approach 3, same manifest
    python scripts/03_features.py --init imagenet --invert   # ablation A2

This is the step that makes the project iterable. The body never changes during
training, so after this a fit takes seconds - which is what turns the class weights,
the input polarity and the whole approach-3 comparison into experiments rather than
commitments.

Augmentation is off, deliberately: these must be the features step 8 will compute at
inference, which is the `augment=False` path. Nothing here may add a resize or a crop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import backend_path  # noqa: F401
import datasets
import models


def cache_path(init: str, *, invert: bool, standardise: bool = False) -> Path:
    """One file per (init, polarity, standardise). All three change the vectors.

    `standardise` is in the FILENAME, not only in the payload, so the two variants can
    coexist and be compared rather than one silently overwriting the other. A cache whose
    input transform is not identifiable from its name is a cache that gets mixed up.
    """
    suffix = f"{'_inverted' if invert else ''}{'_std' if standardise else ''}"
    return backend_path.FEATURES_DIR / f"{init}{suffix}.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=sorted(models.INITS), default="imagenet")
    parser.add_argument("--invert", action="store_true",
                        help="flip the input polarity - ablation A2, dark nuclei")
    parser.add_argument("--standardise", action="store_true",
                        help="divide each tile by its own p99 before the clip. Gate G0b "
                             "measured this taking coverage of the served IHC "
                             "distribution from 4/6 to 6/6 percentiles, so it is the "
                             "setting the shipped model should carry - but it must match "
                             "at inference, which is why it is recorded in the cache and "
                             "in the manifest rather than left as a habit.")
    parser.add_argument(
        "--all-sources", action="store_true",
        help="let every source contribute every class - the old behaviour, in which "
             "BRACS supplied 270 invasive and 662 non-epithelium tiles guessed by "
             "BEETLE inside DCIS-selected ROIs. Kept so the two can be compared.")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers; 0 on Windows, where each worker "
                             "re-imports the module tree and the forward pass is the "
                             "bottleneck anyway")
    args = parser.parse_args()

    backend_path.ensure_dirs()

    if args.init == "imagenet":
        vendored = models.vendor_imagenet_weights(backend_path.PRETRAINED_DIR)
        print(f"vendored {vendored.name}: {vendored.stat().st_size / 1e6:.1f} MB")
        print(f"  sha256 {models.sha256_of(vendored)}")

    net = models.resnet18_backbone(args.init, weights_dir=backend_path.PRETRAINED_DIR)
    body = models.feature_extractor(net)
    print(f"\ninit={args.init}  conv1 weight norm "
          f"{float(torch.linalg.norm(net.conv1.weight)):.4f}")
    print(f"polarity: {'nuclei dark (inverted)' if args.invert else 'nuclei bright'}")

    rows = datasets.read_manifest(backend_path.TILES_DIR / "tiles_manifest.csv")
    if not args.all_sources:
        before = len(rows)
        rows = datasets.by_source_authority(rows)
        dropped = before - len(rows)
        print(f"source authority: dropped {dropped:,} tiles a source is not "
              f"authoritative for ({datasets.SOURCE_CLASSES})")
    fingerprint = datasets.manifest_fingerprint(rows)
    print(f"{len(rows):,} tiles, manifest fingerprint {fingerprint[:16]}")

    print(f"standardise: {args.standardise}"
          + ("" if args.standardise else "   <- G0b wants True for IHC coverage"))

    dataset = datasets.TileDataset(rows, backend_path.TILES_DIR,
                                   augment=False, invert=args.invert,
                                   standardise=args.standardise)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers)

    features = np.zeros((len(dataset), models.FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(len(dataset), dtype=np.int64)

    cursor = 0
    started = time.monotonic()
    for batch_number, (batch, batch_labels) in enumerate(loader, start=1):
        with torch.inference_mode():
            vectors = body(batch).numpy()
        features[cursor:cursor + len(vectors)] = vectors
        labels[cursor:cursor + len(vectors)] = batch_labels.numpy()
        cursor += len(vectors)

        if batch_number % 20 == 0 or cursor == len(dataset):
            elapsed = time.monotonic() - started
            rate = cursor / max(elapsed, 1e-9)
            remaining = (len(dataset) - cursor) / max(rate, 1e-9)
            print(f"  {cursor:>7,}/{len(dataset):,} tiles "
                  f"({cursor / len(dataset):5.1%})  {rate:6.1f} tiles/s  "
                  f"eta {remaining / 60:5.1f} min", flush=True)

    assert cursor == len(dataset), f"extracted {cursor} of {len(dataset)}"

    destination = cache_path(args.init, invert=args.invert,
                             standardise=args.standardise)
    np.savez_compressed(
        destination,
        features=features, labels=labels,
        # The fingerprint is what stops a feature array being paired with a different
        # manifest. That failure trains to about chance and looks exactly like a model
        # that will not learn.
        fingerprint=np.array(fingerprint),
        init=np.array(args.init), invert=np.array(args.invert),
        standardise=np.array(args.standardise),
    )
    print(f"\nwrote {destination.name} ({destination.stat().st_size / 1e6:.1f} MB) "
          f"in {(time.monotonic() - started) / 60:.1f} min")

    # --- G3 ---------------------------------------------------------------------
    print("\n== G3: deterministic, aligned, carrying signal ==")

    subset = datasets.TileDataset(rows[:64], backend_path.TILES_DIR,
                                  standardise=args.standardise,
                                  augment=False, invert=args.invert)
    with torch.inference_mode():
        again = body(torch.stack([subset[i][0] for i in range(len(subset))])).numpy()
    assert np.allclose(again, features[:64], atol=1e-5), "extraction is not deterministic"
    print("  G3a OK - re-extraction reproduces the first 64 vectors")

    stored = np.load(destination, allow_pickle=False)
    assert str(stored["fingerprint"]) == fingerprint
    assert np.array_equal(stored["labels"], labels)
    print("  G3b OK - the cache carries the manifest fingerprint it was built from")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=0, stratify=labels
    )
    accuracy = LogisticRegression(
        max_iter=2000, class_weight="balanced"
    ).fit(x_train, y_train).score(x_test, y_test)
    print(f"  probe accuracy on a RANDOM split (optimistic by design): {accuracy:.3f}")
    assert accuracy > 0.70, (
        f"{accuracy:.3f} is too low even for a flattering split - the features are not "
        "carrying class information. Check the input transform and G2's tile panel."
    )
    print("  G3c OK - the features carry signal; the honest number is 04's\n")
    print("Next: python scripts/04_train.py --init " + args.init)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
