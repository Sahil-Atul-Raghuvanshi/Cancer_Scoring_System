"""Publish a familiarity reference beside every step 8 head - see `familiarity.py`.

    python scripts/build_familiarity_references.py              # all heads
    python scripts/build_familiarity_references.py --only invasive_tile_fov224_he_concat

For each published checkpoint this reads the two frozen-body feature caches it was
fitted on (`03_features.py`'s, the same vectors `08_fit_concat_mlp.py` concatenated),
checks they carry the fingerprint the checkpoint's manifest records, fits one
Ledoit-Wolf Gaussian per class on the training rows, reads the cut off the held-out
institutions, and writes `<name>.familiarity.npz` next to the `.pt`. A head with fewer
held-out tiles than `tissue_type_familiarity_min_held_out` gets no reference at all -
see that setting for why - and is served with the flat test alone.

**Two checks before anything is written, and a reference that fails either is not
published.** A check that cannot fail proves nothing, so both are against things that
exist on disk rather than against the object just built:

  served   the served network, run on three stored training tiles, must reproduce the
           cached features. If it does not, the reference describes vectors the head
           never sees and the distances it produces are meaningless.
  fill     a 224 px window of the measured scanner fill, RGB (181, 180, 186), pushed
           through the head's own input transform, must be refused - by the flat test
           and, where it can, the distance. That is P-05, and a gate that would let it
           through is not one.

Needs the training repository and its caches (`tissue_type_model_training`, under the
active data version) and scikit-learn. Serving needs neither - only the `.npz`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
WORKSPACE = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(WORKSPACE / "tissue_type_model_training" / "src"))

import datasets  # noqa: E402  - the training repo's, so the split is the one it used
from PIL import Image  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.pipeline.step08_tissue_type_segmentation import familiarity  # noqa: E402
from app.pipeline.step08_tissue_type_segmentation import input as model_input  # noqa: E402
from app.pipeline.step08_tissue_type_segmentation import model  # noqa: E402

TRAINING = WORKSPACE / "tissue_type_model_training"
sys.path.insert(0, str(WORKSPACE))
import data_versions  # noqa: E402

#: The fill P-05 was measured on - one constant, CAN_00259 and CAN_00865.
SCANNER_FILL = (181, 180, 186)


def _tiles_dir(pinned: model.Pinned) -> Path:
    branch = "he" if pinned.channel == model_input.CHANNEL_RGB_HE else "h_channel"
    fov = int(round(pinned.tile_px * pinned.mpp))
    return data_versions.data_root() / "tissue_type_model_training" / branch / f"{fov}um"


def _features(tiles_dir: Path, pinned: model.Pinned) -> tuple[np.ndarray, np.ndarray, str]:
    suffix = "" if pinned.channel == model_input.CHANNEL_RGB_HE else "_std"
    halves, labels, prints = [], None, set()
    # The body order the concat head reads: `body_a` is ImageNet, `body_b` SimCLR.
    for init in ("imagenet", "simclr"):
        with np.load(tiles_dir / "features" / f"{init}{suffix}.npz") as stored:
            halves.append(stored["features"].astype(np.float32))
            labels = stored["labels"]
            prints.add(str(stored["fingerprint"]))
    if len(prints) != 1:
        raise SystemExit(f"{tiles_dir}: the two caches describe different tiles {prints}")
    return np.concatenate(halves, axis=1), labels, prints.pop()


def _embed(pinned: model.Pinned, batch: np.ndarray) -> np.ndarray:
    import torch

    captured = {}
    hook = model.penultimate(pinned.net).register_forward_pre_hook(
        lambda _m, inputs: captured.__setitem__("f", inputs[0])
    )
    try:
        with torch.inference_mode():
            pinned.net(torch.from_numpy(batch.astype(np.float32)))
    finally:
        hook.remove()
    return captured["f"].numpy()


def _model_input(pinned: model.Pinned, pixels: np.ndarray) -> np.ndarray:
    if pinned.channel == model_input.CHANNEL_RGB_HE:
        return model_input.rgb_to_model_input(pixels)
    return model_input.to_model_input(
        pixels,
        gamma=pinned.gamma,
        invert=pinned.invert_polarity,
        standardise=pinned.standardise,
    )


def build(name: str, *, quantile: float, write: bool) -> bool:
    pinned = model.load_pinned(name)
    tiles_dir = _tiles_dir(pinned)
    features, labels, fingerprint = _features(tiles_dir, pinned)

    expected = (pinned.manifest.get("training_data") or {}).get("manifest_fingerprint")
    if fingerprint != expected:
        print(f"  REFUSED: caches are export {fingerprint[:12]}, {name} was fitted on "
              f"{str(expected)[:12]}")
        return False

    rows = datasets.by_source_authority(datasets.read_manifest(tiles_dir / "tiles_manifest.csv"))
    if datasets.manifest_fingerprint(rows) != fingerprint or len(rows) != len(features):
        print("  REFUSED: the manifest rows do not line up with the cache rows")
        return False
    _, test_rows = datasets.institution_split(rows)
    test_ids = {row["tile_id"] for row in test_rows}
    held_out = np.array([row["tile_id"] in test_ids for row in rows])

    # Check 1 - served features are the cached features.
    picks = sorted({0, len(rows) // 2, len(rows) - 1})
    stored = []
    for index in picks:
        image = np.asarray(Image.open(tiles_dir / rows[index]["tile_path"]))
        if pinned.channel == model_input.CHANNEL_RGB_HE:
            stored.append(model_input.rgb_to_model_input(image[..., :3]))
        else:
            plane = image if image.ndim == 2 else image[..., 0]
            stored.append(model_input.from_stored(
                plane, gamma=pinned.gamma, invert=pinned.invert_polarity,
                standardise=pinned.standardise,
            ))
    drift = float(np.abs(_embed(pinned, np.stack(stored)) - features[picks]).max())
    if drift > 1e-3:
        print(f"  REFUSED: served features differ from the cache by {drift:.4g}")
        return False

    reference, distances = familiarity.fit(
        features, labels, held_out, quantile=quantile, fingerprint=fingerprint
    )
    path = model.familiarity_path(model.find(name).checkpoint)
    if reference.held_out < settings.tissue_type_familiarity_min_held_out:
        # Too few unseen tiles to know where genuine tissue ends: the cut would be read
        # off the tail of a handful, and it refused real tumour when it was tried.
        print(
            f"  NO DISTANCE GATE: {reference.held_out} held-out tiles, under the "
            f"{settings.tissue_type_familiarity_min_held_out} a cut may be read from. "
            "This head keeps the flat test only."
        )
        if write and path.exists():
            path.unlink()
            print(f"  removed {path.name}")
        return True

    # Check 2 - the measured fill is refused.
    size = pinned.tile_px
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[:] = SCANNER_FILL
    pixels = (
        rgb
        if pinned.channel == model_input.CHANNEL_RGB_HE
        else model_input.haematoxylin_od(rgb.astype(np.float32), (230.0, 230.0, 230.0))
    )
    flat = familiarity.flat_share(pixels) >= settings.tissue_type_max_flat_share
    fill_distance = float(reference.distance(_embed(pinned, _model_input(pinned, pixels)[None]))[0])
    unfamiliar = fill_distance > reference.threshold
    if not flat:
        print("  REFUSED: the scanner fill passes the flat test")
        return False

    print(
        f"  {reference.train:,} training / {reference.held_out:,} held-out tiles, "
        f"served drift {drift:.1e}\n"
        f"  held-out distance p50 {np.median(distances):,.0f}  p99 {np.quantile(distances, 0.99):,.0f}"
        f"  max {distances.max():,.0f}  -> cut {reference.threshold:,.0f} (q={quantile})\n"
        f"  scanner fill: flat yes, distance {fill_distance:,.0f} "
        f"({'refused' if unfamiliar else 'within the cut - the flat test is what refuses it here'})"
    )
    if write:
        familiarity.save(reference, path)
        # Round trip: what is served is what was read back, not what was built.
        again = familiarity.load(path)
        assert np.array_equal(again.means, reference.means)
        assert again.threshold == reference.threshold
        print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", action="append", help="one head by name; repeatable")
    parser.add_argument("--quantile", type=float, default=settings.tissue_type_familiarity_quantile)
    parser.add_argument("--dry-run", action="store_true", help="measure, write nothing")
    args = parser.parse_args()

    names = args.only or [
        candidate.name
        for candidate in model.discover()
        if candidate.usable and candidate.arch == model.CONCAT_ARCH
    ]
    failed = []
    for name in names:
        print(name)
        if not build(name, quantile=args.quantile, write=not args.dry_run):
            failed.append(name)
    if failed:
        print(f"\nnot published: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
