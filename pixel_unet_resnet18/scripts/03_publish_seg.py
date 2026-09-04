"""Pin the trained segmentation model: checkpoint, manifest, SHA-256, and a reload check.

    python scripts/03_publish_seg.py                    # publish the default recipe's run
    python scripts/03_publish_seg.py --name my_model    # a different published name

Idempotent in the way that matters: publishing twice writes the same weights and the same
input contract, and the SHA-256 in the manifest is recomputed from the file each time. What
it will not do is publish a checkpoint whose recipe does not match a report on disk - a
model nobody has scored has no business being pinned.

**The name says the licence.** The default is `invasive_pixel_v1_<init>_nc`, and the `nc`
is load-bearing: this model is trained on BEETLE-labelled BRACS masks, so it inherits
CC BY-NC-SA and BRACS's non-commercial terms. Approach 1's `invasive_tile_v1_imagenet.pt`
is CC0/BSD-3 and still ships; these two must never be confused by a filename.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classes  # noqa: E402
import paths  # noqa: E402
import segdata  # noqa: E402
import segexport  # noqa: E402
import segnet  # noqa: E402
import segtrain  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", choices=("imagenet", "simclr"), default="imagenet")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tiles-per-epoch", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--all-sources", dest="source_authority", action="store_false",
        help="the run being published let every source teach every class. MUST match "
             "what 02_train_seg.py was run with - it is part of the recipe fingerprint, "
             "so a mismatch will simply fail to find the checkpoint.")
    parser.add_argument("--bcss-only", action="store_true",
                        help="record that this run used no BEETLE-labelled data, which "
                             "makes it CC0/BSD-3 and shippable. Only pass this if the "
                             "export was actually run with --bcss-only.")
    args = parser.parse_args()

    paths.ensure_dirs()
    recipe = segtrain.Recipe(
        epochs=args.epochs, batch_size=args.batch_size,
        tiles_per_epoch=args.tiles_per_epoch, init=args.init,
        source_authority=args.source_authority,
    )
    checkpoint = segtrain.checkpoint_path(recipe)
    if not checkpoint.is_file():
        raise SystemExit(
            f"no checkpoint at {checkpoint}. Run scripts/02_train_seg.py with the same "
            "recipe first - the filename carries the recipe's fingerprint, so a "
            "different recipe is a different run."
        )

    report_path = (
        paths.REPORTS_DIR / f"02_train_{recipe.init}_{recipe.fingerprint()}.json"
    )
    if not report_path.is_file():
        raise SystemExit(
            f"no report at {report_path}. A model nobody has scored has no business "
            "being pinned; finish scripts/02_train_seg.py."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["history"]["epochs_done"] < recipe.epochs:
        raise SystemExit(
            f"this checkpoint stopped at epoch {saved['history']['epochs_done']} of "
            f"{recipe.epochs}. Finish the run rather than pinning a partial one."
        )

    net = segnet.freeze_early_encoder(segnet.ResNet18UNet(recipe.init))
    net.load_state_dict(saved["model"])
    net.eval()

    rows = segexport.read_manifest()
    manifest_json = json.loads(
        (paths.TILES_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    sources = sorted({row["source"] for row in rows})
    non_commercial = classes.SOURCE_DCIS in sources and not args.bcss_only
    name = args.name or (
        f"invasive_pixel_v1_{recipe.init}" + ("_nc" if non_commercial else "")
    )

    checkpoint_path, manifest_path = segnet.publish(
        net,
        name=name,
        models_dir=paths.MODELS_DIR,
        tile_px=manifest_json["spec"]["tile_px"],
        mpp=manifest_json["spec"]["mpp"],
        standardise=recipe.standardise,
        invert=recipe.invert,
        metrics=report["held_out"],
        training_data={
            "tiles": report["tiles"],
            "patients": report["patients"],
            "sources": sources,
            "labels_drawn_by": manifest_json["labels_drawn_by"],
            "pixels_per_class": manifest_json["pixels_per_class"],
            # What the export annotated is not what the run learned from. Recording
            # both, and which source was allowed to teach which class, is the single
            # biggest decision behind this checkpoint - a reader comparing two of them
            # needs it beside the metrics rather than in a commit message.
            "pixels_per_class_supervising": segdata.pixel_census(
                rows, source_authority=recipe.source_authority
            ),
            "source_authority": (
                {source: sorted(allowed) for source, allowed in classes.SOURCE_CLASSES.items()}
                if recipe.source_authority
                else "every source taught every class"
            ),
            "source_authority_note": (
                "BRACS regions were selected for being DCIS-heavy, so they supply 98.8% "
                "of the in-situ pixels - and also 58.0 M non-epithelium and 16.0 M "
                "invasive pixels that are BEETLE's predictions rather than anybody's "
                "judgement, against 385 M and 267 M drawn by hand in BCSS. Those are "
                "blanked to IGNORE, so the tile still teaches in-situ and stops "
                "asserting the other two."
                if recipe.source_authority
                else "no source-authority rule was applied"
            ),
            "tile_manifest_fingerprint": segexport.fingerprint_rows(rows),
            "export_spec": manifest_json["spec"],
        },
        provenance={
            "pipeline": "pixel_unet_resnet18",
            "recipe": report["recipe"],
            "run_fingerprint": report["fingerprint"],
            "licence": (
                "CC BY-NC-SA 4.0 (BEETLE labels) + BRACS non-commercial. RESEARCH ONLY - "
                "this model must not be shipped. Approach 1's "
                "invasive_tile_v1_imagenet.pt is the permissive one."
                if non_commercial else
                "CC0 (BCSS) + BSD-3 (torchvision weights). Shippable."
            ),
            "shares_no_code_with": "bcss_bracs_hchannel_resnet18",
            "transform_parity": (
                "tests/test_parity.py asserts this pipeline's H-channel transform is "
                "bit-for-bit identical to approach 1's"
            ),
        },
    )
    print(f"  checkpoint {checkpoint_path}")
    print(f"  manifest   {manifest_path}")

    # --- the reload check ---------------------------------------------------
    # A fresh load from disk must reproduce the logits the in-memory network gives. This
    # is what catches a manifest whose `input` block does not describe what the model was
    # actually trained under - the failure that got past approach 1's gate G6 once,
    # invisibly, because the check rebuilt its input with a default instead of the
    # recorded setting.
    reloaded, reloaded_manifest = segnet.load_pinned(name, models_dir=paths.MODELS_DIR)
    settings = reloaded_manifest["input"]

    probe_rows = rows[:4]
    dataset = segdata.SegTileDataset(
        probe_rows, paths.TILES_DIR, augment_data=False,
        invert=settings["invert_polarity"],
        standardise=settings["standardise_tile_p99"],
    )
    images = torch.stack([dataset[i][0] for i in range(len(probe_rows))])

    with torch.inference_mode():
        before = net(images).numpy()
        after = reloaded(images).numpy()

    worst = float(np.max(np.abs(before - after)))
    print(f"  reload check: worst logit difference {worst:.3e}")
    if worst > 1e-5:
        raise SystemExit(
            "a fresh load does not reproduce the in-memory model's logits. Either the "
            "weights did not round-trip or the manifest's input block does not describe "
            "what this model was trained under. Refusing to call this published."
        )
    print("  G2 OK - reloads in a fresh graph and reproduces its own logits")

    if non_commercial:
        print("\n  RESEARCH ONLY. This model is trained on BEETLE-labelled BRACS masks")
        print("  and inherits CC BY-NC-SA plus BRACS's non-commercial terms. The `_nc`")
        print("  suffix is there so it cannot be mistaken for the shippable tile model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
