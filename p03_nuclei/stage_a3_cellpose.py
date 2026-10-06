"""Approach 3b: fine-tune a nucleus model on hand-labelled IHC nuclei - a CPU pilot.

Runs in the `cellpose` venv. The review names InstanSeg, StarDist or CellViT; Cellpose is
used because it has a documented training API that runs on CPU, which is what this
machine has. The review rates the full approach at 2-6 weeks with 200-500 of our own
labelled nuclei; this is one night on public labels only, so it answers "does training on
IHC nuclei move the count", not "here is the production model".

  cellpose_nuclei     the published `nuclei` model, zero-shot - the baseline the fine-tune
                      has to beat, on exactly the same input
  cellpose_finetuned  that model trained further on LyNSeC's IHC train split

Brightfield is fed as inverted grey (nuclei bright), which is how a nuclear-channel model
reads it. Training is time-capped. A finished model is recorded and never retrained; a
run killed mid-training starts its training again (the zero-shot results already written
are kept).
"""

from __future__ import annotations

import sys
import time

import numpy as np
from PIL import Image

import common

LOG = common.LOGS / "a3_cellpose.log"
TRAIN_HOURS = 3.5
DIAMETER_PX = 14.0  # ~7 um nucleus at 0.5 um/px; replaced by the trained model's own


def _grey(rgb: np.ndarray) -> np.ndarray:
    return (255 - np.asarray(Image.fromarray(rgb).convert("L"))).astype(np.float32)


def finetune() -> str:
    from cellpose import models, train

    done = common.read_json(common.STATE / "a3_cellpose.model.json")
    if done and (common.MODELS / "cellpose_lynsec").exists():
        return done["path"]
    meta = common.read_json(common.LYNSEC / "meta.json")
    images, masks = [], []
    for entry in meta["train"]:
        images.append(_grey(np.asarray(Image.open(common.LYNSEC / "train" / f"{entry['name']}.png").convert("RGB"))))
        masks.append(np.load(common.LYNSEC / "train" / f"{entry['name']}_inst.npz")["labels"].astype(np.int32))

    model = models.CellposeModel(gpu=False, model_type="nuclei")
    # Time one short run to size the epochs to the cap, instead of guessing.
    started = time.monotonic()
    train.train_seg(model.net, train_data=images[:16], train_labels=masks[:16], channels=[0, 0],
                    n_epochs=1, save_path=str(common.MODELS), model_name="probe", min_train_masks=1)
    per_epoch = (time.monotonic() - started) * len(images) / 16
    epochs = int(max(5, min(300, TRAIN_HOURS * 3600 / max(1.0, per_epoch))))
    common.say(f"{len(images)} train images, ~{per_epoch:.0f}s/epoch -> {epochs} epochs", LOG)
    common.heartbeat("a3_cellpose", f"training {epochs} epochs")

    model = models.CellposeModel(gpu=False, model_type="nuclei")
    path, losses, _ = train.train_seg(
        model.net, train_data=images, train_labels=masks, channels=[0, 0],
        n_epochs=epochs, learning_rate=0.05, weight_decay=1e-4, SGD=True,
        save_path=str(common.MODELS), save_every=5, model_name="cellpose_lynsec", min_train_masks=1,
    )
    common.write_json(common.STATE / "a3_cellpose.model.json",
                      {"path": str(path), "epochs": epochs, "final_loss": float(losses[-1]) if len(losses) else None})
    return str(path)


def infer(detector: str, model, diameter: float | None) -> None:
    checkpoint = common.Checkpoint(f"a3_{detector}")
    units = [("LYNSEC", "lynsec")] + [(common.pair_id(c, m), "ihc") for c, m in common.PAIRS]
    for unit, side in units:
        if checkpoint.done(unit):
            continue
        if side == "lynsec":
            meta = common.read_json(common.LYNSEC / "meta.json") or {"test": []}
            items = [(e["name"], common.LYNSEC / "test" / f"{e['name']}.png") for e in meta["test"]]
        else:
            meta = common.read_json(common.FIELDS / unit / "meta.json") or {"ihc": []}
            items = [(e["name"], common.FIELDS / unit / "ihc" / f"{e['name']}.png") for e in meta["ihc"]]
        for name, path in items:
            out = common.LABELS / detector / unit / f"{side}_{name}.npz"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            grey = _grey(np.asarray(Image.open(path).convert("RGB")))
            masks, *_ = model.eval(grey, diameter=diameter, channels=[0, 0])
            np.savez_compressed(out, labels=np.asarray(masks, dtype=np.int32))
            common.heartbeat("a3_cellpose", f"{detector} {unit} {name}")
        checkpoint.mark(unit, images=len(items))
        common.say(f"{detector} {unit}: {len(items)} images", LOG)


def main() -> int:
    from cellpose import models

    common.ensure_dirs()
    infer("cellpose_nuclei", models.CellposeModel(gpu=False, model_type="nuclei"), DIAMETER_PX)
    path = finetune()
    tuned = models.CellposeModel(gpu=False, pretrained_model=path)
    diameter = float(getattr(tuned, "diam_labels", 0) or 0) or DIAMETER_PX
    common.say(f"fine-tuned model {path}, diameter {diameter:.1f}px", LOG)
    infer("cellpose_finetuned", tuned, diameter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
