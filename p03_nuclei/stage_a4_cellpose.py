"""The second run's Cellpose work: other inputs, breast training, and self-training.

    python stage_a4_cellpose.py zero   option 5 + the zero-shot rows the new sets need
    python stage_a4_cellpose.py bc     option 1: fine-tune on BC-DeepLIIF breast IHC
    python stage_a4_cellpose.py self   option 4: RETIRED 6 Oct, kept for the record - see below

Runs in the `cellpose` venv. The first night's best result on our slides was zero-shot
Cellpose on inverted grey (mean gap 34%, 5/10 pairs within 20%), and its fine-tune on
lymphoma IHC made our slides worse. So this run asks three narrower questions:

  zero   Does the input matter? Zero-shot Cellpose on production's haematoxylin render
         (`cellpose_hinput`) and on the OD-sum render (`cellpose_odsum`), from
         `stage_renders.py`. Also fills in the zero-shot and LyNSeC-fine-tuned models on
         the breast test set and on the self-training fields.
  bc     Does breast IHC training transfer where lymphoma did not? (`cellpose_bc`)
  self   Does training on our own slides help? Labels = nuclei that zero-shot Cellpose
         and `instanseg_odsum` both find (IoU > 0.5), on fields the benchmark never scores.
         Unlabelled pixels count as background, so a field is used only when the two
         detectors mostly agree. Nuclei both detectors miss stay missed - this can steady a
         detector, it cannot teach it what neither sees. (`cellpose_selftrained`)
         Result: they agree on only 28-40% of each other's nuclei and 1 of 236 fields
         qualified. Decision: no training on our slides without real labels - consensus at
         that level is not ground truth. `night.py` no longer runs this mode.

All three trained/zero-shot models read inverted grey, as on the first night, except the
two render rows. Same hyper-parameters as the LyNSeC fine-tune, so the runs compare.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import common
from stage_a3_cellpose import DIAMETER_PX, _grey

LOG = common.LOGS / "a4_cellpose.log"
TRAIN_HOURS = 2.5
AGREEMENT = 0.5      # matched / union of the two detectors' nuclei on a field
MIN_AGREED = 10


def _pairs(source=lambda unit, name: common.FIELDS / unit / "ihc" / f"{name}.png"):
    out = []
    for case, marker in common.PAIRS:
        unit = common.pair_id(case, marker)
        meta = common.read_json(common.FIELDS / unit / "meta.json") or {"ihc": []}
        out.append((unit, "ihc", [(e["name"], source(unit, e["name"])) for e in meta["ihc"]]))
    return out


def _labelled(units=("LYNSEC", "BCDL"), source=None):
    out = []
    for unit in units:
        side = common.LABELLED[unit][0]
        items = [(n, source(unit, side, n) if source else p) for n, p, _ in common.labelled_items(unit)]
        out.append((unit, side, items))
    return out


def _pseudo_units():
    out = []
    for case, marker in common.PAIRS:
        unit = common.pair_id(case, marker)
        meta = common.read_json(common.PSEUDO / unit / "meta.json") or {"fields": []}
        out.append((f"PSEUDO_{unit}", "pseudo", [(e["name"], common.PSEUDO / unit / f"{e['name']}.png")
                                                 for e in meta["fields"]]))
    return out


def infer(detector: str, model, diameter: float, units) -> None:
    checkpoint = common.Checkpoint(f"a4_{detector}")
    for unit, side, items in units:
        if checkpoint.done(unit):
            continue
        for name, path in items:
            out = common.LABELS / detector / unit / f"{side}_{name}.npz"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            masks, *_ = model.eval(_grey(np.asarray(Image.open(path).convert("RGB"))),
                                   diameter=diameter, channels=[0, 0])
            np.savez_compressed(out, labels=np.asarray(masks, dtype=np.int32))
            common.heartbeat("a4_cellpose", f"{detector} {unit} {name}")
        checkpoint.mark(unit, images=len(items))
        common.say(f"{detector} {unit}: {len(items)} images", LOG)


def finetune(name: str, images: list, masks: list) -> str:
    from cellpose import models, train

    state = common.STATE / f"a4_{name}.model.json"
    done = common.read_json(state)
    if done and Path(done["path"]).exists():
        return done["path"]
    model = models.CellposeModel(gpu=False, model_type="nuclei")
    started = time.monotonic()
    probe = min(16, len(images))
    train.train_seg(model.net, train_data=images[:probe], train_labels=masks[:probe], channels=[0, 0],
                    n_epochs=1, save_path=str(common.MODELS), model_name=f"probe_{name}", min_train_masks=1)
    per_epoch = (time.monotonic() - started) * len(images) / probe
    epochs = int(max(5, min(300, TRAIN_HOURS * 3600 / max(1.0, per_epoch))))
    common.say(f"{name}: {len(images)} train images, ~{per_epoch:.0f}s/epoch -> {epochs} epochs", LOG)
    common.heartbeat("a4_cellpose", f"{name} training {epochs} epochs")

    model = models.CellposeModel(gpu=False, model_type="nuclei")
    path, losses, _ = train.train_seg(
        model.net, train_data=images, train_labels=masks, channels=[0, 0],
        n_epochs=epochs, learning_rate=0.05, weight_decay=1e-4, SGD=True,
        save_path=str(common.MODELS), save_every=5, model_name=name, min_train_masks=1,
    )
    common.write_json(state, {"path": str(path), "epochs": epochs, "images": len(images),
                              "final_loss": float(losses[-1]) if len(losses) else None})
    return str(path)


def _trained(path: str):
    from cellpose import models

    model = models.CellposeModel(gpu=False, pretrained_model=path)
    return model, float(getattr(model, "diam_labels", 0) or 0) or DIAMETER_PX


def _evaluate(detector: str, path: str) -> None:
    model, diameter = _trained(path)
    common.say(f"{detector}: model {path}, diameter {diameter:.1f}px", LOG)
    infer(detector, model, diameter, _labelled() + _pairs())


def zero() -> int:
    from cellpose import models

    nuclei = models.CellposeModel(gpu=False, model_type="nuclei")
    for render, detector in (("h", "cellpose_hinput"), ("odsum", "cellpose_odsum")):
        rendered = lambda unit, name, render=render: common.RENDERS / render / unit / f"ihc_{name}.png"
        labelled = lambda unit, side, name, render=render: common.RENDERS / render / unit / f"{side}_{name}.png"
        infer(detector, nuclei, DIAMETER_PX, _labelled(source=labelled) + _pairs(rendered))
    infer("cellpose_nuclei", nuclei, DIAMETER_PX, _labelled(("BCDL",)) + _pseudo_units())
    lynsec = common.read_json(common.STATE / "a3_cellpose.model.json")
    if lynsec and Path(lynsec["path"]).exists():
        tuned, diameter = _trained(lynsec["path"])
        infer("cellpose_finetuned", tuned, diameter, _labelled(("BCDL",)))
    return 0


def bc() -> int:
    images, masks = [], []
    for _, path, inst in common.labelled_items("BCDL", "train"):
        images.append(_grey(np.asarray(Image.open(path).convert("RGB"))))
        masks.append(np.load(inst)["labels"].astype(np.int32))
    _evaluate("cellpose_bc", finetune("cellpose_bc", images, masks))
    return 0


def _match(a: np.ndarray, b: np.ndarray) -> list[int]:
    """Labels of `a` that overlap one label of `b` at IoU > 0.5."""
    pairs = np.stack([a.ravel(), b.ravel()], 1)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    if not len(pairs):
        return []
    keys, inter = np.unique(pairs, axis=0, return_counts=True)
    sa, sb = np.bincount(a.ravel()), np.bincount(b.ravel())
    iou = inter / (sa[keys[:, 0]] + sb[keys[:, 1]] - inter)
    return sorted(set(keys[iou > 0.5, 0].tolist()))


def pseudo_labels() -> tuple[list, list]:
    images, masks, stats = [], [], {}
    for unit, side, items in _pseudo_units():
        kept = 0
        for name, path in items:
            c = common.LABELS / "cellpose_nuclei" / unit / f"{side}_{name}.npz"
            i = common.LABELS / "instanseg_odsum" / unit / f"{side}_{name}.npz"
            if not (c.exists() and i.exists()):
                continue
            cp, ins = np.load(c)["labels"], np.load(i)["labels"]
            agreed = _match(cp, ins)
            union = len(np.unique(cp)) - 1 + len(np.unique(ins)) - 1 - len(agreed)
            if len(agreed) < MIN_AGREED or len(agreed) / max(1, union) < AGREEMENT:
                continue
            mask = np.where(np.isin(cp, agreed), cp, 0)
            _, mask = np.unique(mask, return_inverse=True)
            images.append(_grey(np.asarray(Image.open(path).convert("RGB"))))
            masks.append(mask.reshape(cp.shape).astype(np.int32))
            kept += 1
        stats[unit] = {"fields": len(items), "kept": kept}
    common.write_json(common.STATE / "a4_pseudo.json", stats)
    common.say(f"pseudo-labels: {sum(s['kept'] for s in stats.values())} fields kept: {stats}", LOG)
    return images, masks


def self_train() -> int:
    images, masks = pseudo_labels()
    if len(images) < 20:
        common.say(f"only {len(images)} fields where the detectors agree - too few to train on", LOG)
        return 1
    _evaluate("cellpose_selftrained", finetune("cellpose_selftrained", images, masks))
    return 0


if __name__ == "__main__":
    common.ensure_dirs()
    sys.exit({"zero": zero, "bc": bc, "self": self_train}[sys.argv[1]]())
