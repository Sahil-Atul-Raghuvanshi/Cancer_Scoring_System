"""Approach 2: DeepLIIF - nuclei found independently of the DAB.

Runs in the `deepliif` venv. DeepLIIF translates an IHC image into virtual multiplex
channels, including a nuclear one, and segments cells from them, so a nucleus under dense
brown is found from its own virtual stain rather than from a counterstain the DAB hid.

It was trained at 40x (~0.25 um/px); the benchmark fields are 0.5 um/px, so each is
upscaled 2x for inference and the instance map brought back with nearest-neighbour. Its
refined segmentation paints positive cells red and negative cells blue; instances are the
connected components of either. Checkpointed per pair, and per LyNSeC test set.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image
from scipy import ndimage

import common

LOG = common.LOGS / "a2_deepliif.log"

#: DeepLIIF runs nine image-to-image generators on every 512 px tile: measured at ~110 s a
#: field on this CPU, against ~4 s for InstanSeg. The full benchmark (316 images) would
#: run well past morning, so DeepLIIF is scored on a fixed subset - the first fields of each
#: pair's seeded sample and the first LyNSeC test images - and the report says so.
FIELDS_PER_PAIR = 8
LYNSEC_IMAGES = 20
#: Second run. DeepLIIF was built on this data, so its BC-DeepLIIF row is an upper bound
#: on what it can do, not a fair comparison - the report says so.
BCDL_IMAGES = 20


def model_dir():
    for path in (common.DOWNLOADS / "deepliif_model").rglob("train_opt.txt"):
        return path.parent
    raise RuntimeError("DeepLIIF model folder (with train_opt.txt) not found")


def segment(rgb: np.ndarray, directory) -> np.ndarray:
    from deepliif.models import infer_modalities

    height, width = rgb.shape[:2]
    big = Image.fromarray(rgb).resize((width * 2, height * 2), Image.Resampling.BICUBIC)
    images, _ = infer_modalities(big, 512, str(directory))
    refined = np.asarray(images["SegRefined"].convert("RGB")).astype(np.int16)
    cells = ((refined[..., 0] > 100) | (refined[..., 2] > 100)) & (refined[..., 1] < 100)
    labels, _ = ndimage.label(cells)
    small = Image.fromarray(labels.astype(np.int32)).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(small, dtype=np.int32)


def main() -> int:
    common.ensure_dirs()
    directory = model_dir()
    checkpoint = common.Checkpoint("a2_deepliif")
    units = [(common.pair_id(c, m), "ihc") for c, m in common.PAIRS] + [("LYNSEC", "lynsec"), ("BCDL", "bcdl")]
    failures = 0
    for unit, side in units:
        if checkpoint.done(unit):
            continue
        try:
            if side != "ihc":
                limit = LYNSEC_IMAGES if unit == "LYNSEC" else BCDL_IMAGES
                names = [(n, p) for n, p, _ in common.labelled_items(unit, limit=limit)]
            else:
                meta = common.read_json(common.FIELDS / unit / "meta.json") or {"ihc": []}
                names = [(e["name"], common.FIELDS / unit / "ihc" / f"{e['name']}.png") for e in meta["ihc"][:FIELDS_PER_PAIR]]
            for name, path in names:
                out = common.LABELS / "deepliif" / unit / f"{side}_{name}.npz"
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                labels = segment(np.asarray(Image.open(path).convert("RGB")), directory)
                np.savez_compressed(out, labels=labels)
                common.heartbeat("a2_deepliif", f"{unit} {name}")
            checkpoint.mark(unit, images=len(names))
            common.say(f"{unit}: {len(names)} images", LOG)
        except Exception:  # noqa: BLE001
            import traceback
            checkpoint.fail(unit, traceback.format_exc())
            common.say(f"{unit}: FAILED\n{traceback.format_exc()}", LOG)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
