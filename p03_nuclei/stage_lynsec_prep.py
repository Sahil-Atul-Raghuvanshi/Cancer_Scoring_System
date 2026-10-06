"""Turn LyNSeC's labelled nuclei into a test set (all detectors) and a train set (fine-tune).

The archive holds three subsets of 512x512x5 float32 arrays in HoVer-Net's convention -
R, G, B, instance map, class map - and does not say which subset is IHC or what the
pixel size is. Both are measured here rather than assumed, and logged:

  which subset    the one whose images carry the most DAB (Ruifrok un-mixing);
  pixel size      from the median nucleus area against ~35 um2, a lymphocyte nucleus,
                  snapped to 0.25 or 0.5 um/px.

Images are brought to 0.5 um/px - the benchmark fields' resolution - so every detector is
scored on the same scale it runs at on our slides. Split 80/20 by image, seeded; the test
images are never seen by the fine-tune.
"""

from __future__ import annotations

import sys
import zipfile

import numpy as np

import common

sys.path.insert(0, str(common.BACKEND))
LOG = common.LOGS / "lynsec_prep.log"
NUCLEUS_UM2 = 35.0


def _dab_share(rgb: np.ndarray) -> float:
    from app.common import stains
    from app.common.imaging import optical_density

    od = optical_density(rgb.astype(np.float32), (245.0, 245.0, 245.0)).reshape(-1, 3)
    coefficients = od @ stains.RUIFROK_INVERSE.T
    return float(np.mean(coefficients[:, 1] > 0.15))


def main() -> int:
    from PIL import Image

    common.ensure_dirs()
    if not (common.DOWNLOADS / "lynsec_data.ok").exists():
        common.say("lynsec data not downloaded yet", LOG)
        return 2
    if (common.LYNSEC / "meta.json").exists():
        return 0

    root = common.DOWNLOADS / "lynsec_data"
    files = sorted(p for p in root.rglob("*.npy"))
    subsets: dict[str, list] = {}
    for path in files:
        subsets.setdefault(path.parent.name, []).append(path)

    survey = {}
    for name, paths in subsets.items():
        sample = [np.load(p) for p in paths[:: max(1, len(paths) // 12)][:12]]
        dab = float(np.mean([_dab_share(a[..., :3].clip(0, 255)) for a in sample]))
        areas = []
        for a in sample:
            inst = a[..., 3].astype(np.int64)
            counts = np.bincount(inst.ravel())[1:]
            areas.extend(counts[counts > 0].tolist())
        survey[name] = {"images": len(paths), "dab_share": round(dab, 4),
                        "median_area_px": float(np.median(areas)) if areas else None,
                        "shape": list(sample[0].shape), "dtype": str(sample[0].dtype)}
    common.say(f"subsets: {survey}", LOG)

    chosen = max(survey, key=lambda n: survey[n]["dab_share"])
    area_px = survey[chosen]["median_area_px"]
    raw_mpp = (NUCLEUS_UM2 / area_px) ** 0.5 if area_px else 0.25
    native = 0.25 if raw_mpp < 0.375 else 0.5
    scale = native / 0.5  # 0.5 when native is 0.25: shrink by two
    common.say(f"IHC subset = {chosen!r}; estimated {raw_mpp:.3f} um/px -> native {native}, "
               f"rescaled to 0.5 um/px", LOG)

    paths = sorted(subsets[chosen], key=lambda p: p.name)
    rng = np.random.default_rng(common.SEED)
    order = rng.permutation(len(paths))
    cut = int(round(0.8 * len(paths)))
    split = {"train": [paths[i] for i in order[:cut]], "test": [paths[i] for i in order[cut:]]}

    meta = {"subset": chosen, "survey": survey, "native_mpp": native, "mpp": 0.5,
            "train": [], "test": []}
    for part, items in split.items():
        (common.LYNSEC / part).mkdir(parents=True, exist_ok=True)
        for path in items:
            array = np.load(path)
            rgb = array[..., :3].clip(0, 255).astype(np.uint8)
            inst = array[..., 3].astype(np.int32)
            if scale != 1.0:
                size = (int(round(rgb.shape[1] * scale)), int(round(rgb.shape[0] * scale)))
                rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BOX))
                inst = np.asarray(Image.fromarray(inst).resize(size, Image.Resampling.NEAREST))
            name = path.stem
            Image.fromarray(rgb).save(common.LYNSEC / part / f"{name}.png")
            np.savez_compressed(common.LYNSEC / part / f"{name}_inst.npz", labels=inst.astype(np.int32))
            meta[part].append({"name": name})
        common.heartbeat("lynsec_prep", f"{part} {len(items)}")
    common.write_json(common.LYNSEC / "meta.json", meta)
    common.say(f"train {len(meta['train'])}, test {len(meta['test'])}", LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
