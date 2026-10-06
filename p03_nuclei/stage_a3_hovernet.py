"""Approach 3a: an IHC-trained detector - LyNSeC's published IHC model, in HoVer-Net.

Runs in the `hovernet` venv with HoVer-Net's own architecture and post-processing. The
checkpoint does not record HoVer-Net's mode ("fast" 256->164 or "original" 270->80), so
both are tried on a few held-out LyNSeC images and the one that matches the labels better
is used - chosen on data, recorded in the log. Trained at 40x, so 0.5 um/px inputs are
upscaled 2x and the instance map brought back with nearest-neighbour.
"""

from __future__ import annotations

import sys

import numpy as np
from PIL import Image

import common

sys.path.insert(0, str(common.DOWNLOADS / "hovernet_code" / "hover_net-master"))
LOG = common.LOGS / "a3_hovernet.log"
GEOMETRY = {"fast": (256, 164), "original": (270, 80)}

#: Measured ~110 s a field on this CPU. The first run did all 76 LyNSeC images and the
#: first four pairs in full before its 4 h cap; the rest are scored on the first 8 fields
#: of their seeded sample so the night can finish. The report says which.
FIELDS_PER_PAIR = 8


def load(mode: str):
    import torch

    from models.hovernet.net_desc import HoVerNet

    checkpoint = torch.load(common.DOWNLOADS / "ihc.tar", map_location="cpu", weights_only=False)
    state = checkpoint.get("desc", checkpoint)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    tp = [v for k, v in state.items() if k.startswith("decoder.tp.") and v.ndim == 4]
    nr_types = int(tp[-1].shape[0]) if tp else None
    net = HoVerNet(nr_types=nr_types, mode=mode)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net, nr_types


def segment(rgb: np.ndarray, net, nr_types, mode: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    from models.hovernet.post_proc import process

    height, width = rgb.shape[:2]
    image = np.asarray(Image.fromarray(rgb).resize((width * 2, height * 2), Image.Resampling.BICUBIC))
    win, out = GEOMETRY[mode]
    margin = (win - out) // 2
    H, W = image.shape[:2]
    ph, pw = -(-H // out) * out, -(-W // out) * out
    padded = np.pad(image, ((margin, ph - H + margin), (margin, pw - W + margin), (0, 0)), mode="reflect")
    channels = (1 if nr_types else 0) + 3
    canvas = np.zeros((ph, pw, channels), np.float32)
    origins = [(y, x) for y in range(0, ph, out) for x in range(0, pw, out)]
    for start in range(0, len(origins), 8):
        batch = origins[start:start + 8]
        tiles = np.stack([padded[y:y + win, x:x + win] for y, x in batch]).astype(np.float32)
        with torch.inference_mode():
            pred = net(torch.from_numpy(tiles).permute(0, 3, 1, 2))
        np_map = F.softmax(pred["np"].permute(0, 2, 3, 1), dim=-1)[..., 1:]
        hv = pred["hv"].permute(0, 2, 3, 1)
        parts = [np_map, hv]
        if nr_types:
            tp = torch.argmax(F.softmax(pred["tp"].permute(0, 2, 3, 1), dim=-1), dim=-1, keepdim=True).float()
            parts = [tp, np_map, hv]
        merged = torch.cat(parts, -1).numpy()
        for (y, x), tile in zip(batch, merged):
            canvas[y:y + out, x:x + out] = tile[: out, : out]
    pred_map = canvas[:H, :W]
    result = process(pred_map, nr_types=nr_types)
    inst = result[0] if isinstance(result, tuple) else result
    small = Image.fromarray(inst.astype(np.int32)).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(small, dtype=np.int32)


def _f1(pred: np.ndarray, truth: np.ndarray) -> float:
    pairs = np.stack([truth.ravel(), pred.ravel()], 1)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    if not len(pairs):
        return 0.0
    keys, inter = np.unique(pairs, axis=0, return_counts=True)
    ta = np.bincount(truth.ravel()); pa = np.bincount(pred.ravel())
    iou = inter / (ta[keys[:, 0]] + pa[keys[:, 1]] - inter)
    tp = int((iou > 0.5).sum())
    n_t, n_p = int((ta[1:] > 0).sum()), int((pa[1:] > 0).sum())
    return 2 * tp / max(1, n_t + n_p)


def choose_mode() -> str:
    state = common.read_json(common.STATE / "a3_hovernet.mode.json")
    if state:
        return state["mode"]
    meta = common.read_json(common.LYNSEC / "meta.json") or {"test": []}
    sample = meta["test"][:6]
    scores = {}
    for mode in GEOMETRY:
        try:
            net, nr_types = load(mode)
            values = []
            for entry in sample:
                rgb = np.asarray(Image.open(common.LYNSEC / "test" / f"{entry['name']}.png").convert("RGB"))
                truth = np.load(common.LYNSEC / "test" / f"{entry['name']}_inst.npz")["labels"]
                values.append(_f1(segment(rgb, net, nr_types, mode), truth))
            scores[mode] = float(np.mean(values)) if values else 0.0
        except Exception as exc:  # noqa: BLE001
            scores[mode] = -1.0
            common.say(f"mode {mode} failed: {exc}", LOG)
    mode = max(scores, key=scores.get)
    common.say(f"HoVer-Net mode by F1 on {len(sample)} LyNSeC images: {scores} -> {mode}", LOG)
    common.write_json(common.STATE / "a3_hovernet.mode.json", {"mode": mode, "scores": scores})
    return mode


def main() -> int:
    common.ensure_dirs()
    mode = choose_mode()
    net, nr_types = load(mode)
    checkpoint = common.Checkpoint("a3_hovernet")
    units = [("LYNSEC", "lynsec")] + [(common.pair_id(c, m), "ihc") for c, m in common.PAIRS]
    failures = 0
    for unit, side in units:
        if checkpoint.done(unit):
            continue
        try:
            if side == "lynsec":
                meta = common.read_json(common.LYNSEC / "meta.json") or {"test": []}
                items = [(e["name"], common.LYNSEC / "test" / f"{e['name']}.png") for e in meta["test"]]
            else:
                meta = common.read_json(common.FIELDS / unit / "meta.json") or {"ihc": []}
                items = [(e["name"], common.FIELDS / unit / "ihc" / f"{e['name']}.png") for e in meta["ihc"][:FIELDS_PER_PAIR]]
            for name, path in items:
                out = common.LABELS / "lynsec_hovernet" / unit / f"{side}_{name}.npz"
                if out.exists():
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, labels=segment(np.asarray(Image.open(path).convert("RGB")), net, nr_types, mode))
                common.heartbeat("a3_hovernet", f"{unit} {name}")
            checkpoint.mark(unit, images=len(items), mode=mode)
            common.say(f"{unit}: {len(items)} images", LOG)
        except Exception:  # noqa: BLE001
            import traceback
            checkpoint.fail(unit, traceback.format_exc())
            common.say(f"{unit}: FAILED\n{traceback.format_exc()}", LOG)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
