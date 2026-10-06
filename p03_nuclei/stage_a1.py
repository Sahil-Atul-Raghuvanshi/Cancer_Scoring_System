"""Approach 1: use the detector the way it was trained.

    python stage_a1.py fields     the benchmark IHC fields and the H&E reference fields
    python stage_a1.py lynsec     LyNSeC's held-out labelled IHC images

Three detectors, all from the backend's own code so nothing here is a re-implementation:

  instanseg_rgb     InstanSeg's brightfield model on the RGB tile - its native input. The
                    production path feeds it a haematoxylin-only render instead, and under
                    dense DAB that render has nothing left where the nuclei are.
  instanseg_odsum   InstanSeg on a grey rendering of the optical-density sum (H + DAB), so a
                    brown-covered nucleus stays dark instead of disappearing - QuPath's own
                    remedy for DAB masking, given to the deep model.
  watershed_odsum   the backend's watershed on the OD sum - QuPath positive-cell-detection
                    style, a classical reference with no learned input at all.

The H&E reference fields get `instanseg_rgb` only: P-21 showed RGB is the right input there.
Labels are written as `<detector>/<pair>/<side>_<field>.npz`; checkpointed per pair.
"""

from __future__ import annotations

import sys

import numpy as np

import common

sys.path.insert(0, str(common.BACKEND))
LOG = common.LOGS / "a1.log"


def _white(images: list[np.ndarray]) -> np.ndarray:
    """One white point per pair: the bright tail of all its fields, held to plausible glass."""
    stacked = np.concatenate([image.reshape(-1, 3) for image in images])
    return np.clip(np.percentile(stacked, 99.5, axis=0), 200.0, 255.0)


def _od_sum(rgb: np.ndarray, white: np.ndarray) -> np.ndarray:
    from app.common.imaging import optical_density

    return optical_density(rgb.astype(np.float32), tuple(float(v) for v in white)).clip(0, None).sum(axis=-1)


def _render(od_sum: np.ndarray) -> np.ndarray:
    """OD sum as a grey brightfield image: dark where any stain is, white on glass."""
    grey = np.clip(255.0 * np.power(10.0, -od_sum / 1.5), 0, 255).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=-1)


def _production_render(rgb: np.ndarray, white: np.ndarray) -> np.ndarray:
    from app.pipeline.step13_nuclei_segmentation.stain_input import haematoxylin_only_rgb, ruifrok_basis

    return haematoxylin_only_rgb(rgb, tuple(float(v) for v in white), ruifrok_basis())


def run(images: dict[str, np.ndarray], *, mpp: float, sides: dict[str, str], out_pair: str) -> None:
    from app.nuclei.model import segment_array
    from app.pipeline.step13_nuclei_segmentation.watershed import segment as watershed

    white = _white(list(images.values()))
    for name, rgb in images.items():
        side = sides[name]
        targets = {"instanseg_rgb": lambda: segment_array(rgb)}
        if side == "lynsec":
            # Production's own input - the haematoxylin-only render step 13 feeds the
            # model - so the labelled comparison has a "today" row. On our slides that
            # row is production's saved output instead; LyNSeC images have none.
            targets["baseline_h"] = lambda: segment_array(_production_render(rgb, white))
        if side != "he":
            od = _od_sum(rgb, white)
            targets["instanseg_odsum"] = lambda od=od: segment_array(_render(od))
            targets["watershed_odsum"] = lambda od=od: watershed(od, mpp=mpp)
        for detector, make in targets.items():
            path = common.LABELS / detector / out_pair / f"{side}_{name}.npz"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, labels=np.asarray(make(), dtype=np.int32))
        common.heartbeat("a1", f"{out_pair} {name}")


def fields() -> int:
    from PIL import Image

    checkpoint = common.Checkpoint("a1_fields")
    failures = 0
    for case, marker in common.PAIRS:
        unit = common.pair_id(case, marker)
        meta = common.read_json(common.FIELDS / unit / "meta.json")
        if checkpoint.done(unit) or meta is None:
            continue
        try:
            images, sides = {}, {}
            for side in ("ihc", "he"):
                for entry in meta[side]:
                    images[entry["name"]] = np.asarray(
                        Image.open(common.FIELDS / unit / side / f"{entry['name']}.png").convert("RGB"))
                    sides[entry["name"]] = side
            run(images, mpp=float(meta["ihc"][0]["mpp"]), sides=sides, out_pair=unit)
            checkpoint.mark(unit, fields=len(images))
            common.say(f"{unit}: {len(images)} fields", LOG)
        except Exception:  # noqa: BLE001
            import traceback
            checkpoint.fail(unit, traceback.format_exc())
            common.say(f"{unit}: FAILED\n{traceback.format_exc()}", LOG)
            failures += 1
    return 1 if failures else 0


def lynsec() -> int:
    from PIL import Image

    meta = common.read_json(common.LYNSEC / "meta.json")
    if not meta:
        common.say("lynsec: not prepared yet", LOG)
        return 2
    checkpoint = common.Checkpoint("a1_lynsec")
    if checkpoint.done("test"):
        return 0
    images, sides = {}, {}
    for entry in meta["test"]:
        images[entry["name"]] = np.asarray(Image.open(common.LYNSEC / "test" / f"{entry['name']}.png").convert("RGB"))
        sides[entry["name"]] = "lynsec"
    run(images, mpp=float(meta["mpp"]), sides=sides, out_pair="LYNSEC")
    checkpoint.mark("test", images=len(images))
    return 0


if __name__ == "__main__":
    common.ensure_dirs()
    sys.exit(lynsec() if sys.argv[1:] == ["lynsec"] else fields())
