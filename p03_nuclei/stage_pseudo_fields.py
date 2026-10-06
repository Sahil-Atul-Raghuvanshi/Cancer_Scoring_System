"""Option 4, part 1: unlabelled training fields from our own slides, and one detector's view.

    python stage_pseudo_fields.py

Self-training needs fields the benchmark never scores: for each pair, up to `CAP` of the
fields step 13 sampled that are *not* among its benchmark fields, copied as the RGB tile
it read. `instanseg_odsum` is run on them here (it needs the backend); the Cellpose stage
then runs zero-shot Cellpose on the same fields, and nuclei the two independent detectors
agree on become training labels (`stage_a4_cellpose.py pseudo`).

Same slides, different fields: the self-trained model is evaluated on fields it never saw,
but on sections it did - a transductive set-up, which the report says.
Written as `pseudo/<pair>/<name>.png` and `labels/instanseg_odsum/PSEUDO_<pair>/pseudo_<name>.npz`.
"""

from __future__ import annotations

import json
import random
import shutil
import sys

import numpy as np

import common
import stage_a1
import stage_fields

LOG = common.LOGS / "pseudo_fields.log"
CAP = 30


def export(case: str, marker: str) -> int:
    from PIL import Image

    from app.nuclei.model import segment_array

    unit = common.pair_id(case, marker)
    root, _, pair = stage_fields._pair_paths(case, marker)
    benchmark = {e["name"] for e in (common.read_json(common.FIELDS / unit / "meta.json") or {"ihc": []})["ihc"]}
    report = json.loads((root / "nuclei" / pair / "report.json").read_text(encoding="utf-8"))
    candidates = []
    for region in report["regions"]:
        for field in region["fields"]:
            name = f"r{region['rank']}_f{field['index']}"
            raw = root / "nuclei" / pair / f"region{region['rank']}" / f"f{field['index']}_raw.png"
            if name not in benchmark and raw.is_file():
                candidates.append((name, raw))
    chosen = random.Random(common.SEED + 1).sample(candidates, min(CAP, len(candidates)))

    out = common.PSEUDO / unit
    out.mkdir(parents=True, exist_ok=True)
    images = {}
    for name, raw in chosen:
        shutil.copy2(raw, out / f"{name}.png")
        images[name] = np.asarray(Image.open(raw).convert("RGB"))
    white = stage_a1._white(list(images.values())) if images else None
    for name, rgb in images.items():
        path = common.LABELS / "instanseg_odsum" / f"PSEUDO_{unit}" / f"pseudo_{name}.npz"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = segment_array(stage_a1._render(stage_a1._od_sum(rgb, white)))
        np.savez_compressed(path, labels=np.asarray(labels, dtype=np.int32))
        common.heartbeat("pseudo", f"{unit} {name}")
    common.write_json(out / "meta.json", {"fields": [{"name": n} for n in images],
                                          "excluded_benchmark": sorted(benchmark)})
    return len(images)


def main() -> int:
    common.ensure_dirs()
    checkpoint = common.Checkpoint("pseudo_fields")
    failures = 0
    for case, marker in common.PAIRS:
        unit = common.pair_id(case, marker)
        if checkpoint.done(unit):
            continue
        try:
            count = export(case, marker)
            checkpoint.mark(unit, fields=count)
            common.say(f"{unit}: {count} fields", LOG)
        except Exception:  # noqa: BLE001
            import traceback
            checkpoint.fail(unit, traceback.format_exc())
            common.say(f"{unit}: FAILED\n{traceback.format_exc()}", LOG)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
