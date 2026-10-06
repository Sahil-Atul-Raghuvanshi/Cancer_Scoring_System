"""Cut the benchmark: the same IHC fields production measured, and matched H&E fields.

IHC  - `FIELD_CAP` of the fields step 13 sampled for the pair, chosen reproducibly, copied
       as the RGB tile it read (`f*_raw.png`) together with production's own nuclei
       (`f*_labels.npz`), which *are* the baseline detector - no re-run, no drift.
H&E  - the reference: fields drawn on the H&E side of the same regions with step 13's
       area-proportional allocation (P-21), read at the same resolution, RGB.

Every detector in every environment then reads these PNGs, so they all see identical
pixels. Checkpointed per pair.
"""

from __future__ import annotations

import json
import random
import shutil
import sys

import numpy as np

import common

sys.path.insert(0, str(common.BACKEND))
LOG = common.LOGS / "fields.log"


def _pair_paths(case: str, marker: str):
    root = common.HISTORY / case / "markers" / marker
    record = json.loads((root / "case.json").read_text(encoding="utf-8"))
    pair = f"{record['he_upload_id']}__{record['ihc_upload_id']}"
    return root, record, pair


def export(case: str, marker: str) -> dict:
    from PIL import Image

    from app.core.config import settings
    from app.ingestion.slide_reader import open_slide
    from app.pipeline.step05_optical_density.tiles import read_tile
    from app.pipeline.step13_nuclei_segmentation.sampling import allocate, fields_for_region

    root, record, pair = _pair_paths(case, marker)
    out = common.FIELDS / common.pair_id(case, marker)
    (out / "ihc").mkdir(parents=True, exist_ok=True)
    (out / "he").mkdir(parents=True, exist_ok=True)
    (common.LABELS / "baseline_h" / common.pair_id(case, marker)).mkdir(parents=True, exist_ok=True)

    report = json.loads((root / "nuclei" / pair / "report.json").read_text(encoding="utf-8"))
    candidates = []
    for region in report["regions"]:
        for field in region["fields"]:
            raw = root / "nuclei" / pair / f"region{region['rank']}" / f"f{field['index']}_raw.png"
            if raw.is_file():
                candidates.append((region["rank"], field, raw))
    rng = random.Random(common.SEED)
    chosen = rng.sample(candidates, min(common.FIELD_CAP, len(candidates)))

    ihc_meta = []
    for rank, field, raw in chosen:
        name = f"r{rank}_f{field['index']}"
        shutil.copy2(raw, out / "ihc" / f"{name}.png")
        labels = np.load(raw.with_name(f"f{field['index']}_labels.npz"))["labels"]
        np.savez_compressed(common.LABELS / "baseline_h" / common.pair_id(case, marker) / f"ihc_{name}.npz",
                            labels=labels.astype(np.int32))
        ihc_meta.append({"name": name, "rank": rank, "x": field["x"], "y": field["y"],
                         "span": field["span"], "size": field["size"], "mpp": field["mpp"]})

    # H&E reference fields over the same regions, step 13's allocation (P-21).
    alignment = json.loads((root / "ihc_alignment" / pair / "report.json").read_text(encoding="utf-8"))
    regions = alignment["regions"]
    slide_record = None
    for path in list((common.HISTORY / case / "shared" / "slides").glob("*.json")):
        if path.stem == record["he_upload_id"]:
            slide_record = json.loads(path.read_text(encoding="utf-8"))
    if slide_record is None:
        raise RuntimeError("H&E slide record not found")

    he_meta = []
    reader = open_slide(slide_record["final_path"])
    try:
        mpp = reader.mpp
        width, height = reader.dimensions
        shares = allocate([float(r.get("areaMm2") or 0.0) for r in regions])
        fields = []
        for region, share in zip(regions, shares):
            sampled, _ = fields_for_region(region["heRings"], mpp=mpp, slide_width=width,
                                           slide_height=height, count=share)
            fields.extend((region.get("rank"), f) for f in sampled)
        rng = random.Random(common.SEED)
        picked = rng.sample(fields, min(common.FIELD_CAP, len(fields)))
        for k, (rank, field) in enumerate(picked):
            # The IHC fields' own resolution, so both sides are read at one scale.
            tile = read_tile(reader, x=field.x, y=field.y, target_mpp=float(ihc_meta[0]["mpp"]),
                             size=field.size, base_mpp=mpp)
            name = f"r{rank}_h{k}"
            Image.fromarray(np.asarray(tile.rgb, dtype=np.uint8)).save(out / "he" / f"{name}.png")
            he_meta.append({"name": name, "rank": rank, "x": field.x, "y": field.y,
                            "span": tile.span, "size": tile.size, "mpp": tile.mpp})
    finally:
        reader.close()

    meta = {"case": case, "marker": marker, "pair": pair,
            "filed_ihc_density": report.get("densityPerMm2"),
            "border_px": settings.nuclei_border_margin_px,
            "min_area_um2": settings.nuclei_min_area_um2,
            "ihc": ihc_meta, "he": he_meta}
    common.write_json(out / "meta.json", meta)
    return {"ihc": len(ihc_meta), "he": len(he_meta)}


def main() -> int:
    common.ensure_dirs()
    checkpoint = common.Checkpoint("fields")
    failures = 0
    for case, marker in common.PAIRS:
        unit = common.pair_id(case, marker)
        if checkpoint.done(unit):
            continue
        common.heartbeat("fields", unit)
        try:
            counts = export(case, marker)
            checkpoint.mark(unit, **counts)
            common.say(f"{unit}: {counts}", LOG)
        except Exception as exc:  # noqa: BLE001 - one pair must not stop the benchmark
            import traceback
            checkpoint.fail(unit, traceback.format_exc())
            common.say(f"{unit}: FAILED {exc}", LOG)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
