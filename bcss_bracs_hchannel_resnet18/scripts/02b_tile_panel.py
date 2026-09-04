"""G2's eyeball check, as a contact sheet: stored H-channel tile beside its RGB source.

    python scripts/02b_tile_panel.py                # 12 per class
    python scripts/02b_tile_panel.py --per-class 6

The gate asks for this and it is the one check no assertion can replace:

> Eyeball 12 random tiles per class against their RGB source. A human with no pathology
> training can confirm class 0 fat and stroma; **if class 2 tiles look like empty space,
> the mask is misaligned with the image** (check the resample and any off-by-one crop).

Every other G2 check tests self-consistency - the vote adds up, the export is
deterministic, no slide crosses a split. A systematic mask/image offset passes all of
them and produces a confidently wrong model, because the tiles are internally consistent
and simply labelled from the wrong place. Only looking catches it.

Each row is one tile: the RGB crop the exporter read, the H channel it stored, and the
label mask over the same window. If the H channel does not look like the RGB's nuclei,
the transform is wrong; if the mask's colours do not sit on the RGB's structures, the
geometry is wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backend_path
import bcss
import datasets
import export
import hchannel

CELL = 224
PAD = 10
LABEL_H = 22

#: Distinct, colour-blind-safe enough to tell apart at 224 px, and deliberately NOT the
#: red/green a reader might read as "right/wrong".
MASK_COLOURS = {
    bcss.NON_EPITHELIUM: (120, 145, 180),     # blue-grey
    bcss.NON_INVASIVE: (240, 190, 90),        # amber
    bcss.INVASIVE: (200, 110, 150),           # magenta
    bcss.IGNORE: (35, 35, 35),                # near-black: unlabelled, zero weight
}


def colourise(remapped: np.ndarray) -> np.ndarray:
    out = np.zeros((*remapped.shape, 3), dtype=np.uint8)
    for code, colour in MASK_COLOURS.items():
        out[remapped == code] = colour
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = datasets.read_manifest(backend_path.TILES_DIR / "tiles_manifest.csv")
    rng = np.random.default_rng(args.seed)

    regions = {r.roi_id: r for r in bcss.find_regions(backend_path.BCSS_DIR)}
    import json
    summary = json.loads((backend_path.TILES_DIR / "export_summary.json").read_text())
    spec = export.TileSpec(**{k: summary["spec"][k] for k in ("tile_px", "mpp")})
    measured = summary["source_mpp_measured"]

    picked: list[dict] = []
    for cls, name in enumerate(bcss.CLASS_NAMES):
        pool = [r for r in rows if int(r["label"]) == cls]
        if not pool:
            print(f"  {name}: no tiles")
            continue
        take = min(args.per_class, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        chosen = [pool[i] for i in idx]
        picked.extend(chosen)
        note = "" if take == args.per_class else f"   (only {take} exist)"
        print(f"  {name:<26} {take:>3} of {len(pool):,}{note}")

    if not picked:
        print("nothing to show")
        return 1

    width = 3 * CELL + 4 * PAD
    height = len(picked) * (CELL + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)

    draw.text((PAD, 1), "RGB source            H channel (stored)      "
                        "label mask", fill=(20, 20, 20))

    # Group by region so each source is read once - a BCSS region is up to 200 MB
    # decoded, and re-reading it per tile would take longer than the export did.
    by_region: dict[str, list[dict]] = {}
    for row in picked:
        by_region.setdefault(row["roi_id"], []).append(row)

    y = PAD + LABEL_H
    for roi_id, group in by_region.items():
        region = regions[roi_id]
        rgb_full = np.asarray(Image.open(region.image).convert("RGB"))
        mask_full = np.asarray(Image.open(region.mask))
        if mask_full.ndim == 3:
            mask_full = mask_full[..., 0]

        rgb_s, mask_s = export.resample(
            rgb_full, mask_full,
            source_mpp=measured[region.slide_key], target_mpp=spec.mpp,
        )
        remapped = bcss.remap(mask_s)

        for row in group:
            x0, y0 = int(row["x"]), int(row["y"])
            sl = (slice(y0, y0 + spec.tile_px), slice(x0, x0 + spec.tile_px))
            rgb_tile = rgb_s[sl]
            mask_tile = colourise(remapped[sl])

            stored = np.asarray(Image.open(backend_path.TILES_DIR / row["tile_path"]))
            h_tile = np.repeat(stored[:, :, None], 3, axis=2)

            for i, arr in enumerate((rgb_tile, h_tile, mask_tile)):
                img = Image.fromarray(np.ascontiguousarray(arr))
                if img.size != (CELL, CELL):
                    img = img.resize((CELL, CELL), Image.NEAREST)
                sheet.paste(img, (PAD + i * (CELL + PAD), y))

            label = (f"{bcss.CLASS_NAMES[int(row['label'])]}  "
                     f"usable {float(row['usable']):.2f}  "
                     f"inv {float(row['invasive_frac']):.2f}  "
                     f"noninv {float(row['non_invasive_frac']):.2f}  "
                     f"{row['institution']}/{row['slide_id']}")
            draw.text((PAD, y + CELL + 4), label, fill=(60, 60, 60))
            y += CELL + LABEL_H + PAD

    out = backend_path.REPORTS_DIR / "02b_tile_panel.png"
    sheet.save(out, optimize=True)
    print(f"\nwrote {out}  ({sheet.width}x{sheet.height})")
    print("\nWhat to look for, in order of how badly it bites:")
    print("  1. mask colours must sit ON the RGB's structures. If the magenta is beside")
    print("     the tumour rather than on it, the geometry is wrong and every number in")
    print("     the project is measuring the wrong pixels.")
    print("  2. the H channel must look like the RGB's NUCLEI, bright on dark.")
    print("     Inverted or flat means the transform or the polarity is wrong.")
    print("  3. class 2 tiles must not look like empty space or pure stroma.")
    print("  4. class 0 tiles should be recognisably fat, stroma or background -")
    print("     no pathology training needed for that one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
