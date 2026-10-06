"""Option 2: contact sheets to look at - each detector's nuclei drawn on the same field.

    python stage_visual.py

Numbers cannot say whether a detector over-counts or the H&E reference under-counts; a
picture of the field often can, even to a non-pathologist (is there a dark blob with no
outline? an outline round nothing?). For the two worst pairs (CAN_00267 CD44 and ABCC4),
the two pairs zero-shot Cellpose gets closest on, and the breast labelled set, this writes
one sheet per field under `visual/`: the raw tile, production's haematoxylin render, then
each detector's outlines on the raw tile, and for the labelled set the truth first. Each of
our pairs also gets a sheet of H&E fields with the reference's outlines, for scale.
Run last, so every detector that finished is on it.
"""

from __future__ import annotations

import sys

import numpy as np

import common

LOG = common.LOGS / "visual.log"
SHOW = ["baseline_h", "instanseg_rgb", "cellpose_nuclei", "cellpose_odsum", "cellpose_bc",
        "cellpose_selftrained", "deepliif", "lynsec_hovernet"]
FIELDS = 3
ZOOM = 2


def outline(rgb: np.ndarray, labels: np.ndarray, colour=(255, 230, 0)) -> np.ndarray:
    edge = np.zeros(labels.shape, bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    out = rgb.copy()
    out[edge & (labels > 0)] = colour
    return out


def sheet(panels: list[tuple[str, np.ndarray]], path) -> None:
    from PIL import Image, ImageDraw

    tiles = []
    for title, image in panels:
        big = Image.fromarray(image).resize((image.shape[1] * ZOOM, image.shape[0] * ZOOM), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (big.width, big.height + 22), "white")
        canvas.paste(big, (0, 22))
        ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
        tiles.append(canvas)
    per_row = 5
    rows = [tiles[i:i + per_row] for i in range(0, len(tiles), per_row)]
    width = max(sum(t.width + 6 for t in row) for row in rows)
    height = sum(max(t.height for t in row) + 6 for row in rows)
    out = Image.new("RGB", (width, height), (230, 230, 230))
    y = 0
    for row in rows:
        x = 0
        for tile in row:
            out.paste(tile, (x, y))
            x += tile.width + 6
        y += max(t.height for t in row) + 6
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def _labels(detector: str, unit: str, side: str, name: str):
    path = common.LABELS / detector / unit / f"{side}_{name}.npz"
    return np.load(path)["labels"] if path.exists() else None


def _count(labels) -> int:
    return int(len(np.unique(labels)) - 1)


def pair_sheets(unit: str) -> int:
    from PIL import Image

    meta = common.read_json(common.FIELDS / unit / "meta.json")
    if not meta:
        return 0
    for entry in meta["ihc"][:FIELDS]:
        name = entry["name"]
        rgb = np.asarray(Image.open(common.FIELDS / unit / "ihc" / f"{name}.png").convert("RGB"))
        panels = [(f"{unit} {name} raw", rgb)]
        h = common.RENDERS / "h" / unit / f"ihc_{name}.png"
        if h.exists():
            panels.append(("haematoxylin render (production input)", np.asarray(Image.open(h).convert("RGB"))))
        for detector in SHOW:
            labels = _labels(detector, unit, "ihc", name)
            if labels is not None:
                panels.append((f"{detector}: {_count(labels)}", outline(rgb, labels)))
        sheet(panels, common.VISUAL / unit / f"{name}.png")
    panels = []
    for entry in meta["he"][:FIELDS]:
        rgb = np.asarray(Image.open(common.FIELDS / unit / "he" / f"{entry['name']}.png").convert("RGB"))
        labels = _labels("instanseg_rgb", unit, "he", entry["name"])
        panels.append((f"H&E {entry['name']} raw", rgb))
        if labels is not None:
            panels.append((f"H&E reference: {_count(labels)}", outline(rgb, labels)))
    if panels:
        sheet(panels, common.VISUAL / unit / "he_reference.png")
    return 1


def labelled_sheets(unit: str) -> int:
    from PIL import Image

    side = common.LABELLED[unit][0]
    for name, path, inst in common.labelled_items(unit, limit=FIELDS):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        truth = np.load(inst)["labels"]
        panels = [(f"{unit} {name} raw", rgb), (f"TRUTH: {_count(truth)}", outline(rgb, truth, (0, 255, 0)))]
        for detector in SHOW:
            labels = _labels(detector, unit, side, name)
            if labels is not None:
                panels.append((f"{detector}: {_count(labels)}", outline(rgb, labels)))
        sheet(panels, common.VISUAL / unit / f"{name}.png")
    return 1


def chosen_pairs() -> list[str]:
    results = common.read_json(common.OUT / "results.json", {}) or {}
    rows = results.get("pairs", {})
    gaps = {u: abs(r["cellpose_nuclei"]["shortfall"]) for u, r in rows.items()
            if (r.get("cellpose_nuclei") or {}).get("shortfall") is not None}
    best = sorted(gaps, key=gaps.get)[:2]
    return ["CAN_00267_A", "CAN_00267_F"] + [u for u in best if u not in ("CAN_00267_A", "CAN_00267_F")]


def main() -> int:
    common.ensure_dirs()
    made = []
    for unit in chosen_pairs():
        if pair_sheets(unit):
            made.append(unit)
            common.heartbeat("visual", unit)
    for unit in common.LABELLED:
        labelled_sheets(unit)
        made.append(unit)
    common.write_json(common.VISUAL / "index.json", {"sheets_for": made})
    common.say(f"contact sheets for {made}", LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
