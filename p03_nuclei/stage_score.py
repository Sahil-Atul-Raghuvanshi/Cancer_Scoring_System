"""Score every detector the same way, and write REPORT.md.

Run after each detector stage, so the report fills in through the night. Two measures:

  On our slides   counted nuclei per mm2 over the benchmark IHC fields, counted with step
                  13's own rule (`extract`: the border margin and the minimum nucleus area),
                  against the H&E reference - InstanSeg on RGB over matched H&E fields
                  (P-21). Shortfall = (H&E - IHC) / H&E. The review's target: within ~20%.
  Labelled sets   F1 at IoU 0.5 against instance labels on held-out IHC images, and the
                  count ratio (found / labelled): LyNSeC (lymphoma, hand-drawn) and, from
                  the second run, BC-DeepLIIF (breast Ki-67, labels from mpIF, so nuclei
                  under the brown are labelled too).

A density alone can be gamed by over-splitting, which is why the labelled F1 sits beside it.

Per tissue area (second run). Looking at the fields showed CAN_00267's CD44 fields are 2%
tissue against 31% on the matching H&E (they sit on glass), and its ABCC4 fields are mostly
the flat grey scanner-fill rectangle (P-05; these fields predate that fix). A density over
the whole field reads both as missing nuclei that were never there. So a second shortfall
divides each side's count by the area of *tissue* in its fields: optical-density sum over
`TISSUE_OD` (one white point per pair and side) *and* some texture, since fill is dark
enough to pass on density alone but its local variation is exactly zero. Needs no labels.
"""

from __future__ import annotations

import sys

import numpy as np

import common

sys.path.insert(0, str(common.BACKEND))
LOG = common.LOGS / "score.log"
TISSUE_OD = 0.25
TEXTURE_SD = 1.0  # grey-level std over 9x9 px; scanner fill and clean glass are 0


def _tissue_fractions(unit: str, side: str, entries: list) -> dict[str, float]:
    """Share of each field that is tissue; cached, since the fields never change."""
    from PIL import Image
    from scipy import ndimage

    cache = common.FIELDS / unit / f"tissue2_{side}.json"
    known = common.read_json(cache, {}) or {}
    if all(e["name"] in known for e in entries):
        return known
    images = {e["name"]: np.asarray(Image.open(common.FIELDS / unit / side / f"{e['name']}.png").convert("RGB"),
                                    dtype=np.float32) for e in entries}
    stacked = np.concatenate([a.reshape(-1, 3) for a in images.values()])
    white = np.clip(np.percentile(stacked, 99.5, axis=0), 200.0, 255.0)
    for name, rgb in images.items():
        od = (-np.log10(np.clip(rgb, 1.0, None) / white)).clip(0, None).sum(axis=-1)
        grey = rgb.mean(axis=-1)
        mean = ndimage.uniform_filter(grey, 9)
        sd = np.sqrt(np.clip(ndimage.uniform_filter(grey * grey, 9) - mean * mean, 0, None))
        known[name] = round(float(((od > TISSUE_OD) & (sd > TEXTURE_SD)).mean()), 4)
    common.write_json(cache, known)
    return known


def _count(labels: np.ndarray, mpp: float, border: int, min_area: float) -> tuple[int, list[float]]:
    from app.pipeline.step13_nuclei_segmentation.instances import extract

    nuclei = extract(labels, haematoxylin=np.zeros(labels.shape, np.float32), mpp=mpp, x0=0.0, y0=0.0,
                     level0_scale=1.0, border_px=border, min_area_um2=min_area)
    counted = [n for n in nuclei if n.counted]
    return len(counted), [n.area_um2 for n in counted]


def _area_mm2(size: int, border: int, mpp: float) -> float:
    from app.pipeline.step13_nuclei_segmentation.instances import counted_area_mm2

    return counted_area_mm2(size=size, border_px=border, mpp=mpp)


def _f1(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int]:
    pairs = np.stack([truth.ravel(), pred.ravel()], 1)
    pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
    ta = np.bincount(truth.ravel()); pa = np.bincount(pred.ravel())
    n_t, n_p = int((ta[1:] > 0).sum()), int((pa[1:] > 0).sum())
    if not len(pairs):
        return 0, n_t, n_p
    keys, inter = np.unique(pairs, axis=0, return_counts=True)
    iou = inter / (ta[keys[:, 0]] + pa[keys[:, 1]] - inter)
    return int((iou > 0.5).sum()), n_t, n_p


def score() -> dict:
    results = {"pairs": {}, "labelled": {}}
    for case, marker in common.PAIRS:
        unit = common.pair_id(case, marker)
        meta = common.read_json(common.FIELDS / unit / "meta.json")
        if not meta:
            continue
        border, min_area = int(meta["border_px"]), float(meta["min_area_um2"])
        row = {"filed_ihc_density": meta.get("filed_ihc_density")}
        tissue = {side: _tissue_fractions(unit, side, meta[side]) for side in ("ihc", "he")}
        for side in ("ihc", "he"):
            row[f"tissue_share_{side}"] = round(float(np.mean(list(tissue[side].values()))), 3)
        for detector in common.DETECTORS + ["he_reference"]:
            source, side = ("instanseg_rgb", "he") if detector == "he_reference" else (detector, "ihc")
            counted, area, tissue_area, areas, fields = 0, 0.0, 0.0, [], 0
            for entry in meta[side]:
                path = common.LABELS / source / unit / f"{side}_{entry['name']}.npz"
                if not path.exists():
                    continue
                n, a = _count(np.load(path)["labels"], float(entry["mpp"]), border, min_area)
                counted += n
                areas += a
                field_area = _area_mm2(int(entry["size"]), border, float(entry["mpp"]))
                area += field_area
                tissue_area += field_area * tissue[side].get(entry["name"], 0.0)
                fields += 1
            if fields:
                row[detector] = {"density": round(counted / area, 1) if area else None,
                                 "tissue_density": round(counted / tissue_area, 1) if tissue_area else None,
                                 "median_area_um2": round(float(np.median(areas)), 1) if areas else None,
                                 "fields": fields, "nuclei": counted}
        reference = (row.get("he_reference") or {}).get("density")
        for detector in common.DETECTORS:
            if detector in row and reference and row[detector]["density"] is not None:
                row[detector]["shortfall"] = round((reference - row[detector]["density"]) / reference, 3)
        tissue_ref = (row.get("he_reference") or {}).get("tissue_density")
        for detector in common.DETECTORS:
            if detector in row and tissue_ref and row[detector].get("tissue_density") is not None:
                row[detector]["tissue_shortfall"] = round(
                    (tissue_ref - row[detector]["tissue_density"]) / tissue_ref, 3)
        results["pairs"][unit] = row

    for unit, (side, _) in common.LABELLED.items():
        items = common.labelled_items(unit)
        table = results["labelled"].setdefault(unit, {})
        for detector in common.DETECTORS:
            tp = n_t = n_p = 0
            images = 0
            for name, _, inst in items:
                path = common.LABELS / detector / unit / f"{side}_{name}.npz"
                if not path.exists():
                    continue
                a, b, c = _f1(np.load(path)["labels"], np.load(inst)["labels"])
                tp, n_t, n_p, images = tp + a, n_t + b, n_p + c, images + 1
            if images:
                table[detector] = {"images": images, "f1": round(2 * tp / max(1, n_t + n_p), 3),
                                   "count_ratio": round(n_p / max(1, n_t), 2),
                                   "labelled": n_t, "found": n_p}
    return results


def report(results: dict) -> str:
    lines = ["# P-03 overnight benchmark", "",
             f"Generated {__import__('time').strftime('%Y-%m-%d %H:%M')}. Updated after every stage.", "",
             "## Missing nuclei on our slides", "",
             "Shortfall = how far the IHC count falls below the matched H&E reference "
             "(InstanSeg on RGB, P-21). Target: under ~20%. Negative = more than the H&E.", ""]
    totals: dict[str, list[float]] = {d: [] for d in common.DETECTORS}
    first, second = common.DETECTORS[:8], ["baseline_h", "cellpose_nuclei"] + common.DETECTORS[8:]
    for title, detectors in (("First night", first), ("Second run (6 Oct)", second)):
        lines += [f"**{title}**", "", "| Pair | H&E ref /mm2 | " + " | ".join(detectors) + " |",
                  "|" + "---|" * (len(detectors) + 2)]
        for unit, row in results["pairs"].items():
            ref = (row.get("he_reference") or {}).get("density")
            cells = []
            for detector in detectors:
                entry = row.get(detector)
                if entry and entry.get("shortfall") is not None:
                    if title.startswith("First") or detector in common.DETECTORS[8:]:
                        totals[detector].append(abs(entry["shortfall"]))
                    cells.append(f"{entry['density']:,.0f} ({entry['shortfall']:+.0%})")
                else:
                    cells.append("–")
            reference = f"{ref:,.0f}" if ref else "–"
            lines.append(f"| {unit} | {reference} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["", "DeepLIIF is scored on a subset - 8 of each pair's 24 fields and 20 of the 76 "
              "LyNSeC images - because it runs ~110 s a field on this CPU; its rows are "
              "indicative, not directly comparable to the full-sample ones. lynsec_hovernet ran at the "
              "same speed: all 24 fields on the first four pairs, 8 on the rest.",
              "", "Mean |shortfall| per detector (pairs scored):", ""]
    for detector, values in totals.items():
        if values:
            within = sum(1 for v in values if v <= 0.2)
            lines.append(f"- **{detector}**: {np.mean(values):.0%} over {len(values)} pairs; "
                         f"{within} within 20%")
    lines += ["", "## Per tissue area (second run)", "",
              "The same counts divided by the tissue in each side's fields instead of the whole "
              "field, so a field that sits on glass stops reading as missed nuclei. Shown: "
              "tissue share of the IHC / H&E fields, then shortfall per tissue area.", "",
              "| Pair | Tissue IHC / H&E | " + " | ".join(common.DETECTORS) + " |",
              "|" + "---|" * (len(common.DETECTORS) + 2)]
    tissue_totals: dict[str, list[float]] = {d: [] for d in common.DETECTORS}
    for unit, row in results["pairs"].items():
        cells = []
        for detector in common.DETECTORS:
            value = (row.get(detector) or {}).get("tissue_shortfall")
            if value is None:
                cells.append("–")
            else:
                tissue_totals[detector].append(abs(value))
                cells.append(f"{value:+.0%}")
        shares = f"{row.get('tissue_share_ihc', 0):.0%} / {row.get('tissue_share_he', 0):.0%}"
        lines.append(f"| {unit} | {shares} | " + " | ".join(cells) + " |")
    lines += ["", "Mean |shortfall| per tissue area:", ""]
    for detector, values in tissue_totals.items():
        if values:
            within = sum(1 for v in values if v <= 0.2)
            lines.append(f"- **{detector}**: {np.mean(values):.0%} over {len(values)} pairs; {within} within 20%")
    for unit, (_, what) in common.LABELLED.items():
        lines += ["", f"## Against labelled IHC nuclei: {what}, held out", "",
                  "| Detector | Images | F1 @ IoU 0.5 | Found / labelled |", "|---|---|---|---|"]
        for detector, entry in (results["labelled"].get(unit) or {}).items():
            lines.append(f"| {detector} | {entry['images']} | {entry['f1']:.3f} | {entry['count_ratio']:.2f} |")
    lines += ["", "Read the tables together: a detector can close the density gap by "
              "over-splitting, which shows up as a found/labelled ratio well above 1 and a low F1.", "",
              "Second-run notes: BC-DeepLIIF labels each cell as its mask from immunofluorescence, "
              "outline ring included, so its F1 punishes a detector that draws a tighter nucleus; the "
              "found/labelled ratio is the steadier number there. DeepLIIF was built on BC-DeepLIIF, so "
              "its row on that set is an upper bound, not a fair comparison. cellpose_selftrained "
              "learnt from other fields of the same ten sections it is scored on. Contact sheets to "
              "look at: `visual/`.", ""]
    stages = common.read_json(common.STATE / "night.json", {}) or {}
    lines += ["## Stages", "", "| Stage | State | Note |", "|---|---|---|"]
    for name, entry in (stages.get("stages") or {}).items():
        lines.append(f"| {name} | {entry.get('state')} | {str(entry.get('note', ''))[:120]} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    common.ensure_dirs()
    results = score()
    common.write_json(common.OUT / "results.json", results)
    (common.OUT / "REPORT.md").write_text(report(results), encoding="utf-8")
    common.say("scored", LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
