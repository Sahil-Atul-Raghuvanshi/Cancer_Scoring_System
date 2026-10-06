"""Score every detector the same way, and write REPORT.md.

Run after each detector stage, so the report fills in through the night. Two measures:

  On our slides   counted nuclei per mm2 over the benchmark IHC fields, counted with step
                  13's own rule (`extract`: the border margin and the minimum nucleus area),
                  against the H&E reference - InstanSeg on RGB over matched H&E fields
                  (P-21). Shortfall = (H&E - IHC) / H&E. The review's target: within ~20%.
  On LyNSeC       F1 at IoU 0.5 against hand-drawn instance labels on held-out IHC images,
                  and the count ratio (found / labelled).

A density alone can be gamed by over-splitting, which is why the labelled F1 sits beside it.
"""

from __future__ import annotations

import sys

import numpy as np

import common

sys.path.insert(0, str(common.BACKEND))
LOG = common.LOGS / "score.log"


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
    results = {"pairs": {}, "lynsec": {}}
    for case, marker in common.PAIRS:
        unit = common.pair_id(case, marker)
        meta = common.read_json(common.FIELDS / unit / "meta.json")
        if not meta:
            continue
        border, min_area = int(meta["border_px"]), float(meta["min_area_um2"])
        row = {"filed_ihc_density": meta.get("filed_ihc_density")}
        for detector in common.DETECTORS + ["he_reference"]:
            source, side = ("instanseg_rgb", "he") if detector == "he_reference" else (detector, "ihc")
            counted, area, areas, fields = 0, 0.0, [], 0
            for entry in meta[side]:
                path = common.LABELS / source / unit / f"{side}_{entry['name']}.npz"
                if not path.exists():
                    continue
                n, a = _count(np.load(path)["labels"], float(entry["mpp"]), border, min_area)
                counted += n
                areas += a
                area += _area_mm2(int(entry["size"]), border, float(entry["mpp"]))
                fields += 1
            if fields:
                row[detector] = {"density": round(counted / area, 1) if area else None,
                                 "median_area_um2": round(float(np.median(areas)), 1) if areas else None,
                                 "fields": fields, "nuclei": counted}
        reference = (row.get("he_reference") or {}).get("density")
        for detector in common.DETECTORS:
            if detector in row and reference and row[detector]["density"] is not None:
                row[detector]["shortfall"] = round((reference - row[detector]["density"]) / reference, 3)
        results["pairs"][unit] = row

    meta = common.read_json(common.LYNSEC / "meta.json")
    if meta:
        for detector in common.DETECTORS:
            tp = n_t = n_p = 0
            images = 0
            for entry in meta["test"]:
                path = common.LABELS / detector / "LYNSEC" / f"lynsec_{entry['name']}.npz"
                if not path.exists():
                    continue
                truth = np.load(common.LYNSEC / "test" / f"{entry['name']}_inst.npz")["labels"]
                a, b, c = _f1(np.load(path)["labels"], truth)
                tp, n_t, n_p, images = tp + a, n_t + b, n_p + c, images + 1
            if images:
                results["lynsec"][detector] = {"images": images, "f1": round(2 * tp / max(1, n_t + n_p), 3),
                                               "count_ratio": round(n_p / max(1, n_t), 2),
                                               "labelled": n_t, "found": n_p}
    return results


def report(results: dict) -> str:
    lines = ["# P-03 overnight benchmark", "",
             f"Generated {__import__('time').strftime('%Y-%m-%d %H:%M')}. Updated after every stage.", "",
             "## Missing nuclei on our slides", "",
             "Shortfall = how far the IHC count falls below the matched H&E reference "
             "(InstanSeg on RGB, P-21). Target: under ~20%. Negative = more than the H&E.", ""]
    header = "| Pair | H&E ref /mm2 | " + " | ".join(common.DETECTORS) + " |"
    lines += [header, "|" + "---|" * (len(common.DETECTORS) + 2)]
    totals: dict[str, list[float]] = {d: [] for d in common.DETECTORS}
    for unit, row in results["pairs"].items():
        ref = (row.get("he_reference") or {}).get("density")
        cells = []
        for detector in common.DETECTORS:
            entry = row.get(detector)
            if entry and entry.get("shortfall") is not None:
                totals[detector].append(abs(entry["shortfall"]))
                cells.append(f"{entry['density']:,.0f} ({entry['shortfall']:+.0%})")
            else:
                cells.append("–")
        reference = f"{ref:,.0f}" if ref else "–"
        lines.append(f"| {unit} | {reference} | " + " | ".join(cells) + " |")
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
    lines += ["", "## Against hand-labelled IHC nuclei (LyNSeC, held out)", "",
              "| Detector | Images | F1 @ IoU 0.5 | Found / labelled |", "|---|---|---|---|"]
    for detector, entry in results["lynsec"].items():
        lines.append(f"| {detector} | {entry['images']} | {entry['f1']:.3f} | {entry['count_ratio']:.2f} |")
    lines += ["", "Read the two tables together: a detector can close the density gap by "
              "over-splitting, which shows up as a found/labelled ratio well above 1 and a low F1.", ""]
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
