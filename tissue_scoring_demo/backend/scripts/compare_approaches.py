"""Score one case every way we have considered, and judge them against the readers.

    python scripts/compare_approaches.py CAN_00270

Everything here runs over step 14's stored per-cell rows, so a whole sweep of
variants costs seconds and opens no slide. That is the point of step 14 writing
rows rather than a summary: the scoring question can be re-asked without
re-measuring.

**Two tests, and neither of them is "it moved closer".** With one case there are
five (marker, value) pairs per number, which admits exactly two honest
judgements:

*Ordering.* Does the variant rank the five markers as the readers did? Spearman
over five points; a perfect ordering has a 1-in-120 chance of arising by luck,
so a clean result is evidence and a null is informative.

*Residual after fitting.* Fit the variant's free parameters to this case and see
what error remains. A variant that cannot match even when fitted is wrong in
structure rather than mis-tuned. This rejects variants; it never selects one.

A variant that merely nudges one marker toward its target is not thereby better,
and this script prints no ranking that would suggest otherwise - it prints the
numbers and leaves the judgement where it belongs.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from app import panel  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.scoring import cuts as cut_points  # noqa: E402
from app.services.case_score_service import case_score_service  # noqa: E402
from app.services.validation_service import (  # noqa: E402
    _normalise_case,
    _same_case,
    validation_service,
)

PERCENT_TOLERANCE = 10.0


def round5(value: float) -> int:
    return int(math.floor(value / 5 + 0.5) * 5)


def load(case_id: str) -> dict:
    """Per-marker rows, region areas, and the readers' consensus."""
    readers = validation_service.readers()
    key = _normalise_case(case_id)
    out: dict[str, dict] = {}

    for letter in panel.SCORED_MARKERS:
        found = case_score_service.pairs(case_id).get(letter)
        if not found:
            continue
        he, ihc = found
        directory = settings.per_cell_dir / f"{he}__{ihc}"
        if not (directory / "report.json").is_file():
            continue

        rows: list[dict] = []
        for path in sorted(directory.glob("region*/cells.json")):
            rows.extend(json.loads(path.read_text(encoding="utf-8")).get("cells", []))
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        areas = {int(r["rank"]): float(r["areaMm2"]) for r in report["regions"]}

        cases = readers.get(letter, {})
        match = next((k for k in cases if _same_case(key, k)), None)
        if match is None:
            continue
        readings = cases[match]
        intensities = [r[1] for r in readings if r[1] is not None]

        out[letter] = {
            "rows": rows,
            "areas": areas,
            "their_percent": float(np.mean([r[0] for r in readings])),
            "percent_spread": max(r[0] for r in readings) - min(r[0] for r in readings),
            "their_intensity": float(np.mean(intensities)) if intensities else None,
            "intensity_spread": (
                max(intensities) - min(intensities) if intensities else None
            ),
        }
    return out


def by_region(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["regionRank"]), []).append(row)
    return grouped


def positives(cells: list[dict], od_cut: float, second_min: float) -> list[dict]:
    return [
        cell
        for cell in cells
        if float(cell["intensityOd"]) >= od_cut and float(cell["second"]) >= second_min
    ]


# --- percent variants -------------------------------------------------------


def percent(data: dict, od_cut: float, second_min: float, *, how: str) -> float:
    """Percent positive under one region-combining rule."""
    grouped = by_region(data["rows"])
    areas = data["areas"]

    per_region, weights, counts = [], [], []
    for rank, cells in grouped.items():
        per_region.append(100.0 * len(positives(cells, od_cut, second_min)) / len(cells))
        weights.append(areas.get(rank, 0.0))
        counts.append(len(cells))

    if not per_region:
        return 0.0

    if how == "pooled":
        every = data["rows"]
        return 100.0 * len(positives(every, od_cut, second_min)) / len(every)
    if how == "plain":
        return float(np.mean(per_region))
    if how == "cells":
        total = sum(counts) or 1
        return sum(p * c for p, c in zip(per_region, counts, strict=True)) / total
    if how == "dominant":
        # The largest region alone - SOP 1.4's reading, where the block is chosen
        # for its most invasive tumour content.
        biggest = max(grouped, key=lambda rank: areas.get(rank, 0.0))
        cells = grouped[biggest]
        return 100.0 * len(positives(cells, od_cut, second_min)) / len(cells)
    # area-weighted, the default
    total = sum(weights)
    if total <= 0:
        return float(np.mean(per_region))
    return sum(p * w for p, w in zip(per_region, weights, strict=True)) / total


# --- intensity variants -----------------------------------------------------


def intensity_od(data: dict, od_cut: float, second_min: float, *, statistic: str) -> float:
    """Density of the positive cells, area-weighted across regions."""
    grouped = by_region(data["rows"])
    areas = data["areas"]

    values, weights = [], []
    for rank, cells in grouped.items():
        chosen = positives(cells, od_cut, second_min)
        if not chosen:
            continue
        density = np.array([float(c["intensityOd"]) for c in chosen])
        if statistic == "median":
            value = float(np.median(density))
        elif statistic == "p75":
            value = float(np.percentile(density, 75))
        elif statistic == "p90":
            value = float(np.percentile(density, 90))
        else:
            value = float(density.mean())
        values.append(value)
        weights.append(areas.get(rank, 0.0))

    if not values:
        return 0.0
    total = sum(weights)
    if total <= 0:
        return float(np.mean(values))
    return sum(v * w for v, w in zip(values, weights, strict=True)) / total


def band_linear(od: float, lo: float, hi: float) -> float:
    """Map `od` onto the permitted bands by position between `lo` and `hi`.

    The rescaling variant: instead of fixed optical-density breakpoints, place
    the observed range across the reporting scale. It has two free parameters
    rather than five, and it is the operation that answers "can the pipeline
    reach 1.75 and 2.0 at all" without fitting each band separately.
    """
    bands = list(cut_points.PERMITTED_BANDS)
    if hi <= lo:
        return bands[0]
    position = (od - lo) / (hi - lo)
    index = int(round(position * (len(bands) - 1)))
    return bands[max(0, min(len(bands) - 1, index))]


def judge(name: str, ours: dict[str, float], theirs: dict[str, float], spreads: dict) -> dict:
    letters = [k for k in ours if k in theirs and theirs[k] is not None]
    if len(letters) < 2:
        return {"name": name, "n": len(letters)}
    mine = np.array([ours[k] for k in letters], dtype=float)
    yours = np.array([theirs[k] for k in letters], dtype=float)
    errors = mine - yours
    inside = sum(
        1
        for k, e in zip(letters, errors, strict=True)
        if spreads.get(k) is not None and abs(e) <= max(spreads[k], 1e-9)
    )
    return {
        "name": name,
        "n": len(letters),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "rho": float(spearmanr(mine, yours).statistic) if len(letters) > 2 else float("nan"),
        "inside": inside,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    args = parser.parse_args()

    data = load(args.case_id)
    if not data:
        print("no scored marker with a matching reader row; run the pipeline first")
        return 1

    letters = list(data)
    their_percent = {k: data[k]["their_percent"] for k in letters}
    their_intensity = {k: data[k]["their_intensity"] for k in letters}
    percent_spread = {k: data[k]["percent_spread"] for k in letters}
    intensity_spread = {k: data[k]["intensity_spread"] for k in letters}
    shipped = {k: cut_points.for_marker(k) for k in letters}

    control_path = settings.scores_dir / f"{args.case_id}_internal_control.json"
    control = (
        json.loads(control_path.read_text(encoding="utf-8"))
        if control_path.is_file()
        else {}
    )

    print(f"{args.case_id}: {len(letters)} marker(s) with reader data\n")

    # --- percent ------------------------------------------------------------
    print("=" * 78)
    print("PERCENT POSITIVE  (tolerance = each marker's own reader spread)")
    print("=" * 78)
    print(f"{'variant':<34} {'MAE':>7} {'bias':>7} {'rho':>6} {'inside':>7}")

    results = []
    for how, label in (
        ("area", "P0 area-weighted (shipped)"),
        ("pooled", "P3 pooled over all cells"),
        ("plain", "P3 plain mean of regions"),
        ("cells", "P3 cell-count weighted"),
        ("dominant", "P1 dominant region only"),
    ):
        ours = {
            k: float(
                round5(percent(data[k], shipped[k].positivity_od, shipped[k].second_min, how=how))
            )
            for k in letters
        }
        verdict = judge(label, ours, their_percent, percent_spread)
        results.append((verdict, ours))
        if "mae" not in verdict:
            # Fewer than two markers have reader data, so there is nothing to
            # average and nothing to rank. Say so rather than printing zeros.
            print(f"{label:<34} {'too few markers to judge':>30}")
            continue
        print(
            f"{label:<34} {verdict['mae']:>7.1f} {verdict['bias']:>+7.1f} "
            f"{verdict['rho']:>6.2f} {verdict['inside']:>3}/{verdict['n']}"
        )

    print()
    print("  per marker, shipped cuts:")
    header = "  " + " ".join(f"{k:>7}" for k in letters)
    print(header + "     (readers)")
    for verdict, ours in results:
        line = "  " + " ".join(f"{ours.get(k, float('nan')):>7.0f}" for k in letters)
        print(f"{line}   {verdict['name']}")
    print("  " + " ".join(f"{their_percent[k]:>7.1f}" for k in letters) + "   READERS")

    # --- intensity ----------------------------------------------------------
    print()
    print("=" * 78)
    print("INTENSITY  (tolerance = each marker's own reader spread)")
    print("=" * 78)
    print(f"{'variant':<34} {'MAE':>7} {'bias':>7} {'rho':>6} {'inside':>7}")

    raw = {
        k: intensity_od(data[k], shipped[k].positivity_od, shipped[k].second_min, statistic="mean")
        for k in letters
    }

    variants: list[tuple[str, dict[str, float]]] = []

    for statistic in ("mean", "median", "p75", "p90"):
        densities = {
            k: intensity_od(
                data[k], shipped[k].positivity_od, shipped[k].second_min, statistic=statistic
            )
            for k in letters
        }
        banded = {k: shipped[k].band(densities[k]) for k in letters}
        variants.append((f"I0/I3 {statistic}, shipped bands", banded))

    lo, hi = min(raw.values()), max(raw.values())
    variants.append(
        ("I1 mean, bands rescaled to range", {k: band_linear(raw[k], lo, hi) for k in letters})
    )

    if control:
        ratios = {}
        for k in letters:
            reference = float(control.get(k, {}).get("control_mean_od", 0.0))
            ratios[k] = raw[k] / reference if reference > 1e-6 else 0.0
        if any(ratios.values()):
            rlo, rhi = min(ratios.values()), max(ratios.values())
            variants.append(
                (
                    "I2 tumour/control, rescaled",
                    {k: band_linear(ratios[k], rlo, rhi) for k in letters},
                )
            )
            print()
            print("  internal control (SOP 5.7) measured:")
            for k in letters:
                reference = float(control.get(k, {}).get("control_mean_od", 0.0))
                print(
                    f"    {k}: tumour {raw[k]:.4f} OD / control {reference:.4f} OD "
                    f"= {ratios[k]:.3f}"
                )
            print()
    else:
        print("\n  (no internal-control measurement on disk - run "
              "measure_internal_control.py)\n")

    for label, banded in variants:
        verdict = judge(label, banded, their_intensity, intensity_spread)
        if "mae" not in verdict:
            print(f"{label:<34} {'too few markers to judge':>30}")
            continue
        print(
            f"{label:<34} {verdict['mae']:>7.3f} {verdict['bias']:>+7.3f} "
            f"{verdict['rho']:>6.2f} {verdict['inside']:>3}/{verdict['n']}"
        )

    print()
    print("  rho over five markers: +1.00 is a perfect ordering (p about 0.008),")
    print("  and anything under about +0.60 is indistinguishable from chance here.")
    print()
    print("  The raw densities this pipeline produces span "
          f"{lo:.3f} to {hi:.3f} OD.")
    table = cut_points.cut_set().for_marker(letters[0]).od_to_band
    reach = [band for ceiling, band in table if ceiling is None or ceiling > hi]
    print(f"  The shipped band table needs {table[-2][0]:.2f} OD for 1.75 and "
          f"{table[-1][0] if table[-1][0] else 0.80:.2f} for 2.0, so bands "
          f"{sorted(set(reach))} are effectively unreachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
