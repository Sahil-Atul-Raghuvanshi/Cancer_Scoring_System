"""Fit each antibody's positivity cut against the pathologists' readings.

    python scripts/calibrate_marker_cuts.py                 # report only
    python scripts/calibrate_marker_cuts.py --write v2      # write config/marker_cuts.v2.json

**One parameter per marker, fitted on the stored per-cell rows.** Step 14 has
already measured every cell; the cut decides which of them count. So the fit is
a one-dimensional sweep over a number, not a training run, and it can be checked
by hand from the rows it reads.

**Why a fit is legitimate here and overfitting is the whole risk.** The guide's
own instruction is to run steps 1-14 on the scored cases and choose each marker's
cut points to minimise disagreement with that marker's reader consensus, holding
cases out and reporting the fit and the held-out result separately. That is what
this does. What makes it calibration rather than curve-fitting is the holding
out - and with a single scored case there is nothing to hold out, so this script
**refuses to write** below `--min-cases` and says why.

That refusal is the point of the script as much as the fit is. On one case a
five-parameter fit reproduces the readers almost exactly and has demonstrated
nothing, because a one-dimensional sweep against a single target always can.

**It fits the cut and the band table together, because they are one
measurement.** The intensity is the mean density of the cells the *cut* admitted,
so raising the cut raises the intensity: the two reported numbers are coupled
through a single parameter, and fitting one while holding the other fixed would
be calibrating half a model. The cut is fitted against the reader percentages and
the band breakpoints are then placed on whatever density distribution that cut
produces.

**Neither is written below `--min-cases`,** and that is the important half. A
one-dimensional sweep against a single target always reproduces it.

What this does *not* attempt is the mechanism OncoStem actually describe. SOP 5.7
grades intensity against the internal control ducts on the same slide, which is a
different measurement - `measure_internal_control.py` implements it and
`SCORING_APPROACHES.md` F3 records that it does not yet work well enough to use.
An absolute band table is therefore a stand-in for a relative judgement, and no
amount of fitting makes it the same quantity.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app import panel  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.scoring import cuts as cut_points  # noqa: E402
from app.services.case_score_service import case_score_service  # noqa: E402
from app.services.validation_service import _same_case, validation_service  # noqa: E402

#: Optical densities swept, in steps of 0.001. Wide enough to cover a slide
#: whose DAB never clears the noise floor and one that is saturated throughout.
SWEEP = np.arange(0.02, 1.20, 0.001)


def round5(value: float) -> int:
    return int(math.floor(value / 5 + 0.5) * 5)


def _rows_and_areas(he: str, ihc: str):
    directory = settings.per_cell_dir / f"{he}__{ihc}"
    rows: list[dict] = []
    for path in sorted(directory.glob("region*/cells.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")).get("cells", []))
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    areas = {int(r["rank"]): float(r["areaMm2"]) for r in report["regions"]}
    return rows, areas


def percent_at(rows: list[dict], areas: dict[int, float], od_cut: float, second_min: float):
    """Area-weighted percent positive, and the positive cells' mean density."""
    by: dict[int, list[dict]] = {}
    for row in rows:
        by.setdefault(int(row["regionRank"]), []).append(row)

    percents, intensities, weights = [], [], []
    for rank, cells in by.items():
        positive = [
            cell
            for cell in cells
            if float(cell["intensityOd"]) >= od_cut and float(cell["second"]) >= second_min
        ]
        percents.append(100.0 * len(positive) / len(cells))
        intensities.append(
            sum(float(c["intensityOd"]) for c in positive) / len(positive)
            if positive
            else 0.0
        )
        weights.append(areas.get(rank, 0.0))

    total = sum(weights) or float(len(percents))
    if sum(weights) <= 0:
        weights = [1.0] * len(percents)
        total = float(len(percents))

    percent = sum(p * w for p, w in zip(percents, weights, strict=True)) / total
    stained = [(i, w) for i, w in zip(intensities, weights, strict=True) if i > 0]
    intensity = (
        sum(i * w for i, w in stained) / sum(w for _, w in stained) if stained else 0.0
    )
    return percent, intensity


def gather(case_ids: list[str]) -> dict:
    """`{marker: {case: (rows, areas, reader_percent, reader_intensity)}}`."""
    readers = validation_service.readers()
    out: dict[str, dict[str, tuple]] = {}

    for case_id in case_ids:
        for letter, (he, ihc) in case_score_service.pairs(case_id).items():
            directory = settings.per_cell_dir / f"{he}__{ihc}"
            if not (directory / "report.json").is_file():
                continue
            cases = readers.get(letter, {})
            match = next(
                (key for key in cases if _same_case(case_id.replace("_", ""), key)), None
            )
            if match is None:
                continue
            readings = cases[match]
            percent = float(np.mean([r[0] for r in readings]))
            intensities = [r[1] for r in readings if r[1] is not None]
            intensity = float(np.mean(intensities)) if intensities else None

            rows, areas = _rows_and_areas(he, ihc)
            out.setdefault(letter, {})[case_id] = (rows, areas, percent, intensity)
    return out


def fit_bands(cases: dict, od_cut: float, second_min: float) -> list[float] | None:
    """Breakpoints placing the observed densities onto the reported bands.

    Run *after* the cut is fitted, on the densities that cut produces, because
    the intensity is the mean of the cells the cut admitted - move the cut and
    every density here moves with it.

    Each case contributes one (density, reader band) pair per marker, and the
    breakpoints are the midpoints between adjacent bands' mean densities. That is
    deliberately the least clever thing that could work: with a handful of cases
    there is not enough signal to fit six free breakpoints, and a midpoint rule
    has none - it is determined entirely by where the observed groups sit.
    """
    bands = list(cut_points.PERMITTED_BANDS)
    seen: dict[float, list[float]] = {}

    for rows, areas, _target, intensity in cases.values():
        if intensity is None:
            continue
        # The reader value is a mean of four bands and need not be one itself;
        # attribute it to the nearest permitted band.
        nearest = min(bands, key=lambda band: abs(band - intensity))
        _percent, density = percent_at(rows, areas, od_cut, second_min)
        if density > 0:
            seen.setdefault(nearest, []).append(density)

    if len(seen) < 2:
        return None

    centres = {band: float(np.mean(values)) for band, values in sorted(seen.items())}
    ordered = sorted(centres)
    breakpoints: list[float] = []
    for lower, upper in zip(ordered, ordered[1:], strict=False):
        breakpoints.append(round((centres[lower] + centres[upper]) / 2.0, 4))
    return breakpoints


def fit(marker: str, cases: dict, second_min: float, exclude: str | None = None):
    """The cut minimising mean absolute error against the readers' consensus.

    **The middle of the best plateau, not its first point.** Reported percentages
    are multiples of 5, so the error is a step function and many neighbouring
    cuts score identically. Taking the first cut that achieves the minimum is an
    artefact of sweeping upward: on CD44 it returns 0.020, which is beneath the
    optical-density noise floor of clean glass and only "works" because every
    cut from there to 0.14 gives the same rounded answer. The middle of the
    plateau is the cut furthest from the edges where the answer changes, which is
    the one most likely to survive being shown a second case.
    """
    used = {case: data for case, data in cases.items() if case != exclude}
    if not used:
        return None, None

    errors: list[float] = []
    for od in SWEEP:
        per_case = []
        for rows, areas, target, _ in used.values():
            percent, _ = percent_at(rows, areas, float(od), second_min)
            per_case.append(abs(round5(percent) - target))
        errors.append(float(np.mean(per_case)))

    best_error = min(errors)
    plateau = [float(od) for od, error in zip(SWEEP, errors, strict=True) if error == best_error]
    return float(np.median(plateau)), best_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="", help="comma-separated; default all scored")
    parser.add_argument(
        "--min-cases",
        type=int,
        default=4,
        help="refuse to write a fit below this many cases per marker (default 4)",
    )
    parser.add_argument("--write", default="", help="version to write, e.g. v2")
    args = parser.parse_args()

    if args.cases:
        case_ids = [value.strip() for value in args.cases.split(",") if value.strip()]
    else:
        directory = settings.data_dir / "cases"
        case_ids = sorted({p.stem.rsplit("_", 1)[0] for p in directory.glob("*.json")})

    data = gather(case_ids)
    if not data:
        print("no scored case has a matching row in the reader sheet; nothing to fit")
        return 1

    current = cut_points.cut_set()
    print(f"cases scored and matched: {sorted({c for m in data.values() for c in m})}")
    print()
    print(f"{'mk':3} {'cases':>5} {'current':>8} {'fitted':>8} {'MAE@fit':>8} "
          f"{'MAE@current':>11}  held-out")
    print()

    fitted: dict[str, float] = {}
    enough = True
    for letter in panel.SCORED_MARKERS:
        cases = data.get(letter, {})
        if not cases:
            print(f"{letter:3} {'-':>5}  no reader data")
            continue

        marker = current.for_marker(letter)
        cut, error = fit(letter, cases, marker.second_min)
        if cut is None:
            continue
        fitted[letter] = cut

        baseline = float(
            np.mean(
                [
                    abs(
                        round5(
                            percent_at(
                                rows, areas, marker.positivity_od, marker.second_min
                            )[0]
                        )
                        - target
                    )
                    for rows, areas, target, _ in cases.values()
                ]
            )
        )

        # Leave-one-case-out: fit without a case, then score that case. With one
        # case this is empty, and an empty held-out result is reported as such
        # rather than as a pass.
        held: list[float] = []
        if len(cases) > 1:
            for case in cases:
                out_cut, _ = fit(letter, cases, marker.second_min, exclude=case)
                if out_cut is None:
                    continue
                rows, areas, target, _ = cases[case]
                percent, _ = percent_at(rows, areas, out_cut, marker.second_min)
                held.append(abs(round5(percent) - target))

        held_text = f"{np.mean(held):.2f} pts over {len(held)}" if held else "NONE (n=1)"
        if len(cases) < args.min_cases:
            enough = False
        print(
            f"{letter:3} {len(cases):>5} {marker.positivity_od:>8.3f} {cut:>8.3f} "
            f"{error:>8.2f} {baseline:>11.2f}  {held_text}"
        )

    print()
    if not enough:
        print(
            f"FEWER THAN {args.min_cases} CASES for at least one marker. A one-dimensional\n"
            "sweep against a single target always reproduces it, so a fit at this size\n"
            "measures nothing and the held-out column above is empty. The fitted cuts are\n"
            "printed as a diagnostic - they are NOT written."
        )
        if args.write:
            print("--write refused for that reason.")
        return 2

    if not args.write:
        print("report only; pass --write v2 to write the fitted cuts")
        return 0

    payload = json.loads(cut_points.CONFIG_PATH.read_text(encoding="utf-8"))
    version = int(args.write.lstrip("vV"))
    payload["version"] = version
    payload["status"] = "calibrated"
    payload["anchored_on"] = sorted({c for m in data.values() for c in m})
    payload["provenance"] = (
        f"Positivity cuts fitted against the pathologist consensus on "
        f"{len(payload['anchored_on'])} case(s) by scripts/calibrate_marker_cuts.py, "
        "with each marker's band breakpoints placed afterwards on the densities that "
        "cut produces. The two upper optical-density cuts are unchanged. Note that an "
        "absolute band table is a stand-in for the relative judgement SOP 5.7 "
        "describes - see SCORING_APPROACHES.md, finding F3."
    )
    for letter, cut in fitted.items():
        entry = payload["markers"][letter]
        entry["od"] = [round(cut, 3), *entry["od"][1:]]

        breakpoints = fit_bands(
            data.get(letter, {}), cut, current.for_marker(letter).second_min
        )
        if breakpoints:
            entry["od_to_band_note"] = (
                "Breakpoints fitted alongside the cut, on the densities that cut "
                "produces. Intensity and percentage are one measurement."
            )
            entry["od_to_band"] = breakpoints

    destination = cut_points.CONFIG_PATH.with_name(f"marker_cuts.v{version}.json")
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {destination}")
    print("Point app/scoring/cuts.py:CONFIG_PATH at it to make it the active set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
