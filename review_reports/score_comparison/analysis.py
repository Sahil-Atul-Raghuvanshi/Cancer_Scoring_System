"""Deeper analysis of both AI score files against the pathologists.

Run with the backend's Python from tissue_scoring_demo/backend. Writes analysis.json.

1. Reproduce each slide's score from the stored per-cell rows (proves we read the right data).
2. Leave-one-case-out refit of each marker's positivity cut - the pipeline's own calibration,
   pointed at v<N>_data/data/history, fitted on five cases and tested on the sixth.
3. Leave-one-case-out ML recalibration from slide-level AI features, against a no-AI baseline.
4. Agreement statistics: Spearman, Lin's concordance, Bland-Altman.
5. Reliability: the warning flags each score carries, and whether flagged slides err more.
"""
import csv, importlib.util, json, pathlib, re, statistics as st, sys
import numpy as np
from scipy import stats as sps
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# The workspace root, from this file's own place - review_reports/<review>/<script>.py
# - rather than one machine's path (P-20).
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(REPO))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402
BACKEND = REPO / "tissue_scoring_demo" / "backend"
#: Intermediate files (tables, images, charts) live under v<N>_data/data/, not beside the code.
HERE = data_versions.data_root() / "review_reports" / "score_comparison"
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BACKEND))
spec = importlib.util.spec_from_file_location("calib", BACKEND / "scripts" / "calibrate_marker_cuts.py")
calib = importlib.util.module_from_spec(spec); spec.loader.exec_module(calib)
from app.scoring import cuts as cut_points  # noqa: E402

MARKERS = ["A", "F", "R", "U", "W"]
MODES = {"pixel": "oncostem_ai_scores.csv", "tile": "oncostem_ai_scores_tiles.csv"}
readers = {(r["case"], r["marker"]): r for r in json.loads((HERE / "rows.json").read_text(encoding="utf-8"))}
CASES = sorted({k[0] for k in readers})

# --- load both files and the per-cell rows behind every score --------------------------------
data = {}
for mode, name in MODES.items():
    with open(data_versions.results_root() / name, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["case_id"], r["marker"], mode)
            d = data_versions.data_root() / "history" / r["case_id"] / "markers" / r["marker"] / "per_cell" / f"{r['he_upload_id']}__{r['ihc_upload_id']}"
            rows = []
            for p in sorted(d.glob("region*/cells.json")):
                rows.extend(json.loads(p.read_text(encoding="utf-8")).get("cells", []))
            rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
            areas = {int(x["rank"]): float(x["areaMm2"]) for x in rep["regions"]}
            data[key] = {"csv": r, "rows": rows, "areas": areas}

# 1. reproduction check at the provisional cut
repro = []
for (case, m, mode), v in data.items():
    c = cut_points.for_marker(m)
    pct, dens = calib.percent_at(v["rows"], v["areas"], c.positivity_od, c.second_min) if v["rows"] else (0.0, 0.0)
    repro.append(abs(pct - float(v["csv"]["percent_raw"] or 0)))
print(f"reproduction: max |recomputed - percent_raw| = {max(repro):.3f} over {len(repro)} slides")

# 2. leave-one-case-out cut refit, per marker and mode
def band_model(train, od_cut, second_min):
    """Mirror calib.fit_bands, but also return the ordered bands so a held-out density can be mapped."""
    seen = {}
    for rows, areas, _t, inten in train.values():
        nearest = min(cut_points.PERMITTED_BANDS, key=lambda b: abs(b - inten))
        _p, dens = calib.percent_at(rows, areas, od_cut, second_min)
        if dens > 0:
            seen.setdefault(nearest, []).append(dens)
    if not seen:
        return None
    centres = {b: float(np.mean(v)) for b, v in sorted(seen.items())}
    ordered = sorted(centres)
    brk = [(centres[a] + centres[b]) / 2 for a, b in zip(ordered, ordered[1:])]
    return ordered, brk

def to_band(model, dens):
    if model is None or dens <= 0:
        return 0.0
    ordered, brk = model
    return float(ordered[sum(dens >= x for x in brk)])

refit = {}
for mode in MODES:
    for m in MARKERS:
        second = cut_points.for_marker(m).second_min
        cases = {c: (data[(c, m, mode)]["rows"], data[(c, m, mode)]["areas"], readers[(c, m)]["readerPctMean"], readers[(c, m)]["readerIntMean"])
                 for c in CASES if data[(c, m, mode)]["rows"]}
        out = {}
        for held in CASES:
            if held not in cases:
                out[held] = None; continue
            cut, _err = calib.fit(m, cases, second, exclude=held)
            train = {c: v for c, v in cases.items() if c != held}
            rows, areas, _t, _i = cases[held]
            pct, dens = calib.percent_at(rows, areas, cut, second)
            out[held] = {"cut": round(cut, 3), "pct": calib.round5(pct), "int": to_band(band_model(train, cut, second), dens)}
        full_cut, full_err = calib.fit(m, cases, second)
        refit[f"{mode}:{m}"] = {"heldOut": out, "cutAllCases": round(full_cut, 3), "inSampleMae": round(full_err, 2),
                                "provisionalCut": cut_points.for_marker(m).positivity_od}

# 3. ML recalibration from slide-level AI features, leave-one-case-out
def feats(case, m):
    x = []
    for mode in MODES:
        r = data[(case, m, mode)]["csv"]
        x += [float(r["percent_raw"] or 0), float(r["intensity_raw"] or 0), float(r["h_score"] or 0), np.log1p(float(r["cells"] or 0))]
    return x + [1.0 if m == k else 0.0 for k in MARKERS]
X = {(c, m): feats(c, m) for c in CASES for m in MARKERS}
ml = {"ridge": {}, "baseline": {}}
for target in ("pct", "int"):
    tkey = "readerPctMean" if target == "pct" else "readerIntMean"
    for held in CASES:
        tr = [(c, m) for c in CASES for m in MARKERS if c != held]
        model = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit([X[k] for k in tr], [readers[k][tkey] for k in tr])
        for m in MARKERS:
            ml["ridge"][f"{target}:{held}:{m}"] = float(model.predict([X[(held, m)]])[0])
            # no-AI baseline: this marker's mean reader score on the other five cases
            ml["baseline"][f"{target}:{held}:{m}"] = float(np.mean([readers[(c, m)][tkey] for c in CASES if c != held]))

# --- collect every prediction on one scoreboard -------------------------------------------------
def err_table(pred_of):
    """pred_of(case, marker) -> (pct, int) or None. Returns overall and per-marker agreement."""
    res = {"pct": [], "int": [], "inPct": 0, "inInt": 0, "n": 0, "byMarker": {}}
    for m in MARKERS:
        mp, mi = [], []
        for c in CASES:
            p = pred_of(c, m)
            if p is None: continue
            r = readers[(c, m)]
            dp, di = p[0] - r["readerPctMean"], p[1] - r["readerIntMean"]
            res["pct"].append(dp); res["int"].append(di); mp.append(abs(dp)); mi.append(abs(di)); res["n"] += 1
            res["inPct"] += r["readerPctMin"] <= p[0] <= r["readerPctMax"]
            res["inInt"] += r["readerIntMin"] <= p[1] <= r["readerIntMax"]
        res["byMarker"][m] = {"pctMae": round(st.mean(mp), 1), "intMae": round(st.mean(mi), 2)}
    return {"pctMae": round(st.mean(abs(x) for x in res["pct"]), 1), "pctBias": round(st.mean(res["pct"]), 1),
            "intMae": round(st.mean(abs(x) for x in res["int"]), 2), "intBias": round(st.mean(res["int"]), 2),
            "inPct": res["inPct"], "inInt": res["inInt"], "n": res["n"], "byMarker": res["byMarker"]}

scoreboard = {}
for mode in MODES:
    scoreboard[f"{mode}: as delivered"] = err_table(lambda c, m, mode=mode: (float(data[(c, m, mode)]["csv"]["percent"]), float(data[(c, m, mode)]["csv"]["intensity"])))
    scoreboard[f"{mode}: cut refit (held-out)"] = err_table(lambda c, m, mode=mode: None if refit[f"{mode}:{m}"]["heldOut"][c] is None else (refit[f"{mode}:{m}"]["heldOut"][c]["pct"], refit[f"{mode}:{m}"]["heldOut"][c]["int"]))
scoreboard["ML recalibration, both methods (held-out)"] = err_table(lambda c, m: (ml["ridge"][f"pct:{c}:{m}"], ml["ridge"][f"int:{c}:{m}"]))
scoreboard["No-AI baseline (marker's usual reader score)"] = err_table(lambda c, m: (ml["baseline"][f"pct:{c}:{m}"], ml["baseline"][f"int:{c}:{m}"]))

# 4. agreement statistics, as delivered
def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(2 * np.cov(x, y, bias=True)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))
agreement = {}
for mode in MODES:
    ai = [float(data[(c, m, mode)]["csv"]["percent"]) for m in MARKERS for c in CASES]
    rd = [readers[(c, m)]["readerPctMean"] for m in MARKERS for c in CASES]
    diff = np.array(ai) - np.array(rd)
    per_marker_rho = {}
    for m in MARKERS:
        a = [float(data[(c, m, mode)]["csv"]["percent"]) for c in CASES]; b = [readers[(c, m)]["readerPctMean"] for c in CASES]
        per_marker_rho[m] = None if len(set(b)) < 2 else round(float(sps.spearmanr(a, b).statistic), 2)
    agreement[mode] = {"spearman": round(float(sps.spearmanr(ai, rd).statistic), 2), "ccc": round(ccc(ai, rd), 2),
                       "baBias": round(float(diff.mean()), 1), "baLow": round(float(diff.mean() - 1.96 * diff.std(ddof=1)), 1),
                       "baHigh": round(float(diff.mean() + 1.96 * diff.std(ddof=1)), 1), "spearmanByMarker": per_marker_rho}
pix_err = [abs(float(data[(c, m, "pixel")]["csv"]["percent"]) - readers[(c, m)]["readerPctMean"]) for m in MARKERS for c in CASES]
til_err = [abs(float(data[(c, m, "tile")]["csv"]["percent"]) - readers[(c, m)]["readerPctMean"]) for m in MARKERS for c in CASES]
agreement["pixelVsTileWilcoxonP"] = round(float(sps.wilcoxon(pix_err, til_err).pvalue), 3)

# 5. reliability flags
FLAGS = {"NOT A MEASUREMENT": "Too few cells to measure", "THIN DENOMINATOR": "Thin cell count",
         "DENOMINATOR INCOMPLETE": "Nuclei missed on the IHC slide", "ALIGNMENT FAILED": "Registration failed its check",
         "OUTSIDE THE EXPECTED RANGE": "Outside OncoStem's usual range", "PROVISIONAL CUT POINTS": "Provisional cut points"}
flags = {}
for mode in MODES:
    tally = {v: 0 for v in FLAGS.values()}
    flagged_err, clean_err = [], []
    for c in CASES:
        for m in MARKERS:
            r = data[(c, m, mode)]["csv"]
            hit = [label for key, label in FLAGS.items() if key in r["caveats"]]
            for label in hit: tally[label] += 1
            e = abs(float(r["percent"]) - readers[(c, m)]["readerPctMean"])
            serious = [h for h in hit if h in ("Too few cells to measure", "Nuclei missed on the IHC slide", "Registration failed its check")]
            (flagged_err if serious else clean_err).append(e)
    flags[mode] = {"tally": tally, "seriousFlaggedN": len(flagged_err), "seriousFlaggedMae": round(st.mean(flagged_err), 1) if flagged_err else None,
                   "cleanN": len(clean_err), "cleanMae": round(st.mean(clean_err), 1) if clean_err else None}
cells = {mode: {f"{c}:{m}": int(float(data[(c, m, mode)]["csv"]["cells"] or 0)) for c in CASES for m in MARKERS} for mode in MODES}

out = {"reproductionMaxAbs": round(max(repro), 3), "refit": refit, "scoreboard": scoreboard, "agreement": agreement,
       "flags": flags, "cells": cells, "mlPredictions": ml}
(HERE / "analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

for k, v in scoreboard.items():
    print(f"{k:48s} %MAE {v['pctMae']:5.1f} bias {v['pctBias']:+5.1f} in-range {v['inPct']:2d}/{v['n']}  | int MAE {v['intMae']:.2f} bias {v['intBias']:+.2f} in-range {v['inInt']}/{v['n']}")
print(json.dumps(agreement, indent=0))
print(json.dumps(flags, indent=0))
for k, v in refit.items():
    print(k, "provisional", v["provisionalCut"], "fitted all-cases", v["cutAllCases"], "in-sample MAE", v["inSampleMae"], "held-out cuts", [x["cut"] if x else None for x in v["heldOut"].values()])
