"""Evidence for the "Things that need attention" report. Run with the backend's Python.

Everything is measured on this cohort's stored runs, not quoted from elsewhere:
  - the measurement funnel per slide (tissue -> invasive -> outlined -> sampled -> nuclei -> tumour cells)
  - region selection (step 10): candidates, selected, dropped, chosen by a person or by default
  - pooled vs area-weighted vs plain-mean percent, each scored against the pathologists
  - pixel-wise vs tile-based: how much the outline alone moves the score
  - cell count vs error; nuclei shortfall vs how much of the tumour is stained
  - sensitivity of the percent to the ring-completeness / stained-fraction and density cuts
  - the full agreement set (MAE, median, RMSE, r, within +/-5/10/20, weighted kappa on intensity)
"""
import csv, importlib.util, json, math, pathlib, statistics as st, sys
import numpy as np
from scipy import stats as sps
from sklearn.metrics import cohen_kappa_score

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

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PIXEL, TILE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "Calibri", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
def tidy(ax):
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    ax.tick_params(length=0); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)

MARKERS = ["A", "F", "R", "U", "W"]
MODES = {"pixel": "oncostem_ai_scores.csv", "tile": "oncostem_ai_scores_tiles.csv"}
readers = {(r["case"], r["marker"]): r for r in json.loads((HERE / "rows.json").read_text(encoding="utf-8"))}
slides = {(s["case"], s["marker"], s["mode"]): s for s in json.loads((HERE / "slides.json").read_text(encoding="utf-8"))}
CASES = sorted({k[0] for k in readers})
NAME = {m: next(r["markerName"] for r in readers.values() if r["marker"] == m) for m in MARKERS}

csvrows, cells_rows = {}, {}
for mode, name in MODES.items():
    with open(data_versions.results_root() / name, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["case_id"], r["marker"], mode); csvrows[key] = r
            d = data_versions.data_root() / "history" / r["case_id"] / "markers" / r["marker"] / "per_cell" / f"{r['he_upload_id']}__{r['ihc_upload_id']}"
            rows = []
            for p in sorted(d.glob("region*/cells.json")):
                rows.extend(json.loads(p.read_text(encoding="utf-8")).get("cells", []))
            rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
            cells_rows[key] = (rows, {int(x["rank"]): float(x["areaMm2"]) for x in rep["regions"]})
out = {}

# --- funnel ------------------------------------------------------------------------------------
funnel = {}
for (c, m, mode), s in slides.items():
    funnel[f"{c}:{m}:{mode}"] = {"tissueMm2": s["heTissueMm2"], "invasiveMm2": s["invasiveMm2"], "outlinedMm2": s["carriedMm2"],
        "sampledMm2": s["sampledMm2"], "nucleiDetected": s["nucleiDetected"], "nucleiCounted": s["nucleiCounted"], "tumourCells": s["cells"], "positive": s["positive"]}
out["funnel"] = funnel
share = {mode: [slides[(c, m, mode)]["sampledMm2"] / slides[(c, m, mode)]["carriedMm2"] for c in CASES for m in MARKERS
                if slides[(c, m, mode)]["carriedMm2"] and slides[(c, m, mode)]["sampledMm2"] is not None] for mode in MODES}
out["sampledShareOfOutlined"] = {m: {"median": round(float(np.median(v)), 3), "min": round(min(v), 3), "max": round(max(v), 3)} for m, v in share.items()}
tumour_share = {mode: [slides[(c, m, mode)]["cells"] / slides[(c, m, mode)]["nucleiCounted"] for c in CASES for m in MARKERS if slides[(c, m, mode)]["nucleiCounted"]] for mode in MODES}
out["tumourShareOfNuclei"] = {m: {"median": round(float(np.median(v)), 3), "min": round(min(v), 3), "max": round(max(v), 3)} for m, v in tumour_share.items()}

ex = [("CAN_00270", "A", "pixel"), ("CAN_00270", "A", "tile")]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 2.9))
for ax, key in zip(axes, ex):
    s = slides[key]
    labels = ["Tissue on the H&E", "Invasive tumour (classifier)", "Outlined and carried to IHC", "Actually sampled for nuclei"]
    vals = [s["heTissueMm2"], s["invasiveMm2"], s["carriedMm2"], s["sampledMm2"]]
    bars = ax.barh(labels[::-1], vals[::-1], color=PIXEL if key[2] == "pixel" else TILE, height=0.55, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"{v:.1f} mm²", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(vals) * 1.22); tidy(ax); ax.grid(axis="y", visible=False)
    ax.set_title(f"{key[0]} · CD44 · {'pixel-wise' if key[2] == 'pixel' else 'tile-based'}: "
                 f"{s['nucleiCounted']:,} nuclei → {s['cells']:,} tumour cells → {s['positive']:,} positive", fontsize=9, color=INK, loc="left")
axes[1].set_yticklabels([])
fig.tight_layout(); fig.savefig(HERE / "fig_funnel.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# --- region selection ----------------------------------------------------------------------------
sel = {}
for c in CASES:
    for mode in MODES:
        he = csvrows[(c, "A", mode)]["he_upload_id"]
        f = data_versions.data_root() / "history" / c / "shared" / "roi_selection" / he / "report.json"
        if not f.is_file():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        sel[f"{c}:{mode}"] = {"candidates": len(r["candidates"]), "selected": len(r["selected"]), "chosenByPerson": r.get("chosenByPerson"),
            "invasiveMm2": r.get("invasiveMm2"), "offeredMm2": r.get("offeredMm2"), "selectedMm2": r.get("selectedMm2"),
            "droppedSmall": r.get("droppedSmall"), "droppedSmallMm2": r.get("droppedSmallMm2"), "droppedCapped": r.get("droppedCapped"), "droppedCappedMm2": r.get("droppedCappedMm2")}
out["selection"] = sel

# --- weighting --------------------------------------------------------------------------------
def mae(vals):
    return round(st.mean(abs(v) for v in vals), 1)
weighting = {}
for mode in MODES:
    res = {}
    for col, lab in (("percent_area_weighted", "Area-weighted (delivered)"), ("percent_pooled", "Pooled by cell"), ("percent_plain_mean", "Plain mean of regions")):
        d = [float(csvrows[(c, m, mode)][col] or 0) - readers[(c, m)]["readerPctMean"] for c in CASES for m in MARKERS]
        res[lab] = {"mae": mae(d), "bias": round(st.mean(d), 1)}
    spread = [max(float(csvrows[(c, m, mode)][k] or 0) for k in ("percent_area_weighted", "percent_pooled", "percent_plain_mean"))
              - min(float(csvrows[(c, m, mode)][k] or 0) for k in ("percent_area_weighted", "percent_pooled", "percent_plain_mean")) for c in CASES for m in MARKERS]
    res["spreadMedian"] = round(float(np.median(spread)), 1); res["spreadMax"] = round(max(spread), 1)
    res["spreadOver10"] = sum(x >= 10 for x in spread)
    weighting[mode] = res
out["weighting"] = weighting
ex270 = {k: float(csvrows[("CAN_00270", "A", "pixel")][k]) for k in ("percent_area_weighted", "percent_pooled", "percent_plain_mean")}
out["weighting270A"] = ex270
labs = ["Area-weighted (delivered)", "Pooled by cell", "Plain mean of regions"]
fig, ax = plt.subplots(figsize=(6.4, 2.8)); x = np.arange(3); w = 0.36
for i, (mode, col) in enumerate((("pixel", PIXEL), ("tile", TILE))):
    vals = [weighting[mode][l]["mae"] for l in labs]
    bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.92, color=col, label="Pixel-wise" if mode == "pixel" else "Tile-based", edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(x, labs); ax.set_ylabel("Average difference from\npathologists (points)"); tidy(ax); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16)); ax.set_ylim(0, max(weighting[m][l]["mae"] for m in MODES for l in labs) * 1.2)
fig.tight_layout(); fig.savefig(HERE / "fig_weighting.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# --- pixel vs tile -------------------------------------------------------------------------------
pt = [slides[(c, m, "pixel")]["percent"] - slides[(c, m, "tile")]["percent"] for c in CASES for m in MARKERS]
out["pixelVsTile"] = {"changedBy5": sum(abs(x) >= 5 for x in pt), "changedBy20": sum(abs(x) >= 20 for x in pt), "min": min(pt), "max": max(pt),
    "cellRatio": {f"{c}:{m}": [slides[(c, m, "pixel")]["cells"], slides[(c, m, "tile")]["cells"]] for c in CASES for m in MARKERS}}
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.fill_between([0, 100], [-5, 95], [5, 105], color=GRID, alpha=0.7, lw=0, label="within ±5 points")
ax.plot([0, 100], [0, 100], color=INK2, lw=1)
ax.scatter([slides[(c, m, "tile")]["percent"] for c in CASES for m in MARKERS], [slides[(c, m, "pixel")]["percent"] for c in CASES for m in MARKERS],
           s=40, color=PIXEL, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="one slide")
ax.set_xlabel("AI tile-based percent"); ax.set_ylabel("AI pixel-wise percent"); ax.set_xlim(-3, 103); ax.set_ylim(-3, 103); tidy(ax)
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "fig_pixel_vs_tile.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# --- cells vs error; shortfall vs stain -------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.5))
for mode, col, mk in (("pixel", PIXEL, "o"), ("tile", TILE, "s")):
    xs = [max(slides[(c, m, mode)]["cells"], 1) for c in CASES for m in MARKERS]
    ys = [abs(slides[(c, m, mode)]["percent"] - readers[(c, m)]["readerPctMean"]) for c in CASES for m in MARKERS]
    a1.scatter(xs, ys, s=38, marker=mk, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Pixel-wise" if mode == "pixel" else "Tile-based")
    pts = [(readers[(c, m)]["readerPctMean"], slides[(c, m, mode)]["densityShortfall"] * 100) for c in CASES for m in MARKERS if slides[(c, m, mode)]["densityShortfall"] is not None]
    a2.scatter([p[0] for p in pts], [p[1] for p in pts], s=38, marker=mk, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="Pixel-wise" if mode == "pixel" else "Tile-based")
    rho = sps.spearmanr([p[0] for p in pts], [p[1] for p in pts]).statistic
    out.setdefault("shortfallVsStain", {})[mode] = round(float(rho), 2)
    small = [e for x_, e in zip(xs, ys) if x_ < 400]; big = [e for x_, e in zip(xs, ys) if x_ >= 400]
    out.setdefault("errorByCells", {})[mode] = {"under400": {"n": len(small), "mae": round(st.mean(small), 1) if small else None},
                                               "atLeast400": {"n": len(big), "mae": round(st.mean(big), 1) if big else None}}
a1.set_xscale("log"); a1.axvline(50, color=INK2, lw=1, ls="--"); a1.axvline(400, color=INK2, lw=1, ls=":")
a1.text(52, a1.get_ylim()[1] * 0.97 if a1.get_ylim()[1] else 60, "50: not a measurement", fontsize=8, color=INK2, va="top")
a1.text(420, 2, "400: thin-count warning", fontsize=8, color=INK2)
a1.set_xlabel("Tumour cells measured (log scale)"); a1.set_ylabel("Difference from pathologists (points)")
a1.set_title("Error against the number of cells measured", fontsize=10, color=INK, loc="left"); tidy(a1)
a2.set_xlabel("Pathologists' mean percent positive"); a2.set_ylabel("Nuclei missing vs H&E (%)"); a2.set_ylim(-5, 105)
a2.set_title("Nuclei missing against how much of the tumour is stained", fontsize=10, color=INK, loc="left"); tidy(a2)
a2.legend(frameon=False, loc="lower right", fontsize=8); a1.legend(frameon=False, loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "fig_cells_shortfall.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# --- sensitivity sweeps on the stored per-cell rows ---------------------------------------------
second_grid = [0.20, 0.30, 0.35, 0.40, 0.50]
od_grid = [0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
sens = {}
for mode in MODES:
    by_second, by_od = [], []
    for sm in second_grid:
        d = []
        for c in CASES:
            for m in MARKERS:
                rows, areas = cells_rows[(c, m, mode)]
                if not rows: continue
                cut = cut_points.for_marker(m)
                # the membrane markers' completeness minimum is swept; the cytoplasmic markers keep theirs
                second = sm if cut.compartment == "membrane" else cut.second_min
                pct, _ = calib.percent_at(rows, areas, cut.positivity_od, second)
                d.append(calib.round5(pct) - readers[(c, m)]["readerPctMean"])
        by_second.append(mae(d))
    for od in od_grid:
        d = []
        for c in CASES:
            for m in MARKERS:
                rows, areas = cells_rows[(c, m, mode)]
                if not rows: continue
                cut = cut_points.for_marker(m)
                pct, _ = calib.percent_at(rows, areas, od, cut.second_min)
                d.append(calib.round5(pct) - readers[(c, m)]["readerPctMean"])
        by_od.append(mae(d))
    sens[mode] = {"second": dict(zip(map(str, second_grid), by_second)), "od": dict(zip(map(str, od_grid), by_od))}
# how much one slide's percent moves across the completeness range
move = {}
for mode in MODES:
    mv = []
    for c in CASES:
        for m in ("A", "F", "R"):
            rows, areas = cells_rows[(c, m, mode)]
            if not rows: continue
            cut = cut_points.for_marker(m)
            vals = [calib.percent_at(rows, areas, cut.positivity_od, sm)[0] for sm in (0.20, 0.50)]
            mv.append(vals[0] - vals[1])
    move[mode] = {"median": round(float(np.median(mv)), 1), "max": round(max(mv), 1)}
sens["completenessMove"] = move
out["sensitivity"] = sens
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.1))
for mode, col, mk in (("pixel", PIXEL, "o"), ("tile", TILE, "s")):
    a1.plot(second_grid, sens[mode]["second"].values(), color=col, lw=2, marker=mk, markersize=7, markeredgecolor=SURFACE, label="Pixel-wise" if mode == "pixel" else "Tile-based")
    a2.plot(od_grid, sens[mode]["od"].values(), color=col, lw=2, marker=mk, markersize=7, markeredgecolor=SURFACE, label="Pixel-wise" if mode == "pixel" else "Tile-based")
a1.axvline(0.35, color=INK2, lw=1, ls="--"); a1.text(0.352, a1.get_ylim()[0] + 0.3, "current 0.35", fontsize=8, color=INK2)
a2.axvline(0.15, color=INK2, lw=1, ls="--"); a2.text(0.152, a2.get_ylim()[0] + 0.3, "current 0.15 / 0.12", fontsize=8, color=INK2)
a1.set_xlabel("Ring completeness a membrane cell needs (CD44, ABCC4, ABCC11)"); a2.set_xlabel("Brown density a cell needs (all five markers)")
for a in (a1, a2):
    a.set_ylabel("Average difference from\npathologists (points)"); tidy(a); a.legend(frameon=False, fontsize=8)
a1.set_title("Sensitivity to the ring-completeness rule", fontsize=10, color=INK, loc="left")
a2.set_title("Sensitivity to the positivity cut", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig(HERE / "fig_sensitivity.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# --- full agreement set ----------------------------------------------------------------------------
bands = list(cut_points.PERMITTED_BANDS)
nearest = lambda v: min(bands, key=lambda b: abs(b - v))
agree = {}
for mode in MODES:
    ai = np.array([slides[(c, m, mode)]["percent"] for c in CASES for m in MARKERS]); rd = np.array([readers[(c, m)]["readerPctMean"] for c in CASES for m in MARKERS])
    e = ai - rd
    aiI = [slides[(c, m, mode)]["intensity"] for c in CASES for m in MARKERS]; rdI = [readers[(c, m)]["readerIntMean"] for c in CASES for m in MARKERS]
    rdB = [nearest(v) for v in rdI]
    to_idx = {b: i for i, b in enumerate(bands)}
    agree[mode] = {"mae": round(float(np.abs(e).mean()), 1), "medianAe": round(float(np.median(np.abs(e))), 1), "rmse": round(float(np.sqrt((e ** 2).mean())), 1),
        "bias": round(float(e.mean()), 1), "pearson": round(float(sps.pearsonr(ai, rd).statistic), 2), "spearman": round(float(sps.spearmanr(ai, rd).statistic), 2),
        "within5": int((np.abs(e) <= 5).sum()), "within10": int((np.abs(e) <= 10).sum()), "within20": int((np.abs(e) <= 20).sum()),
        "intExact": sum(a == b for a, b in zip(aiI, rdB)), "intWithin025": sum(abs(a - b) <= 0.25 for a, b in zip(aiI, rdI)), "intWithin05": sum(abs(a - b) <= 0.5 for a, b in zip(aiI, rdI)),
        "intKappa": round(float(cohen_kappa_score([to_idx[a] for a in aiI], [to_idx[b] for b in rdB], weights="quadratic", labels=list(range(len(bands))))), 2)}
# the pathologists against their own consensus, for scale: each reader vs the mean of the other three
self_e = []
for (c, m), r in readers.items():
    for k, v in r["readers"].items():
        others = [w[0] for kk, w in r["readers"].items() if kk != k]
        self_e.append(v[0] - st.mean(others))
agree["readerLeaveOneOut"] = {"mae": round(st.mean(abs(x) for x in self_e), 1), "within10": sum(abs(x) <= 10 for x in self_e), "n": len(self_e)}
out["agreement"] = agree

# --- data lineage: region files on disk must be the regions the report lists ---------------------
stale = []
for (c, m, mode), r in csvrows.items():
    d = data_versions.data_root() / "history" / c / "markers" / m / "per_cell" / f"{r['he_upload_id']}__{r['ihc_upload_id']}"
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    listed = {int(x["rank"]) for x in rep["regions"]}
    on_disk = {int(p.parent.name.replace("region", "")) for p in d.glob("region*/cells.json")}
    if on_disk == listed:
        continue
    rows = []
    for rank in sorted(listed):
        f = d / f"region{rank}" / "cells.json"
        if f.is_file():
            rows.extend(json.loads(f.read_text(encoding="utf-8")).get("cells", []))
    areas = {int(x["rank"]): float(x["areaMm2"]) for x in rep["regions"]}
    cut = cut_points.for_marker(m)
    pct, _ = calib.percent_at(rows, areas, cut.positivity_od, cut.second_min)
    stale.append({"case": c, "marker": m, "markerName": NAME[m], "mode": mode, "filesOnDisk": len(on_disk), "regionsInReport": len(listed),
                  "csvCells": int(float(r["cells"] or 0)), "correctCells": len(rows), "csvPercent": float(r["percent"]), "correctPercent": calib.round5(pct)})
out["stale"] = stale
out["rowsChecked"] = len(csvrows)
print("stale per-cell folders:", stale)

(HERE / "attention.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("sampledShareOfOutlined", "tumourShareOfNuclei", "weighting", "weighting270A", "shortfallVsStain", "errorByCells", "sensitivity", "agreement")}, indent=0))
print("pixelVsTile", {k: v for k, v in out["pixelVsTile"].items() if k != "cellRatio"})
print("selection", json.dumps(sel, indent=0)[:1500])
