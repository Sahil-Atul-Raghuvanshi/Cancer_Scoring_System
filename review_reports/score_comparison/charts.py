"""Small-multiple comparison charts (one panel per marker) and the per-marker agreement table."""
import json, pathlib, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402

#: Intermediate files (tables, images, charts) live under v<N>_data/data/, not beside the code.
HERE = data_versions.data_root() / "review_reports" / "score_comparison"
HERE.mkdir(parents=True, exist_ok=True)
rows = json.loads((HERE / "rows.json").read_text(encoding="utf-8"))
MARKERS = ["A", "F", "R", "U", "W"]
CASES = sorted({r["case"] for r in rows})

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
PIXEL, TILE, RANGE = "#2a78d6", "#eb6834", "#c9c8c3"   # validated categorical slots 1-2
plt.rcParams.update({"font.family": "Calibri", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def figure(field, lo_key, hi_key, mean_key, ylim, yticks, ylabel, out):
    fig, axes = plt.subplots(1, 5, figsize=(10.2, 3.3), sharey=True)
    for ax, m in zip(axes, MARKERS):
        sub = {r["case"]: r for r in rows if r["marker"] == m}
        for i, c in enumerate(CASES):
            r = sub[c]
            # readers' range as a thin bar, their mean as a dark tick
            ax.plot([i, i], [r[lo_key], r[hi_key]], color=RANGE, lw=9, solid_capstyle="round", zorder=1)
            ax.plot([i - 0.22, i + 0.22], [r[mean_key]] * 2, color=INK, lw=2, zorder=2)
            ax.scatter(i - 0.16, r["pixel"][field], s=46, marker="o", color=PIXEL, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.scatter(i + 0.16, r["tile"][field], s=46, marker="s", color=TILE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_title(sub[CASES[0]]["markerName"], fontsize=10, color=INK, fontweight="bold")
        ax.set_xticks(range(len(CASES)), [c.replace("CAN_", "") for c in CASES], rotation=45, fontsize=8)
        ax.set_xlim(-0.6, len(CASES) - 0.4)
        ax.set_ylim(*ylim); ax.set_yticks(yticks)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel(ylabel)
    handles = [
        plt.Line2D([], [], color=RANGE, lw=9, solid_capstyle="round", label="Pathologists' range (4 readers)"),
        plt.Line2D([], [], color=INK, lw=2, label="Pathologists' mean"),
        plt.Line2D([], [], marker="o", ls="", color=PIXEL, markersize=7, label="AI pixel-wise (BEETLE outline)"),
        plt.Line2D([], [], marker="s", ls="", color=TILE, markersize=7, label="AI tile-based (step 9 squares)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.02, "Case", ha="center", color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


figure("pct", "readerPctMin", "readerPctMax", "readerPctMean", (0, 105), [0, 20, 40, 60, 80, 100], "Percent positive (%)", HERE / "fig_percent.png")
figure("int", "readerIntMin", "readerIntMax", "readerIntMean", (0, 2.15), [0, 0.5, 1, 1.5, 2], "Intensity (0–2 scale)", HERE / "fig_intensity.png")

# per-marker and overall agreement, for the document's tables
def agree(sub, key, field):
    mean_k, lo_k, hi_k = {"pct": ("readerPctMean", "readerPctMin", "readerPctMax"), "int": ("readerIntMean", "readerIntMin", "readerIntMax")}[field]
    d = [r[key][field] - r[mean_k] for r in sub]
    return {"mae": round(st.mean(abs(x) for x in d), 2), "bias": round(st.mean(d), 2),
            "within": sum(r[lo_k] <= r[key][field] <= r[hi_k] for r in sub),
            "within10": sum(abs(r[key][field] - r[mean_k]) <= 10 for r in sub) if field == "pct" else None, "n": len(sub)}

stats = {"overall": {k: {f: agree(rows, k, f) for f in ("pct", "int")} for k in ("pixel", "tile")}, "markers": {}}
for m in MARKERS:
    sub = [r for r in rows if r["marker"] == m]
    stats["markers"][m] = {"name": sub[0]["markerName"], **{k: {f: agree(sub, k, f) for f in ("pct", "int")} for k in ("pixel", "tile")},
                           "readerSpread": round(st.mean(r["readerPctMax"] - r["readerPctMin"] for r in sub), 1)}
stats["readerMad"] = round(st.mean(st.mean(abs(v[0] - r["readerPctMean"]) for v in r["readers"].values()) for r in rows), 2)
stats["pixelVsTileMae"] = round(st.mean(abs(r["pixel"]["pct"] - r["tile"]["pct"]) for r in rows), 2)
(HERE / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
print(json.dumps(stats["overall"], indent=0)); print("reader MAD", stats["readerMad"], "pixel-vs-tile", stats["pixelVsTileMae"])
for m, s in stats["markers"].items():
    print(m, s["name"], "pix", s["pixel"]["pct"], "tile", s["tile"]["pct"], "spread", s["readerSpread"])


# --- figure 3: how far each approach is from the pathologists (held-out where fitted) -----------
an = json.loads((HERE / "analysis.json").read_text(encoding="utf-8"))
board = an["scoreboard"]
order = [("pixel: as delivered", "AI pixel-wise, as delivered"), ("tile: as delivered", "AI tile-based, as delivered"),
         ("pixel: cut refit (held-out)", "Pixel-wise, cut points refitted*"), ("tile: cut refit (held-out)", "Tile-based, cut points refitted*"),
         ("ML recalibration, both methods (held-out)", "ML recalibration of both*"), ("No-AI baseline (marker's usual reader score)", "No AI: marker's usual score*")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.4))
labels = [lab for _, lab in order][::-1]
pct = [board[k]["pctMae"] for k, _ in order][::-1]
inr = [board[k]["inInt"] for k, _ in order][::-1]
for ax, vals, title, fmt in ((a1, pct, "Percent positive: average difference\nfrom pathologists (points, lower is better)", "{:.1f}"),
                             (a2, inr, "Intensity: slides inside the\npathologists' range (of 30, higher is better)", "{:d}")):
    bars = ax.barh(labels, vals, color=PIXEL, height=0.55, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + (0.4 if ax is a1 else 0.4), b.get_y() + b.get_height() / 2, fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
a1.axvline(json.loads((HERE / "stats.json").read_text())["readerMad"], color=INK2, lw=1.2, ls="--")
a1.text(json.loads((HERE / "stats.json").read_text())["readerMad"] + 0.4, -0.62, "pathologists vs their own mean", fontsize=8, color=INK2, va="center")
a2.set_yticklabels([]); a2.set_xlim(0, 30); a1.set_ylim(-0.95, 5.5); a2.set_ylim(-0.95, 5.5)
fig.text(0.01, -0.03, "* Tested on held-out cases: fitted on five cases, scored on the sixth, repeated for every case.", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(HERE / "fig_scoreboard.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# --- figure 4: nuclei shortfall heatmap, IHC vs its H&E -------------------------------------
slides = json.loads((HERE / "slides.json").read_text(encoding="utf-8"))
from matplotlib.colors import LinearSegmentedColormap
ramp = LinearSegmentedColormap.from_list("blue", ["#eaf2fc", "#9cc3ef", "#2a78d6", "#123f75"])
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.3))
for ax, mode, title in ((axes[0], "pixel", "Pixel-wise"), (axes[1], "tile", "Tile-based")):
    def short(c, m):
        v = next(s["densityShortfall"] for s in slides if s["case"] == c and s["marker"] == m and s["mode"] == mode)
        return None if v is None else max(v, 0.0) * 100
    grid = [[short(c, m) for c in CASES] for m in MARKERS]
    import numpy as np
    masked = np.ma.masked_invalid(np.array([[np.nan if v is None else v for v in row] for row in grid], float))
    ramp.set_bad("#e6e5e1")
    ax.imshow(masked, cmap=ramp, vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            raw = next(s["densityShortfall"] for s in slides if s["case"] == CASES[j] and s["marker"] == MARKERS[i] and s["mode"] == mode)
            label = "—" if v is None else ("more" if raw < 0 else f"{v:.0f}%")
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=INK2 if v is None else ("#ffffff" if v >= 60 else INK))
    ax.set_xticks(range(len(CASES)), [c.replace("CAN_", "") for c in CASES], fontsize=8)
    names = {r["marker"]: r["markerName"] for r in rows}
    ax.set_yticks(range(len(MARKERS)), [names[m] for m in MARKERS], fontsize=8)
    ax.set_title(title, fontsize=10, color=INK, fontweight="bold")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks([x - 0.5 for x in range(1, len(CASES))], minor=True); ax.set_yticks([y - 0.5 for y in range(1, len(MARKERS))], minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2); ax.tick_params(which="minor", length=0)
fig.suptitle("Nuclei missing on the IHC slide, compared with the same regions on the H&E (0% = none missing)", fontsize=10, color=INK, y=1.0)
fig.tight_layout(); fig.savefig(HERE / "fig_shortfall.png", dpi=220, bbox_inches="tight"); plt.close(fig)
print("figures 3 and 4 written")
