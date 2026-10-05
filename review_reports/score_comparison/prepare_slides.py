"""Everything the per-slide review pages need, taken from the exact run behind each CSV row.

For every case, marker and mode (pixel / tile) the CSV names an H&E and IHC upload id; the
registration images, nuclei fields and reports are read from v<N>_data/data/history under that pair,
so a page never shows a picture from a different run than the score beside it.
"""
import csv, json, pathlib, statistics as st, sys
from PIL import Image

REPO = pathlib.Path(r"C:\Users\Coditas\Desktop\Healthcare_Projects\Cancer_Scoring_System")
sys.path.append(str(REPO))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402
HIST = data_versions.data_root() / "history"
#: Intermediate files (tables, images, charts) live under v<N>_data/data/, not beside the code.
HERE = data_versions.data_root() / "review_reports" / "score_comparison"
HERE.mkdir(parents=True, exist_ok=True)
ASSETS = HERE / "assets"; ASSETS.mkdir(exist_ok=True)
MODES = {"pixel": "oncostem_ai_scores.csv", "tile": "oncostem_ai_scores_tiles.csv"}
readers = {(r["case"], r["marker"]): r for r in json.loads((HERE / "rows.json").read_text(encoding="utf-8"))}


def jpg(src, dst, side, q=82):
    im = Image.open(src).convert("RGB"); im.thumbnail((side, side), Image.LANCZOS); im.save(dst, "JPEG", quality=q, optimize=True)


def strip(field_dir, dst):
    """The IHC field as scanned, the counterstain channel the detector sees, and what it found."""
    names = ("f0_raw", "f0_input", "f0_over")
    if not all((field_dir / f"{n}.png").is_file() for n in names):
        return False
    tiles = [Image.open(field_dir / f"{n}.png").convert("RGB").resize((420, 420), Image.LANCZOS) for n in names]
    canvas = Image.new("RGB", (420 * 3 + 24, 420), "white")
    for i, t in enumerate(tiles):
        canvas.paste(t, (i * 432, 0))
    canvas.save(dst, "JPEG", quality=82, optimize=True)
    return True


def caveats(text):
    """`TITLE. First sentence.` for each pipeline caveat, in the order the pipeline gave them."""
    out = []
    for part in [p.strip() for p in text.split(" | ") if p.strip()]:
        title, _, rest = part.partition(". ")
        first = rest.split(". ")[0].rstrip(".") + "." if rest else ""
        out.append({"title": title.strip().rstrip(".").capitalize(), "text": first})
    return out


slides = []
for mode, name in MODES.items():
    with open(data_versions.results_root() / name, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            case, m = r["case_id"], r["marker"]
            pair = f"{r['he_upload_id']}__{r['ihc_upload_id']}"
            base = HIST / case / "markers" / m
            align = json.loads((base / "ihc_alignment" / pair / "report.json").read_text(encoding="utf-8"))
            nuc_dir = base / "nuclei" / pair
            nuc = json.loads((nuc_dir / "report.json").read_text(encoding="utf-8"))
            stem = f"{case}_{m}_{mode}"
            jpg(base / "ihc_alignment" / pair / "he_borders.png", ASSETS / f"{stem}_he.jpg", 820)
            jpg(base / "ihc_alignment" / pair / "ihc_borders.png", ASSETS / f"{stem}_ihc.jpg", 820)
            # the first sampled field of the largest region
            top = min(nuc["regions"], key=lambda x: x["rank"])
            has_strip = strip(nuc_dir / f"region{top['rank']}", ASSETS / f"{stem}_nuclei.jpg")
            d = align.get("diagnostics") or {}
            slides.append({
                "case": case, "marker": m, "markerName": r["marker_name"], "mode": mode,
                "percent": float(r["percent"]), "intensity": float(r["intensity"]), "intensityLabel": r["intensity_label"],
                "percentRaw": float(r["percent_raw"] or 0), "intensityRawOd": float(r["intensity_raw"] or 0),
                "cells": int(float(r["cells"] or 0)), "positive": int(float(r["positive_cells"] or 0)), "hScore": float(r["h_score"] or 0),
                "compartment": r["compartment"], "caveats": caveats(r["caveats"]),
                "heImage": str(ASSETS / f"{stem}_he.jpg"), "ihcImage": str(ASSETS / f"{stem}_ihc.jpg"),
                "nucleiImage": str(ASSETS / f"{stem}_nuclei.jpg") if has_strip else None,
                "regions": len(align["regions"]), "carriedMm2": align.get("carriedMm2"), "invasiveMm2": align.get("invasiveMm2"),
                "coverage": align.get("areaCoverage"), "nmi": d.get("alignmentNmi"), "heTissueMm2": d.get("heTissueMm2"),
                "ihcTissueMm2": d.get("ihcTissueMm2"), "tissueRatio": d.get("tissueAreaRatio"), "method": d.get("method"),
                "confirmedBy": align.get("confirmedBy"), "refusalReasons": align.get("refusalReasons"),
                "nucleiDetected": nuc.get("detected"), "nucleiCounted": nuc.get("counted"), "sampledMm2": nuc.get("sampledMm2"),
                "densityPerMm2": nuc.get("densityPerMm2"), "heDensityPerMm2": nuc.get("heDensityPerMm2"),
                "densityShortfall": nuc.get("densityShortfall"), "fieldsSegmented": sum(len(x.get("fields", [])) for x in nuc["regions"]),
            })

(HERE / "slides.json").write_text(json.dumps(slides, indent=1), encoding="utf-8")
print(len(slides), "slide records;", sum(1 for s in slides if s["nucleiImage"]), "with nuclei strips")
print("shortfall pixel mean", round(st.mean(s["densityShortfall"] for s in slides if s["mode"] == "pixel" and s["densityShortfall"] is not None), 3),
      "tile mean", round(st.mean(s["densityShortfall"] for s in slides if s["mode"] == "tile" and s["densityShortfall"] is not None), 3))
print("confirmedBy:", {s["confirmedBy"] for s in slides})
