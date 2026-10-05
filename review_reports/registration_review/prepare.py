"""Collect the 60 step-12 registrations (6 cases x 5 markers x 2 modes) for the review doc.

Picks, per case/marker/mode, the most recent confirmed report in v<N>_data/data/history, copies its
he_borders.png / ihc_borders.png as compact JPEGs, and measures the carried area on both
slides from the stored rings (outer ring positive, holes negative, so the signed sum is
the region's area).
"""

import collections
import json
import pathlib
import sys

from PIL import Image

# The workspace root, from this file's own place - review_reports/<review>/<script>.py
# - rather than one machine's path (P-20).
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tissue_scoring_demo" / "backend"))
from app.ingestion.slide_reader import open_slide  # noqa: E402

sys.path.append(str(REPO))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402

HISTORY = data_versions.data_root() / "history"
#: Intermediate files live under v<N>_data/data/, not beside the code; build.js reads them from there.
OUT = data_versions.data_root() / "review_reports" / "registration_review" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
MARKERS = ["A", "F", "R", "U", "W"]
MODES = {"beetle_pixel": "pixel", "step9_tile": "tile"}
_mpp_cache: dict[str, float] = {}


def mpp_of(path: str) -> float:
    if path not in _mpp_cache:
        value = open_slide(pathlib.Path(path)).mpp
        _mpp_cache[path] = float(value[0] if isinstance(value, (tuple, list)) else value)
    return _mpp_cache[path]


def ring_area_px(rings) -> float:
    total = 0.0
    for ring in rings:
        s = 0.0
        for i in range(len(ring)):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % len(ring)]
            s += x1 * y2 - x2 * y1
        total += s / 2
    return abs(total)


def slide_record(case: str, upload_id: str) -> dict:
    for candidate in [HISTORY / case / "shared" / "slides" / f"{upload_id}.json",
                      *(HISTORY / case / "markers").glob(f"*/slides/{upload_id}.json")]:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"{case}: no slide record for {upload_id}")


def compact(src: pathlib.Path, dst: pathlib.Path, side: int = 1000, quality: int = 84) -> None:
    image = Image.open(src).convert("RGB")
    image.thumbnail((side, side), Image.LANCZOS)
    image.save(dst, "JPEG", quality=quality, optimize=True)


best: dict[tuple, tuple] = {}
for report_path in HISTORY.glob("*/markers/*/ihc_alignment/*/report.json"):
    case, marker = report_path.parts[-6], report_path.parts[-4]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sources = {r.get("source") for r in report.get("regions", [])}
    if len(sources) != 1 or report.get("state") != "ready":
        continue
    mode = MODES.get(sources.pop())
    if mode is None:
        continue
    key = (case, marker, mode)
    stamp = report.get("generatedAt") or ""
    if key not in best or stamp > best[key][0]:
        best[key] = (stamp, report_path, report)

cases = sorted({k[0] for k in best})
rows = []
for case in cases:
    case_json = json.loads((HISTORY / case / "manifest.json").read_text(encoding="utf-8"))
    for mode in ("pixel", "tile"):
        for marker in MARKERS:
            stamp, report_path, report = best[(case, marker, mode)]
            folder = report_path.parent
            he = slide_record(case, report["heUploadId"])
            ihc = slide_record(case, report["ihcUploadId"])
            he_mpp, ihc_mpp = mpp_of(he["final_path"]), mpp_of(ihc["final_path"])
            he_mm2 = sum(ring_area_px(r["heRings"]) for r in report["regions"]) * he_mpp**2 / 1e6
            ihc_mm2 = sum(ring_area_px(r["ihcRings"]) for r in report["regions"]) * ihc_mpp**2 / 1e6
            stem = f"{case}_{marker}_{mode}"
            compact(folder / "he_borders.png", OUT / f"{stem}_he.jpg")
            compact(folder / "ihc_borders.png", OUT / f"{stem}_ihc.jpg")
            # The largest region, cropped from both slides at full resolution: the check that
            # the two outlines sit on the same cells, which the whole-slide view cannot show.
            top = min(report["regions"], key=lambda r: r["rank"])
            compact(folder / "he_invasive1.png", OUT / f"{stem}_he_zoom.jpg", 720, 78)
            compact(folder / "invasive1.png", OUT / f"{stem}_ihc_zoom.jpg", 720, 78)
            d = report.get("diagnostics") or {}
            rows.append({
                "case": case,
                "marker": marker,
                "markerName": case_json["markers"][marker]["name"],
                "mode": mode,
                "heFile": he["filename"],
                "ihcFile": ihc["filename"],
                "heImage": str(OUT / f"{stem}_he.jpg"),
                "ihcImage": str(OUT / f"{stem}_ihc.jpg"),
                "heZoom": str(OUT / f"{stem}_he_zoom.jpg"),
                "ihcZoom": str(OUT / f"{stem}_ihc_zoom.jpg"),
                "topRegionMm2": round(top["areaMm2"], 3),
                "regions": len(report["regions"]),
                "invasiveMm2": report.get("invasiveMm2"),
                "carriedMm2": report.get("carriedMm2"),
                "coverage": report.get("areaCoverage"),
                "heRegionMm2": round(he_mm2, 3),
                "ihcRegionMm2": round(ihc_mm2, 3),
                "heTissueMm2": d.get("heTissueMm2"),
                "ihcTissueMm2": d.get("ihcTissueMm2"),
                "nmi": d.get("alignmentNmi"),
                "roundTripUm": d.get("roundTripMedianUm"),
                "folded": d.get("foldedPoints"),
                "method": d.get("method"),
                "confirmed": report.get("confirmed"),
                "generatedAt": stamp,
            })

(OUT.parent / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
print(len(rows), "rows;", len(cases), "cases")
for r in rows[:3] + rows[-2:]:
    print(r["case"], r["marker"], r["mode"], r["regions"], r["carriedMm2"], r["heRegionMm2"], r["ihcRegionMm2"])
