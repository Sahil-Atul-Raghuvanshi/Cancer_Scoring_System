"""How much of the tile-drawn region did BEETLE actually call invasive?

    python beetle_coverage.py [--out beetle_coverage.csv]

Read-only. It touches nothing the pipeline owns and runs in seconds, so it is safe beside
an overnight pass.

**The question.** Step 9 draws candidate regions on step 8's tile grid - squares of a
fixed field of view, a staircase around whatever the tile model called tumour. Step 11
then runs BEETLE inside each of those boxes at the pixel level and traces the real
boundary. The ratio between the two is the honest answer to whether the tile-based
approach is finding tumour or finding tumour-shaped neighbourhoods, and it is worth having
at three different grains:

    per slide   one number for the whole slide: of all the area the tiles claimed, how
                much survived BEETLE
    per region  the same for each ROI, which is where a single bad candidate shows up
    per tile    the distribution behind those averages, which is the only grain that can
                tell "the tiles are roughly right and BEETLE trims the edges" apart from
                "a few tiles are right and the rest are empty"

The third is the one the averages hide. A slide at 15% kept could be every tile 15%
invasive - a systematic over-reach the tile grid could be shrunk to fix - or 15% of tiles
at 100% and the rest at nothing, which is a different problem with a different fix.

**Two area rules, both reported, because they answer different questions.**
`maskShare` is BEETLE's raw per-pixel verdict. `contourShare` is what survives tracing,
which drops components below `roi_refinement_min_component_mm2` and is what actually gets
carried onto the IHC slide. The second is always the smaller and is the one the score
depends on; the first says how much of the gap is tracing rather than BEETLE.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

#: BEETLE's overlay colour per class code, copied rather than imported so this script
#: needs no backend on the path. Checked against
#: `app/pipeline/step08_tissue_type_segmentation/beetle.py`.
CLASS_COLOURS = {
    0: (100, 116, 139),  # unannotated - glass and background
    1: (250, 204, 21),   # other - stroma, fat, lymphocytes, vessels
    2: (59, 130, 246),   # non-invasive epithelium - DCIS, normal ducts
    3: (239, 68, 68),    # invasive epithelium - the only class a score is measured on
    4: (168, 85, 247),   # necrosis
}
SCORED_CODE = 3
CLASS_NAMES = {
    0: "unannotated",
    1: "other",
    2: "non_invasive_epithelium",
    3: "invasive_epithelium",
    4: "necrosis",
}

DEMO = common.DATA / "demo"
REFINEMENT = DEMO / "roi_refinement"
CASES = DEMO / "cases"

#: Finished runs are **moved** out of the live tree, not copied, and the shared H&E work -
#: which is where every BEETLE refinement lives - moves with the last marker of a case.
#: Walking only `v<N>_data/data/demo/roi_refinement` therefore measures whatever happens not to have
#: been filed yet, and silently loses a case the moment it completes: this report went from
#: three slides to two while cases were finishing, and the two it kept were the two still
#: in flight. Nothing was lost on disk - it was all under `v<N>_data/data/history` - but the coverage
#: CSVs, which are the deliverable, were quietly emptying out.
HISTORY = common.DATA / "history"


def refinement_dirs() -> list[pathlib.Path]:
    """Every refinement directory, live or filed, newest location last.

    A case that is filed mid-run can appear in both trees; the live copy wins, because it
    is the one the run is still writing to.
    """
    found: dict[str, pathlib.Path] = {}
    if HISTORY.is_dir():
        for case_dir in sorted(HISTORY.iterdir()):
            filed = case_dir / "shared" / "roi_refinement"
            if not filed.is_dir():
                continue
            for upload_dir in sorted(filed.iterdir()):
                if upload_dir.is_dir():
                    found[upload_dir.name] = upload_dir
    if REFINEMENT.is_dir():
        for upload_dir in sorted(REFINEMENT.iterdir()):
            if upload_dir.is_dir():
                found[upload_dir.name] = upload_dir
    return [found[name] for name in sorted(found)]


def slide_owner() -> dict[str, tuple[str, str]]:
    """upload id -> (case, role), built from the demo's own case records.

    An upload that no case record mentions is reported as `unknown`, not skipped: a
    refinement with no owner is a thing worth seeing, not a thing to hide.
    """
    owner: dict[str, tuple[str, str]] = {}

    # Filed cases first. Their live records move out with the rest of the run, so a case
    # that has completed is invisible to `v<N>_data/data/demo/cases` - which is why nearly every
    # archived refinement reported `unknown/unknown` once this script learned to read them.
    # The manifest carries the same mapping in a different shape.
    if HISTORY.is_dir():
        for manifest_path in sorted(HISTORY.glob("*/manifest.json")):
            manifest = common.read_json(manifest_path)
            if not manifest:
                continue
            case = manifest.get("caseId", "unknown")
            if manifest.get("heUploadId"):
                owner[manifest["heUploadId"]] = (case, "HE")
            for letter, entry in (manifest.get("markers") or {}).items():
                if isinstance(entry, dict) and entry.get("ihcUploadId"):
                    owner[entry["ihcUploadId"]] = (case, letter)

    # Live records win: a case being re-run right now has the newer upload ids here.
    if not CASES.is_dir():
        return owner
    for path in sorted(CASES.glob("*.json")):
        record = common.read_json(path)
        if not record:
            continue
        case = record.get("case_id", "unknown")
        if record.get("he_upload_id"):
            owner[record["he_upload_id"]] = (case, "HE")
        if record.get("ihc_upload_id"):
            owner[record["ihc_upload_id"]] = (case, record.get("marker", "?"))
    return owner


def classes_from_png(path: pathlib.Path) -> np.ndarray | None:
    """Recover BEETLE's class codes from the flat colour map it wrote.

    `beetle_mask.png` is painted with exactly the five overlay colours and nothing
    between them - the renderer paints labels, never interpolates - so the inverse is
    exact. Anything that is not one of the five (there should be nothing) reads as
    unannotated rather than as a sixth class.
    """
    if not path.is_file():
        return None
    rgb = np.asarray(Image.open(path).convert("RGB"))
    codes = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for code, colour in CLASS_COLOURS.items():
        hit = np.all(rgb == np.asarray(colour, dtype=np.uint8), axis=2)
        codes[hit] = code
    return codes


def tile_polygon_mask(rings, shape, crop_x, crop_y, scale) -> np.ndarray:
    """Rasterise step 9's tile staircase into the BEETLE mask's frame."""
    from PIL import ImageDraw

    canvas = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for ring in rings or []:
        if len(ring) < 3:
            continue
        points = [((x - crop_x) * scale, (y - crop_y) * scale) for x, y in ring]
        draw.polygon(points, fill=255)
    return np.asarray(canvas) > 127


def measure_region(directory: pathlib.Path) -> dict | None:
    """One ROI: the tile region, BEETLE's verdict on it, and the per-tile breakdown."""
    record = common.read_json(directory / "metadata.json")
    if not record or record.get("state") != "complete":
        return None

    codes = classes_from_png(directory / "beetle_mask.png")
    if codes is None:
        return None

    crop_x = float(record["cropX"])
    crop_y = float(record["cropY"])
    base_mpp = float(record["baseMpp"])
    # The PNG is written at the mask's own resolution, then saved; recover the actual
    # level-0 pixels per mask pixel from the image rather than trusting maskMpp, because
    # the renderer may have resized on the way out.
    mask_h, mask_w = codes.shape
    scale = mask_w / float(record["cropWidth"])  # mask px per level-0 px
    mask_mpp = base_mpp / scale
    px_mm2 = (mask_mpp / 1000.0) ** 2

    tiles_mask = tile_polygon_mask(record.get("tileRings"), codes.shape, crop_x, crop_y, scale)
    if not tiles_mask.any():
        return None

    invasive = (codes == SCORED_CODE) & tiles_mask
    tile_px = int(tiles_mask.sum())

    # Per tile. Step 8's windows are `fieldOfViewUm` across at level 0 and land on a
    # global grid, so the tile a pixel belongs to is its level-0 coordinate divided by
    # the window size - computed against the slide's own grid, not the crop's, or the
    # tiles would be offset by wherever the crop happened to start.
    field_px = float(record["fieldOfViewUm"]) / base_mpp
    ys, xs = np.nonzero(tiles_mask)
    level0_x = crop_x + xs / scale
    level0_y = crop_y + ys / scale
    tile_ix = np.floor(level0_x / field_px).astype(np.int64)
    tile_iy = np.floor(level0_y / field_px).astype(np.int64)
    key = tile_iy * (2**20) + tile_ix

    hit = invasive[ys, xs]
    order = np.argsort(key, kind="stable")
    key_sorted, hit_sorted = key[order], hit[order]
    boundaries = np.flatnonzero(np.diff(key_sorted)) + 1
    groups = np.split(np.arange(key_sorted.size), boundaries)

    per_tile = []
    for group in groups:
        if group.size == 0:
            continue
        index = int(key_sorted[group[0]])
        per_tile.append(
            {
                "tileX": index % (2**20),
                "tileY": index // (2**20),
                "tilePixels": int(group.size),
                "invasivePixels": int(hit_sorted[group].sum()),
                "share": round(float(hit_sorted[group].mean()), 4),
            }
        )

    shares = np.asarray([one["share"] for one in per_tile], dtype=float)
    contour_mm2 = float(record.get("areaMm2") or 0.0)
    tile_mm2 = float(record.get("tileAreaMm2") or 0.0)

    return {
        "roiId": record.get("roiId"),
        "tileAreaMm2": round(tile_mm2, 4),
        "maskInvasiveMm2": round(float(invasive.sum()) * px_mm2, 4),
        "contourInvasiveMm2": round(contour_mm2, 4),
        "maskShare": round(float(invasive.sum()) / tile_px, 4) if tile_px else 0.0,
        "contourShare": round(contour_mm2 / tile_mm2, 4) if tile_mm2 else 0.0,
        "tiles": len(per_tile),
        # The distribution behind the average. A median far below the mean means a few
        # tiles carry the region; a tight spread means the grid is uniformly over-reaching.
        "tileShareMean": round(float(shares.mean()), 4) if shares.size else 0.0,
        "tileShareMedian": round(float(np.median(shares)), 4) if shares.size else 0.0,
        "tileSharesP10": round(float(np.percentile(shares, 10)), 4) if shares.size else 0.0,
        "tileSharesP90": round(float(np.percentile(shares, 90)), 4) if shares.size else 0.0,
        "tilesMostlyInvasive": int((shares >= 0.5).sum()),
        "tilesNearlyEmpty": int((shares < 0.05).sum()),
        "classShare": record.get("classShare") or {},
        "perTile": per_tile,
    }


def collect() -> tuple[list[dict], list[dict], list[dict]]:
    """Walk every refinement on disk and measure it at all three grains."""
    owner = slide_owner()
    slide_rows, region_rows, tile_rows = [], [], []

    for upload_dir in refinement_dirs():
        upload = upload_dir.name
        case, role = owner.get(upload, ("unknown", "unknown"))

        regions = []
        for roi_dir in sorted(upload_dir.glob("ROI-*")):
            measured = measure_region(roi_dir)
            if measured is None:
                continue
            regions.append(measured)

            region_rows.append(
                {
                    "case": case,
                    "slideRole": role,
                    "uploadId": upload,
                    "roiId": measured["roiId"],
                    "tileAreaMm2": measured["tileAreaMm2"],
                    "maskInvasiveMm2": measured["maskInvasiveMm2"],
                    "contourInvasiveMm2": measured["contourInvasiveMm2"],
                    "maskSharePct": round(measured["maskShare"] * 100, 2),
                    "contourSharePct": round(measured["contourShare"] * 100, 2),
                    "tiles": measured["tiles"],
                    "tileShareMeanPct": round(measured["tileShareMean"] * 100, 2),
                    "tileShareMedianPct": round(measured["tileShareMedian"] * 100, 2),
                    "tileShareP10Pct": round(measured["tileSharesP10"] * 100, 2),
                    "tileShareP90Pct": round(measured["tileSharesP90"] * 100, 2),
                    "tilesMostlyInvasive": measured["tilesMostlyInvasive"],
                    "tilesNearlyEmpty": measured["tilesNearlyEmpty"],
                }
            )
            for tile in measured["perTile"]:
                tile_rows.append(
                    {
                        "case": case,
                        "slideRole": role,
                        "uploadId": upload,
                        "roiId": measured["roiId"],
                        "tileX": tile["tileX"],
                        "tileY": tile["tileY"],
                        "tilePixels": tile["tilePixels"],
                        "invasivePixels": tile["invasivePixels"],
                        "invasiveSharePct": round(tile["share"] * 100, 2),
                    }
                )

        if not regions:
            continue
        tile_mm2 = sum(one["tileAreaMm2"] for one in regions)
        mask_mm2 = sum(one["maskInvasiveMm2"] for one in regions)
        contour_mm2 = sum(one["contourInvasiveMm2"] for one in regions)
        all_tiles = sum(one["tiles"] for one in regions)
        mostly = sum(one["tilesMostlyInvasive"] for one in regions)
        empty = sum(one["tilesNearlyEmpty"] for one in regions)
        slide_rows.append(
            {
                "case": case,
                "slideRole": role,
                "uploadId": upload,
                "regions": len(regions),
                "tileAreaMm2": round(tile_mm2, 4),
                "maskInvasiveMm2": round(mask_mm2, 4),
                "contourInvasiveMm2": round(contour_mm2, 4),
                "maskSharePct": round(mask_mm2 / tile_mm2 * 100, 2) if tile_mm2 else 0.0,
                "contourSharePct": round(contour_mm2 / tile_mm2 * 100, 2) if tile_mm2 else 0.0,
                "tiles": all_tiles,
                "tilesMostlyInvasive": mostly,
                "tilesNearlyEmpty": empty,
                "tilesMostlyInvasivePct": round(mostly / all_tiles * 100, 2) if all_tiles else 0.0,
                "tilesNearlyEmptyPct": round(empty / all_tiles * 100, 2) if all_tiles else 0.0,
            }
        )

    return slide_rows, region_rows, tile_rows


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    """Rebuilt from scratch every time, never appended to.

    Appending means a slide measured twice appears twice, and a run interrupted mid-write
    leaves a half-row a spreadsheet reads as data. Rebuilding makes the file a pure
    function of what is on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(common.RESULTS))
    args = parser.parse_args()

    out = pathlib.Path(args.out_dir)
    slide_rows, region_rows, tile_rows = collect()

    write_csv(out / "beetle_coverage_by_slide.csv", slide_rows)
    write_csv(out / "beetle_coverage_by_region.csv", region_rows)
    write_csv(out / "beetle_coverage_by_tile.csv", tile_rows)

    common.say(
        f"BEETLE coverage: {len(slide_rows)} slide(s), {len(region_rows)} region(s), "
        f"{len(tile_rows)} tile(s)"
    )
    for row in slide_rows:
        common.say(
            f"  {row['case']}/{row['slideRole']}: tiles claimed {row['tileAreaMm2']} mm2, "
            f"BEETLE kept {row['contourSharePct']}% (raw mask {row['maskSharePct']}%), "
            f"{row['tilesMostlyInvasivePct']}% of tiles mostly invasive, "
            f"{row['tilesNearlyEmptyPct']}% nearly empty"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
