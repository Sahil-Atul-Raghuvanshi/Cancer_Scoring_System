"""Tissue areas for the gate's ratio test, measured the same way on every slide of a case.

    python gate_areas.py                 every case
    python gate_areas.py --only CAN_00270
    python gate_areas.py --validate      compare against the published figures

**A ratio between two measurements is only meaningful if both were measured the same
way**, and neither rule this pipeline has used satisfied that.

*   Step 3's mask keys off colour **saturation**, so it measures how stained a slide is.
    It put CAN_00865's H&E at 357 mm² against a published 74.3 and refused four good
    markers.
*   `render.py` fixed the rule but not the comparison: it chooses an Otsu threshold **per
    slide**, and Otsu reliably picks a lower cut for an H&E than for a pale IHC section -
    0.190 against 0.251 on CAN_00270. The H&E is then measured generously and the IHC
    conservatively, and the ratio between them is systematically depressed. Two markers
    that had scored fine were refused.

So the cut is chosen **once per case, from the H&E**, and applied unchanged to every
section of that block. The H&E is the reference every ROI is drawn on, so it is the
natural slide to calibrate against; and whatever bias the cut carries, it now falls on
both sides of the ratio equally, which is the only property the ratio actually needs.

These areas are for the **gate** and nothing else. `render.py` keeps its per-slide Otsu
for placement, where a tight, well-chosen mask per slide is exactly right and no
comparison is being made.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from scipy import ndimage

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tissue_scoring_demo" / "backend"))

import common  # noqa: E402
import render  # noqa: E402

#: Resolution the areas are measured at. Matches the renders, so the numbers are
#: comparable with everything else this directory produces.
MPP = 8.0

#: The published figures, for `--validate`. From `cohort_characterisation.md` section 3,
#: measured independently of anything here.
PUBLISHED = {
    "CAN_00251": {"HE": 166.3, "A": 95.0, "F": 176.8, "R": 133.6, "U": 203.4, "W": 129.1},
    "CAN_00259": {"HE": 315.4, "A": 261.1, "F": 368.5, "R": 295.6, "U": 338.5, "W": 315.5},
    "CAN_00267": {"HE": 163.5, "A": 39.3, "F": 160.0, "R": 144.6, "U": 293.1, "W": 114.0},
    "CAN_00270": {"HE": 189.8, "A": 101.2, "F": 228.3, "R": 177.1, "U": 247.4, "W": 123.0},
    "CAN_00303": {"HE": 35.8, "A": 31.6, "F": 35.9, "R": 33.4, "U": 43.6, "W": 31.5},
    "CAN_00865": {"HE": 74.3, "A": 17.5, "F": 91.1, "R": 84.8, "U": 96.1, "W": 84.8},
}


def _area_at(path: pathlib.Path, cut: float | None) -> tuple[float, float]:
    """Tissue area in mm2 at a given optical-density cut, and the cut Otsu would choose.

    Padding is excluded by flatness first, exactly as `render.py` does - that part was
    never the problem.
    """
    rgb, _base, level = render.read_at(path, MPP)
    optical = render.absorbance(rgb)

    smooth = ndimage.uniform_filter(optical, size=5)
    variance = ndimage.uniform_filter(optical * optical, size=5) - smooth * smooth
    considered = (np.sqrt(np.clip(variance, 0, None)) * 255.0) > render.PADDING_STD
    if considered.sum() < optical.size * 0.01:
        considered = np.ones_like(considered)

    otsu = render._otsu(optical[considered], render.OD_FLOOR, render.OD_CEILING)
    use = otsu if cut is None else cut

    mask = (optical > use) & considered
    radius = max(1, int(round(30.0 / level)))
    mask = ndimage.binary_closing(mask, np.ones((3, 3)), iterations=radius)
    return float(mask.sum()) * (level / 1000.0) ** 2, otsu


def measure_case(case: str, log=None) -> dict:
    """Every slide of one case, at the cut the H&E chose."""
    slides = common.slides_for(case)
    if "HE" not in slides:
        raise ValueError(f"{case}: no H&E")

    # The reference section picks the cut; everything else is measured with it.
    he_area, cut = _area_at(slides["HE"], None)
    common.say(f"{case}: H&E chose cut {cut:.3f} -> {he_area:.1f} mm2", log)

    areas = {"HE": round(he_area, 2)}
    for code, path in sorted(slides.items()):
        if code == "HE":
            continue
        area, _ = _area_at(path, cut)
        areas[code] = round(area, 2)
        ratio = area / he_area if he_area else 0.0
        common.say(f"{case}:   {code} {area:8.1f} mm2  ratio {ratio:.2f}", log)

    payload = {"case": case, "mpp": MPP, "cut": round(cut, 4), "areas": areas,
               "rule": "optical density, one cut per case chosen from the H&E"}
    common.write_json(common.case_dir(case) / "gate_areas.json", payload)
    return payload


def validate() -> int:
    """Do the shared-cut ratios agree with the published ones on which side of the gate?"""
    print(f"{'case':12} {'mk':3} {'mine':>8} {'published':>10} {'my ratio':>9} "
          f"{'pub ratio':>10}  verdict")
    print("-" * 74)
    wrong = 0
    for case, published in sorted(PUBLISHED.items()):
        payload = common.read_json(common.case_dir(case) / "gate_areas.json")
        if not payload:
            print(f"{case:12} (not measured)")
            continue
        areas = payload["areas"]
        he, pub_he = areas.get("HE"), published["HE"]
        for code in ("A", "F", "R", "U", "W"):
            if code not in areas:
                continue
            mine = areas[code] / he if he else 0.0
            theirs = published[code] / pub_he
            same = (mine >= 0.5) == (theirs >= 0.5)
            if not same:
                wrong += 1
            print(f"{case:12} {code:3} {areas[code]:8.1f} {published[code]:10.1f} "
                  f"{mine:9.2f} {theirs:10.2f}  "
                  f"{'agree' if same else 'DISAGREE'}")
    print("-" * 74)
    print(f"{wrong} pair(s) where my ratio and the published one fall on different sides "
          f"of the 0.5 gate")
    return 0 if wrong == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    common.ensure_dirs()

    if args.validate:
        return validate()

    names = [c.strip() for c in args.only.split(",") if c.strip()] or common.cases()
    for case in names:
        try:
            measure_case(case, common.RUN_LOG)
        except Exception as failure:  # noqa: BLE001
            common.say(f"{case}: FAILED - {type(failure).__name__}: {failure}", common.RUN_LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
