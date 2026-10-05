"""Measure BRACS's resolution instead of assuming it, by comparing nuclei with BCSS.

`config.BRACS_MPP = 0.25` is the one number in this app that nothing on disk confirms.
BCSS writes its resolution into every filename (`..._MPP-0.2500.png`); BRACS ships no
sidecar, no EXIF and no convention, so 0.25 is inferred from "the WSIs were scanned at
40x" and nothing else.

It is also a number that is expensive to get wrong in a way that never raises. It sets
the physical size of a training tile - 224 px is meant to be 112 um, about nine cells,
because that is the smallest field that can show whether epithelium sits inside a duct
or has broken out of one. At the wrong scale the tiles still export, still look like
tiles, and pose a different question from the one BCSS's tiles pose.

**The measurement.** A cell nucleus is the same physical size in both datasets - about
7 um across for breast epithelium - so if BRACS nuclei are the same size *in pixels* as
BCSS nuclei, the two datasets are at the same resolution. Nuclei are found the same way
in both: haematoxylin optical density above a per-image threshold, connected components,
median area. Using approach 1's own `hchannel` for the density means the two sides are
measured by identical code, which is the only reason the comparison means anything.

This is a scale check, not a calibration. It answers "is BRACS at 0.25 or is it at 0.5",
which is the mistake worth catching; it will not resolve 0.25 from 0.27.

    python scripts/check_resolution.py
    python scripts/check_resolution.py --n 24
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from bracs_app import config  # noqa: E402

config.install_approach1_path()
import hchannel  # noqa: E402  - approach 1's, and deliberately not a copy

Image.MAX_IMAGE_PIXELS = None

#: A crop rather than a whole region: 1024 px carries thousands of nuclei, which is far
#: more than enough for a median, and reading 40 MPx per image would make this take
#: minutes for no extra precision.
CROP = 1024

#: Optical density above which a pixel is called nuclear. Haematoxylin in a nucleus runs
#: well above this and the surrounding cytoplasm and stroma well below; the threshold
#: only has to separate those two populations, not find a boundary precisely.
NUCLEAR_OD = 0.35

#: Connected components outside this range in pixels are not nuclei at 0.25 um/px - the
#: small ones are noise and lymphocyte fragments, the large ones are merged clumps and
#: dark artefacts. Both tails would drag a mean; the median is used anyway, and this
#: keeps the population being taken a median *of* honest.
AREA_RANGE = (30, 3000)


def centre_crop(path: Path, size: int = CROP) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        left = max(0, (image.width - size) // 2)
        top = max(0, (image.height - size) // 2)
        return np.asarray(image.crop((left, top, left + size, top + size)))


def median_nucleus_area(rgb: np.ndarray) -> float | None:
    """Median connected-component area, in pixels, of the haematoxylin-dense regions."""
    white = hchannel.white_point(rgb, percentile=99.0)
    density = hchannel.haematoxylin_od(rgb, white)

    labelled, count = ndimage.label(density > NUCLEAR_OD)
    if count == 0:
        return None
    areas = np.bincount(labelled.ravel())[1:]
    kept = areas[(areas >= AREA_RANGE[0]) & (areas <= AREA_RANGE[1])]
    return float(np.median(kept)) if kept.size >= 20 else None


def sample(paths: list[Path], n: int, seed: int = 0) -> list[Path]:
    return paths if len(paths) <= n else random.Random(seed).sample(paths, n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16, help="images per dataset")
    args = parser.parse_args()

    bcss_dir = config.ORIGINAL_DATA / "bcss" / "images"
    bracs_dir = config.DCIS_DIR / "train"
    if not bcss_dir.is_dir() or not bracs_dir.is_dir():
        print("need both BCSS images and BRACS regions on disk", file=sys.stderr)
        return 2

    results: dict[str, list[float]] = {}
    for name, directory in (("BCSS", bcss_dir), ("BRACS", bracs_dir)):
        paths = sample(sorted(directory.glob("*.png")), args.n)
        print(f"{name}: measuring {len(paths)} images", file=sys.stderr)
        areas = []
        for path in paths:
            area = median_nucleus_area(centre_crop(path))
            if area is not None:
                areas.append(area)
        results[name] = areas

    if not results["BCSS"] or not results["BRACS"]:
        print("not enough nuclei found to compare", file=sys.stderr)
        return 2

    bcss_area = float(np.median(results["BCSS"]))
    bracs_area = float(np.median(results["BRACS"]))
    # Area scales as the square of the linear resolution, so the linear ratio - which is
    # what microns per pixel is - is the square root.
    linear = float(np.sqrt(bcss_area / bracs_area))
    implied = 0.25 * linear

    print()
    print(f"  BCSS  median nucleus area   {bcss_area:8.1f} px   (known 0.2500 um/px)")
    print(f"  BRACS median nucleus area   {bracs_area:8.1f} px")
    print(f"  linear scale BRACS/BCSS     {1 / linear:8.3f}")
    print(f"  implied BRACS resolution    {implied:8.4f} um/px")
    print(f"  configured in this app      {config.BRACS_MPP:8.4f} um/px")
    print()

    ratio = implied / config.BRACS_MPP
    if 0.8 <= ratio <= 1.25:
        print("  CONSISTENT - the assumption survives the measurement.")
        print("  Nuclei are the same physical size in both datasets, so a 224 px tile")
        print(f"  covers {224 * config.TILE_MPP:.0f} um of BRACS tissue as intended.")
        return 0

    print(f"  INCONSISTENT - measured {implied:.3f} against configured {config.BRACS_MPP:.3f}.")
    print("  Before trusting any tile from this app, check whether BRACS ships at a")
    print("  different magnification than assumed. A factor near 2 means 20x, not 40x.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
