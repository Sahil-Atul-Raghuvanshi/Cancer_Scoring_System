"""Phase 3 - rotate every section into the H&E's frame and record the affine.

    python align.py CAN_00303

Writes `aligned/`: each Phase-1 render turned by the angle Phase 2 measured against the
H&E, with the full level-0-to-aligned affine recorded beside it in `aligned.json`. That is
the frame `methods_run.py` fits its transforms in and `transform_warp.py` maps points
through.

Split out of the retired `stack.py`, whose other half handed these images to VALIS. VALIS
is no longer used - Mattes mutual information over a similarity transform replaced it
(`tissue_scoring_demo/docs/registration/REGISTRATION-PLAN.md` section 14) - but the
rotation is still what lets every method start from one orientation.

**Rotating first is not a nicety.** Two of CAN_00267's sections are mounted about 180
degrees round. Turning the image is stain-blind, costs milliseconds, and removes that
failure rather than asking a registration to survive it.

**The affine is written down, never re-derived.** Everything downstream of registration is
expressed in a slide's own level-0 pixels. The chain from there to the aligned image is:
scale by mpp, translate to centre the tissue, rotate about the canvas centre. Three steps,
each trivial, and composing them by eye at the call site is how an off-by-a-scale-factor
bug gets in. So it is composed once here, stored as a 2x3 matrix, and inverted
numerically when a point has to come back.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402


def affine_for(scale: float, offset, angle_deg: float, centre: float) -> np.ndarray:
    """level-0 pixels -> aligned-image pixels, as a 2x3 matrix.

    Composed in the order the pixels actually move: scale to the render's resolution,
    translate so the tissue centroid sits at the canvas centre, then rotate about that
    centre by the angle measured against the H&E.
    """
    place = np.array([[scale, 0.0, offset[0]], [0.0, scale, offset[1]], [0.0, 0.0, 1.0]])
    radians = math.radians(angle_deg)
    cos, sin = math.cos(radians), math.sin(radians)
    turn = np.array(
        [
            [cos, -sin, centre - centre * cos + centre * sin],
            [sin, cos, centre - centre * sin - centre * cos],
            [0.0, 0.0, 1.0],
        ]
    )
    return (turn @ place)[:2]


def write_aligned(case: str, log=None) -> dict:
    """Turn each render into the H&E's frame and record the transform."""
    case_dir = common.case_dir(case)
    manifest = common.read_json(case_dir / "render.json")
    measured = common.read_json(case_dir / "order.json")
    if manifest is None or measured is None:
        raise ValueError(f"{case}: needs phase 1 and phase 2 first")

    out = case_dir / "aligned"
    out.mkdir(parents=True, exist_ok=True)
    side = int(manifest["canvas"][0])
    centre = side / 2.0

    aligned = {"case": case, "canvas": manifest["canvas"], "targetMpp": manifest["targetMpp"], "slides": {}}
    for code, entry in manifest["slides"].items():
        angle = float(measured["rotationToHE"].get(code, 0.0))
        image = Image.open(case_dir / "render" / entry["file"]).convert("RGB")
        # PIL rotates anticlockwise about the centre for a positive angle, and expand=False
        # keeps the canvas identical across all six, so every method sees six images of
        # exactly the same size.
        turned = image.rotate(-angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        turned.save(out / f"{code}.png")

        matrix = affine_for(entry["scale"], entry["offset"], angle, centre)
        aligned["slides"][code] = {
            "file": f"{code}.png",
            "source": entry["source"],
            "marker": entry.get("marker"),
            "rotationDeg": angle,
            "tissueMm2": entry["tissueMm2"],
            # aligned_xy = M @ [level0_x, level0_y, 1]
            "level0ToAligned": matrix.tolist(),
        }
        common.say(f"{case}:   aligned {code} turned {angle:.1f} deg", log)

    common.write_json(case_dir / "aligned.json", aligned)
    return aligned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    args = parser.parse_args()
    common.ensure_dirs()
    write_aligned(args.case, log=common.RUN_LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
