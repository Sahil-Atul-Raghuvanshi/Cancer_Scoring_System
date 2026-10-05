"""Is a fitted transform anatomically possible? JSON in, JSON out.

    python check_plausible.py request.json response.json

**Mutual information cannot answer this, and that is why this file exists.** A free-form
B-spline has enough freedom to fold tissue over itself, tear it, or stretch one region to
three times its size, and any of those can *raise* mutual information by lining up
intensity statistics. The result would score well and be biologically impossible - and
since the transform's job here is to carry an invasive-tumour boundary that the final
percentages are measured inside, an impossible warp puts the mask on the wrong cells while
every number downstream looks entirely healthy.

Three things are measured, on a grid covering the tissue:

    Jacobian determinant   the local area change the transform applies. **Negative
                           anywhere means the tissue has been folded through itself** -
                           topologically impossible and immediately disqualifying. Far
                           from 1 means implausible local stretching or compression.
    displacement           how far the transform actually moves tissue, in microns.
                           Serial sections of one block sit within a few hundred microns
                           of each other; millimetres of local warp is not a registration.
    non-rigid excess       how much of that displacement the free-form part adds on top of
                           the rigid/affine part it was initialised from. This is the part
                           that can invent deformation, so it is reported on its own.

The Jacobian is computed numerically by sampling the transform rather than analytically,
so it works for every transform type this pipeline stores - affine, similarity, B-spline
or a composite of them - without a special case per type.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUEST = Path(sys.argv[1]) if len(sys.argv) > 1 else None
RESPONSE = Path(sys.argv[2]) if len(sys.argv) > 2 else None

#: Spacing of the sampling grid, in working pixels. Fine enough to catch a fold inside one
#: B-spline control cell (the mesh is 8x8 over ~1000 px, so a cell is ~125 px).
GRID_STEP = 8

#: Step used for the numerical derivative. Small against the control-cell size, large
#: against floating-point noise.
EPS = 1.0


def main(request: dict) -> dict:
    import numpy as np
    import SimpleITK as sitk
    from PIL import Image

    transform = sitk.ReadTransform(str(request["transform"]))
    mpp = float(request["mpp_at_work"])

    # Sample only where there is tissue: a fold in empty canvas is not a finding, and
    # including it would drown the real signal.
    mask_img = Image.open(request["mask"]).convert("L")
    work_dim = int(request["work_dim"])
    scale = min(1.0, work_dim / max(mask_img.size))
    if scale < 1.0:
        mask_img = mask_img.resize(
            (max(1, int(mask_img.size[0] * scale)), max(1, int(mask_img.size[1] * scale))),
            Image.NEAREST,
        )
    mask = np.asarray(mask_img) > 127

    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return {"ok": False, "error": "no tissue in the mask"}
    keep = (rows % GRID_STEP == 0) & (cols % GRID_STEP == 0)
    rows, cols = rows[keep], cols[keep]
    if rows.size < 50:
        rows, cols = np.nonzero(mask)

    def warp(xs, ys):
        return np.array([transform.TransformPoint((float(x), float(y))) for x, y in zip(xs, ys)])

    base = warp(cols, rows)
    dx = warp(cols + EPS, rows)
    dy = warp(cols, rows + EPS)

    # Jacobian of the map, column by column: how a unit step in x and in y is transformed.
    j11 = (dx[:, 0] - base[:, 0]) / EPS
    j21 = (dx[:, 1] - base[:, 1]) / EPS
    j12 = (dy[:, 0] - base[:, 0]) / EPS
    j22 = (dy[:, 1] - base[:, 1]) / EPS
    det = j11 * j22 - j12 * j21

    displacement = np.linalg.norm(base - np.column_stack([cols, rows]), axis=1) * mpp

    result = {
        "ok": True,
        "points": int(rows.size),
        "jacobian_min": round(float(det.min()), 4),
        "jacobian_max": round(float(det.max()), 4),
        "jacobian_median": round(float(np.median(det)), 4),
        "jacobian_p01": round(float(np.percentile(det, 1)), 4),
        "jacobian_p99": round(float(np.percentile(det, 99)), 4),
        # The disqualifying one. Any negative determinant is tissue folded through itself.
        "folded_fraction": round(float((det <= 0).mean()), 6),
        "folded_points": int((det <= 0).sum()),
        "displacement_median_um": round(float(np.median(displacement)), 1),
        "displacement_p95_um": round(float(np.percentile(displacement, 95)), 1),
        "displacement_max_um": round(float(displacement.max()), 1),
    }

    # How much of the displacement the deformable part added, if a rigid comparison was
    # supplied. Reported separately because that is the part free to invent anatomy.
    rigid_path = request.get("rigid_transform")
    if rigid_path and Path(rigid_path).is_file():
        rigid = sitk.ReadTransform(str(rigid_path))
        rigid_base = np.array(
            [rigid.TransformPoint((float(x), float(y))) for x, y in zip(cols, rows)]
        )
        excess = np.linalg.norm(base - rigid_base, axis=1) * mpp
        result["nonrigid_excess_median_um"] = round(float(np.median(excess)), 1)
        result["nonrigid_excess_p95_um"] = round(float(np.percentile(excess, 95)), 1)
        result["nonrigid_excess_max_um"] = round(float(excess.max()), 1)

    return result


if __name__ == "__main__":
    if REQUEST is None or RESPONSE is None:
        print("usage: check_plausible.py <request.json> <response.json>", file=sys.stderr)
        sys.exit(2)
    try:
        payload = main(json.loads(REQUEST.read_text(encoding="utf-8")))
        code = 0
    except Exception as exc:  # noqa: BLE001 - every failure is reported as data
        import traceback

        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-3000:]}
        code = 1
    RESPONSE.write_text(json.dumps(payload), encoding="utf-8")
    sys.exit(code)
