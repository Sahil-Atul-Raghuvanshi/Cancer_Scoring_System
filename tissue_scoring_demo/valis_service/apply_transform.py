"""Apply a stored SimpleITK transform to a list of points. JSON in, JSON out.

    python apply_transform.py request.json response.json

Lives here rather than in the backend for one reason: SimpleITK is installed in this
virtual environment and not in the backend's. The work itself is microseconds - loading a
transform and mapping a few thousand points - so the subprocess hop costs far more than
the arithmetic, and that is fine. It happens once per pair, not once per point.

**No coordinate conversion happens here.** The caller passes points already in the
transform's own space and converts back itself. Splitting that responsibility would mean
two places could disagree about what a coordinate means, which is the single most
expensive kind of bug in this pipeline: a mask on the wrong tissue looks exactly like a
mask on the right one.

The round trip is measured and returned, because a transform that does not agree with
itself over the points actually being moved is not one to carry a mask with - and unlike a
keypoint residual, that check needs no features to be meaningful.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUEST = Path(sys.argv[1]) if len(sys.argv) > 1 else None
RESPONSE = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def main(request: dict) -> dict:
    import numpy as np
    import SimpleITK as sitk

    transform = sitk.ReadTransform(str(request["transform"]))

    # The inverse is needed for the round-trip check. A B-spline has no closed-form
    # inverse, so SimpleITK iterates for one; when that is unavailable the check is
    # reported as unavailable rather than silently skipped.
    inverse = None
    try:
        inverse = transform.GetInverse()
    except Exception:  # noqa: BLE001 - not every transform can be inverted
        inverse = None

    out_rings = []
    round_trip = []
    for ring in request["rings"]:
        if not ring:
            out_rings.append([])
            continue
        mapped = [transform.TransformPoint((float(x), float(y))) for x, y in ring]
        out_rings.append([[float(p[0]), float(p[1])] for p in mapped])
        if inverse is not None:
            back = [inverse.TransformPoint(p) for p in mapped]
            original = np.asarray(ring, dtype=float)
            returned = np.asarray(back, dtype=float)
            round_trip.extend(np.linalg.norm(returned - original, axis=1).tolist())

    result = {"ok": True, "warped_rings": out_rings}
    if round_trip:
        values = sorted(round_trip)
        result["round_trip_median_px"] = round(values[len(values) // 2], 4)
        result["round_trip_max_px"] = round(values[-1], 4)
    else:
        result["round_trip_note"] = "this transform could not be inverted, so no round trip"
    return result


if __name__ == "__main__":
    if REQUEST is None or RESPONSE is None:
        print("usage: apply_transform.py <request.json> <response.json>", file=sys.stderr)
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
