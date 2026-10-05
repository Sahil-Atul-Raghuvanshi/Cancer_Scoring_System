"""Register one case's sections by several methods that do not depend on feature matching.

    python register_methods.py request.json response.json

**Why this exists.** VALIS estimates its transform from detected and matched image
features, so when detection fails the whole thing fails - and on this cohort it fails
exactly where the section is nearly unstained, because a near-negative slide has tissue
with almost no absorbance contrast for a detector to key on. CAN_00267's CD44 section
returns thirteen matched features and CAN_00865's returns five, however the images are
rotated, whatever detector is used, at any resolution tried.

Feature matching is not the only way to register two images. The methods here need no
features at all:

    outline     the identity: what phases 1-2 already achieved by rendering both sections
                to one physical scale, cropping to tissue and rotating by the angle
                measured from the tissue outline. Included as a method so it competes on
                the same scoreboard rather than being assumed to be the floor.
    mask        a similarity transform (translation, rotation, uniform scale) fitted by
                maximising agreement between the two TISSUE masks. Geometry only, blind to
                stain. Fitted on the whole-tissue mask, never on the invasive mask - two
                tumour regions do not constrain a transform, a section outline does.
    mattes      Mattes mutual information, optimised over a similarity transform with a
                multi-resolution pyramid. **The important one.** Mutual information is the
                standard metric for registering images that do not look alike, and it uses
                every pixel rather than a few hundred keypoints, so a section with no
                crisp features but real tissue still carries signal.
    affine      the same, with shear and anisotropic scale allowed, initialised from the
                mattes result.
    bspline     a free-form non-rigid refinement on top of affine, which is the only
                method here that can follow the local deformation a section picks up
                being floated onto glass.

**Every method is scored the same way and the scoreboard decides.** The score is
normalised mutual information between the two sections' absorbance over a lattice covering
the H&E's tissue - the same probe `register_stack.py` uses, so these numbers are directly
comparable with VALIS's. Critically it is computed on the *result*, not on whatever the
method optimised, so a method cannot win by being graded on its own objective.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REQUEST = Path(sys.argv[1]) if len(sys.argv) > 1 else None
RESPONSE = Path(sys.argv[2]) if len(sys.argv) > 2 else None

#: Longest edge the intensity methods optimise at. They are O(pixels) per iteration and
#: multi-resolution already handles the coarse work, so paying for more than this buys
#: little: the renders are 8 um/px and a 1024 px working image is ~16 um/px on a large
#: section, finer than the deformation being corrected.
WORK_DIM = 1024

#: Points in the quality probe. Enough for a stable 32x32 joint histogram, small enough
#: that scoring six methods costs seconds.
PROBE_POINTS = 30_000


def absorbance(rgb):
    """How much light the tissue took out. Not saturation - see `render.py`."""
    import numpy as np

    grey = np.asarray(rgb[..., :3], dtype=np.float64).mean(axis=2) / 255.0
    return -np.log10(np.clip(grey, 1e-3, 1.0))


def nmi(first, second, bins: int = 32) -> float:
    """Normalised mutual information. 1.0 means the two are independent; higher is better."""
    import numpy as np

    histogram, _, _ = np.histogram2d(first, second, bins=bins)
    joint = histogram / max(1.0, histogram.sum())

    def entropy(values):
        nonzero = values[values > 0]
        return float(-(nonzero * np.log(nonzero)).sum())

    joint_entropy = entropy(joint.ravel())
    if joint_entropy <= 0:
        return 0.0
    return (entropy(joint.sum(axis=1)) + entropy(joint.sum(axis=0))) / joint_entropy


def score_sitk(fixed_od, moving_od, fixed_mask, transform):
    """NMI for a transform that is not expressible as a matrix (the non-rigid ones)."""
    import numpy as np

    rows, cols = np.nonzero(fixed_mask)
    if rows.size == 0:
        return 0.0, 0.0
    stride = max(1, rows.size // PROBE_POINTS)
    rows, cols = rows[::stride], cols[::stride]

    mapped = np.array([transform.TransformPoint((float(c), float(r))) for c, r in zip(cols, rows)])
    height, width = moving_od.shape
    sample_c = np.round(mapped[:, 0]).astype(int)
    sample_r = np.round(mapped[:, 1]).astype(int)
    inside = (sample_r >= 0) & (sample_r < height) & (sample_c >= 0) & (sample_c < width)
    if inside.sum() < 100:
        return 0.0, 0.0
    return (
        nmi(fixed_od[rows[inside], cols[inside]], moving_od[sample_r[inside], sample_c[inside]]),
        float(inside.mean()),
    )


def score_transform(fixed_od, moving_od, fixed_mask, matrix):
    """NMI between the two sections once `matrix` is applied to the moving image.

    `matrix` maps fixed pixel coordinates to moving pixel coordinates, so it is what a
    resampler wants and needs no inversion here. Sampled over the fixed image's tissue
    only, because empty matching empty scores well under any transform at all.
    """
    import numpy as np

    rows, cols = np.nonzero(fixed_mask)
    if rows.size == 0:
        return 0.0, 0.0
    stride = max(1, rows.size // PROBE_POINTS)
    rows, cols = rows[::stride], cols[::stride]

    ones = np.ones(len(rows))
    source = np.column_stack([cols, rows, ones])
    mapped = source @ np.asarray(matrix).T

    height, width = moving_od.shape
    sample_c = np.round(mapped[:, 0]).astype(int)
    sample_r = np.round(mapped[:, 1]).astype(int)
    inside = (sample_r >= 0) & (sample_r < height) & (sample_c >= 0) & (sample_c < width)
    if inside.sum() < 100:
        return 0.0, 0.0
    return (
        nmi(fixed_od[rows[inside], cols[inside]], moving_od[sample_r[inside], sample_c[inside]]),
        float(inside.mean()),
    )


# --- the methods -------------------------------------------------------------


def method_outline(fixed_od, moving_od, fixed_mask, moving_mask):
    """The identity. Phases 1-2 already put both sections on one scale and orientation."""
    import numpy as np

    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), {}


def method_mask(fixed_od, moving_od, fixed_mask, moving_mask):
    """A similarity transform fitted to the two tissue masks. Geometry only.

    Optimised on the *distance transform* of the moving mask rather than on raw overlap,
    because IoU between two binary masks is piecewise constant - it gives an optimiser no
    gradient to follow until the masks already touch. A distance transform turns the same
    objective into a smooth basin that a downhill method can actually descend.
    """
    import numpy as np
    from scipy import ndimage, optimize

    # Distance from the moving tissue, in pixels: 0 inside, growing outside.
    outside = ndimage.distance_transform_edt(~moving_mask)

    rows, cols = np.nonzero(fixed_mask)
    stride = max(1, rows.size // 20_000)
    rows, cols = rows[::stride], cols[::stride]
    centre = np.array([cols.mean(), rows.mean()])
    points = np.column_stack([cols, rows]).astype(float)

    height, width = moving_mask.shape

    def matrix_for(params):
        dx, dy, theta, log_scale = params
        scale = np.exp(log_scale)
        cos, sin = np.cos(theta) * scale, np.sin(theta) * scale
        linear = np.array([[cos, -sin], [sin, cos]])
        offset = centre + np.array([dx, dy]) - linear @ centre
        return np.column_stack([linear, offset])

    def cost(params):
        matrix = matrix_for(params)
        mapped = points @ matrix[:, :2].T + matrix[:, 2]
        c = np.clip(np.round(mapped[:, 0]).astype(int), 0, width - 1)
        r = np.clip(np.round(mapped[:, 1]).astype(int), 0, height - 1)
        return float(outside[r, c].mean())

    best = optimize.minimize(
        cost,
        x0=np.zeros(4),
        method="Powell",
        options={"maxiter": 2000, "xtol": 0.05, "ftol": 0.01},
    )
    return matrix_for(best.x), {"mask_cost_px": round(float(best.fun), 3)}


def _sitk_image(array):
    import SimpleITK as sitk

    return sitk.GetImageFromArray(array.astype("float32"))


def _sitk_matrix(transform, shape):
    """A SimpleITK 2D transform as a 2x3 matrix mapping fixed pixels to moving pixels.

    SimpleITK's `TransformPoint` maps *physical* points, and these images are created with
    the default identity spacing and origin, so physical coordinates and pixel indices
    coincide. Three points determine an affine map, so it is read off rather than
    reconstructed from the parameter vector - which is how a centre-of-rotation term gets
    dropped by mistake.
    """
    import numpy as np

    height, width = shape
    probes = [(0.0, 0.0), (float(width), 0.0), (0.0, float(height))]
    mapped = [transform.TransformPoint(p) for p in probes]
    origin = np.array(mapped[0])
    column_x = (np.array(mapped[1]) - origin) / width
    column_y = (np.array(mapped[2]) - origin) / height
    return np.column_stack([column_x, column_y, origin])


def _mattes(fixed_od, moving_od, fixed_mask, transform_kind, initial=None, iterations=300):
    """Optimise Mattes mutual information over one transform family."""
    import SimpleITK as sitk

    fixed = _sitk_image(fixed_od)
    moving = _sitk_image(moving_od)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    # Sample rather than use every pixel: the metric is evaluated hundreds of times and
    # a random 20% is enough for a stable gradient on images this size.
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.2, seed=1)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=iterations,
        gradientMagnitudeTolerance=1e-6,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    # Coarse to fine. A multimodal metric has local minima, and starting at one eighth
    # resolution is what keeps the optimiser out of them.
    method.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    method.SetSmoothingSigmasPerLevel([4, 2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    if initial is not None:
        start = initial
    elif transform_kind == "affine":
        start = sitk.AffineTransform(2)
    else:
        start = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Similarity2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
    method.SetInitialTransform(start, inPlace=False)
    return method.Execute(fixed, moving)


def method_mattes(fixed_od, moving_od, fixed_mask, moving_mask):
    """Mutual information over a similarity transform: the feature-free workhorse."""
    result = _mattes(fixed_od, moving_od, fixed_mask, "similarity")
    return _sitk_matrix(result, fixed_od.shape), {"sitk": result.GetName()}


def method_affine(fixed_od, moving_od, fixed_mask, moving_mask):
    """The same, allowing shear and anisotropic scale, started from the similarity fit."""
    import SimpleITK as sitk

    similarity = _mattes(fixed_od, moving_od, fixed_mask, "similarity")
    affine = sitk.AffineTransform(2)
    affine.SetCenter(similarity.GetCenter() if hasattr(similarity, "GetCenter") else (0.0, 0.0))
    composite = sitk.CompositeTransform([similarity])
    result = _mattes(fixed_od, moving_od, fixed_mask, "affine", initial=sitk.AffineTransform(2))
    # Keep whichever of the two the scoreboard prefers; affine can overfit a weak signal.
    return _sitk_matrix(result, fixed_od.shape), {"sitk": result.GetName()}


def method_bspline(fixed_od, moving_od, fixed_mask, moving_mask):
    """Free-form non-rigid refinement on top of the similarity fit.

    **The only method here that can follow local deformation.** A section is floated onto
    glass and picks up stretch and folding that no rigid or affine transform can express,
    which is exactly what the outline method cannot do however well it places the section.

    Returned as a dense displacement field sampled onto the same 2x3 convention as the
    others cannot express - so this one reports its score and its field separately, and
    the caller keeps the field rather than a matrix.
    """
    import numpy as np
    import SimpleITK as sitk

    fixed = _sitk_image(fixed_od)
    moving = _sitk_image(moving_od)
    start = _mattes(fixed_od, moving_od, fixed_mask, "similarity")

    # A coarse control grid. Fine grids overfit a weak multimodal signal and start
    # inventing deformation to explain noise; eight control points across the section is
    # about one per centimetre of tissue here.
    mesh = [8, 8]
    bspline = sitk.BSplineTransformInitializer(fixed, mesh, order=3)

    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.2, seed=1)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=120)
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    method.SetMovingInitialTransform(start)
    method.SetInitialTransform(bspline, inPlace=False)
    deformable = method.Execute(fixed, moving)

    composite = sitk.CompositeTransform([start])
    composite.AddTransform(deformable)
    return composite, {"mesh": mesh, "nonrigid": True}


METHODS = {
    "outline": method_outline,
    "mask": method_mask,
    "mattes": method_mattes,
    "affine": method_affine,
    "bspline": method_bspline,
}

#: Methods that return a SimpleITK transform rather than a 2x3 matrix, because a non-rigid
#: warp cannot be written as one.
NON_LINEAR = {"bspline"}



# --- persisting a fitted transform ------------------------------------------


def _as_sitk(fitted, name):
    """Whatever a method returned, as a SimpleITK transform that can be written to disk.

    The linear methods hand back a 2x3 matrix and the non-rigid one hands back a
    SimpleITK transform already. Storing both the same way means the warp path has one
    code path instead of two, and a matrix that has been round-tripped through
    `AffineTransform` is still exactly the same map.
    """
    import numpy as np
    import SimpleITK as sitk

    if name in NON_LINEAR:
        return fitted
    m = np.asarray(fitted, dtype=float)
    affine = sitk.AffineTransform(2)
    # SimpleITK wants the matrix row-major and the translation separately.
    affine.SetMatrix([m[0, 0], m[0, 1], m[1, 0], m[1, 1]])
    affine.SetTranslation([m[0, 2], m[1, 2]])
    return affine


def fit_best(request: dict) -> dict:
    """Refit one named method per pair and write the transform to disk.

    Separate from the sweep because the sweep's job is to *compare* and this one's job is
    to *keep*. Refitting is cheap next to re-running every method, and it avoids carrying
    five transforms per pair around when four of them will never be used.

    The transform is written in the **working** resolution's pixel space, and
    `work_scale` is written beside it, because the caller's coordinates are in the
    aligned render's pixels and the conversion has to be explicit rather than inferred.
    """
    import SimpleITK as sitk

    work_dim = int(request.get("work_dim", WORK_DIM))
    out_dir = Path(request["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_od, fixed_mask, scale = load(request["reference"], work_dim)
    saved = {}

    for code, spec in request["targets"].items():
        path, name = spec["path"], spec["method"]
        entry = {"method": name}
        try:
            moving_od, moving_mask, _ = load(path, work_dim)
            fitted, extra = METHODS[name](fixed_od, moving_od, fixed_mask, moving_mask)
            transform = _as_sitk(fitted, name)
            # **HDF5, not the legacy .tfm text format.** A B-spline fit here is a
            # composite (similarity, then the deformable part), and SimpleITK's own
            # optimiser returns a composite too, so the result is a *nested* composite -
            # which .tfm refuses with "Composite Transform can only be 1st transform in a
            # file". HDF5 represents it without flattening, and flattening would risk
            # changing the map rather than merely its storage.
            target = out_dir / f"{code}.h5"

            # **Flatten before writing.** `Execute()` with a moving initial transform
            # hands back a CompositeTransform, and adding that to another composite makes
            # a *nested* one - which neither .tfm nor HDF5 will store: both throw
            # "Composite Transform can only be 1st transform in a file". Worse, the write
            # throws *after* creating the file, leaving a valid-looking HDF5 holding an
            # empty composite - an identity transform that would read back cleanly and
            # silently carry every mask nowhere. Flattening preserves the map exactly
            # (verified below) and removes the nesting the writer objects to.
            if isinstance(transform, sitk.CompositeTransform):
                transform = sitk.CompositeTransform(transform)
                transform.FlattenTransform()

            try:
                sitk.WriteTransform(transform, str(target))
            except Exception:
                # Never leave a half-written transform behind for something else to read.
                target.unlink(missing_ok=True)
                raise
            # Read it straight back and check it still maps a point the same way. A
            # transform that wrote without error but reloads differently is the kind of
            # failure that would only show up as a mask on the wrong tissue.
            reread = sitk.ReadTransform(str(target))
            probe = (float(fixed_od.shape[1]) / 3, float(fixed_od.shape[0]) / 3)
            before, after = transform.TransformPoint(probe), reread.TransformPoint(probe)
            drift = max(abs(before[0] - after[0]), abs(before[1] - after[1]))
            if drift > 1e-6:
                raise RuntimeError(f"transform changed on reload: {drift:.2e} px drift")

            if name in NON_LINEAR:
                score, on_slide = score_sitk(fixed_od, moving_od, fixed_mask, fitted)
            else:
                score, on_slide = score_transform(fixed_od, moving_od, fixed_mask, fitted)
            entry.update(ok=True, file=target.name, nmi=round(score, 4),
                         on_slide=round(on_slide, 4))
        except Exception as failure:  # noqa: BLE001 - one pair failing is data
            entry.update(ok=False, error=f"{type(failure).__name__}: {failure}")
        saved[code] = entry

    return {"ok": True, "work_scale": scale, "work_dim": work_dim, "saved": saved}


def load(path, work_dim):
    """An aligned render as absorbance plus tissue mask, downscaled to the working size."""
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    scale = min(1.0, work_dim / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale))),
            Image.BILINEAR,
        )
    rgb = np.asarray(image)
    optical = absorbance(rgb)
    # The renders already have everything outside the tissue whitened, so anything with
    # real absorbance is tissue. No threshold search needed here.
    return optical, optical > 0.05, scale


def main(request: dict) -> dict:
    started = time.time()
    import numpy as np

    work_dim = int(request.get("work_dim", WORK_DIM))
    fixed_od, fixed_mask, scale = load(request["reference"], work_dim)

    wanted = request.get("methods") or list(METHODS)
    results = {}

    for code, path in request["targets"].items():
        moving_od, moving_mask, _ = load(path, work_dim)
        per_method = {}
        for name in wanted:
            if name not in METHODS:
                continue
            attempt = {"method": name}
            try:
                began = time.time()
                fitted, extra = METHODS[name](fixed_od, moving_od, fixed_mask, moving_mask)
                if name in NON_LINEAR:
                    score, on_slide = score_sitk(fixed_od, moving_od, fixed_mask, fitted)
                    shape = None
                else:
                    score, on_slide = score_transform(fixed_od, moving_od, fixed_mask, fitted)
                    shape = np.asarray(fitted).tolist()
                attempt.update(
                    ok=True,
                    nmi=round(score, 4),
                    on_slide=round(on_slide, 4),
                    seconds=round(time.time() - began, 1),
                    # Reported at the working resolution. The caller rescales to the
                    # render's own pixels using `work_scale`, which is recorded rather
                    # than assumed. None for a non-rigid fit, which has no matrix.
                    matrix=shape,
                    **extra,
                )
            except Exception as failure:  # noqa: BLE001 - one method failing is data
                attempt.update(ok=False, error=f"{type(failure).__name__}: {failure}")
            per_method[name] = attempt

        usable = [a for a in per_method.values() if a.get("ok") and a.get("nmi")]
        best = max(usable, key=lambda a: a["nmi"]) if usable else None
        results[code] = {
            "methods": per_method,
            "best": best["method"] if best else None,
            "best_nmi": best["nmi"] if best else None,
            # What the outline alone achieves, so every other method is read as a delta
            # against doing nothing rather than as an absolute nobody can place.
            "outline_nmi": per_method.get("outline", {}).get("nmi"),
        }

    return {
        "ok": True,
        "work_scale": scale,
        "work_dim": work_dim,
        "slides": results,
        "seconds": round(time.time() - started, 1),
    }


if __name__ == "__main__":
    if REQUEST is None or RESPONSE is None:
        print("usage: register_methods.py <request.json> <response.json>", file=sys.stderr)
        sys.exit(2)
    try:
        req = json.loads(REQUEST.read_text(encoding="utf-8"))
        payload = fit_best(req) if req.get("mode") == "fit_best" else main(req)
        code = 0
    except Exception as exc:  # noqa: BLE001 - every failure is reported as data
        import traceback

        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-4000:]}
        code = 1
    RESPONSE.write_text(json.dumps(payload), encoding="utf-8")
    sys.exit(code)
