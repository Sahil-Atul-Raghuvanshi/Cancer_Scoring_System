"""The classical baseline, kept because the comparison is the argument.

Step 11's whole claim is that *instance* segmentation is a different problem from
*semantic* segmentation - that "which pixels are nuclei" is easy and "which
pixels are nucleus #4,182" is not, because nuclei touch. A screen that only shows
InstanSeg's outlines does not make that case; it just shows some outlines. So the
textbook method runs beside it on the same field: threshold the counterstain,
distance-transform, seed at the local maxima, flood.

This is a fair implementation of that method, not a strawman. It is also the
method's honest limit: a distance transform splits two touching nuclei only when
the neck between them is narrower than either, so it merges the pairs that sit
shoulder to shoulder and shatters the one large irregular nucleus whose distance
map has two peaks. Both failures are visible on a 40x field, both are the reason
the pre-trained model is worth its 15 MB, and both move the count in the
direction that matters - a merge removes a cell from the denominator.

Written on scipy alone. `scipy.ndimage.watershed_ift` is the image-foresting
transform, which is a watershed; scikit-image is not a dependency of this backend
and this step is not a good enough reason to make it one.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from app.common.imaging import otsu_threshold

#: How far apart two seeds must be before both are kept, in microns. Below a
#: nucleus radius, every lumpy nucleus becomes two objects; the shipped value is
#: about one small-nucleus radius.
SEED_SEPARATION_UM = 3.0


def _seeds(distance: np.ndarray, separation_px: int) -> np.ndarray:
    """Local maxima of the distance transform, one marker per plateau."""
    span = max(1, int(round(separation_px)) * 2 + 1)
    peaks = distance >= ndimage.maximum_filter(distance, size=span)
    peaks &= distance > 0
    markers, _ = ndimage.label(peaks, structure=np.ones((3, 3), dtype=bool))
    return markers.astype(np.int32)


def segment(haematoxylin: np.ndarray, *, mpp: float) -> np.ndarray:
    """Instance labels for one field, by threshold-distance-watershed.

    Takes the haematoxylin coefficient rather than RGB, so the comparison with
    the model is like for like: both are shown the counterstain with the DAB
    already removed, and any difference between them is the segmentation method
    rather than the input.
    """
    if haematoxylin.size == 0:
        return np.zeros_like(haematoxylin, dtype=np.int32)

    # Otsu on the density, quantised to 8 bits because that is what the shared
    # histogram helpers take. Density is unbounded above, so it is scaled by its
    # own 99.5th percentile first - a fixed ceiling would clip a strongly
    # counterstained field into a single level.
    ceiling = float(np.percentile(haematoxylin, 99.5))
    if ceiling <= 0:
        return np.zeros(haematoxylin.shape, dtype=np.int32)

    scaled = np.clip(haematoxylin / ceiling, 0.0, 1.0)
    quantised = (scaled * 255).astype(np.uint8)
    foreground = quantised > otsu_threshold(quantised)

    # Fill the small holes a threshold leaves in a nucleus' pale centre, or the
    # distance transform seeds a ring of markers around it.
    foreground = ndimage.binary_fill_holes(foreground)
    if not foreground.any():
        return np.zeros(haematoxylin.shape, dtype=np.int32)

    distance = ndimage.distance_transform_edt(foreground)
    separation_px = max(1, int(round(SEED_SEPARATION_UM / mpp))) if mpp > 0 else 3
    markers = _seeds(distance, separation_px)
    if markers.max() == 0:
        labelled, _ = ndimage.label(foreground, structure=np.ones((3, 3), dtype=bool))
        return labelled.astype(np.int32)

    # watershed_ift floods uphill from the markers over a uint8 surface, so the
    # distance map is inverted: the ridges between two nuclei become the basins'
    # watershed lines.
    surface = (255 - np.clip(distance / max(distance.max(), 1e-6) * 255, 0, 255)).astype(np.uint8)

    # Background is seeded as -1 so the flood stops at the tissue edge instead of
    # assigning every glass pixel to the nearest nucleus.
    seeded = markers.copy()
    seeded[~foreground] = -1

    filled = ndimage.watershed_ift(surface, seeded)
    filled[filled < 0] = 0
    filled[~foreground] = 0
    return filled.astype(np.int32)


__all__ = ["SEED_SEPARATION_UM", "segment"]
