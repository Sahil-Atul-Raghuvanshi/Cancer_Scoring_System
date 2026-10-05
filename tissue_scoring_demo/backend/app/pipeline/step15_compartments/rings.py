"""Turning step 13's label maps into outlines the browser can draw on the slide.

Step 13 has always stored its compartments as label maps, which is right for
step 14 - it measures pixels - and useless for a screen, which needs outlines in
the slide's own coordinates. The old screen worked around that with one rendered
PNG per region: a 512 px picture of one sampled square, at a fixed zoom, with no
way to tell which cell was which. It was the least legible screen in the
walkthrough, and the reason was that the geometry never left the server.

So this traces the bands into rings, once, when the compartments are built, and
stores them beside the arrays. The tracer is step 9's, reused rather than
rewritten: it walks cell edges and cancels shared ones, so holes come out for
free and there is no tolerance to tune.

**Collinear vertices are dropped, and nothing else is.** The tracer is exact and
rectilinear, so a 30-pixel-wide cell arrives as a staircase carrying a vertex per
pixel edge. Collapsing runs that continue in the same direction is lossless - the
polygon is identical - and it takes a region's geometry from tens of megabytes to
a few. No Douglas-Peucker, no tolerance: a simplification that moved a boundary
would be the screen disagreeing with the measurement about where a cell ends.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from app.pipeline.step09_roi_mask.mask import trace_rings
from app.pipeline.step13_nuclei_segmentation.instances import pixel_mapper

#: Below this many pixels a traced band is a handful of stray cells' worth of
#: noise rather than a compartment, and drawing it adds vertices and no meaning.
MIN_BAND_PX = 4


def drop_collinear(ring: tuple[tuple[int, int], ...]) -> list[list[float]]:
    """Keep only the vertices where the boundary actually turns.

    Lossless: three consecutive points on one axis describe the same edge as the
    two ends of it. On this project's fields it removes about nine vertices in
    ten.
    """
    if len(ring) < 3:
        return [[round(float(x), 1), round(float(y), 1)] for x, y in ring]

    out: list[list[float]] = []
    count = len(ring)
    for index in range(count):
        previous = ring[index - 1]
        current = ring[index]
        following = ring[(index + 1) % count]
        # Cross product of the two edge vectors. Zero means "straight on".
        turn = (current[0] - previous[0]) * (following[1] - current[1]) - (
            current[1] - previous[1]
        ) * (following[0] - current[0])
        # A tolerance, not an equality, because the tracer has already mapped
        # these into slide coordinates: the multiply is exact for the grid but
        # the comparison should not depend on that.
        if abs(turn) > 1e-9:
            # One decimal, because these are grid corners scaled into level-0
            # pixels: a tenth of a pixel is already finer than the geometry is,
            # and the full double costs eight characters a vertex in a file that
            # runs to tens of thousands of them.
            out.append([round(float(current[0]), 1), round(float(current[1]), 1)])

    return (
        out
        if len(out) >= 3
        else [[round(float(x), 1), round(float(y), 1)] for x, y in ring]
    )


def _trace(
    band: np.ndarray, identifier: int, window, *, x0: float, y0: float, scale: float
) -> list[list[list[float]]]:
    """One cell's share of one band, as rings in slide coordinates."""
    rows, cols = window
    patch = band[rows, cols] == identifier
    if int(patch.sum()) < MIN_BAND_PX:
        return []

    mapper = pixel_mapper(x0 + cols.start * scale, y0 + rows.start * scale, scale)
    return [drop_collinear(ring) for ring in trace_rings(patch, mapper)]


def trace_field(
    built,
    *,
    x0: float,
    y0: float,
    level0_scale: float,
) -> list[dict]:
    """Every cell, as `{id, nucleus, cytoplasm, membrane, alternate}`.

    `built` is a `geometry.Anatomy`. `level0_scale` is how many level-0 slide
    pixels one of its array pixels covers - a different number whenever the field
    was resampled to the model's input scale, which is why it is passed in rather
    than assumed to be one.

    Cells are found from the widest band rather than from the nuclei, so a cell
    whose nucleus sits outside the crop of one band still gets traced in the
    other. The window comes from the same array it is used on, per band, because
    the bands have different extents.
    """
    out: list[dict] = []

    identifiers = np.unique(built.nucleus)
    nucleus_windows = ndimage.find_objects(built.nucleus)
    membrane_windows = ndimage.find_objects(built.membrane)
    cytoplasm_windows = ndimage.find_objects(built.cytoplasm)
    alternate_windows = ndimage.find_objects(built.alternate)

    for identifier in identifiers:
        if identifier == 0:
            continue
        index = int(identifier) - 1

        def band(array: np.ndarray, windows: list) -> list[list[list[float]]]:
            window = windows[index] if index < len(windows) else None
            if window is None:
                return []
            return _trace(
                array, int(identifier), window, x0=x0, y0=y0, scale=level0_scale
            )

        nucleus = band(built.nucleus, nucleus_windows)
        if not nucleus:
            continue

        out.append(
            {
                "id": int(identifier),
                "nucleus": nucleus,
                "cytoplasm": band(built.cytoplasm, cytoplasm_windows),
                "membrane": band(built.membrane, membrane_windows),
                "alternate": band(built.alternate, alternate_windows),
            }
        )

    return out


__all__ = ["MIN_BAND_PX", "drop_collinear", "trace_field"]
