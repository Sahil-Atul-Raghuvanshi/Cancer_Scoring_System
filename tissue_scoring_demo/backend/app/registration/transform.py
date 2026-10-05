"""Ring geometry either side of a warp: densify before, tidy after.

A ring from step 9 is traced on step 8's tile grid, so its vertices sit a whole
window apart - at the default field of view, 224 microns. That spacing is fine
for a shape made of straight tile edges, and wrong for one about to be pushed
through a non-rigid transform, because such a transform does not map a straight
line to a straight line. Warping only the corners keeps the corners honest and
throws away every deformation between them.

So: densify to a fine spacing, warp every vertex, then optionally drop the
vertices the warped shape does not need. The order matters and this module
exists to make it hard to get wrong.
"""

from __future__ import annotations

import math

Point = tuple[float, float]
Ring = tuple[Point, ...]


def densify_ring(ring: Ring, max_spacing_px: float) -> list[list[float]]:
    """`ring` with intermediate vertices so no edge is longer than `max_spacing_px`.

    The ring is treated as closed: the edge from the last vertex back to the
    first is subdivided too, because it is as much a part of the border as any
    other and a warp will bend it just the same.
    """
    if len(ring) < 2 or max_spacing_px <= 0:
        return [[float(x), float(y)] for x, y in ring]

    out: list[list[float]] = []
    count = len(ring)

    for index in range(count):
        x0, y0 = ring[index]
        x1, y1 = ring[(index + 1) % count]
        out.append([float(x0), float(y0)])

        length = math.hypot(x1 - x0, y1 - y0)
        if length <= max_spacing_px:
            continue

        steps = int(math.ceil(length / max_spacing_px))
        for step in range(1, steps):
            fraction = step / steps
            out.append([x0 + (x1 - x0) * fraction, y0 + (y1 - y0) * fraction])

    return out


def simplify_ring(points: list[list[float]], tolerance_px: float) -> list[list[float]]:
    """Douglas-Peucker, to undo densification the warped shape does not need.

    Purely a payload and draw-cost measure, applied *after* warping: the dense
    vertices did their job carrying the deformation, and a border drawn at
    screen resolution does not need one vertex per 40 microns. The tolerance is
    what bounds the error this introduces, so it is expressed in pixels of the
    slide the points now live on and kept well under one displayed pixel.
    """
    if len(points) < 3 or tolerance_px <= 0:
        return points

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue

        x0, y0 = points[start]
        x1, y1 = points[end]
        dx, dy = x1 - x0, y1 - y0
        span = math.hypot(dx, dy)

        worst, worst_index = -1.0, start
        for index in range(start + 1, end):
            px, py = points[index]
            if span == 0:
                distance = math.hypot(px - x0, py - y0)
            else:
                distance = abs(dy * px - dx * py + x1 * y0 - y1 * x0) / span
            if distance > worst:
                worst, worst_index = distance, index

        if worst > tolerance_px:
            keep[worst_index] = True
            stack.append((start, worst_index))
            stack.append((worst_index, end))

    return [point for point, kept in zip(points, keep, strict=True) if kept]


def um_to_px(microns: float, mpp: float | None) -> float:
    """Microns to pixels on a slide of scale `mpp`.

    Falls back to treating the number as pixels when a slide records no scale,
    which is honest: without a scale there is no conversion to do, and refusing
    to run because a file lacks one would fail a slide the rest of the pipeline
    handles.
    """
    if not mpp or mpp <= 0:
        return microns
    return microns / mpp


__all__ = ["Point", "Ring", "densify_ring", "simplify_ring", "um_to_px"]
