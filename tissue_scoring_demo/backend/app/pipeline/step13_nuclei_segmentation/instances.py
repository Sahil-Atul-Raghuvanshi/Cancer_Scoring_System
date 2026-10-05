"""A label map into nuclei: outlines, and the measurements step 12 classifies on.

Three jobs, in order.

**Trace each object.** `step09_roi_mask.mask.trace_rings` already does this,
exactly and with holes for free, and it is reused rather than reimplemented -
the only change being what a "cell" of its grid means. There it is a step 8
window of 224 microns; here it is one pixel at the model's 0.5 um/px. The
walk, the cancellation of shared edges and the leftmost-turn rule at a corner
touch are all the same arithmetic, and having one copy of it means an outline in
step 11 and a region border in step 9 cannot disagree about what a boundary is.

**Drop what cannot be measured.** A nucleus whose centroid falls in the border
band was seen truncated, so its shape is wrong and it may have been split in two.
It is kept as an object - it is really there, and hiding it would make the
overlay look wrong to anyone comparing it with the tile - but it is marked
`counted = False`, and the area it sits in is removed from the denominator with
it. That is what makes nuclei per mm2 an unbiased estimate rather than one that
drifts with tile size.

**Measure.** Area, perimeter, circularity, eccentricity, and mean haematoxylin.
These five are not a general-purpose feature set; they are precisely what the
guide names as separating lymphocytes (small, round, very dark) from tumour
nuclei (large, irregular), and step 12 uses them for nothing else. Every length
is in microns, computed from the model's own mpp - never in pixels, which would
mean something different on the next scanner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from app.pipeline.step09_roi_mask.mask import trace_rings


@dataclass(frozen=True)
class Nucleus:
    """One segmented nucleus, in the slide's own level-0 pixels."""

    label: int
    #: Centroid, level-0 pixels of the slide this was read from.
    x: float
    y: float
    area_um2: float
    perimeter_um: float
    #: `4*pi*area / perimeter^2`. 1.0 is a circle; a lymphocyte sits near it.
    circularity: float
    #: 0 is a disc, approaching 1 is a line. From the second central moments.
    eccentricity: float
    #: Mean haematoxylin coefficient inside the nucleus. Dark small nuclei are
    #: the lymphocyte signature the guide names.
    haematoxylin: float
    #: Outer ring first, then holes; level-0 `[x, y]` vertices.
    rings: list[list[list[float]]]
    #: False when the object touches the border band and must not enter a count.
    counted: bool
    #: Whether the outline touches the edge of the segmented field at all.
    on_edge: bool


def pixel_mapper(x0: float, y0: float, scale: float):
    """A mapper from a field-pixel vertex to a level-0 slide point.

    `trace_rings` hands back integer grid vertices - corners, so they run one
    past the last cell - and this turns them into slide coordinates. `scale` is
    level-0 pixels per field pixel, which is the resampling factor the field was
    read at rather than the slide's mpp: the field may have been read from a
    pyramid level and scaled, and it is the composition of the two that maps
    back.
    """

    def to_level0(row: int, col: int) -> tuple[float, float]:
        return (x0 + col * scale, y0 + row * scale)

    return to_level0


def _shape(mask: np.ndarray, mpp: float) -> tuple[float, float, float, float]:
    """Area, perimeter, circularity and eccentricity of one binary object.

    Perimeter is the *crack* boundary - the number of pixel edges between object
    and background - corrected by pi/4.

    The correction is not a fudge and its size is not arbitrary. A digitised disc
    of radius r has a crack boundary of 8r, because the staircase around it is
    the perimeter of its bounding square however fine the grid gets, while its
    true circumference is 2*pi*r. The ratio is exactly 4/pi, so multiplying by
    pi/4 makes a circle measure as a circle - `circularity` comes out at 1.0
    rather than 0.79.

    Calibrated on a circle deliberately, because circularity is a feature for
    telling round nuclei from irregular ones, and a lymphocyte is the round case.
    The cost is that it under-states a genuinely rectangular outline's perimeter
    by the same 21%; no nucleus is rectangular, and the factor is constant, so it
    cannot reorder two objects relative to each other.
    """
    area_px = int(mask.sum())
    if area_px == 0:
        return 0.0, 0.0, 0.0, 0.0

    padded = np.pad(mask, 1)
    horizontal = int(np.count_nonzero(padded[:, 1:] != padded[:, :-1]))
    vertical = int(np.count_nonzero(padded[1:, :] != padded[:-1, :]))
    perimeter_px = (horizontal + vertical) * (math.pi / 4.0)

    area = area_px * mpp * mpp
    perimeter = perimeter_px * mpp
    circularity = (
        float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0.0, 1.0))
        if perimeter > 0
        else 0.0
    )

    rows, cols = np.nonzero(mask)
    row_mean, col_mean = rows.mean(), cols.mean()
    dr, dc = rows - row_mean, cols - col_mean
    # Second central moments, with the 1/12 pixel-extent term that keeps a
    # single-pixel object from reporting a degenerate axis.
    mu_rr = float((dr * dr).mean()) + 1.0 / 12.0
    mu_cc = float((dc * dc).mean()) + 1.0 / 12.0
    mu_rc = float((dr * dc).mean())

    common = math.sqrt(max(0.0, (mu_rr - mu_cc) ** 2 + 4.0 * mu_rc * mu_rc))
    major = (mu_rr + mu_cc + common) / 2.0
    minor = (mu_rr + mu_cc - common) / 2.0
    eccentricity = math.sqrt(max(0.0, 1.0 - minor / major)) if major > 0 else 0.0

    return area, perimeter, circularity, float(eccentricity)


def extract(
    labels: np.ndarray,
    *,
    haematoxylin: np.ndarray | None,
    mpp: float,
    x0: float,
    y0: float,
    level0_scale: float,
    border_px: int,
    min_area_um2: float,
) -> list[Nucleus]:
    """Every object in `labels`, measured and traced.

    `labels` is the model's instance map for one field: 0 background, every other
    value one nucleus. `mpp` is the microns each of its pixels covers;
    `level0_scale` is how many level-0 slide pixels that is, which is a different
    number whenever the field was resampled.
    """
    if labels.size == 0:
        return []

    height, width = labels.shape
    objects = ndimage.find_objects(labels)

    out: list[Nucleus] = []
    for index, window in enumerate(objects, start=1):
        if window is None:
            continue

        rows, cols = window
        patch = labels[rows, cols] == index
        area, perimeter, circularity, eccentricity = _shape(patch, mpp)
        if area < min_area_um2:
            continue

        local_rows, local_cols = np.nonzero(patch)
        centre_row = float(local_rows.mean()) + rows.start
        centre_col = float(local_cols.mean()) + cols.start

        counted = (
            border_px <= centre_col <= width - border_px
            and border_px <= centre_row <= height - border_px
        )
        on_edge = (
            rows.start == 0 or cols.start == 0 or rows.stop >= height or cols.stop >= width
        )

        stain = 0.0
        if haematoxylin is not None:
            stain = float(haematoxylin[rows, cols][patch].mean())

        # Traced in the crop's own frame, then offset - the mapper carries both
        # the crop's origin and the field's, so the ring comes out in slide
        # coordinates in one step.
        crop_mapper = pixel_mapper(
            x0 + cols.start * level0_scale, y0 + rows.start * level0_scale, level0_scale
        )
        rings = trace_rings(patch, crop_mapper)

        out.append(
            Nucleus(
                label=index,
                x=x0 + centre_col * level0_scale,
                y=y0 + centre_row * level0_scale,
                area_um2=area,
                perimeter_um=perimeter,
                circularity=circularity,
                eccentricity=eccentricity,
                haematoxylin=stain,
                rings=[[[float(x), float(y)] for x, y in ring] for ring in rings],
                counted=counted,
                on_edge=on_edge,
            )
        )

    return out


def counted_area_mm2(*, size: int, border_px: int, mpp: float) -> float:
    """The denominator: the field's area minus the border band, in mm2.

    Shrinking the area to match the objects that were dropped is the whole point
    of the band. Counting over the full field while excluding its edge objects
    would undercount density by roughly the band's share of the area - about 18%
    at the shipped 512 px field and 24 px band, which is far larger than any
    difference this step is being asked to detect between two serial sections.
    """
    inner = max(0, size - 2 * border_px)
    return (inner * mpp / 1000.0) ** 2


__all__ = ["Nucleus", "counted_area_mm2", "extract", "pixel_mapper"]
