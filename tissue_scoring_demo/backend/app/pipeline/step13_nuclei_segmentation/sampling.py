"""Which fields inside a carried region actually get segmented.

**This step samples, and the sample is the headline.** The three regions step 10
carries across total about 26 mm2; at the measured 60 s/mm2 that is over half an
hour of CPU to segment whole, for a screen whose question is "what do the nuclei
in this region look like, and how many are there per mm2". Twelve 256 um fields
per region answer that in about two minutes. So the count this step publishes is
**an estimate from a stated sample**, never a census, and every number it emits
carries the sampled area beside it so nobody can read it as one.

**The sample is spread, not ranked.** The obvious implementation - score every
candidate field by how much haematoxylin it holds and take the best - is wrong
here, and wrong in the specific direction that matters: it would pick the densest
fields in the region and report their density as the region's. Since nuclei per
mm2 is exactly the QC metric the guide asks for (compare it across the case's six
serial sections, and a 30% shortfall is a segmentation failure rather than
biology), sampling on the thing being measured would destroy the only check the
step has. So candidates are filtered for *being tissue at all*, and then taken at
a uniform stride across the region in raster order.

**And it is deterministic.** No RNG: the same region yields the same fields on
every run, so two runs that disagree disagree about segmentation rather than
about which pixels they looked at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from app.core.config import settings

#: Resolution, in microns per pixel, of the scratch mask a region's rings are
#: rasterised onto to decide which fields are inside it. Coarse on purpose - it
#: answers "is this field's centre in the region", which is a question about
#: hundreds of microns, and a full-resolution raster of a 22 mm2 region would be
#: half a gigapixel.
FOOTPRINT_MPP = 8.0


@dataclass(frozen=True)
class Field:
    """One square of slide about to be segmented, in the IHC slide's own pixels."""

    index: int
    #: Top-left corner, level-0 pixels.
    x: int
    y: int
    #: Side of the square in level-0 pixels - what is read.
    span: int
    #: Side in model pixels - what is segmented, after resampling to its mpp.
    size: int
    #: Share of the field that is tissue rather than glass, from the footprint mask.
    tissue: float


@dataclass(frozen=True)
class Footprint:
    """A region's rings, rasterised coarsely, plus the frame they were drawn in."""

    mask: np.ndarray
    scale: float
    x0: int
    y0: int

    def covered(self, x: int, y: int, span: int) -> float:
        """Share of the level-0 square at `(x, y, span)` that is inside the region."""
        col0 = int((x - self.x0) * self.scale)
        row0 = int((y - self.y0) * self.scale)
        col1 = max(col0 + 1, int((x + span - self.x0) * self.scale))
        row1 = max(row0 + 1, int((y + span - self.y0) * self.scale))

        height, width = self.mask.shape
        col0, col1 = max(0, col0), min(width, col1)
        row0, row1 = max(0, row0), min(height, row1)
        if col1 <= col0 or row1 <= row0:
            return 0.0

        window = self.mask[row0:row1, col0:col1]
        return float(window.mean())


def bounds(rings: list[list[list[float]]]) -> tuple[int, int, int, int]:
    """The level-0 bounding box of a region's rings, as `(x0, y0, x1, y1)`."""
    xs = [x for ring in rings for x, _ in ring]
    ys = [y for ring in rings for _, y in ring]
    if not xs or not ys:
        raise ValueError("a region with no vertices has no bounding box")
    return int(min(xs)), int(min(ys)), int(np.ceil(max(xs))), int(np.ceil(max(ys)))


def rasterise(
    rings: list[list[list[float]]], *, mpp: float, target_mpp: float = FOOTPRINT_MPP
) -> Footprint:
    """Fill a region's rings into a coarse boolean mask.

    Outer ring filled, every later ring cut back out - the convention step 9 and
    step 10 both use, where a hole is a different class fully enclosed by this
    one. Drawn with PIL rather than a polygon library because Pillow is already a
    core dependency and this is a fill, not a geometry engine: the rings may be
    self-intersecting (step 9 traces a tile grid, and diagonally adjacent cells
    pinch), which an even-odd scanline fill handles without complaint and a
    topology-checking library refuses outright.
    """
    x0, y0, x1, y1 = bounds(rings)
    scale = mpp / target_mpp if mpp and mpp > 0 else 1.0
    width = max(1, int(np.ceil((x1 - x0) * scale)))
    height = max(1, int(np.ceil((y1 - y0) * scale)))

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for index, ring in enumerate(rings):
        if len(ring) < 3:
            continue
        points = [((x - x0) * scale, (y - y0) * scale) for x, y in ring]
        draw.polygon(points, fill=0 if index else 1)

    return Footprint(mask=np.asarray(canvas, dtype=bool), scale=scale, x0=x0, y0=y0)


def _spread(candidates: list[Field], count: int) -> list[Field]:
    """`count` fields taken at a uniform stride through `candidates`.

    A stride rather than the first N, so the sample covers the region instead of
    its top-left corner, and rather than a random draw so the answer is the same
    every run. Candidates arrive in raster order, so a stride walks down the
    region.
    """
    if count <= 0 or not candidates:
        return []
    if len(candidates) <= count:
        return list(candidates)

    step = len(candidates) / count
    picked = [candidates[min(len(candidates) - 1, int(index * step))] for index in range(count)]

    # A stride can land twice on the same field when the list is barely longer
    # than the sample; fill from what is left rather than returning fewer.
    seen = {id(field) for field in picked}
    if len(seen) < len(picked):
        picked = list({id(f): f for f in picked}.values())
        for field in candidates:
            if len(picked) >= count:
                break
            if id(field) not in {id(f) for f in picked}:
                picked.append(field)
    return picked[:count]


def allocate(
    areas: list[float], *, budget: int | None = None, minimum: int | None = None
) -> list[int]:
    """Split a field budget between regions in proportion to their area.

    **The allocation is the sample, so getting it wrong is not a loss of
    precision but a change of subject.** Giving every region the same number of
    fields makes the measured cells a sample of the *regions* rather than of the
    tumour: on CAN_00270 the largest region is 84% of the invasive area and was
    contributing 42% of the cells. Nothing downstream can fully undo that -
    weighting the regions afterwards corrects their influence on the mean but
    cannot recover the resolution never spent on the large one.

    Proportional allocation also collapses open question Q3 for the common case.
    When each region's share of the fields matches its share of the area, the
    plain mean over all sampled fields and the area-weighted mean of the regions
    are the same number, so "which averaging did you use" stops having a
    material answer.

    `minimum` is a floor per region, and it is why this returns an allocation
    rather than a formula: a region whose proportional share rounds to one field
    has a percentage of 0 or 100 and no way to tell which. The floor is taken
    out of the budget before the remainder is shared, so the proportionality
    holds over what is left rather than being quietly overspent.

    Largest-remainder rather than rounding each share independently: rounding
    gives away or overspends fields depending on where the fractions fall, and
    the caller has a fixed budget to respect.
    """
    wanted = settings.nuclei_field_budget if budget is None else budget
    floor = settings.nuclei_min_tiles_per_region if minimum is None else minimum

    if not areas:
        return []
    if wanted <= 0:
        return [0] * len(areas)

    # The floor can exceed the budget on a slide with many carried regions. Honour
    # the floor and let the budget grow rather than sampling a region too thinly to
    # read: the budget is a cost control, and the floor is a validity condition.
    base = [floor] * len(areas)
    remaining = wanted - sum(base)
    total = sum(areas)
    if remaining <= 0 or total <= 0:
        return base

    exact = [remaining * area / total for area in areas]
    shares = [int(value) for value in exact]
    for index in sorted(
        range(len(areas)), key=lambda i: exact[i] - shares[i], reverse=True
    )[: remaining - sum(shares)]:
        shares[index] += 1

    return [floor + share for floor, share in zip(base, shares, strict=True)]


def fields_for_region(
    rings: list[list[list[float]]],
    *,
    mpp: float,
    slide_width: int,
    slide_height: int,
    count: int | None = None,
    tile_px: int | None = None,
    model_mpp: float = 0.5,
    min_tissue: float | None = None,
) -> tuple[list[Field], int]:
    """The fields to segment inside one region, and how many were available.

    Returns `(chosen, candidates)`. The second number is what makes the first
    honest: "12 of 340 fields" says what fraction of the region was looked at,
    and the report prints both.
    """
    wanted = settings.nuclei_tiles_per_region if count is None else count
    size = settings.nuclei_tile_px if tile_px is None else tile_px
    floor = settings.nuclei_min_tile_tissue if min_tissue is None else min_tissue

    # How much slide one model pixel covers. The model is scale-specific - 0.5
    # um/px is in its own metadata - so the field is sized in microns and the
    # read is resampled to it, never the other way round.
    span = max(1, int(round(size * model_mpp / mpp))) if mpp and mpp > 0 else size

    footprint = rasterise(rings, mpp=mpp)
    x0, y0, x1, y1 = bounds(rings)

    candidates: list[Field] = []
    index = 0
    for y in range(y0, y1, span):
        for x in range(x0, x1, span):
            if x + span > slide_width or y + span > slide_height:
                continue
            covered = footprint.covered(x, y, span)
            if covered < floor:
                continue
            candidates.append(
                Field(index=index, x=x, y=y, span=span, size=size, tissue=covered)
            )
            index += 1

    chosen = _spread(candidates, wanted)
    # Renumber so a field's index is its rank in the sample, which is what the
    # UI labels and the crop endpoints key on.
    return (
        [
            Field(index=rank, x=f.x, y=f.y, span=f.span, size=f.size, tissue=f.tissue)
            for rank, f in enumerate(chosen)
        ],
        len(candidates),
    )


__all__ = [
    "FOOTPRINT_MPP",
    "Field",
    "Footprint",
    "allocate",
    "bounds",
    "fields_for_region",
    "rasterise",
]
