"""BEETLE's pixels, turned back into slide coordinates as polygons.

This is the module that has to get coordinate integrity right, so it is worth naming the
chain it sits in the middle of:

    candidate box (H&E level-0)
        -> padded box, the canvas BEETLE paints into
        -> mask pixels at `mask_mpp`, origin at the padded box's corner
        -> rings in H&E level-0 pixels          <- this module
        -> VALIS warp                            <- step 12
        -> rings in IHC level-0 pixels

Every arrow is a multiplication and an addition, and the one this module owns is the
only one where a *picture* becomes *geometry*. Get it wrong and nothing fails: a mask
appears, in the right shape, on the wrong tissue.

--------------------------------------------------------------------------------
Why polygons rather than the raster mask
--------------------------------------------------------------------------------

Because that is what the registration takes. `transform_warp.warp_for_pair` moves vertices,
not images, and the whole pipeline downstream of step 12 - the field sampler, the crops,
the panels - already speaks rings. Handing step 12 a raster would mean either warping an
image (resampling a boundary twice, once into the warp and once out of it) or inventing
a second registration path beside the working one.

It also means the refined boundary and the coarse one are *the same kind of object*, so
step 12 needs no fork: it warps whatever rings it is given.

--------------------------------------------------------------------------------
Three clean-ups, and why each one is not optional
--------------------------------------------------------------------------------

**Specks are dropped.** A per-pixel network at 1 um/px produces isolated pixels a
per-window one cannot. Left in, one tumour boundary becomes four hundred rings, and
every one of them is then densified, warped by VALIS, simplified, stored, drawn and
counted as a region.

**Small holes are filled.** A lumen, a capillary and a patch the network was unsure
about all read as holes. A hole is meaningful here - it is how the boundary says "not
tumour, inside the tumour" - but a three-pixel one is not saying that.

**The staircase is simplified.** The trace is exact and rectilinear: it runs along mask
pixel edges, so a millimetre of boundary at 1 um/px is a thousand vertices. Douglas-
Peucker at twice the mask pixel removes the staircase and cannot remove anything
larger, which is the bound that makes this safe to do before a warp rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from app.pipeline.step09_roi_mask.mask import trace_rings
from app.registration.transform import simplify_ring


@dataclass(frozen=True)
class RefinedRegion:
    """One connected piece of refined tumour, in H&E level-0 pixels.

    `area_mm2` is counted **from the mask pixels, not from the polygon**. The polygon has
    been simplified and would give a slightly different number, and this step's whole
    claim is that the area it reports is the area the network drew - so the number comes
    from the thing the network produced and the rings are only how it is shown and
    warped.
    """

    #: Outer ring first, then holes, `[x, y]` in H&E level-0 pixels.
    rings: list[list[list[float]]]
    pixels: int
    area_mm2: float

    @property
    def holes(self) -> int:
        return max(0, len(self.rings) - 1)


def clean(
    binary: np.ndarray, *, min_component_px: int, min_hole_px: int
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Split `binary` into components, dropping specks and filling small holes.

    Returns `(component, (row0, col0))` per surviving component, largest first, **each
    cropped to its own bounding box** with the offset needed to put it back.

    **It used to return full-size arrays, and that did not terminate.** One boolean array
    the size of the whole mask per component reads well - the caller needs no offsets -
    but the cost is `components x mask`, in both memory and time, and neither is bounded
    by how big the components actually are.

    CAN_00259's ROI-002 measured it: a 12,507 x 12,507 mask is 156 MB per component, and
    at `min_component_mm2` of 0.005 (5,000 px at 1 um/px) roughly 110 fragments survive.
    The run climbed 6.9 -> 9.9 -> 12.8 -> 15.1 -> 17.0 -> 17.3 GB - a straight line at
    156 MB a component - while `_fill_small_holes` made four more full-mask passes for
    each one, single-threaded, and never came back. It was killed twice at 96 and 114
    minutes having written nothing.

    The sibling region ROI-002 was compared against, ROI-001, is **larger** at 199 mm2 and
    finished in 55 minutes - because its refined mask is empty, so `count == 0` returns
    here immediately and the loop never runs. The bigger region worked; the smaller one
    hung. That is the shape of an O(components x mask) cost, not a big-input cost.

    Cropping makes both bounded by the tumour rather than the canvas: a 0.005 mm2 speck
    now costs a bounding box of a few thousand pixels instead of 156 MB.
    """
    labelled, count = ndimage.label(binary, structure=np.ones((3, 3)))
    if count == 0:
        return []

    sizes = ndimage.sum_labels(binary, labelled, index=range(1, count + 1))
    order = np.argsort(sizes)[::-1]
    # One bounding box per label, indexed by label - 1. Computed once for all of them,
    # which is a single pass rather than one per component.
    boxes = ndimage.find_objects(labelled)

    kept: list[tuple[np.ndarray, tuple[int, int]]] = []
    for position in order:
        if sizes[position] < min_component_px:
            # Sorted largest first, so the first speck means every remaining one is a
            # speck too.
            break
        box = boxes[position]
        if box is None:  # pragma: no cover - a label with no pixels cannot occur here
            continue
        rows, cols = box
        # `== position + 1` inside the box, so a speck costs its own area and not the
        # canvas. Holes are enclosed by their component by definition, so they are
        # inside the box too and filling within the crop is exact.
        window = labelled[rows, cols] == position + 1
        kept.append(
            (_fill_small_holes(window, min_hole_px), (int(rows.start), int(cols.start)))
        )
    return kept


def _fill_small_holes(component: np.ndarray, min_hole_px: int) -> np.ndarray:
    """`component` with every enclosed gap below `min_hole_px` filled in.

    `binary_fill_holes` alone would fill *every* hole, which would throw away the real
    ones - a duct lumen inside a sheet of tumour, or a vessel - and those are part of
    what a per-pixel boundary is for. So the holes are found first, measured, and only
    the ones too small to be anatomy are given back to the component.
    """
    if min_hole_px <= 1:
        return component

    filled = ndimage.binary_fill_holes(component)
    holes = filled & ~component
    if not holes.any():
        return component

    labelled, count = ndimage.label(holes)
    if count == 0:
        return component

    sizes = ndimage.sum_labels(holes, labelled, index=range(1, count + 1))
    small = np.isin(labelled, [index + 1 for index, size in enumerate(sizes) if size < min_hole_px])
    return component | small


def to_level0(
    *,
    origin: tuple[int, int],
    mask_mpp: float,
    base_mpp: float,
    slide_width: int,
    slide_height: int,
):
    """A `(row, col)` in the mask to an `(x, y)` in the slide, clamped to the canvas.

    The same shape of function `mask.level0_mapper` returns for the tile grid, and it is
    passed to the same `trace_rings` - which is what lets one tracer serve a boundary
    made of 224 um windows and one made of 1 um pixels without knowing which it has.

    The clamp is the same defence step 9's mapper makes: the padded box was clamped to
    the slide, but rounding at the far edge can still put a vertex one pixel past it,
    and a vertex outside the canvas is one VALIS would have to invent a warp for.
    """
    origin_x, origin_y = origin
    scale = float(mask_mpp) / float(base_mpp)

    def convert(row: int, col: int) -> tuple[int, int]:
        x = origin_x + col * scale
        y = origin_y + row * scale
        return (
            int(np.clip(round(x), 0, slide_width)),
            int(np.clip(round(y), 0, slide_height)),
        )

    return convert


def _shifted(convert, origin_rc: tuple[int, int]):
    """`convert`, with a cropped component's bounding-box offset added back on."""
    row0, col0 = origin_rc

    def shifted(row: int, col: int, _r0: int = row0, _c0: int = col0) -> tuple[int, int]:
        return convert(row + _r0, col + _c0)

    return shifted


def regions(
    mask: np.ndarray,
    *,
    code: int,
    origin: tuple[int, int],
    mask_mpp: float,
    base_mpp: float,
    slide_width: int,
    slide_height: int,
    min_component_mm2: float,
    min_hole_mm2: float,
    simplify_um: float,
) -> list[RefinedRegion]:
    """Every piece of `code` in `mask`, as simplified polygons in H&E level-0 pixels.

    `mask` is a `PixelMap.mask` - BEETLE's five codes at `mask_mpp`, with `OUTSIDE`
    where no window ran - and `origin` is its `mask_origin`. `code` is which class to
    trace, which is always `beetle.SCORED_CODE` on the pipeline's own path: Rule 5 scores
    invasive carcinoma and nothing else, and a refined mask that included in-situ disease
    would put back exactly what step 9 spent its `protect_in_situ` parameter removing.
    """
    pixel_mm2 = (float(mask_mpp) / 1000.0) ** 2
    min_component_px = max(1, int(round(min_component_mm2 / pixel_mm2)))
    min_hole_px = max(1, int(round(min_hole_mm2 / pixel_mm2)))
    # Microns to level-0 pixels: the rings are in level-0 by the time this is applied.
    tolerance_px = float(simplify_um) / max(float(base_mpp), 1e-9)

    convert = to_level0(
        origin=origin,
        mask_mpp=mask_mpp,
        base_mpp=base_mpp,
        slide_width=slide_width,
        slide_height=slide_height,
    )

    out: list[RefinedRegion] = []
    for component, origin_rc in clean(
        np.asarray(mask) == code,
        min_component_px=min_component_px,
        min_hole_px=min_hole_px,
    ):
        pixels = int(component.sum())
        # `clean` hands back each component cropped to its bounding box, so the tracer's
        # (row, col) are box-relative and the box's own offset has to go back on before
        # the mask-to-slide conversion. Bound as default arguments rather than captured,
        # because a closure over the loop variable would give every component the last
        # box's offset - and the failure would be silent: rings of the right shape on the
        # wrong tissue, which is exactly what this module's header warns about.
        traced = trace_rings(component, _shifted(convert, origin_rc))
        simplified = [
            simplify_ring([[float(x), float(y)] for x, y in ring], tolerance_px)
            for ring in traced
        ]
        # A ring that simplification took below a triangle describes no area and cannot
        # be warped into one, so it is dropped rather than carried as a degenerate
        # polygon that every later stage has to special-case.
        simplified = [ring for ring in simplified if len(ring) >= 3]
        if not simplified:
            continue

        out.append(
            RefinedRegion(
                rings=simplified,
                pixels=pixels,
                area_mm2=round(pixels * pixel_mm2, 5),
            )
        )
    return out


__all__ = ["RefinedRegion", "clean", "regions", "to_level0"]
