"""One field, end to end: read it, take the brown out, segment it, measure it.

The order here is the step's argument in code. Pixels are read at the *model's*
scale rather than the slide's, the DAB is removed before anything looks for a
nucleus, and only then does a segmenter see anything. Nothing downstream can put
those back in a different order.

**Read at 0.5 um/px, look at 40x.** The guide says to zoom to 40x, and the screen
does. The model does not: its own metadata declares an input scale of 0.5 um/px,
and 0.25 would be out of distribution. Those are different resolutions for
different purposes - one is what the network was trained on, the other is what a
person needs to judge an outline - and conflating them would quietly degrade the
segmentation to make a caption true.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.pipeline.step05_optical_density.tiles import read_tile

from .instances import Nucleus, counted_area_mm2, extract
from .sampling import Field
from .stain_input import StainBasis, haematoxylin_density, haematoxylin_only_rgb


@dataclass(frozen=True)
class SegmentedField:
    """One field's pixels, its nuclei, and the arithmetic behind its density."""

    field: Field
    #: What the segmenter was shown: DAB removed, unless the caller asked for raw.
    shown: np.ndarray
    #: The field as it really looks, for the overlay and the RGB comparison.
    rgb: np.ndarray
    haematoxylin: np.ndarray
    #: The segmenter's raw instance map, kept for the overlay: drawing seams
    #: between touching nuclei needs the labels, not the polygons.
    labels: np.ndarray
    nuclei: list[Nucleus]
    #: Microns per pixel of `shown` - the model's, not the slide's.
    mpp: float
    #: Level-0 pixels per pixel of `shown`.
    level0_scale: float
    counted_mm2: float

    @property
    def counted(self) -> int:
        return sum(1 for nucleus in self.nuclei if nucleus.counted)

    @property
    def density_per_mm2(self) -> float:
        return self.counted / self.counted_mm2 if self.counted_mm2 > 0 else 0.0


def _white_field(white, field: Field, mpp: float):
    """Step 4's white point over this block, as something broadcastable.

    Asked of the calibration object rather than recomputed, for the reason step 8
    records: step 4 may have fitted an illumination surface, and a flat white
    point taken from the slide average would undo the vignetting correction over
    a field at the edge of the section. `field_for` returns the triple or an
    HxWx3 surface and `optical_density` broadcasts both, so nothing here
    branches on which came back.
    """
    if hasattr(white, "field_for"):
        return white.field_for(x=field.x, y=field.y, size=field.size, mpp=mpp)
    return white


def segment_field(
    reader,
    field: Field,
    *,
    white,
    basis: StainBasis,
    base_mpp: float,
    model_mpp: float,
    remove_dab: bool | None = None,
    engine: str = "instanseg",
) -> SegmentedField:
    """Read `field` from `reader` and return its nuclei.

    `engine` is "instanseg" or "watershed". Both are shown the same pixels, which
    is what makes the comparison on screen a comparison of methods.
    """
    strip_dab = settings.nuclei_remove_dab if remove_dab is None else remove_dab

    tile = read_tile(
        reader,
        x=field.x,
        y=field.y,
        target_mpp=model_mpp,
        size=field.size,
        base_mpp=base_mpp,
    )
    rgb = np.asarray(tile.rgb, dtype=np.uint8)
    white_field = _white_field(white, field, tile.mpp)

    stain = haematoxylin_density(rgb, white_field, basis)
    shown = haematoxylin_only_rgb(rgb, white_field, basis) if strip_dab else rgb

    if engine == "watershed":
        from .watershed import segment as watershed_segment

        labels = watershed_segment(stain, mpp=tile.mpp)
    else:
        from app.nuclei.model import segment_array

        labels = segment_array(shown)

    nuclei = extract(
        labels,
        haematoxylin=stain,
        mpp=tile.mpp,
        x0=float(field.x),
        y0=float(field.y),
        level0_scale=tile.span / max(1, tile.size),
        border_px=settings.nuclei_border_margin_px,
        min_area_um2=settings.nuclei_min_area_um2,
    )

    return SegmentedField(
        field=field,
        shown=shown,
        rgb=rgb,
        haematoxylin=stain,
        labels=labels,
        nuclei=nuclei,
        mpp=tile.mpp,
        level0_scale=tile.span / max(1, tile.size),
        counted_mm2=counted_area_mm2(
            size=tile.size, border_px=settings.nuclei_border_margin_px, mpp=tile.mpp
        ),
    )


__all__ = ["SegmentedField", "segment_field"]
