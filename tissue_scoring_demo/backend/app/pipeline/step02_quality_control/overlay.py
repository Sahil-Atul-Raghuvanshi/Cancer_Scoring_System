"""Turning masks and feature grids into images the browser can show.

Three renderings, and they answer different questions:

  artefact overlay   the slide, with only the artefacts tinted. Tissue and
                     glass stay untouched, because the point of the picture is
                     "here is what will be thrown away", not "here is a
                     segmentation".
  tissue overlay     pass 1's output on its own, so the two models are visibly
                     two models rather than one opaque result.
  feature heatmap    one classical metric over the patch grid, which is the
                     explanation that a red blob on the overlay is asking for.

Colours come from `app.pipeline.step02_quality_control.classes` - GrandQC's own
palette - so anything rendered here can be compared directly against the
figures in their paper.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from app.pipeline.step02_quality_control import classes as qc_classes
from app.pipeline.step02_quality_control.features import metric_spec

#: How strongly an artefact is tinted. High enough to find at thumbnail size,
#: low enough that the tissue underneath is still legible.
ARTEFACT_ALPHA = 0.55

#: Longest edge of a rendered overlay. A whole-slide mask at 1.5 um/px is
#: ~19,000 px across; nothing in the UI shows more than a couple of thousand.
DEFAULT_MAX_SIZE = 1600


def _to_png(image: Image.Image, *, optimise: bool = True) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=optimise)
    return buffer.getvalue()


def _fit(size: tuple[int, int], longest: int) -> tuple[int, int]:
    width, height = size
    scale = min(1.0, longest / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


# --- mask rendering ----------------------------------------------------------


def mask_png(mask: np.ndarray) -> bytes:
    """The raw class mask as an indexed-colour PNG.

    Indexed rather than RGB so the file stays one byte per pixel and the class
    ids survive a round-trip - this is the artefact a user can download and
    load into QuPath alongside the slide.
    """
    image = Image.fromarray(mask, mode="P")
    flat: list[int] = []
    for colour in qc_classes.palette_rows():
        flat.extend(colour)
    image.putpalette(flat)
    return _to_png(image)


def colourise(mask: np.ndarray) -> np.ndarray:
    """Class ids to RGB, using GrandQC's palette."""
    lookup = np.zeros((256, 3), dtype=np.uint8)
    for item in qc_classes.QC_CLASSES:
        lookup[item.id] = item.colour
    return lookup[mask]


def artefact_overlay_png(
    reader: Any,
    mask: np.ndarray,
    *,
    max_size: int = DEFAULT_MAX_SIZE,
    alpha: float = ARTEFACT_ALPHA,
) -> bytes:
    """The slide with its artefacts tinted, and nothing else touched.

    Only classes flagged `is_artefact` are painted. Clean tissue, glass and the
    unanalysed margin are left as the original pixels, so what the viewer sees
    highlighted is exactly the set of pixels step 3 will not receive.
    """
    height, width = mask.shape
    target = _fit((width, height), max_size)

    base = reader.thumbnail_pil(max(target)).convert("RGB")
    if base.size != target:
        base = base.resize(target, Image.Resampling.LANCZOS)

    small = np.asarray(
        Image.fromarray(mask).resize(target, Image.Resampling.NEAREST)
    )

    canvas = np.asarray(base).astype(np.float32)
    tint = colourise(small).astype(np.float32)

    artefacts = np.isin(small, [item.id for item in qc_classes.ARTEFACT_CLASSES])
    blended = np.where(
        artefacts[..., None], canvas * (1.0 - alpha) + tint * alpha, canvas
    )

    return _to_png(Image.fromarray(blended.astype(np.uint8)))


def mask_overlay_png(mask: np.ndarray, *, max_size: int = DEFAULT_MAX_SIZE) -> bytes:
    """The full class map as a flat colour image - every class, no slide under it."""
    height, width = mask.shape
    target = _fit((width, height), max_size)
    small = np.asarray(Image.fromarray(mask).resize(target, Image.Resampling.NEAREST))
    return _to_png(Image.fromarray(colourise(small)))


def tissue_overlay_png(
    thumbnail: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    max_size: int = DEFAULT_MAX_SIZE,
    alpha: float = 0.35,
) -> bytes:
    """Pass 1's tissue map over the thumbnail it was computed from.

    Tissue is tinted, glass is left alone - the inverse of how it is tempting
    to draw it, but the tinted region is the region that will be worked on, and
    consistency with the artefact overlay matters more than novelty.
    """
    target = _fit((thumbnail.shape[1], thumbnail.shape[0]), max_size)

    base = np.asarray(
        Image.fromarray(thumbnail).resize(target, Image.Resampling.LANCZOS)
    ).astype(np.float32)
    small = np.asarray(
        Image.fromarray(tissue_mask).resize(target, Image.Resampling.NEAREST)
    )

    # GrandQC's tissue-detector palette: blue tissue, grey background.
    tint = np.zeros((*small.shape, 3), dtype=np.float32)
    tint[small == 0] = (50, 50, 250)

    blended = np.where((small == 0)[..., None], base * (1.0 - alpha) + tint * alpha, base)
    return _to_png(Image.fromarray(blended.astype(np.uint8)))


# --- feature heatmaps --------------------------------------------------------

#: A perceptually-ordered ramp, dark to bright, defined inline so the backend
#: does not pull in matplotlib for five colours.
_RAMP = np.array(
    [
        (12, 16, 24),
        (38, 60, 110),
        (30, 120, 150),
        (90, 180, 130),
        (220, 210, 110),
        (250, 250, 235),
    ],
    dtype=np.float32,
)


def _ramp(values: np.ndarray) -> np.ndarray:
    """Map 0..1 through the ramp, linearly interpolating between stops."""
    clipped = np.clip(values, 0.0, 1.0) * (len(_RAMP) - 1)
    low = np.floor(clipped).astype(int)
    high = np.minimum(low + 1, len(_RAMP) - 1)
    weight = (clipped - low)[..., None]
    return (_RAMP[low] * (1 - weight) + _RAMP[high] * weight).astype(np.uint8)


def feature_heatmap_png(
    cells: list[Any],
    *,
    metric: str,
    grid_cols: int,
    grid_rows: int,
    cell_px: int = 12,
) -> tuple[bytes, dict[str, float]]:
    """One metric over the patch grid, plus the value range it was scaled to.

    Scaled between the 2nd and 98th percentiles of the measured patches, not
    the min and max: a single dust speck otherwise compresses the whole slide
    into one flat colour. Patches that were never measured - glass - are drawn
    as the background, not as a zero, because a metric of zero and a metric of
    "not asked" are different statements.
    """
    if metric_spec(metric) is None:
        raise ValueError(f"unknown metric {metric!r}")

    grid = np.full((grid_rows, grid_cols), np.nan, dtype=np.float32)
    for cell in cells:
        if cell.features is None or metric not in cell.features:
            continue
        if 0 <= cell.row < grid_rows and 0 <= cell.col < grid_cols:
            grid[cell.row, cell.col] = cell.features[metric]

    measured = grid[~np.isnan(grid)]
    if measured.size:
        low, high = (float(value) for value in np.percentile(measured, (2.0, 98.0)))
    else:
        low, high = 0.0, 1.0
    span = high - low if high > low else 1.0

    normalised = (grid - low) / span
    rgb = _ramp(np.nan_to_num(normalised, nan=0.0))
    rgb[np.isnan(grid)] = (12, 16, 24)

    image = Image.fromarray(rgb).resize(
        (grid_cols * cell_px, grid_rows * cell_px), Image.Resampling.NEAREST
    )
    return _to_png(image), {"min": low, "max": high}


def region_png(image: Image.Image) -> bytes:
    """A single inspected region, unmodified."""
    return _to_png(image.convert("RGB"))
