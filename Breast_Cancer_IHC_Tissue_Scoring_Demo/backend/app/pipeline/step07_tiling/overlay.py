"""The two panels step 7 is meant to be read as.

  grid    the slide with the tile grid on it, kept tiles lit and dropped ones dim
  sample  one kept tile, as the model will actually receive it

The first is the guide's own picture and it carries the funnel by itself: a reader
sees the lit cells trace the section and the dim ones fill the empty two thirds of
the frame, which is what "412,164 tiles became 24,908" looks like.

The second exists because the funnel is about *counts* and the step's output is
about *pixels*, and a reader is owed a look at what a tile actually is. It is
drawn as step 6's haematoxylin channel rather than as RGB, because that is what
step 8 receives - and drawing the RGB here would quietly show the reader
something the model never sees.

**The grid is aggregated when it is finer than the lines that would draw it.** A
whole-slide grid can be six hundred cells across; drawn at 1600 px that is two
pixels a cell, so outlines would merge into a texture and the picture would stop
being a grid. Past `MAX_DRAWN` the cells are filled rather than outlined, which
is the same information at a scale that survives.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from app.pipeline.step05_optical_density.overlay import GROUND
from app.pipeline.step06_colour_deconvolution.overlay import HAEMATOXYLIN_RAMP, ramp

from .index import MAX_DRAWN, TileIndex

#: Longest edge of a panel served for display. Steps 5 and 6's number.
MAX_SIZE = 1600

#: Kept tiles carry the app's accent, because "kept" is this step's one decision.
KEPT_TINT = (56, 189, 248)
#: Dropped for having no tissue: the commonest case and the least interesting, so
#: it is drawn faintest.
OFF_TISSUE_TINT = (78, 90, 110)
#: Dropped for artefacts: rarer, and a fact about the slide rather than about its
#: shape, so it keeps the app-wide "be suspicious of this" amber.
UNCLEAN_TINT = (251, 191, 36)

#: The panels this step serves, in reading order.
PANELS: tuple[str, ...] = ("grid", "sample")


def _to_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _fit(image: Image.Image, longest: int = MAX_SIZE) -> Image.Image:
    width, height = image.size
    scale = min(1.0, longest / max(1, max(width, height)))
    if scale >= 1.0:
        return image
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )


def grid_png(thumbnail: np.ndarray, index: TileIndex) -> bytes:
    """Panel 1 - the slide with the grid on it, and what each cell was judged to be.

    All three populations are drawn, and the dropped ones matter as much as the
    kept ones. A reader shown only the survivors has to take the filter on trust;
    a reader shown the whole grid sees that the lit region is the section and the
    dark region is glass, which is the funnel as a picture rather than as a claim.
    """
    display = _fit(Image.fromarray(thumbnail, mode="RGB"))
    # Dimmed so the drawn cells read as an overlay rather than as part of the scan.
    faded = (np.asarray(display, dtype=np.float32) * 0.55).astype(np.uint8)
    canvas = Image.fromarray(faded, mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")

    width, height = canvas.size
    dense = max(index.cols, index.rows) > MAX_DRAWN

    for tile in index.tiles:
        box = (
            tile.fx * width,
            tile.fy * height,
            (tile.fx + tile.fw) * width,
            (tile.fy + tile.fh) * height,
        )

        if tile.kept:
            tint, fill, outline = KEPT_TINT, 62, 200
        elif tile.rejected_by == "clean":
            tint, fill, outline = UNCLEAN_TINT, 52, 150
        else:
            # Off the section. Outlined only, and faintly: there are tens of
            # thousands of these and filling them would bury the section under
            # its own background.
            tint, fill, outline = OFF_TISSUE_TINT, 0, 70

        if dense:
            # Too fine to outline - see the module docstring. Filled at the
            # outline's weight so the picture keeps the same reading.
            if fill or tile.rejected_by != "tissue":
                draw.rectangle(box, fill=(*tint, max(fill, 40)))
            continue

        draw.rectangle(
            box,
            fill=(*tint, fill) if fill else None,
            outline=(*tint, outline),
            width=1,
        )

    return _to_png(canvas)


def sample_png(haematoxylin: np.ndarray, *, high: float) -> bytes:
    """Panel 2 - one kept tile as the model receives it: the haematoxylin channel.

    Drawn through step 6's own ramp and step 6's own scale, imported rather than
    redefined, so a reader moving between the two screens is looking at the same
    picture of the same quantity. A second blue ramp here would be a second answer
    to "how much counterstain is that", and the two would drift.

    `high` is the stretch, handed in from step 6's report for the same reason.
    """
    scaled = np.maximum(np.asarray(haematoxylin, dtype=np.float32), 0.0) / max(high, 1e-6)
    return _to_png(_fit(Image.fromarray(ramp(scaled, HAEMATOXYLIN_RAMP), mode="RGB")))


def empty_png(size: int = 512) -> bytes:
    """A plain ground, for when there is no sample tile to draw.

    A real state - a slide whose every tile failed both gates has no sample - and
    the report says so in words. A panel that 404'd would make a reported fact
    look like a failure.
    """
    return _to_png(Image.new("RGB", (size, size), GROUND))
