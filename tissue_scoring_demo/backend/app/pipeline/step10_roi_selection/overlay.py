"""One card per candidate: the slide's own pixels, with the tile boundary on them.

The card has to answer one question - *is this tissue worth spending BEETLE on* - and
that means it has to show two things at once: what the tissue looks like, and what the
tile model claimed about it. Either alone is useless here. The coloured class map is not
tissue a person can judge; the bare crop does not say where the model drew its line.

So the card is the crop with the region's own boundary drawn on it, and the boundary is
drawn as the staircase it actually is. **Not smoothed, and that is the whole argument of
the next step.** Step 8 answered once per 224 um window, so its boundary runs along
window edges; drawing it as a curve would claim a precision the tile model does not have
and would make step 11's refinement look like a cosmetic change rather than the
difference between a square and a tumour.

**A piece is an outer ring followed by its holes**, which is the convention step 9 traces
in, step 12 warps in and `step13_nuclei_segmentation.sampling.rasterise` fills in. This
module honours it rather than treating a list of rings as a flat set, because a hole is
how in-situ disease carved out of a tumour is represented and filling it would paint the
one thing Rule 5 says is not scored.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from app.ingestion.slide_reader import SlideReader
from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_COLOURS, SCORED
from app.pipeline.step09_roi_mask.crops import Bbox, padded_bbox

#: Outline width in the card's own display pixels.
OUTLINE_PX = 3

#: Opacity of the fill under a boundary. Light: the point is to read the tissue
#: *through* the claim, so the fill says "this whole box is one answer" without hiding
#: the ducts that make the answer judgeable.
FILL_ALPHA = 60

TILE_COLOUR = CLASS_COLOURS[SCORED]


def _to_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def read_crop(reader: SlideReader, bbox: Bbox, *, max_size: int) -> tuple[Image.Image, float]:
    """The slide inside `bbox`, capped at `max_size`, and the scale that got it there.

    The scale is returned rather than recomputed by the caller because everything drawn
    on this image has to be mapped through exactly the one that produced it, and a second
    derivation of "how many display pixels is a level-0 pixel" is a second chance to put
    a boundary somewhere the tissue is not.

    Reads at the coarsest pyramid level that still clears `max_size` - `crops.crop_png`'s
    argument, and for its reason: a multi-millimetre region at level 0 is tens of
    thousands of pixels a side.
    """
    x0, y0, x1, y1 = bbox
    width, height = max(1, x1 - x0), max(1, y1 - y0)

    downsample = max(1.0, max(width, height) / max_size)
    level = reader.best_level_for_downsample(downsample)
    factor = reader.level_downsamples[level]

    size = (max(1, round(width / factor)), max(1, round(height / factor)))
    image = reader.read_region_pil((x0, y0), level, size).convert("RGB")

    longest = max(image.size)
    if longest > max_size:
        scale = max_size / longest
        image = image.resize(
            (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))),
            Image.Resampling.LANCZOS,
        )

    return image, image.size[0] / width


def draw_pieces(
    image: Image.Image,
    pieces,
    bbox: Bbox,
    scale: float,
    *,
    colour: tuple[int, int, int] = TILE_COLOUR,
    fill_alpha: int = 0,
    width: int = OUTLINE_PX,
) -> Image.Image:
    """Draw `pieces` - each an outer ring then its holes - onto a crop of `bbox`.

    The fill is built as a single coverage mask before any colour is applied, rather than
    as one translucent polygon per piece. Two reasons, and the second is the one that
    would bite: compositing per polygon makes overlapping pieces darker than isolated
    ones, so a fill opacity would read as a claim about density; and a hole punched by
    subtraction has to be punched out of the *same* layer the outer ring was drawn into,
    which is only true if there is one layer.
    """
    x0, y0 = bbox[0], bbox[1]

    def project(ring):
        return [((x - x0) * scale, (y - y0) * scale) for x, y in ring]

    if fill_alpha > 0:
        coverage = Image.new("L", image.size, 0)
        painter = ImageDraw.Draw(coverage)
        for rings in pieces:
            for index, ring in enumerate(rings):
                if len(ring) < 3:
                    continue
                # 0 for every ring after the first: a hole is cut back out of the piece
                # it sits in. The same convention `sampling.rasterise` fills with.
                painter.polygon(project(ring), fill=0 if index else fill_alpha)

        layer = Image.new("RGB", image.size, colour)
        image = Image.composite(
            Image.blend(image.convert("RGB"), layer, fill_alpha / 255.0),
            image.convert("RGB"),
            coverage.point(lambda value: 255 if value else 0),
        )

    draw = ImageDraw.Draw(image)
    for rings in pieces:
        for ring in rings:
            if len(ring) < 2:
                continue
            points = project(ring)
            draw.line([*points, points[0]], fill=colour, width=width)
    return image


def card_png(
    reader: SlideReader,
    rings,
    bbox: Bbox,
    *,
    pad_px: int,
    slide_width: int,
    slide_height: int,
    max_size: int,
) -> bytes:
    """One candidate's card: the padded crop with its tile boundary outlined and filled.

    Padded, because a crop tight to the boundary shows nothing beyond it and the
    judgement being made is partly about what surrounds the region - a focus sitting in
    stroma reads differently from one sitting against the section's edge.

    `rings` is one candidate's rings, which are one piece: step 9's components are
    connected by construction, so a candidate is an outer boundary and its holes.
    """
    padded = padded_bbox(bbox, pad_px, slide_width, slide_height)
    image, scale = read_crop(reader, padded, max_size=max_size)
    image = draw_pieces(image, [rings], padded, scale, fill_alpha=FILL_ALPHA)
    return _to_png(image)


__all__ = [
    "FILL_ALPHA",
    "OUTLINE_PX",
    "TILE_COLOUR",
    "card_png",
    "draw_pieces",
    "read_crop",
]
