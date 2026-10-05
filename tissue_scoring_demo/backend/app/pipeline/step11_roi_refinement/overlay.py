"""Step 11's pictures: the square, the shape, and the two of them on one slide.

The step's claim is a comparison - *this coarse box became this boundary* - so the
pictures are built as a matched pair. Same crop, same box, same scale, same drawing code,
and the only difference between the left panel and the right one is which geometry is
drawn on it. Rendering them from two paths would let a difference between the two
drawings be mistaken for a difference between the two answers.

Four per region:

    input.png           the padded H&E crop, nothing drawn - what BEETLE was shown
    tile.png            the same crop with step 8's window staircase on it   (left)
    beetle_mask.png     BEETLE's five classes as a flat map, no slide under it
    beetle_overlay.png  the same crop with the refined boundary on it        (right)

and two for the slide as a whole, `coarse.png` and `refined.png`, which are the same
comparison at the scale of the section.

**`beetle_mask.png` earns its place by being the panel that cannot flatter.** The overlay
draws a boundary, and a boundary drawn on tissue is persuasive whatever it encloses. The
flat class map shows what the network actually said about every pixel of the box - how
much is stroma, how much is in-situ epithelium, how much it called background - and a
refinement that had simply returned the whole rectangle would be obvious in it at a
glance.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from app.ingestion.slide_reader import SlideReader
from app.pipeline.step08_tissue_type_segmentation import beetle
from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_COLOURS, SCORED
from app.pipeline.step08_tissue_type_segmentation.overlay import MAX_SIZE
from app.pipeline.step08_tissue_type_segmentation.pixels import OUTSIDE
from app.pipeline.step09_roi_mask.crops import Bbox
from app.pipeline.step10_roi_selection.overlay import draw_pieces, read_crop

#: Colour of the coarse tile boundary - step 8's own invasive colour, so the left panel
#: here, the cards on step 10 and the class map on step 8 all mean the same thing by the
#: same colour.
TILE_COLOUR = CLASS_COLOURS[SCORED]

#: Colour of the refined boundary. BEETLE's own invasive code, for the same reason: the
#: right panel is BEETLE's answer and it is drawn in BEETLE's palette.
REFINED_COLOUR = beetle.CLASS_COLOURS[beetle.SCORED_CODE]

#: Fill opacity under each boundary. Light enough to read the tissue through.
FILL_ALPHA = 64

OUTLINE_PX = 3

#: What `OUTSIDE` is drawn as in the flat class map: near-white, so "no window ran here"
#: reads as absence rather than as a sixth class. BEETLE's own code 0 is a *claim* -
#: "this looks like background" - and the two must not share a colour.
OUTSIDE_COLOUR = (244, 244, 245)


def _to_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def before_pngs(
    reader: SlideReader,
    box: Bbox,
    *,
    tile_rings,
    max_size: int,
) -> dict[str, bytes]:
    """The two panels that do not need the segmentation: the crop, and the square on it.

    Written when a region *starts* rather than when it finishes, because both are
    knowable before BEETLE has seen anything and the screen needs them then: `input.png`
    is the picture the live paint lands on, and `tile.png` is the "before" it is being
    compared against. Without this the comparison could only appear once the region was
    over, which is precisely when watching it stops being useful.

    `crop_pngs` rewrites both at the end from the read that also produces the overlay,
    and that is deliberate rather than wasteful: the left and right panels have to come
    from one read to be a fair pair, and this one is a preview of the left, not a
    replacement for it.
    """
    base, scale = read_crop(reader, box, max_size=max_size)
    tile = draw_pieces(
        base.copy(), [tile_rings], box, scale,
        colour=TILE_COLOUR, fill_alpha=FILL_ALPHA, width=OUTLINE_PX,
    )
    return {"input.png": _to_png(base), "tile.png": _to_png(tile)}


def crop_pngs(
    reader: SlideReader,
    box: Bbox,
    *,
    tile_rings,
    refined_pieces,
    max_size: int,
) -> dict[str, bytes]:
    """The three slide-backed panels for one region, from a single read.

    One read, three pictures: they are the same pixels with different things drawn on
    them, and reading the box three times would be three chances for the pyramid level to
    differ and the panels to stop being comparable.

    `tile_rings` is one piece - step 9's candidate is connected by construction.
    `refined_pieces` is a list of them, because a single coarse box routinely refines
    into several separate foci, and each carries its own holes.
    """
    base, scale = read_crop(reader, box, max_size=max_size)

    tile = draw_pieces(
        base.copy(), [tile_rings], box, scale,
        colour=TILE_COLOUR, fill_alpha=FILL_ALPHA, width=OUTLINE_PX,
    )
    refined = draw_pieces(
        base.copy(), refined_pieces, box, scale,
        colour=REFINED_COLOUR, fill_alpha=FILL_ALPHA, width=OUTLINE_PX,
    )
    return {
        "input.png": _to_png(base),
        "tile.png": _to_png(tile),
        "beetle_overlay.png": _to_png(refined),
    }


def class_map_png(mask: np.ndarray, *, max_size: int) -> bytes:
    """BEETLE's five classes over the box, as flat colour with no slide underneath.

    Nearest-neighbour on the way down, unlike everything else in this pipeline that
    resamples: this is a *label* map and there is no probability left to average - the
    argmax already happened, in `pixels.segment`, where the averaging was done properly
    over the probability planes. Interpolating labels here would invent colours between
    two classes that mean a third one.
    """
    height, width = mask.shape
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = OUTSIDE_COLOUR
    for code, colour in beetle.CLASS_COLOURS.items():
        canvas[mask == code] = colour
    canvas[mask == OUTSIDE] = OUTSIDE_COLOUR

    image = Image.fromarray(canvas, mode="RGB")
    longest = max(image.size)
    if longest > max_size:
        scale = max_size / longest
        image = image.resize(
            (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))),
            Image.Resampling.NEAREST,
        )
    return _to_png(image)


def slide_png(
    reader: SlideReader,
    pieces: list[list[list[list[float]]]],
    *,
    slide_width: int,
    slide_height: int,
    colour: tuple[int, int, int],
    max_size: int = MAX_SIZE,
    fill_alpha: int = 0,
    label: bool = True,
) -> bytes:
    """The whole slide with `pieces` drawn on it, each an outer ring then its holes.

    Used twice with the same code and different geometry - once for the coarse squares
    and once for the refined boundaries - which is what makes the pair of panels a fair
    comparison. Nothing is transformed here; the caller is responsible for the rings
    already being in this slide's coordinates.
    """
    thumbnail = reader.thumbnail_pil(max_size=max_size).convert("RGB")
    width, height = thumbnail.size
    scale_x = width / max(1, slide_width)
    scale_y = height / max(1, slide_height)

    # `draw_pieces` projects through one scale, and a thumbnail's two axes can round to
    # scales that differ in the last decimal. The rings are rescaled here instead, so the
    # projection is exact on both axes and the drawing code stays shared.
    scaled = [
        [[[x * scale_x, y * scale_y] for x, y in ring] for ring in rings]
        for rings in pieces
    ]
    thumbnail = draw_pieces(
        thumbnail,
        scaled,
        (0, 0, width, height),
        1.0,
        colour=colour,
        fill_alpha=fill_alpha,
        width=2,
    )

    if label:
        draw = ImageDraw.Draw(thumbnail)
        for index, rings in enumerate(scaled, start=1):
            if not rings or not rings[0]:
                continue
            xs = [x for x, _ in rings[0]]
            ys = [y for _, y in rings[0]]
            draw.text((min(xs) + 6, min(ys) + 6), str(index), fill=colour)

    return _to_png(thumbnail)


__all__ = [
    "FILL_ALPHA",
    "OUTLINE_PX",
    "OUTSIDE_COLOUR",
    "REFINED_COLOUR",
    "TILE_COLOUR",
    "class_map_png",
    "crop_pngs",
    "slide_png",
]
