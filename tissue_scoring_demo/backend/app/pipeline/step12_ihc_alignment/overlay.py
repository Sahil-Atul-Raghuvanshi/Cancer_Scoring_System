"""Step 10's pictures: the same borders, drawn on each of the two slides.

Borders only, nothing filled, and that is the point rather than a style
choice. This screen exists to be *judged* - a viewer is deciding whether the
regions landed on the right tissue - and a filled region hides the very pixels
they need to compare. An outline lets the tissue show through on both sides,
so the question "is that the same duct" can actually be answered.

The two panels are rendered the same way from the same code, so any difference
between them is a difference between the slides and not between two drawing
paths.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from app.ingestion.slide_reader import SlideReader
from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_COLOURS, SCORED
from app.pipeline.step08_tissue_type_segmentation.overlay import MAX_SIZE

#: Outline width in display pixels. Thick enough to follow on a 1600 px panel
#: of a 127,000 px slide without hiding the tissue it is claiming.
OUTLINE_PX = 3

#: Each region is labelled with its rank, because the three crops below the
#: panels are numbered and a viewer has to be able to tell which is which.
LABEL_OFFSET_PX = 6

REGION_COLOUR = CLASS_COLOURS[SCORED]


def _to_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def borders_on_slide_png(
    reader: SlideReader,
    regions: list[list[list[list[float]]]],
    *,
    slide_width: int,
    slide_height: int,
    max_size: int = MAX_SIZE,
    label: bool = True,
) -> bytes:
    """One slide's thumbnail with `regions` outlined on it.

    `regions` is a list of regions, each a list of rings, each a list of
    level-0 `[x, y]` vertices **of this slide**. The caller is responsible for
    having warped them into this slide's space first; nothing is transformed
    here, so the same function draws the H&E's own rings and the IHC's warped
    ones without knowing which it has.
    """
    thumbnail = reader.thumbnail_pil(max_size=max_size).convert("RGB")
    width, height = thumbnail.size
    scale_x = width / max(1, slide_width)
    scale_y = height / max(1, slide_height)

    draw = ImageDraw.Draw(thumbnail)
    for index, rings in enumerate(regions, start=1):
        for ring in rings:
            if len(ring) < 2:
                continue
            points = [(x * scale_x, y * scale_y) for x, y in ring]
            draw.line([*points, points[0]], fill=REGION_COLOUR, width=OUTLINE_PX)

        if label and rings and rings[0]:
            xs = [x for x, _ in rings[0]]
            ys = [y for _, y in rings[0]]
            anchor = (
                min(xs) * scale_x + LABEL_OFFSET_PX,
                min(ys) * scale_y + LABEL_OFFSET_PX,
            )
            draw.text(anchor, str(index), fill=REGION_COLOUR)

    return _to_png(thumbnail)


__all__ = ["OUTLINE_PX", "REGION_COLOUR", "borders_on_slide_png"]
