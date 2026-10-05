"""Top-3 ROI crops: a `ClassRegion`'s bounding box, read straight off the slide.

The border regions in `borders.py` are traced on step 8's tile grid - a few hundred
microns to a side per cell. That is fine for a boundary but not for something a
pathologist is meant to look at, so the picture shown for each of the largest DCIS and
invasive regions is not the coloured tile map: it is the original slide, cropped to
that region's bounding box (padded, so the boundary itself is visible with tissue
around it) and read at whatever pyramid level keeps the crop a sane number of pixels.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from app.ingestion.slide_reader import SlideReader
from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_COLOURS

from .borders import BORDER_CLASSES, ClassRegion

Bbox = tuple[int, int, int, int]  # x0, y0, x1, y1 - level-0 pixels, half-open

#: Outline width for `region_crop_with_borders_png`, in the crop's own display pixels.
#: Thicker than the six-panel picture's `OUTLINE_PX`: this image is shown large and on
#: its own, and a hairline that read fine at 1600 px would be lost against real slide
#: texture at this size.
CROP_OUTLINE_PX = 4


def region_bbox_level0(region: ClassRegion) -> Bbox:
    """The region's bounding box in level-0 pixels, from its outer ring.

    The outer ring is already in level-0 pixels (see `mask.level0_mapper`), so this
    needs no grid geometry of its own - just the extremes of its vertices.
    """
    outer = region.rings[0]
    xs = [x for x, _ in outer]
    ys = [y for _, y in outer]
    return min(xs), min(ys), max(xs), max(ys)


def padded_bbox(bbox: Bbox, pad_px: int, slide_width: int, slide_height: int) -> Bbox:
    """`bbox` expanded by `pad_px` on every side and clamped to the slide.

    The padding is what turns "exactly the region" into "the region with enough
    surrounding tissue to read" - a crop tight to the boundary shows nothing beyond it,
    which is the least useful crop of a lesion a pathologist can be shown.
    """
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - pad_px),
        max(0, y0 - pad_px),
        min(slide_width, x1 + pad_px),
        min(slide_height, y1 + pad_px),
    )


def _to_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def crop_png(reader: SlideReader, bbox: Bbox, *, max_size: int) -> bytes:
    """The slide's own pixels inside `bbox`, capped at `max_size` on the longest side.

    Reads at the coarsest pyramid level that still clears `max_size`, rather than
    always reading level 0 and downscaling after: a multi-millimetre invasive focus at
    level 0 is tens of thousands of pixels on a side, and decoding that just to throw
    most of it away is the kind of read that turns a fast step slow.
    """
    x0, y0, x1, y1 = bbox
    width, height = max(1, x1 - x0), max(1, y1 - y0)

    downsample = max(1.0, max(width, height) / max_size)
    level = reader.best_level_for_downsample(downsample)
    factor = reader.level_downsamples[level]

    size = (max(1, round(width / factor)), max(1, round(height / factor)))
    image = reader.read_region_pil((x0, y0), level, size)

    # The chosen level's own downsample may still leave the read a little over
    # budget - `best_level_for_downsample` picks the finest level *under or at* the
    # target, so one more resize guarantees the cap rather than approximating it.
    longest = max(image.size)
    if longest > max_size:
        scale = max_size / longest
        image = image.resize(
            (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))),
            Image.Resampling.LANCZOS,
        )

    return _to_png(image)


def _bbox_intersects(a: Bbox, b: Bbox) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def region_crop_with_borders_png(
    reader: SlideReader,
    regions: dict[int, list[ClassRegion]],
    class_id: int,
    index: int,
    *,
    pad_px: int,
    slide_width: int,
    slide_height: int,
    max_size: int,
) -> bytes:
    """One region a viewer selected off the borders panel, enlarged from the slide.

    Not limited to the top 3, and not limited to DCIS and invasive: `borders.py` keeps
    every component with no area cutoff, and a viewer can click any of them - uncertain
    included - so this reads `index` straight against `regions[class_id]` rather than
    against a pre-rendered top-3 file.

    Every border-class region whose bounding box reaches into the (padded) crop is
    drawn on it, in its own class colour - invasive included, whichever class was
    actually selected - the same way `borders_png` draws the flat class map, but on
    the real slide pixels rather than on the tile grid.
    """
    found = regions.get(class_id, [])
    if not 0 <= index < len(found):
        raise IndexError(f"no region at index {index} for class {class_id}")
    target = found[index]

    bbox = padded_bbox(region_bbox_level0(target), pad_px, slide_width, slide_height)
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

    scale_x = image.size[0] / width
    scale_y = image.size[1] / height
    draw = ImageDraw.Draw(image)

    for other_class_id in BORDER_CLASSES:
        colour = CLASS_COLOURS[other_class_id]
        for region in regions.get(other_class_id, []):
            if not _bbox_intersects(region_bbox_level0(region), bbox):
                continue
            for ring in region.rings:
                if len(ring) < 2:
                    continue
                points = [((x - x0) * scale_x, (y - y0) * scale_y) for x, y in ring]
                draw.line([*points, points[0]], fill=colour, width=CROP_OUTLINE_PX)

    return _to_png(image)


__all__ = [
    "Bbox",
    "crop_png",
    "padded_bbox",
    "region_bbox_level0",
    "region_crop_with_borders_png",
]
