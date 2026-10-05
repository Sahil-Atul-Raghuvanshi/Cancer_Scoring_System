"""Step 9's pictures: the seven panels the guide asks this screen to march through.

    seed       the raw invasive windows, blocky and speckled - where we start
    smoothed   P(invasive) blurred, as a heatmap - the reason we smooth probabilities
    binary     the threshold, before any morphology
    region     the finished ROI, with the protected in-situ drawn in its own colour
    outline    the ROI boundary on the scan, which is the thing a pathologist judges
    borders    every class as an outline only, tissue included - see `borders_png` below
    borders_on_slide   the same borders on the scan, each class faintly tinted

The march is the argument, not decoration. A reader who sees only the last panel
cannot tell how much of that clean region was measured and how much was invented by a
closing radius, and this step invents more than any other in the pipeline.

**`region` draws the protected in-situ, and that is the panel's whole point.** A hole
in the ROI is easy to read as a gap in the model's confidence. Here it is usually the
opposite - a duct the model identified as in-situ, which Rule 5 then removed from the
denominator on purpose. Colouring those cells in step 8's own in-situ blue says which
of the two happened, on the picture, rather than in a footnote.

Colours are step 8's, imported rather than restated: the ROI is drawn in the class
colour of the class it is made of, so a viewer moving between the two screens is not
asked to learn a second palette for the same tissue.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from scipy import ndimage

from app.pipeline.step05_optical_density.overlay import GROUND
from app.pipeline.step06_colour_deconvolution.overlay import ramp
from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_COLOURS, SCORED
from app.pipeline.step08_tissue_type_segmentation.inference import ClassMap
from app.pipeline.step08_tissue_type_segmentation.overlay import MAX_SIZE, cell_field, label_image
from app.pipeline.step08_tissue_type_segmentation.uncertainty import UNCERTAIN

from .mask import BORDER_GAP_CELLS, IN_SITU, RoiMask

#: The panels this step serves, in reading order.
PANELS: tuple[str, ...] = (
    "seed",
    "smoothed",
    "binary",
    "region",
    "outline",
    "borders",
    "borders_on_slide",
)

#: The ROI's colour is the invasive class's colour. Same tissue, same colour.
ROI_COLOUR = CLASS_COLOURS[SCORED]
#: And the carved-out in-situ keeps step 8's in-situ blue.
PROTECTED_COLOUR = CLASS_COLOURS[IN_SITU]

ROI_ALPHA = 0.5
SLIDE_DIM = 0.6

#: Outline width in display pixels. Thick enough to survive a 1600 px panel of a
#: 127 000 px slide, thin enough not to hide the tissue it is claiming.
OUTLINE_PX = 3

#: How strongly a class tints the scan underneath it on `borders_on_slide_png`. Faint
#: on purpose: the fill only has to say which side of a border is which class, and the
#: tissue has to stay legible through it - a reader who cannot see the nuclei inside a
#: red region cannot check the call the colour is making.
CLASS_FILL_ALPHA = 0.22

#: The classes that get a fill there. Stroma is outlined like the rest and never
#: filled: it is most of the section, so tinting it would tint the whole slide and
#: leave the three classes a reader is actually looking for competing with a wash.
FILLED_CLASSES: tuple[int, ...] = (IN_SITU, SCORED, UNCERTAIN)

#: Heatmap for the smoothed probability: the ground where there is nothing, through
#: the invasive colour where the model is sure. Starts at 0 rather than 1/3 because
#: this is one class's probability, not a three-way argmax - it really can be zero.
HEAT_RAMP: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, GROUND),
    (0.35, (69, 26, 34)),
    (0.5, (158, 42, 43)),
    (0.75, ROI_COLOUR),
    (1.0, (255, 228, 226)),
)


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


def _resample(field: np.ndarray, class_map: ClassMap, shape: tuple[int, int], fill):
    """A grid-shaped array onto a display canvas, `fill` off the grid.

    Nearest-neighbour through step 8's `cell_field`, which is the same mapping the
    class map is drawn with - so the ROI and the classes underneath it land on the
    same pixels and a viewer comparing the two screens is comparing like with like.
    """
    rows, cols = cell_field(class_map, shape)
    grid = class_map.grid
    valid = ((rows >= 0) & (rows < grid.rows))[:, None] & ((cols >= 0) & (cols < grid.cols))[None, :]
    picked = field[
        np.clip(rows, 0, grid.rows - 1)[:, None], np.clip(cols, 0, grid.cols - 1)[None, :]
    ]
    return np.where(valid, picked, fill)


def _edge(mask: np.ndarray, width: int) -> np.ndarray:
    """The boundary band of a boolean image, `width` pixels thick, drawn inward.

    Inward so the outline never claims a pixel the region does not hold - an outline
    drawn outward would report a larger ROI than the one that was measured, which on
    a 1600 px panel of a 127 000 px slide is a visible amount of tissue.
    """
    from scipy import ndimage

    return mask & ~ndimage.binary_erosion(mask, iterations=max(1, width))


def _flat(field: np.ndarray, colour, shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    canvas[:] = GROUND
    canvas[field] = colour
    return canvas


def seed_png(class_map: ClassMap, roi: RoiMask, shape: tuple[int, int]) -> bytes:
    """Panel 1 - the windows the model called invasive, exactly as it called them."""
    field = _resample(class_map.labels == SCORED, class_map, shape, False)
    return _to_png(Image.fromarray(_flat(field, ROI_COLOUR, shape), mode="RGB"))


def smoothed_png(class_map: ClassMap, roi: RoiMask, shape: tuple[int, int]) -> bytes:
    """Panel 2 - P(invasive) after the blur. The reason smoothing runs on probabilities.

    Recomputed from the stored probabilities rather than carried on `RoiMask`: a
    picture is not worth `rows x cols` floats of extra state on an object every later
    step reads, and the blur is one call.
    """
    from scipy import ndimage

    sigma = roi.params.sigma
    invasive = np.asarray(class_map.probabilities, dtype=np.float32)[..., SCORED]
    smoothed = ndimage.gaussian_filter(invasive, sigma) if sigma > 0 else invasive
    field = _resample(np.where(class_map.grid.inside, smoothed, 0.0), class_map, shape, 0.0)
    return _to_png(Image.fromarray(ramp(field, HEAT_RAMP), mode="RGB"))


def binary_png(class_map: ClassMap, roi: RoiMask, shape: tuple[int, int]) -> bytes:
    """Panel 3 - the threshold, before morphology. What the ROI would be with no help."""
    from scipy import ndimage

    sigma = roi.params.sigma
    invasive = np.asarray(class_map.probabilities, dtype=np.float32)[..., SCORED]
    smoothed = ndimage.gaussian_filter(invasive, sigma) if sigma > 0 else invasive
    seed = (smoothed >= roi.params.threshold) & class_map.grid.inside
    field = _resample(seed, class_map, shape, False)
    return _to_png(Image.fromarray(_flat(field, ROI_COLOUR, shape), mode="RGB"))


def region_png(class_map: ClassMap, roi: RoiMask, shape: tuple[int, int]) -> bytes:
    """Panel 4 - the finished ROI, with the in-situ that was carved out of it.

    The blue is the Rule 5 decision made visible: those cells were inside the closed
    region and were removed because the model calls them in-situ. Without them the
    holes in this panel would be unreadable.
    """
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    canvas[:] = GROUND
    canvas[_resample(roi.mask, class_map, shape, False)] = ROI_COLOUR

    if roi.protected_cells:
        protect = (
            np.asarray(class_map.probabilities, dtype=np.float32)[..., IN_SITU]
            >= roi.params.protect_in_situ
        ) & class_map.grid.inside
        canvas[_resample(protect, class_map, shape, False)] = PROTECTED_COLOUR

    return _to_png(Image.fromarray(canvas, mode="RGB"))


def outline_png(thumbnail: np.ndarray, class_map: ClassMap, roi: RoiMask) -> bytes:
    """Panel 5 - the ROI boundary on the scan. The panel a pathologist argues with.

    Filled at low alpha *and* outlined: the fill says which side of the boundary is
    scored, the outline says exactly where the boundary is, and on a region with forty
    holes neither alone is legible.
    """
    display = _fit(Image.fromarray(thumbnail, mode="RGB"))
    width, height = display.size
    shape = (height, width)

    faded = np.asarray(display, dtype=np.float32) * SLIDE_DIM
    region = _resample(roi.mask, class_map, shape, False)

    alpha = (region[..., None] * ROI_ALPHA).astype(np.float32)
    blended = faded * (1.0 - alpha) + np.array(ROI_COLOUR, dtype=np.float32) * alpha
    blended[_edge(region, OUTLINE_PX)] = ROI_COLOUR

    return _to_png(Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB"))


def borders_png(class_map: ClassMap, shape: tuple[int, int]) -> bytes:
    """Panel 6 - step 8's class map, as borders only - nothing on it is filled.

    Every class is traced, tissue included: `non_epithelium` is mostly one component,
    so outlining it draws the section's own silhouette rather than thousands of stroma
    patches, and a viewer sees where the tissue is before they see where the model
    changed its mind inside it. Each class is drawn in its own colour - the same
    `CLASS_COLOURS` the filled panels use - so the outline a viewer is looking at is
    still identifiable without a second legend.

    Adjacent tiles of the same class merge into one region wherever they touch -
    `borders.class_regions` traces exactly this for the three the frontend lets a
    viewer click through - but this picture is drawn straight from the label field
    rather than from those regions, so stroma's outline does not need a `ClassRegion`
    of its own.
    """
    field = label_image(class_map, shape)

    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    canvas[:] = GROUND

    # The same gap the exported geometry carries, in display pixels. Two regions of
    # different classes that touch share their cell edge exactly, so an outline drawn
    # flush against it would sit against its neighbour's with nothing between them; each
    # is stood off by this first, leaving twice it as a visible gap. Derived from the
    # grid rather than fixed, so it stays the same distance on the slide whichever field
    # of view drew the map.
    # Floored at 2 px: the exact conversion rounds to 1 px at most fields of view, and a
    # single pixel between two dark outlines is not a gap anybody can see. The exported
    # geometry keeps the exact value - this floor is legibility on a downsampled panel,
    # and nothing measures the picture.
    cell_px = shape[1] / max(1, class_map.grid.cols)
    gap = max(2, int(round(BORDER_GAP_CELLS * cell_px)))

    for class_id, colour in CLASS_COLOURS.items():
        here = field == class_id
        if not here.any():
            continue
        if gap > 0:
            here = ndimage.binary_erosion(here, iterations=gap)
        canvas[_edge(here, OUTLINE_PX)] = colour

    return _to_png(Image.fromarray(canvas, mode="RGB"))


def borders_on_slide_png(thumbnail: np.ndarray, class_map: ClassMap) -> bytes:
    """Panel 7 - the same borders as `borders_png`, drawn on the scan itself.

    `borders_png` is the geometry with nothing else in the frame, which is the right
    picture for checking a shape and the wrong one for checking a call: an outline on
    black says where the model drew a boundary but not what tissue it drew it around.
    This panel answers the second question by putting the identical borders on the
    scan and tinting each enclosed region with its own class colour at
    `CLASS_FILL_ALPHA` - faint enough that the nuclei inside it still read, strong
    enough to say which side of a shared border is which class.

    Nothing here is measured and nothing is re-derived. The label field, the
    stand-off gap and the outline width are `borders_png`'s, so the two panels are the
    same geometry twice and a reader moving between them is not comparing two
    different traces.

    Every fill is laid down before any border, so a region's tint never washes over
    its neighbour's outline - with the classes iterated in dictionary order the last
    class drawn would otherwise dim the borders of the ones under it.
    """
    display = _fit(Image.fromarray(thumbnail, mode="RGB"))
    width, height = display.size
    shape = (height, width)

    field = label_image(class_map, shape)
    canvas = np.asarray(display, dtype=np.float32)

    cell_px = shape[1] / max(1, class_map.grid.cols)
    gap = max(2, int(round(BORDER_GAP_CELLS * cell_px)))

    edges: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for class_id, colour in CLASS_COLOURS.items():
        here = field == class_id
        if not here.any():
            continue
        if gap > 0:
            here = ndimage.binary_erosion(here, iterations=gap)
        if class_id in FILLED_CLASSES:
            alpha = (here[..., None] * CLASS_FILL_ALPHA).astype(np.float32)
            canvas = canvas * (1.0 - alpha) + np.array(colour, dtype=np.float32) * alpha
        edges.append((_edge(here, OUTLINE_PX), colour))

    for band, colour in edges:
        canvas[band] = colour

    return _to_png(Image.fromarray(canvas.clip(0, 255).astype(np.uint8), mode="RGB"))


__all__ = [
    "PANELS",
    "binary_png",
    "borders_on_slide_png",
    "borders_png",
    "outline_png",
    "region_png",
    "seed_png",
    "smoothed_png",
]
