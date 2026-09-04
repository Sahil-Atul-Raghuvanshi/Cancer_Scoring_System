"""Running the two GrandQC passes over a slide.

This is a reimplementation of GrandQC's inference loop against this project's
own slide reader, not a copy of their scripts. The method, the class coding and
the weights are theirs (see `app.pipeline.step02_quality_control` for the
citation); what changed is the plumbing, for three reasons:

  * their scripts drive a *folder* of slides through argparse and write results
    to disk. This app has one slide, an upload id, and an HTTP response.
  * they read every patch from level 0 and downsample it in software. On a 40x
    scan that is 15x more pixels off disk per patch than the model can use, so
    by default we read the nearest at-or-finer pyramid level instead. Set
    `qc_read_from_level_0` to reproduce upstream exactly.
  * they require the OpenSlide C library. This app reads slides through
    tiffslide, which is pure Python and therefore actually installable on
    Windows without a binary hunt.

The order of the two passes is not an optimisation, it is the design. Tissue
detection runs first at 10 um/px, where a whole slide is a few thousand pixels
and a forward pass costs under a second. The artefact model then only visits
patches that contain tissue - typically a third of the grid - which is what
makes a whole-slide artefact map affordable on a CPU at all.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from app.common.imaging import otsu_threshold, saturation_channel
from app.core.logging import get_logger
from app.pipeline.step02_quality_control import classes as qc_classes
from app.pipeline.step02_quality_control.features import METRIC_KEYS, block_features, tile_features
from app.pipeline.step02_quality_control.models import LoadedModel, preprocessing_fn

logger = get_logger(__name__)

#: The tissue detector was trained on JPEG-compressed thumbnails, so the input
#: is round-tripped through JPEG at the same quality before inference. Skipping
#: this measurably degrades it - upstream flags it in a comment, and it is the
#: kind of detail that is invisible until the masks come back slightly wrong.
_TISSUE_JPEG_QUALITY = 80

#: Minimum tissue pixels in a patch's tissue-map footprint before the artefact
#: model is asked about it. Upstream's threshold, kept: below it the patch is
#: glass and gets written as background without a forward pass.
_MIN_TISSUE_PIXELS = 50

Progress = Callable[[str, int, int], None]


def _noop(phase: str, done: int, total: int) -> None:  # pragma: no cover - default
    return None


# --- shared helpers ----------------------------------------------------------


def _forward(loaded: LoadedModel, tiles: list[np.ndarray], preprocess: Any) -> np.ndarray:
    """Argmax class ids for a batch of HxWx3 uint8 tiles."""
    import torch

    batch = np.stack(
        [np.ascontiguousarray(preprocess(tile).transpose(2, 0, 1)) for tile in tiles]
    ).astype(np.float32)

    tensor = torch.from_numpy(batch).to(loaded.device)
    with torch.no_grad():
        logits = loaded.model(tensor)

    return logits.argmax(dim=1).cpu().numpy().astype(np.uint8)


def _read_patch(
    reader: Any, x0: int, y0: int, size_l0: int, out: int, *, from_level_0: bool
) -> Image.Image:
    """One square patch, resampled to `out` x `out`.

    `size_l0` is the patch's extent in level-0 pixels; `out` is what the model
    wants. Reading a coarser pyramid level and resampling the short remaining
    distance gives the model the same physical scale for a fraction of the I/O.
    """
    if from_level_0 or size_l0 <= out:
        image = reader.read_region_pil((x0, y0), 0, (size_l0, size_l0))
    else:
        level = reader.best_level_for_downsample(size_l0 / out)
        downsample = float(reader.level_downsamples[level])
        side = max(1, int(round(size_l0 / downsample)))
        image = reader.read_region_pil((x0, y0), level, (side, side))

    if image.size != (out, out):
        image = image.resize((out, out), Image.Resampling.LANCZOS)
    return image


def read_patch(reader: Any, *, x: int, y: int, size_l0: int, out: int) -> Image.Image:
    """Public form of the patch reader, for callers that just want the pixels."""
    return _read_patch(reader, x, y, size_l0, out, from_level_0=False)


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# --- pass 1: tissue detection ------------------------------------------------


@dataclass
class TissueResult:
    """The tissue map, and enough context to place it on the slide."""

    #: uint8, GrandQC coding: 0 = tissue, 1 = background.
    mask: np.ndarray
    #: Actual mpp of one mask pixel, derived from the thumbnail we got back
    #: rather than the one we asked for.
    mpp: float
    #: The thumbnail the mask was computed from, for the tissue overlay.
    thumbnail: np.ndarray
    tissue_fraction: float
    #: "grandqc" or "otsu" - the UI must not present the fallback as the model.
    source: str

    @property
    def size(self) -> tuple[int, int]:
        return self.mask.shape[1], self.mask.shape[0]


def _thumbnail_at_mpp(reader: Any, base_mpp: float, target_mpp: float) -> tuple[Image.Image, float]:
    """A whole-slide thumbnail at roughly `target_mpp`, plus the mpp achieved."""
    width, height = reader.dimensions
    scale = target_mpp / base_mpp
    longest = max(1, int(round(max(width, height) / scale)))

    image = reader.thumbnail_pil(longest)
    achieved = base_mpp * (width / max(1, image.width))
    return image.convert("RGB"), achieved


def detect_tissue(
    reader: Any,
    *,
    base_mpp: float,
    model: LoadedModel,
    target_mpp: float = 10.0,
    batch: int = 4,
    progress: Progress = _noop,
) -> TissueResult:
    """GrandQC pass 1 - separate tissue from empty glass at 10 um/px.

    Deliberately a segmentation model rather than the saturation-and-Otsu
    threshold of step 3: this has to survive pen marks, which are darker and
    far more saturated than any stain and which a threshold happily calls
    tissue. Step 3's classical mask is then applied to what survives here.
    """
    thumbnail, achieved_mpp = _thumbnail_at_mpp(reader, base_mpp, target_mpp)
    thumbnail = _jpeg_roundtrip(thumbnail, _TISSUE_JPEG_QUALITY)

    patch = 512
    width, height = thumbnail.size

    # Pad out to whole patches with white. White is glass, so the padding
    # predicts as background and never invents tissue at the margin; the mask
    # is cropped back to the real extent below.
    cols = max(1, math.ceil(width / patch))
    rows = max(1, math.ceil(height / patch))
    padded = Image.new("RGB", (cols * patch, rows * patch), (255, 255, 255))
    padded.paste(thumbnail, (0, 0))
    source = np.asarray(padded)

    preprocess = preprocessing_fn()
    mask = np.empty((rows * patch, cols * patch), dtype=np.uint8)

    positions = [(row, col) for row in range(rows) for col in range(cols)]
    total = len(positions)
    progress("tissue", 0, total)

    for start in range(0, total, batch):
        window = positions[start : start + batch]
        tiles = [
            source[row * patch : (row + 1) * patch, col * patch : (col + 1) * patch]
            for row, col in window
        ]
        predictions = _forward(model, tiles, preprocess)
        for (row, col), prediction in zip(window, predictions, strict=True):
            mask[row * patch : (row + 1) * patch, col * patch : (col + 1) * patch] = prediction
        progress("tissue", min(start + batch, total), total)

    mask = mask[:height, :width]
    tissue_fraction = float(np.count_nonzero(mask == 0) / mask.size)

    return TissueResult(
        mask=mask,
        mpp=achieved_mpp,
        thumbnail=np.asarray(thumbnail),
        tissue_fraction=tissue_fraction,
        source="grandqc",
    )


def detect_tissue_fallback(
    reader: Any, *, base_mpp: float, target_mpp: float = 10.0
) -> TissueResult:
    """A saturation-and-Otsu stand-in for pass 1, for when the checkpoint is absent.

    This exists so the classical half of step 2 can run before anyone has
    downloaded 100 MB of weights. It is *not* GrandQC, and it is not step 3
    either - it is step 3's method run *before* QC instead of after, which is
    the one ordering the pipeline forbids: with no artefact map to subtract
    first, it calls pen marks tissue, which is precisely the failure the real
    model is here to avoid. It is labelled `otsu` everywhere it surfaces so the
    UI can say so.
    """
    from scipy import ndimage

    thumbnail, achieved_mpp = _thumbnail_at_mpp(reader, base_mpp, target_mpp)

    quantised = saturation_channel(np.asarray(thumbnail))
    threshold = otsu_threshold(quantised)
    tissue = quantised > threshold

    structure = np.ones((5, 5), dtype=bool)
    tissue = ndimage.binary_closing(tissue, structure=structure)
    tissue = ndimage.binary_opening(tissue, structure=structure)

    mask = np.where(tissue, 0, 1).astype(np.uint8)
    return TissueResult(
        mask=mask,
        mpp=achieved_mpp,
        thumbnail=np.asarray(thumbnail),
        tissue_fraction=float(np.count_nonzero(mask == 0) / mask.size),
        source="otsu",
    )


# --- pass 2: artefact segmentation ------------------------------------------


@dataclass
class GridCell:
    """One measurement cell: what it was called, and how it measures.

    A cell is a *sub-block* of a model patch, not a whole one. A 512 px patch at
    1.5 um/px covers 768 um of tissue, and almost nothing is wrong with a whole
    768 um square - a pen line or a fold occupies a slice of it. Classing whole
    patches therefore labels nearly all of them "tissue" or "edge" and the
    explanation panel has no folds, no pen and no blur to say anything about.
    Measuring in sub-blocks is what makes the per-class comparison exist at all.
    """

    col: int
    row: int
    #: Pixel counts per class id within this cell.
    class_counts: dict[int, int]
    dominant: int
    tissue_share: float
    #: Classical metrics, or None for a cell with no tissue in it.
    features: dict[str, float] | None = None


@dataclass
class ArtefactResult:
    """The artefact map and the per-patch measurements taken alongside it.

    Two resolutions live here, and conflating them is the easy mistake:

      `class_totals` is exact. It is accumulated from every patch at the
      model's own resolution, so every percentage in the report is computed
      from full-detail pixel counts.

      `mask` is reduced, at `mask_mpp`. A whole-slide mask at 1.5 um/px is
      around 350 MB, and nothing displays more than a couple of thousand
      pixels across, so the mask is block-reduced as it is built. It is for
      drawing, never for counting.
    """

    #: uint8 at `mask_mpp`, GrandQC class coding, reduced for display.
    mask: np.ndarray
    mask_mpp: float
    #: Resolution the model actually ran at, which the counts are exact at.
    mpp: float
    patch_size: int
    #: The *patch* grid - one entry per model forward pass.
    grid_cols: int
    grid_rows: int
    #: One patch's extent in level-0 pixels.
    grid_extent_px: int
    #: The *measurement* grid, finer than the patch grid by `blocks_per_patch`.
    cell_cols: int = 0
    cell_rows: int = 0
    blocks_per_patch: int = 1
    #: Exact pixel counts per class id at `mpp`, summed over every patch.
    class_totals: dict[int, int] = field(default_factory=dict)
    #: Pixels of the slide the patch grid never covered - the right and bottom
    #: margins, each narrower than a single patch.
    unanalysed_pixels: int = 0
    cells: list[GridCell] = field(default_factory=list)
    patches_inferred: int = 0
    patches_skipped: int = 0
    model_classes: int = 0


def plan_grid(
    reader: Any, *, base_mpp: float, model_mpp: float, patch_size: int
) -> tuple[int, int, int]:
    """Patch extent in level-0 pixels, and the grid it makes. Cheap - no pixels read."""
    width, height = reader.dimensions
    extent = max(1, int(round(model_mpp / base_mpp * patch_size)))
    return extent, width // extent, height // extent


def _reduce_patch_mask(patch_mask: np.ndarray, cell: int) -> np.ndarray:
    """Block-reduce a patch's class mask to `cell` x `cell`, favouring artefacts.

    Plain nearest-neighbour decimation would erase a pen line one pixel wide,
    which is exactly the finding a viewer most wants to see. So each block
    resolves to the artefact class that covers most of it if any artefact
    covers more than a twentieth, and to the majority class otherwise.
    """
    size = patch_mask.shape[0]
    factor = size // cell
    if factor <= 1:
        return patch_mask

    blocks = patch_mask.reshape(cell, factor, cell, factor)
    block_area = factor * factor

    counts = {
        class_id: (blocks == class_id).sum(axis=(1, 3))
        for class_id in (*qc_classes.TISSUE_CLASS_IDS, qc_classes.BACKGROUND)
    }

    artefact_ids = [item.id for item in qc_classes.ARTEFACT_CLASSES]
    artefact_stack = np.stack([counts[class_id] for class_id in artefact_ids])
    best_artefact = artefact_stack.max(axis=0)
    winning_artefact = np.take(artefact_ids, artefact_stack.argmax(axis=0))

    all_ids = [*qc_classes.TISSUE_CLASS_IDS, qc_classes.BACKGROUND]
    full_stack = np.stack([counts[class_id] for class_id in all_ids])
    majority = np.take(all_ids, full_stack.argmax(axis=0))

    return np.where(best_artefact * 20 >= block_area, winning_artefact, majority).astype(np.uint8)


def _measure_blocks(
    rgb: np.ndarray,
    patch_mask: np.ndarray,
    *,
    col: int,
    row: int,
    blocks: int,
    block: int,
    collect_features: bool,
) -> list[GridCell]:
    """Split one patch into sub-blocks and measure each on its own.

    A block with no tissue in it gets no metrics rather than zeros: "nothing to
    measure here" and "measured, and it came out zero" are different statements,
    and the heatmap draws them differently.
    """
    out: list[GridCell] = []

    # One pass of every filter over the whole patch, reduced per block. Calling
    # tile_features per block instead costs about twelve times as much, almost
    # all of it fixed per-call overhead in scipy's filters.
    measured = block_features(rgb, blocks=blocks) if collect_features else None

    for by in range(blocks):
        for bx in range(blocks):
            top, left = by * block, bx * block
            sub_mask = patch_mask[top : top + block, left : left + block]

            ids, counts = np.unique(sub_mask, return_counts=True)
            class_counts = {int(i): int(c) for i, c in zip(ids, counts, strict=True)}
            on_tissue = sum(
                count
                for class_id, count in class_counts.items()
                if class_id in qc_classes.TISSUE_CLASS_IDS
            )

            # A block with no tissue gets no metrics rather than zeros: "nothing
            # to measure" and "measured zero" are different, and the heatmap
            # draws them differently.
            features = measured[by * blocks + bx] if measured is not None and on_tissue else None

            out.append(
                GridCell(
                    col=col * blocks + bx,
                    row=row * blocks + by,
                    class_counts=class_counts,
                    dominant=_dominant_class(class_counts),
                    tissue_share=on_tissue / (block * block),
                    features=features,
                )
            )

    return out


def segment_artefacts(
    reader: Any,
    *,
    base_mpp: float,
    tissue: TissueResult,
    model: LoadedModel,
    model_mpp: float,
    patch_size: int = 512,
    mask_cell: int = 64,
    feature_block: int = 128,
    from_level_0: bool = False,
    collect_features: bool = True,
    progress: Progress = _noop,
) -> ArtefactResult:
    """GrandQC pass 2 - a multi-class artefact map over the tissue found in pass 1.

    The tissue map decides which patches are worth a forward pass and, for the
    patches that get one, overrides the model wherever pass 1 already said
    background. That override is why the two models are run in this order: the
    artefact model can recognise background itself, but less reliably and 40x
    more slowly than the detector does at 10 um/px.
    """
    extent, cols, rows = plan_grid(
        reader, base_mpp=base_mpp, model_mpp=model_mpp, patch_size=patch_size
    )
    width, height = reader.dimensions

    # The tissue map is at 10 um/px; the artefact grid is at model_mpp. Each
    # patch's footprint is cropped straight out of the small map and scaled to
    # the patch, rather than resampling the whole map onto the working grid
    # first: at 10x that single intermediate would be 28,000 px square, which
    # is 800 MB of uint8 for something only ever read 512 px at a time.
    tissue_image = Image.fromarray(tissue.mask)
    tissue_scale = base_mpp / tissue.mpp
    tissue_w, tissue_h = tissue_image.size

    def tissue_footprint(col: int, row: int) -> np.ndarray:
        """Pass 1's verdict over one patch, at the patch's own resolution."""
        left = col * extent * tissue_scale
        top = row * extent * tissue_scale
        box = (
            min(left, tissue_w),
            min(top, tissue_h),
            min(left + extent * tissue_scale, tissue_w),
            min(top + extent * tissue_scale, tissue_h),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            # Wholly outside the tissue map: treat as background.
            return np.ones((patch_size, patch_size), dtype=np.uint8)
        return np.asarray(
            tissue_image.resize((patch_size, patch_size), Image.Resampling.NEAREST, box=box)
        )

    # Measurement cells per patch edge. A patch is one forward pass; a cell is
    # one row in the explanation table.
    blocks = max(1, patch_size // feature_block)

    preprocess = preprocessing_fn()
    mask = np.full((rows * mask_cell, cols * mask_cell), qc_classes.BACKGROUND, dtype=np.uint8)

    cells: list[GridCell] = []
    class_totals: dict[int, int] = {}
    inferred = 0
    skipped = 0
    total = cols * rows
    progress("artefacts", 0, total)

    for row in range(rows):
        for col in range(cols):
            footprint = tissue_footprint(col, row)
            tissue_pixels = int(np.count_nonzero(footprint == 0))

            if tissue_pixels <= _MIN_TISSUE_PIXELS:
                skipped += 1
                area = patch_size * patch_size
                class_totals[qc_classes.BACKGROUND] = (
                    class_totals.get(qc_classes.BACKGROUND, 0) + area
                )
                block_area = feature_block * feature_block
                cells.extend(
                    GridCell(
                        col=col * blocks + bx,
                        row=row * blocks + by,
                        class_counts={qc_classes.BACKGROUND: block_area},
                        dominant=qc_classes.BACKGROUND,
                        tissue_share=0.0,
                    )
                    for by in range(blocks)
                    for bx in range(blocks)
                )
                progress("artefacts", row * cols + col + 1, total)
                continue

            patch = _read_patch(
                reader,
                col * extent,
                row * extent,
                extent,
                patch_size,
                from_level_0=from_level_0,
            )
            rgb = np.asarray(patch)

            prediction = _forward(model, [rgb], preprocess)[0]
            # Pass 1 has the final say on background.
            patch_mask = np.where(footprint == 1, qc_classes.BACKGROUND, prediction).astype(
                np.uint8
            )

            mask[
                row * mask_cell : (row + 1) * mask_cell,
                col * mask_cell : (col + 1) * mask_cell,
            ] = _reduce_patch_mask(patch_mask, mask_cell)

            # Exact, full-resolution counts for the report come from the whole
            # patch; the per-cell breakdown below is only for the explanation.
            ids, counts = np.unique(patch_mask, return_counts=True)
            for class_id, count in zip(ids, counts, strict=True):
                class_totals[int(class_id)] = class_totals.get(int(class_id), 0) + int(count)

            cells.extend(
                _measure_blocks(
                    rgb,
                    patch_mask,
                    col=col,
                    row=row,
                    blocks=blocks,
                    block=feature_block,
                    collect_features=collect_features,
                )
            )

            inferred += 1
            progress("artefacts", row * cols + col + 1, total)

    # The right and bottom margins are narrower than one patch, so the grid
    # never reached them. Count them as unanalysed rather than background: they
    # are not known to be glass, they are simply outside the grid. Reported so
    # the shares add up to something a reader can check.
    slide_pixels_at_model_mpp = int(
        round(width * base_mpp / model_mpp) * round(height * base_mpp / model_mpp)
    )
    covered = cols * rows * patch_size * patch_size
    unanalysed = max(0, slide_pixels_at_model_mpp - covered)

    # Pad the display mask out to the slide's own proportions. Without this the
    # mask covers only the whole-patch grid - up to 2% narrower than the slide -
    # and every overlay drawn from it would sit slightly askew over a
    # full-extent thumbnail. The padding is UNANALYSED, not background: that
    # margin was never looked at, which is a different claim from "it is glass".
    mask_mpp = model_mpp * (patch_size / mask_cell)
    full_width = max(mask.shape[1], int(round(width * base_mpp / mask_mpp)))
    full_height = max(mask.shape[0], int(round(height * base_mpp / mask_mpp)))

    if full_height > mask.shape[0]:
        mask = np.concatenate(
            [
                mask,
                np.full(
                    (full_height - mask.shape[0], mask.shape[1]),
                    qc_classes.UNANALYSED,
                    np.uint8,
                ),
            ],
            axis=0,
        )
    if full_width > mask.shape[1]:
        mask = np.concatenate(
            [
                mask,
                np.full(
                    (mask.shape[0], full_width - mask.shape[1]),
                    qc_classes.UNANALYSED,
                    np.uint8,
                ),
            ],
            axis=1,
        )

    return ArtefactResult(
        mask=mask,
        mask_mpp=mask_mpp,
        mpp=model_mpp,
        patch_size=patch_size,
        grid_cols=cols,
        grid_rows=rows,
        grid_extent_px=extent,
        cell_cols=cols * blocks,
        cell_rows=rows * blocks,
        blocks_per_patch=blocks,
        class_totals=class_totals,
        unanalysed_pixels=unanalysed,
        cells=cells,
        patches_inferred=inferred,
        patches_skipped=skipped,
        model_classes=model.classes,
    )


def _dominant_class(class_counts: dict[int, int]) -> int:
    """The most common class on tissue, ignoring background and the margin.

    A patch that is 90% glass and 10% pen ink is a pen patch as far as the
    explanation panel is concerned - taking a plain mode would label almost
    every edge patch "background" and say nothing.
    """
    on_tissue = {
        class_id: count
        for class_id, count in class_counts.items()
        if class_id in qc_classes.TISSUE_CLASS_IDS
    }
    if on_tissue:
        return max(on_tissue, key=lambda key: on_tissue[key])
    return max(class_counts, key=lambda key: class_counts[key]) if class_counts else 0


# --- region inspection -------------------------------------------------------


@dataclass
class RegionInspection:
    """A close look at one region: its pixels, its classes, its metrics."""

    image: Image.Image
    mpp: float
    features: dict[str, float]
    class_counts: dict[int, int]
    dominant: int


def inspect_region(
    reader: Any,
    *,
    base_mpp: float,
    x: int,
    y: int,
    size_l0: int,
    target_mpp: float,
    mask: np.ndarray | None = None,
    mask_mpp: float | None = None,
) -> RegionInspection:
    """Measure one region at a resolution where the metrics actually mean something.

    The grid pass measures at the artefact model's 1.5 um/px, which is coarse
    enough that blur is partly averaged away. Focus is a claim about fine
    detail, so the explanation panel re-measures the single region a viewer
    clicked at the pipeline's own working resolution.
    """
    out = max(64, min(2048, int(round(size_l0 * base_mpp / target_mpp))))
    image = _read_patch(reader, x, y, size_l0, out, from_level_0=False)
    achieved_mpp = size_l0 * base_mpp / out

    features = tile_features(np.asarray(image), fast_texture=False).as_dict()

    class_counts: dict[int, int] = {}
    if mask is not None and mask_mpp:
        scale = base_mpp / mask_mpp
        x0 = int(x * scale)
        y0 = int(y * scale)
        x1 = min(mask.shape[1], x0 + max(1, int(size_l0 * scale)))
        y1 = min(mask.shape[0], y0 + max(1, int(size_l0 * scale)))
        if x1 > x0 and y1 > y0:
            window = mask[y0:y1, x0:x1]
            ids, counts = np.unique(window, return_counts=True)
            class_counts = {int(i): int(c) for i, c in zip(ids, counts, strict=True)}

    return RegionInspection(
        image=image,
        mpp=achieved_mpp,
        features=features,
        class_counts=class_counts,
        dominant=_dominant_class(class_counts) if class_counts else 0,
    )


# --- aggregation -------------------------------------------------------------


def feature_summary(cells: list[GridCell]) -> dict[str, dict[str, float]]:
    """Mean of each metric per dominant class, plus the clean-tissue baseline.

    Returns `{metric_key: {class_key: mean}}` with `tissue` present whenever any
    clean patch was measured. The ratios the UI draws are computed against that
    baseline - the slide's own clean tissue - never against a constant, because
    these metrics are not comparable between slides or between resolutions.
    """
    buckets: dict[int, list[dict[str, float]]] = {}
    for cell in cells:
        if cell.features is None:
            continue
        buckets.setdefault(cell.dominant, []).append(cell.features)

    summary: dict[str, dict[str, float]] = {key: {} for key in METRIC_KEYS}
    for class_id, rows in buckets.items():
        item = qc_classes.class_by_id(class_id)
        if item is None:
            continue
        for key in METRIC_KEYS:
            values = [row[key] for row in rows if key in row]
            if values:
                summary[key][item.key] = float(np.mean(values))
    return summary


def patch_counts_by_class(cells: list[GridCell]) -> dict[str, int]:
    """How many measured patches each class dominates, for the sample sizes."""
    counts: dict[str, int] = {}
    for cell in cells:
        if cell.features is None:
            continue
        item = qc_classes.class_by_id(cell.dominant)
        if item is not None:
            counts[item.key] = counts.get(item.key, 0) + 1
    return counts
