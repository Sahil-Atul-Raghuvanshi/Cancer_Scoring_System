"""One BRACS test slide, whole tissue, both models, against pathologist annotations.

This is the only place in the project where a model is scored against a **person**. BCSS
supplies nine human-drawn in-situ tiles; BRACS's `.qpdata` files supply boxes a
pathologist drew round lesions on a whole slide neither model has seen. So this is the
first honest answer to "is the trained model right", as opposed to "does it agree with
the model that taught it".

--------------------------------------------------------------------------------
The rule that decides whether the score means anything
--------------------------------------------------------------------------------

**Tissue outside an annotation box is UNLABELLED, not "other tissue".**

BRACS annotates lesions only. A normal duct, a benign lobule, an unremarkable stretch of
epithelium sitting outside any box is real epithelium that nobody drew - and scoring it as
class 0 would punish both models for correctly finding it. That would not be a strict
evaluation, it would be a wrong one, and it would flatter whichever model finds *less*
epithelium.

So the truth grid carries `bcss.IGNORE` everywhere outside a box, exactly as approach 1
treats BCSS's `outside_roi`, and every number below is computed on annotated tiles only.
The whole tissue is still segmented and displayed - that is what the pictures are for -
but it is not scored.

--------------------------------------------------------------------------------
Why it streams
--------------------------------------------------------------------------------

The slide is 89,640 x 81,211 at 0.2519 um/px, which is 45,163 x 40,914 at the models'
0.5 um/px - 1,848 megapixels. As one RGB array that is 5.5 GB, and BEETLE's five-class
float32 accumulator over it would be 37 GB. Neither fits, so the slide is walked in
blocks and only three small things are kept: one label per 224 px tile from each model,
plus the truth. That is a 201 x 182 grid - 37,000 entries - rather than 1.8 gigapixels.

Blocks carry a **halo** that is read, segmented and then discarded. A U-Net is worst at
the edge of its receptive field, so a block segmented to its own border would show a seam
at every block boundary - and those seams would fall across ducts, which is exactly what
is being measured.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import config, labels, qpdata, student, teacher

Progress = Callable[[str, float], None]


def _noop(message: str, fraction: float) -> None:
    return None


#: BRACS PathClass -> our three classes.
#:
#: `Malignant-sure` is invasive carcinoma and `DCIS-sure` is in-situ, which are the two
#: the project turns on. **`UDH-sure` is class 1 as well** and that is a judgement worth
#: stating: usual ductal hyperplasia is a benign proliferation *inside* a duct, so it is
#: non-invasive epithelium by the same definition that puts DCIS there - BEETLE's own
#: class covers "healthy glands and DCIS ... LCIS, atypical ductal hyperplasia, apocrine
#: metaplasia", and approach 1 maps BCSS's `normal_acinus_or_duct` to class 1 too. Calling
#: it anything else would mean the two models are marked wrong for finding epithelium that
#: is really there.
#:
#: The `-non` suffixes BRACS uses for uncertain calls are deliberately absent: an
#: annotation the pathologist was not sure of is not ground truth, and `truth_grid`
#: leaves those tiles unlabelled rather than scoring against a maybe.
BRACS_TO_OURS: dict[str, int] = {
    "Malignant-sure": 2,   # bcss.INVASIVE
    "DCIS-sure": 1,        # bcss.NON_INVASIVE
    "UDH-sure": 1,         # bcss.NON_INVASIVE - see above
}

#: Tiles per block edge. 16 x 224 = 3584 px of core, plus the halo below. Bigger blocks
#: waste less on halo but hold a larger float32 accumulator; at 4096 px square that peaks
#: around 340 MB, which is affordable beside a 46 M-parameter network.
BLOCK_TILES = 16

#: Halo in working pixels, discarded after segmentation. Half of BEETLE's 512 px patch.
HALO_PX = 256

#: BEETLE's sliding-window step on this path, as a fraction of its 512 px patch.
#:
#: **1.0, not the 0.5 it uses on a single region, and the cost is quadratic.** At 0.5 a
#: 4096 px block needs 256 forward passes; at 1.0 it needs 64. Over a whole slide that is
#: an hour and a quarter against twenty minutes, and it buys nothing here: the output is
#: reduced to one label per 224 px tile before anything is scored, and displayed at a ~20x
#: downscale, so a seam a few pixels wide is below the resolution of every number and
#: every picture this module produces. The halo above already removes the seams that would
#: have mattered - the ones at block boundaries, which fall across whole ducts.
WSI_TILE_STEP = 1.0


@dataclass
class Grids:
    """Three aligned label grids, one entry per 224 px tile of the whole slide."""

    ours: np.ndarray
    beetle: np.ndarray
    truth: np.ndarray
    tissue: np.ndarray
    tile_px: int
    origin: tuple[int, int]     # level-0 pixel of tile (0, 0)
    working_mpp: float


def _block_has_tissue(
    coarse: np.ndarray, mask_mpp: float, base_mpp: float,
    lx: int, ly: int, lw: int, lh: int,
) -> bool:
    """Does this level-0 rectangle overlap any tissue in the coarse mask?

    Deliberately generous: one tissue pixel in the coarse mask is enough. The mask is at
    ~2.8 um/px, so one of its pixels is 11x11 working pixels, and a block that clips the
    very edge of a section must still be segmented. This skips glass, not tissue.
    """
    step = mask_mpp / base_mpp
    r0 = max(0, int(ly / step))
    c0 = max(0, int(lx / step))
    r1 = min(coarse.shape[0], int((ly + lh) / step) + 1)
    c1 = min(coarse.shape[1], int((lx + lw) / step) + 1)
    if r1 <= r0 or c1 <= c0:
        return False
    return bool(coarse[r0:r1, c0:c1].any())


def tissue_bounds(reader) -> tuple[int, int, int, int, float, np.ndarray]:
    """The tissue bounding box in level-0 pixels, via step 3's mask.

    Reused from approach 4a rather than rewritten: `tissue_at_score_mpp` builds the demo
    backend's own saturation-threshold mask and deliberately excludes the non-commercial
    GrandQC model, which is the right call here for the same reason it was there.
    """
    import regions as beetle_regions

    mask, mask_mpp = beetle_regions.tissue_at_score_mpp(reader)
    rows, cols = np.where(mask)
    if not rows.size:
        raise ValueError("this slide has no tissue at all by step 3's mask")

    scale = mask_mpp / reader.mpp
    return (
        int(cols.min() * scale), int(rows.min() * scale),
        int((cols.max() + 1) * scale), int((rows.max() + 1) * scale),
        mask_mpp, mask,
    )


def truth_grid(
    annotations: list[qpdata.Annotation],
    grid_shape: tuple[int, int],
    origin: tuple[int, int],
    tile_px: int,
    scale: float,
) -> np.ndarray:
    """One truth label per tile, `IGNORE` everywhere no box was drawn.

    A tile counts as annotated only if its **centre** falls inside a box. Anything looser
    would label the ring of tiles around every ROI with a class the pathologist drew a
    boundary specifically to exclude, and those edge tiles are where the two models
    disagree most - so a generous rule here would quietly decide the result.
    """
    import bcss

    rows, cols = grid_shape
    grid = np.full((rows, cols), bcss.IGNORE, dtype=np.uint8)

    # Tile centres in level-0 coordinates.
    centre_y = origin[1] + (np.arange(rows) + 0.5) * tile_px * scale
    centre_x = origin[0] + (np.arange(cols) + 0.5) * tile_px * scale

    for annotation in annotations:
        label = BRACS_TO_OURS.get(annotation.path_class)
        if label is None:
            continue  # an uncertain or unmapped class: left unlabelled, never guessed
        inside_y = (centre_y >= annotation.y) & (centre_y < annotation.y2)
        inside_x = (centre_x >= annotation.x) & (centre_x < annotation.x2)
        if not inside_y.any() or not inside_x.any():
            continue
        block = np.ix_(inside_y, inside_x)
        # Later boxes win where two overlap. BRACS's boxes are drawn per lesion and do
        # overlap slightly; invasive takes precedence because a field containing both is
        # invasive for the purposes of the score (guide Rule 5).
        current = grid[block]
        grid[block] = np.where(
            (current == bcss.IGNORE) | (np.uint8(label) == 2), np.uint8(label), current
        )
    return grid


def run_slide(
    svs: Path,
    annotation_file: Path,
    out_dir: Path,
    *,
    folds: tuple[int, ...] = (0,),
    model_name: str | None = None,
    block_tiles: int = BLOCK_TILES,
    progress: Progress = _noop,
) -> dict:
    """Segment the whole tissue with both models and score both inside the annotations."""
    import bcss
    from app.ingestion.slide_reader import SlideReader

    config.install_beetle_path()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    annotations = qpdata.parse(annotation_file)
    net, manifest = student.load_student(model_name)
    settings = student.input_settings(manifest)
    tile_px = settings["tile_px"]
    working_mpp = settings["mpp"]
    archive = teacher.read_archive()

    with SlideReader(svs) as reader:
        base_mpp = reader.mpp
        if base_mpp is None:
            raise ValueError(f"{svs.name} records no microns-per-pixel")
        scale = working_mpp / base_mpp          # level-0 px per working px
        progress("finding the tissue", 0.01)
        x0, y0, x1, y1, mask_mpp, coarse = tissue_bounds(reader)

        width_w = int((x1 - x0) / scale)
        height_w = int((y1 - y0) / scale)
        cols = width_w // tile_px
        rows = height_w // tile_px
        if rows == 0 or cols == 0:
            raise ValueError("the tissue is smaller than one tile at the working scale")

        ours = np.full((rows, cols), bcss.IGNORE, np.uint8)
        beetle = np.full((rows, cols), bcss.IGNORE, np.uint8)
        tissue = np.zeros((rows, cols), bool)

        block_rows = (rows + block_tiles - 1) // block_tiles
        block_cols = (cols + block_tiles - 1) // block_tiles
        total_blocks = block_rows * block_cols
        done = 0

        for br in range(block_rows):
            for bc in range(block_cols):
                done += 1
                r0, c0 = br * block_tiles, bc * block_tiles
                r1, c1 = min(rows, r0 + block_tiles), min(cols, c0 + block_tiles)
                core_h, core_w = (r1 - r0) * tile_px, (c1 - c0) * tile_px

                # Ask the low-resolution mask FIRST. Reading an 8192 px region off the
                # slide and area-averaging it costs ~2 s whether or not there is anything
                # in it, and on a 22 mm slide three quarters of the blocks are glass. The
                # mask was already built before this loop; consulting it is free.
                if not _block_has_tissue(
                    coarse, mask_mpp, base_mpp,
                    x0 + int(c0 * tile_px * scale), y0 + int(r0 * tile_px * scale),
                    int(core_w * scale), int(core_h * scale),
                ):
                    progress(f"block {done}/{total_blocks} (glass)", done / total_blocks)
                    continue

                # Read core + halo at level 0, then area-average to the working spacing.
                lx = x0 + int(c0 * tile_px * scale) - int(HALO_PX * scale)
                ly = y0 + int(r0 * tile_px * scale) - int(HALO_PX * scale)
                read_w = int((core_w + 2 * HALO_PX) * scale)
                read_h = int((core_h + 2 * HALO_PX) * scale)
                lx, ly = max(0, lx), max(0, ly)

                image = reader.read_region_pil((lx, ly), 0, (read_w, read_h))
                block = np.asarray(
                    image.resize(
                        (core_w + 2 * HALO_PX, core_h + 2 * HALO_PX),
                        Image.Resampling.BOX,
                    ).convert("RGB")
                )

                # Skip blocks with no tissue: on a 22 mm slide most blocks are glass, and
                # segmenting glass costs the same as segmenting tumour.
                core = block[HALO_PX : HALO_PX + core_h, HALO_PX : HALO_PX + core_w]
                if core.size == 0:
                    continue
                stained = (core.astype(np.int16).max(2) - core.astype(np.int16).min(2)) > 18
                covered = stained.reshape(
                    r1 - r0, tile_px, c1 - c0, tile_px
                ).mean(axis=(1, 3))
                tissue[r0:r1, c0:c1] = covered > 0.10
                if not tissue[r0:r1, c0:c1].any():
                    progress(f"block {done}/{total_blocks} (no tissue)", done / total_blocks)
                    continue

                result = student.classify_region(core, net, manifest)
                ours[r0:r1, c0:c1] = result["tile_labels"]

                prediction = teacher.predict(
                    block, archive, folds=folds, tile_step=WSI_TILE_STEP
                )
                mapped = labels.to_our_classes(prediction.mask)
                mapped_core = mapped[HALO_PX : HALO_PX + core_h, HALO_PX : HALO_PX + core_w]
                beetle[r0:r1, c0:c1] = _majority(mapped_core, tile_px)

                progress(
                    f"block {done} of {total_blocks}",
                    0.02 + 0.93 * done / total_blocks,
                )

        progress("reading the annotations", 0.96)
        truth = truth_grid(annotations, (rows, cols), (x0, y0), tile_px, scale)

        progress("drawing", 0.97)
        thumb = np.asarray(
            reader.read_region_pil((x0, y0), reader.best_level_for_downsample(
                max(1.0, (x1 - x0) / 2000)
            ), (
                max(1, int((x1 - x0) / max(1.0, reader.level_downsamples[
                    reader.best_level_for_downsample(max(1.0, (x1 - x0) / 2000))]))),
                max(1, int((y1 - y0) / max(1.0, reader.level_downsamples[
                    reader.best_level_for_downsample(max(1.0, (x1 - x0) / 2000))]))),
            )).convert("RGB")
        )

    grids = Grids(ours, beetle, truth, tissue, tile_px, (x0, y0), working_mpp)
    _write_pictures(out_dir, thumb, grids)

    result = {
        "slide": svs.name,
        "annotations_file": annotation_file.name,
        "slide_size": [int(x1 - x0), int(y1 - y0)],
        "slide_mpp": base_mpp,
        "working_mpp": working_mpp,
        "tile_px": tile_px,
        "grid": [int(rows), int(cols)],
        "tiles_total": int(rows * cols),
        "tiles_tissue": int(tissue.sum()),
        "tissue_mm2": round(float(tissue.sum()) * (tile_px * working_mpp / 1000) ** 2, 2),
        "annotations": qpdata.summarise(annotations, base_mpp),
        "bracs_to_ours": BRACS_TO_OURS,
        "student": {"model": model_name or student.default_model(), "input": settings},
        "teacher": {"folds": list(folds), "tile_step": WSI_TILE_STEP},
        "scored": score(grids),
        "seconds": round(time.time() - started, 1),
        "unlabelled_rule": (
            "Tissue outside a BRACS box is UNLABELLED, not 'other tissue'. BRACS "
            "annotates lesions only, so unboxed epithelium is real epithelium nobody "
            "drew; scoring it as class 0 would punish both models for finding it."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "grids.npz", ours=ours, beetle=beetle, truth=truth, tissue=tissue
    )
    progress("done", 1.0)
    return result


def _majority(pixel_labels: np.ndarray, tile_px: int) -> np.ndarray:
    """BEETLE's pixels to one label per tile. Vectorised; `IGNORE` never wins."""
    import bcss

    rows = pixel_labels.shape[0] // tile_px
    cols = pixel_labels.shape[1] // tile_px
    blocks = pixel_labels[: rows * tile_px, : cols * tile_px].reshape(
        rows, tile_px, cols, tile_px
    )
    counts = np.stack(
        [(blocks == cls).sum(axis=(1, 3)) for cls in range(3)], axis=-1
    )
    out = counts.argmax(axis=-1).astype(np.uint8)
    out[counts.sum(axis=-1) == 0] = bcss.NON_EPITHELIUM
    return out


def score(grids: Grids) -> dict:
    """Both models against the pathologist, on annotated tiles only."""
    import bcss

    annotated = grids.truth != bcss.IGNORE
    out: dict = {
        "annotated_tiles": int(annotated.sum()),
        "truth_tiles": {
            name: int((grids.truth == index).sum())
            for index, name in enumerate(bcss.CLASS_NAMES)
        },
    }
    if not annotated.any():
        out["note"] = "no tile centre falls inside any annotation box"
        return out

    for who, grid in (("ours", grids.ours), ("beetle", grids.beetle)):
        predicted = grid[annotated]
        actual = grids.truth[annotated]
        matrix = np.zeros((3, 3), int)
        for t in range(3):
            for p in range(3):
                matrix[t, p] = int(((actual == t) & (predicted == p)).sum())
        rows_sum = matrix.sum(axis=1)
        out[who] = {
            "accuracy": round(float((predicted == actual).mean()), 4),
            "confusion": matrix.tolist(),
            "confusion_axes": {"rows": "pathologist", "cols": who,
                               "order": list(bcss.CLASS_NAMES)},
            "recall": {
                name: (round(float(matrix[i, i] / rows_sum[i]), 4) if rows_sum[i] else None)
                for i, name in enumerate(bcss.CLASS_NAMES)
            },
            "in_situ_called_invasive": int(matrix[bcss.NON_INVASIVE, bcss.INVASIVE]),
            "invasive_called_in_situ": int(matrix[bcss.INVASIVE, bcss.NON_INVASIVE]),
        }
    return out


def _write_pictures(out_dir: Path, thumb: np.ndarray, grids: Grids) -> None:
    """The four columns, all at the thumbnail's size so they overlay pixel for pixel."""
    import bcss

    height, width = thumb.shape[:2]
    Image.fromarray(thumb).save(out_dir / "original.png", optimize=True)

    for name, grid in (("ours", grids.ours), ("beetle", grids.beetle), ("truth", grids.truth)):
        painted = np.zeros((*grid.shape, 3), np.uint8)
        painted[:] = labels.IGNORE_COLOUR
        for cls, colour in labels.CLASS_COLOURS.items():
            painted[grid == cls] = colour

        picture = Image.fromarray(painted).resize((width, height), Image.Resampling.NEAREST)
        base = Image.fromarray(thumb).convert("RGB")
        blended = Image.blend(base, picture, 0.55)

        if name == "truth":
            # Unannotated tissue keeps the tissue unmodified: `IGNORE` is the absence of
            # a statement, and tinting it would imply the pathologist said something.
            unlabelled = np.asarray(
                Image.fromarray((grid == bcss.IGNORE).astype(np.uint8) * 255)
                .resize((width, height), Image.Resampling.NEAREST)
            ) > 127
            out = np.asarray(blended).copy()
            out[unlabelled] = thumb[unlabelled]
            blended = Image.fromarray(out)

        blended.save(out_dir / f"{name}.png", optimize=True)
