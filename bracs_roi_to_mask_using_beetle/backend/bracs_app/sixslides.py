"""The six clinical H&E slides, through both models, on the same pixels.

One row per case: the tissue, our trained tile classifier's answer, and BEETLE's. Both
read the *same* 2048 px window at the *same* 0.5 um/px, and both answers are expressed in
the *same* three classes, so the only thing that differs between the two right-hand
columns is the model.

--------------------------------------------------------------------------------
What this comparison can and cannot establish
--------------------------------------------------------------------------------

**Our model was trained on BEETLE's labels.** 1,843 of the 1,852 in-situ tiles in its
training set are BEETLE's predictions on BRACS regions. So where the two agree, that
mostly demonstrates that distillation worked - not that either is right. Agreement here
is close to guaranteed by construction and must never be quoted as accuracy.

Three things make it worth running anyway:

1. **These slides are a third domain.** Our own laboratory's scanner and staining, seen
   by neither model: BEETLE trained on its own cohort, our head on TCGA (BCSS) and an
   Italian cohort (BRACS). This is the first look at either on the data they would
   actually be deployed against.
2. **Disagreement is the signal.** Where the student diverges from the teacher on tissue
   neither has seen is precisely where the distillation failed to generalise, and it is
   localised - a picture, not a number.
3. **It is the evidence for dropping BEETLE.** A 46 M-parameter five-fold nnU-Net against
   a frozen ResNet18 and a linear head, roughly thirty times the compute, plus a
   CC BY-NC-SA licence on inference. Equivalence on our own slides is what would justify
   removing it.

--------------------------------------------------------------------------------
Why a region and not a slide
--------------------------------------------------------------------------------

Both models must run at 0.5 um/px - BEETLE's `dataset.json` spacing and the checkpoint's
`manifest["input"]["mpp"]`. These slides are 0.2222 um/px and up to 126,976 px square, so
a whole one at the working spacing is around 400 megapixels: roughly an hour per slide per
fold on this machine. Showing a low-magnification thumbnail instead would be worse than
slow, it would be wrong - both models would be running at a scale neither was fitted at,
and the comparison would measure that rather than the models.

So one window per slide, and the window is chosen by nuclear density: `regions.py`'s
sampler already grids each slide at 2 um/px, masks its tissue and scores every window by
the share of tissue pixels carrying haematoxylin. That score is a **stain measurement, not
a model output**, so it favours neither model, and epithelium is nuclear-dense so it lands
the window where the invasive/in-situ boundary actually lives rather than on a blank field
of stroma that both models would trivially agree about.

--------------------------------------------------------------------------------
Why agreement is scored per tile
--------------------------------------------------------------------------------

The two outputs are not the same kind of object. BEETLE labels every pixel; our model
labels a 224 px tile. Comparing them per pixel would charge our model for the coarseness
of its output format rather than for its judgement - it would lose every duct boundary by
construction, and the number would say more about tile size than about either network.

So BEETLE's mask is reduced to one label per 224 px tile by majority vote and the two are
compared label to label. The per-pixel figure is computed too and reported beside it,
labelled as the pessimistic one, because a reader is entitled to both.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import config, labels, pipeline, student, teacher

Progress = Callable[[str, float], None]


def _noop(message: str, fraction: float) -> None:
    return None


def cases() -> tuple[str, ...]:
    """The six case ids, from approach 4a rather than restated here."""
    config.install_beetle_path()
    import backend_path as beetle_paths

    return tuple(beetle_paths.CASES)


def slide_path(case: str) -> Path:
    """The H&E slide for a case, globbed rather than formatted.

    Five cases are `CAN_00251_26_H&E.svs`; `CAN_00865_26` is `CAN_00865_26-_H&E.svs`,
    with both separators. `beetle_paths.he_slide` already handles that and raises on
    zero or multiple matches, so it is used rather than reimplemented.
    """
    config.install_beetle_path()
    import backend_path as beetle_paths

    return beetle_paths.he_slide(case)


def slide_summary(case: str) -> dict:
    """Header-level facts about one slide, without reading a single tile of raster."""
    from app.ingestion.slide_reader import SlideReader

    path = slide_path(case)
    with SlideReader(path) as reader:
        width, height = reader.dimensions
        mpp = reader.mpp
    return {
        "case": case,
        "file": path.name,
        "width": int(width),
        "height": int(height),
        "mpp": mpp,
        "mm": [round(width * mpp / 1000, 1), round(height * mpp / 1000, 1)] if mpp else None,
    }


# --- picking the window --------------------------------------------------------


def pick_window(reader, case: str) -> dict:
    """The most nucleus-dense window on the slide, via approach 4a's own scorer.

    `score_candidates` grids the slide at 2 um/px, builds step 3's tissue mask, and scores
    each window by the share of its **tissue** pixels carrying haematoxylin - measured
    over tissue rather than over the window, so a field half off the section is not
    penalised for being half empty. Taking the maximum is the whole "densest nuclei"
    rule; everything else about the choice was already written and tested there.
    """
    import regions as beetle_regions

    candidates = beetle_regions.score_candidates(reader, case, min_tissue=0.35)
    if not candidates:
        raise ValueError(
            f"{case}: no window on this slide is at least 35 % tissue, so there is "
            "nothing worth comparing two models on."
        )
    best = max(candidates, key=lambda c: c.nuclear_density)
    return {
        "x": int(best.x),
        "y": int(best.y),
        "span_l0": int(best.span_l0),
        "tissue_share": round(float(best.tissue_share), 4),
        "nuclear_density": round(float(best.nuclear_density), 4),
        "candidates_scored": len(candidates),
    }


def read_window(reader, window: dict, size_px: int) -> np.ndarray:
    """The chosen window as uint8 RGB at exactly `config.TEACHER_MPP`.

    Read at whichever pyramid level is nearest and then area-averaged down with
    `Image.Resampling.BOX`. These slides are 0.2222 um/px with no level near 0.5, so in
    practice this is always a level-0 read followed by a 2.25x downsample - and the
    averaging has to happen on intensities, before any logarithm, for the reason step 5
    and `export.resample` both give: averaging densities takes the mean of logarithms,
    which is a number no instrument records.
    """
    level = reader.best_level_for_mpp(config.TEACHER_MPP)
    downsample = float(reader.level_downsamples[level])
    read_px = max(1, int(round(window["span_l0"] / downsample)))

    image = reader.read_region_pil((window["x"], window["y"]), level, (read_px, read_px))
    if image.size != (size_px, size_px):
        image = image.resize((size_px, size_px), Image.Resampling.BOX)
    return np.asarray(image.convert("RGB"))


# --- comparing -----------------------------------------------------------------


def tile_majority(pixel_labels: np.ndarray, tile_px: int) -> np.ndarray:
    """BEETLE's per-pixel mask reduced to one label per tile, by majority.

    Plain majority over labelled pixels, not `export.vote`'s asymmetric thresholds. Those
    thresholds exist to decide whether a tile is *good enough to train on*; here the
    question is only "what does the teacher mostly say here", and importing a training
    rule into a scoring rule would silently discard the mixed tiles that are exactly where
    the two models are most likely to differ.
    """
    import bcss

    rows = pixel_labels.shape[0] // tile_px
    cols = pixel_labels.shape[1] // tile_px
    out = np.zeros((rows, cols), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            block = pixel_labels[
                row * tile_px : (row + 1) * tile_px,
                col * tile_px : (col + 1) * tile_px,
            ]
            labelled = block[block != bcss.IGNORE]
            if labelled.size == 0:
                out[row, col] = bcss.NON_EPITHELIUM
                continue
            out[row, col] = np.bincount(labelled, minlength=3).argmax()
    return out


def agreement(ours: np.ndarray, theirs: np.ndarray) -> dict:
    """How often the student and the teacher say the same thing, and where they do not."""
    import bcss

    if ours.shape != theirs.shape:
        raise ValueError(f"grids disagree: {ours.shape} against {theirs.shape}")

    n = int(ours.size)
    matrix = np.zeros((3, 3), dtype=int)
    for teacher_cls in range(3):
        for student_cls in range(3):
            matrix[teacher_cls, student_cls] = int(
                ((theirs == teacher_cls) & (ours == student_cls)).sum()
            )

    per_class = {}
    for index, name in enumerate(bcss.CLASS_NAMES):
        both = int(((ours == index) & (theirs == index)).sum())
        either = int(((ours == index) | (theirs == index)).sum())
        per_class[name] = {
            "iou": round(both / either, 4) if either else None,
            "teacher_tiles": int((theirs == index).sum()),
            "student_tiles": int((ours == index).sum()),
        }

    return {
        "tiles": n,
        "tile_agreement": round(float((ours == theirs).mean()), 4),
        "confusion": matrix.tolist(),
        "confusion_axes": {
            "rows": "BEETLE (teacher)",
            "cols": "our model (student)",
            "order": list(bcss.CLASS_NAMES),
        },
        "per_class": per_class,
        # The pair the whole project turns on, named rather than left to be read off the
        # matrix by a reader who may not know which axis is which.
        "teacher_in_situ_student_invasive": int(
            matrix[bcss.NON_INVASIVE, bcss.INVASIVE]
        ),
        "teacher_invasive_student_in_situ": int(
            matrix[bcss.INVASIVE, bcss.NON_INVASIVE]
        ),
    }


def disagreement_picture(rgb: np.ndarray, ours: np.ndarray, theirs: np.ndarray) -> np.ndarray:
    """The tissue, with the tiles the two models label differently marked out.

    Deliberately not a third class map. The question this picture answers is "where do
    they differ", and a reader tracing two class maps against each other by eye will miss
    scattered single tiles - which is the pattern a domain shift usually produces.
    """
    painted = rgb.astype(np.float32).copy()
    differs = ours != theirs
    if differs.any():
        red = np.array([216, 27, 96], dtype=np.float32)
        painted[differs] = 0.45 * painted[differs] + 0.55 * red
    return painted.clip(0, 255).astype(np.uint8)


# --- one case ------------------------------------------------------------------


def compare_slide(
    case: str,
    *,
    folds: tuple[int, ...] = (0,),
    size_px: int | None = None,
    model_name: str | None = None,
    progress: Progress = _noop,
) -> dict:
    """Run both models on one slide's densest window and write the row's three pictures."""
    import bcss
    from app.ingestion.slide_reader import SlideReader

    config.install_beetle_path()
    size_px = int(size_px or config.COMPARE_REGION_PX)
    out_dir = config.SIXSLIDES_DIR / case
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    path = slide_path(case)
    progress("opening the slide and finding the tissue", 0.02)

    with SlideReader(path) as reader:
        if reader.mpp is None:
            # The standing rule in this codebase: exclude and say why, never guess. The
            # cohort spans 0.16-0.50 um/px, so a guess can be twice wrong and every tile
            # would land at the wrong physical scale without anything raising.
            raise ValueError(
                f"{path.name} records no microns-per-pixel, so neither model can be run "
                "at its trained scale on it."
            )
        base_mpp = float(reader.mpp)
        progress("scoring windows by nuclear density", 0.06)
        window = pick_window(reader, case)
        progress("reading the chosen window at 0.5 um/px", 0.18)
        rgb = read_window(reader, window, size_px)

    # Our model first: it is seconds, and if its transform is wrong there is no point
    # spending a minute of nnU-Net on the same pixels.
    progress("our model", 0.22)
    net, manifest = student.load_student(model_name)
    ours = student.classify_region(
        rgb, net, manifest,
        progress=lambda message, fraction: progress(message, 0.22 + 0.08 * fraction),
    )
    rows, cols = ours["grid"]
    tile_px = ours["tile_px"]
    covered_h, covered_w = ours["covered"]

    # Everything downstream is cropped to the tiles our model actually covered. A 2048 px
    # region holds 9x9 tiles of 224 px = 2016 px, and the 32 px margin is not classified -
    # so comparing over the full region would score BEETLE against nothing on that strip.
    rgb_cropped = rgb[:covered_h, :covered_w]

    progress("BEETLE", 0.32)
    archive = teacher.read_archive()
    prediction = teacher.predict(
        rgb_cropped, archive, folds=folds,
        progress=lambda message, fraction: progress(message, 0.32 + 0.55 * fraction),
    )
    beetle_classes = labels.to_our_classes(prediction.mask)

    progress("comparing", 0.90)
    beetle_tiles = tile_majority(beetle_classes, tile_px)
    scores = agreement(ours["tile_labels"], beetle_tiles)
    # The pessimistic number: our blocky map against BEETLE's pixels, which charges us for
    # the tile size. Reported beside the tile figure, never instead of it.
    ours_pixels = ours["pixel_labels"][:covered_h, :covered_w]
    scores["pixel_agreement"] = round(
        float((ours_pixels == beetle_classes).mean()), 4
    )

    progress("drawing", 0.94)
    pipeline._downscale_for_web(rgb_cropped).save(out_dir / "original.png", optimize=True)
    pipeline._downscale_for_web(
        pipeline.overlay(rgb_cropped, ours_pixels)
    ).save(out_dir / "ours.png", optimize=True)
    pipeline._downscale_for_web(
        pipeline.overlay(rgb_cropped, beetle_classes)
    ).save(out_dir / "beetle.png", optimize=True)
    pipeline._downscale_for_web(
        disagreement_picture(
            rgb_cropped,
            np.repeat(np.repeat(ours["tile_labels"], tile_px, 0), tile_px, 1),
            np.repeat(np.repeat(beetle_tiles, tile_px, 0), tile_px, 1),
        )
    ).save(out_dir / "disagreement.png", optimize=True)

    result = {
        "case": case,
        "file": path.name,
        "slide_mpp": base_mpp,
        "working_mpp": config.TEACHER_MPP,
        "window": window,
        "region_px": size_px,
        "region_um": round(size_px * config.TEACHER_MPP, 1),
        "compared_px": [int(covered_w), int(covered_h)],
        "grid": [int(rows), int(cols)],
        "tile_px": tile_px,
        "student": {
            "model": model_name or student.default_model(),
            "input": ours["input"],
            "class_area_fraction": ours["class_area_fraction"],
            "mean_confidence": round(ours["mean_confidence"], 4),
        },
        "teacher": {
            "folds": list(prediction.folds),
            "mean_confidence": round(float(prediction.confidence.mean()), 4),
            "class_area_fraction": {
                name: round(float((beetle_tiles == index).mean()), 4)
                for index, name in enumerate(bcss.CLASS_NAMES)
            },
        },
        "agreement": scores,
        "seconds": round(time.time() - started, 1),
        "caveat": (
            "Our model was trained on BEETLE's labels, so agreement here largely shows "
            "that distillation worked rather than that either model is correct. What is "
            "genuinely new is the tissue: neither model has seen this laboratory's "
            "slides."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress("done", 1.0)
    return result


def load_result(case: str) -> dict | None:
    path = config.SIXSLIDES_DIR / case / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def summarise(results: list[dict]) -> dict:
    """The six rows in one line, weighted by tiles rather than by slide.

    A per-slide mean would give a 3x3 window the same weight as a 9x9 one, which is a
    quiet way of letting the smallest field decide the headline.
    """
    if not results:
        return {"cases": 0}

    tiles = sum(r["agreement"]["tiles"] for r in results)
    agreed = sum(
        r["agreement"]["tile_agreement"] * r["agreement"]["tiles"] for r in results
    )
    return {
        "cases": len(results),
        "tiles": tiles,
        "tile_agreement": round(agreed / tiles, 4) if tiles else None,
        "teacher_in_situ_student_invasive": sum(
            r["agreement"]["teacher_in_situ_student_invasive"] for r in results
        ),
        "teacher_invasive_student_in_situ": sum(
            r["agreement"]["teacher_invasive_student_in_situ"] for r in results
        ),
        "seconds": round(sum(r["seconds"] for r in results), 1),
    }
