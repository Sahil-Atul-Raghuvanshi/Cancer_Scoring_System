"""Per-pixel metrics, and the boundary metric that is the point of this whole pipeline.

Area Dice is the number everyone reports and it is not sufficient here. A model can score a
high Dice on a large duct while placing its edge ten pixels out, and "the edge is in the
right place" is the entire reason for building a pixel model rather than keeping the tile
one. So `boundary_f1` is computed alongside, on a narrow band around the true boundary, and
both are reported.

**`IGNORE` is excluded everywhere, and that is not a detail.** `confusion` builds a 3x3
matrix; a truth value of 255 indexed into it is an `IndexError` if you are lucky and a
silent wrap if you are not. More importantly, unlabelled pixels are 3.2 % of BCSS and
everything outside a BRACS annotation - counting them as class 0 would mark the model wrong
for finding epithelium nobody drew, and would flatter whichever model finds less.

**The two label sources are never averaged.** BCSS's masks were drawn by pathologists;
the BRACS masks are BEETLE's predictions, reviewed by nobody. A single class-1 Dice over
both would be dominated by the teacher-labelled majority and would be reporting agreement
with a model as though it were accuracy. `by_source` keeps them apart, and the number that
leaves this pipeline is the `bcss` one.
"""

from __future__ import annotations

import numpy as np

import classes


def confusion(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Rows are truth, columns are prediction, over labelled pixels only."""
    truth = np.asarray(truth).ravel()
    predicted = np.asarray(predicted).ravel()
    if truth.shape != predicted.shape:
        raise ValueError(f"shapes disagree: {truth.shape} against {predicted.shape}")

    keep = truth != classes.IGNORE
    truth, predicted = truth[keep].astype(np.int64), predicted[keep].astype(np.int64)

    n = len(classes.CLASS_NAMES)
    if truth.size and (truth.max() >= n or predicted.max() >= n):
        raise ValueError(
            f"labels outside 0..{n - 1} reached the matrix: truth max {truth.max()}, "
            f"predicted max {predicted.max()}. IGNORE should already be gone."
        )
    matrix = np.zeros((n, n), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    return matrix


def dice(matrix: np.ndarray, cls: int) -> float:
    """Dice / F1 for one class against all others: `2TP / (2TP + FP + FN)`."""
    tp = float(matrix[cls, cls])
    fp = float(matrix[:, cls].sum() - tp)
    fn = float(matrix[cls, :].sum() - tp)
    denominator = 2.0 * tp + fp + fn
    return 2.0 * tp / denominator if denominator > 0 else float("nan")


def iou(matrix: np.ndarray, cls: int) -> float:
    tp = float(matrix[cls, cls])
    union = float(matrix[:, cls].sum() + matrix[cls, :].sum() - tp)
    return tp / union if union > 0 else float("nan")


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Pixels of a binary mask that touch a different value - its outline, one pixel wide.

    Written with shifts rather than `scipy.ndimage` so this module has no dependency
    beyond numpy, and so the definition is visible: a pixel is on the boundary if any
    4-neighbour differs from it.
    """
    mask = mask.astype(bool)
    edge = np.zeros_like(mask)
    edge[:-1, :] |= mask[:-1, :] != mask[1:, :]
    edge[1:, :] |= mask[:-1, :] != mask[1:, :]
    edge[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    edge[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return edge & mask


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation by `radius`, by repeated shifts. Cheap and dependency-free."""
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[:-1, :] |= out[1:, :]
        grown[1:, :] |= out[:-1, :]
        grown[:, :-1] |= out[:, 1:]
        grown[:, 1:] |= out[:, :-1]
        out = grown
    return out


def boundary_f1(
    truth: np.ndarray, predicted: np.ndarray, cls: int, *, tolerance: int = 3
) -> float:
    """How well the predicted outline of one class sits on the true outline.

    The standard trimap measure. A predicted boundary pixel counts as correct if a true
    boundary pixel lies within `tolerance`, and vice versa; precision and recall over those
    two counts give the F1.

    **This is the number area Dice cannot give you.** A duct predicted slightly too large
    keeps a high Dice - the interior dominates - while its boundary F1 collapses, and a
    slightly-too-large duct is exactly the error a tile model makes at 224 px granularity.
    `tolerance=3` at 0.5 um/px is 1.5 um, well under one nuclear diameter, so it is a real
    constraint rather than a formality.

    Unlabelled pixels are excluded from both outlines: an edge between annotated tissue and
    unannotated tissue is the edge of the *annotation*, not of the tissue, and scoring
    against it would reward a model for stopping where the pathologist's pen did.
    """
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    labelled = truth != classes.IGNORE
    if not labelled.any():
        return float("nan")

    truth_mask = (truth == cls) & labelled
    predicted_mask = (predicted == cls) & labelled
    if not truth_mask.any() and not predicted_mask.any():
        return float("nan")

    truth_edge = _binary_boundary(truth_mask) & labelled
    predicted_edge = _binary_boundary(predicted_mask) & labelled
    if not truth_edge.any() or not predicted_edge.any():
        return 0.0 if (truth_edge.any() or predicted_edge.any()) else float("nan")

    near_truth = _dilate(truth_edge, tolerance)
    near_predicted = _dilate(predicted_edge, tolerance)

    precision = float((predicted_edge & near_truth).sum()) / float(predicted_edge.sum())
    recall = float((truth_edge & near_predicted).sum()) / float(truth_edge.sum())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    tolerance: int = 3,
) -> dict:
    """Per-class Dice, IoU, recall and boundary F1 over one stack of masks.

    `truth` and `predicted` are `(N, H, W)` or `(H, W)`. Boundary F1 is computed per tile
    and averaged over the tiles where the class appears, not over a concatenated array:
    concatenating would create boundaries at the joins between unrelated tiles.
    """
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    if truth.ndim == 2:
        truth, predicted = truth[None], predicted[None]

    matrix = confusion(truth, predicted)
    rows = matrix.sum(axis=1)
    total = int(matrix.sum())

    per_class: dict[str, dict] = {}
    for index, name in enumerate(classes.CLASS_NAMES):
        scores = [
            boundary_f1(truth[i], predicted[i], index, tolerance=tolerance)
            for i in range(truth.shape[0])
        ]
        finite = [s for s in scores if np.isfinite(s)]
        per_class[name] = {
            "dice": _finite(dice(matrix, index)),
            "iou": _finite(iou(matrix, index)),
            "recall": _finite(matrix[index, index] / rows[index]) if rows[index] else None,
            "pixels": int(rows[index]),
            "boundary_f1": _finite(float(np.mean(finite))) if finite else None,
            "boundary_tiles": len(finite),
        }

    return {
        "labelled_pixels": total,
        "accuracy": _finite(np.trace(matrix) / total) if total else None,
        "confusion": matrix.tolist(),
        "confusion_axes": {
            "rows": "truth", "cols": "prediction", "order": list(classes.CLASS_NAMES),
        },
        "per_class": per_class,
        "boundary_tolerance_px": tolerance,
        # The two cells the project turns on, named rather than left to be read off the
        # matrix by someone who may not know which axis is which.
        "in_situ_called_invasive": int(matrix[classes.NON_INVASIVE, classes.INVASIVE]),
        "invasive_called_in_situ": int(matrix[classes.INVASIVE, classes.NON_INVASIVE]),
    }


def evaluate_by_source(
    truth: np.ndarray, predicted: np.ndarray, sources: list[str], *, tolerance: int = 3
) -> dict:
    """`evaluate` for the whole set and again per label source, never averaged together.

    The `bcss` block is the one that answers "is the model right". The `bracs_dcis` block
    answers "does it agree with BEETLE", which is a different and weaker question.
    """
    sources = list(map(str, sources))
    out = {"all": evaluate(truth, predicted, tolerance=tolerance), "by_source": {}}

    for source in sorted(set(sources)):
        picked = [i for i, s in enumerate(sources) if s == source]
        block = evaluate(truth[picked], predicted[picked], tolerance=tolerance)
        block["labels_drawn_by"] = (
            "pathologists" if source == classes.SOURCE_BCSS
            else "BEETLE's nnU-Net, reviewed by nobody"
        )
        block["tiles"] = len(picked)
        out["by_source"][source] = block
    return out


def _finite(value: float | None, places: int = 4) -> float | None:
    """`None` for a NaN or an infinity, because JSON cannot spell either."""
    if value is None:
        return None
    value = float(value)
    return round(value, places) if np.isfinite(value) else None


def headline(scored: dict) -> str:
    """One line, naming **whose labels** it was scored against.

    The `against` prefix is not decoration. An earlier version said "vs pathologists"
    unconditionally, and on a subset containing no BCSS tiles it printed BEETLE's numbers
    under a claim that a person had drawn them - which is the single most misleading thing
    this report could say, because the whole value of the `bcss` block is that it is the
    only human-labelled one.
    """
    by_source = scored.get("by_source", {})
    if classes.SOURCE_BCSS in by_source:
        block, against = by_source[classes.SOURCE_BCSS], "vs pathologists"
    elif len(by_source) == 1:
        only = next(iter(by_source))
        block = by_source[only]
        against = f"vs {only} labels ({block['labels_drawn_by']}) - NO human-labelled tiles"
    else:
        block, against = scored.get("all", scored), "vs all sources POOLED - read with care"

    per_class = block["per_class"]
    return (
        f"{against}: in-situ Dice {per_class['non_invasive_epithelium']['dice']}, "
        f"boundary F1 {per_class['non_invasive_epithelium']['boundary_f1']} | "
        f"invasive Dice {per_class['invasive_epithelium']['dice']}, "
        f"boundary F1 {per_class['invasive_epithelium']['boundary_f1']} | "
        f"n={block['labelled_pixels']:,} px"
    )
