"""Fitting the three-class head, and the two ways it gets evaluated.

Here rather than in a notebook cell because **both approaches must fit identically**.
Approach 3 differs from approach 1 in exactly one respect - where the body's weights
came from - and if each ran its own copy of the optimiser loop, the A/B would be free
to become a comparison of two learning rates instead. One definition, three callers:
approach 1's notebook, approach 3's notebook, and the scripts that actually produced
the published checkpoints.

The head is fitted on **cached features**, which is what makes the whole plan
iterable: the frozen body is run once (notebook 03), and after that a fit is seconds,
so the class weights, the input polarity and the choice of initialisation stop being
decisions to defend in advance and become experiments to run.

Full batch, fixed step count, fixed seed. A linear head on frozen features is convex
up to the optimiser, so there is no early-stopping ritual to perform and nothing to
gain from a schedule - and a fixed recipe is reproducible, which matters more here than
the last decimal place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import bcss
import datasets
import report

#: The recipe. Fixed rather than tuned: every number here was chosen once, and the
#: comparison between two initialisations is only meaningful while they stay fixed.
EPOCHS = 400
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 0


def fit_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    weights: torch.Tensor,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    seed: int = SEED,
) -> tuple[torch.nn.Linear, float]:
    """A class-weighted linear head on cached features. Returns it and the final loss.

    `weights` is not optional and not a refinement. BCSS is overwhelmingly
    non-epithelium, so an unweighted loss is minimised by a model that answers `0` to
    everything: it scores around 80% accuracy, never finds a tumour, and the accuracy
    figure hides that completely.
    """
    torch.manual_seed(seed)
    head = torch.nn.Linear(features.shape[1], len(bcss.CLASS_NAMES))
    optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    x = torch.from_numpy(np.ascontiguousarray(features))
    y = torch.from_numpy(np.ascontiguousarray(labels))

    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = loss_fn(head(x), y)
        loss.backward()
        optimiser.step()

    return head, float(loss.item())


def predict(head: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return head(torch.from_numpy(np.ascontiguousarray(features))).argmax(dim=1).numpy()


def logits(head: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return head(torch.from_numpy(np.ascontiguousarray(features))).numpy()


# --- the two evaluations ------------------------------------------------------


@dataclass
class Fit:
    """One initialisation, fitted and evaluated both ways."""

    init: str
    head: torch.nn.Linear
    final_loss: float
    #: Per-fold headline Dice from the grouped cross-validation on the training
    #: institutions. Its spread is the yardstick any claimed improvement must beat.
    folds: list[float]
    #: The report on the held-out institutions. This is the number that leaves.
    held_out: report.Report

    @property
    def fold_mean(self) -> float:
        return float(np.mean(self.folds)) if self.folds else float("nan")

    @property
    def fold_sd(self) -> float:
        return float(np.std(self.folds)) if self.folds else float("nan")


def cross_validate(
    features: np.ndarray,
    labels: np.ndarray,
    train_rows: list[dict[str, str]],
    index_of: dict[str, int],
    *,
    folds: int = 5,
    seed: int = SEED,
    on_fold=None,
) -> list[float]:
    """Grouped cross-validation on the training institutions only.

    Grouped by slide, and the leak is asserted inside the loop rather than trusted.
    Tiles from one slide are near-duplicates - same patient, same scanner, same stain
    batch, often literal cells at a shared edge - so a random split would report
    memory as generalisation, and would report it as a beautiful number.

    The held-out institutions are not touched here. They are touched once, at the end,
    by `evaluate`.
    """
    weights = datasets.class_weights(train_rows)
    scores: list[float] = []

    for number, held in enumerate(datasets.slide_folds(train_rows, folds=folds, seed=seed)):
        held_set = set(held.tolist())
        fit_rows = [row for i, row in enumerate(train_rows) if i not in held_set]
        val_rows = [train_rows[i] for i in held]
        datasets.assert_no_leak(fit_rows, val_rows)

        fit_index = np.array([index_of[row["tile_id"]] for row in fit_rows])
        val_index = np.array([index_of[row["tile_id"]] for row in val_rows])

        head, _ = fit_head(features[fit_index], labels[fit_index], weights=weights, seed=seed)
        scored = report.evaluate(labels[val_index], predict(head, features[val_index]))
        scores.append(scored.dice_invasive)

        if on_fold is not None:
            on_fold(number, scored, len(val_rows))

    return scores


def evaluate(
    init: str,
    features: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str]],
    *,
    tumour_content: dict[str, float],
    folds: int = 5,
    seed: int = SEED,
    on_fold=None,
) -> Fit:
    """Cross-validate on the training institutions, then fit once and report held out.

    `tumour_content` maps `roi_id` to that region's invasive share, read off the labels
    at export time. It stratifies the report: a slide that is 80-90% tumour is easy, and
    a 2-3 mm focus inside a 2-3 cm block is the hard case *and* the clinically common
    one, so an average over both is a number that can flatter a useless model.
    """
    index_of = {row["tile_id"]: i for i, row in enumerate(rows)}
    train_rows, test_rows = datasets.institution_split(rows)
    datasets.assert_no_leak(train_rows, test_rows)

    fold_scores = cross_validate(
        features, labels, train_rows, index_of, folds=folds, seed=seed, on_fold=on_fold
    )

    train_index = np.array([index_of[row["tile_id"]] for row in train_rows])
    test_index = np.array([index_of[row["tile_id"]] for row in test_rows])

    head, final_loss = fit_head(
        features[train_index], labels[train_index],
        weights=datasets.class_weights(train_rows), seed=seed,
    )

    held_out = report.evaluate(
        labels[test_index],
        predict(head, features[test_index]),
        slide_ids=np.array([row["slide_id"] for row in test_rows]),
        tumour_content=np.array([tumour_content[row["roi_id"]] for row in test_rows]),
        # Keeps BCSS's nine human-drawn in-situ tiles reportable separately from the
        # teacher-labelled majority. Without this the class-1 recall that leaves this
        # function is mostly a measure of how well the head reproduces BEETLE.
        sources=np.array([row.get("source", "bcss") for row in test_rows]),
    )

    return Fit(init=init, head=head, final_loss=final_loss, folds=fold_scores,
               held_out=held_out)


def ablate_class_weights(
    features: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str]],
    *,
    seed: int = SEED,
) -> report.Report:
    """The same fit with an unweighted loss, to show what the weights are for.

    Run and reported rather than asserted. "We used class weights" is a claim; "here
    is the confusion matrix without them" is evidence, and the evidence is usually
    that the unweighted model has the *higher* accuracy - which is the point being
    made about accuracy.
    """
    index_of = {row["tile_id"]: i for i, row in enumerate(rows)}
    train_rows, test_rows = datasets.institution_split(rows)

    train_index = np.array([index_of[row["tile_id"]] for row in train_rows])
    test_index = np.array([index_of[row["tile_id"]] for row in test_rows])

    head, _ = fit_head(
        features[train_index], labels[train_index],
        weights=torch.ones(len(bcss.CLASS_NAMES)), seed=seed,
    )
    return report.evaluate(labels[test_index], predict(head, features[test_index]))


def as_json(fit: Fit) -> dict[str, object]:
    """One fit, as the JSON the reports directory keeps and the manifest embeds."""
    scored = fit.held_out
    return {
        "init": fit.init,
        "final_loss": fit.final_loss,
        "recipe": {"epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED},
        "groupkfold_dice": {
            "mean": fit.fold_mean, "sd": fit.fold_sd, "folds": fit.folds,
        },
        "held_out": {
            "tiles": int(scored.tiles),
            "accuracy": scored.accuracy,
            "macro_f1": scored.macro_f1,
            "dice_invasive_vs_non_invasive": scored.dice_invasive,
            "per_class_dice": scored.per_class_dice,
            "confusion": scored.matrix.tolist(),
            "confusion_axes": {"rows": "truth", "cols": "prediction",
                               "order": list(bcss.CLASS_NAMES)},
            "invasive_called_non_invasive": scored.invasive_called_non_invasive,
            "non_invasive_called_invasive": scored.non_invasive_called_invasive,
            "invasive_recall": scored.invasive_recall,
            "non_invasive_recall": scored.non_invasive_recall,
            "by_tumour_content": scored.by_tumour_content,
            "per_slide_dice": scored.by_slide,
            # The two class-1 populations, never averaged together. `bcss` is nine
            # human-drawn tiles; `bracs_dcis` is thousands of BEETLE's predictions.
            "by_source": scored.by_source,
        },
    }
