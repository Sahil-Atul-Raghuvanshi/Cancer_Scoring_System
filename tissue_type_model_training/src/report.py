"""The report: the confusion matrix, the one cell that matters, and the paired A/B.

Everything here exists to stop a single average from being the answer. Three
separate reasons, each of which has sunk a pathology model before:

  the cell         a 3x3 confusion matrix averaged into one number hides the
                   invasive-versus-in-situ confusion, which *is* the project. It is
                   reported on its own line, always.
  the strata       a slide that is 80-90% tumour is easy; a 2-3 mm invasive focus in
                   a 2-3 cm block, mixed with DCIS and normal ducts, is the hard case
                   and the clinically common one. A model that averages well can be
                   useless on exactly the cases that matter, so accuracy is binned by
                   tumour content and each bin carries its tile count.
  the pairing      the expected gap between approach 1 and approach 3 is about three
                   points, and fold-to-fold scatter on 151 regions is the same size.
                   Two unpaired averages three points apart are a coin toss wearing a
                   decimal point, so the comparison is per slide and paired.

No plotting here. Numbers and tables, returned as data, so the notebook decides how
to draw them and a test can assert on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import bcss

# --- the matrix ---------------------------------------------------------------


def confusion(truth: np.ndarray, predicted: np.ndarray, *, classes: int = 3) -> np.ndarray:
    """Rows are truth, columns are prediction. The convention every caller assumes."""
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)), 1)
    return matrix


def dice(matrix: np.ndarray, cls: int) -> float:
    """Dice / F1 for one class against all others: `2TP / (2TP + FP + FN)`."""
    true_positive = float(matrix[cls, cls])
    false_positive = float(matrix[:, cls].sum() - true_positive)
    false_negative = float(matrix[cls, :].sum() - true_positive)
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator > 0 else 0.0


def dice_invasive_vs_non_invasive(truth: np.ndarray, predicted: np.ndarray) -> float:
    """**The project's number.** Dice on invasive, over the epithelial tiles only.

    Restricted to tiles whose truth is epithelium - invasive or not - because that is
    the boundary the whole model exists to draw. Including the non-epithelial tiles
    would let a model that merely separates tissue from stroma score well here, which
    is the easy half of the problem and not the half anyone is worried about.

    Reported alone, never folded into a macro average. Guide step 9 and both design
    documents say the same thing: that one number is the project.
    """
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    epithelial = np.isin(truth, (bcss.NON_INVASIVE, bcss.INVASIVE))
    if not epithelial.any():
        return float("nan")

    matrix = confusion(truth[epithelial], predicted[epithelial])
    return dice(matrix, bcss.INVASIVE)


@dataclass
class Report:
    """Everything one model's evaluation produced, as data rather than as printout."""

    matrix: np.ndarray
    per_class_dice: dict[str, float]
    macro_f1: float
    accuracy: float
    #: The headline. Invasive vs non-invasive, epithelial tiles only.
    dice_invasive: float
    #: The two off-diagonal cells that matter, as counts and as rates.
    invasive_called_non_invasive: int
    non_invasive_called_invasive: int
    invasive_recall: float
    non_invasive_recall: float
    tiles: int
    by_tumour_content: list[dict[str, object]] = field(default_factory=list)
    by_slide: dict[str, float] = field(default_factory=dict)
    #: The same numbers computed separately for each label source. Present only when
    #: `evaluate` is given `sources`, and the reason it exists is that this project's
    #: class 1 now comes from two places that are not equally trustworthy: BCSS's
    #: nine tiles were drawn by people, and every other class-1 tile is BEETLE's
    #: opinion. A single class-1 recall averaged over both would be dominated by the
    #: teacher-labelled majority and would be reporting agreement with a model as
    #: though it were accuracy.
    by_source: dict[str, dict[str, object]] = field(default_factory=dict)

    def headline(self) -> str:
        """One line, with the cell that matters spelled out rather than averaged."""
        return (
            f"Dice invasive vs non-invasive: {self.dice_invasive:.3f}   "
            f"(invasive called in-situ: {self.invasive_called_non_invasive}, "
            f"in-situ called invasive: {self.non_invasive_called_invasive}, "
            f"n={self.tiles:,})"
        )


def evaluate(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    slide_ids: np.ndarray | None = None,
    tumour_content: np.ndarray | None = None,
    sources: np.ndarray | None = None,
) -> Report:
    """Build the whole report from one set of predictions.

    `tumour_content` is the *region's* invasive share, one value per tile, used only
    for the stratification. It comes from the manifest rather than from the model, so
    a bin is a property of the tissue and not of the prediction.
    """
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    matrix = confusion(truth, predicted)

    per_class = {name: dice(matrix, index) for index, name in enumerate(bcss.CLASS_NAMES)}
    row_sums = matrix.sum(axis=1)

    report = Report(
        matrix=matrix,
        per_class_dice=per_class,
        macro_f1=float(np.mean(list(per_class.values()))),
        accuracy=float(np.trace(matrix) / max(1, matrix.sum())),
        dice_invasive=dice_invasive_vs_non_invasive(truth, predicted),
        invasive_called_non_invasive=int(matrix[bcss.INVASIVE, bcss.NON_INVASIVE]),
        non_invasive_called_invasive=int(matrix[bcss.NON_INVASIVE, bcss.INVASIVE]),
        invasive_recall=float(
            matrix[bcss.INVASIVE, bcss.INVASIVE] / max(1, row_sums[bcss.INVASIVE])
        ),
        non_invasive_recall=float(
            matrix[bcss.NON_INVASIVE, bcss.NON_INVASIVE] / max(1, row_sums[bcss.NON_INVASIVE])
        ),
        tiles=int(matrix.sum()),
    )

    if tumour_content is not None:
        report.by_tumour_content = stratify_by_tumour_content(truth, predicted, tumour_content)
    if slide_ids is not None:
        report.by_slide = per_slide_dice(truth, predicted, slide_ids)
    if sources is not None:
        report.by_source = split_by_source(truth, predicted, sources)

    return report


def split_by_source(
    truth: np.ndarray, predicted: np.ndarray, sources: np.ndarray
) -> dict[str, dict[str, object]]:
    """One sub-report per label source, so the two class-1 populations stay apart.

    `bcss` rows are human-drawn labels. `bracs_dcis` rows are a model's prediction that
    no pathologist has reviewed. Scoring against the second measures agreement with
    BEETLE; only the first measures agreement with a person. They are both worth having
    and they are not the same number, so they are never added together.
    """
    out: dict[str, dict[str, object]] = {}
    for source in sorted(set(map(str, sources.tolist()))):
        mask = sources == source
        sub = confusion(truth[mask], predicted[mask])
        row_sums = sub.sum(axis=1)
        out[source] = {
            "tiles": int(mask.sum()),
            "accuracy": float(np.trace(sub) / max(1, sub.sum())),
            "confusion": sub.tolist(),
            "per_class_dice": {
                name: dice(sub, index) for index, name in enumerate(bcss.CLASS_NAMES)
            },
            "class_tiles": {
                name: int(row_sums[index]) for index, name in enumerate(bcss.CLASS_NAMES)
            },
            "non_invasive_recall": (
                float(sub[bcss.NON_INVASIVE, bcss.NON_INVASIVE]
                      / row_sums[bcss.NON_INVASIVE])
                if row_sums[bcss.NON_INVASIVE] else None
            ),
            "invasive_recall": (
                float(sub[bcss.INVASIVE, bcss.INVASIVE] / row_sums[bcss.INVASIVE])
                if row_sums[bcss.INVASIVE] else None
            ),
            # Asked of every borrowed source, not of `bracs_dcis` alone: `bracs_ic`
            # and `bracs_normal` masks are the same nnU-Net's output, reviewed by the
            # same nobody, and a report that called them human-drawn would be a lie
            # about the one thing this breakdown exists to say.
            "labels_drawn_by": (
                "BEETLE (unreviewed)" if bcss.is_borrowed(source) else "people"
            ),
        }
    return out


#: Tumour-content bins, as shares of a region that is invasive. The first bin is the
#: clinically common hard case - a small focus in a large block - and it is the one an
#: overall average hides most effectively.
TUMOUR_BINS: tuple[tuple[float, float, str], ...] = (
    (0.00, 0.05, "0-5%"),
    (0.05, 0.20, "5-20%"),
    (0.20, 0.50, "20-50%"),
    (0.50, 1.01, ">50%"),
)


def stratify_by_tumour_content(
    truth: np.ndarray, predicted: np.ndarray, content: np.ndarray
) -> list[dict[str, object]]:
    """Accuracy and the headline Dice per tumour-content bin, with tile counts.

    The counts are not decoration. A bin holding forty tiles is a curiosity and not a
    result, and without the count beside it a reader cannot tell which is which.
    """
    content = np.asarray(content, dtype=float)
    rows: list[dict[str, object]] = []

    for low, high, name in TUMOUR_BINS:
        selected = (content >= low) & (content < high)
        count = int(selected.sum())
        if count == 0:
            rows.append({"bin": name, "tiles": 0, "accuracy": None, "dice_invasive": None})
            continue

        matrix = confusion(truth[selected], predicted[selected])
        rows.append(
            {
                "bin": name,
                "tiles": count,
                "accuracy": float(np.trace(matrix) / max(1, matrix.sum())),
                "dice_invasive": dice_invasive_vs_non_invasive(
                    truth[selected], predicted[selected]
                ),
            }
        )
    return rows


def per_slide_dice(
    truth: np.ndarray, predicted: np.ndarray, slide_ids: np.ndarray
) -> dict[str, float]:
    """The headline number per slide - the unit the A/B comparison is paired on."""
    slide_ids = np.asarray(slide_ids)
    result: dict[str, float] = {}
    for slide in np.unique(slide_ids):
        selected = slide_ids == slide
        value = dice_invasive_vs_non_invasive(truth[selected], predicted[selected])
        if not np.isnan(value):
            result[str(slide)] = value
    return result


# --- the A/B ------------------------------------------------------------------


@dataclass
class Paired:
    """The result of comparing two models slide by slide."""

    slides: int
    mean_difference: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    ties: int
    #: Wilcoxon signed-rank p, or None when scipy is unavailable or n is too small.
    p_value: float | None

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero. Stated plainly, either way."""
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)

    def headline(self) -> str:
        verdict = "a real difference" if self.significant else "inside the noise"
        return (
            f"mean per-slide difference {self.mean_difference:+.3f} "
            f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}], "
            f"{self.wins}W/{self.losses}L/{self.ties}T over {self.slides} slides - {verdict}"
        )


def paired_comparison(
    baseline: dict[str, float], candidate: dict[str, float], *, seed: int = 0
) -> Paired:
    """Compare two models on the slides they were both scored on.

    Paired, because the slides are the same slides: a slide that is hard for one model
    is hard for the other, and pairing removes that shared difficulty from the
    comparison instead of leaving it in the variance where it drowns a three-point
    effect.

    The interval is a bootstrap over slides rather than a normal approximation - with
    forty-odd slides and a bounded, skewed statistic there is no reason to assume a
    shape - and the Wilcoxon signed-rank test is reported beside it as the standard
    non-parametric paired test. If they disagree, believe neither and say so.
    """
    shared = sorted(set(baseline) & set(candidate))
    if len(shared) < 3:
        raise ValueError(
            f"only {len(shared)} slides were scored by both models; a paired "
            "comparison on that is not a comparison. Check that both were evaluated "
            "on the same held-out set."
        )

    differences = np.asarray([candidate[slide] - baseline[slide] for slide in shared])

    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(10_000, differences.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, (2.5, 97.5))

    p_value: float | None = None
    try:
        from scipy.stats import wilcoxon

        if np.any(differences != 0):
            p_value = float(wilcoxon(differences).pvalue)
    except Exception:  # noqa: BLE001 - scipy absent or degenerate input; report None
        p_value = None

    return Paired(
        slides=len(shared),
        mean_difference=float(differences.mean()),
        ci_low=float(low),
        ci_high=float(high),
        wins=int((differences > 0).sum()),
        losses=int((differences < 0).sum()),
        ties=int((differences == 0).sum()),
        p_value=p_value,
    )


# --- printing -----------------------------------------------------------------


def format_matrix(matrix: np.ndarray) -> str:
    """The confusion matrix as text, with the two cells that matter marked.

    Marked rather than merely present: the invasive/non-invasive pair is the reason
    the matrix is being printed, and a reader scanning nine numbers should not have to
    work out which two they are.
    """
    short = ("non-epi", "in-situ", "invasive")
    lines = [f"{'truth \\ pred':>14}" + "".join(f"{name:>11}" for name in short)]

    for row, name in enumerate(short):
        cells = []
        for col in range(3):
            marker = (
                " *"
                if {row, col} == {bcss.INVASIVE, bcss.NON_INVASIVE} and row != col
                else "  "
            )
            cells.append(f"{int(matrix[row, col]):>9,}{marker}")
        lines.append(f"{name:>14}" + "".join(cells))

    lines.append("  * the invasive <-> in-situ confusion. This pair is the project.")
    return "\n".join(lines)
