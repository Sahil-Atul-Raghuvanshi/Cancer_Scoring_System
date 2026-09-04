"""The three classes this step emits, and which one the score is gated on.

**The class ids are the checkpoint's, not this module's.** They are the order the head
was fitted in - `0 non-epithelium`, `1 non-invasive epithelium`, `2 invasive
epithelium` - and the manifest carries that order as data. `verify_order` is called at
load, so a checkpoint trained in some other order is refused rather than served with
every prediction silently permuted. That failure is worth naming because it is
invisible: a permuted model still produces a plausible-looking class map, still fills
in a confusion matrix, and would put stroma in the denominator of a clinical score.

Two of the pipeline guide's six ordering rules land here, and they are the entire
reason this step exists rather than a brightness threshold:

  **Rule 1 - fat leaves here, not at the tissue mask.** Fat is pale, so a saturation
  threshold at step 3 would drop it, and dropping it there would also drop pale tumour
  and pale in-situ ducts. Fat is a *semantic* class, so it is removed by a model that
  knows what fat is - at this step, where `non_epithelium` absorbs it along with
  stroma, inflammation, necrosis and vessels.

  **Rule 5 - scoring is gated on invasive tumour, not on "tumour".** Ductal carcinoma
  in situ is carcinoma that has not broken out of the duct. It is not scored, it is not
  in the denominator, and a pipeline that pools it with invasive disease reports a
  percentage of the wrong population. So in-situ epithelium is its own output class and
  `SCORED` names the one class that is not excluded.

Why a model and not a threshold: nuclear density - the classical proxy - cannot
separate in-situ from invasive, because both are dense. The difference is
*architectural*: in-situ nuclei sit inside an intact duct with a myoepithelial layer
around it, invasive nuclei have broken out into the stroma. That is a spatial-pattern
judgement, which is what learned features are for and what thresholds are not.
"""

from __future__ import annotations

#: Class names in checkpoint order, so `CLASS_NAMES[label]` is the label's name and a
#: confusion matrix's axes are this tuple in this order.
CLASS_NAMES: tuple[str, str, str] = (
    "non_epithelium",
    "non_invasive_epithelium",
    "invasive_epithelium",
)

#: The one class that enters the score. Everything else is excluded, and the two
#: exclusions have different reasons - see the module docstring.
SCORED: int = 2

#: What each class actually holds, in the words the report uses.
CLASS_MEANING: dict[int, str] = {
    0: "stroma, fat, inflammation, necrosis, vessels - excluded (Rule 1)",
    1: "DCIS, LCIS, normal ducts and lobules - excluded (Rule 5)",
    2: "invasive carcinoma - the only thing scored",
}

#: Plain-language labels for the screen. The UI must read for someone with basic
#: knowledge, so "in-situ" and "epithelium" are spelled out here rather than assumed;
#: `CLASS_MEANING` above keeps the rigorous version for the notes.
CLASS_LABELS: dict[int, str] = {
    0: "Not tumour tissue",
    1: "Tumour still inside the duct",
    2: "Tumour that has broken out",
}

#: One-line explanation per class, again for the screen rather than for a reviewer.
CLASS_PLAIN: dict[int, str] = {
    0: "Supporting tissue, fat, inflammation and dead tissue. None of it is scored.",
    1: "Abnormal cells that are still contained inside a duct. Not scored - a "
    "treatment decision is made on the tumour that has escaped.",
    2: "Abnormal cells that have grown out into the surrounding tissue. This is the "
    "only tissue the final score is measured on.",
}

#: Overlay colours, one per class, chosen to survive being drawn at 40% opacity over
#: a pink-and-brown scan. Slate for the tissue that is not scored, amber for the
#: in-situ class - the app's "be careful with this one" colour, and this is the cell
#: where the errors live - and the accent cyan for the class the score is gated on.
CLASS_COLOURS: dict[int, tuple[int, int, int]] = {
    0: (100, 116, 139),
    1: (251, 191, 36),
    2: (56, 189, 248),
}


class ClassOrderError(ValueError):
    """A checkpoint's class order is not the one this step reads its output as."""


def verify_order(classes: list[str] | tuple[str, ...]) -> None:
    """Refuse a checkpoint whose head was fitted in a different class order.

    Not a formality. Every number after this step - the invasive area in mm2, step 9's
    ROI mask, the denominator of the final score - is `label == SCORED`, and a
    permutation of the head's output changes which tissue that selects without
    changing anything a reader could notice. So the order travels in the manifest as
    data and is compared, rather than being a convention two files happen to share.
    """
    if tuple(classes) != CLASS_NAMES:
        raise ClassOrderError(
            f"this checkpoint reports classes {tuple(classes)}, but step 8 reads its "
            f"output as {CLASS_NAMES}. Serving it would relabel every tile - the "
            "invasive class is what the whole score is gated on - so it is refused. "
            "Re-publish the checkpoint with the pipeline's class order."
        )


__all__ = [
    "CLASS_COLOURS",
    "CLASS_LABELS",
    "CLASS_MEANING",
    "CLASS_NAMES",
    "CLASS_PLAIN",
    "SCORED",
    "ClassOrderError",
    "verify_order",
]
