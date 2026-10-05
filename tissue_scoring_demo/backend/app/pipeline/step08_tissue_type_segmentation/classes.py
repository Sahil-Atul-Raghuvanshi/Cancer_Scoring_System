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

**There is a fourth colour on the map and it is not a fourth class.** `uncertain` is a
*display* label the post-pass in `uncertainty.py` writes over class-1 windows it would
not stand behind; the model has three logits and no way to decline, and the flag is
derived from the finished class map rather than predicted. So it lives in the display
tables here - `CLASS_COLOURS`, `CLASS_LABELS`, `CLASS_PLAIN` - and pointedly not in
`CLASS_NAMES`, which is the checkpoint's contract. Two consequences worth carrying:
`verify_order` is unaffected, and because the flag can only ever land on class 1 - which
Rule 5 already excludes - nothing it does can move `SCORED`, the score or step 9's ROI.
"""

from __future__ import annotations

from .uncertainty import UNCERTAIN

#: Class names in checkpoint order, so `CLASS_NAMES[label]` is the label's name and a
#: confusion matrix's axes are this tuple in this order.
#:
#: **Three, and it stays three however many colours the map ends up with.** This tuple is
#: what `verify_order` compares a checkpoint's manifest against, and it is the shape of
#: `ClassMap.probabilities`, `counts` and `areas_mm2`. The `uncertain` display class
#: below is *not* here: nothing was fitted for it, no head emits it, and appending it
#: would make every published checkpoint fail load.
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
    UNCERTAIN: (
        "an in-situ call the uncertainty layer would not stand behind - excluded, and "
        "already excluded by Rule 5 before it was flagged"
    ),
}

#: Plain-language labels for the screen. The UI must read for someone with basic
#: knowledge, so "in-situ" and "epithelium" are spelled out here rather than assumed;
#: `CLASS_MEANING` above keeps the rigorous version for the notes.
CLASS_LABELS: dict[int, str] = {
    0: "Not tumour",
    1: "Tumour inside a duct",
    2: "Invasive tumour",
    UNCERTAIN: "Cannot be determined",
}

#: One-line explanation per class, again for the screen rather than for a reviewer.
CLASS_PLAIN: dict[int, str] = {
    0: "Supporting tissue, fat, inflammation and dead tissue. Not scored.",
    1: "Tumour cells still held inside a duct. Not scored, because treatment "
    "decisions are based on tumour that has spread out.",
    2: "Tumour cells that have grown out into the surrounding tissue. This is the "
    "only tissue the final score is measured on.",
    UNCERTAIN: "The model called this tumour inside a duct, but the surroundings or "
    "the shape do not support that. Shown separately rather than counted, and worth "
    "a human look. It was not going to be scored either way.",
}

#: Overlay colours, one per class, chosen to survive being drawn at 55% opacity over
#: a pink-and-brown scan. Red for invasive tumour - the class the score is gated on,
#: and the only one a reader should be drawn to first - blue for in-situ disease, and
#: yellow for the non-epithelium bulk. The three are far apart in hue rather than in
#: brightness, because a class map is read on a scan that is already pink and brown
#: and two colours a step apart on the same ramp stop being separable there.
CLASS_COLOURS: dict[int, tuple[int, int, int]] = {
    0: (250, 204, 21),
    1: (59, 130, 246),
    2: (239, 68, 68),
    UNCERTAIN: (168, 85, 247),
}

#: The names the *map* has, which is the three the model emits plus the one the
#: uncertainty layer adds. Everything that draws or tabulates the picture iterates this;
#: everything that indexes the model's output iterates `CLASS_NAMES`. Keeping the two
#: apart is what lets a fourth colour exist without a fourth logit.
DISPLAY_NAMES: tuple[str, ...] = (*CLASS_NAMES, "uncertain")


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


#: The one class the uncertainty layer may qualify, and the one class whose presence
#: nearby counts as evidence against it. Named here rather than written as literals at
#: `uncertainty.derive`'s call site, so the pair travels with the class table it indexes.
#:
#: `CANDIDATE` is in-situ because that is the class with somewhere to fall back to: it is
#: already outside the score under Rule 5, so flagging it costs nothing downstream.
#: `RIVAL` is invasive because that is the competing *reading of the same tissue* - the
#: distinction Rule 5 turns on. Non-epithelium is neither, and the reason is the whole
#: argument in `uncertainty.py`: stroma around a duct is that duct's context, not a
#: rival claim to it.
CANDIDATE: int = 1
RIVAL: int = SCORED

__all__ = [
    "CLASS_COLOURS",
    "CLASS_LABELS",
    "CLASS_MEANING",
    "CLASS_NAMES",
    "CLASS_PLAIN",
    "CANDIDATE",
    "DISPLAY_NAMES",
    "RIVAL",
    "SCORED",
    "UNCERTAIN",
    "ClassOrderError",
    "verify_order",
]
