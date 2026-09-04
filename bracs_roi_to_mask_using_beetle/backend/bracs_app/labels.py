"""BEETLE's classes, BCSS's codes, and the translation between them.

This module is the whole experiment in one file. Everything else here is plumbing:
loading a network, cutting an image into patches, drawing a picture. What decides
whether the answer is worth anything is what a BEETLE prediction is allowed to *mean*
when it is written into a BCSS-shaped mask, and that decision is here, alone, written
out rather than inlined at the two places that need it.

--------------------------------------------------------------------------------
The correction this module carries
--------------------------------------------------------------------------------

`beetle_teacher_resnet18/src/beetle.py` records BEETLE's label codes as

    unannotated 0, other 1, invasive_epithelium 2, non_invasive_epithelium 3, necrosis 4

read off the paper, with a docstring saying they are unconfirmed until checked against
the release, and an `assert_codes_verified` that fails closed until they are. They are
now checked, against two files inside `model.zip` itself:

    dataset.json               -> "non-invasive epithelium": 2, "invasive epithelium": 3
    checkpoint init_args       -> "non-invasive epithelium": 2, "invasive epithelium": 3

**Codes 2 and 3 are the other way round.** Non-invasive is 2 and invasive is 3, in the
weights this app actually runs. That is exactly the failure `assert_codes_verified` was
written to prevent, and it is the single most damaging error possible in this project:
with the paper's ordering, every duct BEETLE calls in-situ would be exported as class 2
`invasive_epithelium` and every invasive focus as class 1 - a training set that is not
merely noisy but inverted precisely on the boundary the region model exists to draw. It
would train to a plausible accuracy and be wrong about the only thing that matters.

So this module reads its codes from the archive at run time (`codes_from_dataset_json`)
and treats the constants below as an assertion to check against, never as the source.

--------------------------------------------------------------------------------
Why the output is a BCSS-coded mask rather than our three classes
--------------------------------------------------------------------------------

The point of the exercise is to feed approach 1's existing exporter, unmodified. That
exporter takes a raw BCSS mask and calls `bcss.remap` on it, which is the function
carrying every judgement call approach 1 has already made and tested - `dcis` and
`normal_acinus_or_duct` to class 1, `outside_roi` to `IGNORE`, and the assertion that
`tumor` and `dcis` never collapse together. Emitting our three classes directly would
mean bypassing `bcss.remap`, which means a second definition of what class 1 is, which
is the exact failure mode `beetle_teacher_resnet18/src/backend_path.py` is written to
prevent one layer down. So the teacher's opinion is written in BCSS's own vocabulary
and handed to approach 1's pipeline as though it had come from a pathologist.

Three of the four translations are forced. One is a judgement call and is flagged.
"""

from __future__ import annotations

import numpy as np

# --- BEETLE, as the released archive defines it -------------------------------

#: What `dataset.json` inside `model.zip` says, checked at load time by
#: `codes_from_dataset_json`. Note 2 and 3 against the paper's ordering; see above.
BEETLE_CODES: dict[str, int] = {
    "unannotated": 0,
    "other": 1,
    "non_invasive_epithelium": 2,
    "invasive_epithelium": 3,
    "necrosis": 4,
}

#: `dataset.json` spells these with spaces and hyphens, and calls code 0 `unannotated`
#: in the released copy but `background` in the checkpoint's own `init_args`. Both
#: names mean the same channel. Normalising here means the verification below compares
#: meanings rather than spellings.
_ALIASES: dict[str, str] = {
    "background": "unannotated",
    "unannotated": "unannotated",
    "other": "other",
    "non-invasive epithelium": "non_invasive_epithelium",
    "non_invasive_epithelium": "non_invasive_epithelium",
    "invasive epithelium": "invasive_epithelium",
    "invasive_epithelium": "invasive_epithelium",
    "necrosis": "necrosis",
}

#: Index into the network's 5 output channels, in channel order. The model has an
#: opinion about every pixel including channel 0, which is *not* an abstention - it is
#: a learned "this looks like the unannotated regions of my training slides", mostly
#: glass and background. Approach 4a's notes assume a 4-channel output with no such
#: channel; the checkpoint's `seg_layers` are 5-channel and this is that fifth.
CHANNEL_NAMES: tuple[str, ...] = (
    "unannotated",
    "other",
    "non_invasive_epithelium",
    "invasive_epithelium",
    "necrosis",
)


def codes_from_dataset_json(labels: dict[str, int]) -> dict[str, int]:
    """The released `labels` block, normalised, with the paper's ordering re-checked.

    Raises if the archive disagrees with `BEETLE_CODES`. That is not paranoia about a
    file that has already been read once: the check costs nothing and it is the only
    thing standing between a differently-ordered future release and a silently
    inverted training set.
    """
    normalised: dict[str, int] = {}
    for name, code in labels.items():
        key = _ALIASES.get(name.strip().lower())
        if key is None:
            raise ValueError(
                f"dataset.json names a class {name!r} that this app has no translation "
                "for. A fifth tissue class cannot be mapped by guesswork."
            )
        normalised[key] = int(code)

    if normalised != BEETLE_CODES:
        raise ValueError(
            f"the released label codes are {normalised}, and this app was written "
            f"against {BEETLE_CODES}. Refusing to run: if 2 and 3 have moved, every "
            "in-situ duct would be exported as invasive carcinoma and the resulting "
            "training set would be inverted on the one boundary that matters."
        )
    return normalised


# --- BCSS, as approach 1's exporter expects it --------------------------------

#: The BCSS codes this app writes. Imported from approach 1 at call time rather than
#: restated - see `to_bcss_codes` - but named here so the mapping reads as a mapping.
#:
#: `other` -> `stroma` is the one judgement call in the table, and it is a real one.
#: BEETLE's `other` is everything that is not epithelium or necrosis: stroma,
#: lymphocytes, fat, vessels, glass. BCSS keeps those apart, and this collapses them
#: all onto code 2. Nothing downstream can tell the difference - approach 1's
#: `bcss.remap` sends stroma, lymphocytic_infiltrate, fat, blood_vessel and eleven
#: others to the same class 0 - so the collapse is lossless *for this pipeline* and
#: would not be for any other consumer of these masks. Two consequences worth stating
#: next to any number derived from them:
#:
#:   * a mask written here must never be used to train anything that separates stroma
#:     from inflammation. It does not contain that information.
#:   * approach 1's tile vote requires 80 % agreement to call a tile non-epithelium,
#:     and a merged class clears that threshold more easily than three separate ones
#:     would. This makes class 0 tiles slightly *easier* to obtain from BRACS than
#:     from BCSS. Since class 0 is the class this experiment is not short of, that
#:     asymmetry costs nothing - but it means BRACS class 0 counts are not comparable
#:     with BCSS class 0 counts.
#:
#: The other three are forced, and are the reason approach 4a is cheaper in label risk
#: than approach 1 was. BEETLE's own definition of its non-invasive class is "healthy
#: glands and DCIS ... LCIS, atypical ductal hyperplasia, apocrine metaplasia", which
#: is what BCSS splits between `dcis` (20) and `normal_acinus_or_duct` (13) and what
#: `bcss.remap` reunites into class 1.
#:
#: **Which of those two BEETLE's non-invasive class becomes depends on the region, so
#: it is the one target a caller may override.** `dcis` is the default because it is
#: right for the two carcinoma trees: in a BRACS DCIS region the epithelium is in-situ
#: carcinoma by the dataset's own annotation, and in an IC region BEETLE's non-invasive
#: prediction is most likely the adjacent DCIS the ROI label does not cover.
#:
#: It is **wrong for a normal region**, and the error is worth spelling out because
#: `bcss.remap` hides it: 13 and 20 both land in class 1, so the training tiles are
#: identical either way and no downstream number moves. What differs is what the mask
#: *asserts*. Writing code 20 over a consensus-normal region produces a file claiming
#: 43% of it is carcinoma in-situ where three pathologists said there is none - and
#: these masks are handed to approach 1's exporter "as though they had come from a
#: pathologist". Anything that later separates DCIS from normal duct - a four-class
#: head, `pixel_unet_resnet18` reading these directories, or a person auditing them -
#: would be reading an inversion. So the normal tree passes 13.
BCSS_TARGETS: dict[str, str] = {
    "unannotated": "outside_roi",
    "other": "stroma",
    "non_invasive_epithelium": "dcis",
    "invasive_epithelium": "tumor",
    "necrosis": "necrosis_or_debris",
}

#: The default target for BEETLE's non-invasive class, and the only one overridable.
#: `config.RoiTree.non_invasive_code` supplies the per-tree value.
DEFAULT_NON_INVASIVE_TARGET: str = BCSS_TARGETS["non_invasive_epithelium"]


def targets_for(non_invasive_target: str = DEFAULT_NON_INVASIVE_TARGET) -> dict[str, str]:
    """`BCSS_TARGETS` with BEETLE's non-invasive class redirected.

    Kept as a function rather than letting callers build the dict so the four forced
    translations cannot be overridden by accident - only the one that is a genuine
    per-region judgement.
    """
    import bcss

    if non_invasive_target not in bcss.GT_CODES:
        raise ValueError(
            f"{non_invasive_target!r} is not a BCSS label. The non-invasive target must "
            f"be one of BCSS's own codes, and for this pipeline one that `bcss.remap` "
            "sends to class 1 - `dcis` or `normal_acinus_or_duct`."
        )
    return {**BCSS_TARGETS, "non_invasive_epithelium": non_invasive_target}


def to_bcss_codes(
    beetle_mask: np.ndarray,
    *,
    non_invasive_target: str = DEFAULT_NON_INVASIVE_TARGET,
) -> np.ndarray:
    """A BEETLE argmax to a mask approach 1's exporter can read unmodified.

    Returns uint16, which is what the BCSS release ships and what `PIL` writes as a
    single-channel `I;16` PNG. uint8 would round-trip through PIL as mode `L` and be
    read back identically, but a mask that is byte-for-byte the same *kind of file* as
    `TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500.png` is one fewer difference to
    argue about when the tiles turn out different from BCSS's.

    `non_invasive_target` picks which BCSS in-situ code BEETLE's non-invasive class is
    written as - see `BCSS_TARGETS` for why a normal region needs 13 and not 20.
    """
    import bcss  # approach 1's, via backend_path; the codes come from its GT_CODES

    targets = targets_for(non_invasive_target)
    beetle_mask = np.asarray(beetle_mask)
    if beetle_mask.dtype.kind not in "ui":
        raise TypeError(f"a label mask must be integer-valued, got {beetle_mask.dtype}")

    present = set(np.unique(beetle_mask).tolist())
    if not present <= set(BEETLE_CODES.values()):
        raise ValueError(
            f"mask carries codes {sorted(present - set(BEETLE_CODES.values()))}, which "
            "the BEETLE release does not use. This is not a teacher prediction."
        )

    out = np.zeros(beetle_mask.shape, dtype=np.uint16)
    for beetle_name, code in BEETLE_CODES.items():
        out[beetle_mask == code] = bcss.GT_CODES[targets[beetle_name]]
    return out


def to_our_classes(
    beetle_mask: np.ndarray,
    *,
    non_invasive_target: str = DEFAULT_NON_INVASIVE_TARGET,
) -> np.ndarray:
    """The same mask in approach 1's three classes, for display and for counting.

    Goes the long way round on purpose - through `to_bcss_codes` and then approach 1's
    own `bcss.remap` - so that what this function shows and what the exporter acts on
    cannot drift apart. A shortcut here would be a second definition of class 1, which
    is the failure this whole module is arranged to make impossible.
    """
    import bcss

    return bcss.remap(
        to_bcss_codes(beetle_mask, non_invasive_target=non_invasive_target)
    )


#: What each of our three classes is painted in the overlay, and why anyone should
#: care. Kept beside the mapping so a legend cannot disagree with the pixels.
CLASS_COLOURS: dict[int, tuple[int, int, int]] = {
    0: (120, 144, 156),  # non-epithelium - slate, deliberately dull
    1: (255, 179, 0),    # non-invasive epithelium - amber, the class being harvested
    2: (216, 27, 96),    # invasive carcinoma - magenta
}

#: `bcss.IGNORE`, drawn as nothing. Repeated as a literal only in the overlay, where a
#: colour is needed for "no label"; the sentinel itself is always imported.
IGNORE_COLOUR: tuple[int, int, int] = (28, 28, 32)


def colour_legend() -> list[dict]:
    """What each overlay colour means, derived through `bcss.remap` rather than restated.

    A legend is a claim about the pixels, so it is computed by pushing one pixel of each
    BEETLE class through the same two functions that painted them - `to_bcss_codes` and
    approach 1's `bcss.remap` - and reading off where it lands. Writing the grouping out
    by hand would be a third definition of what class 1 is, which is the failure this
    module's docstring is about; it would also be the definition most likely to be read
    and believed, because it is the one on the screen.

    Returns one entry per class in `bcss.CLASS_NAMES` order, plus a final entry for
    `bcss.IGNORE`, which is not a class but is a colour on the picture.
    """
    import bcss

    grouped: dict[int, list[str]] = {}
    for beetle_name, code in BEETLE_CODES.items():
        one_pixel = np.full((1, 1), code, dtype=np.uint8)
        our_class = int(bcss.remap(to_bcss_codes(one_pixel))[0, 0])
        grouped.setdefault(our_class, []).append(beetle_name)

    legend = []
    for index, class_name in enumerate(bcss.CLASS_NAMES):
        legend.append(
            {
                "class_index": index,
                "class_name": class_name,
                "rgb": list(CLASS_COLOURS[index]),
                "hex": "#%02x%02x%02x" % CLASS_COLOURS[index],
                "beetle_classes": grouped.get(index, []),
                "bcss_codes": {
                    name: bcss.GT_CODES[BCSS_TARGETS[name]] for name in grouped.get(index, [])
                },
                "scored": index != 0,
            }
        )

    legend.append(
        {
            "class_index": int(bcss.IGNORE),
            "class_name": "unlabelled",
            "rgb": list(IGNORE_COLOUR),
            "hex": "#%02x%02x%02x" % IGNORE_COLOUR,
            "beetle_classes": grouped.get(int(bcss.IGNORE), []),
            "bcss_codes": {
                name: bcss.GT_CODES[BCSS_TARGETS[name]]
                for name in grouped.get(int(bcss.IGNORE), [])
            },
            "scored": False,
        }
    )
    return legend
