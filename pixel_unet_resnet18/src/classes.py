"""BCSS's 22 label codes, the 3 classes we collapse them into, and the ignore sentinel.

A standalone restatement of the mapping `bcss_bracs_hchannel_resnet18/src/bcss.py` carries. It
is restated rather than imported because this folder depends on no code from there - see
`paths.py` - and it is checked against that module by `tests/test_parity.py`, which skips
when approach 1 is absent. Two definitions of class 1 is precisely the failure this
project cannot afford, so the copy is *tested*, not trusted.

**The trap this module exists to make impossible.** There is a widely-used "5-class"
version of BCSS - tumour / stroma / inflammatory / necrosis / other - and it merges `dcis`
into `tumor`. That is the one merge this project cannot accept: separating in-situ from
invasive carcinoma is the entire point. So the raw 22-code map is written out in full,
`TO_CLASS` is asserted total at import, and a test checks that codes 1 and 20 land in
different classes.

Amgad M, Elfandy H, ... Cooper LAD. Structured crowdsourcing enables convolutional
segmentation of histology images. Bioinformatics 35(18):3461-3467 (2019).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: Every code the raw BCSS masks use, by name. All 22, written out rather than parsed so
#: the mapping below can be checked against `meta/gtruth_codes.tsv` by eye.
GT_CODES: dict[str, int] = {
    "outside_roi": 0,
    "tumor": 1,
    "stroma": 2,
    "lymphocytic_infiltrate": 3,
    "necrosis_or_debris": 4,
    "glandular_secretions": 5,
    "blood": 6,
    "exclude": 7,
    "metaplasia_NOS": 8,
    "fat": 9,
    "plasma_cells": 10,
    "other_immune_infiltrate": 11,
    "mucoid_material": 12,
    "normal_acinus_or_duct": 13,
    "lymphatics": 14,
    "undetermined": 15,
    "nerve": 16,
    "skin_adnexa": 17,
    "blood_vessel": 18,
    "angioinvasion": 19,
    "dcis": 20,
    "other": 21,
}

CODE_NAMES: dict[int, str] = {code: name for name, code in GT_CODES.items()}

# --- our three classes --------------------------------------------------------

NON_EPITHELIUM = 0
NON_INVASIVE = 1
INVASIVE = 2

CLASS_NAMES: tuple[str, str, str] = (
    "non_epithelium",
    "non_invasive_epithelium",
    "invasive_epithelium",
)

#: Sentinel for "no class". 255 rather than -1 so a mask stays uint8, and far from any
#: real class so an off-by-one cannot land on it.
#:
#: **This is not a fourth class and it must never become one.** At pixel level it is far
#: more load-bearing than it was for the tile model: BCSS is 3.2 % unlabelled and every
#: pixel outside a BRACS annotation carries it too. Folding those into class 0 would teach
#: the model that unannotated tissue is stroma, and would inflate every accuracy figure.
#: `segtrain.py` passes it as `ignore_index`; `segreport.py` masks it out before counting.
IGNORE = 255

#: Codes that are not a class. The BCSS authors are explicit that zero pixels lie outside
#: the annotated region; `exclude` and `undetermined` are the annotators saying they could
#: not tell, which is not a label either.
IGNORE_CODES: frozenset[int] = frozenset(
    {GT_CODES["outside_roi"], GT_CODES["exclude"], GT_CODES["undetermined"]}
)

_INVASIVE_LABELS = ("tumor", "angioinvasion")
_NON_INVASIVE_LABELS = ("dcis", "normal_acinus_or_duct")
_NON_EPITHELIUM_LABELS = (
    "stroma", "lymphocytic_infiltrate", "necrosis_or_debris", "glandular_secretions",
    "blood", "metaplasia_NOS", "fat", "plasma_cells", "other_immune_infiltrate",
    "mucoid_material", "lymphatics", "nerve", "skin_adnexa", "blood_vessel", "other",
)

TO_CLASS: dict[int, int] = {
    **{GT_CODES[name]: INVASIVE for name in _INVASIVE_LABELS},
    **{GT_CODES[name]: NON_INVASIVE for name in _NON_INVASIVE_LABELS},
    **{GT_CODES[name]: NON_EPITHELIUM for name in _NON_EPITHELIUM_LABELS},
}


def _check_total() -> None:
    """Every code is mapped or explicitly ignored, and nothing is both. Run at import."""
    assigned = set(TO_CLASS) | set(IGNORE_CODES)
    missing = set(GT_CODES.values()) - assigned
    if missing:
        raise AssertionError(
            "these BCSS codes are neither mapped nor ignored: "
            + ", ".join(f"{code} ({CODE_NAMES[code]})" for code in sorted(missing))
        )
    if both := set(TO_CLASS) & set(IGNORE_CODES):
        raise AssertionError(f"codes both mapped and ignored: {sorted(both)}")
    if TO_CLASS[GT_CODES["tumor"]] == TO_CLASS[GT_CODES["dcis"]]:
        raise AssertionError(
            "tumor and dcis have landed in the same class. That is the 5-class merge "
            "this project cannot accept - it erases the invasive-versus-in-situ boundary "
            "the whole model exists to draw."
        )


_check_total()

#: Code to class as a lookup table, so remapping is one fancy-index rather than 22
#: comparisons per pixel. Unmapped codes give `IGNORE`.
_LUT = np.full(256, IGNORE, dtype=np.uint8)
for _code, _cls in TO_CLASS.items():
    _LUT[_code] = _cls


def remap(mask: np.ndarray) -> np.ndarray:
    """A raw BCSS-coded mask to our three classes, with `IGNORE` where there is no label.

    Raises on a code the release should not contain. A mask carrying a code above 21 is
    not the raw 22-class BCSS release - most likely the merged 5-class version, which
    folds `dcis` into `tumor` and cannot be used here - and finding that out now beats
    finding it out from a confusion matrix three scripts later.
    """
    raw = np.asarray(mask)
    if raw.ndim == 3:
        # Some mirrors ship masks as RGB with the code triplicated. Take one channel,
        # after checking they really are identical.
        if not (
            np.array_equal(raw[..., 0], raw[..., 1])
            and np.array_equal(raw[..., 0], raw[..., 2])
        ):
            raise ValueError(
                "this mask has three different channels, so it is a colour rendering of "
                "the labels rather than the labels themselves."
            )
        raw = raw[..., 0]

    unknown = [int(v) for v in np.unique(raw) if int(v) not in CODE_NAMES]
    if unknown:
        raise ValueError(
            f"mask contains codes {unknown}, which are not in BCSS's 22. This is not the "
            "raw release - most likely the merged 5-class version."
        )
    return _LUT[raw.astype(np.uint8)]


# --- where a region came from -------------------------------------------------

#: The two label sources, kept apart for the whole pipeline. `bcss` masks were drawn by
#: pathologists; `bracs_dcis` masks are BEETLE's predictions, reviewed by nobody. They are
#: indistinguishable once exported, so this string is the only thing that knows, and
#: `segreport.py` never averages the two together.
SOURCE_BCSS = "bcss"
SOURCE_DCIS = "bracs_dcis"

#: Which classes each source is allowed to teach.
#:
#: **BCSS teaches all three.** 151 TCGA regions with pixel masks drawn by pathologists,
#: abundant in invasive carcinoma and in stroma, fat and the rest.
#:
#: **BRACS teaches in-situ only.** Its regions were chosen *because* they are DCIS-heavy,
#: so in-situ epithelium is the one thing it represents well - and it does: BRACS supplies
#: **98.8 %** of this export's in-situ pixels, 89.5 M against BCSS's 1.09 M. Its other two
#: classes are a side effect of that selection, and they are not small: 58.0 M
#: non-epithelium pixels and 16.0 M invasive ones, **all of them BEETLE's predictions
#: rather than anybody's judgement**, against 385 M and 267 M drawn by hand in BCSS. A
#: DCIS-selected, model-labelled minority should not get a vote on what invasive carcinoma
#: looks like; it is a biased sample of the class its labeller is worst at.
#:
#: There is no "conflict resolution" to do beyond this, because the two sources are
#: *different images*: no pixel carries both a human's label and BEETLE's, so there is
#: never a disagreement to arbitrate. The only question is which source is authoritative
#: for which class, and this table is the answer.
SOURCE_CLASSES: dict[str, frozenset[int]] = {
    SOURCE_BCSS: frozenset({NON_EPITHELIUM, NON_INVASIVE, INVASIVE}),
    SOURCE_DCIS: frozenset({NON_INVASIVE}),
}


def apply_source_authority(
    mask: np.ndarray,
    source: str,
    *,
    allowed: dict[str, frozenset[int]] | None = None,
) -> np.ndarray:
    """Blank the pixels whose source is not authoritative for their class.

    **The pixel-level form of this rule is strictly better than the tile-level one**, and
    the difference is worth understanding. A tile classifier has to drop a whole tile whose
    label it distrusts, losing everything else on it. Here the tile stays: a BRACS tile
    holding a duct surrounded by stroma keeps teaching in-situ epithelium, and its stroma
    pixels simply stop being a statement about stroma. They become `IGNORE`, which the loss
    already skips (`ignore_index`), `segreport` already excludes from every count, and
    `segtrain`'s Dice already zeroes out. Nothing new had to be built to absorb them -
    which is the sign this is the right place to express the rule.

    Note what this is *not*: it is not saying those pixels are unlabelled in the data. It
    is saying **this project does not accept BEETLE's opinion about them**, which is a
    judgement about label provenance rather than about the slide, and so belongs at load
    time where it can be turned off (`--all-sources`) and compared.

    A source nobody has ruled on keeps every class, so adding a third dataset later fails
    open rather than silently blanking all of it.
    """
    table = SOURCE_CLASSES if allowed is None else allowed
    permitted = table.get(source)
    if permitted is None:
        return mask

    out = np.asarray(mask).copy()
    for label in range(len(CLASS_NAMES)):
        if label not in permitted:
            out[out == label] = IGNORE
    return out


def authoritative_pixels(row: dict) -> dict[str, int]:
    """A manifest row's per-class pixel counts, after the authority rule.

    Used wherever counts drive a decision - class weights, the census, the report's
    denominators - because weights computed over pixels that no longer supervise would be
    balancing against supervision that is not there.
    """
    permitted = SOURCE_CLASSES.get(row.get("source", ""), frozenset(range(len(CLASS_NAMES))))
    return {
        name: (int(row[f"px_{name}"]) if index in permitted else 0)
        for index, name in enumerate(CLASS_NAMES)
    }


#: BCSS's published held-out tissue source sites. Holding out hospitals rather than
#: regions is the closest this dataset offers to "a slide from somewhere we have never
#: seen": two regions from one hospital share a scanner, a stain protocol and a fixation
#: habit.
TEST_INSTITUTIONS: frozenset[str] = frozenset({"OL", "LL", "E2", "EW", "GM", "S3"})

#: BRACS regions have no institution. Deliberately not two letters - a two-character code
#: here would collide with a TCGA tissue source site by coincidence.
DCIS_INSTITUTION = "BRACS"

_BARCODE = re.compile(r"^(TCGA)-([0-9A-Z]{2})-([0-9A-Z]{4})", re.IGNORECASE)


def parse_bcss(name: str) -> tuple[str, str]:
    """`(slide_id, institution)` from a BCSS filename. Raises rather than guessing.

    Every split downstream groups on this, so a wrong guess would produce a leak that no
    later check could see.
    """
    match = _BARCODE.match(Path(name).stem)
    if not match:
        raise ValueError(
            f"cannot read a TCGA barcode from {name!r}. BCSS regions are named "
            "TCGA-XX-YYYY-DX1_xmin..._ymin..., and the grouping for every split comes "
            "from that - so this is refused rather than guessed at."
        )
    _, institution, participant = match.groups()
    return f"TCGA-{institution.upper()}-{participant.upper()}", institution.upper()


def parse_dcis(name: str) -> tuple[str, str]:
    """`(patient, institution)` from a BRACS filename: `BRACS_1247_DCIS_1` -> `BRACS_1247`.

    Grouped by patient rather than region because 42 of BRACS's 665 training regions come
    from case 1247 alone, and four near-identical fields of one duct on both sides of a
    split would report memorisation as generalisation.
    """
    parts = Path(name).stem.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"cannot read a patient from {name!r}; expected something like "
            "BRACS_1247_DCIS_1, because every split groups by patient."
        )
    return "_".join(parts[:2]), DCIS_INSTITUTION


def slide_key(roi_id: str) -> str:
    """`TCGA-A1-A0SK-DX1` - the key `slide_mpp.json` uses.

    Not `slide_id` plus `"-DX1"`: the section number is part of the identity and is not
    always 1. Guessing would look up the wrong slide's resolution on any case with two
    sections, and be right often enough that nobody would notice.
    """
    head = roi_id.split("_")[0]
    parts = head.split("-")
    section = next((p for p in parts if p.upper().startswith("DX")), None)
    if len(parts) < 3 or section is None:
        raise ValueError(
            f"cannot read a slide key from {roi_id!r}; expected TCGA-A1-A0SK-DX1_xmin..."
        )
    return f"{parts[0]}-{parts[1]}-{parts[2]}-{section}".upper()
