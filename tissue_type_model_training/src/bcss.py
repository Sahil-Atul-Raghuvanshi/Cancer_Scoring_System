"""BCSS: the label codes, the three classes we collapse them into, and the splits.

BCSS is 151 regions of interest from 151 TCGA-BRCA slides, released **CC0**, with a
pixel label per pixel drawn by pathologists, residents and medical students through
the Digital Slide Archive. It is the whole label source for this model.

**The trap this module exists to make impossible.** There is a widely-used "5-class"
version of BCSS - tumour / stroma / inflammatory / necrosis / other - and it merges
`dcis` into `tumor`. That is the one merge this project cannot accept: separating
in-situ carcinoma from invasive carcinoma is the entire point of the region model,
because only invasive tumour is scored (guide Rule 5). The merged version is the
default in most tutorials and in TIAToolbox's `fcn_resnet50_unet-bcss`. So the raw
22-code map is written out in full below, `TO_CLASS` is asserted total, and a unit
test checks that codes 1 and 20 land in different classes. A comment would not have
survived contact with a tutorial; a failing test will.

Three classes, not eight, and that is a modelling decision rather than laziness. Fat,
stroma, inflammation and necrosis do not need telling apart from each other - they
are all excluded from the score - so collapsing them puts every training example
behind the one boundary that matters.

Amgad M, Elfandy H, ... Cooper LAD. Structured crowdsourcing enables convolutional
segmentation of histology images. Bioinformatics 35(18):3461-3467 (2019).
Codes verified against meta/gtruth_codes.tsv of
CancerDataScience/CrowdsourcingDataset-Amgadetal2019.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- the raw label codes, verbatim from gtruth_codes.tsv -----------------------

#: Every code the raw masks use, by name. All 22, written out rather than parsed, so
#: the mapping below can be read and checked against the TSV by eye. Notebook 01
#: re-reads the TSV and asserts this dictionary still matches it - the file is the
#: source of truth, this is the copy the code compiles against.
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

#: Class index to the name used in every report, manifest and overlay. Ordered by
#: index so `CLASS_NAMES[label]` is the label's name and a confusion matrix's axes
#: cannot be mislabelled.
CLASS_NAMES: tuple[str, str, str] = (
    "non_epithelium",
    "non_invasive_epithelium",
    "invasive_epithelium",
)

#: What each class means, for the report's own legend.
CLASS_MEANING: dict[int, str] = {
    NON_EPITHELIUM: "stroma, fat, inflammation, necrosis, vessels - excluded (Rule 1)",
    NON_INVASIVE: "DCIS, LCIS, normal ducts and lobules - excluded (Rule 5)",
    INVASIVE: "invasive carcinoma - the only thing scored",
}

#: Codes that are **not a class** and must carry zero weight. The BCSS authors are
#: explicit that zero pixels lie outside the annotated region; folding them into
#: `non_epithelium` would teach the model that unannotated background is stroma and
#: inflate every accuracy figure. `exclude` and `undetermined` are the annotators
#: saying they could not tell, which is not a label either.
IGNORE_CODES: frozenset[int] = frozenset(
    {GT_CODES["outside_roi"], GT_CODES["exclude"], GT_CODES["undetermined"]}
)

#: The mapping, by name so it can be checked against the paper's own vocabulary.
_INVASIVE_LABELS = ("tumor", "angioinvasion")
_NON_INVASIVE_LABELS = ("dcis", "normal_acinus_or_duct")
_NON_EPITHELIUM_LABELS = (
    "stroma",
    "lymphocytic_infiltrate",
    "necrosis_or_debris",
    "glandular_secretions",
    "blood",
    "metaplasia_NOS",
    "fat",
    "plasma_cells",
    "other_immune_infiltrate",
    "mucoid_material",
    "lymphatics",
    "nerve",
    "skin_adnexa",
    "blood_vessel",
    "other",
)

TO_CLASS: dict[int, int] = {
    **{GT_CODES[name]: INVASIVE for name in _INVASIVE_LABELS},
    **{GT_CODES[name]: NON_INVASIVE for name in _NON_INVASIVE_LABELS},
    **{GT_CODES[name]: NON_EPITHELIUM for name in _NON_EPITHELIUM_LABELS},
}

#: Sentinel for "no class" in the remapped mask. 255 rather than -1 so the remapped
#: mask stays uint8, and far from any real class so an off-by-one cannot land on it.
IGNORE = 255


def _check_total() -> None:
    """Every code is either mapped or explicitly ignored, and nothing is both.

    Run at import. A code that is neither would be silently dropped by the lookup
    table below and its pixels would count as ignored - which is a plausible-looking
    tile export that quietly threw away a tissue type.
    """
    assigned = set(TO_CLASS) | set(IGNORE_CODES)
    missing = set(GT_CODES.values()) - assigned
    if missing:
        raise AssertionError(
            "these BCSS codes are neither mapped to a class nor ignored: "
            + ", ".join(f"{code} ({CODE_NAMES[code]})" for code in sorted(missing))
        )
    both = set(TO_CLASS) & set(IGNORE_CODES)
    if both:
        raise AssertionError(f"codes both mapped and ignored: {sorted(both)}")
    if TO_CLASS[GT_CODES["tumor"]] == TO_CLASS[GT_CODES["dcis"]]:
        raise AssertionError(
            "tumor and dcis have landed in the same class. That is the 5-class merge "
            "this project cannot accept - it erases the invasive-versus-in-situ "
            "boundary the whole region model exists to draw."
        )


_check_total()

#: Code to class as a lookup table, so remapping a mask is one fancy-index rather
#: than 22 comparisons per pixel. Index by the raw code; unmapped codes give IGNORE.
_LUT = np.full(256, IGNORE, dtype=np.uint8)
for _code, _cls in TO_CLASS.items():
    _LUT[_code] = _cls


def remap(mask: np.ndarray) -> np.ndarray:
    """A raw BCSS mask to our three classes, with `IGNORE` where there is no label.

    Raises on a code the dataset should not contain. That is the release check doing
    its job: a mask carrying a code above 21 is not the raw BCSS release, and finding
    out here beats finding out from a confusion matrix three notebooks later.
    """
    raw = np.asarray(mask)
    if raw.ndim == 3:
        # Some mirrors ship the masks as RGB PNGs with the code triplicated across
        # channels. Take one channel, after checking they really are identical.
        if not (np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 0], raw[..., 2])):
            raise ValueError(
                "this mask has three different channels, so it is a colour rendering "
                "of the labels rather than the labels themselves. The exporter needs "
                "the code-valued mask - re-download from a source that ships it."
            )
        raw = raw[..., 0]

    present = np.unique(raw)
    unknown = [int(v) for v in present if int(v) not in CODE_NAMES]
    if unknown:
        raise ValueError(
            f"mask contains codes {unknown}, which are not in gtruth_codes.tsv. This "
            "is not the raw 22-class BCSS release - most likely the merged 5-class "
            "version, which folds dcis into tumor and cannot be used here."
        )

    return _LUT[raw.astype(np.uint8)]


# --- which slide, which hospital ----------------------------------------------

#: BCSS region filenames are TCGA barcodes with the crop appended, e.g.
#: `TCGA-A2-A0YE-DX1_xmin45749_ymin25055_MPP-0.2500.png`. Fields 2 and 3 are the
#: tissue source site and the participant.
_BARCODE = re.compile(r"^(TCGA)-([0-9A-Z]{2})-([0-9A-Z]{4})", re.IGNORECASE)

#: The published held-out institutions. BCSS's own convention is to test on whole
#: tissue source sites rather than on random regions, which is the right unit: two
#: regions from one hospital share a scanner, a stain protocol and a fixation habit,
#: so a split that separates regions but not hospitals measures memory rather than
#: generalisation. Roughly 108 training and 43 test regions.
TEST_INSTITUTIONS: frozenset[str] = frozenset({"OL", "LL", "E2", "EW", "GM", "S3"})


@dataclass(frozen=True)
class Region:
    """One BCSS region of interest, and the two groupings a split may use."""

    roi_id: str
    #: `TCGA-A2-A0YE` - the case. Every tile from one case stays on one side of a
    #: split, because tiles from one slide are near-duplicates of each other.
    slide_id: str
    #: `A2` - the tissue source site, i.e. the hospital.
    institution: str
    image: Path
    mask: Path

    @property
    def is_test(self) -> bool:
        """Whether this region is in the published held-out set."""
        return self.institution.upper() in TEST_INSTITUTIONS

    @property
    def slide_key(self) -> str:
        """`TCGA-A1-A0SK-DX1` - the key `meta/roiBounds.csv` and the mpp sidecar use.

        Not `slide_id` plus `"-DX1"`. The section number is part of the identity and it
        is not always 1; guessing it would look up the wrong slide's resolution on any
        case with two sections, and be right often enough that nobody would notice.
        """
        head = self.roi_id.split("_")[0]
        parts = head.split("-")
        section = next((part for part in parts if part.upper().startswith("DX")), None)
        if len(parts) < 3 or section is None:
            raise ValueError(
                f"cannot read a slide key from {self.roi_id!r}; expected a name like "
                "TCGA-A1-A0SK-DX1_xmin..._ymin..."
            )
        return f"{parts[0]}-{parts[1]}-{parts[2]}-{section}".upper()


def parse_roi(name: str) -> tuple[str, str]:
    """`(slide_id, institution)` from a BCSS filename.

    Raises rather than guessing. A filename this cannot parse means the download is
    laid out differently from the release, and every split downstream depends on
    grouping being right - so a wrong guess here would produce a leak that no later
    check could see.
    """
    match = _BARCODE.match(Path(name).stem)
    if not match:
        raise ValueError(
            f"cannot read a TCGA barcode from {name!r}. BCSS regions are named "
            "TCGA-XX-YYYY-DX1_xmin..._ymin..., and the grouping for every split is "
            "read from that - so this is refused rather than guessed at."
        )
    _, institution, participant = match.groups()
    return f"TCGA-{institution.upper()}-{participant.upper()}", institution.upper()


def locate_pairs(root: Path) -> tuple[Path, Path]:
    """Find the `images/` and `masks/` directories under a BCSS download.

    Searched for rather than assumed, because the download routes disagree about the
    layout. The Google Drive folder nests everything one level down, under
    `0_Public-data-Amgad2019_0.25MPP/`, while the girder script writes `images/` and
    `masks/` directly into its save path. Both are the same dataset, and neither is
    worth making a caller think about.

    Only the root and its immediate children are searched. A deeper walk would find
    an `images/` directory belonging to something else entirely - a report's figures,
    say - and quietly train on it.
    """
    root = Path(root)
    candidates = [root]
    if root.is_dir():
        candidates += sorted(entry for entry in root.iterdir() if entry.is_dir())

    for candidate in candidates:
        images, masks = candidate / "images", candidate / "masks"
        if images.is_dir() and masks.is_dir():
            return images, masks

    raise FileNotFoundError(
        f"no images/ and masks/ pair under {root} or its immediate subdirectories. "
        "Run notebook 01 (or scripts/01_download.py) to download BCSS."
    )


def find_regions(root: Path) -> list[Region]:
    """Every region under a BCSS download, image paired with mask.

    An image with no mask is reported rather than skipped: a partial download that
    silently trains on two thirds of the dataset is worse than one that stops.
    """
    images_dir, masks_dir = locate_pairs(root)

    regions: list[Region] = []
    orphans: list[str] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
            continue
        mask = masks_dir / image.name
        if not mask.exists():
            candidates = list(masks_dir.glob(image.stem + ".*"))
            if not candidates:
                orphans.append(image.name)
                continue
            mask = candidates[0]

        slide_id, institution = parse_roi(image.name)
        regions.append(
            Region(
                roi_id=image.stem,
                slide_id=slide_id,
                institution=institution,
                image=image,
                mask=mask,
            )
        )

    if orphans:
        raise FileNotFoundError(
            f"{len(orphans)} image(s) have no matching mask, e.g. {orphans[:3]}. "
            "The download is incomplete; finish it rather than training on the part "
            "that arrived."
        )
    if not regions:
        raise FileNotFoundError(f"no BCSS regions found under {root}")

    return regions


# --- the borrowed in-situ class ------------------------------------------------
#
# BCSS carries nine `dcis` tiles in the whole 151-region release, seven of them from one
# patient, and none at all in the six held-out institutions. A three-class model cannot
# be fitted on that and, worse, cannot be *measured* on it: class 1 recall on the
# published split is 0/0, not a low number but no number.
#
# So class 1 is borrowed from BRACS, labelled by BEETLE's released nnU-Net. Everything
# below exists to keep that borrowing honest, because the tiles themselves are
# indistinguishable from BCSS's once exported - same 224 px, same uint8, same directory.
# The distinction lives in the manifest's `source` column and in the two rules here.

#: The `institution` value every borrowed tile carries, whatever its lesion.
#: Deliberately not two letters: `Region.is_test` and `datasets.institution_split` both
#: read that field, and a two-character code here would land BRACS tiles in a TCGA
#: tissue source site by coincidence.
#:
#: One value across all three trees on purpose. It answers "who scanned this", and the
#: answer is the same Italian centre for a DCIS, an IC and a normal region alike; what
#: differs between them is the *lesion*, and that is `source`, below.
BRACS_INSTITUTION = "BRACS"

#: Share of BRACS *patients* held out, never mixed into training.
#:
#: Held out by patient rather than by region because 42 of the 665 regions come from
#: case 1247 alone; a region-level split would put four near-identical fields of one
#: duct on both sides and report memorisation as generalisation.
#:
#: Hashed over the union of all three trees, never per tree: a patient can contribute a
#: DCIS region and an IC region, and a per-tree draw would put the same person on both
#: sides of the split. `bracs_test_cases` hashes each id independently, so the union is
#: the same assignment as any subset of it.
BRACS_TEST_FRACTION = 0.25


@dataclass(frozen=True)
class BorrowedTree:
    """One BRACS lesion type as it arrives here: its directory, its `source`, its remit.

    The mirror of `tissue_label_generation`'s `config.RoiTree`, which is what
    writes these trees. Two tables rather than one because they answer two different
    questions - that one decides which regions to segment and what the mask on disk may
    claim; this one decides which tiles may vote on which class, and what the export
    gate checks. They meet at `source`, and `02_export.py` asserts that `teaches` here
    equals the `teaches_classes` the producing manifest recorded, so the two cannot
    drift apart in silence.

    **`teaches` is the authority rule, not a filter on pixels.** BRACS labels a region
    by its predominant lesion, by three-pathologist consensus; BEETLE says where within
    it. Where the two agree the tile has two independent sources behind it, and where
    they disagree nobody has adjudicated which is wrong - so the tile is dropped rather
    than assigned to whichever source is convenient. `datasets.by_source_authority` is
    that drop, and `datasets.SOURCE_CLASSES` is built from this column.

    **Why three trees and not one pooled `bracs`.** A DCIS consensus corroborates
    BEETLE's in-situ and contradicts its invasive; an IC consensus does the opposite; a
    normal consensus licenses both of its non-carcinoma answers and never asks it to
    separate invasive from in-situ at all. Pooling them would give one `source` string
    one authority, which is exactly the distinction being drawn.
    """

    #: The `data/` subdirectory, and `--include-<name>` on `02_export.py`.
    name: str
    #: The `source` value stamped on every tile cut from this tree, and the key
    #: `datasets.SOURCE_CLASSES` looks up.
    source: str
    #: The classes this tree's consensus corroborates, so the only ones its tiles vote
    #: on. Checked against the producing manifest's `teaches_classes`.
    teaches: frozenset[int]
    #: The lesion as a person says it, for the export log.
    lesion: str
    #: G2d: the combined tile share of `gate_classes` must clear `gate_min_share`.
    #:
    #: The mirror image of the G2a gate on BCSS, catching the same fault from the other
    #: side. These are regions a consensus of three pathologists annotated as this
    #: lesion, so a share near zero is not a rare class - it is the teacher's label
    #: codes, the resolution, or the class mapping being wrong.
    gate_classes: frozenset[int]
    gate_min_share: float
    #: How the gate's subject reads in its failure message. Written rather than
    #: assembled from `CLASS_NAMES`, which would produce worse English than a sentence.
    gate_reads: str
    #: How to read a patient off an image filename, because every split groups by
    #: patient and a wrong answer is a leak nothing downstream can detect.
    #:
    #: `"bracs_prefix"` takes the first two underscore-separated fields, so
    #: `BRACS_1247_DCIS_1` groups under `BRACS_1247` - 42 of BRACS's 665 training
    #: regions are case 1247's alone, and grouping by region would put four
    #: near-identical fields of one duct on both sides of the split.
    #:
    #: `"whole_stem"` makes each image its own case. That is not a weaker version of the
    #: same rule, it is the correct rule for BACH: the release publishes no patient
    #: grouping, each of its 400 images is an independent field, and there is therefore
    #: no patient that could span the split. Stated rather than inferred from the
    #: filename's shape, so a later source whose names happen to contain an underscore
    #: cannot silently inherit BRACS's rule.
    case_rule: str = "bracs_prefix"
    #: The `institution` stamped on this tree's tiles. Deliberately not two letters:
    #: `Region.is_test` and `datasets.institution_split` both read that field and a
    #: two-character code would land these tiles in a TCGA tissue source site by
    #: coincidence. BACH is a different laboratory from BRACS and saying so is the whole
    #: point of having it - a manifest that called Porto's images "BRACS" would be a
    #: false provenance claim in the one column that records provenance.
    institution: str = "BRACS"


#: The BRACS lesion trees this project borrows from, keyed by directory name.
#:
#: **Why there are three and what each is for.** BCSS ships nine `dcis` tiles in its
#: entire 151-region release, so class 1 cannot be fitted or measured on it - `dcis` is
#: the borrow that fixes that. But once class 1 is 1,843 BRACS tiles against BCSS's 9,
#: `source` becomes very nearly a synonym for class 1, and `06_leakage_check.py` reports
#: INCONCLUSIVE rather than a pass: a model can score on in-situ by recognising which
#: dataset it is looking at. `ic` and `normal` are the correction. BRACS supplying
#: invasive carcinoma as well means dataset identity stops predicting the class, and
#: BRACS supplying consensus-normal stroma is what teaches "BRACS-looking connective
#: tissue is not DCIS" - which is the measured failure: on `CAN_00251_26_H&E` the v2
#: tile model called 45.9% of the section in-situ at 0.809 mean confidence, and for
#: 88.5% of those windows the runner-up class was non-epithelium.
BORROWED_TREES: dict[str, BorrowedTree] = {
    "dcis": BorrowedTree(
        name="dcis",
        source="bracs_dcis",
        teaches=frozenset({1}),
        lesion="DCIS",
        gate_classes=frozenset({1}),
        # Measured at 0.61 mean / 0.60 median area share over the 120 exported regions,
        # the lowest region being 0.20. A floor of 0.20 on the tile share is therefore
        # well below anything a correct run produces and well above a mapping fault.
        gate_min_share=0.20,
        gate_reads="in-situ epithelium",
    ),
    "ic": BorrowedTree(
        name="ic",
        source="bracs_ic",
        teaches=frozenset({2}),
        lesion="invasive carcinoma",
        gate_classes=frozenset({2}),
        # Lower than DCIS's floor on purpose, and not because the gate is weaker. An IC
        # region of interest is drawn around a tumour but carries its stroma, its
        # inflammation and often an adjacent in-situ focus with it, so the invasive
        # *share* of an IC region is honestly smaller than the in-situ share of a DCIS
        # region. The fault this catches - wrong codes, wrong resolution - puts the
        # share near zero, not near a fifth.
        gate_min_share=0.15,
        gate_reads="invasive epithelium",
    ),
    "normal": BorrowedTree(
        name="normal",
        source="bracs_normal",
        teaches=frozenset({0, 1}),
        lesion="normal",
        # Both corroborated classes together, which is the whole region: a consensus of
        # "no carcinoma here" says class 2 should be absent, and says nothing about how
        # the rest divides between stroma and duct. Gating on class 1 alone would fire
        # on a legitimate field of pure stroma.
        gate_classes=frozenset({0, 1}),
        # Measured at 0.9962 mean / 0.9995 median area share over the 150 exported
        # regions, the lowest region being 0.9304. A floor of 0.90 clears every one of
        # them and still fails loudly if invasive starts appearing in regions three
        # pathologists agreed contain no carcinoma.
        gate_min_share=0.90,
        gate_reads="non-epithelium plus normal duct",
    ),
}

#: BACH (ICIAR 2018) - a **second laboratory**, and the only thing here that attacks the
#: cause rather than the symptom.
#:
#: In-situ carcinoma reaches this project through BRACS alone: BCSS ships nine `dcis`
#: tiles in its entire release, so class 1 is ~99.7% one Italian centre's crops. Every
#: in-situ number the model reports is therefore largely *agreement with BEETLE*, and
#: the measured consequence is confident nonsense on a third laboratory - 45.9% of
#: `CAN_00251_26_H&E` called in-situ at 0.809 mean confidence, the runner-up class being
#: non-epithelium in 88.5% of those windows. The `ic` and `normal` trees break
#: `source == class 1` and cut that field by about 40%, but they cannot fix its cause:
#: after them, class 1 is *still* ~99.7% BRACS, and the residual false in-situ is still
#: firing on stroma 81.7% of the time.
#:
#: BACH is 400 microscopy images from Porto - a different scanner, a different
#: laboratory - labelled per image by two pathologists. That per-image label is the same
#: kind of object as a BRACS ROI consensus, so it obeys the same rule: BEETLE says
#: where, the label says what is corroborated, disagreement drops the image. `Benign` is
#: deliberately absent for the reason BRACS's four atypia categories are - it has no
#: home in a three-class ontology.
#:
#: **LICENCE: CC BY-NC-ND 4.0 - NoDerivatives**, stricter than BEETLE's ShareAlike and
#: than BRACS's non-commercial. Whether a checkpoint fitted on these is a derivative
#: work is a legal question nobody here has answered. See
#: `tissue_label_generation/data/bach/LICENCE_NOTE.md`. These trees exist to
#: *measure*; a model trained on them must not be published on that basis alone.
BORROWED_TREES.update({
    "bach_insitu": BorrowedTree(
        name="bach_insitu",
        source="bach_insitu",
        case_rule="whole_stem",
        institution="BACH",
        teaches=frozenset({1}),
        lesion="BACH in-situ carcinoma",
        gate_classes=frozenset({1}),
        # Looser than BRACS DCIS's 0.20. A BACH image is a fixed 2048x1536 field chosen
        # to contain the lesion, not a region drawn around it, so more of the frame is
        # ordinary breast tissue. The fault this catches still puts the share near zero.
        gate_min_share=0.10,
        gate_reads="in-situ epithelium",
    ),
    "bach_invasive": BorrowedTree(
        name="bach_invasive",
        source="bach_invasive",
        case_rule="whole_stem",
        institution="BACH",
        teaches=frozenset({2}),
        lesion="BACH invasive carcinoma",
        gate_classes=frozenset({2}),
        gate_min_share=0.10,
        gate_reads="invasive epithelium",
    ),
    "bach_normal": BorrowedTree(
        name="bach_normal",
        source="bach_normal",
        case_rule="whole_stem",
        institution="BACH",
        teaches=frozenset({0, 1}),
        lesion="BACH normal",
        gate_classes=frozenset({0, 1}),
        gate_min_share=0.90,
        gate_reads="non-epithelium plus normal duct",
    ),
})

#: Every `source` value that means "a region, mask written by BEETLE, reviewed by
#: nobody". The complement of `"bcss"`, and the test every report uses to keep
#: human-drawn labels apart from model-generated ones.
BORROWED_SOURCES: frozenset[str] = frozenset(
    tree.source for tree in BORROWED_TREES.values()
)

#: Reverse lookup, for a manifest row that records only its `source`.
BORROWED_BY_SOURCE: dict[str, BorrowedTree] = {
    tree.source: tree for tree in BORROWED_TREES.values()
}

#: The `source` value stamped on BRACS's borrowed in-situ tiles. Kept as a name of its
#: own because the BRACS DCIS tree has a relationship to class 1 the others do not - it
#: was the only source of it before BACH, which is what `datasets.real_dcis_slides` and the
#: `holdout_real_dcis` default are about. Anything asking "is this row borrowed at all"
#: wants `is_borrowed`, not this.
DCIS_SOURCE = BORROWED_TREES["dcis"].source


def is_borrowed(source: str | None) -> bool:
    """Is this tile's label BEETLE's opinion rather than a person's?

    The one question every split and every report needs answered, and it must not be
    spelled `source != "bcss"` at each call site: a fourth source added later would
    silently become "borrowed" everywhere it is asked that way. An unknown source is
    *not* borrowed here, which fails the same way `by_source_authority` does - open,
    and visibly, because such a row then has to survive a report that says its labels
    were drawn by people.
    """
    return source in BORROWED_SOURCES


def bracs_test_cases(case_ids) -> frozenset[str]:
    """Which BRACS patients are held out, decided by hash rather than by shuffle.

    A `random.sample` with a seed would be reproducible only as long as the *set* of
    cases never changes; add ten more patients and every previous assignment moves, so
    a model trained last week and one trained today would be scored on different tiles
    while both claim the same seed. Hashing each case id independently means a patient's
    side of the split is a property of that patient and nothing else, and a later export
    that adds cases leaves the existing assignment untouched.
    """
    import hashlib

    held = set()
    for case in set(case_ids):
        digest = hashlib.sha256(case.encode("utf-8")).digest()
        # First four bytes as a fraction of the range - a uniform draw in [0, 1) that
        # depends only on this case id.
        draw = int.from_bytes(digest[:4], "big") / 2**32
        if draw < BRACS_TEST_FRACTION:
            held.add(case)
    return frozenset(held)


def find_borrowed_regions(root: Path, *, case_rule: str | None = None) -> list[Region]:
    """The BRACS/BEETLE regions of one tree, paired, in the same `Region` shape as BCSS's.

    One reader for all three lesion trees, because they differ in nothing this function
    looks at: `data/dcis/`, `data/ic/` and `data/normal/` each hold `images/` and
    `masks/` of identically named PNGs, written by the same script at the same spacing.
    What differs between them is which classes their tiles may vote on, and that is
    `BORROWED_TREES`, decided by the caller from the directory it passed in - not
    guessable from a filename, and deliberately not guessed here.

    `parse_roi` is deliberately not used: it demands a TCGA barcode and would raise on
    `BRACS_1247_DCIS_1`. The grouping key is the patient, `BRACS_1247`, taken off the
    first two underscore-separated fields - the same rule
    `tissue_label_generation` uses, and the reason its filenames were left in BRACS's own
    format rather than being rewritten to look like TCGA barcodes. A borrowed tile that
    looks native is a borrowed tile that ends up in the wrong split.
    """
    root = Path(root)
    # The tree named by the directory decides how its filenames are read. Looked up
    # rather than passed by every caller, so the answer cannot differ between the
    # exporter and the determinism gate.
    tree = BORROWED_TREES.get(root.name)
    if case_rule is None:
        case_rule = tree.case_rule if tree else "bracs_prefix"
    if case_rule not in ("bracs_prefix", "whole_stem"):
        raise ValueError(f"unknown case_rule {case_rule!r}")
    institution = tree.institution if tree else BRACS_INSTITUTION
    images_dir, masks_dir = root / "images", root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            f"no images/ and masks/ under {root}. Run\n"
            f"  tissue_label_generation/scripts/export_to_approach1.py "
            f"--roi-type {root.name}\n"
            "which segments BRACS regions with BEETLE and writes the pairs."
        )

    regions: list[Region] = []
    orphans: list[str] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in {".png", ".tif", ".tiff"}:
            continue
        mask = masks_dir / image.name
        if not mask.exists():
            orphans.append(image.name)
            continue

        if case_rule == "whole_stem":
            slide_id = image.stem
        else:
            parts = image.stem.split("_")
            if len(parts) < 2:
                raise ValueError(
                    f"cannot read a patient from {image.name!r} under the "
                    f"`bracs_prefix` rule; expected something like BRACS_1247_DCIS_1, "
                    "because every split groups by patient. A source that names images "
                    "individually wants `case_rule=\"whole_stem\"` on its "
                    "`BorrowedTree` - see that field."
                )
            slide_id = "_".join(parts[:2])
        regions.append(
            Region(
                roi_id=image.stem,
                slide_id=slide_id,
                institution=institution,
                image=image,
                mask=mask,
            )
        )

    if orphans:
        raise FileNotFoundError(
            f"{len(orphans)} borrowed image(s) under {root.name} have no mask, e.g. "
            f"{orphans[:3]}. Re-run the "
            "exporter rather than training on the half that arrived."
        )
    if not regions:
        raise FileNotFoundError(f"no borrowed regions found under {root}")
    return regions
