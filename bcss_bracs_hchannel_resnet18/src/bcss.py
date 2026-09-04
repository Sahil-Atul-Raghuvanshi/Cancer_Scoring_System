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

#: The `source` value the exporter stamps on every borrowed tile.
DCIS_SOURCE = "bracs_dcis"

#: The `institution` value they carry. Deliberately not two letters: `Region.is_test`
#: and `datasets.institution_split` both read that field, and a two-character code here
#: would land BRACS tiles in a TCGA tissue source site by coincidence.
DCIS_INSTITUTION = "BRACS"

#: Share of BRACS *patients* held out, never mixed into training.
#:
#: Held out by patient rather than by region because 42 of the 665 regions come from
#: case 1247 alone; a region-level split would put four near-identical fields of one
#: duct on both sides and report memorisation as generalisation.
DCIS_TEST_FRACTION = 0.25


def dcis_test_cases(case_ids) -> frozenset[str]:
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
        if draw < DCIS_TEST_FRACTION:
            held.add(case)
    return frozenset(held)


def find_dcis_regions(root: Path) -> list[Region]:
    """The BRACS/BEETLE regions, paired, in the same `Region` shape as BCSS's.

    `parse_roi` is deliberately not used: it demands a TCGA barcode and would raise on
    `BRACS_1247_DCIS_1`. The grouping key is the patient, `BRACS_1247`, taken off the
    first two underscore-separated fields - the same rule
    `bracs_roi_to_mask_using_beetle` uses, and the reason its filenames were left in BRACS's own
    format rather than being rewritten to look like TCGA barcodes. A borrowed tile that
    looks native is a borrowed tile that ends up in the wrong split.
    """
    root = Path(root)
    images_dir, masks_dir = root / "images", root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            f"no images/ and masks/ under {root}. Run "
            "bracs_roi_to_mask_using_beetle/scripts/export_to_approach1.py to build them."
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

        parts = image.stem.split("_")
        if len(parts) < 2:
            raise ValueError(
                f"cannot read a patient from {image.name!r}; expected something like "
                "BRACS_1247_DCIS_1, because every split groups by patient."
            )
        regions.append(
            Region(
                roi_id=image.stem,
                slide_id="_".join(parts[:2]),
                institution=DCIS_INSTITUTION,
                image=image,
                mask=mask,
            )
        )

    if orphans:
        raise FileNotFoundError(
            f"{len(orphans)} DCIS image(s) have no mask, e.g. {orphans[:3]}. Re-run the "
            "exporter rather than training on the half that arrived."
        )
    if not regions:
        raise FileNotFoundError(f"no DCIS regions found under {root}")
    return regions
