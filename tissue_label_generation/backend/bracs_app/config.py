"""Where everything is, and the handful of numbers that are assumptions rather than facts.

Every path is derived from this file's own location, so the app runs from a clone
anywhere as long as the sibling approach folders sit beside it. Nothing is configured by
environment variable: an experiment whose answer depends on a shell variable is an
experiment nobody can reproduce.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

#: `tissue_label_generation/`.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: The directory holding every stage folder side by side, the shared `data/` tree and
#: the per-version `v<N>_data/` trees.
WORKSPACE_ROOT: Path = PROJECT_ROOT.parent

sys.path.append(str(WORKSPACE_ROOT))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402

#: The model-training stage's tree. This folder writes the labelled regions it trains
#: on; it is named for what it does rather than for the architecture it happened to
#: start with.
APPROACH1_ROOT: Path = WORKSPACE_ROOT / "tissue_type_model_training"
APPROACH1_SRC: Path = APPROACH1_ROOT / "src"
#: The training stage's exports (tiles, features), under this version's data tree.
APPROACH1_DATA: Path = data_versions.data_root() / "tissue_type_model_training"

#: The demo backend, which the training stage's `hchannel` imports `app.common.imaging`
#: from. **This is why this app's own package is `bracs_app` and not `app`.** Two
#: packages called `app` on one `sys.path` resolve to whichever was imported first, and
#: the symptom is `ModuleNotFoundError: No module named 'app.common'` raised from inside
#: a file that has nothing to do with the collision.
DEMO_ROOT: Path = WORKSPACE_ROOT / "tissue_scoring_demo"
DEMO_BACKEND: Path = DEMO_ROOT / "backend"

#: The 1.9 GB nnU-Net ensemble. Read straight out of the zip - unpacking it would put
#: a second 1.9 GB on the disk for no gain, and `zipfile` streams a member fine.
#:
#: **Under the active version's `models/` (`data_versions.models_root()`), not approach 4a's `data/`.** It moved there when
#: `beetle_teacher_resnet18/` was retired, which puts every model this project runs in
#: one place beside `tissue_type/` and `grandqc/`. The first candidate that exists wins,
#: so a machine that still has the old layout keeps working; `teacher.read_archive`
#: raises naming all of them when none is present.
MODEL_ZIP_CANDIDATES: tuple[Path, ...] = (
    data_versions.models_root() / "beetle" / "model.zip",
    # The retired approach-4a location, spelled out rather than via a constant: that
    # folder's other paths went with the six-slide screen, and this one line is all that
    # is left of them.
    WORKSPACE_ROOT / "beetle_teacher_resnet18" / "data" / "model" / "model.zip",
)


def _first_existing(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # None on disk: return the preferred one so the error message names where it
    # *should* be rather than where it used to be.
    return candidates[0]


def _first_existing_dir(candidates: tuple[Path, ...]) -> Path:
    """`_first_existing` for directories, with the same fall back to the preferred one."""
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


MODEL_ZIP: Path = _first_existing(MODEL_ZIP_CANDIDATES)

#: The single directory inside `model.zip`, named for the trainer that produced it.
MODEL_MEMBER_ROOT: str = (
    "model/nnUNetTrainer_WSD_wei_i0_nnunet_aug_json__"
    "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d/"
)

#: This stage's own artefacts: the teacher's segmentations in `runs/`, the region trees
#: it writes for the training stage in `regions/`. Everything here is **derived** and
#: re-derivable - `runs/` at 25 s a region, `regions/` in minutes from `runs/`.
#:
#: **Under the version's data tree, not inside this folder.** Every byte the three
#: stages write lives under `<workspace>/v<N>_data/data/`, one subdirectory per
#: producer, so the whole tree can be moved to another disk or deleted without touching
#: a line of code - and so a folder of source code stays a folder of source code.
DATA_DIR: Path = data_versions.data_root() / "tissue_label_generation"

#: **The sources, and they sit apart from everything derived from them.**
#:
#: BRACS's regions of interest, BACH's microscopy images and BCSS's annotated crops are
#: downloads: 43 GB, hours over services that rate-limit, and nothing in this project
#: can regenerate them. Keeping them in the same directory as the exports cut from them
#: meant one folder held both the irreplaceable input and the disposable output - so
#: "clear the exports" and "keep the sources" became the same rmdir with different
#: intentions.
#:
#: So they live at the shared `data/original/`, apart from every version's
#: `v<N>_data/` rather than inside any of them. They are read-only from here: nothing in this app
#: writes into `ORIGINAL_DATA`.
#:
#: **The first candidate that exists wins**, the same rule `MODEL_ZIP_CANDIDATES` uses:
#: a clone that still keeps the sources in the old workspace-level `original_data/`, or
#: under this stage's own `data/`, goes on working, and the layout is discovered rather
#: than asserted.
ORIGINAL_DATA_CANDIDATES: tuple[Path, ...] = (
    data_versions.ORIGINAL_ROOT,
    WORKSPACE_ROOT / "original_data",
    DATA_DIR,
)
ORIGINAL_DATA: Path = next(
    (
        candidate
        for candidate in ORIGINAL_DATA_CANDIDATES
        if (candidate / "bracs").is_dir() or (candidate / "bach").is_dir()
    ),
    ORIGINAL_DATA_CANDIDATES[0],
)

#: The region trees this app writes for approach 1: one `images/` and `masks/` pair per
#: lesion, in BCSS's label codes. **Derived**, so under `DATA_DIR` and not beside the
#: sources - and no longer under approach 1's `data/`, which now holds only the tile
#: exports cut from these.
REGIONS_DIR: Path = DATA_DIR / "regions"

#: The BRACS DCIS regions of interest, by split. `train` is what the UI lists first.
DCIS_DIR: Path = ORIGINAL_DATA / "bracs" / "dcis"
#: The BRACS invasive-carcinoma regions of interest, same `<split>/*.png` layout.
IC_DIR: Path = ORIGINAL_DATA / "bracs" / "ic"


@dataclass(frozen=True)
class RoiTree:
    """One BRACS lesion type: where its PNGs live, and what the teacher may say there.

    **`teaches` is the whole point of this table and it is not a filter on pixels - it
    is a statement about which of BEETLE's opinions this lesion type corroborates.**

    BRACS labels a region by its predominant lesion, by three-pathologist consensus.
    That is a human label, and it is the strongest thing in the pipeline; BEETLE's
    per-pixel argmax is a model's opinion two steps from a pathologist. So `teaches`
    says which class a tile from this tree may carry, and `datasets.SOURCE_CLASSES`
    enforces it one repository downstream - v1 shipped BRACS's 270 DCIS-region
    "invasive" tiles and scored 0.19 Dice on them, which is what a training set full of
    unadjudicated disagreement looks like from the outside.

    **What changed, and it is the important entry in this table: `epithelium_code`.**
    The rule above used to be carried out by *deletion*. BEETLE decided the lesion as
    well as its location, and a region where its verdict contradicted the consensus was
    thrown away whole - one BRACS DCIS region in five, and 47 of 100 BACH in-situ
    images. That is not a neutral filter. A large solid or comedo duct read 112 um at a
    time shows no duct wall, so the teacher calls it invasive, so the region was
    rejected: the filter selected against exactly the morphology the student goes on to
    get wrong, and the measured consequence is a served model that calls 22.4 % of
    held-out DCIS invasive and fills a pathologist's DCIS contour with invasive on
    `CAN_00270_26_H&E`.

    So the disagreement is now *resolved* in the consensus's favour rather than
    deleted. `epithelium_code` writes both of BEETLE's epithelium classes as one code -
    the union is where the epithelium is, and the consensus says which lesion it is. On
    a DCIS region BEETLE's invasive becomes in-situ; on an IC region its in-situ becomes
    invasive; on a normal region either becomes normal duct. `teaches` is unchanged and
    still binding, and it is now unanimous by construction rather than by rejection.

    The cost is stated where it is taken, in `labels.targets_for`: BRACS labels the
    *predominant* lesion, so a DCIS region holding a true focus of invasion now teaches
    that focus as in-situ. That error is bounded by how often the consensus is
    incomplete, and it is the smaller of the two on offer.

    **A normal region corroborates two classes, which is why `teaches` is a set.** A
    consensus of "no carcinoma here" licenses both of BEETLE's non-carcinoma answers -
    its `other` becomes non-epithelium and its normal ducts become class 1 - and asks it
    to decide only epithelium versus not, the axis it is strong on. It never has to
    separate invasive from in-situ, which is the axis that made the DCIS-region invasive
    tiles worthless. That makes `0_N` the safest of the three trees and the one that
    speaks to the measured failure: on `CAN_00251_26_H&E` the in-situ head claimed 45.9%
    of the section at 0.809 confidence, and for 88.5% of those windows the runner-up was
    non-epithelium - it is firing on stroma.

    `compare_name` is the class the contradiction is weighed against, and the one
    `flag_when_below` guards. For DCIS and IC that is the single corroborated class. For
    a normal region it is the *epithelial* member of `teaches`: more invasive than normal
    duct in a region annotated normal is the contradiction worth hearing, where
    "non-epithelium plus normal duct is under 5% of this region" would fire on every
    legitimate field of pure stroma.
    """

    #: `--roi-type` on the command line, and the subdirectory under `data/`.
    name: str
    #: The lesion directory on the BRACS FTP server, under `<split>/`.
    ftp_dir: str
    #: Stamped into every manifest and, downstream, into the tile manifest's `source`
    #: column. `datasets.SOURCE_CLASSES` keys off exactly this string.
    source: str
    #: Our class indices whose BEETLE prediction this lesion type corroborates.
    teaches: frozenset[int]
    #: The same, as `bcss.CLASS_NAMES` spells them, in class order.
    teaches_names: tuple[str, ...]
    #: The class the contradiction is weighed against, and that `flag_when_below`
    #: guards. See the note above on why this is not always `teaches_names[0]`.
    compare_name: str
    #: The class whose BEETLE prediction contradicts the ROI label.
    contradicts_name: str
    #: How a person says the lesion, for the notes. `compare_name` is a field name and
    #: reads like one; "more invasive (40.6%) than in-situ (29.3%)" is the sentence a
    #: pathologist would write, and these notes are read by people.
    teaches_short: str
    contradicts_short: str
    #: The contradicting finding as a noun, for "either the region genuinely contains
    #: ___". Kept separate because the adjective and the noun differ in English:
    #: "invasive" epithelium, but the region contains "invasion".
    contradicts_noun: str
    #: The lesion as it appears in a note. Stated rather than sliced off `ftp_dir`,
    #: which would turn `0_N` into the unreadable "an N region".
    lesion: str
    #: `"a"` or `"an"`, for `lesion`. English, not logic.
    article: str = "a"
    #: Which BCSS in-situ code BEETLE's non-invasive class is written as. `dcis` (20)
    #: for the two carcinoma trees; `normal_acinus_or_duct` (13) for a normal region,
    #: where writing 20 would produce a mask asserting carcinoma a consensus of three
    #: pathologists says is absent. Both land in class 1 via `bcss.remap`, so this
    #: changes no training label - only what the mask on disk claims. See
    #: `labels.BCSS_TARGETS`.
    non_invasive_code: str = "dcis"
    #: The one BCSS code that **both** of BEETLE's epithelium classes are written as
    #: inside this tree's regions - the label rule these trees are now built under.
    #:
    #: `None` restores the older division of labour, where BEETLE decided the lesion as
    #: well as the location and a region whose consensus disagreed was deleted. That
    #: deletion is what this field exists to stop: it removed about one BRACS DCIS
    #: region in five and 47 of 100 BACH in-situ images, and it removed them
    #: non-randomly. A solid or comedo duct read 112 um at a time shows no duct wall, so
    #: the teacher calls it invasive, so the region was rejected - the filter selected
    #: against precisely the morphology the student is failing on.
    #:
    #: So the two sources are asked only what each is good for: BEETLE for `invasive OR
    #: in-situ -> epithelium is here`, the three-pathologist consensus for which lesion
    #: that epithelium is. `dcis` on a DCIS region, `tumor` on an IC one,
    #: `normal_acinus_or_duct` where the consensus is that no carcinoma is present.
    #:
    #: **It must remap into `teaches`, and that is asserted rather than trusted** - in
    #: `pipeline.run_region` for the region being written, and over the whole table in
    #: `tests/test_config.py`. A tree that collapsed its epithelium onto a class its
    #: consensus has no authority for would be manufacturing exactly the unadjudicated
    #: label `teaches` exists to keep out, and `datasets.by_source_authority` would then
    #: silently drop every tile of it one repository downstream.
    #:
    #: Where it is set, `non_invasive_code` no longer reaches the mask. It is kept
    #: alongside rather than replaced because `--split-epithelium` still needs it, and
    #: because it is what `recode_epithelium` reads a cached run's mask back through.
    epithelium_code: str | None = None
    #: Below this area share of `compare_name`, the region is flagged. `0.0` disables
    #: the check, which is right for a normal region - see the note above.
    #:
    #: **Under `epithelium_code` this guards a different and better number, without any
    #: change to the arithmetic.** `class_area_fraction` is measured after the collapse,
    #: so on a DCIS region `non_invasive_epithelium` *is* the epithelium union and
    #: `invasive_epithelium` is empty by construction. The contradiction check below can
    #: therefore no longer fire, and this check stops asking "did the teacher find
    #: in-situ" - a question a solid duct answers wrongly - and starts asking "did the
    #: teacher find epithelium at all", which is the smoke alarm it was always meant to
    #: be: at 5 % it catches a wrong resolution or an unseen stain, and nothing else.
    flag_when_below: float = 0.05
    #: Where the images actually sit under `data/`, when that is not `name`. BACH ships
    #: one directory per class with its own capitalisation (`bach/InSitu`), and renaming
    #: a downloaded tree to suit a table here would mean a re-download is not idempotent.
    data_subdir: str | None = None
    #: What the images are called. BRACS ships PNG, BACH ships TIFF.
    suffix: str = ".png"
    #: Whether the tree is laid out `<dir>/<split>/*` or flat.
    #:
    #: **Every tree is flat now, and BRACS's own splits are not a loss.** They used to
    #: be honoured because they were how the release arrived, but nothing here ever
    #: divided on them: `bcss.bracs_test_cases` holds out 25 % of *patients* by hashing
    #: each patient id, because a BRACS split puts regions of one patient on both sides
    #: and a model scored across that boundary is scored on its own memory. So the
    #: directory layout was carrying a division the pipeline deliberately ignored. The
    #: assignment itself is still on disk, in each tree's `.bracs_splits/`, for anyone
    #: who needs to reconstruct the release's own division.
    has_splits: bool = False
    #: The source resolution in microns per pixel. `None` means "BRACS's, see BRACS_MPP".
    #: BACH states 0.42 um/px in its challenge paper, and unlike BRACS that is published
    #: rather than inferred.
    source_mpp: float | None = None

    @property
    def directory(self) -> Path:
        """Where this tree's *source* images sit - under `ORIGINAL_DATA`, never `DATA_DIR`."""
        return ORIGINAL_DATA / (self.data_subdir or self.name)


#: The lesion types this app knows how to harvest. The four atypia categories BRACS
#: also ships - `1_PB`, `2_UDH`, `3_FEA`, `4_ADH` - are deliberately absent: they have
#: no home in a three-class ontology, so admitting them would mean asking BEETLE to
#: place them on the invasive/in-situ boundary, which is the one judgement its labels
#: are least worth trusting for.
ROI_TREES: dict[str, RoiTree] = {
    "dcis": RoiTree(
        name="dcis",
        ftp_dir="5_DCIS",
        source="bracs_dcis",
        teaches=frozenset({1}),
        teaches_names=("non_invasive_epithelium",),
        compare_name="non_invasive_epithelium",
        contradicts_name="invasive_epithelium",
        teaches_short="in-situ",
        contradicts_short="invasive",
        contradicts_noun="invasion",
        lesion="DCIS",
        article="a",
        data_subdir="bracs/dcis",
        has_splits=False,
        # The consensus is DCIS, so every duct the teacher finds in here is in-situ by
        # the annotation - including the solid and comedo ones it reads as invasive.
        epithelium_code="dcis",
    ),
    "ic": RoiTree(
        name="ic",
        ftp_dir="6_IC",
        source="bracs_ic",
        teaches=frozenset({2}),
        teaches_names=("invasive_epithelium",),
        compare_name="invasive_epithelium",
        contradicts_name="non_invasive_epithelium",
        teaches_short="invasive",
        contradicts_short="in-situ",
        contradicts_noun="adjacent in-situ disease",
        lesion="IC",
        article="an",
        data_subdir="bracs/ic",
        has_splits=False,
        # The mirror: three pathologists called this invasive carcinoma, so the
        # epithelium in it is invasive, including the part the teacher reads as the
        # adjacent in-situ the ROI label does not separately cover.
        epithelium_code="tumor",
    ),
    "normal": RoiTree(
        name="normal",
        ftp_dir="0_N",
        source="bracs_normal",
        teaches=frozenset({0, 1}),
        teaches_names=("non_epithelium", "non_invasive_epithelium"),
        compare_name="non_invasive_epithelium",
        contradicts_name="invasive_epithelium",
        teaches_short="normal duct",
        contradicts_short="invasive",
        contradicts_noun="invasion",
        lesion="normal",
        article="a",
        data_subdir="bracs/normal",
        has_splits=False,
        non_invasive_code="normal_acinus_or_duct",
        # Same rule, and here it is the least contentious of the three: a consensus of
        # "no carcinoma present" says the teacher's invasive prediction is wrong on its
        # own terms, so the duct it found is a duct.
        epithelium_code="normal_acinus_or_duct",
        # A normal region may legitimately be all stroma and fat with no duct in it, so
        # "the teacher found almost no normal duct" is not a smoke alarm here.
        flag_when_below=0.0,
    ),
}

#: BACH (ICIAR 2018), a **second laboratory**, and the whole reason it is here.
#:
#: In-situ carcinoma reaches this project only through BRACS: BCSS ships nine `dcis`
#: tiles in its entire release, so class 1 is ~99.7% one Italian centre's crops. Every
#: in-situ number the pipeline reports is therefore largely *agreement with BEETLE*, and
#: the measured consequence is a model that fires in-situ on stroma from a third
#: laboratory - 45.9% of `CAN_00251_26_H&E` at 0.809 confidence, runner-up
#: non-epithelium in 88.5% of those windows. The `ic` and `normal` trees reduce that
#: symptom by breaking `source == class 1`; they cannot fix its cause, because after
#: them class 1 is still ~99.7% BRACS.
#:
#: BACH is 400 microscopy images from Porto, a different scanner and a different
#: laboratory, labelled Normal / Benign / InSitu / Invasive **per image** by two
#: pathologists. That per-image label is the same kind of object as a BRACS ROI
#: consensus - a human statement about what the field predominantly contains - so it
#: slots into exactly the same rule: BEETLE says where, the label says what is
#: corroborated, and disagreement drops the image rather than being assigned.
#:
#: **Benign is deliberately absent**, for the reason the four BRACS atypia categories
#: are: it has no home in a three-class ontology, and admitting it would mean asking
#: BEETLE to place adenosis and fibroadenoma on the invasive/in-situ boundary.
#:
#: **LICENCE: CC BY-NC-ND 4.0.** NoDerivatives, which is stricter than BEETLE's
#: ShareAlike and than BRACS's non-commercial. Whether a checkpoint fitted on these is a
#: derivative work is a legal question nobody here has answered - see
#: `data/bach/LICENCE_NOTE.md`. These trees exist to *measure* whether a second
#: laboratory's in-situ fixes the confound.
BACH_MPP: float = 0.42

_BACH_COMMON = dict(article="a", has_splits=False, suffix=".tif",
                    source_mpp=BACH_MPP)

ROI_TREES.update({
    "bach_insitu": RoiTree(
        name="bach_insitu",
        data_subdir="bach/InSitu",
        ftp_dir="Photos/InSitu",
        source="bach_insitu",
        teaches=frozenset({1}),
        teaches_names=("non_invasive_epithelium",),
        compare_name="non_invasive_epithelium",
        contradicts_name="invasive_epithelium",
        teaches_short="in-situ",
        contradicts_short="invasive",
        contradicts_noun="invasion",
        lesion="BACH in-situ carcinoma",
        # The tree this rule changes most. BEETLE disagreed with BACH's two pathologists
        # on 47 of these 100 images, every one of them by calling invasive where they
        # said in-situ - so under the old rule nearly half of the second laboratory's
        # in-situ was dropped, and what remained was whatever BEETLE already found
        # familiar. A "second laboratory" filtered down to the first one's taste is not
        # a second laboratory.
        epithelium_code="dcis",
        **_BACH_COMMON,
    ),
    "bach_invasive": RoiTree(
        name="bach_invasive",
        data_subdir="bach/Invasive",
        ftp_dir="Photos/Invasive",
        source="bach_invasive",
        teaches=frozenset({2}),
        teaches_names=("invasive_epithelium",),
        compare_name="invasive_epithelium",
        contradicts_name="non_invasive_epithelium",
        teaches_short="invasive",
        contradicts_short="in-situ",
        contradicts_noun="adjacent in-situ disease",
        lesion="BACH invasive carcinoma",
        epithelium_code="tumor",
        **_BACH_COMMON,
    ),
    "bach_normal": RoiTree(
        name="bach_normal",
        data_subdir="bach/Normal",
        ftp_dir="Photos/Normal",
        source="bach_normal",
        teaches=frozenset({0, 1}),
        teaches_names=("non_epithelium", "non_invasive_epithelium"),
        compare_name="non_invasive_epithelium",
        contradicts_name="invasive_epithelium",
        teaches_short="normal duct",
        contradicts_short="invasive",
        contradicts_noun="invasion",
        lesion="BACH normal",
        non_invasive_code="normal_acinus_or_duct",
        epithelium_code="normal_acinus_or_duct",
        # As for BRACS normal: a normal field may legitimately be all stroma.
        flag_when_below=0.0,
        **_BACH_COMMON,
    ),
})

#: Reverse lookup, for a manifest that records only the source string.
ROI_TREE_BY_SOURCE: dict[str, RoiTree] = {t.source: t for t in ROI_TREES.values()}


def tree_for_roi_id(roi_id: str) -> RoiTree | None:
    """`BRACS_1247_DCIS_1` -> the DCIS tree; `BRACS_1003667_IC_1` -> the IC tree.

    BRACS encodes the lesion type in the third underscore-separated field, so a bare
    roi_id is enough to say which tree it belongs to. `None` for anything that is not a
    BRACS region name - an upload, for instance - so the caller decides what that means
    rather than getting a default it did not ask for.
    """
    parts = roi_id.split("_")
    if len(parts) < 3:
        return None
    key = parts[2].lower()
    return ROI_TREES.get(key)
#: One directory per processed region: its mask, its overlay, its tiles, its manifest.
RUNS_DIR: Path = DATA_DIR / "runs"

#: The BRACS test whole slides and their QuPath annotations - the only pathologist-drawn
#: ground truth in the project. `annotations/test/test/Group_*/Type_*/<case>.qpdata`
#: mirrors the FTP tree; `qpdata.parse` reads them without QuPath.
#:
#: A download like BRACS's regions and BCSS's crops, so it belongs beside them under
#: `ORIGINAL_DATA` rather than loose at the workspace root. The old root location is
#: kept as a second candidate under the same first-one-wins rule as everything else.
#:
#: **May be absent.** It is the only pathologist-drawn ground truth in the project and
#: it is not needed to harvest regions, so its absence is a skip rather than an error -
#: `test_the_real_bracs_annotations_parse_to_the_expected_lesions` skips on it, and the
#: whole-slide screen reports it missing rather than raising.
BRACS_WSI_DIR_CANDIDATES: tuple[Path, ...] = (
    ORIGINAL_DATA / "bracs_wsi",
    WORKSPACE_ROOT / "Testing_H&E_images_BRACS",
)
BRACS_WSI_DIR: Path = _first_existing_dir(BRACS_WSI_DIR_CANDIDATES)

# --- the assumptions ----------------------------------------------------------

#: BRACS's own resolution, in microns per pixel.
#:
#: **This is an assumption and it is the one most likely to be wrong.** BRACS whole
#: slide images were scanned on an Aperio AT2 at 40x, which is 0.25 um/px, and the
#: released regions of interest are crops at native magnification - but unlike BCSS,
#: whose every filename ends `MPP-0.2500`, BRACS ships no resolution in the filename or
#: in a sidecar, so nothing on disk can confirm it.
#:
#: Getting it wrong is not subtle. It sets the physical size of a 224 px tile, and the
#: whole argument for that tile size is that 112 um is about nine cells across - enough
#: to see whether epithelium sits inside a duct or has broken out of one. At half that
#: it cannot contain the architecture, and a model trained on such tiles would be
#: learning a different question from the one BCSS's tiles pose. It also sets what the
#: teacher sees: BEETLE was trained at 0.5 um/px and is handed a 2x downsample on this
#: assumption.
#:
#: `scripts/check_resolution.py` measures it from the images instead of assuming it,
#: by comparing nuclear size against BCSS at its known 0.25, and the UI exposes it as
#: an editable field. The default is stated here so that it is stated once.
BRACS_MPP: float = 0.25

#: What BEETLE was trained at, from `dataset.json`'s `"spacing": 0.5`. Not a choice.
TEACHER_MPP: float = 0.5

#: What approach 1's tiles are cut at, from `export.TileSpec`. Not a choice either; it
#: is restated here only so the two can be compared in one place. That they are equal
#: is a convenience - the region is resampled once, for the teacher, and the exporter
#: then has nothing to do.
TILE_MPP: float = 0.5

#: The teacher's patch size, from `plans.json`'s `patch_size`.
PATCH_PX: int = 512

#: Sliding-window step as a fraction of the patch. nnU-Net's own inference default.
#: Halving the step quadruples the work and buys a smoother seam; this is the value the
#: released model's own reported numbers were produced with, so it is the value here.
TILE_STEP: float = 0.5

#: Threads for the forward pass. Left at the machine's count minus one so the API stays
#: answerable while a region is being segmented - a UI that stops responding for four
#: minutes is indistinguishable from a crashed one.
def torch_threads() -> int:
    import os

    return max(1, (os.cpu_count() or 4) - 1)


def install_approach1_path() -> None:
    """Put approach 1's `src/` and the demo backend on `sys.path`, failing loudly.

    This app imports `bcss` and `export` from approach 1 rather than owning copies, for
    the reason `beetle_teacher_resnet18/src/backend_path.py` states at length: a second
    exporter is where a stray transform gets introduced, and the failure looks like a
    model problem for a week. The whole question being asked here is whether BRACS
    tiles are interchangeable with BCSS tiles, and that question is meaningless if they
    were not cut by the same code.

    The demo backend comes along because approach 1's `hchannel` imports
    `app.common.imaging` from it - the optical-density transform is step 4's, and it is
    shared for the same reason.
    """
    if not (APPROACH1_SRC / "export.py").exists():
        raise RuntimeError(
            f"approach 1's src is not at {APPROACH1_SRC}. This app cuts its tiles with "
            "approach 1's exporter rather than a copy of it, so tissue_type_model_training/ "
            "must sit beside this folder."
        )
    if not (DEMO_BACKEND / "app" / "common" / "imaging.py").exists():
        raise RuntimeError(
            f"the demo backend is not at {DEMO_BACKEND}. Approach 1's hchannel imports "
            "its optical-density transform from there rather than restating it."
        )
    for path in (str(APPROACH1_SRC), str(DEMO_BACKEND)):
        if path not in sys.path:
            sys.path.insert(0, path)


def ensure_dirs() -> None:
    """Create what this app writes into. `ORIGINAL_DATA` is deliberately absent - it is
    read-only, and creating an empty tree there would make a missing download look like
    an empty one."""
    for directory in (DATA_DIR, RUNS_DIR, REGIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
