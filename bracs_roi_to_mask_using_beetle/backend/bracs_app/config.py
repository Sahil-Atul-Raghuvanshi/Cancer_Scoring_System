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

#: `bracs_roi_to_mask_using_beetle/`.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: The directory holding every approach folder side by side.
WORKSPACE_ROOT: Path = PROJECT_ROOT.parent

#: Approach 1's tree. Renamed from `bcss_bracs_hchannel_resnet18` once BRACS became a second
#: source of regions rather than a borrowed afterthought.
APPROACH1_ROOT: Path = WORKSPACE_ROOT / "bcss_bracs_hchannel_resnet18"
APPROACH1_SRC: Path = APPROACH1_ROOT / "src"

#: Approach 4a's tree, if it is still on this machine. It held `regions.py` and the
#: teacher archive; the archive has since moved under the demo's own `models/` (see
#: `MODEL_ZIP`) and this folder may be absent entirely. Nothing needed to harvest
#: regions reads it - only the six-slide comparison does, through
#: `install_beetle_path`, which raises a sentence naming this constant when it is gone.
BEETLE_ROOT: Path = WORKSPACE_ROOT / "beetle_teacher_resnet18"

#: Approach 4a's `src/`, for `regions.py` alone. That module already knows how to grid a
#: whole slide, mask its tissue and score every window by nuclear density at 2 um/px -
#: which is exactly the "pick the most nucleus-dense field" rule the six-slide comparison
#: needs, and writing a second one would be a second definition of "where the interesting
#: tissue is".
BEETLE_SRC: Path = BEETLE_ROOT / "src"

#: Where the six clinical H&E whole slides live. Not `DEMO_ROOT/data/slides`, whose
#: uploads are all the same IHC file - the same distinction `beetle_teacher_resnet18`
#: draws, for the same reason.
HE_IMAGES_DIR: Path = (
    WORKSPACE_ROOT / "Breast_Cancer_IHC_Tissue_Scoring_Demo" / "images"
)

#: The published tile classifier this app compares BEETLE against.
MODELS_DIR: Path = (
    WORKSPACE_ROOT / "Breast_Cancer_IHC_Tissue_Scoring_Demo" / "models" / "tissue_type"
)

#: The demo backend, which approach 1's `hchannel` imports `app.common.imaging` from.
#: **This is why this app's own package is `bracs_app` and not `app`.** Two packages
#: called `app` on one `sys.path` resolve to whichever was imported first, and the
#: symptom is `ModuleNotFoundError: No module named 'app.common'` raised from inside a
#: file that has nothing to do with the collision.
DEMO_BACKEND: Path = WORKSPACE_ROOT / "Breast_Cancer_IHC_Tissue_Scoring_Demo" / "backend"

#: The 1.9 GB nnU-Net ensemble. Read straight out of the zip - unpacking it would put
#: a second 1.9 GB on the disk for no gain, and `zipfile` streams a member fine.
#:
#: **Under the demo's `models/`, not approach 4a's `data/`.** It moved there when
#: `beetle_teacher_resnet18/` was retired, which puts every model this project runs in
#: one place beside `tissue_type/` and `grandqc/`. The first candidate that exists wins,
#: so a machine that still has the old layout keeps working; `teacher.read_archive`
#: raises naming all of them when none is present.
MODEL_ZIP_CANDIDATES: tuple[Path, ...] = (
    WORKSPACE_ROOT / "Breast_Cancer_IHC_Tissue_Scoring_Demo" / "models" / "beetle" / "model.zip",
    BEETLE_ROOT / "data" / "model" / "model.zip",
)


def _first_existing(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # None on disk: return the preferred one so the error message names where it
    # *should* be rather than where it used to be.
    return candidates[0]


MODEL_ZIP: Path = _first_existing(MODEL_ZIP_CANDIDATES)

#: The single directory inside `model.zip`, named for the trainer that produced it.
MODEL_MEMBER_ROOT: str = (
    "model/nnUNetTrainer_WSD_wei_i0_nnunet_aug_json__"
    "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d/"
)

DATA_DIR: Path = PROJECT_ROOT / "data"
#: The BRACS DCIS regions of interest, by split. `train` is what the UI lists first.
DCIS_DIR: Path = DATA_DIR / "dcis"
#: The BRACS invasive-carcinoma regions of interest, same `<split>/*.png` layout.
IC_DIR: Path = DATA_DIR / "ic"


@dataclass(frozen=True)
class RoiTree:
    """One BRACS lesion type: where its PNGs live, and what the teacher may say there.

    **`teaches` is the whole point of this table and it is not a filter on pixels - it
    is a statement about which of BEETLE's opinions this lesion type corroborates.**

    BRACS labels a region by its predominant lesion, by three-pathologist consensus.
    That is a human label, and it is the strongest thing in the pipeline; BEETLE's
    per-pixel argmax is a model's opinion two steps from a pathologist. Where the two
    agree the tile has two independent sources behind it. Where they disagree, one of
    them is wrong and nobody has adjudicated which - so the tile is discarded rather
    than assigned to whichever source is more convenient.

    On a DCIS region that means BEETLE's in-situ is kept and its invasive is dropped,
    which is exactly the rule `datasets.SOURCE_CLASSES` already enforces one repository
    downstream - v1 shipped BRACS's 270 DCIS-region "invasive" tiles and scored 0.19
    Dice on them, which is what a training set full of unadjudicated disagreement looks
    like from the outside. On an IC region the same rule points the other way: BEETLE's
    invasive is corroborated, and its in-situ is probably adjacent DCIS that the ROI
    label does not cover.

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
    #: Below this area share of `compare_name`, the region is flagged. `0.0` disables
    #: the check, which is right for a normal region - see the note above.
    flag_when_below: float = 0.05

    @property
    def directory(self) -> Path:
        return DATA_DIR / self.name


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
        non_invasive_code="normal_acinus_or_duct",
        # A normal region may legitimately be all stroma and fat with no duct in it, so
        # "the teacher found almost no normal duct" is not a smoke alarm here.
        flag_when_below=0.0,
    ),
}

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

#: One directory per image a user uploaded through the app, holding the same artefacts.
#:
#: **Deliberately not under `RUNS_DIR`.** Everything that reads `RUNS_DIR` treats what
#: it finds there as a BRACS DCIS region: `compare.compare_all` pools its tiles into the
#: BRACS side of the stain comparison, `/api/results` lists it as a harvested region, and
#: `scripts/export_to_approach1.py` copies its tiles into approach 1's training set. An
#: arbitrary image a user dropped on the page is none of those things - it has no BRACS
#: annotation saying it contains in-situ carcinoma, and pooling it would corrupt the one
#: measurement this app exists to make. Uploads therefore live in their own tree, are not
#: tile-exported, and never appear in a number quoted about BRACS.
UPLOADS_DIR: Path = DATA_DIR / "uploads"

#: One directory per clinical case compared on the six-slide screen. A sibling of
#: `RUNS_DIR` and `UPLOADS_DIR` for the reason spelled out above: a region cut from one of
#: our own H&E slides is not a BRACS DCIS region, carries no annotation saying it contains
#: in-situ carcinoma, and must never be pooled into the stain comparison or copied into
#: approach 1's training set.
SIXSLIDES_DIR: Path = DATA_DIR / "sixslides"

#: The BRACS test whole slides and their QuPath annotations - the only pathologist-drawn
#: ground truth in the project. `annotations/test/test/Group_*/Type_*/<case>.qpdata`
#: mirrors the FTP tree; `qpdata.parse` reads them without QuPath.
#:
#: **May be absent.** It is the only pathologist-drawn ground truth in the project and
#: it is not needed to harvest regions, so its absence is a skip rather than an error -
#: `test_the_real_bracs_annotations_parse_to_the_expected_lesions` skips on it, and the
#: whole-slide screen reports it missing rather than raising.
BRACS_WSI_DIR: Path = WORKSPACE_ROOT / "Testing_H&E_images_BRACS"

#: One directory per whole slide run: the four pictures, the label grids, the manifest.
WSI_DIR: Path = DATA_DIR / "wsi"

#: The field compared on that screen, in pixels at `TEACHER_MPP`.
#:
#: 2048 px at 0.5 um/px is 1.024 mm - the same figure `beetle_teacher_resnet18`'s sampler
#: uses, and not a coincidence: it is 4x4 of the teacher's 512 px patches and 9x9 of the
#: tile model's 224 px tiles, so neither model is asked to work on a partial window. A
#: whole slide at this spacing is ~400 MPx and about an hour per fold, which is why this
#: is a region and not a slide.
COMPARE_REGION_PX: int = 2048

#: The largest upload accepted, in pixels. The teacher holds region-sized float32
#: accumulators - five classes plus an importance map, so roughly 24 bytes per working
#: pixel - and the browser is handed a PNG of the result. 120 MPx at 0.25 um/px is a
#: 2.7 cm square of tissue, larger than any BRACS region of interest and larger than
#: anything worth waiting an hour for on a CPU. Above it the request is refused with a
#: number rather than accepted and killed by the OOM killer twenty minutes later.
MAX_UPLOAD_PIXELS: int = 120_000_000

#: The largest upload accepted, in bytes, checked while the body is still being read.
MAX_UPLOAD_BYTES: int = 400 * 1024 * 1024

FRONTEND_DIR: Path = PROJECT_ROOT / "frontend"

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
            "approach 1's exporter rather than a copy of it, so bcss_bracs_hchannel_resnet18/ "
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


def install_beetle_path() -> None:
    """Additionally put approach 4a's `src/` on `sys.path`, for `regions.py`.

    Separate from `install_approach1_path` because only the six-slide comparison needs it
    and it is the heavier import: `regions` pulls in the demo backend's slide reader and
    step 3's tissue mask, so a machine with no whole slides on it can still run every
    other screen in this app.
    """
    install_approach1_path()
    if not (BEETLE_SRC / "regions.py").exists():
        raise RuntimeError(
            f"approach 4a's src is not at {BEETLE_SRC}. The six-slide comparison grids "
            "and scores whole slides with its sampler rather than a copy of it, so "
            "beetle_teacher_resnet18/ must sit beside this folder."
        )
    if str(BEETLE_SRC) not in sys.path:
        sys.path.insert(0, str(BEETLE_SRC))


def ensure_dirs() -> None:
    for directory in (DATA_DIR, RUNS_DIR, UPLOADS_DIR, SIXSLIDES_DIR, WSI_DIR):
        directory.mkdir(parents=True, exist_ok=True)
