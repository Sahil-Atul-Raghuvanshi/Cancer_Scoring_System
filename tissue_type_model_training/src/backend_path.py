"""Put the demo backend on `sys.path`, so a notebook can import the real pipeline.

One module, imported for its side effect, and that is the point. The training work
must reach the *same* colour deconvolution the API serves - `app.common.stains` and
`step06_colour_deconvolution.deconvolution.separate` - because a model trained on a
different definition of "haematoxylin channel" from the one it is served scores
nothing like its validation number, and the failure looks like a model problem for a
week before anyone finds the plumbing.

So there is exactly one place that knows where the backend lives, and every notebook
starts with `import backend_path`. A notebook that instead appends to `sys.path` in a
cell is a second answer to the same question, and the second answer is the one that
goes stale.

Import it before anything from `app.*`:

    import backend_path            # noqa: F401 - imported for the side effect
    from app.common.stains import RUIFROK_HDAB
"""

from __future__ import annotations

import sys
from pathlib import Path

#: This file is `<workspace>/tissue_type_model_training/src/backend_path.py`, so the
#: backend is two directories up and across. Resolved rather than relative, because a
#: notebook's working directory is wherever Jupyter was started and not where the
#: notebook lives.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT: Path = PROJECT_ROOT.parent
sys.path.append(str(WORKSPACE_ROOT))  # data_versions.py lives at the workspace root
import data_versions  # noqa: E402

#: This version's produced data - `<workspace>/v<N>_data/data/`.
VERSION_DATA_ROOT: Path = data_versions.data_root()
DEMO_ROOT: Path = WORKSPACE_ROOT / "tissue_scoring_demo"
BACKEND_ROOT: Path = DEMO_ROOT / "backend"

#: Where this stage's artefacts live. Exported here so no notebook builds these paths
#: by hand - `data/tiles` written by one notebook and `data/tile` read by the next is
#: a fifteen-minute bug that a constant makes impossible.
#:
#: **Exports only, and that is now enforceable by looking at it.** Every input this
#: stage reads lives elsewhere: the sources in the shared `data/original/`, and
#: the BEETLE-labelled region trees in `v<N>_data/data/tissue_label_generation/`. What is left
#: here is one directory per field of view, all of it re-cuttable - so "clear it and
#: export again" is a safe sentence, which it was not while the 8 GB BCSS download sat
#: in the same tree as the tiles cut from it.
#:
#: **Under the version's data tree, not inside this folder.** All three stages
#: write into `<workspace>/v<N>_data/data/<stage>/`, so the whole of the project's storage can be
#: moved to another disk or deleted without touching a line of code, and a folder of
#: source code stays a folder of source code.
DATA_DIR: Path = VERSION_DATA_ROOT / "tissue_type_model_training"

#: The BEETLE-labelled region trees - `<tree>/images/` and `<tree>/masks/` in BCSS's own
#: label codes - written by `tissue_label_generation/scripts/export_to_approach1.py`.
#:
#: **In the repository that produces them, not this one.** They are derived twice over
#: (a resample of `data/original`, then a teacher's prediction over it) and they
#: re-derive from that repository's `data/runs/` cache in minutes, so they are neither a
#: source to be preserved here nor an export of this plan. Its `config.REGIONS_DIR` is
#: the same path; the two are separate because this folder imports no code from there,
#: and `tests/test_borrowed_trees.py` compares them rather than either importing the
#: other.
#:
#: First candidate that exists wins, so a clone with the old layout - the trees under
#: this repository's own `data/` - keeps working unchanged.
REGIONS_DIR_CANDIDATES: tuple[Path, ...] = (
    VERSION_DATA_ROOT / "tissue_label_generation" / "regions",
    WORKSPACE_ROOT / "bracs_roi_to_mask_using_beetle" / "data" / "regions",
    DATA_DIR,
)
REGIONS_DIR: Path = next(
    (
        candidate
        for candidate in REGIONS_DIR_CANDIDATES
        if (candidate / "dcis").is_dir() or (candidate / "bach_insitu").is_dir()
    ),
    REGIONS_DIR_CANDIDATES[0],
)

#: The 151 human-annotated BCSS regions, ~8 GB, and **the one input here that is not
#: cheaply rebuildable** - every other tree under `DATA_DIR` is a copy out of
#: `v<N>_data/data/tissue_label_generation/runs/` and re-derives in minutes, while this is an
#: hour of download over a service that rate-limits.
#:
#: So it lives outside `DATA_DIR`, in the shared `data/original/`, beside the
#: other sources rather than among the exports cut from them. `DATA_DIR` is then exactly
#: what its docstring says: this plan's artefacts, all of them regenerable.
#:
#: **The first candidate that exists wins**, which is the same rule
#: `bracs_app.config.MODEL_ZIP_CANDIDATES` uses and for the same reason: a clone that
#: still has the old layout keeps working, and neither location is asserted as the only
#: one. `download_bcss` writes to whichever this resolves to, so a fresh machine fills
#: the new location and an existing one is left alone.
BCSS_DIR_CANDIDATES: tuple[Path, ...] = (
    data_versions.ORIGINAL_ROOT / "bcss",
    WORKSPACE_ROOT / "original_data" / "bcss",
    DATA_DIR / "bcss",
)
BCSS_DIR: Path = next(
    (candidate for candidate in BCSS_DIR_CANDIDATES if candidate.is_dir()),
    BCSS_DIR_CANDIDATES[0],
)

#: The in-situ class BCSS does not have, written by
#: `tissue_label_generation/scripts/export_to_approach1.py`: BRACS regions of interest
#: with masks predicted by BEETLE's released nnU-Net and remapped into BCSS's own label
#: codes. Same `images/` and `masks/` layout as `BCSS_DIR`, so `bcss.find_dcis_regions`
#: is a thin wrapper rather than a second reader.
#:
#: **These labels are model-generated and nobody has reviewed them.** BCSS's are drawn
#: by people. That difference does not show up anywhere in the tile files, so it is
#: carried in the manifest's `source` column instead, and every split and every report
#: keeps the two apart on the strength of it.
DCIS_DIR: Path = REGIONS_DIR / "dcis"

#: Invasive carcinoma borrowed from the same place, by the same script with
#: `--roi-type ic`. Identical layout, identical caveat about who drew the labels - and a
#: **separate directory and a separate `source` column value on purpose.**
#:
#: BRACS annotates a region by its predominant lesion. On a DCIS region that consensus
#: corroborates BEETLE's in-situ prediction and contradicts its invasive one; on an IC
#: region it does the opposite. Two trees is what lets `datasets.SOURCE_CLASSES` say so
#: - `bracs_dcis -> {1}`, `bracs_ic -> {2}` - and one pooled directory would collapse
#: the distinction that rule exists to draw.
#:
#: **What this tree is for.** With in-situ coming only from BRACS (1,843 tiles against
#: BCSS's 9), `source` and `class 1` are nearly the same question, and
#: `scripts/06_leakage_check.py` reports that as INCONCLUSIVE rather than a pass.
#: BRACS supplying invasive as well is what breaks it: dataset identity stops predicting
#: the class. That is the measurement to re-run after this tree is added, and it is the
#: reason to add it.
IC_DIR: Path = REGIONS_DIR / "ic"

#: The consensus-normal regions, by the same script with `--roi-type normal`. Same
#: layout, same caveat about who drew the labels, and a third `source` column value.
#:
#: **What this tree is for, and why it is not just more data.** It is the one that
#: targets a measured failure rather than a gap in a census. On `CAN_00251_26_H&E` the
#: v2 tile model called 45.9% of the section in-situ epithelium - 141 mm2 - at a mean
#: confidence of 0.809, and for 88.5% of those windows the runner-up class was
#: non-epithelium: the in-situ head is firing on STROMA, confidently, on a third
#: laboratory's slide. The cause is the confound `IC_DIR` describes, seen from the other
#: end - the head has learned "looks like a BRACS crop", and BRACS-like connective
#: tissue clears it.
#:
#: A region three pathologists agree contains no carcinoma is the correction, and it is
#: the safest of the three trees: BEETLE is asked only for epithelium versus not, the
#: axis it is strong on, and never to adjudicate invasive against in-situ. Its stroma
#: and fat become class 0 and its normal ducts class 1, both under that consensus -
#: which is what teaches "BRACS-looking stroma is not DCIS". Hence
#: `bcss.BORROWED_TREES["normal"].teaches == {0, 1}`, the only tree that teaches two.
NORMAL_DIR: Path = REGIONS_DIR / "normal"


#: BACH's three trees, written by `export_to_approach1.py --roi-type bach_*`. A second
#: laboratory's tissue, which is the point - see `bcss.BORROWED_TREES` for what that
#: buys and for the CC BY-NC-ND licence that comes with it.
BACH_INSITU_DIR: Path = REGIONS_DIR / "bach_insitu"
BACH_INVASIVE_DIR: Path = REGIONS_DIR / "bach_invasive"
BACH_NORMAL_DIR: Path = REGIONS_DIR / "bach_normal"


def borrowed_dir(name: str) -> Path:
    """The directory holding one borrowed lesion tree, by its `BORROWED_TREES` key.

    Derived rather than looked up in a second table: the tree's name *is* the directory
    name, decided by `export_to_approach1.py --roi-type`, and a mapping here would be a
    second place for that to be true. The constants above stay because they are what the
    prose above hangs off, and because a caller naming `DCIS_DIR` reads better than one
    naming `borrowed_dir("dcis")`.
    """
    return REGIONS_DIR / name


TILES_DIR: Path = DATA_DIR / "tiles"
FEATURES_DIR: Path = DATA_DIR / "features"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

#: The one directory checkpoints are published to, shared with the API. Step 8 will
#: load from here, so the trained model and the served model are the same file.
MODELS_DIR: Path = data_versions.models_root() / "tissue_type"
PRETRAINED_DIR: Path = MODELS_DIR / "pretrained"


def publish_dir() -> Path:
    """Where to WRITE a new checkpoint - this version's own models, never inherited ones."""
    return data_versions.publish_models_root() / "tissue_type"

#: Slides the demo has ingested. Used only by the smoke test, which checks that a
#: notebook and the API compute the same H channel on the same real tile.
SLIDES_DIR: Path = VERSION_DATA_ROOT / "demo" / "slides"


def _install() -> None:
    """Prepend the backend to `sys.path`, once, and fail loudly if it is not there."""
    if not (BACKEND_ROOT / "app" / "common" / "stains.py").exists():
        raise RuntimeError(
            f"the demo backend is not where this module expects it: {BACKEND_ROOT}. "
            "Everything here imports the pipeline's own deconvolution rather than "
            "reimplementing it, so there is nothing useful to do without it. Check "
            "that this folder still sits beside tissue_scoring_demo."
        )

    path = str(BACKEND_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_dirs() -> None:
    """Create the directories the notebooks write into. Idempotent."""
    # `BCSS_DIR` is deliberately included: a fresh machine needs it to exist before
    # `download_bcss` writes into it, and it resolves to `data/original/bcss` unless a
    # clone still carries the legacy `data/bcss` - in which case that one already
    # exists and this is a no-op. It cannot resurrect the old layout, because the
    # legacy candidate is only ever chosen when it is already a directory.
    for directory in (DATA_DIR, BCSS_DIR, TILES_DIR, FEATURES_DIR, REPORTS_DIR,
                      MODELS_DIR, PRETRAINED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


_install()
