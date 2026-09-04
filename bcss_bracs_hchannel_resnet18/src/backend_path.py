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

#: This file is `<repo>/bcss_bracs_hchannel_resnet18/src/backend_path.py`, so the
#: backend is two directories up and across. Resolved rather than relative, because a
#: notebook's working directory is wherever Jupyter was started and not where the
#: notebook lives.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT: Path = PROJECT_ROOT.parent
DEMO_ROOT: Path = WORKSPACE_ROOT / "Breast_Cancer_IHC_Tissue_Scoring_Demo"
BACKEND_ROOT: Path = DEMO_ROOT / "backend"

#: Where this plan's artefacts live. Exported here so no notebook builds these paths
#: by hand - `data/tiles` written by one notebook and `data/tile` read by the next is
#: a fifteen-minute bug that a constant makes impossible.
DATA_DIR: Path = PROJECT_ROOT / "data"
BCSS_DIR: Path = DATA_DIR / "bcss"

#: The in-situ class BCSS does not have, written by
#: `bracs_roi_to_mask_using_beetle/scripts/export_to_approach1.py`: BRACS regions of interest
#: with masks predicted by BEETLE's released nnU-Net and remapped into BCSS's own label
#: codes. Same `images/` and `masks/` layout as `BCSS_DIR`, so `bcss.find_dcis_regions`
#: is a thin wrapper rather than a second reader.
#:
#: **These labels are model-generated and nobody has reviewed them.** BCSS's are drawn
#: by people. That difference does not show up anywhere in the tile files, so it is
#: carried in the manifest's `source` column instead, and every split and every report
#: keeps the two apart on the strength of it.
DCIS_DIR: Path = DATA_DIR / "dcis"

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
IC_DIR: Path = DATA_DIR / "ic"

TILES_DIR: Path = DATA_DIR / "tiles"
FEATURES_DIR: Path = DATA_DIR / "features"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

#: The one directory checkpoints are published to, shared with the API. Step 8 will
#: load from here, so the trained model and the served model are the same file.
MODELS_DIR: Path = DEMO_ROOT / "models" / "tissue_type"
PRETRAINED_DIR: Path = MODELS_DIR / "pretrained"

#: Slides the demo has ingested. Used only by the smoke test, which checks that a
#: notebook and the API compute the same H channel on the same real tile.
SLIDES_DIR: Path = DEMO_ROOT / "data" / "slides"


def _install() -> None:
    """Prepend the backend to `sys.path`, once, and fail loudly if it is not there."""
    if not (BACKEND_ROOT / "app" / "common" / "stains.py").exists():
        raise RuntimeError(
            f"the demo backend is not where this module expects it: {BACKEND_ROOT}. "
            "Everything here imports the pipeline's own deconvolution rather than "
            "reimplementing it, so there is nothing useful to do without it. Check "
            "that this folder still sits beside Breast_Cancer_IHC_Tissue_Scoring_Demo."
        )

    path = str(BACKEND_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_dirs() -> None:
    """Create the directories the notebooks write into. Idempotent."""
    for directory in (DATA_DIR, BCSS_DIR, TILES_DIR, FEATURES_DIR, REPORTS_DIR,
                      MODELS_DIR, PRETRAINED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


_install()
