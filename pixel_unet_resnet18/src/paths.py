"""Where everything is. The only module that knows a path.

**This folder imports no code from `bcss_bracs_hchannel_resnet18/`.** That is a
deliberate break from how `moco_init_resnet18` and `beetle_teacher_resnet18` were built,
which both put approach 1's `src/` on `sys.path` and import its transforms. Those folders
are *variations* on approach 1 - same tiles, same splits, same head - so sharing code is
right for them. This one is a different model with a different output, and it must be
runnable, publishable and movable without approach 1 being present or correct. Both of
those sibling folders have since been retired; this one still runs, which is the
independence being described paying for itself.

What is shared is the **data**, read-only:

    bcss_bracs_hchannel_resnet18/data/bcss/{images,masks}/    150 regions, human masks
    bcss_bracs_hchannel_resnet18/data/dcis/{images,masks}/    BRACS DCIS, BEETLE masks
    bcss_bracs_hchannel_resnet18/data/normal/{images,masks}/  BRACS normal, BEETLE masks
    bcss_bracs_hchannel_resnet18/data/ic/{images,masks}/      BRACS IC, BEETLE masks

Nothing here writes into those directories, and nothing here reads approach 1's
`data/tiles/` - this pipeline cuts its own tiles into `pixel_unet_resnet18/data/`, because
approach 1's tiles carry no masks and were filtered by a majority vote this model must not
inherit.

The cost of independence is that the H-channel transform is reimplemented in `hstain.py`
rather than imported, and a reimplemented transform is exactly what approach 1's exporter
warns about in its own docstring:

    "A second exporter is exactly where a stray `skimage.color.rgb2hed` gets introduced,
    and the failure looks like a model problem for a week."

That warning is taken seriously rather than ignored. `tests/test_parity.py` imports
approach 1 **in the test only** and asserts this folder's transform is bit-for-bit
identical to it, skipping if approach 1 is absent. So the runtime code has no dependency
and the equivalence is still proven on any machine that has both.
"""

from __future__ import annotations

import os
from pathlib import Path

#: `pixel_unet_resnet18/`.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: The directory holding every approach folder side by side.
WORKSPACE_ROOT: Path = PROJECT_ROOT.parent

# --- shared inputs, read-only -------------------------------------------------

#: Approach 1's data directory. Read for its regions and masks; never written to, and its
#: `tiles/` are deliberately not read.
#:
#: Renamed from `bcss_bracs_hchannel_resnet18` once BRACS became a second source of regions
#: rather than a borrowed afterthought. The first candidate that exists wins, so a clone
#: with either layout works and neither name is asserted as the only one.
SHARED_DATA: Path = next(
    (
        candidate
        for candidate in (
            WORKSPACE_ROOT / "bcss_bracs_hchannel_resnet18" / "data",
            WORKSPACE_ROOT / "bcss_bracs_hchannel_resnet18" / "data",
        )
        if candidate.is_dir()
    ),
    WORKSPACE_ROOT / "bcss_bracs_hchannel_resnet18" / "data",
)

#: 150 BCSS regions with pathologist-drawn pixel masks, at ~0.25 um/px.
BCSS_IMAGES: Path = SHARED_DATA / "bcss" / "images"
BCSS_MASKS: Path = SHARED_DATA / "bcss" / "masks"

#: The per-slide measured resolution, written by approach 1's download. **Read rather
#: than assumed**: every BCSS filename ends `MPP-0.2500` and the slides are not 0.25 -
#: the first measures 0.2521 - so a fixed factor would put every tile at the wrong
#: physical size, and a 224 px tile is only defensible as a claim about microns.
BCSS_SLIDE_MPP: Path = SHARED_DATA / "bcss" / "slide_mpp.json"

#: 120 BRACS regions with BEETLE-drawn masks, already at 0.5 um/px. Model-generated
#: labels; `dcis_manifest.json` records that, and `classes.py` keeps the two sources
#: distinguishable for the whole of this pipeline.
DCIS_IMAGES: Path = SHARED_DATA / "dcis" / "images"
DCIS_MASKS: Path = SHARED_DATA / "dcis" / "masks"
DCIS_MANIFEST: Path = SHARED_DATA / "dcis" / "dcis_manifest.json"

#: The ImageNet backbone, vendored by the demo rather than downloaded at run time. Not
#: approach 1's directory - it belongs to the demo's model store, which is where a
#: pinned checkpoint has to live so that the trained and the served model are one file.
PRETRAINED: Path = (
    WORKSPACE_ROOT / "Breast_Cancer_IHC_Tissue_Scoring_Demo"
    / "models" / "tissue_type" / "pretrained"
)
MODELS_DIR: Path = (
    WORKSPACE_ROOT / "Breast_Cancer_IHC_Tissue_Scoring_Demo" / "models" / "tissue_type"
)

# --- our own outputs ----------------------------------------------------------

DATA_DIR: Path = PROJECT_ROOT / "data"

#: `(image, mask)` tile pairs. Our own, not approach 1's: its tiles have no masks and
#: were filtered by the majority vote whose discards are the boundary tiles this model
#: most needs.
TILES_DIR: Path = DATA_DIR / "seg_tiles"

#: One JSON shard per exported region. The export is idempotent because of these: a
#: region with a shard whose fingerprint matches the current settings is skipped, and the
#: combined manifest is rebuilt from the shards rather than accumulated in memory. An
#: hour-long export that dies at minute fifty then costs ten minutes, not fifty.
SHARDS_DIR: Path = TILES_DIR / "shards"

CHECKPOINTS_DIR: Path = DATA_DIR / "checkpoints"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


def torch_threads() -> int:
    """Threads for training. One less than the machine has, so it stays usable."""
    return max(1, (os.cpu_count() or 4) - 1)


def ensure_dirs() -> None:
    for directory in (DATA_DIR, TILES_DIR, SHARDS_DIR, CHECKPOINTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def check_inputs() -> dict[str, bool]:
    """What is present. Reported per item so a missing half names itself."""
    return {
        "bcss_images": BCSS_IMAGES.is_dir(),
        "bcss_masks": BCSS_MASKS.is_dir(),
        "bcss_slide_mpp": BCSS_SLIDE_MPP.is_file(),
        "dcis_images": DCIS_IMAGES.is_dir(),
        "dcis_masks": DCIS_MASKS.is_dir(),
        "imagenet_weights": (PRETRAINED / "resnet18-imagenet.pth").is_file(),
    }


def require_inputs() -> None:
    missing = [name for name, ok in check_inputs().items() if not ok]
    if missing:
        raise RuntimeError(
            f"these inputs are missing: {missing}. This pipeline reads "
            f"{SHARED_DATA} for its regions and masks and "
            f"{PRETRAINED} for the backbone; it writes only into {DATA_DIR}."
        )
