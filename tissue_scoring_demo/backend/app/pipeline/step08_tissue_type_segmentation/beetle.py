"""BEETLE's released nnU-Net, run per window, per pixel, and on its own terms.

This is step 8's third branch and it is **independent of the two ResNet18 branches in
every way that matters**. It loads no checkpoint of ours, imports neither `model.py` nor
`classes.py` nor `input.py`, declares no manifest, and reports its own five classes
rather than our three. The only thing it shares with the rest of step 8 is *geometry* -
where on the slide to look, which is step 7's answer and not a model's - and that
sharing is named where it happens, in `pixels.py`.

Why the separation is worth enforcing rather than merely observing: the ResNet branches
are a fitted head whose input contract, class order, polarity, standardisation and
decision threshold all have to be verified against a manifest at load, because a
mismatch there is invisible. BEETLE has none of those degrees of freedom - it is a
published network with a published preprocessing rule - so routing it through machinery
built to check a manifest it does not have would mean inventing one, and an invented
manifest is a claim nobody made. It gets its own path instead.

What this module has to get right is not a model but four things around one:

--------------------------------------------------------------------------------
1. The physical scale, which is the one preprocessing error that looks like a bad model
--------------------------------------------------------------------------------

`dataset.json` records `spacing: 0.5`, in microns per pixel. That is not a suggestion
and it is not a resolution this step may choose: a network shown tissue at 2 um/px sees
structures a quarter the size it was fitted on, and it does not fail - it returns a
plausible, confident, wrong segmentation. So **every window is resampled to 0.5 um/px
before the network sees it**, whatever field of view step 7 chose, and `read_archive`
refuses an archive whose recorded spacing is not the one this module resamples to.

That is what makes the four fields of view a real comparison rather than four
resolutions: at every one of them the network is shown the same physical scale, and what
changes is only **how much slide is in view at once** - 112, 224, 448 or 672 um.
`window_px` is that conversion and it is the whole geometry of the branch.

Contrast the ResNet branches, where a field of view is a 224 px window read at four
different resolutions: there the model sees a constant pixel count at four physical
scales, here a constant physical scale at four extents. The two are not comparable
window for window, and the report says so rather than putting the numbers side by side.

--------------------------------------------------------------------------------
2. Preprocessing, which is division by 255 and nothing else
--------------------------------------------------------------------------------

`plans.json` names the normalisation `RGBTo01Normalization` and `dataset.json` names all
three channels `rgb_to_0_1`. That is it - no z-scoring, no per-image percentile, no
ImageNet mean and standard deviation, no optical density, no deconvolution, no polarity
flip, no per-tile standardisation. `to_model_input` below is that one division, and its
shortness is the point: every one of the things it does not do is a statement about dye
concentration that the ResNet branches make and this one does not.

Applying a z-score here - nnU-Net's default for every non-pathology dataset, and
therefore the thing a reader expects - would shift the input away from the batch-norm
running statistics baked into the checkpoint and degrade the output smoothly instead of
failing.

--------------------------------------------------------------------------------
3. The label codes, where the paper and the release disagree
--------------------------------------------------------------------------------

The BEETLE paper's ordering has invasive epithelium at 2 and non-invasive at 3. **The
released weights are the other way round** - `dataset.json` says `non-invasive
epithelium: 2, invasive epithelium: 3` - and this is the single most damaging error
available here: with the paper's ordering every duct BEETLE calls in-situ would be
reported as invasive carcinoma and every invasive focus as in-situ, an inversion on
precisely the boundary Rule 5 exists to draw and the one a clinical score is gated on.
It would look entirely plausible on screen.

So the codes are **read from the archive at load time** and compared against
`BEETLE_CODES`; a release that moves them is refused rather than served. This repeats
`tissue_label_generation/backend/bracs_app/labels.py`, deliberately: that module found
the discrepancy, and the check is cheap enough that both places running these weights
should make it rather than one trusting the other.

--------------------------------------------------------------------------------
4. Five classes, kept as five
--------------------------------------------------------------------------------

**What this branch draws is BEETLE's own five classes.** `other` and `necrosis` are
different things and the network was trained to tell them apart; collapsing them before
anything is drawn would throw away a distinction it offers for free, and it is one that
matters - necrosis inside a duct is comedo DCIS, and necrosis is most of what a treated
tumour bed is made of. So `PIXEL_CLASSES` is the five, each gets a colour and a legend
entry, and the class filter toggles all five.

`unannotated` - channel 0 - is the one worth having rather than merely tolerating. It is
not an abstention; it is a learned "this looks like the unannotated parts of my training
slides", which is mostly glass and background. The ResNet branches have no way to say
that, and the consequence is on the record: a one-window in-situ rim traced the
section's edge until a window-level tissue gate removed it, because a mostly-empty
window standardised against its own weak p99 acquires texture the slide has not got.
BEETLE can simply say so, per pixel, and the tissue gate is correspondingly not this
branch's only defence against glass.

`SCORED_CODE` is invasive epithelium, code 3, and it is the one number here that is a
pipeline judgement rather than BEETLE's: Rule 5 says a score is measured on invasive
carcinoma alone, with in-situ disease, stroma, fat and necrosis all out of the
denominator. The share is reported over the four *tissue* classes, with `unannotated`
excluded, because glass in a denominator is a way of reporting a smaller tumour content
for a slide that happens to have more empty space on it.

--------------------------------------------------------------------------------
Cost, stated plainly
--------------------------------------------------------------------------------

One 512x512 forward pass is about 1.05 s on four CPU threads with `channels_last`, and a
whole section at 0.5 um/px is on the order of a gigapixel however it is cut up. So a
whole-slide pass is hours, and **the four fields of view are not equally priced** -
measured on CAN_00270_26_H&E (28 x 28 mm, 45% tissue) at zero overlap:

    112 um   224 px   1 patch/window   28,702 windows   80 patches/mm2   8.5 h
    224 um   448 px   1 patch/window    7,202 windows   20 patches/mm2   2.2 h
    448 um   896 px   9 patches/window  1,804 windows   45 patches/mm2   5.1 h
    672 um  1344 px  25 patches/window    795 windows   55 patches/mm2   5.4 h

Two effects pull against each other and neither is obvious:

  **Padding waste.** A window smaller than one patch is mirror-padded up to 512, so the
  forward pass is paid for in full and only part of its output is used. At 112 um that
  is `(512/224)**2` = 5.2x wasted; at 224 um the 448 px window nearly fills the patch and
  wastes only 1.31x.

  **Intra-window overlap, which only exists above one patch.** The half-step's 4x cost
  applies to windows *bigger* than a patch and to no others - at 112 and 224 um the
  window is a single patch, so there is no seam inside it to blend and the step is free.
  It is what makes 448 um nine passes and 672 um twenty-five.

**224 um is therefore the cheapest option and by a wide margin**, because it is the one
that nearly fills exactly one patch. That is a coincidence of BEETLE's 512 px patch at
0.5 um/px rather than anything about the tissue, and it happens to be
`tiling_field_of_view_um`'s default.

Two settings exist because of the cost, and both are honest trades rather than tuning:

  `tissue_type_beetle_folds`  how many of the five released folds to average. One by
      default. Five is five times the work for an uncertainty estimate nothing
      downstream currently reads, and a single fold's spread is zero - which is *one
      fold*, not consensus, and must never be reported as agreement.
  `tissue_type_beetle_patch_step`  the sliding window's step as a fraction of the patch.
      **0.5, nnU-Net's own default, and the largest single cost in the pass.** A full
      step is four times cheaper and was tried first; it fails. Measured on a 672 um
      window over a duct/invasive boundary, a full step makes the 3x3 patch grid visible
      as a *checkerboard* whose squares disagree between in-situ and invasive - the seam
      does not blur the boundary, it inverts the class across it, on precisely the
      distinction Rule 5 exists to draw. `patch_grid`'s even spreading helps but cannot
      fix it, because with a 96 px overlap the Gaussian has almost nothing to blend.
      The setting's comment in `config.py` carries the three-way comparison.

BEETLE, Zenodo record 16812932, CC BY-NC-SA 4.0. **Non-commercial: every number this
branch produces is research-only and must not ship.**
"""

from __future__ import annotations

import io
import json
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import MODELS_ROOT, settings

#: Where the release is expected. Not committed - 1.9 GB - and fetched by `setup.py`.
#: A tuple so a clone keeping it elsewhere can add a location without a code change.
MODEL_ZIP_CANDIDATES: tuple[Path, ...] = (
    MODELS_ROOT / "beetle" / "model.zip",
)

#: The single directory inside `model.zip`, named for the trainer that produced it.
MEMBER_ROOT: str = (
    "model/nnUNetTrainer_WSD_wei_i0_nnunet_aug_json__"
    "nnUNetWholeSlideDataPlans__wsd_None_iterator_nnunet_aug__2d/"
)

#: The five folds, as members of `model.zip`. `checkpoint_best.pth` is what the release
#: ships; nnU-Net normally writes `checkpoint_final.pth` too, and the archive has only
#: the best-validation one.
FOLD_MEMBERS: tuple[str, ...] = tuple(
    f"{MEMBER_ROOT}fold_{fold}/checkpoint_best.pth" for fold in range(5)
)

#: Microns per pixel the release was trained at, and therefore the resolution every
#: window is resampled to. Asserted against `dataset.json` in `read_archive`; see the
#: module docstring for why this is a refusal and not a default.
SPACING: float = 0.5

#: The name this branch reports itself under. Deliberately not a filename in
#: `models/tissue_type/`: nothing about this branch is a published checkpoint of ours,
#: and a name that looked like one would invite `model.find` to be called on it.
MODEL_NAME: str = "beetle_nnunet_2d"

#: The architecture string the report carries. Not `resnet18` and not
#: `concat_resnet18_mlp`, which are the two the other branches record.
ARCH: str = "beetle_nnunet_2d"

#: The released 2d configuration's patch side, for descriptions that must not pay to
#: open a 1.9 GB archive - the capability listing, and a cost estimate on step 7.
#: **Not the authority.** `Archive.patch` is, read from `plans.json` by
#: `beetle_net.patch_size` at load, and every forward pass is sized from that. If a
#: future release changes it, this constant is wrong in a listing and the pass is still
#: correct, which is the right way round for the two to disagree.
NOMINAL_PATCH_PX: int = 512

#: The licence the whole archive carries, and the reason it does.
LICENCES: dict[str, str] = {
    "beetle": "CC BY-NC-SA 4.0 (Zenodo 16812932) - non-commercial research only",
}
LICENCE_TRACK: str = "research-only"

CITATION: str = (
    "BEETLE: a released nnU-Net for breast epithelium and tumour segmentation, "
    "Zenodo record 16812932, CC BY-NC-SA 4.0. Isensee F, Jaeger PF, Kohl SAA, "
    "Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep "
    "learning-based biomedical image segmentation. Nature Methods 18:203-211 (2021) - "
    "the architecture and the sliding-window inference reproduced here."
)

#: BEETLE's own label codes, as the released `dataset.json` defines them - checked at
#: load by `codes_from_dataset_json`, never trusted from here. **Note 2 and 3 against
#: the paper's ordering**; see the module docstring's section 3.
BEETLE_CODES: dict[str, int] = {
    "unannotated": 0,
    "other": 1,
    "non_invasive_epithelium": 2,
    "invasive_epithelium": 3,
    "necrosis": 4,
}

#: `dataset.json` spells these with spaces and hyphens, and calls code 0 `unannotated`
#: in the released copy but `background` in the checkpoint's own `init_args`. Both names
#: mean the same channel, so normalising here compares meanings rather than spellings.
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

#: What each BEETLE channel is called, in channel order. **These are the classes this
#: branch reports and draws** - see the module docstring's section 4 - so this tuple is
#: indexed by pixel value throughout `pixels.py` and the overlay.
PIXEL_CLASSES: tuple[str, ...] = (
    "unannotated",
    "other",
    "non_invasive_epithelium",
    "invasive_epithelium",
    "necrosis",
)

#: The code the final score would be gated on. Rule 5: invasive carcinoma alone.
SCORED_CODE: int = 3

#: The code that is not tissue. Excluded from the tumour-content denominator, because
#: glass in a denominator reports a smaller tumour content for a slide that merely has
#: more empty space on it.
GLASS_CODE: int = 0

#: The codes that are tissue - the denominator `tumour_content` is taken over.
TISSUE_CODES: tuple[int, ...] = (1, 2, 3, 4)

#: Plain-language label per code, for the screen. The UI has to read for someone without
#: a pathology background, so "in-situ" and "epithelium" are spelled out here rather
#: than assumed; `CLASS_MEANING` keeps the rigorous version for the notes.
CLASS_LABELS: dict[int, str] = {
    0: "Not tissue",
    1: "Not tumour tissue",
    2: "Tumour still inside the duct",
    3: "Tumour that has broken out",
    4: "Dead tissue",
}

#: One line per class, again for the screen rather than for a reviewer.
CLASS_PLAIN: dict[int, str] = {
    0: "Background and empty glass. The model is saying there is nothing here to "
    "judge.",
    1: "Supporting tissue, fat, inflammation and blood vessels. Not scored.",
    2: "Tumour cells still held inside a duct. Not scored, because treatment "
    "decisions are based on tumour that has spread out.",
    3: "Tumour cells that have grown out into the surrounding tissue. This is the "
    "only tissue the final score is measured on.",
    4: "Tissue that has died. Not scored.",
}

#: What each class actually holds, in the words the report uses.
CLASS_MEANING: dict[int, str] = {
    0: "BEETLE's `unannotated` channel - glass and background. Not tissue, and out of "
    "the tumour-content denominator",
    1: "BEETLE's `other` - stroma, fat, lymphocytes, vessels. Excluded (Rule 1)",
    2: "DCIS, LCIS, normal ducts and lobules. Excluded (Rule 5)",
    3: "invasive carcinoma - the only class a score is measured on",
    4: "necrosis and debris. Excluded",
}

#: Overlay colour per code, chosen to survive 55% opacity over a pink-and-brown scan.
#: The five are far apart in hue rather than in brightness, for a reason worth stating:
#: a class map is read on a section that is already pink and brown, and two colours a
#: step apart on one ramp stop being separable there. Slate for `unannotated`,
#: deliberately dull - it is the class a reader should be drawn to last - and red for
#: invasive, the one they should reach first.
CLASS_COLOURS: dict[int, tuple[int, int, int]] = {
    0: (100, 116, 139),
    1: (250, 204, 21),
    2: (59, 130, 246),
    3: (239, 68, 68),
    4: (168, 85, 247),
}


class BeetleError(RuntimeError):
    """The archive is absent, unreadable, or not the release this module expects."""


def model_zip() -> Path:
    """The first candidate location that exists, or the first for the error message."""
    for candidate in MODEL_ZIP_CANDIDATES:
        if candidate.is_file():
            return candidate
    return MODEL_ZIP_CANDIDATES[0]


def available() -> tuple[bool, str | None]:
    """Whether this branch can run at all, and why not when it cannot.

    Cheap on purpose - a `stat`, no torch and no 371 MB read - because the capability
    endpoint and step 7's picker both ask it on every visit, and a branch that is
    offered has to be offered with a reason attached when it is unavailable.
    """
    path = model_zip()
    if not path.is_file():
        return False, (
            f"BEETLE's weights are not at {path}. The archive is 1.9 GB and is not "
            "committed; fetch it from Zenodo record 16812932 (CC BY-NC-SA 4.0, "
            f"non-commercial) into {path.parent} - `setup.py` does this - before this "
            "option can run."
        )
    return True, None


def codes_from_dataset_json(labels: dict[str, int]) -> dict[str, int]:
    """The released `labels` block, normalised, with the paper's ordering re-checked.

    Raises if the archive disagrees with `BEETLE_CODES`. Not paranoia about a file that
    has already been read once: the check costs nothing and it is the only thing
    standing between a differently-ordered future release and a class map inverted on
    exactly the in-situ/invasive boundary a score is gated on.
    """
    normalised: dict[str, int] = {}
    for name, code in labels.items():
        key = _ALIASES.get(name.strip().lower())
        if key is None:
            raise BeetleError(
                f"dataset.json names a class {name!r} that this module has no "
                "translation for. A sixth tissue class cannot be mapped by guesswork."
            )
        normalised[key] = int(code)

    if normalised != BEETLE_CODES:
        raise BeetleError(
            f"the released label codes are {normalised}, and this module was written "
            f"against {BEETLE_CODES}. Refusing to run: if 2 and 3 have moved, every "
            "in-situ duct would be reported as invasive carcinoma and the tumour "
            "content would be wrong in the one direction nobody would question."
        )
    return normalised


@dataclass(frozen=True)
class Archive:
    """The two JSON files in `model.zip`, read once and checked."""

    plans: dict
    dataset: dict
    patch: int
    path: Path

    @property
    def num_classes(self) -> int:
        return len(self.dataset["labels"])

    @property
    def spacing(self) -> float:
        return float(self.dataset["spacing"])


def read_archive(path: Path | None = None) -> Archive:
    """`plans.json` and `dataset.json`, with the codes and the spacing verified as read.

    Both checks happen here rather than at first use, so a mismatched archive fails when
    the model is loaded and not two hours into a pass.
    """
    zip_path = Path(path) if path is not None else model_zip()
    if not zip_path.is_file():
        _, problem = available()
        raise BeetleError(problem or f"BEETLE's weights are not at {zip_path}")

    try:
        with zipfile.ZipFile(zip_path) as archive:
            plans = json.loads(archive.read(MEMBER_ROOT + "plans.json"))
            dataset = json.loads(archive.read(MEMBER_ROOT + "dataset.json"))
    except (KeyError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise BeetleError(
            f"{zip_path} is not the BEETLE release this module can read ({exc}). "
            "Re-fetch it from Zenodo record 16812932 and check it against "
            "models/beetle/CHECKSUMS."
        ) from exc

    codes_from_dataset_json(dataset["labels"])

    spacing = float(dataset["spacing"])
    if abs(spacing - SPACING) > 1e-9:
        raise BeetleError(
            f"the release was trained at {spacing} um/px and this module resamples "
            f"every window to {SPACING}. Feeding a network the wrong physical scale is "
            "the one preprocessing error that looks like a bad model rather than a bug."
        )

    from . import beetle_net

    return Archive(
        plans=plans,
        dataset=dataset,
        patch=beetle_net.patch_size(plans),
        path=zip_path,
    )


# --- the input contract ------------------------------------------------------


def to_model_input(rgb: np.ndarray) -> np.ndarray:
    """An HxWx3 uint8 window to the 3xHxW float32 array the network sees.

    `RGBTo01Normalization`, which is division by 255 and nothing else. **The shortness
    is the contract.** No optical density, no white point, no colour deconvolution, no
    clip, no shape term, no polarity flip, no per-tile standardisation and no ImageNet
    statistics - every one of those is a statement about dye concentration that the
    ResNet branches make and this one does not, and applying any of them would move the
    input away from the batch-norm running statistics the checkpoint carries.

    There is correspondingly nothing here that can drift between what trained this
    network and what serves it, which is the failure the other branches need a verified
    manifest to prevent.
    """
    x = np.asarray(rgb)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 window, got shape {x.shape}")
    return (x.astype(np.float32) / 255.0).transpose(2, 0, 1)


# --- geometry ----------------------------------------------------------------


def window_px(fov_um: float) -> int:
    """The window side, in the network's own pixels, for a field of view in microns.

    The whole geometry of this branch. 112 um is 224 px, 224 um is 448 px, 448 um is
    896 px and 672 um is 1344 px - all at BEETLE's fixed 0.5 um/px, because the
    resolution is the release's and only the extent is step 7's choice.
    """
    side = int(round(float(fov_um) / SPACING))
    if side < 16:
        raise BeetleError(
            f"{fov_um:g} um is {side} px at {SPACING} um/px, which is not a field of view"
        )
    return side


def patch_grid(extent: int, patch: int, step: float) -> list[int]:
    """Patch origins along one axis, spread evenly rather than packed flush.

    nnU-Net's own rule is `range(0, extent - patch + 1, stride)` with the flush-right
    origin appended when the last stride does not land on the edge. That covers
    everything, but for some extents it leaves two patches meeting **edge to edge** in
    the middle with no overlap to blend across: a 1344 px window at a full-patch step
    gives origins 0, 512 and 832, and the first two share not one pixel. A U-Net is
    worst at the edge of its receptive field, so that join is a visible line - and it
    falls across ducts, which is to say across exactly the boundary this step exists to
    draw.

    So the patch *count* is computed nnU-Net's way and the origins are then distributed
    evenly across the available range. Same number of forward passes, no join without
    overlap: the same window becomes 0, 416, 832, each pair sharing 96 px for the
    Gaussian to blend over. Where the flush rule already overlapped everywhere - an
    896 px window, or any step of 0.5 - this returns the origins it did.

    **One case needs a patch more, and it is the case even spreading cannot fix.** When
    `extent - patch` is an exact multiple of the stride, the even spread *is* the flush
    packing - 2048 px at a full step gives 0, 512, 1024, 1536, every join a hard edge -
    so the count is raised until neighbours actually overlap. That costs one origin per
    axis and it never triggers for the four window sizes this branch uses (224, 448, 896
    and 1344 px, at either step), so it is a guarantee the function can make honestly
    rather than a cost anyone pays. Making the promise and then not keeping it in the
    one arithmetically special case is how a seam gets shipped.
    """
    if extent <= patch:
        return [0]
    stride = max(1, int(round(patch * step)))
    count = int(np.ceil((extent - patch) / stride)) + 1
    if count <= 1:
        return [0]

    span = extent - patch
    # Strictly less than `patch`, so every consecutive pair shares pixels for the
    # Gaussian to blend across. At most one extra origin in practice.
    while span / (count - 1) >= patch:
        count += 1

    return [int(round(index * span / (count - 1))) for index in range(count)]


def _step(step: float | None) -> float:
    value = settings.tissue_type_beetle_patch_step if step is None else float(step)
    if not 0.0 < value <= 1.0:
        raise BeetleError(
            f"the patch step must be greater than 0 and at most 1, not {value}; above 1 "
            "the sliding window would leave unsegmented strips between patches"
        )
    return value


def patches_per_window(fov_um: float, patch: int, step: float | None = None) -> int:
    """Forward passes one window costs per fold. For an honest time estimate."""
    per_axis = len(patch_grid(window_px(fov_um), patch, _step(step)))
    return per_axis * per_axis


def folds_to_run(count: int | None = None) -> tuple[int, ...]:
    """Which of the five released folds a pass averages, from the setting."""
    wanted = settings.tissue_type_beetle_folds if count is None else int(count)
    if not 1 <= wanted <= len(FOLD_MEMBERS):
        raise BeetleError(
            f"the release has {len(FOLD_MEMBERS)} folds and {wanted} were asked for"
        )
    return tuple(range(wanted))


# --- loading -----------------------------------------------------------------


_CACHE: dict[int, Any] = {}
_CACHE_LOCK = threading.Lock()


def load_fold(archive: Archive, fold: int) -> Any:
    """One fold, in eval mode, ready for inference. Cached, because it is 371 MB.

    ~371 MB of checkpoint is streamed from the zip and about 46 M parameters survive it;
    the optimiser state, which is most of the file, is dropped on the floor by reading
    only `network_weights`.

    Cached across passes so a viewer who reruns step 8, or runs it on a second slide,
    does not pay a 371 MB read and a rebuild again. One resident fold is ~185 MB of
    float32 parameters, which is why the default is one fold.
    """
    with _CACHE_LOCK:
        cached = _CACHE.get(fold)
    if cached is not None:
        return cached

    import torch

    from . import beetle_net

    try:
        with zipfile.ZipFile(archive.path) as zf, zf.open(FOLD_MEMBERS[fold]) as member:
            # `torch.load` needs a seekable file and a zip member is not one, so this
            # buffers ~371 MB. Materialising it is deliberate: extracting all five folds
            # would cost 1.9 GB of disk permanently, and holding one costs it for a few
            # seconds.
            blob = member.read()
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise BeetleError(
            f"fold {fold} could not be read out of {archive.path} ({exc})"
        ) from exc

    checkpoint = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    if "network_weights" not in checkpoint:
        raise BeetleError(
            f"fold {fold} of {archive.path} carries no `network_weights`, so it is not "
            "an nnU-Net checkpoint this module can load."
        )

    net = beetle_net.from_plans(archive.plans, num_classes=archive.num_classes)
    beetle_net.load_beetle_weights(net, checkpoint["network_weights"])
    net.eval()

    # `channels_last` is a memory layout and not a numerical change - the same weights,
    # the same arithmetic, reordered so oneDNN's convolutions hit their fast path.
    # Measured on this box: 1.26 s to 1.05 s per 512 px patch, about 20% off the whole
    # pass for one line. It is applied here rather than at the call site so a cached
    # fold is already in the right layout and `predict_window` need not know.
    net = net.to(memory_format=torch.channels_last)

    with _CACHE_LOCK:
        _CACHE[fold] = net
    return net


def clear_cache() -> None:
    """Drop the resident folds. For the maintenance endpoint and the tests."""
    with _CACHE_LOCK:
        _CACHE.clear()


@dataclass
class Loaded:
    """A loaded BEETLE and everything a run records about it.

    Its own type rather than the ResNet branches' `model.Pinned`: that dataclass exists
    to carry a *verified manifest's* input contract - polarity, standardisation, the
    shape term, a decision threshold - and BEETLE has none of those. Constructing one
    would mean filling six fields with identity values that assert a contract nobody
    published. `net` is typed loosely so only the modules that may import torch mention
    its types.
    """

    archive: Archive
    nets: tuple[Any, ...]
    folds: tuple[int, ...]

    #: The window's side in the network's own pixels, and the fixed spacing it reads at.
    #: `window_px * mpp` is the field of view, so step 7 can price this branch with the
    #: same arithmetic it prices the others with even though nothing else is shared.
    window_px: int
    mpp: float = SPACING

    name: str = MODEL_NAME
    arch: str = ARCH
    licence_track: str = LICENCE_TRACK
    licences: dict[str, str] = field(default_factory=lambda: dict(LICENCES))

    @property
    def classes(self) -> tuple[str, ...]:
        return PIXEL_CLASSES

    @property
    def patch(self) -> int:
        return self.archive.patch

    @property
    def field_um(self) -> float:
        return round(self.window_px * self.mpp, 2)

    def describe(self) -> dict[str, Any]:
        """What the run record and the report carry about this model, as plain data."""
        return {
            "name": self.name,
            "arch": self.arch,
            "path": str(self.archive.path),
            "classes": list(PIXEL_CLASSES),
            "beetle_codes": dict(BEETLE_CODES),
            "folds": list(self.folds),
            "folds_available": len(FOLD_MEMBERS),
            "patch_px": self.patch,
            "spacing": self.mpp,
            "window_px": self.window_px,
            "field_of_view_um": self.field_um,
            "licence_track": self.licence_track,
            "licences": dict(self.licences),
            "citation": CITATION,
        }


def load(fov_um: float, *, folds: int | None = None) -> Loaded:
    """BEETLE, ready to segment windows of `fov_um` microns.

    The field of view reaches the network only as the size of the array it is handed -
    the weights are identical at all four - so this loads once and records the geometry
    beside it. That is the opposite of the ResNet branches, where a field of view
    selects a *different checkpoint*, and it is why this branch is available at every
    field of view the moment the archive is on disk rather than as heads are trained.
    """
    archive = read_archive()
    wanted = folds_to_run(folds)
    return Loaded(
        archive=archive,
        nets=tuple(load_fold(archive, fold) for fold in wanted),
        folds=wanted,
        window_px=window_px(fov_um),
    )


# --- inference ---------------------------------------------------------------


def gaussian_weights(patch: int, sigma_scale: float = 1.0 / 8) -> np.ndarray:
    """nnU-Net's patch importance map: a Gaussian, floored so no pixel weighs nothing.

    The floor matters. Without it the extreme corner of a patch contributes an
    effectively zero weight, and a pixel covered *only* by patch corners - which happens
    at a window's border, where the step cannot be centred - would be divided by
    something near zero and blow up.
    """
    from scipy.ndimage import gaussian_filter

    centre = np.zeros((patch, patch), dtype=np.float32)
    centre[patch // 2, patch // 2] = 1.0
    weights = gaussian_filter(centre, sigma=patch * sigma_scale, mode="constant", cval=0)
    weights = weights / weights.max()
    return np.maximum(weights, weights[weights > 0].min()).astype(np.float32)


_WEIGHTS: dict[int, np.ndarray] = {}


def _weights_for(patch: int) -> np.ndarray:
    """`gaussian_weights`, memoised - it is the same array for every window."""
    cached = _WEIGHTS.get(patch)
    if cached is None:
        cached = gaussian_weights(patch)
        _WEIGHTS[patch] = cached
    return cached


def predict_window(
    rgb: np.ndarray,
    loaded: Loaded,
    *,
    step: float | None = None,
    batch_size: int = 1,
) -> np.ndarray:
    """Segment one window, already resampled to BEETLE's 0.5 um/px.

    `rgb` is HxWx3 uint8 **at the network's own spacing** - resampling is the caller's
    job, because `pixels.segment` reads a whole block at that spacing and cuts windows
    out of it, and doing the resample twice is how a mask ends up a pixel out of
    register with the image it labels.

    Returns `(5, H, W)` float32 probabilities over BEETLE's own codes, at the window's
    own size. Nothing is collapsed here: the five are the branch's output.

    **The seam.** A U-Net is worst at the edge of its receptive field, so patches are
    overlapped and blended by a Gaussian rather than tiled and stitched. Hard tiling
    leaves a visible grid in the argmax, and that grid falls across ducts. The weights
    are nnU-Net's own; the origins are `patch_grid`'s, which is nnU-Net's count spread
    evenly.

    **A window smaller than one patch is mirror-padded, not resized.** The network is
    fully convolutional and would accept the smaller array, but its batch-norm
    statistics and receptive field were fitted at 512, and mirroring is what nnU-Net
    itself pads with. This is the ordinary case at 112 um (224 px) and 224 um (448 px).

    Folds are averaged **in probability space, after the softmax**, which is what
    nnU-Net does and is not the same as averaging logits.
    """
    import torch

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 window, got {rgb.shape} {rgb.dtype}")

    height, width = rgb.shape[:2]
    patch = loaded.patch

    pad_y = max(0, patch - height)
    pad_x = max(0, patch - width)
    if pad_y or pad_x:
        rgb = np.pad(rgb, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")
    padded_h, padded_w = rgb.shape[:2]

    ys = patch_grid(padded_h, patch, _step(step))
    xs = patch_grid(padded_w, patch, _step(step))
    weights = _weights_for(patch)

    tensor = torch.from_numpy(to_model_input(rgb))

    classes = loaded.archive.num_classes
    total = np.zeros((classes, padded_h, padded_w), dtype=np.float32)
    norm = np.zeros((padded_h, padded_w), dtype=np.float32)

    origins = [(y, x) for y in ys for x in xs]
    width_of_batch = max(1, int(batch_size))

    for net in loaded.nets:
        with torch.inference_mode():
            for start in range(0, len(origins), width_of_batch):
                chunk = origins[start : start + width_of_batch]
                stack = torch.stack(
                    [tensor[:, y : y + patch, x : x + patch] for y, x in chunk]
                ).contiguous(memory_format=torch.channels_last)
                probs = torch.softmax(net(stack), dim=1).float().numpy()
                for (y, x), plane in zip(chunk, probs, strict=True):
                    total[:, y : y + patch, x : x + patch] += plane * weights
                    norm[y : y + patch, x : x + patch] += weights

    mean = total / np.maximum(norm, 1e-8)[None]
    return mean[:, :height, :width]


def self_check() -> dict[str, Any]:
    """Assert the rebuilt network answers something structured, on synthetic input.

    A strict weight load cannot catch a *wiring* error between correctly-shaped layers -
    a skip attached to the wrong stage - and such a model runs and segments noise. What
    it cannot do is produce a spatially structured answer, so this is the check from the
    other side: run one patch and report the class shares and the mean top probability.
    A single distinct label, or a mean top probability near 1/5, is the failure this
    names.

    Returned rather than asserted so the caller decides what to do with it; the tests
    assert on it and the capability endpoint does not pay for it.
    """
    import torch

    archive = read_archive()
    net = load_fold(archive, 0)
    patch = archive.patch

    grid = np.indices((patch, patch)).sum(axis=0) % 64
    synthetic = np.clip(
        np.stack([grid * 4, 255 - grid * 3, grid * 2], axis=-1), 0, 255
    ).astype(np.uint8)

    with torch.inference_mode():
        tensor = torch.from_numpy(to_model_input(synthetic))
        probs = torch.softmax(net(tensor.unsqueeze(0)), dim=1).numpy()[0]

    labels = probs.argmax(axis=0)
    return {
        "patch": patch,
        "classes": int(archive.num_classes),
        "distinct_labels": int(np.unique(labels).size),
        "mean_top_probability": float(probs.max(axis=0).mean()),
        "channel_shares": {
            PIXEL_CLASSES[code]: round(float((labels == code).mean()), 4)
            for code in range(archive.num_classes)
        },
    }


__all__ = [
    "ARCH",
    "BEETLE_CODES",
    "CITATION",
    "CLASS_COLOURS",
    "CLASS_LABELS",
    "CLASS_MEANING",
    "CLASS_PLAIN",
    "GLASS_CODE",
    "LICENCES",
    "LICENCE_TRACK",
    "MODEL_NAME",
    "NOMINAL_PATCH_PX",
    "PIXEL_CLASSES",
    "SCORED_CODE",
    "SPACING",
    "TISSUE_CODES",
    "Archive",
    "BeetleError",
    "Loaded",
    "available",
    "clear_cache",
    "codes_from_dataset_json",
    "folds_to_run",
    "gaussian_weights",
    "load",
    "load_fold",
    "model_zip",
    "patch_grid",
    "patches_per_window",
    "predict_window",
    "read_archive",
    "self_check",
    "to_model_input",
    "window_px",
]
