"""Running BEETLE's released ensemble over a region of interest, on a CPU.

Two things here are easy to get wrong in ways that produce a picture rather than an
error, so both are written out with the reason attached:

**Preprocessing.** `plans.json` names the normalisation `RGBTo01Normalization` and
`dataset.json` names all three channels `rgb_to_0_1`. That is division by 255 and
nothing else - no z-scoring, no per-image percentile, no ImageNet mean and standard
deviation. Applying a z-score here, which is nnU-Net's default for every non-pathology
dataset and therefore the thing a reader expects, would shift the input distribution
away from the batch-norm running statistics baked into the checkpoint and degrade the
output smoothly instead of failing.

**The seam.** A U-Net is worst at the edge of its receptive field, so patches are
overlapped and blended by a Gaussian rather than tiled edge to edge and stitched. Hard
tiling leaves a visible grid in the argmax, and that grid falls across ducts - which
means it falls across exactly the class this app exists to harvest, splitting single
ducts into two differently-labelled halves at the seam. The weights and the half-step
are nnU-Net's own inference defaults, reproduced rather than reinvented.

Folds are averaged in probability space, after the softmax, which is what nnU-Net does
and is not the same as averaging logits. Their disagreement is kept: it is the only
uncertainty signal available without a pathologist, and it is what a review sample
should be drawn by.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from . import config, labels, unet

#: The five folds, as members of `model.zip`. `checkpoint_best.pth` is what the release
#: ships; nnU-Net normally writes `checkpoint_final.pth` too, and the archive has only
#: the best-validation one.
FOLD_MEMBERS: tuple[str, ...] = tuple(
    f"{config.MODEL_MEMBER_ROOT}fold_{fold}/checkpoint_best.pth" for fold in range(5)
)


@dataclass(frozen=True)
class Archive:
    """The two JSON files in `model.zip`, read once and checked."""

    plans: dict
    dataset: dict

    @property
    def num_classes(self) -> int:
        return len(self.dataset["labels"])

    @property
    def spacing(self) -> float:
        return float(self.dataset["spacing"])


def read_archive(model_zip: Path | None = None) -> Archive:
    """`plans.json` and `dataset.json`, with the label codes verified as they are read.

    `labels.codes_from_dataset_json` raises if the release's ordering is not the one
    this app was written against. It is called here rather than at first use so that a
    mismatched archive fails when the model is loaded, not four minutes into a run.
    """
    path = Path(model_zip or config.MODEL_ZIP)
    if not path.exists():
        raise FileNotFoundError(
            f"BEETLE's weights are not at {path}. The archive is 1.9 GB and is not "
            "committed; fetch it from Zenodo record 16812932 into "
            f"{path.parent} before starting the app."
        )

    with zipfile.ZipFile(path) as archive:
        plans = json.loads(archive.read(config.MODEL_MEMBER_ROOT + "plans.json"))
        dataset = json.loads(archive.read(config.MODEL_MEMBER_ROOT + "dataset.json"))

    labels.codes_from_dataset_json(dataset["labels"])
    if abs(float(dataset["spacing"]) - config.TEACHER_MPP) > 1e-9:
        raise ValueError(
            f"the release was trained at {dataset['spacing']} um/px and this app "
            f"resamples to {config.TEACHER_MPP}. Feeding a network the wrong physical "
            "scale is the one preprocessing error that looks like a bad model."
        )
    return Archive(plans=plans, dataset=dataset)


def load_fold(archive: Archive, fold: int, model_zip: Path | None = None):
    """One fold, in eval mode, ready for inference.

    ~370 MB of checkpoint is streamed from the zip and about 46 M parameters survive
    it; the optimiser state, which is most of the file, is dropped on the floor by
    reading only `network_weights`.
    """
    path = Path(model_zip or config.MODEL_ZIP)
    with zipfile.ZipFile(path) as zf, zf.open(FOLD_MEMBERS[fold]) as member:
        # `torch.load` needs a seekable file and a zip member is not one, so this
        # buffers ~370 MB. Materialising it is deliberate: extracting all five folds to
        # disk would cost 1.9 GB permanently, and holding one in memory costs it for a
        # few seconds.
        import io

        checkpoint = torch.load(
            io.BytesIO(member.read()), map_location="cpu", weights_only=False
        )

    model = unet.from_plans(archive.plans, num_classes=archive.num_classes)
    unet.load_beetle_weights(model, checkpoint["network_weights"])
    model.eval()
    return model


def _gaussian_weights(patch: int, sigma_scale: float = 1.0 / 8) -> np.ndarray:
    """nnU-Net's patch importance map: a Gaussian, floored so no pixel weighs nothing.

    The floor matters. Without it the extreme corner of a patch contributes an
    effectively zero weight, and a pixel covered *only* by patch corners - which
    happens at the image border, where the step cannot be centred - would be divided by
    something near zero and blow up.
    """
    from scipy.ndimage import gaussian_filter

    centre = np.zeros((patch, patch), dtype=np.float32)
    centre[patch // 2, patch // 2] = 1.0
    weights = gaussian_filter(centre, sigma=patch * sigma_scale, mode="constant", cval=0)
    weights = weights / weights.max()
    return np.maximum(weights, weights[weights > 0].min()).astype(np.float32)


def _starts(extent: int, patch: int, step: int) -> list[int]:
    """Patch origins along one axis, always including the last full patch.

    `range(0, extent - patch + 1, step)` alone leaves an unsegmented strip at the right
    and bottom whenever the extent is not a whole number of steps - a strip up to 255 px
    wide, which at 0.5 um/px is 128 um of tissue silently labelled nothing. Appending
    the flush-right origin costs one more patch and covers it.
    """
    if extent <= patch:
        return [0]
    origins = list(range(0, extent - patch + 1, step))
    if origins[-1] != extent - patch:
        origins.append(extent - patch)
    return origins


@dataclass
class Prediction:
    """What the ensemble thinks, and how much it disagreed with itself."""

    #: `(H, W)` uint8 argmax over BEETLE's five codes.
    mask: np.ndarray
    #: `(H, W)` float32 in [0, 1] - the winning class's mean probability.
    confidence: np.ndarray
    #: `(H, W)` float32 in [0, 1] - the spread across folds at the winning class, 0 for
    #: a single fold. Zero everywhere is not agreement, it is one fold; the caller must
    #: not report it as consensus.
    disagreement: np.ndarray
    folds: tuple[int, ...]


def predict(
    rgb: np.ndarray,
    archive: Archive,
    folds: tuple[int, ...] = (0,),
    *,
    model_zip: Path | None = None,
    tile_step: float | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> Prediction:
    """Segment one region, already resampled to the teacher's 0.5 um/px.

    `rgb` is HxWx3 uint8 **at the teacher's own spacing**. Resampling is the caller's
    job (`pipeline.prepare_region`) because the same resampled array is also what the
    tile exporter must see - doing it twice, once here and once there, is how a mask
    ends up a pixel out of register with the image it labels.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise TypeError(f"expected an HxWx3 uint8 image, got {rgb.shape} {rgb.dtype}")
    if not folds:
        raise ValueError("at least one fold is needed")

    torch.set_num_threads(config.torch_threads())
    height, width = rgb.shape[:2]
    patch = config.PATCH_PX
    # `tile_step` overrides the default half-step. It exists because the step is a
    # QUADRATIC cost: at 0.5 a region needs four times the forward passes it needs at
    # 1.0, and on a whole slide that is the difference between twenty minutes and an
    # hour and a quarter. Half-step is right for a single region, where the seam would
    # be visible; whole-slide output is reduced to one label per 224 px tile and then
    # displayed at a 20x downscale, so a seam a few pixels wide survives neither step.
    step = max(1, int(round(patch * (config.TILE_STEP if tile_step is None else tile_step))))

    # A region smaller than one patch is mirror-padded to it rather than resized. The
    # network is fully convolutional and would accept the small array, but its batch
    # norm statistics and receptive field were fitted at 512, and mirroring is what
    # nnU-Net itself pads with.
    pad_y = max(0, patch - height)
    pad_x = max(0, patch - width)
    if pad_y or pad_x:
        rgb = np.pad(rgb, ((0, pad_y), (0, pad_x), (0, 0)), mode="reflect")
    padded_h, padded_w = rgb.shape[:2]

    ys = _starts(padded_h, patch, step)
    xs = _starts(padded_w, patch, step)
    weights = _gaussian_weights(patch)

    # RGBTo01Normalization, and nothing else. See the module docstring.
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

    n_classes = archive.num_classes
    # float32 accumulators at region size: a 3000x3000 region costs 3000*3000*5*4 =
    # 180 MB per accumulator, which is why the folds are accumulated in place rather
    # than kept as a stack.
    total = np.zeros((n_classes, padded_h, padded_w), dtype=np.float32)
    total_sq = np.zeros((n_classes, padded_h, padded_w), dtype=np.float32)
    norm = np.zeros((padded_h, padded_w), dtype=np.float32)

    steps_done = 0
    steps_total = len(folds) * len(ys) * len(xs)

    for fold in folds:
        if progress:
            progress(f"loading fold {fold}", steps_done / steps_total)
        model = load_fold(archive, fold, model_zip=model_zip)

        fold_probs = np.zeros((n_classes, padded_h, padded_w), dtype=np.float32)
        fold_norm = np.zeros((padded_h, padded_w), dtype=np.float32)

        with torch.inference_mode():
            for y in ys:
                for x in xs:
                    window = tensor[:, y : y + patch, x : x + patch].unsqueeze(0)
                    logits = model(window)[0]
                    probs = torch.softmax(logits, dim=0).numpy()
                    fold_probs[:, y : y + patch, x : x + patch] += probs * weights
                    fold_norm[y : y + patch, x : x + patch] += weights
                    steps_done += 1
                    if progress:
                        progress(
                            f"fold {fold}: patch {steps_done} of {steps_total}",
                            steps_done / steps_total,
                        )

        fold_probs /= np.maximum(fold_norm, 1e-8)
        total += fold_probs
        total_sq += fold_probs**2
        norm += 1.0
        # A fold is 370 MB of tensors; dropping it before loading the next one keeps
        # peak memory at one fold rather than five.
        del model, fold_probs, fold_norm

    mean = total / norm
    mask = mean.argmax(axis=0).astype(np.uint8)

    winner = np.take_along_axis(mean, mask[None], axis=0)[0]
    if len(folds) > 1:
        variance = np.maximum(
            np.take_along_axis(total_sq / norm, mask[None], axis=0)[0] - winner**2, 0.0
        )
        spread = np.sqrt(variance)
    else:
        spread = np.zeros_like(winner)

    crop = (slice(0, height), slice(0, width))
    return Prediction(
        mask=mask[crop],
        confidence=winner[crop].astype(np.float32),
        disagreement=spread[crop].astype(np.float32),
        folds=tuple(folds),
    )


def patch_count(height: int, width: int) -> int:
    """How many forward passes one region costs per fold, for an honest time estimate."""
    patch, step = config.PATCH_PX, max(1, int(round(config.PATCH_PX * config.TILE_STEP)))
    return len(_starts(max(height, patch), patch, step)) * len(
        _starts(max(width, patch), patch, step)
    )
