"""Our trained tile classifier, run over a region so it can be compared with BEETLE.

The model this loads is approach 1's published checkpoint - a frozen ResNet18 with a
three-class linear head, fitted on BCSS plus the in-situ tiles this app borrowed from
BRACS. It answers one question per 224 px tile, so "segmenting" a region with it means
classifying a grid of tiles and painting each one its answer. The result is blocky, and
`sixslides.py` renders it blocky on purpose: 224 px at 0.5 um/px **is** this model's
resolution, and smoothing the picture would claim a precision it does not have.

--------------------------------------------------------------------------------
The only thing in this module that is hard to get right
--------------------------------------------------------------------------------

**The input transform must be the one the model was fitted under, exactly, with every
setting read from the manifest and none from a default.**

This is not a general principle stated for safety; it is the bug that was found in this
project three hours ago. `05_publish.py`'s gate G6 rebuilt its check input without
passing `standardise`, so a freshly loaded checkpoint was fed a differently-scaled tensor
and its logits moved by 1.8 - and nobody noticed for a while because a second bug meant
the gate was verifying a stale checkpoint whose settings happened to match the defaults.
`to_model_input` defaults `standardise=False`; the checkpoints in use need `True`. A
wrong transform here does not raise. It produces a plausible class map that is wrong, and
the six-slide comparison would then be reporting a preprocessing mistake as a difference
of opinion between two models.

So: `_tile_tensor` takes its `gamma`, `invert` and `standardise` from
`manifest["input"]`, `tests/test_sixslides.py` asserts the tensors are identical to
`datasets.TileDataset`'s, and `verify_against_training` re-derives one tile through
approach 1's own dataset class and refuses to differ.

--------------------------------------------------------------------------------
Two details that look like details and are not
--------------------------------------------------------------------------------

**One white point per region, not per tile.** `export.py` takes a single `I0` across a
whole BCSS region of interest and deconvolves it in one pass, recording
`i0_rule = "p99_per_channel_over_region"` on every tile it writes. Neighbouring tiles must
share a white point because they are the same physical section under the same lamp. A
per-tile `I0` would rescale each tile against its own brightest pixels, so a tile of pale
stroma and a tile of dense tumour would be normalised to the same density - erasing
exactly the difference the model reads. The 2048 px region here is the analogue of a BCSS
region of interest, so the white point is taken once over it.

**The quantise/dequantise round trip is reproduced, not skipped.** Training read tiles
back off disk as 8-bit PNGs, so every tensor the model ever saw had been through
`quantise` and `dequantise` - a round trip to a 1/255 grid of [0, 1.5] OD. It is nearly
lossless and skipping it would change almost nothing, but "almost nothing" is not a thing
worth being unable to rule out when a class map looks wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import config

#: Fall back to this if the models directory holds nothing better. The `_std` variants are
#: the ones trained with per-tile standardisation, which gate G0b certified as the setting
#: that covers the served IHC distribution (percentile coverage 1/6 -> 6/6).
PREFERRED_MODELS: tuple[str, ...] = (
    "invasive_tile_v1_imagenet_std",
    "invasive_tile_v1_simclr_std",
    "invasive_tile_v1_imagenet",
)

#: Tiles per forward pass. 64 x 3 x 224 x 224 float32 is about 38 MB, which keeps the
#: peak far below the teacher's region-sized accumulators running in the same process.
BATCH = 64


def available_models() -> list[str]:
    """Every published checkpoint that has both its weights and its manifest."""
    if not config.MODELS_DIR.is_dir():
        return []
    names = []
    for manifest in sorted(config.MODELS_DIR.glob("invasive_tile_v1_*.manifest.json")):
        name = manifest.name[: -len(".manifest.json")]
        if (config.MODELS_DIR / f"{name}.pt").exists():
            names.append(name)
    return names


def default_model() -> str:
    """The checkpoint to compare against, preferring the standardised ImageNet arm.

    Named rather than "the newest file": a comparison whose subject changes because
    somebody published another checkpoint is a comparison nobody can reproduce.
    """
    found = available_models()
    if not found:
        raise FileNotFoundError(
            f"no published tile classifier in {config.MODELS_DIR}. Run "
            "bcss_bracs_hchannel_resnet18/scripts/05_publish.py --both --standardise first."
        )
    for name in PREFERRED_MODELS:
        if name in found:
            return name
    return found[0]


def load_student(name: str | None = None):
    """The published checkpoint and its manifest, SHA-256 verified.

    `models.load_pinned` is approach 1's own loader and it refuses a checkpoint whose
    hash disagrees with its manifest. Imported rather than reimplemented: the manifest is
    the input contract, and a second loader would be a second opinion about what that
    contract says.
    """
    import models  # approach 1's, via config.install_approach1_path

    chosen = name or default_model()
    net, manifest = models.load_pinned(chosen, models_dir=config.MODELS_DIR)
    net.eval()
    return net, manifest


def input_settings(manifest: dict) -> dict:
    """The three transform knobs, read off the manifest and nothing else.

    Raises on a manifest that does not carry them rather than filling in defaults. A
    missing key here means the checkpoint was published by an older script that did not
    record what it was trained under, and guessing would reproduce the G6 failure.
    """
    block = manifest.get("input")
    if not isinstance(block, dict):
        raise ValueError("this manifest has no `input` block, so its transform is unknown")

    missing = [
        key for key in ("gamma", "invert_polarity", "standardise_tile_p99", "tile_px", "mpp")
        if key not in block
    ]
    if missing:
        raise ValueError(
            f"the manifest does not record {missing}, so the transform this model was "
            "fitted under cannot be reproduced. Refusing to guess - a wrong transform "
            "does not raise, it just returns a wrong class map."
        )
    return {
        "gamma": float(block["gamma"]),
        "invert": bool(block["invert_polarity"]),
        "standardise": bool(block["standardise_tile_p99"]),
        "tile_px": int(block["tile_px"]),
        "mpp": float(block["mpp"]),
    }


def _tile_tensor(h_od_tile: np.ndarray, settings: dict) -> np.ndarray:
    """One tile of haematoxylin density to the 3xHxW the network was trained on.

    The round trip through `quantise`/`dequantise` is what training saw - tiles were
    written to disk as 8-bit PNGs and read back - so it is reproduced here rather than
    short-circuited. See the module docstring.
    """
    import hchannel

    return hchannel.from_stored(
        hchannel.quantise(h_od_tile),
        alpha=1.0,
        beta=0.0,
        gamma=settings["gamma"],
        invert=settings["invert"],
        standardise=settings["standardise"],
    )


def classify_region(rgb: np.ndarray, net, manifest: dict, *, progress=None) -> dict:
    """Classify a region tile by tile. `rgb` must already be at the model's own mpp.

    Returns the tile labels, the grid they sit on, and a full-resolution label image in
    which every pixel carries its tile's answer - the blocky map `sixslides.py` draws.

    Partial tiles at the right and bottom edge are dropped rather than padded, exactly as
    `export.export_region` does: padding invents pixels and the model would classify
    them. With a 2048 px region and a 224 px tile that is a 9x9 grid covering 2016 px, and
    the 32 px margin is reported so the caller can crop everything else to match.
    """
    import bcss

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise TypeError(f"expected an HxWx3 uint8 region, got {rgb.shape} {rgb.dtype}")

    settings = input_settings(manifest)
    tile_px = settings["tile_px"]

    import hchannel

    # ONE white point over the whole region, then one deconvolution pass. See the module
    # docstring - this is `export.py`'s rule and departing from it would rescale every
    # tile against its own content.
    white = hchannel.white_point(rgb, percentile=99.0)
    h_od = hchannel.haematoxylin_od(rgb, white)

    height, width = h_od.shape
    rows, cols = height // tile_px, width // tile_px
    if rows == 0 or cols == 0:
        raise ValueError(
            f"a {width}x{height} region does not hold a single {tile_px}px tile at the "
            "model's working resolution"
        )

    tensors: list[np.ndarray] = []
    for row in range(rows):
        for col in range(cols):
            window = (
                slice(row * tile_px, (row + 1) * tile_px),
                slice(col * tile_px, (col + 1) * tile_px),
            )
            tensors.append(_tile_tensor(h_od[window], settings))

    labels = np.empty(len(tensors), dtype=np.uint8)
    confidence = np.empty(len(tensors), dtype=np.float32)
    torch.set_num_threads(config.torch_threads())
    with torch.inference_mode():
        for start in range(0, len(tensors), BATCH):
            chunk = tensors[start : start + BATCH]
            batch = torch.from_numpy(np.stack(chunk))
            logits = net(batch)
            probs = torch.softmax(logits, dim=1)
            best = probs.argmax(dim=1)
            labels[start : start + len(chunk)] = best.numpy().astype(np.uint8)
            confidence[start : start + len(chunk)] = (
                probs.gather(1, best[:, None])[:, 0].numpy()
            )
            if progress:
                progress(
                    f"our model: tile {min(start + BATCH, len(tensors))} of {len(tensors)}",
                    min(1.0, (start + BATCH) / len(tensors)),
                )

    tile_labels = labels.reshape(rows, cols)
    # The blocky full-resolution map: every pixel of a tile carries that tile's answer.
    pixel_labels = np.repeat(np.repeat(tile_labels, tile_px, axis=0), tile_px, axis=1)

    return {
        "tile_labels": tile_labels,
        "tile_confidence": confidence.reshape(rows, cols),
        "pixel_labels": pixel_labels,
        "covered": (rows * tile_px, cols * tile_px),
        "grid": (rows, cols),
        "tile_px": tile_px,
        "class_area_fraction": {
            name: float((tile_labels == index).mean())
            for index, name in enumerate(bcss.CLASS_NAMES)
        },
        "mean_confidence": float(confidence.mean()),
        "model": manifest.get("name"),
        "input": settings,
    }


def verify_against_training(net, manifest: dict, tiles_dir: Path, tile_path: str) -> float:
    """Push a real stored training tile through both paths and return the worst difference.

    `datasets.TileDataset` is what produced every tensor the head was fitted on. If this
    module's transform and that one disagree by anything above float noise, the class maps
    on the six-slide screen are measuring a preprocessing bug rather than two models.

    Returns the maximum absolute difference so a caller can assert on it. Used by the
    tests and by `scripts/verify_student.py`.
    """
    import datasets
    from PIL import Image

    settings = input_settings(manifest)
    rows = [{"tile_path": tile_path, "label": 0, "tile_id": "probe"}]
    reference = datasets.TileDataset(
        rows, tiles_dir, augment=False,
        invert=settings["invert"], standardise=settings["standardise"],
    )[0][0].numpy()

    stored = np.asarray(Image.open(Path(tiles_dir) / tile_path))
    import hchannel

    ours = hchannel.from_stored(
        stored, alpha=1.0, beta=0.0, gamma=settings["gamma"],
        invert=settings["invert"], standardise=settings["standardise"],
    )
    return float(np.max(np.abs(reference - ours)))
