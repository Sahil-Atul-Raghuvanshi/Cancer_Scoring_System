"""Loading Cellpose's published `nuclei` model, and proving it before it is used.

Step 13's detector since 6 October 2026 (P-03). Every alternative was benchmarked on
the same fields - production's InstanSeg-on-haematoxylin, InstanSeg on RGB and on the
optical-density sum, a watershed, DeepLIIF, the LyNSeC IHC model, and Cellpose
zero-shot, fine-tuned on lymphoma IHC and fine-tuned on breast IHC - and zero-shot
Cellpose was best on both independent checks: nuclei per mm2 of tissue against the
H&E (20% mean gap against 58% for the old path) and F1 against breast IHC cells
labelled from immunofluorescence (0.77 against 0.70). Both fine-tunes did worse on
our slides. See `p03_nuclei/` and storage/v1_data/data/p03_nuclei/REPORT.md.

**What it is shown.** The field as an inverted grey image: `255 - luminance`, so a
nucleus is bright, which is how a nuclear-channel model reads brightfield. That is
the input the benchmark measured; the haematoxylin-only render and the
optical-density sum were both tried and both did worse.

**Two silent traps, both guarded here.**

  * Loaded from a file, Cellpose treats the checkpoint as its built-in nuclei model -
    base diameter 17 px - only if the file name ends in `nucleitorch_0`. Under any
    other name it falls back to 30 px, rescales every field differently, and returns
    plausible but different nuclei on every field with no warning. The checkpoint
    keeps the upstream name, and `load` refuses a model whose base diameter is not 17.
  * Like InstanSeg, a model that loads is not a model that segments as published. So
    it is run on InstanSeg's published test tile (`parity-input.npy`) and must give
    exactly the label map recorded in the manifest, by sha256.

Nothing here knows what a slide is. Arrays in, label maps out.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.nuclei.model import ModelUnavailable, ParityFailure, _sha256, models_dir

logger = get_logger(__name__)

CHECKPOINT = "cellpose_nucleitorch_0"
MANIFEST = "cellpose_nuclei.manifest.json"
#: The built-in nuclei model's base diameter. Anything else means the checkpoint was
#: loaded as a custom model and is rescaling fields the wrong way.
BASE_DIAMETER_PX = 17.0


@dataclass(frozen=True)
class LoadedCellpose:
    """A Cellpose model that has proved itself, plus the contract it met."""

    module: Any
    manifest: dict[str, Any]

    @property
    def mpp(self) -> float:
        return float(self.manifest["input"]["mpp"])


_lock = threading.Lock()
_loaded: LoadedCellpose | None = None
_failure: str | None = None


def manifest() -> dict[str, Any]:
    path = models_dir() / MANIFEST
    if not path.exists():
        raise ModelUnavailable(
            f"{MANIFEST} is missing from {models_dir()}. It records the checkpoint's "
            "sha256 and the parity answer; without it the weights cannot be trusted."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def inverted_grey(rgb: np.ndarray) -> np.ndarray:
    """`255 - luminance`, float32: nuclei bright, glass dark - the benchmarked input.

    PIL's ITU-R 601 luma, the conversion the benchmark used, so production feeds the
    model exactly the pixels it was measured on.
    """
    from PIL import Image

    return (255 - np.asarray(Image.fromarray(rgb).convert("L"))).astype(np.float32)


def _eval(module: Any, rgb: np.ndarray, *, diameter_px: float) -> np.ndarray:
    masks, *_ = module.eval(inverted_grey(rgb), diameter=diameter_px, channels=[0, 0])
    return np.asarray(masks, dtype=np.int32)


def diameter_px(mpp: float) -> float:
    """The expected nucleus diameter in pixels at `mpp` (7 um at 0.5 um/px = 14 px)."""
    return settings.nuclei_cellpose_diameter_um / mpp


def _check_parity(module: Any, spec: dict[str, Any]) -> None:
    parity = spec["parity"]
    source = models_dir() / parity["input"]
    if not source.exists():
        raise ModelUnavailable(
            f"{parity['input']} is missing. It is InstanSeg's published test tile, listed "
            "in models.lock.json; run `python setup.py` to fetch it."
        )
    tile = np.load(source)[0].transpose(1, 2, 0).astype(np.uint8)
    produced = _eval(module, tile, diameter_px=float(parity["diameter_px"]))
    digest = hashlib.sha256(np.ascontiguousarray(produced).tobytes()).hexdigest()
    if digest != parity["labels_sha256"]:
        raise ParityFailure(
            "the Cellpose nuclei model loaded but did not reproduce its recorded output: "
            f"{int(produced.max())} nuclei where the manifest records "
            f"{parity['instances']}. The weights verified against their sha256, so the "
            "surrounding arithmetic changed - most likely a torch, numpy or cellpose "
            "upgrade. Nothing is served until it matches."
        )


def load() -> LoadedCellpose:
    """The model, loaded once per process and proved before it is returned."""
    global _loaded, _failure

    with _lock:
        if _loaded is not None:
            return _loaded
        if _failure is not None:
            raise ModelUnavailable(_failure)
        try:
            spec = manifest()
            checkpoint = models_dir() / CHECKPOINT
            if not checkpoint.exists():
                raise ModelUnavailable(
                    f"{CHECKPOINT} is not in {models_dir()}. It is Cellpose's published "
                    "nuclei model (BSD-3-Clause), listed in models.lock.json; run "
                    "`python setup.py`."
                )
            digest = _sha256(checkpoint)
            if digest != spec["sha256"]:
                raise ModelUnavailable(
                    f"{CHECKPOINT} does not match its manifest's sha256 "
                    f"({digest[:12]}… against {spec['sha256'][:12]}…)."
                )
            try:
                from cellpose import models as cellpose_models
            except ImportError as exc:
                raise ModelUnavailable(
                    "cellpose is not installed. Install backend/requirements-qc.txt "
                    "(cellpose itself goes in with --no-deps; see the comment there)."
                ) from exc

            module = cellpose_models.CellposeModel(gpu=False, pretrained_model=str(checkpoint))
            if float(module.diam_mean) != BASE_DIAMETER_PX:
                raise ModelUnavailable(
                    f"Cellpose loaded {CHECKPOINT} with a base diameter of "
                    f"{float(module.diam_mean):g} px instead of {BASE_DIAMETER_PX:g}: it was "
                    "treated as a custom model, which rescales every field differently. "
                    "The file must keep a name ending in 'nucleitorch_0'."
                )
            _check_parity(module, spec)
            _loaded = LoadedCellpose(module=module, manifest=spec)
            logger.info("cellpose nuclei model ready, parity %s nuclei", spec["parity"]["instances"])
            return _loaded
        except ModelUnavailable as exc:
            _failure = str(exc)
            raise


def reset() -> None:
    """Forget the loaded model. For tests."""
    global _loaded, _failure
    with _lock:
        _loaded = None
        _failure = None


def segment_array(rgb: np.ndarray, *, mpp: float) -> np.ndarray:
    """Instance labels for one `HxWx3` uint8 RGB field read at `mpp` microns per pixel."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("the segmenter takes one HxWx3 RGB field")
    loaded = load()
    with _lock:  # one forward pass at a time: Cellpose's eval is not re-entrant
        return _eval(loaded.module, rgb, diameter_px=diameter_px(mpp))


__all__ = [
    "BASE_DIAMETER_PX",
    "CHECKPOINT",
    "LoadedCellpose",
    "MANIFEST",
    "diameter_px",
    "inverted_grey",
    "load",
    "manifest",
    "reset",
    "segment_array",
]
