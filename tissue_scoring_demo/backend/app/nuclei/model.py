"""Loading InstanSeg, and the one transform it cannot be run without.

**The preprocessing is not a detail.** Fed the raw 0-255 tile the upstream ships
as its own test input, this model returns an empty label map - no nuclei, no
warning, no error. Apply `scale_range` first and the same tile returns 336, which
is exactly the reference output byte for byte. So the difference between "this
model works" and "this tissue apparently has no cells" is one percentile stretch,
and a silent zero is the most dangerous possible failure for a step whose job is
to produce a denominator.

That is why loading is not just `torch.jit.load`:

  * the checkpoint is checked against the manifest's sha256 before it is trusted,
  * and then it is *run*, on the upstream's own test pair, and required to
    reproduce the published output exactly.

The parity run costs about a second, once per process. It is cheap against being
told a slide has no nuclei because a torch release changed an operator.

Nothing in this module knows what a slide or a region is. Arrays in, label maps
out.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import MODELS_ROOT, settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: File names inside `models/nuclei/`, so the four move together.
CHECKPOINT = "instanseg_brightfield_nuclei.pt"
MANIFEST = "instanseg_brightfield_nuclei.manifest.json"
PARITY_INPUT = "parity-input.npy"
PARITY_OUTPUT = "parity-output.npy"


class ModelUnavailable(RuntimeError):
    """The segmenter cannot be loaded. Reported to the UI, never raised at a viewer."""


class ParityFailure(ModelUnavailable):
    """It loaded, and then did not reproduce the output it is published with.

    Its own type because it means something different from "not installed": the
    weights are present and the arithmetic has changed under them. Serving a
    model in that state would produce counts nobody could compare with anything.
    """


def models_dir() -> Path:
    """Where the nuclei checkpoint lives, beside step 2's and step 8's.

    Anchored on `MODELS_ROOT` rather than on `data_dir`, for the reason step 8's
    equivalent records: storage moved out of the repository, so any expression
    reaching the models through the data directory now names the wrong tree.
    """
    if settings.nuclei_models_dir is not None:
        return Path(settings.nuclei_models_dir)
    return MODELS_ROOT / "nuclei"


def manifest() -> dict[str, Any]:
    """The loading contract, read from disk rather than duplicated in code."""
    path = models_dir() / MANIFEST
    if not path.exists():
        raise ModelUnavailable(
            f"{MANIFEST} is missing from {models_dir()}. It is committed to the "
            "repository, so a missing manifest means the checkout is incomplete "
            "rather than the download being."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def scale_range(
    image: np.ndarray,
    *,
    min_percentile: float,
    max_percentile: float,
    eps: float,
) -> np.ndarray:
    """The model's declared input normalisation: a per-channel percentile stretch.

    `image` is NCHW float. Each channel is mapped so that its `min_percentile`
    lands on 0 and its `max_percentile` on 1, percentiles being taken over the
    spatial axes of the whole batch exactly as the upstream `rdf.yaml` specifies
    (`axes: [x, y]`).

    Percentiles rather than min/max because a single dust speck or a saturated
    highlight would otherwise set the scale for the whole tile. And *per channel*
    because on a deconvolved or heavily stained field the three channels do not
    share a range - stretching them together would preserve a colour cast that
    the model was never trained to see through.
    """
    out = np.asarray(image, dtype=np.float32).copy()
    for channel in range(out.shape[1]):
        plane = out[:, channel]
        low, high = np.percentile(plane, [min_percentile, max_percentile])
        out[:, channel] = (plane - low) / (high - low + eps)
    return out


@dataclass(frozen=True)
class LoadedModel:
    """A TorchScript module that has proved itself, plus the contract it met."""

    module: Any
    manifest: dict[str, Any]

    @property
    def mpp(self) -> float:
        return float(self.manifest["input"]["mpp"])


_lock = threading.Lock()
_loaded: LoadedModel | None = None
_failure: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _run(module: Any, batch: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """One forward pass, normalisation included. NCHW float in, NHW int32 out."""
    import torch

    prepared = scale_range(
        batch,
        min_percentile=float(spec["min_percentile"]),
        max_percentile=float(spec["max_percentile"]),
        eps=float(spec["eps"]),
    )
    with torch.no_grad():
        raw = module(torch.from_numpy(prepared))

    # TorchScript returns a tensor here, but the upstream has shipped a tuple in
    # the past and may again; take the first output either way rather than
    # depending on which.
    tensor = raw[0] if isinstance(raw, list | tuple) else raw
    array = tensor.detach().cpu().numpy()

    # NCHW with a single channel -> NHW. Labels are integers stored as float.
    if array.ndim == 4:
        array = array[:, 0]
    return array.astype(np.int32)


def _check_parity(module: Any, spec: dict[str, Any], parity: dict[str, Any]) -> None:
    """Run the upstream's own test pair and require the published answer."""
    root = models_dir()
    source = root / parity["input"]
    expected_path = root / parity["output"]
    if not source.exists() or not expected_path.exists():
        raise ModelUnavailable(
            "the parity tensors are missing. They are listed in models.lock.json; "
            "run `python setup.py` to fetch them."
        )

    produced = _run(module, np.load(source).astype(np.float32), spec)
    expected = np.load(expected_path)
    if expected.ndim == 4:
        expected = expected[:, 0]
    expected = expected.astype(np.int32)

    if not np.array_equal(produced, expected):
        found = int(produced.max())
        want = int(expected.max())
        raise ParityFailure(
            "the nuclei model loaded but did not reproduce its published output: "
            f"{found} instances where the reference has {want}. The weights verified "
            "against their sha256, so this is the surrounding arithmetic changing, "
            "not a corrupt download - most likely a torch upgrade. Nothing is served "
            "until it matches, because a model that quietly segments differently "
            "produces counts that cannot be compared with any run before it."
        )


def load() -> LoadedModel:
    """The segmenter, loaded once per process and proved before it is returned."""
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
                    f"{CHECKPOINT} is not in {models_dir()}. It is a 15 MB Apache-2.0 "
                    "download listed in models.lock.json; run `python setup.py`."
                )

            digest = _sha256(checkpoint)
            if digest != spec["sha256"]:
                raise ModelUnavailable(
                    f"{CHECKPOINT} does not match the sha256 its manifest declares "
                    f"({digest[:12]}… against {spec['sha256'][:12]}…). Delete it and "
                    "re-run setup rather than serving an unknown checkpoint."
                )

            try:
                import torch
            except ImportError as exc:  # pragma: no cover - torch is a qc dependency
                raise ModelUnavailable(
                    "torch is not installed. Step 11 needs it for the same reason "
                    "step 2 does; install backend/requirements-qc.txt."
                ) from exc

            module = torch.jit.load(str(checkpoint), map_location="cpu")
            module.eval()

            _check_parity(module, spec["input"]["scale_range"], spec["parity"])

            _loaded = LoadedModel(module=module, manifest=spec)
            logger.info(
                "nuclei model ready: %s %s, parity %s instances",
                spec["name"],
                spec["version"],
                spec["parity"]["labels"],
            )
            return _loaded
        except ModelUnavailable as exc:
            _failure = str(exc)
            raise


def reset() -> None:
    """Forget the loaded model. For tests, which swap the models directory."""
    global _loaded, _failure
    with _lock:
        _loaded = None
        _failure = None


def segment_array(rgb: np.ndarray) -> np.ndarray:
    """Instance labels for one `HxWx3` uint8 RGB field read at the model's mpp.

    Returns an `HxW` int32 label map: 0 is background, every other value is one
    nucleus. Labels are dense from 1 but carry no meaning beyond identity.

    The caller is responsible for having read the pixels at 0.5 microns per
    pixel - `manifest()["input"]["mpp"]` - because this function has no way to
    know what scale it is being handed, and the model is scale-specific.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("the segmenter takes one HxWx3 RGB field")

    loaded = load()
    batch = rgb.astype(np.float32).transpose(2, 0, 1)[None]
    labels = _run(loaded.module, batch, loaded.manifest["input"]["scale_range"])
    return labels[0]


__all__ = [
    "CHECKPOINT",
    "LoadedModel",
    "MANIFEST",
    "ModelUnavailable",
    "ParityFailure",
    "load",
    "manifest",
    "models_dir",
    "reset",
    "scale_range",
    "segment_array",
]
