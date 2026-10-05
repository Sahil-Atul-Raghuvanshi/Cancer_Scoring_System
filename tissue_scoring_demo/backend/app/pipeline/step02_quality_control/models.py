"""Finding, loading and caching the two GrandQC checkpoints.

Everything torch-shaped is confined to this module and imported lazily, so the
API still starts - and every other endpoint still works - on a machine with no
torch installed and no checkpoints downloaded. What you get in that case is a
capability report saying exactly which piece is missing.

The two checkpoints are not the same kind of file, and that is worth knowing
before debugging one:

  Tissue_Detection_MPP10.pth   a plain state_dict. UnetPlusPlus on a
                               timm-efficientnet-b0 encoder, 2 classes.

  GrandQC_MPP{1,15,2}.pth      a *pickled nn.Module* - a plain Unet on the same
                               encoder, 8 channels - saved with torch.save on
                               the whole model.

The pickled ones are the awkward case. Unpickling runs against whatever
segmentation-models-pytorch and timm are installed now, which are several years
newer than the ones GrandQC saved with, so two things go wrong: names have
moved (`timm.models.layers.activations`, smp's `DecoderBlock`), and the
reconstructed object's `forward` belongs to today's class while its stored
attributes belong to the old one - it fails with things like
"'DepthwiseSeparableConv' object has no attribute 'has_skip'".

So a pickled checkpoint is treated as a *container of tensors, never as code*.
It is unpickled behind small scoped shims that bridge the moved names, its
state_dict is lifted out, the object is dropped, and the weights are loaded
into a network built from the installed library's current code. The layers that
actually run are always the maintained ones.

Both paths end in the same thing: an eval-mode module on the requested device,
plus the number of classes it actually emits - read off the segmentation head
rather than assumed, because the artefact models carry an unused channel 0 and
hard-coding 7 would silently shift every class id.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import MODELS_ROOT, REPO_ROOT, settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Encoder the published checkpoints were trained with. Not a knob - changing
#: it makes the weights unloadable.
ENCODER = "timm-efficientnet-b0"

#: ImageNet statistics for that encoder, used only if smp's own preprocessing
#: lookup is unavailable in the installed version.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

TISSUE_CHECKPOINT = "Tissue_Detection_MPP10.pth"

#: Artefact checkpoint filenames, keyed by the mpp they were trained at.
ARTEFACT_CHECKPOINTS: dict[float, str] = {
    1.0: "GrandQC_MPP1.pth",
    1.5: "GrandQC_MPP15.pth",
    2.0: "GrandQC_MPP2.pth",
}

#: Human labels for those, since the paper talks in magnifications.
MPP_LABELS: dict[float, str] = {1.0: "10x", 1.5: "7x", 2.0: "5x"}


# --- discovery ---------------------------------------------------------------


def _candidate_roots() -> list[Path]:
    """Directories that may hold a `td/` and `qc/` pair, best first."""
    roots: list[Path] = []

    if settings.qc_models_dir is not None:
        roots.append(Path(settings.qc_models_dir))

    # Anchored on `MODELS_ROOT`, not on `data_dir`. It was `data_dir.parent`, which was
    # the repository root only while the data sat inside the repository; storage now
    # lives at `<workspace>/data/demo/`, so that expression would name
    # `<workspace>/data/models/` and no checkpoint would ever be found.
    repo_root = REPO_ROOT
    roots.append(MODELS_ROOT / "grandqc")

    # An existing clone of the GrandQC repository sitting next to this project.
    # Their README tells you to put the checkpoints here, so if the user
    # followed it, this is where they already are.
    sibling = repo_root.parent / "grandqc"
    roots.append(sibling / "01_WSI_inference_OPENSLIDE_QC" / "models")
    roots.append(sibling / "02_WSI_inference_OME_TIFF_QC" / "models")

    return roots


def _find(root: Path, subdir: str, filename: str) -> Path | None:
    """Look for a checkpoint under `root/subdir/` and, failing that, `root/`."""
    for candidate in (root / subdir / filename, root / filename):
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class Checkpoints:
    """Which checkpoints were found, and where the search looked."""

    root: Path | None
    tissue: Path | None
    artefacts: dict[float, Path] = field(default_factory=dict)
    searched: tuple[Path, ...] = ()

    def artefact_for(self, model_mpp: float) -> Path | None:
        return self.artefacts.get(round(model_mpp, 1))

    @property
    def complete(self) -> bool:
        return self.tissue is not None and bool(self.artefacts)


def discover_checkpoints() -> Checkpoints:
    """Locate the GrandQC weights without loading anything.

    The first root that yields *any* checkpoint wins, so a half-populated
    directory is reported as half-populated rather than silently completed
    from a second location - which would make it very hard to tell which
    weights actually ran.
    """
    searched = _candidate_roots()

    for root in searched:
        if not root.is_dir():
            continue

        tissue = _find(root, "td", TISSUE_CHECKPOINT)
        artefacts = {
            mpp: path
            for mpp, name in ARTEFACT_CHECKPOINTS.items()
            if (path := _find(root, "qc", name)) is not None
        }

        if tissue is not None or artefacts:
            return Checkpoints(
                root=root, tissue=tissue, artefacts=artefacts, searched=tuple(searched)
            )

    return Checkpoints(root=None, tissue=None, artefacts={}, searched=tuple(searched))


# --- runtime -----------------------------------------------------------------


@dataclass(frozen=True)
class Runtime:
    """What the torch side of the world looks like right now."""

    torch_version: str | None
    smp_version: str | None
    device: str
    device_name: str | None
    problem: str | None

    @property
    def usable(self) -> bool:
        return self.problem is None


def _pick_device() -> tuple[str, str | None]:
    """CUDA, then Apple Silicon, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", "Apple Silicon (MPS)"
    return "cpu", None


def runtime() -> Runtime:
    """Report the torch stack, without raising when it is absent."""
    try:
        import torch
    except ImportError as exc:
        return Runtime(None, None, "cpu", None, f"PyTorch is not installed ({exc})")

    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        return Runtime(
            torch.__version__,
            None,
            "cpu",
            None,
            f"segmentation-models-pytorch is not installed ({exc})",
        )

    device, name = _pick_device()
    return Runtime(torch.__version__, getattr(smp, "__version__", "unknown"), device, name, None)


# --- loading -----------------------------------------------------------------

_CACHE: dict[tuple[str, str], Any] = {}
_CACHE_LOCK = threading.Lock()


class ModelError(RuntimeError):
    """A checkpoint could not be loaded. The message is meant for the user."""


def _classes_from_state_dict(state: dict[str, Any]) -> int:
    """Read the output-channel count off the segmentation head."""
    for key, value in state.items():
        if key.endswith("segmentation_head.0.weight"):
            return int(value.shape[0])
    raise ModelError(
        "checkpoint has no recognisable segmentation head - it does not look like a "
        "segmentation-models-pytorch model"
    )


def _normalise_state_dict(raw: Any) -> dict[str, Any]:
    """Unwrap the usual containers a state_dict arrives in."""
    state = raw
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            inner = state.get(key)
            if isinstance(inner, dict):
                state = inner
                break
    if not isinstance(state, dict):
        raise ModelError(f"expected a state_dict, got {type(raw).__name__}")

    # DataParallel prefixes every key; strip it so the plain module accepts them.
    if all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    return state


#: The two decoder architectures GrandQC ships. The tissue detector is a
#: UnetPlusPlus; the artefact models are plain Unets. Detected rather than
#: assumed - loading a checkpoint into the wrong one fails loudly, but only
#: because both are tried and an exact match is required.
ARCHITECTURES = ("UnetPlusPlus", "Unet")


def _build(architecture: str, classes: int) -> Any:
    """A GrandQC-shaped network with randomly initialised weights.

    `encoder_weights=None` on purpose: every parameter is about to be
    overwritten by the checkpoint, and asking for `imagenet` here would make
    model loading depend on a download from the internet for no benefit.
    """
    import segmentation_models_pytorch as smp

    if architecture not in ARCHITECTURES:
        raise ModelError(f"unsupported architecture {architecture!r}")

    return getattr(smp, architecture)(
        encoder_name=ENCODER, encoder_weights=None, classes=classes, activation=None
    )


def _rebuild(state: dict[str, Any], classes: int, *, prefer: str | None) -> tuple[Any, str]:
    """Build a fresh network of the right shape and pour the weights into it.

    Only an *exact* fit is accepted - no missing and no unexpected parameters.
    That strictness is what makes architecture detection trustworthy: a Unet
    and a UnetPlusPlus have different decoder parameter names, so exactly one
    of them can absorb a given checkpoint cleanly.
    """
    order = [name for name in (prefer, *ARCHITECTURES) if name in ARCHITECTURES]
    seen: list[str] = []
    problems: list[str] = []

    for architecture in order:
        if architecture in seen:
            continue
        seen.append(architecture)

        model = _build(architecture, classes)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if not missing and not unexpected:
            return model, architecture

        problems.append(
            f"{architecture}: {len(missing)} missing, {len(unexpected)} unexpected"
        )

    raise ModelError(
        f"checkpoint fits neither architecture ({'; '.join(problems)}). It was not "
        f"trained on {ENCODER}, or it is not a GrandQC checkpoint."
    )


# --- unpickling a model saved against an older timm ---------------------------
#
# The artefact checkpoints were pickled when timm still exposed
# `timm.models.layers.*` and `timm.models.efficientnet_blocks`. timm 1.x moved
# both - to `timm.layers.*` and `timm.models._efficientnet_blocks` - and a
# pickle stores module paths as plain strings, so unpickling one now fails with
# `No module named 'timm.models.layers.activations'`.
#
# The classes themselves are unchanged; only their address is. So rather than
# pinning timm back below what segmentation-models-pytorch requires, the old
# addresses are resolved to the new ones for the duration of the load.
#
# The finder is appended to the *end* of sys.meta_path, so it is consulted only
# after every real finder has failed. It can therefore never shadow a module
# that genuinely exists.


def _modern_timm_name(fullname: str) -> str | None:
    """The current home of a timm module that has moved, or None."""
    if fullname.startswith("timm.models.layers."):
        return "timm.layers." + fullname.removeprefix("timm.models.layers.")

    # timm 0.9 made the internal model-building modules private.
    if fullname.startswith("timm.models."):
        tail = fullname.removeprefix("timm.models.")
        if "." not in tail and not tail.startswith("_"):
            return f"timm.models._{tail}"

    return None


class _MovedTimmModules:
    """A meta-path finder that resolves pre-1.0 timm module paths."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        modern = _modern_timm_name(fullname)
        if modern is None:
            return None

        import importlib
        import importlib.util

        try:
            module = importlib.import_module(modern)
        except ImportError:
            return None

        logger.debug("resolving legacy module %s to %s", fullname, modern)
        loader = _AliasLoader(module)
        return importlib.util.spec_from_loader(fullname, loader)


class _AliasLoader:
    """Loader that hands back an already-imported module under another name."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def create_module(self, spec: Any) -> Any:
        return self._module

    def exec_module(self, module: Any) -> None:
        return None


#: Classes segmentation-models-pytorch renamed in 0.5.0. The pickle names the
#: old ones, and unpickling instantiates them, so they have to resolve to
#: something. Pointing them at the renamed class is safe *because* nothing
#: unpickled is executed: pickle restores instance state directly without
#: calling __init__, and the object is discarded as soon as its tensors have
#: been read out.
_RENAMED_SMP_CLASSES: dict[str, dict[str, str]] = {
    "segmentation_models_pytorch.decoders.unet.decoder": {
        "DecoderBlock": "UnetDecoderBlock",
        "CenterBlock": "UnetCenterBlock",
    },
}


class _legacy_paths:  # noqa: N801 - used as a context manager, not a type
    """Bridge pre-1.0 timm module paths and pre-0.5 smp class names.

    Both bridges are installed only for the duration of one unpickle and
    removed afterwards, so nothing here changes how the rest of the process
    imports anything.
    """

    def __enter__(self) -> None:
        import importlib
        import sys

        self._finder = _MovedTimmModules()
        sys.meta_path.append(self._finder)

        self._patched: list[tuple[Any, str]] = []
        for module_name, renames in _RENAMED_SMP_CLASSES.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            for old, new in renames.items():
                if not hasattr(module, old) and hasattr(module, new):
                    setattr(module, old, getattr(module, new))
                    self._patched.append((module, old))

    def __exit__(self, *_: object) -> None:
        import sys

        with contextlib.suppress(ValueError):
            sys.meta_path.remove(self._finder)
        for module, name in self._patched:
            with contextlib.suppress(AttributeError):
                delattr(module, name)


def _torch_load(path: Path) -> Any:
    """`torch.load`, spelt so it works on both sides of the 2.6 default change.

    torch 2.6 flipped `weights_only` to True. The artefact checkpoints are
    pickled modules, so they need it False - which is safe here only because
    these files come from a source the operator chose to download.
    """
    import torch

    with _legacy_paths():
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # torch < 1.13 has no weights_only parameter at all.
            return torch.load(path, map_location="cpu")


def _finish(model: Any, device: str) -> Any:
    model.eval()
    model.to(device)
    return model


def _load(path: Path, device: str, *, expect_classes: int | None) -> tuple[Any, int]:
    """Load one checkpoint, whether it is a state_dict or a pickled module."""
    try:
        raw = _torch_load(path)
    except ModuleNotFoundError as exc:
        raise ModelError(
            f"{path.name} is a pickled model and unpickling it needs a module that is not "
            f"installed: {exc}. Install segmentation-models-pytorch (and timm) into this "
            "environment - see backend/requirements-qc.txt."
        ) from exc
    except AttributeError as exc:
        raise ModelError(
            f"{path.name} is a pickled model saved against a different version of "
            f"segmentation-models-pytorch, and a class it references has moved: {exc}. "
            "Pin the version listed in backend/requirements-qc.txt."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        raise ModelError(f"could not read {path.name}: {exc}") from exc

    import torch

    # A pickled checkpoint is treated strictly as a container of tensors, never
    # as code to run. Its classes come from whatever segmentation-models-pytorch
    # and timm are installed *now*, which are years newer than the ones it was
    # saved against, so the reconstructed object's `forward` can reference
    # attributes the stored instance never had - it fails with things like
    # "'DepthwiseSeparableConv' object has no attribute 'has_skip'".
    #
    # So the weights are lifted out and poured into a network built from the
    # installed library's current code. Unpickling still has to succeed, but
    # nothing unpickled is ever executed, and the layer implementations that do
    # run are the maintained ones.
    if isinstance(raw, torch.nn.Module):
        prefer: str | None = type(raw).__name__
        state = raw.state_dict()
        del raw
    else:
        prefer = None
        state = _normalise_state_dict(raw)

    classes = _classes_from_state_dict(state)

    if expect_classes is not None and classes != expect_classes:
        raise ModelError(
            f"{path.name} emits {classes} classes, expected {expect_classes} - "
            "this is not the checkpoint it is named after"
        )

    model, architecture = _rebuild(state, classes, prefer=prefer)
    logger.info("%s: %s, %d classes", path.name, architecture, classes)

    return _finish(model, device), classes


@dataclass
class LoadedModel:
    """A model that is on the device and ready to run."""

    model: Any
    classes: int
    path: Path
    device: str


def _cached(path: Path, device: str, *, expect_classes: int | None) -> LoadedModel:
    """Load once per (file, device); a load costs seconds and a run needs both."""
    key = (str(path), device)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit

    logger.info("loading QC checkpoint %s onto %s", path.name, device)
    model, classes = _load(path, device, expect_classes=expect_classes)
    loaded = LoadedModel(model=model, classes=classes, path=path, device=device)

    with _CACHE_LOCK:
        _CACHE[key] = loaded
    return loaded


def load_tissue_model(path: Path, device: str) -> LoadedModel:
    """GrandQC tissue detection: 2 classes, 0 = tissue, 1 = background."""
    return _cached(path, device, expect_classes=2)


def load_artefact_model(path: Path, device: str) -> LoadedModel:
    """GrandQC artefact segmentation. Class count is read off the checkpoint."""
    return _cached(path, device, expect_classes=None)


def unload_all() -> None:
    """Drop every cached model. Called at shutdown."""
    with _CACHE_LOCK:
        _CACHE.clear()


# --- preprocessing -----------------------------------------------------------


def preprocessing_fn() -> Any:
    """The encoder's own input normalisation.

    Prefers smp's lookup so the numbers track the encoder definition, and falls
    back to the ImageNet constants that lookup would have returned - the point
    of the fallback is to survive an smp release that moves the helper, not to
    guess at unknown statistics.
    """
    import numpy as np

    try:
        import segmentation_models_pytorch as smp

        return smp.encoders.get_preprocessing_fn(ENCODER, "imagenet")
    except Exception:  # noqa: BLE001 - any smp-layout change lands here
        logger.warning("smp preprocessing lookup unavailable; using ImageNet constants")

        mean = np.asarray(_IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(_IMAGENET_STD, dtype=np.float32)

        def normalise(image: Any) -> Any:
            return (np.asarray(image, dtype=np.float32) / 255.0 - mean) / std

        return normalise
