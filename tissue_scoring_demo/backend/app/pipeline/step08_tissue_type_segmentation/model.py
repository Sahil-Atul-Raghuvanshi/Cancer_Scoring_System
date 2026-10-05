"""Finding, checking and loading the region-model checkpoint.

Everything torch-shaped is confined to this module and imported lazily, so the API
still starts - and every other endpoint still works - on a machine with no torch
installed and no checkpoint published. What you get in that case is a capability
report naming exactly which piece is missing, the same shape step 2 uses for GrandQC.

A checkpoint here is a **state_dict plus a manifest**, never a pickled `nn.Module`.
Step 2 has to unpickle foreign checkpoints and goes to real trouble to lift tensors
out without executing anything; this project's own model has no excuse to inflict the
same problem, so it loads under `weights_only=True` and always will.

Four things are checked before a checkpoint is allowed to score a slide, and each one
is a refusal rather than a warning:

  **the bytes**       the file's SHA-256 must equal the one its manifest recorded. A
                      pipeline that runs whatever bytes happen to be at a path cannot
                      say which model produced a number, and this one feeds a test
                      that helps decide whether someone receives chemotherapy.
  **the class order** `classes.verify_order`. A permuted head still produces a
                      plausible class map and would put stroma in the denominator.
  **the input**       the manifest's `input` block must equal `input.descriptor(...)`
                      rebuilt from that same block's geometry. This is what catches a
                      model fitted with `standardise` on and served with it off - the
                      quietest failure in the whole plan, invisible in every metric
                      computed inside the training process.
  **the licence**     read out of the manifest and carried into the report. Not a
                      refusal, but see below.

**The licence track is data, not a comment.** Two checkpoints in `models/tissue_type/`
are indistinguishable by looking at them: one is trained on BCSS alone (CC0 +
BSD-3, commercially clean) and one adds BRACS regions labelled by BEETLE's nnU-Net
(non-commercial, ShareAlike). A model fitted on those inherits both restrictions and
so does every number computed from it. `Pinned.licence_track` is therefore read from
the manifest and travels all the way to the screen, because a non-commercial
dependency reaching a client is exactly what happens when the only record of it is a
sentence in a README.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.config import MODELS_ROOT, settings
from app.core.logging import get_logger

from . import branches
from . import input as model_input
from .classes import CLASS_NAMES, verify_order

logger = get_logger(__name__)

#: Feature width of a ResNet18's pooled body, so the head can be built before the
#: state dict is read. Not a knob - it is the architecture's.
FEATURE_DIM = 512

#: Wording in a manifest licence field that makes a checkpoint research-only. Matched
#: case-insensitively against every licence string, because the terms arrive from
#: several upstream projects and none of them phrase it the same way.
_RESTRICTED_MARKERS = ("non-commercial", "noncommercial", "research only", "-nc-", "nc-sa")


class ModelError(RuntimeError):
    """A checkpoint could not be found, verified or loaded. The message is for the user."""


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file, in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- discovery ---------------------------------------------------------------


def models_dir() -> Path:
    """Where published checkpoints live.

    The training folder publishes *here* rather than keeping its own copy, so the
    trained model and the served model are the same file on disk. That is the whole
    reason this is one directory and not two.

    Anchored on `MODELS_ROOT`, not on `data_dir`. It used to be `data_dir.parent`, which
    was the repository root only for as long as the data sat inside the repository;
    once storage moved to `<workspace>/data/demo/` that expression started naming
    `<workspace>/data/models/`, and the checkpoints silently stopped being found.
    """
    if settings.tissue_type_models_dir is not None:
        return Path(settings.tissue_type_models_dir)
    return MODELS_ROOT / "tissue_type"


@dataclass(frozen=True)
class Candidate:
    """One checkpoint found on disk, described from its manifest without loading torch.

    Cheap by design: the capability endpoint lists every published model with its
    licence track and its headline metric, and doing that must not cost a 45 MB read
    per model or require torch to be installed.
    """

    name: str
    checkpoint: Path
    manifest_path: Path

    #: From the manifest. `None` when the manifest is unreadable, in which case
    #: `problem` says so and the candidate cannot be loaded.
    init: str | None = None
    tile_px: int | None = None
    mpp: float | None = None
    standardise: bool | None = None
    invert_polarity: bool | None = None
    created: str | None = None
    sha256: str | None = None
    bytes: int | None = None

    #: The architecture the manifest records. Every served checkpoint answers **once
    #: per window**: the per-pixel U-Net arm was removed with `pixel_unet_resnet18`, and
    #: an architecture this file does not know is built as the plain ResNet18, where
    #: `strict=True` turns the mismatch into a refusal at load rather than a bad score.
    arch: str = "resnet18"

    #: Which input contract the manifest declares - the fact that decides whether
    #: this checkpoint wants optical density or a colour photograph. Defaulted rather
    #: than optional because a manifest recording nothing *is* a haematoxylin
    #: checkpoint, and `branch` below is what step 7 filters on. This is why the
    #: branch can be resolved without loading torch.
    channel: str = model_input.CHANNEL_HAEMATOXYLIN

    licence_track: str = "unknown"
    licences: dict[str, str] = field(default_factory=dict)
    training_source: str | None = None

    #: The headline numbers, so a chooser can be made without loading anything.
    held_out_accuracy: float | None = None
    dice_invasive: float | None = None
    dice_non_invasive: float | None = None
    non_invasive_tiles: int | None = None

    problem: str | None = None

    @property
    def usable(self) -> bool:
        return self.problem is None

    @property
    def branch(self) -> branches.ModelBranch:
        """Which of step 7's options this checkpoint belongs to.

        Derived from `channel` and never from `name`. Two heads now share a geometry
        at every field of view - one per branch - so matching on `tile_px * mpp`
        alone would let alphabetical order decide which one runs, and a colour model
        fed optical density scores badly in a way that looks like a modelling result.
        """
        return branches.branch_of_channel(self.channel)


def _licence_track(manifest: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Read a checkpoint's effective licence out of its manifest.

    Restrictive wins, always, and the whole set of terms travels with the verdict so
    the report can say *which* dependency caused it rather than only that one did.
    """
    provenance = manifest.get("provenance") or {}
    licences = {
        str(key): str(value) for key, value in (provenance.get("licences") or {}).items()
    }

    position = str((manifest.get("training_data") or {}).get("licence_position", ""))
    haystack = " ".join([*licences.values(), position]).lower()

    if not licences and not position:
        return "unknown", licences
    if any(marker in haystack for marker in _RESTRICTED_MARKERS):
        return "research-only", licences
    return "permissive", licences


def _describe(manifest_path: Path) -> Candidate:
    """One candidate, read from its manifest. Never raises - `problem` carries it."""
    name = manifest_path.name.removesuffix(".manifest.json")
    checkpoint = manifest_path.with_name(f"{name}.pt")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Candidate(
            name=name,
            checkpoint=checkpoint,
            manifest_path=manifest_path,
            problem=f"its manifest is unreadable ({exc})",
        )

    if not checkpoint.is_file():
        return Candidate(
            name=name,
            checkpoint=checkpoint,
            manifest_path=manifest_path,
            problem=(
                f"{checkpoint.name} is missing. A manifest without its checkpoint is a "
                "model nobody can run, so both are required."
            ),
        )

    spec = manifest.get("input") or {}
    held_out = (manifest.get("metrics") or {}).get("held_out") or {}
    per_class = held_out.get("per_class_dice") or {}
    by_source = held_out.get("by_source") or {}
    bcss = by_source.get("bcss") or {}
    track, licences = _licence_track(manifest)

    arch = str(manifest.get("arch", "resnet18"))
    return Candidate(
        name=name,
        checkpoint=checkpoint,
        manifest_path=manifest_path,
        init=manifest.get("init"),
        tile_px=spec.get("tile_px"),
        mpp=spec.get("mpp"),
        standardise=spec.get("standardise_tile_p99"),
        invert_polarity=spec.get("invert_polarity"),
        channel=str(spec.get("channel", model_input.CHANNEL_HAEMATOXYLIN)),
        created=manifest.get("created"),
        sha256=manifest.get("sha256"),
        bytes=checkpoint.stat().st_size,
        arch=arch,
        licence_track=track,
        licences=licences,
        training_source=(manifest.get("training_data") or {}).get("source"),
        held_out_accuracy=held_out.get("accuracy"),
        dice_invasive=per_class.get("invasive_epithelium"),
        dice_non_invasive=per_class.get("non_invasive_epithelium"),
        # Class 1's tile count on the human-labelled source, because a Dice figure
        # ships with its tile count or it does not ship - and on BCSS alone that
        # count is nine.
        non_invasive_tiles=(bcss.get("class_tiles") or {}).get("non_invasive_epithelium"),
    )


def discover() -> list[Candidate]:
    """Every published checkpoint, best-known first, described without loading torch.

    Ordered with the configured default first and then by name, so the report's list
    reads as "the one that will run, and the alternatives" rather than as a directory
    listing whose order depends on the filesystem.
    """
    directory = models_dir()
    if not directory.is_dir():
        return []

    found = [_describe(path) for path in sorted(directory.glob("*.manifest.json"))]
    default = settings.tissue_type_model
    found.sort(key=lambda candidate: (candidate.name != default, candidate.name))
    return found


def find(name: str) -> Candidate | None:
    """One candidate by name, or None."""
    for candidate in discover():
        if candidate.name == name:
            return candidate
    return None


# --- runtime -----------------------------------------------------------------


@dataclass(frozen=True)
class Runtime:
    """What the torch side of the world looks like right now."""

    torch_version: str | None
    device: str
    device_name: str | None
    threads: int | None
    problem: str | None

    @property
    def usable(self) -> bool:
        return self.problem is None


def runtime() -> Runtime:
    """Report the torch stack without raising when it is absent."""
    try:
        import torch
    except ImportError as exc:
        return Runtime(None, "cpu", None, None, f"PyTorch is not installed ({exc})")

    try:
        import torchvision  # noqa: F401 - presence is what is being tested
    except ImportError as exc:
        return Runtime(
            torch.__version__, "cpu", None, None, f"torchvision is not installed ({exc})"
        )

    if torch.cuda.is_available():
        return Runtime(
            torch.__version__, "cuda", torch.cuda.get_device_name(0), None, None
        )
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return Runtime(torch.__version__, "mps", "Apple Silicon (MPS)", None, None)
    return Runtime(torch.__version__, "cpu", None, torch.get_num_threads(), None)


# --- loading -----------------------------------------------------------------


@dataclass
class Pinned:
    """A loaded, verified checkpoint and everything serving needs to know about it.

    `net` is deliberately typed loosely: this module is the only place that may import
    torch, and annotating it `torch.nn.Module` would put the import in the signature
    of a dataclass the service layer holds.
    """

    name: str
    net: Any
    manifest: dict[str, Any]

    #: The input contract, lifted out of the manifest after it has been checked
    #: against `input.descriptor`. Serving reads these and never the settings.
    tile_px: int
    mpp: float
    standardise: bool
    invert_polarity: bool
    gamma: float

    #: The verified input contract's channel. Serving branches on this to decide
    #: whether to hand the network deconvolved density or the tile's own RGB, so it
    #: is read from the manifest that passed `_verify_input_contract` and never from
    #: a setting.
    channel: str

    sha256: str
    licence_track: str
    licences: dict[str, str]

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("classes") or CLASS_NAMES)

    @property
    def arch(self) -> str:
        return str(self.manifest.get("arch", "resnet18"))

    @property
    def branch(self) -> branches.ModelBranch:
        """Which of step 7's options actually ran, for the run record."""
        return branches.branch_of_channel(self.channel)

    @property
    def tau(self) -> float | None:
        """The invasive threshold this checkpoint was validated under, if it has one.

        **A model with a tau served under a plain argmax is a different, worse model**,
        and nothing else in the pipeline would notice: the class map would still be
        three-valued, the confidences would still look reasonable, and only the invasive
        area - the number the whole score is gated on - would be wrong.

        The rule exists because a three-way argmax lets a badly-conditioned in-situ head
        outvote invasive. On `CAN_00251_26_H&E` the shipped v2 model called 45.9% of the
        section in-situ, and for 88.5% of those windows the runner-up was
        non-epithelium - p(invasive) was *last*, so thresholding p(invasive) drops them
        correctly while a three-way argmax cannot.

        `None` for every checkpoint published before v3, which were validated under
        argmax and must keep being served under it.
        """
        rule = self.manifest.get("metrics", {}).get("decision_rule") or {}
        value = rule.get("tau")
        return float(value) if value is not None else None


_CACHE: dict[str, Pinned] = {}
_CACHE_LOCK = threading.Lock()


def _verify_input_contract(spec: dict[str, Any], *, name: str) -> None:
    """Refuse a manifest whose input block this code cannot reproduce.

    The comparison is against `descriptor` rebuilt from the manifest's *own* geometry
    and polarity, so what is actually being asserted is that nothing else in the
    contract has moved: the clip, the floor, the stain basis, the normalisation
    statistics, and both standardisation constants. If someone retunes
    `STANDARDISE_TARGET` here without re-fitting, this is what stops the old
    checkpoint being served under the new transform.
    """
    required = (
        "channel",
        "tile_px",
        "mpp",
        "invert_polarity",
        "gamma",
        "standardise_tile_p99",
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise ModelError(
            f"{name}'s manifest does not record {missing} in its `input` block, so there "
            "is no way to tell what transform it was fitted under. Re-publish it with a "
            "full input descriptor - serving a model whose input contract is unknown is "
            "the one failure this check exists to prevent."
        )

    channel = str(spec["channel"])
    try:
        expected = model_input.descriptor(
            tile_px=int(spec["tile_px"]),
            mpp=float(spec["mpp"]),
            invert=bool(spec["invert_polarity"]),
            gamma=float(spec["gamma"]),
            standardise=bool(spec["standardise_tile_p99"]),
            channel=channel,
        )
    except ValueError as exc:
        # An unknown channel, or a channel combined with a transform it does not have
        # - `rgb_he` with a shape term, say. Either way this code cannot compute the
        # input the checkpoint claims, which is the same refusal as a disagreement
        # below, reached one step earlier.
        raise ModelError(
            f"{name}'s manifest declares an input contract this code cannot build: "
            f"{exc}"
        ) from None

    disagreements = {
        key: (spec.get(key), value)
        for key, value in expected.items()
        if spec.get(key) != value
    }
    if disagreements:
        lines = "\n".join(
            f"  {key}: manifest {found!r}, this code {wanted!r}"
            for key, (found, wanted) in sorted(disagreements.items())
        )
        raise ModelError(
            f"{name} was fitted under a different input transform from the one this code "
            f"computes:\n{lines}\n"
            "Refusing to score with it. A model served an input it never saw loses "
            "accuracy silently, and the loss looks like a modelling failure rather than "
            "the plumbing failure it is."
        )


def load_pinned(name: str | None = None, *, verify: bool = True) -> Pinned:
    """Load a published checkpoint, refusing one that fails any of the four checks.

    Cached per name for the process, because the load is a 45 MB read plus a hash of
    the same 45 MB and a slide is scored in tens of thousands of batches against one
    model. `verify=False` skips only the hash - never the class order or the input
    contract - and exists for the test suite, which builds throwaway checkpoints whose
    bytes are not worth hashing twice.
    """
    chosen = name or settings.tissue_type_model

    with _CACHE_LOCK:
        cached = _CACHE.get(chosen)
    if cached is not None:
        return cached

    state = runtime()
    if not state.usable:
        raise ModelError(str(state.problem))

    candidate = find(chosen)
    if candidate is None:
        available = [entry.name for entry in discover()]
        raise ModelError(
            f"no checkpoint named {chosen!r} in {models_dir()}. "
            + (
                f"Published there: {available}."
                if available
                else "That directory holds no published checkpoint at all - train one "
                "with tissue_type_model_training, whose publish step writes here so the "
                "trained model and the served model are the same file."
            )
        )
    if not candidate.usable:
        raise ModelError(f"{chosen} cannot be loaded: {candidate.problem}")

    manifest = json.loads(candidate.manifest_path.read_text(encoding="utf-8"))

    if verify:
        actual = sha256_of(candidate.checkpoint)
        if actual != manifest.get("sha256"):
            raise ModelError(
                f"{candidate.checkpoint.name} hashes to {actual[:16]}... but its manifest "
                f"records {str(manifest.get('sha256'))[:16]}.... Refusing to score with a "
                "model that is not the one that was validated."
            )

    verify_order(manifest.get("classes") or [])
    spec = manifest.get("input") or {}
    _verify_input_contract(spec, name=chosen)

    net = _build_and_load(candidate.checkpoint, classes=len(manifest["classes"]),
                          arch=str(manifest.get("arch", "resnet18")))
    track, licences = _licence_track(manifest)

    pinned = Pinned(
        name=chosen,
        net=net,
        manifest=manifest,
        tile_px=int(spec["tile_px"]),
        mpp=float(spec["mpp"]),
        standardise=bool(spec["standardise_tile_p99"]),
        invert_polarity=bool(spec["invert_polarity"]),
        gamma=float(spec["gamma"]),
        channel=str(spec.get("channel", model_input.CHANNEL_HAEMATOXYLIN)),
        sha256=str(manifest.get("sha256")),
        licence_track=track,
        licences=licences,
    )

    # `checkpoint`, not `name`: `name` is a reserved `LogRecord` attribute and
    # `logging` raises `KeyError: Attempt to overwrite 'name' in LogRecord` when it is
    # passed in `extra`. That raise happens inside `makeRecord`, so it only fires when
    # the record is actually built - which means it is invisible to every script and
    # test that leaves the root logger at its default WARNING, and fatal in the app,
    # which configures INFO. It reached production as a 500 on every step 8 run.
    logger.info(
        "tissue_type.model_loaded",
        extra={
            "checkpoint": chosen,
            "licence_track": track,
            "tile_px": pinned.tile_px,
            "standardise": pinned.standardise,
        },
    )

    with _CACHE_LOCK:
        _CACHE[chosen] = pinned
    return pinned


#: The architecture string a two-body checkpoint records. A manifest without `arch`, or
#: with anything else, is the plain ResNet18 every checkpoint before v3 was.
CONCAT_ARCH = "concat_resnet18_mlp"

def _concat_module(classes: int, hidden: int = 256) -> Any:
    """Two ResNet18 bodies, features concatenated, one hidden layer.

    Restated here rather than imported from approach 1 because the serving path must not
    depend on the training repository being on disk - the demo ships without it. The
    shape is asserted against the checkpoint by `strict=True` below, which is what stops
    the two definitions drifting: a mismatch fails loudly at load, not quietly at score.
    """
    import torch
    import torchvision

    def body():
        net = torchvision.models.resnet18(weights=None)
        net.fc = torch.nn.Identity()
        return net

    class ConcatResNet18MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body_a = body()
            self.body_b = body()
            self.head = torch.nn.Sequential(
                torch.nn.Linear(2 * FEATURE_DIM, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(hidden, classes),
            )

        def forward(self, x):
            a = torch.flatten(self.body_a(x), 1)
            b = torch.flatten(self.body_b(x), 1)
            return self.head(torch.cat([a, b], dim=1))

    return ConcatResNet18MLP()


def _build_and_load(checkpoint: Path, *, classes: int, arch: str = "resnet18") -> Any:
    """The architecture this manifest names, holding these weights.

    `weights=None` and then an **exact** state-dict fit: no missing tensors and no
    unexpected ones. `strict=True` is the whole point - a partial load initialises
    part of the network at random, runs without complaining, and scores a few points
    worse for reasons nobody can find.
    """
    import torch
    import torchvision

    if arch == CONCAT_ARCH:
        net = _concat_module(classes)
        what = "a two-body concat ResNet18"
    else:
        net = torchvision.models.resnet18(weights=None)
        net.fc = torch.nn.Linear(FEATURE_DIM, classes)
        what = "a torchvision ResNet18"

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ModelError(
            f"{checkpoint.name} does not fit {what} with a {classes}-class "
            f"head ({exc}). Refusing rather than loading partially."
        ) from exc

    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad = False
    return net


def clear_cache() -> None:
    """Drop the loaded-model cache. For the tests, which publish and re-publish."""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = [
    "Candidate",
    "ModelError",
    "Pinned",
    "Runtime",
    "clear_cache",
    "discover",
    "find",
    "load_pinned",
    "models_dir",
    "runtime",
    "sha256_of",
]
