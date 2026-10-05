"""The backbone, its two initialisations, and how a checkpoint is pinned.

Both approaches live here, and deliberately in one file. Approach 3 *is* approach 1
with different starting weights - "one box swaps" - so if it had its own model factory
the two would be free to drift in some second respect and the A/B would stop being a
comparison of initialisations. One factory, one `init` argument, one code path.

  imagenet   torchvision's ResNet18, BSD-3. Knows edges, fur and wheels.
  simclr     Ciga, Xu & Martel's ResNet18, MIT, self-supervised on 57 histopathology
             datasets. Knows nuclei and glands. Referred to as "MoCo" in
             segmentation-approaches-comparison.md; the released checkpoint is
             **SimCLR**, and the architecture is ResNet18 rather than the commonly
             cited ResNet50.

**The head is fitted on a frozen body.** Not a compromise for the CPU - though it is
that too - but what makes the whole project iterable: with the body frozen, its
features can be computed once and cached, after which fitting the three-class head
takes seconds. Class definitions, loss weights, input polarity and the entire approach
3 comparison become experiments you run in a coffee break rather than decisions you
have to defend in advance.

Ciga O, Xu T, Martel AL. Self supervised contrastive learning for digital
histopathology. Machine Learning with Applications 7 (2022). arXiv:2011.13971.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision

import bcss
import hchannel

#: The two initialisations, and the file each expects to find in `pretrained/`.
INITS: dict[str, str] = {
    "imagenet": "resnet18-imagenet.pth",
    "simclr": "pytorchnative_tenpercent_resnet18.ckpt",
}

#: Released asset sizes in bytes, as reported by the GitHub API. Checked on load, so
#: a truncated or rate-limited download is caught at the point of use rather than
#: showing up as a mysteriously worse model.
EXPECTED_BYTES: dict[str, int] = {
    "pytorchnative_tenpercent_resnet18.ckpt": 46_104_139,
}

FEATURE_DIM = 512
NUM_CLASSES = len(bcss.CLASS_NAMES)


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file, in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vendor_imagenet_weights(pretrained_dir: Path) -> Path:
    """Download torchvision's ImageNet ResNet18 once and copy it into `pretrained/`.

    Vendored rather than left in the torch hub cache, because a hub cache is a
    machine's property and a pinned checkpoint has to be the project's. Nothing in
    this pipeline is fetched from a model hub at run time - not at training, and
    certainly not at inference, where a silently updated checkpoint would change a
    clinical number without a commit.
    """
    pretrained_dir = Path(pretrained_dir)
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    destination = pretrained_dir / INITS["imagenet"]

    if not destination.exists():
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        state = weights.get_state_dict(progress=True)
        torch.save(state, destination)

    return destination


def _load_state(init: str, weights_dir: Path) -> dict[str, torch.Tensor]:
    """The state dict for one initialisation, with the upstream key names repaired."""
    path = Path(weights_dir) / INITS[init]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Notebook 01 (approach 1) vendors the ImageNet "
            f"weights; notebook 01 of approach 3 downloads the SimCLR checkpoint."
        )

    expected = EXPECTED_BYTES.get(path.name)
    actual = path.stat().st_size
    if expected is not None and actual != expected:
        raise ValueError(
            f"{path.name} is {actual:,} bytes, expected {expected:,}. A truncated or "
            "rate-limited download would otherwise load partially and quietly cost a "
            "few points of accuracy - re-download it."
        )

    # weights_only=True for both, which is the reason approach 3 takes the *native*
    # asset rather than the PyTorch Lightning one: a Lightning checkpoint is a pickled
    # object graph that has to be executed to be read, and this codebase already went
    # to real trouble over that once (step02_quality_control/models.py, for GrandQC).
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if init == "simclr":
        # Upstream keys carry `model.` and `resnet.` prefixes from the training rig.
        state = {
            key.replace("model.", "").replace("resnet.", ""): value
            for key, value in state.items()
        }

    return state


def resnet18_backbone(init: str, *, weights_dir: Path, num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    """A ResNet18 with a fresh `num_classes` head, from one of the two initialisations.

    **`strict=False` with an assertion, never `strict=False` alone.** A silent partial
    load is the failure mode that would sink the approach-3 comparison: the network
    initialises half-randomly, trains without complaining, scores a couple of points
    *worse* than ImageNet, and the conclusion written into the report - "histopathology
    pretraining does not help here" - is false and gets believed. The two assertions
    below are the difference between an A/B and a rumour.

    The only tensors allowed to be missing are `fc.weight` and `fc.bias`: the head
    belongs to our task and not to anyone's pretraining.
    """
    if init not in INITS:
        raise ValueError(f"unknown initialisation {init!r}; expected one of {sorted(INITS)}")

    net = torchvision.models.resnet18(weights=None)
    state = _load_state(init, Path(weights_dir))

    missing, unexpected = net.load_state_dict(state, strict=False)

    unexpected_body = [key for key in unexpected if not key.startswith("fc.")]
    if set(missing) - {"fc.weight", "fc.bias"} or unexpected_body:
        raise ValueError(
            f"the {init!r} checkpoint does not fit a torchvision ResNet18.\n"
            f"  missing:    {sorted(missing)}\n"
            f"  unexpected: {sorted(unexpected_body)}\n"
            "Refusing rather than loading partially: a half-random initialisation "
            "trains quietly and loses, and the wrong conclusion is then permanent."
        )

    net.fc = torch.nn.Linear(FEATURE_DIM, num_classes)
    return net


def feature_extractor(net: torch.nn.Module) -> torch.nn.Module:
    """The frozen body: everything up to and including the pooling, `fc` removed.

    `fc` is replaced by an identity rather than sliced off, so the module still runs
    end to end and the 512-vector it returns is exactly what the head will be handed.
    """
    body = net
    body.fc = torch.nn.Identity()
    for parameter in body.parameters():
        parameter.requires_grad = False
    return body.eval()


def freeze_body(net: torch.nn.Module) -> torch.nn.Module:
    """Everything frozen except the head. The fit in the plan's Notebook 04."""
    for parameter in net.parameters():
        parameter.requires_grad = False
    for parameter in net.fc.parameters():
        parameter.requires_grad = True
    return net


def unfreeze_layer4(net: torch.nn.Module) -> torch.nn.Module:
    """`layer4` and the head trainable. The optional overnight fine-tune, Notebook 05."""
    freeze_body(net)
    for parameter in net.layer4.parameters():
        parameter.requires_grad = True
    return net


# --- pinning ------------------------------------------------------------------


#: The architecture string a concat checkpoint records, and the value `load_pinned`
#: branches on. A published manifest that does not carry it is a plain ResNet18, which
#: is what every checkpoint before this one was.
CONCAT_ARCH = "concat_resnet18_mlp"


class ConcatResNet18MLP(torch.nn.Module):
    """Two frozen ResNet18 bodies side by side, their features concatenated, one MLP.

    **Why two bodies.** The two initialisations disagree about what a tile looks like -
    one is ImageNet-supervised, the other is Ciga's SimCLR on histology - and neither
    body is ever updated here, so their 512-vectors are two genuinely different
    descriptions of the same pixels rather than one description twice. Concatenating
    them measured as the single largest gain of any change tried: +0.020 cross-validated
    weakest-class recall over the ImageNet body alone, more than the MLP head, the class
    rebalancing or the extra training data contributed individually.

    **What it costs, stated plainly because it is the reason not to do this.** Every
    tile goes through two ResNet18 forward passes instead of one, so step 8's wall clock
    roughly doubles. That is the price of the accuracy and it is paid on every slide,
    for ever.

    **The bodies are frozen and are not a training artefact.** They hold their published
    initialisation weights unchanged; only `head` was fitted. They are nonetheless saved
    into the checkpoint rather than re-downloaded at load time, because a checkpoint that
    reconstructs half of itself from the internet is a checkpoint whose behaviour depends
    on what a CDN served that morning - and `load_pinned` exists to make that impossible.
    """

    def __init__(self, *, classes: int = NUM_CLASSES, hidden: int = 256,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.body_a = torch.nn.Identity()
        self.body_b = torch.nn.Identity()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * FEATURE_DIM, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, classes),
        )

    @classmethod
    def build(cls, *, weights_dir: Path, classes: int = NUM_CLASSES,
              hidden: int = 256, dropout: float = 0.2) -> "ConcatResNet18MLP":
        """With both bodies constructed from their published initialisations."""
        model = cls(classes=classes, hidden=hidden, dropout=dropout)
        model.body_a = feature_extractor(
            resnet18_backbone("imagenet", weights_dir=weights_dir))
        model.body_b = feature_extractor(
            resnet18_backbone("simclr", weights_dir=weights_dir))
        return model

    @classmethod
    def empty(cls, *, classes: int = NUM_CLASSES, hidden: int = 256,
              dropout: float = 0.2) -> "ConcatResNet18MLP":
        """Shaped for `load_state_dict`, with no initialisation weights fetched.

        The serving path uses this: the checkpoint carries every tensor, so downloading
        an initialisation only to overwrite it would be slower and would make loading
        depend on the network.
        """
        import torchvision

        model = cls(classes=classes, hidden=hidden, dropout=dropout)
        model.body_a = feature_extractor(torchvision.models.resnet18(weights=None))
        model.body_b = feature_extractor(torchvision.models.resnet18(weights=None))
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.flatten(self.body_a(x), 1)
        b = torch.flatten(self.body_b(x), 1)
        return self.head(torch.cat([a, b], dim=1))


def publish(
    net: torch.nn.Module,
    *,
    name: str,
    models_dir: Path,
    init: str,
    invert_polarity: bool,
    standardise: bool = False,
    tile_px: int,
    mpp: float,
    channel: str = hchannel.CHANNEL_HAEMATOXYLIN,
    metrics: dict[str, object],
    training_data: dict[str, object],
    provenance: dict[str, object],
    probe_tiles: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    """Write the checkpoint and its manifest, and record the checkpoint's own SHA-256.

    A `state_dict`, never a pickled `nn.Module`. Step 2 of the pipeline has to unpickle
    foreign checkpoints and goes to considerable trouble to lift tensors out without
    executing anything; there is no reason to inflict the same problem on whoever
    maintains this in two years.

    `probe_tiles` carries a handful of tile ids and the logits this model produced for
    them. That is gate G6: a fresh process reloads the checkpoint, runs those tiles and
    has to reproduce those numbers. It is the test that catches a normalisation
    mismatch between training and serving, and it is the most valuable test in the
    whole plan - the failure it catches is invisible in every metric computed inside
    the training process.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = models_dir / f"{name}.pt"
    torch.save(net.state_dict(), checkpoint)

    manifest = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_of(checkpoint),
        "arch": (CONCAT_ARCH if isinstance(net, ConcatResNet18MLP)
                 else "resnet18"),
        "init": init,
        "classes": list(bcss.CLASS_NAMES),
        "class_meaning": {bcss.CLASS_NAMES[k]: v for k, v in bcss.CLASS_MEANING.items()},
        # `standardise` belongs here, not only in the training script: step 8 reads
        # this descriptor and must apply the identical transform. A model fitted with
        # it and served without is the drift `hchannel` exists to prevent.
        # `channel` belongs here for the same reason `standardise` does, one step
        # further out: step 7 filters the published heads on it to decide which of
        # two checkpoints at one geometry is the one it means, and step 8 branches on
        # it to decide whether to hand the network density or a photograph.
        "input": hchannel.descriptor(tile_px=tile_px, mpp=mpp, invert=invert_polarity,
                                     standardise=standardise, channel=channel),
        "held_out_institutions": sorted(bcss.TEST_INSTITUTIONS),
        "training_data": training_data,
        "metrics": metrics,
        "provenance": provenance,
        "probe_tiles": probe_tiles or [],
    }
    manifest_path = models_dir / f"{name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return checkpoint, manifest_path


def load_pinned(name: str, *, models_dir: Path, verify: bool = True) -> tuple[torch.nn.Module, dict]:
    """Load a published checkpoint, refusing one whose SHA-256 does not match.

    The refusal is the point. A scoring pipeline that runs whatever bytes happen to be
    at a path cannot say which model produced a number, and this one feeds a test that
    helps decide whether someone receives chemotherapy. Step 8 calls this, and
    `scripts/check_tissue_model.py` calls it as a pre-flight.
    """
    models_dir = Path(models_dir)
    checkpoint = models_dir / f"{name}.pt"
    manifest_path = models_dir / f"{name}.manifest.json"

    if not checkpoint.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"expected {checkpoint.name} and {manifest_path.name} in {models_dir}. "
            "A checkpoint without its manifest is a model nobody can describe, so "
            "both are required."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify:
        actual = sha256_of(checkpoint)
        if actual != manifest.get("sha256"):
            raise ValueError(
                f"{checkpoint.name} hashes to {actual[:16]}... but its manifest records "
                f"{str(manifest.get('sha256'))[:16]}.... Refusing to score with a model "
                "that is not the one that was validated."
            )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if manifest.get("arch") == CONCAT_ARCH:
        net = ConcatResNet18MLP.empty(classes=len(manifest["classes"]))
    else:
        net = torchvision.models.resnet18(weights=None)
        net.fc = torch.nn.Linear(FEATURE_DIM, len(manifest["classes"]))
    # `strict=True`: a partial load initialises part of the network at random, runs
    # without complaining, and scores a few points worse for reasons nobody can find.
    net.load_state_dict(state, strict=True)
    return net.eval(), manifest
