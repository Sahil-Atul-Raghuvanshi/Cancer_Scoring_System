"""A ResNet18 encoder with a U-Net decoder, and the pinning that makes it servable.

The encoder is `torchvision.models.resnet18` with ImageNet weights, exactly the body the
tile model uses - same architecture, same starting point, same input transform. What is
new is the decoder: instead of pooling to a 512-vector and asking one question about the
whole tile, it walks back up through the encoder's own intermediate resolutions and answers
per pixel.

    input 3 x 224 x 224
      conv1+bn+relu  ->  64 x 112  ────────────────────────┐
      maxpool+layer1 ->  64 x  56  ───────────────┐        │
      layer2         -> 128 x  28  ──────┐        │        │
      layer3         -> 256 x  14  ─┐    │        │        │
      layer4         -> 512 x   7   │    │        │        │
      up 512->256 + skip ───────────┘    │        │        │
      up 256->128 + skip ────────────────┘        │        │
      up 128-> 64 + skip ─────────────────────────┘        │
      up  64-> 32 + skip ──────────────────────────────────┘
      up  32-> 16
      1x1 conv -> 3 x 224 x 224

**Why skips at all.** `layer4`'s 7x7 map knows *what* is in the tile but has thrown away
*where* to within 32 pixels. The boundary information is in the early layers, at 112 and 56
px, and the skip connections are the only route by which it reaches the output. A decoder
without them produces exactly the blocky output this model exists to replace, just at 32 px
instead of 224.

**Why the encoder is partly frozen.** `conv1`, `layer1` and `layer2` are edge and texture
detectors that ImageNet already fits well and that a 20,000-tile dataset will not improve;
freezing them roughly halves the backward pass, which on CPU is the difference between a
10-hour run and a 20-hour one. `layer3`, `layer4` and the decoder train, because the
semantic distinction between in-situ and invasive epithelium is not in ImageNet.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from torch import nn

import classes
import hstain
import paths

NUM_CLASSES = len(classes.CLASS_NAMES)

#: The initialisations, and the file each expects in `pretrained/`.
INITS: dict[str, str] = {
    "imagenet": "resnet18-imagenet.pth",
    "simclr": "pytorchnative_tenpercent_resnet18.ckpt",
}


def _load_state(init: str, weights_dir: Path) -> dict[str, torch.Tensor]:
    """The backbone weights, from a plain `state_dict` file.

    `weights_only=True` throughout: these are downloaded assets, and a checkpoint that
    must be *executed* to be read is a checkpoint that can run anything. The SimCLR
    release ships a native `state_dict` for exactly this reason.
    """
    path = Path(weights_dir) / INITS[init]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. `imagenet` is vendored by the demo's own setup; "
            "`simclr` is Ciga's released native checkpoint."
        )
    state = torch.load(path, map_location="cpu", weights_only=True)

    if init == "simclr":
        # Ciga's keys are prefixed from the SimCLR wrapper it was trained inside.
        state = {
            key.replace("model.", "", 1).replace("resnet.", "", 1): value
            for key, value in state.items()
        }
    return {k: v for k, v in state.items() if not k.startswith("fc.")}


class DecoderBlock(nn.Module):
    """Upsample, concatenate the skip, convolve twice.

    Bilinear upsampling plus a 3x3 rather than a transposed convolution: a transposed
    convolution with an even kernel and a stride of two produces checkerboard artefacts,
    and on a boundary task the artefact lands exactly where the answer is being read.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Interpolate rather than assume: a non-multiple-of-32 input (which the
            # fully-convolutional inference path allows) leaves the two off by a pixel.
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResNet18UNet(nn.Module):
    """ResNet18 encoder, U-Net decoder, three-class per-pixel output."""

    def __init__(self, init: str = "imagenet", *, weights_dir: Path | None = None,
                 num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        if init not in INITS:
            raise ValueError(f"unknown initialisation {init!r}; expected {sorted(INITS)}")

        encoder = torchvision.models.resnet18(weights=None)
        state = _load_state(init, Path(weights_dir or paths.PRETRAINED))
        missing, unexpected = encoder.load_state_dict(state, strict=False)

        # **`strict=False` with an assertion, never `strict=False` alone.** A silent
        # partial load initialises half the body randomly, trains without complaining, and
        # scores a couple of points worse - and the conclusion drawn from that is wrong and
        # permanent. Only the head may be missing; it belongs to our task.
        unexpected_body = [k for k in unexpected if not k.startswith("fc.")]
        if set(missing) - {"fc.weight", "fc.bias"} or unexpected_body:
            raise ValueError(
                f"the {init!r} checkpoint does not fit a torchvision ResNet18.\n"
                f"  missing:    {sorted(missing)}\n"
                f"  unexpected: {sorted(unexpected_body)}"
            )

        self.init = init
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.up4 = DecoderBlock(512, 256, 256)
        self.up3 = DecoderBlock(256, 128, 128)
        self.up2 = DecoderBlock(128, 64, 64)
        self.up1 = DecoderBlock(64, 64, 32)
        self.up0 = DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)          # 64 x 112
        s1 = self.layer1(self.pool(s0))   # 64 x 56
        s2 = self.layer2(s1)       # 128 x 28
        s3 = self.layer3(s2)       # 256 x 14
        bottom = self.layer4(s3)   # 512 x 7

        d = self.up4(bottom, s3)   # 256 x 14
        d = self.up3(d, s2)        # 128 x 28
        d = self.up2(d, s1)        # 64 x 56
        d = self.up1(d, s0)        # 32 x 112
        d = self.up0(d, None)      # 16 x 224
        logits = self.head(d)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits


def freeze_early_encoder(net: ResNet18UNet) -> ResNet18UNet:
    """Freeze `stem`, `layer1`, `layer2`. Everything else trains.

    Halves the backward pass for the cost of not adapting filters ImageNet already fits.
    """
    for module in (net.stem, net.layer1, net.layer2):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return net


def parameter_groups(net: ResNet18UNet, *, encoder_lr: float, decoder_lr: float):
    """Two learning rates: the pretrained encoder moves slowly, the fresh decoder fast.

    One rate for both would either wreck `layer4`'s pretrained features or leave the
    randomly-initialised decoder barely trained after thirty epochs.
    """
    encoder = [p for m in (net.layer3, net.layer4) for p in m.parameters() if p.requires_grad]
    decoder = [
        p for m in (net.up4, net.up3, net.up2, net.up1, net.up0, net.head)
        for p in m.parameters() if p.requires_grad
    ]
    return [
        {"params": encoder, "lr": encoder_lr},
        {"params": decoder, "lr": decoder_lr},
    ]


# --- publishing ---------------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(
    net: ResNet18UNet,
    *,
    name: str,
    models_dir: Path,
    tile_px: int,
    mpp: float,
    standardise: bool,
    invert: bool = False,
    gamma: float = 1.0,
    metrics: dict,
    training_data: dict,
    provenance: dict,
) -> tuple[Path, Path]:
    """Write the checkpoint and a manifest that records what it needs at inference.

    `arch` is `resnet18_unet`, not `resnet18`. The tile model's loader reconstructs a bare
    ResNet18 with a `Linear` head and loads strictly, so it would refuse this file outright
    - which is the correct behaviour and the reason the two must be distinguishable by
    their manifest rather than by their filename.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = models_dir / f"{name}.pt"
    torch.save(net.state_dict(), checkpoint)

    manifest = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256_of(checkpoint),
        "arch": "resnet18_unet",
        "output": "per_pixel_logits",
        "init": net.init,
        "classes": list(classes.CLASS_NAMES),
        "ignore_index": classes.IGNORE,
        "input": hstain.descriptor(
            tile_px=tile_px, mpp=mpp, invert=invert, gamma=gamma, standardise=standardise
        ),
        "held_out_institutions": sorted(classes.TEST_INSTITUTIONS),
        "training_data": training_data,
        "metrics": metrics,
        "provenance": provenance,
    }
    manifest_path = models_dir / f"{name}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return checkpoint, manifest_path


def load_pinned(name: str, *, models_dir: Path, verify: bool = True):
    """Load a published segmentation checkpoint, refusing one whose SHA-256 disagrees.

    Refusing rather than warning: a model that is not the one that was validated has no
    claim on any number in the report.
    """
    models_dir = Path(models_dir)
    checkpoint = models_dir / f"{name}.pt"
    manifest_path = models_dir / f"{name}.manifest.json"
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"expected {checkpoint.name} and {manifest_path.name} in {models_dir}. A "
            "checkpoint without its manifest is a model nobody can describe."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("arch") != "resnet18_unet":
        raise ValueError(
            f"{name} records arch {manifest.get('arch')!r}, not 'resnet18_unet'. This "
            "loader builds a segmentation network; a tile classifier needs its own."
        )
    if verify and (actual := sha256_of(checkpoint)) != manifest.get("sha256"):
        raise ValueError(
            f"{checkpoint.name} hashes to {actual[:16]}... but its manifest records "
            f"{str(manifest.get('sha256'))[:16]}.... Refusing to score with a model that "
            "is not the one that was validated."
        )

    net = ResNet18UNet(manifest["init"], num_classes=len(manifest["classes"]))
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return net.eval(), manifest
