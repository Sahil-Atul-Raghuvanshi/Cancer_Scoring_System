"""nnU-Net's `PlainConvUNet`, rebuilt from the shapes in BEETLE's own checkpoint.

**Why a reimplementation rather than `pip install nnunetv2`.** The released archive is
five `checkpoint_best.pth` files plus `plans.json`; nnU-Net itself is only needed to
turn those plans into a `torch.nn.Module`. Installing it drags in
`dynamic_network_architectures`, `batchgenerators`, `acvl_utils` and SimpleITK for
exactly one function call, on a CPU box where several of those have no wheel. The
architecture that `plans.json` describes is 200 lines of plain torch, and rebuilding it
here keeps the backend installable with nothing but the packages it already has.

**Why it lives in the backend.** `tissue_label_generation/backend/bracs_app/unet.py` is
the same reimplementation, written first, for the research stage that harvests labels
with this network. This is not that file imported: the dependency between the two trees
runs **research -> backend and never the other way** - the rule `input.py`'s docstring
states and `bracs_app/config.py` is arranged around - so the backend cannot import from
a sibling research folder, and the module that *serves* a network belongs where the
serving happens. If the two are ever unified it is the research app that should import
this one.

**What makes a rebuild safe.** Guessing an architecture and loading weights into it is
normally an excellent way to produce a model that runs, converges on nothing, and
returns confident nonsense. Two things stop that here:

  * `load_beetle_weights` compares key *sets* before loading and refuses any mismatch
    at all. A misplaced layer, a transposed convolution built the wrong way round, a
    missing skip - all of them change the key set or a tensor shape, and all of them
    raise.
  * every structural number below is read off `plans.json` at construction time rather
    than hard-coded, so the same class would rebuild a differently-planned nnU-Net.

The one thing a strict load cannot catch is a *wiring* error between correctly-shaped
layers - a skip connection attached to the wrong stage, say. `beetle.self_check` covers
that end from the other side, by asserting the network's output on real tissue is
structured rather than uniform.

Layout notes, each of which is a shape in the checkpoint rather than a preference:

  * `conv -> norm -> nonlin`, with `BatchNorm2d`. nnU-Net's own 2d default is
    `InstanceNorm2d`, but the checkpoint carries `running_mean`, `running_var` and a
    `num_batches_tracked` per block, which only a batch norm has.
  * `LeakyReLU(0.01)`, nnU-Net's default nonlinearity.
  * the decoder builds a `seg_layer` per stage because deep supervision was on during
    training and the checkpoint therefore carries all seven. Only the last - the one at
    full resolution - is evaluated at inference, and the other six are built anyway so
    the key sets match.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ConvNormNonlin(nn.Module):
    """One `conv -> norm -> nonlin`, named to match `dynamic_network_architectures`.

    The submodule names `conv` and `norm` are load-bearing: they are half of every
    parameter key in the checkpoint. `all_modules` in the released state dict is the
    same two tensors under a second name - the upstream class registers a `Sequential`
    alongside the individual modules - and `load_beetle_weights` drops that alias
    rather than reproducing it.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=True
        )
        self.norm = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1, affine=True)
        self.nonlin = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlin(self.norm(self.conv(x)))


class StackedConvBlocks(nn.Module):
    """`n` convolutions, only the first of which may carry a stride."""

    def __init__(
        self, n: int, in_ch: int, out_ch: int, kernel: int, initial_stride: int
    ) -> None:
        super().__init__()
        blocks = [ConvNormNonlin(in_ch, out_ch, kernel, initial_stride)]
        blocks += [ConvNormNonlin(out_ch, out_ch, kernel, 1) for _ in range(n - 1)]
        self.convs = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class PlainConvEncoder(nn.Module):
    """The contracting path. Returns every stage's output, for the decoder's skips."""

    def __init__(
        self,
        in_channels: int,
        features_per_stage: Sequence[int],
        kernels: Sequence[int],
        strides: Sequence[int],
        n_conv_per_stage: Sequence[int],
    ) -> None:
        super().__init__()
        stages = []
        current = in_channels
        for stage, out_ch in enumerate(features_per_stage):
            # Each stage is wrapped in a `Sequential` of one, which is what puts the
            # `.0.` in `encoder.stages.3.0.convs...`. Cosmetic upstream; here it is
            # the difference between the keys matching and not.
            stages.append(
                nn.Sequential(
                    StackedConvBlocks(
                        n_conv_per_stage[stage],
                        current,
                        out_ch,
                        kernels[stage],
                        strides[stage],
                    )
                )
            )
            current = out_ch
        self.stages = nn.ModuleList(stages)
        self.output_channels = list(features_per_stage)
        self.strides = list(strides)
        self.kernels = list(kernels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class UNetDecoder(nn.Module):
    """The expanding path: transposed convolution, concatenate the skip, convolve.

    `self.encoder = encoder` reproduces the upstream class's back-reference. It
    duplicates every encoder key under a `decoder.encoder.` prefix in the state dict -
    the same tensors, not copies - and `load_beetle_weights` drops those too. Keeping
    the attribute costs nothing and means the checkpoint's key set and this module's
    agree without a rename table.
    """

    def __init__(
        self,
        encoder: PlainConvEncoder,
        num_classes: int,
        n_conv_per_stage: Sequence[int],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages = len(encoder.output_channels)

        transpconvs, stages, seg_layers = [], [], []
        for s in range(1, n_stages):
            below = encoder.output_channels[-s]
            skip = encoder.output_channels[-(s + 1)]
            stride = encoder.strides[-s]
            transpconvs.append(
                nn.ConvTranspose2d(below, skip, stride, stride=stride, bias=True)
            )
            # `2 * skip` in: the upsampled tensor concatenated with the skip.
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    2 * skip,
                    skip,
                    encoder.kernels[-(s + 1)],
                    1,
                )
            )
            seg_layers.append(nn.Conv2d(skip, num_classes, 1, 1, 0, bias=True))

        self.transpconvs = nn.ModuleList(transpconvs)
        self.stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        """Full-resolution logits only. Deep supervision is a training-time output."""
        x = skips[-1]
        for s in range(len(self.stages)):
            x = self.transpconvs[s](x)
            x = torch.cat((x, skips[-(s + 2)]), dim=1)
            x = self.stages[s](x)
        return self.seg_layers[-1](x)


class PlainConvUNet(nn.Module):
    """The network `plans.json` describes, with full-resolution logits its only output."""

    def __init__(
        self,
        in_channels: int,
        n_stages: int,
        features_per_stage: Sequence[int],
        kernels: Sequence[int],
        strides: Sequence[int],
        n_conv_per_stage: Sequence[int],
        n_conv_per_stage_decoder: Sequence[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        if not (
            n_stages
            == len(features_per_stage)
            == len(kernels)
            == len(strides)
            == len(n_conv_per_stage)
        ):
            raise ValueError(
                "the per-stage lists disagree with n_stages; this is a malformed "
                "plans.json rather than a model that will merely train badly"
            )
        self.encoder = PlainConvEncoder(
            in_channels, features_per_stage, kernels, strides, n_conv_per_stage
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def from_plans(plans: dict, num_classes: int, in_channels: int = 3) -> PlainConvUNet:
    """Build the 2d configuration described by a `plans.json`, refusing anything else.

    `UNet_class_name` is checked rather than ignored. BEETLE ships `PlainConvUNet`; a
    residual-encoder variant has different keys and would fail the strict load anyway,
    but failing here says why.
    """
    cfg = plans["configurations"]["2d"]
    if cfg["UNet_class_name"] != "PlainConvUNet":
        raise ValueError(
            f"this module builds PlainConvUNet, and the plans ask for "
            f"{cfg['UNet_class_name']!r}. Rebuilding a different architecture from "
            "these shapes would load weights into the wrong graph."
        )

    strides = [tuple(s) for s in cfg["pool_op_kernel_sizes"]]
    kernels = [tuple(k) for k in cfg["conv_kernel_sizes"]]
    if any(len(set(s)) != 1 for s in strides) or any(len(set(k)) != 1 for k in kernels):
        raise ValueError(
            "anisotropic strides or kernels are in the plans; this module assumes "
            "square ones, as every 2d pathology configuration uses"
        )

    n_stages = len(kernels)
    base, cap = cfg["UNet_base_num_features"], cfg["unet_max_num_features"]
    features = [min(base * 2**stage, cap) for stage in range(n_stages)]

    return PlainConvUNet(
        in_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features,
        kernels=[k[0] for k in kernels],
        strides=[s[0] for s in strides],
        n_conv_per_stage=cfg["n_conv_per_stage_encoder"],
        n_conv_per_stage_decoder=cfg["n_conv_per_stage_decoder"],
        num_classes=num_classes,
    )


#: The patch side the released 2d configuration was trained at, read off `plans.json`
#: rather than assumed. Named here because the sliding window in `beetle.py` is built
#: from it and a hard-coded 512 there would be a second source of truth.
def patch_size(plans: dict) -> int:
    """The square patch side `plans.json` records for the 2d configuration."""
    size = plans["configurations"]["2d"]["patch_size"]
    if len(size) != 2 or size[0] != size[1]:
        raise ValueError(
            f"this module assumes a square patch and the plans record {size}. A "
            "non-square window would need its own sliding grid per axis."
        )
    return int(size[0])


#: Keys in the released state dict that are a second name for a tensor already present.
#: `all_modules` is upstream's `Sequential` view of a conv block; `decoder.encoder` is
#: the decoder's back-reference to the encoder it was handed.
_ALIAS_FRAGMENTS = (".all_modules.",)
_ALIAS_ROOTS = ("decoder.encoder.",)


def _is_alias(key: str) -> bool:
    return any(part in key for part in _ALIAS_FRAGMENTS) or key.startswith(_ALIAS_ROOTS)


def load_beetle_weights(model: PlainConvUNet, state_dict: dict) -> None:
    """Load a released checkpoint, refusing any mismatch at all.

    The comparison is on key *sets*, before `load_state_dict` is called, so the error
    names the layers that disagree instead of the first shape that happens to differ.
    A model built from the wrong plans loads about 80 % of its weights cleanly and then
    segments noise, and that failure looks exactly like a bad dataset for a week.

    Aliases are dropped from *both* sides. This module reproduces upstream's shared
    references, so it emits the same duplicate keys the checkpoint does; loading the
    canonical name fills the alias too, because they are one tensor. `strict=False` is
    therefore not a relaxation - what it skips is re-checked below, and anything left
    over that is not an alias is still an error.
    """
    incoming = {k: v for k, v in state_dict.items() if not _is_alias(k)}
    expected = {k for k in model.state_dict() if not _is_alias(k)}
    got = set(incoming)

    if missing := sorted(expected - got):
        raise RuntimeError(
            f"{len(missing)} parameters this network needs are absent from the "
            f"checkpoint, e.g. {missing[:4]}. The architecture does not match the "
            "weights."
        )
    if unexpected := sorted(got - expected):
        raise RuntimeError(
            f"{len(unexpected)} checkpoint parameters have nowhere to go, e.g. "
            f"{unexpected[:4]}. The architecture does not match the weights."
        )

    result = model.load_state_dict(incoming, strict=False)
    if stragglers := [k for k in result.missing_keys if not _is_alias(k)]:
        raise RuntimeError(f"unfilled parameters after load: {stragglers[:4]}")
    if result.unexpected_keys:
        raise RuntimeError(f"rejected parameters: {result.unexpected_keys[:4]}")


__all__ = [
    "PlainConvUNet",
    "from_plans",
    "load_beetle_weights",
    "patch_size",
]
