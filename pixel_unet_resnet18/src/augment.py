"""Augmentation for a segmentation model, where the mask has to move with the image.

Two things here differ from the tile model's augmentation, and both were found by reading
its code rather than guessed at.

**1. The geometric transform is sampled once and applied twice.**
`bcss_bracs_hchannel_resnet18/src/datasets.geometric` takes an array and a generator and draws
three values from it - two floats for the flips, then an integer for the turns, and the
integer is *always* drawn even when the result is zero, so the stream advances identically
either way. That is fine for a classifier, which transforms one array. Calling it twice
with the same generator would consume three fresh draws and give the mask a **different**
transform from its image - a silent misalignment that would train the model to predict
boundaries in the wrong place, and which no shape check would catch because both arrays
are still 224x224.

So `sample_geometry` draws the three values once into a plain dataclass, and `apply` puts
the same three onto whatever it is given.

**2. The draw depends on the epoch.**
`TileDataset` seeds `np.random.default_rng((seed, index))`, which is deliberately
epoch-independent: it fits a linear head in 400 full-batch steps over cached features, so
"the augmentation of tile 7" only ever needs to mean one thing. Here the model sees every
tile ~30 times, and a fixed transform per index would mean tile 7 is flipped the same way
in every epoch - a third of the augmentation's value thrown away. `seed_for` folds the
epoch in.

Flips and 90-degree rotations only. A slide has no canonical orientation - the section
landed on the glass however it landed - so the eight symmetries of the square are all
genuinely the same tissue. Other angles need interpolation, which invents pixel values in
the image and invents *classes* in the mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """One draw from the eight symmetries of the square, reusable across arrays."""

    flip_lr: bool
    flip_ud: bool
    turns: int

    def apply(self, array: np.ndarray) -> np.ndarray:
        """The same transform, on an image or on a mask.

        No interpolation anywhere: `fliplr`, `flipud` and `rot90` all move pixels without
        creating new values, which is what makes this safe to apply to a label array.
        """
        if self.flip_lr:
            array = np.fliplr(array)
        if self.flip_ud:
            array = np.flipud(array)
        if self.turns:
            array = np.rot90(array, self.turns)
        return np.ascontiguousarray(array)


def sample_geometry(rng: np.random.Generator) -> Geometry:
    """Draw the three values in the same order approach 1's `geometric` draws them.

    The order and the unconditional integer draw are preserved so that, given the same
    generator state, this makes the same choice approach 1 would - which is what
    `tests/test_parity.py` checks. It matters only for comparability, but comparability is
    the whole reason the two models can be put side by side.
    """
    flip_lr = bool(rng.random() < 0.5)
    flip_ud = bool(rng.random() < 0.5)
    turns = int(rng.integers(0, 4))
    return Geometry(flip_lr=flip_lr, flip_ud=flip_ud, turns=turns)


@dataclass(frozen=True)
class Jitter:
    """Stain-strength and shape jitter, as ranges rather than values.

    The same ranges the tile model uses, restated here because this folder imports no code
    from approach 1. Kept identical on purpose: these two models are meant to be
    comparable, and a different augmentation would make any difference between them
    partly a difference of augmentation.

    `alpha` and `beta` are strength - what varying stain concentration actually does to
    the haematoxylin channel. `gamma` is **shape**, and it is not a nicety: approach 1's
    gate G0b measured the H&E-to-IHC density ratio running from 0.06 at p75 to 2.06 at
    p99, a 37x spread where a pure strength difference would be one constant. No `alpha`
    fixes that, because a monotone map cannot change the ratio between two quantiles of
    the same image. `gamma` can, and fitting the measured quantiles wants 2.0 to 3.6.

    Log-uniform on `gamma` because it is a ratio: a uniform draw over (0.5, 3.5) would
    spend most of its mass above 1.
    """

    alpha: tuple[float, float] = (0.80, 1.25)
    beta: tuple[float, float] = (-0.05, 0.05)
    gamma: tuple[float, float] = (0.5, 3.5)

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float]:
        lo, hi = self.gamma
        g = float(lo) if lo == hi else float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return (float(rng.uniform(*self.alpha)), float(rng.uniform(*self.beta)), g)


#: The identity. Used on the validation and inference paths, where the transform must be
#: exactly what serving does.
NO_JITTER = Jitter(alpha=(1.0, 1.0), beta=(0.0, 0.0), gamma=(1.0, 1.0))


def seed_for(seed: int, index: int, epoch: int) -> np.random.Generator:
    """A generator for one tile in one epoch.

    Per-item rather than global so a `DataLoader` worker cannot fork a shared stream and
    hand every worker the same draws - and epoch-dependent so the same tile is not flipped
    the same way thirty times.
    """
    return np.random.default_rng((int(seed), int(index), int(epoch)))
