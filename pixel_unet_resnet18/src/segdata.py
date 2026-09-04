"""The torch `Dataset`: a stored H-channel tile and its mask, augmented together.

Two things here are the whole reason this module exists rather than being a few lines in
the trainer.

**The mask moves with the image, exactly.** `augment.sample_geometry` draws the flip and
rotation once and `Geometry.apply` puts the same one onto both arrays. Drawing twice - the
obvious thing, and what calling approach 1's `geometric(array, rng)` twice would do - gives
the mask a different transform from its image. Nothing raises: both are still 224x224, the
loss still falls, and the model learns to predict boundaries in the wrong place.

**Only the image gets stain jitter.** `alpha`, `beta` and `gamma` change how much dye the
tile appears to carry. The mask is a set of labels, and there is nothing to jitter about a
label.

`augment=False` is not merely "no jitter" - it is the exact transform inference uses, which
is what makes a validation number mean anything. Do not add a resize or a crop to that
path: `standardise` divides by the tile's own p99, which survives flips and rotations but
not cropping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import augment
import classes
import hstain


class SegTileDataset(Dataset):
    """`(3xHxW float32 image, HxW int64 mask)` pairs from the exporter's manifest.

    The mask is returned as `int64` with `IGNORE` left in place, because that is what
    `CrossEntropyLoss(ignore_index=...)` wants. Do **not** remap `IGNORE` to a class here:
    it is the absence of a statement, and 3.2 % of BCSS plus everything outside a BRACS
    annotation carries it.
    """

    def __init__(
        self,
        rows: list[dict],
        tiles_dir: Path,
        *,
        augment_data: bool = False,
        jitter: augment.Jitter | None = None,
        invert: bool = False,
        standardise: bool = True,
        source_authority: bool = True,
        seed: int = 0,
    ) -> None:
        self.rows = rows
        self.tiles_dir = Path(tiles_dir)
        # Applied at load rather than at export, so turning it off is a flag rather than
        # a re-export - the same reason approach 1 filters its manifest at feature-cache
        # time. The tiles on disk stay the full record of what was annotated; this is a
        # statement about whose annotation this project accepts.
        self.source_authority = source_authority
        self.augment_data = augment_data
        self.jitter = (jitter or augment.Jitter()) if augment_data else augment.NO_JITTER
        self.invert = invert
        # `standardise` is a property of the INPUT, not of the augmentation, so it applies
        # on both paths. Setting it for training and not for inference would be exactly the
        # training/serving drift this class exists to prevent.
        self.standardise = standardise
        self.seed = seed
        #: Bumped by the trainer between epochs so the same tile is not flipped the same
        #: way thirty times. See `augment.seed_for`.
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        stored = np.asarray(Image.open(self.tiles_dir / row["image_path"]))
        mask = np.asarray(Image.open(self.tiles_dir / row["mask_path"]))

        # Before the geometry, though it commutes with it: a per-pixel relabel does not
        # care about flips. Done here so every later line sees the mask the loss will
        # actually be given.
        if self.source_authority:
            mask = classes.apply_source_authority(mask, row.get("source", ""))

        if self.augment_data:
            rng = augment.seed_for(self.seed, index, self.epoch)
            geometry = augment.sample_geometry(rng)
            stored = geometry.apply(stored)
            mask = geometry.apply(mask)
            alpha, beta, gamma = self.jitter.sample(rng)
        else:
            alpha, beta, gamma = 1.0, 0.0, 1.0

        tensor = hstain.from_stored(
            stored,
            alpha=alpha, beta=beta, gamma=gamma,
            invert=self.invert, standardise=self.standardise,
        )
        return torch.from_numpy(tensor), torch.from_numpy(mask.astype(np.int64))


def loader(
    dataset: SegTileDataset,
    *,
    batch_size: int = 8,
    shuffle: bool,
    workers: int = 0,
) -> torch.utils.data.DataLoader:
    """A `DataLoader`, with `workers=0` the honest default on this machine.

    Every forward pass here is CPU-bound and already using every core
    (`paths.torch_threads`), so worker processes would compete with the maths rather than
    hide I/O behind it. Reading two 224 px PNGs is well under a millisecond; the forward
    and backward pass is hundreds. Raise `workers` only on a box where the GPU is waiting.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
        pin_memory=False,
    )


def pixel_census(rows: list[dict], *, source_authority: bool = True) -> dict[str, int]:
    """Labelled pixels per class over a set of rows, read off the manifest.

    From the counts the exporter recorded rather than by re-reading every mask, so a
    sanity check costs a sum instead of a pass over two gigabytes.

    `source_authority` counts only the pixels that will actually supervise. It defaults
    to True because a census that disagreed with the loss would be a census of a training
    run that is not happening - and the gap is not small here: BRACS contributes 58.0 M
    non-epithelium and 16.0 M invasive pixels that the rule blanks.
    """
    out = {name: 0 for name in classes.CLASS_NAMES}
    out["unlabelled"] = 0
    for row in rows:
        counts = (
            classes.authoritative_pixels(row)
            if source_authority
            else {name: int(row[f"px_{name}"]) for name in classes.CLASS_NAMES}
        )
        blanked = 0
        for name in classes.CLASS_NAMES:
            out[name] += counts[name]
            blanked += int(row[f"px_{name}"]) - counts[name]
        # Pixels the rule turned into "no statement" are unlabelled now, and saying so
        # keeps the census summing to the tile area.
        out["unlabelled"] += int(row["px_unlabelled"]) + blanked
    return out
