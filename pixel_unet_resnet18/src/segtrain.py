"""Training the segmentation network. Resumable, because ten hours is long enough to lose.

--------------------------------------------------------------------------------
The loss, and the two decisions inside it
--------------------------------------------------------------------------------

`CrossEntropyLoss(weight=..., ignore_index=IGNORE)` plus soft Dice.

**`ignore_index` is load-bearing.** BCSS is 3.2 % unlabelled and every pixel outside a
BRACS annotation carries `IGNORE` too. Folding those into class 0 would teach the model
that unannotated tissue is stroma, which is both wrong and flattering - the largest class
would absorb every uncertainty. Nothing in the tile pipeline needed this, because a tile
label is never `IGNORE`.

**The weights are per pixel, not per tile.** Class 1 is 13.7 % of the tile model's *tiles*
and around 0.15 % of BCSS's *pixels*. Weighting by tile frequency would under-weight the
rare class by two orders of magnitude and the model would answer "stroma" everywhere while
its loss fell nicely.

**Why Dice as well as cross-entropy.** Weighted cross-entropy fixes the class imbalance but
is still a per-pixel average, so it is nearly indifferent to whether a thin structure is
found at all - losing a whole duct costs little if the duct is small. Dice is computed over
the region and punishes a missed structure properly. Together: cross-entropy gives a
well-behaved gradient everywhere, Dice makes small objects matter.

--------------------------------------------------------------------------------
Resumability
--------------------------------------------------------------------------------

A checkpoint is written after **every** epoch, atomically, carrying the model, the
optimiser state, the epoch number, the history, and a fingerprint of the manifest and the
recipe. `resume` reloads it only when that fingerprint matches: a checkpoint from a
different tile set or a different learning rate is not a checkpoint of this run, and
silently continuing from one would produce a model whose history is a fiction.

So `python scripts/02_train_seg.py` can be run repeatedly. It picks up where it stopped,
and once the epochs are done it is a no-op that reports the final numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

import classes
import paths
import segdata
import segnet
import segreport
import splits


@dataclass(frozen=True)
class Recipe:
    """Every number that changes what the trained model is.

    Hashed into the resume fingerprint, so changing any of them starts a fresh run rather
    than continuing a different one under a new name.
    """

    epochs: int = 12
    batch_size: int = 8
    encoder_lr: float = 1e-4
    decoder_lr: float = 1e-3
    weight_decay: float = 1e-4
    dice_weight: float = 0.5
    seed: int = 0
    init: str = "imagenet"
    standardise: bool = True
    invert: bool = False
    #: Whether a source may only teach the classes it is authoritative for - BRACS
    #: in-situ only, BCSS all three. See `classes.SOURCE_CLASSES`.
    #:
    #: In the `Recipe` and therefore in the resume fingerprint, deliberately: it changes
    #: which pixels supervise, so a run started under one setting must not resume under
    #: the other. That is the same reason `standardise` lives here.
    source_authority: bool = True
    #: Tiles drawn per epoch. `None` means all of them.
    #:
    #: A CPU forward-and-backward pass over a U-Net at full resolution costs far more than
    #: the tile model's cached-feature fit, so a full pass over ~20,000 tiles is hours. A
    #: fixed random subsample per epoch trades epochs against coverage: the model still
    #: sees every tile over the course of the run, and each epoch is short enough that the
    #: validation curve is informative while it is still running.
    tiles_per_epoch: int | None = None

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class History:
    epochs_done: int = 0
    train_loss: list[float] = field(default_factory=list)
    val_dice_in_situ: list[float] = field(default_factory=list)
    val_dice_invasive: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, *, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice over the labelled pixels, averaged across classes.

    `IGNORE` pixels are zeroed out of both the prediction and the target rather than
    dropped, so the spatial shape survives and the arithmetic stays vectorised.
    """
    valid = (target != classes.IGNORE)
    if not valid.any():
        return logits.sum() * 0.0

    probability = torch.softmax(logits, dim=1)
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    one_hot = nn.functional.one_hot(safe_target, num_classes=probability.shape[1])
    one_hot = one_hot.permute(0, 3, 1, 2).to(probability.dtype)

    keep = valid.unsqueeze(1).to(probability.dtype)
    probability = probability * keep
    one_hot = one_hot * keep

    dims = (0, 2, 3)
    intersection = (probability * one_hot).sum(dims)
    cardinality = probability.sum(dims) + one_hot.sum(dims)
    dice = (2 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


class Loss(nn.Module):
    """Weighted cross-entropy plus soft Dice. See the module docstring for why both."""

    def __init__(self, weights: torch.Tensor, *, dice_weight: float) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=weights, ignore_index=classes.IGNORE
        )
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.cross_entropy(logits, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * soft_dice_loss(logits, target)
        return loss


def _clock(seconds: float) -> str:
    """`4h20m`, `18m`, `45s`. Short enough to sit at the end of a progress line."""
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds)}s"


def tick(message: str, *, newline: bool = False) -> None:
    """One progress line, overwritten in place in a terminal and appended in a log.

    A carriage return keeps an interactive window to a single moving line rather than
    thousands of scrolled ones; but the same bytes in a redirected file make it
    unreadable, so a non-tty gets ordinary lines instead. `flush` on both paths,
    because a progress line buffered for four minutes is not progress.
    """
    import sys

    interactive = sys.stdout.isatty()
    if newline:
        if interactive:
            sys.stdout.write(chr(10))
            sys.stdout.flush()
        return
    if interactive:
        # Pad to a fixed width so a shorter line cannot leave the tail of a longer
        # one behind it on screen.
        sys.stdout.write(chr(13) + message.ljust(118)[:118])
    else:
        sys.stdout.write(message + chr(10))
    sys.stdout.flush()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """Write then rename. A kill during `torch.save` must not destroy the last good state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_path(recipe: Recipe) -> Path:
    return paths.CHECKPOINTS_DIR / f"seg_{recipe.init}_{recipe.fingerprint()}.pt"


def run_fingerprint(recipe: Recipe, rows: list[dict]) -> str:
    """What this run *is*: the recipe plus the exact tile set, in order."""
    import segexport

    digest = hashlib.sha256()
    digest.update(recipe.fingerprint().encode("utf-8"))
    digest.update(segexport.fingerprint_rows(rows).encode("utf-8"))
    return digest.hexdigest()[:16]


def _spread(rows: list[dict], limit: int) -> list[dict]:
    """`limit` rows spread evenly across the set, not the first `limit`.

    A prefix is the wrong sample and it bit once: tile ids sort with `BRACS_...` before
    `TCGA-...`, so "the first 100 held-out tiles" was 100 BEETLE-labelled tiles and not one
    human-labelled tile - and the between-epoch validation curve was therefore measuring
    agreement with the teacher while appearing to measure accuracy.

    Deterministic (a stride, not a sample) so the same subset is scored every epoch and the
    curve is comparable from one epoch to the next.
    """
    if limit >= len(rows):
        return rows
    step = len(rows) / limit
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(limit)]


@torch.inference_mode()
def evaluate_split(
    net: nn.Module,
    dataset: segdata.SegTileDataset,
    *,
    batch_size: int,
    limit: int | None = None,
    tolerance: int = 3,
) -> dict:
    """Score a dataset per pixel, keeping the label sources apart.

    Predictions are accumulated as uint8 masks, not as logits: 3 classes at float32 over
    20,000 tiles of 224 px would be 12 GB, and the argmax is all any metric needs.
    """
    net.eval()
    rows = dataset.rows if limit is None else _spread(dataset.rows, limit)
    subset = segdata.SegTileDataset(
        rows, dataset.tiles_dir, augment_data=False,
        invert=dataset.invert, standardise=dataset.standardise,
        source_authority=dataset.source_authority,
    )
    loader = segdata.loader(subset, batch_size=batch_size, shuffle=False)

    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for images, masks in loader:
        logits = net(images)
        predictions.append(logits.argmax(dim=1).to(torch.uint8).numpy())
        truths.append(masks.to(torch.uint8).numpy())

    truth = np.concatenate(truths, axis=0)
    predicted = np.concatenate(predictions, axis=0)
    return segreport.evaluate_by_source(
        truth, predicted, [row["source"] for row in rows], tolerance=tolerance
    )


def train(
    rows: list[dict],
    *,
    recipe: Recipe = Recipe(),
    tiles_dir: Path | None = None,
    val_limit: int | None = 400,
    final_limit: int | None = None,
    heartbeat: int = 5,
    progress: Callable[[str], None] = print,
) -> dict:
    """Fit the network, resuming if a matching checkpoint exists. Returns the report."""
    tiles_dir = Path(tiles_dir or paths.TILES_DIR)
    torch.manual_seed(recipe.seed)
    torch.set_num_threads(paths.torch_threads())

    train_rows, test_rows = splits.split(rows)
    weights = torch.tensor(
        splits.pixel_class_weights(train_rows, source_authority=recipe.source_authority),
        dtype=torch.float32,
    )
    progress(
        f"  {len(train_rows):,} train / {len(test_rows):,} test tiles"
        f"   weights {[round(float(w), 3) for w in weights]}"
    )

    net = segnet.freeze_early_encoder(segnet.ResNet18UNet(recipe.init))
    optimiser = torch.optim.AdamW(
        segnet.parameter_groups(
            net, encoder_lr=recipe.encoder_lr, decoder_lr=recipe.decoder_lr
        ),
        weight_decay=recipe.weight_decay,
    )
    loss_fn = Loss(weights, dice_weight=recipe.dice_weight)
    history = History()
    fingerprint = run_fingerprint(recipe, rows)
    path = checkpoint_path(recipe)

    # --- resume -------------------------------------------------------------
    if path.is_file():
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError):
            saved = None
            progress(f"  checkpoint {path.name} is unreadable; starting fresh")
        if saved is not None:
            if saved.get("fingerprint") != fingerprint:
                # Not this run. A checkpoint from a different tile set or recipe would
                # produce a model whose recorded history is a fiction.
                progress(
                    f"  checkpoint {path.name} belongs to a different run "
                    f"({saved.get('fingerprint')} != {fingerprint}); starting fresh"
                )
            else:
                net.load_state_dict(saved["model"])
                optimiser.load_state_dict(saved["optimiser"])
                history = History(**saved["history"])
                progress(f"  resumed at epoch {history.epochs_done}/{recipe.epochs}")

    train_set = segdata.SegTileDataset(
        train_rows, tiles_dir, augment_data=True,
        invert=recipe.invert, standardise=recipe.standardise,
        source_authority=recipe.source_authority, seed=recipe.seed,
    )
    test_set = segdata.SegTileDataset(
        test_rows, tiles_dir, augment_data=False,
        invert=recipe.invert, standardise=recipe.standardise,
        source_authority=recipe.source_authority,
    )

    rng = np.random.default_rng(recipe.seed)
    for epoch in range(history.epochs_done, recipe.epochs):
        started = time.time()
        train_set.set_epoch(epoch)

        indices = np.arange(len(train_rows))
        if recipe.tiles_per_epoch and recipe.tiles_per_epoch < len(indices):
            indices = rng.choice(indices, recipe.tiles_per_epoch, replace=False)
        epoch_rows = [train_rows[int(i)] for i in indices]
        epoch_set = segdata.SegTileDataset(
            epoch_rows, tiles_dir, augment_data=True,
            invert=recipe.invert, standardise=recipe.standardise,
        source_authority=recipe.source_authority, seed=recipe.seed,
        )
        epoch_set.set_epoch(epoch)

        net.train()
        total = 0.0
        batches = 0
        seen = 0
        total_batches = (len(epoch_set) + recipe.batch_size - 1) // recipe.batch_size
        batch_started = time.time()

        for images, masks in segdata.loader(
            epoch_set, batch_size=recipe.batch_size, shuffle=True
        ):
            optimiser.zero_grad()
            loss = loss_fn(net(images), masks)
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            batches += 1
            seen += int(images.shape[0])

            # A line every `heartbeat` batches, because an epoch is three quarters of an
            # hour and a window that prints nothing for that long is indistinguishable
            # from a hung one. The rate is measured over the epoch so far rather than
            # over the last batch, which on a CPU bounces by a factor of two.
            if heartbeat and (batches % heartbeat == 0 or batches == total_batches):
                elapsed = time.time() - batch_started
                rate = seen / elapsed if elapsed > 0 else 0.0
                remaining = (len(epoch_set) - seen) / rate if rate > 0 else 0.0
                epochs_left = recipe.epochs - epoch - 1
                run_left = remaining + epochs_left * (len(epoch_set) / rate if rate else 0)
                tick(
                    f"    epoch {epoch + 1}/{recipe.epochs}  "
                    f"batch {batches}/{total_batches}  "
                    f"loss {total / batches:.4f}  "
                    f"{rate:.1f} tiles/s  "
                    f"epoch left {_clock(remaining)}  run left {_clock(run_left)}"
                )

        if heartbeat:
            tick("", newline=True)
        mean_loss = total / max(1, batches)
        scored = evaluate_split(
            net, test_set, batch_size=recipe.batch_size, limit=val_limit
        )
        human = scored["by_source"].get(classes.SOURCE_BCSS, scored["all"])
        in_situ = human["per_class"]["non_invasive_epithelium"]["dice"]
        invasive = human["per_class"]["invasive_epithelium"]["dice"]

        history.epochs_done = epoch + 1
        history.train_loss.append(round(mean_loss, 5))
        history.val_dice_in_situ.append(in_situ)
        history.val_dice_invasive.append(invasive)
        history.seconds.append(round(time.time() - started, 1))

        progress(
            f"  epoch {epoch + 1:>2}/{recipe.epochs}  loss {mean_loss:.4f}  "
            f"in-situ Dice {in_situ}  invasive Dice {invasive}  "
            f"{history.seconds[-1] / 60:.1f} min"
        )

        # After every epoch, atomically. Ten hours is long enough to lose.
        _atomic_torch_save({
            "model": net.state_dict(),
            "optimiser": optimiser.state_dict(),
            "history": asdict(history),
            "fingerprint": fingerprint,
            "recipe": asdict(recipe),
        }, path)

    # Scoring 4,537 tiles is minutes of forward passes plus a per-tile boundary F1, so a
    # smoke run caps it: proving the loss falls should not cost as much as an epoch.
    scope = "full" if final_limit is None else f"first {final_limit}"
    progress(f"  scoring the {scope} held-out set")
    final = evaluate_split(
        net, test_set, batch_size=recipe.batch_size, limit=final_limit
    )

    return {
        "recipe": asdict(recipe),
        "fingerprint": fingerprint,
        "checkpoint": str(path),
        "tiles": {"train": len(train_rows), "test": len(test_rows)},
        "patients": {
            "train": len({r["patient"] for r in train_rows}),
            "test": len({r["patient"] for r in test_rows}),
        },
        "class_weights": [round(float(w), 4) for w in weights],
        "history": asdict(history),
        "held_out": final,
        "held_out_scope": "full" if final_limit is None else f"first {final_limit} tiles",
        "headline": segreport.headline(final),
        "net": net,
    }
