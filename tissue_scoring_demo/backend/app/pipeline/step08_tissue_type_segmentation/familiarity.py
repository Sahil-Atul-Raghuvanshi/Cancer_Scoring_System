"""Step 8's refusal: windows the model is not allowed to answer for.

The model has three outputs and no way to decline, so anything it is shown gets a
class - and a class with a confident probability, because softmax confidence measures
which of the three a window is nearest, not whether it is near any of them. P-05 is
that failure measured: on CAN_00259 and CAN_00865 a scanner's flat lilac fill reached
the H&E head and came back invasive at P = 0.61, became the slide's largest ROI, and
was carried onto the IHC. Step 3 now keeps fill out of the tissue mask; this module is
the second, independent line, so the same failure needs both to break at once.

Two tests, run on every window before its answer is kept. Either refuses it.

**1. Flat pixels - nothing a scanner imaged is constant.** The share of a window's
pixels exactly equal to both their right and lower neighbours. Measured on the demo's
slides, at 0.5, 1 and 2 um/px: scanner fill 1.000; H&E tissue at most 0.001; fat at
most 0.028; the worst genuine window anywhere, a pale IHC field, 0.392. Refusing at
0.75 sits nearly twice above that and well below the fill. It needs no model, no
training data and no colour, so it holds for a fill of any colour on any head.

**2. Unfamiliar features - the model has never seen anything like this.** The 1,024
numbers the head reads (two frozen ResNet18 bodies, concatenated) compared with the
training set's, as the smallest class-conditional Mahalanobis distance under a
Ledoit-Wolf covariance. A window farther than the threshold is outside what the head
was fitted on, and its probabilities are not evidence of anything.

Cosine k-nearest-neighbour distance, the obvious alternative, was measured first and
cannot see the fill at all: on the H&E head it lands at 0.630, inside real tissue's
range (median 0.598, 95th percentile 0.652). Mahalanobis does better, and covariance is
why - the fill is not far from the training tiles in every direction, it is far along
the few directions real tissue never moves in.

**But the distance is the backstop, and the flat test is what stops the fill.** Measured
over six OncoStem H&E cases, the fill's distance sits *inside* genuine tissue's spread on
four of the eight heads (112 um H&E: fill 1,571, tissue 95th percentile 1,830). No cut on
those heads refuses the fill without refusing real tumour, and a refused window leaves
the score. The flat test separates the two on all eight - 1.000 against 0.0% of genuine
windows over the threshold - so it is the one that decides P-05.

**So the cut sits at the farthest held-out tile, not at a percentile of them.** Held-out
institutions are the closest thing on file to "a slide this model has not seen, of
tissue it should answer for", and the OncoStem slides sit farther out still: their 99th
percentile refused 2-8% of genuine OncoStem tissue, 33% on CAN_00251 with one head. At
their maximum it refuses 0-0.3%, and what it still catches is what nothing genuine on
file resembles - IHC tissue through the 224 um H&E head has a median of 4,896 against a
cut of 4,226, so most of it is refused. A head calibrated on fewer than `tissue_type_familiarity_min_held_out`
held-out tiles gets no distance gate at all: the 672 um heads have 76, and their maximum
still refused 3.7-5% of genuine tissue.

**What a refusal does.** The window leaves the class map as though it had never been
run: no label, no probabilities, out of `grid.inside`. That last part is load-bearing -
step 9 closes holes *within* `inside`, and a window that only had its probabilities
zeroed could be annexed by the close. The verdict is kept in its own array, drawn as
"cannot be determined", and counted in the report.

**What it does not do.** It does not catch the IHC failure on the haematoxylin head
(P-12, P-18). There, IHC tissue measures *nearer* the training set than H&E does - the
haematoxylin channel of an IHC slide is a very plausible haematoxylin channel - so the
features genuinely do not know. That failure is about what the tissue *is*, not what it
looks like, and no distance in this space can see it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Refusal codes, stored per window. 0 is "answered".
ANSWERED: int = 0
FLAT: int = 1
UNFAMILIAR: int = 2

REASONS: dict[int, str] = {
    FLAT: "flat",
    UNFAMILIAR: "unfamiliar",
}

#: Suffix of the reference file published beside a checkpoint.
SUFFIX = ".familiarity.npz"

#: Bumped when what a reference holds, or how a distance is computed from it, changes.
#: Recorded in step 8's run key, so a pass gated under an older rule is recomputed.
VERSION = 1


def flat_share(pixels: np.ndarray) -> float:
    """Share of pixels equal to both their right and lower neighbour, in every channel.

    Works on either input contract: HxW optical density or HxWx3 sRGB. A digital fill is
    one value repeated, so it scores 1.0 in both; anything imaged carries sensor noise
    and scores near zero. See the module docstring for the measured spread.
    """
    a = pixels if pixels.ndim == 3 else pixels[..., None]
    if a.shape[0] < 2 or a.shape[1] < 2:
        return 0.0
    core = a[:-1, :-1]
    same = np.all(core == a[1:, :-1], axis=-1) & np.all(core == a[:-1, 1:], axis=-1)
    return float(same.mean())


@dataclass(frozen=True)
class Reference:
    """What "familiar" means for one checkpoint: per-class Gaussians and a cut."""

    #: (classes, dim) float32 class means of the training features.
    means: np.ndarray
    #: (classes, dim, dim) float32 shrunk precision matrices.
    precisions: np.ndarray
    #: Distances above this are refused.
    threshold: float
    #: The held-out quantile the threshold was read at, and the calibration population.
    quantile: float
    held_out: int
    train: int
    #: Fingerprint of the training manifest the features came from - the same one the
    #: checkpoint's manifest records, compared at load.
    fingerprint: str

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    def distance(self, features: np.ndarray) -> np.ndarray:
        """Smallest class-conditional squared Mahalanobis distance, per row."""
        f = np.asarray(features, dtype=np.float32)
        best = np.full(len(f), np.inf, dtype=np.float32)
        for mean, precision in zip(self.means, self.precisions, strict=True):
            centred = f - mean
            d = np.einsum("ij,jk,ik->i", centred, precision, centred)
            best = np.minimum(best, d)
        return best


def fit(
    features: np.ndarray,
    labels: np.ndarray,
    held_out: np.ndarray,
    *,
    quantile: float,
    fingerprint: str,
) -> tuple[Reference, np.ndarray]:
    """A reference from training features; the cut from the held-out ones.

    The Gaussians are fitted on the training rows only, so the held-out distances are
    honest - a held-out tile was never part of the distribution it is measured against.
    Returns the reference and the held-out distances, for the build script to report.
    """
    from sklearn.covariance import LedoitWolf

    features = np.asarray(features, dtype=np.float32)
    train = ~np.asarray(held_out, dtype=bool)

    means, precisions = [], []
    for label in np.unique(labels[train]):
        rows = features[train & (labels == label)]
        model = LedoitWolf().fit(rows)
        means.append(rows.mean(axis=0))
        precisions.append(model.precision_)

    partial = Reference(
        means=np.stack(means).astype(np.float32),
        precisions=np.stack(precisions).astype(np.float32),
        threshold=float("inf"),
        quantile=quantile,
        held_out=int((~train).sum()),
        train=int(train.sum()),
        fingerprint=fingerprint,
    )
    distances = partial.distance(features[~train])
    threshold = float(np.quantile(distances, quantile))
    return (
        Reference(**{**partial.__dict__, "threshold": threshold}),
        distances,
    )


def save(reference: Reference, path: Path) -> None:
    np.savez_compressed(
        path,
        version=np.array(VERSION),
        means=reference.means,
        precisions=reference.precisions,
        threshold=np.array(reference.threshold),
        quantile=np.array(reference.quantile),
        held_out=np.array(reference.held_out),
        train=np.array(reference.train),
        fingerprint=np.array(reference.fingerprint),
    )


def load(path: Path) -> Reference:
    """Read a reference, refusing one written under a different rule."""
    with np.load(path) as stored:
        version = int(stored["version"])
        if version != VERSION:
            raise ValueError(
                f"{path.name} is a version {version} familiarity reference and this code "
                f"reads version {VERSION}. Rebuild it with "
                "scripts/build_familiarity_references.py."
            )
        return Reference(
            means=stored["means"].astype(np.float32),
            precisions=stored["precisions"].astype(np.float32),
            threshold=float(stored["threshold"]),
            quantile=float(stored["quantile"]),
            held_out=int(stored["held_out"]),
            train=int(stored["train"]),
            fingerprint=str(stored["fingerprint"]),
        )


@dataclass(frozen=True)
class Gate:
    """Both tests, as `classify` applies them."""

    #: None when the checkpoint has no reference - the flat test still runs.
    reference: Reference | None
    max_flat_share: float

    @property
    def signature(self) -> str:
        """What changes a run's answer, for step 8's cache key."""
        cut = f"{self.reference.threshold:.3f}" if self.reference else "none"
        return f"v{VERSION}/flat{self.max_flat_share:g}/maha{cut}"


__all__ = [
    "ANSWERED",
    "FLAT",
    "REASONS",
    "SUFFIX",
    "UNFAMILIAR",
    "VERSION",
    "Gate",
    "Reference",
    "fit",
    "flat_share",
    "load",
    "save",
]
