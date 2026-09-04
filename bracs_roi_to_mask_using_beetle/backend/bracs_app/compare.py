"""Are BRACS tiles interchangeable with BCSS tiles? The measurement, not the hope.

Producing 224 px uint8 PNGs in the right directory is not the same as producing
training data. This module asks the two questions that stand between the two:

**1. Are they the same kind of file?** Shape, dtype, mode, and the range of stored
values. This one is cheap and it is the one everybody checks.

**2. Do they carry the same distribution?** This is the one that decides whether adding
them helps. The stored tile is a haematoxylin optical density quantised onto [0, 255],
and optical density is a physical measurement of how much dye is in the section - so
two datasets stained in two laboratories on two scanners produce genuinely different
numbers for the same tissue. A model trained on one and shown the other is
extrapolating.

There is a measured precedent for exactly this, in this project, and it is why this
module exists rather than a comment saying "check the histograms". Approach 1's gate
G0b compared the H channel of an H&E slide against the IHC section of the *same case*
and found the ratio between them running from 0.06 at p75 to 2.06 at p99 - a 37-fold
spread where a pure strength difference would have been one constant. The conclusion
recorded there is the one that matters here: **no monotone correction can fix a shape
difference**, because a monotone map cannot change the ratio between two quantiles of
the same image. Stain normalisation, per-slide white points and the `alpha` jitter are
all monotone. Only `gamma` - and, for the scale part alone, `standardise` - move shape.

So this module reports the quantile *ratios*, not just the means, and says which of
approach 1's existing knobs already covers the gap it finds. A single "distributions
differ, p < 0.001" would be true, useless, and would not tell anyone what to do.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

#: The quantiles compared. p50 and p75 describe the bulk of the tissue, p95 and p99 the
#: nuclei. Approach 1's gate G0b used this same ladder, and the whole point of its
#: finding was that the ratio is not constant across it.
QUANTILES: tuple[float, ...] = (50.0, 75.0, 90.0, 95.0, 99.0)

#: Approach 1's stain-strength jitter range, from `datasets.Jitter`. A density ratio
#: inside this is already covered by the training augmentation and needs nothing done
#: about it; outside it, the model would be extrapolating on every BRACS tile.
JITTER_ALPHA: tuple[float, float] = (0.80, 1.25)

#: How many tiles to read per class. Every tile would be more precise and slower; 300
#: puts the standard error on a quantile ratio well below the effect being looked for.
SAMPLE = 300


def _finite(value: float, places: int = 2) -> float | None:
    """`None` for a NaN or an infinity, because JSON has no way to spell either.

    Python's `json` emits a bare `NaN`, which is not JSON and which `JSON.parse` in a
    browser rejects outright - so an empty tile class would take down the whole
    comparison endpoint rather than showing as empty.
    """
    return round(float(value), places) if np.isfinite(value) else None


@dataclass(frozen=True)
class Profile:
    """What one pool of tiles looks like, in the stored uint8 units."""

    name: str
    n: int
    shape: tuple[int, ...] | None
    dtype: str | None
    mode: str | None
    mean: float
    sd: float
    quantiles: dict[str, float]

    def as_json(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "shape": list(self.shape) if self.shape else None,
            "dtype": self.dtype,
            "mode": self.mode,
            "mean": _finite(self.mean),
            "sd": _finite(self.sd),
            "quantiles": {k: _finite(v, 1) for k, v in self.quantiles.items()},
        }


def profile(paths: list[Path], name: str, sample: int = SAMPLE) -> Profile:
    """Read up to `sample` tiles and describe them."""
    if not paths:
        return Profile(name, 0, None, None, None, float("nan"), float("nan"), {})

    chosen = paths if len(paths) <= sample else random.Random(0).sample(paths, sample)
    arrays, mode = [], None
    for path in chosen:
        with Image.open(path) as image:
            mode = image.mode
            arrays.append(np.asarray(image))

    stacked = np.stack(arrays)
    return Profile(
        name=name,
        n=len(chosen),
        shape=tuple(arrays[0].shape),
        dtype=str(arrays[0].dtype),
        mode=mode,
        mean=float(stacked.mean()),
        sd=float(stacked.std()),
        quantiles={
            f"p{q:g}": float(np.percentile(stacked, q)) for q in QUANTILES
        },
    )


def _bcss_tiles(class_name: str) -> list[Path]:
    directory = config.APPROACH1_ROOT / "data" / "tiles" / class_name
    return sorted(directory.glob("*.png")) if directory.is_dir() else []


def _bracs_tiles(class_name: str, runs: list[str] | None = None) -> list[Path]:
    found: list[Path] = []
    for run_dir in sorted(config.RUNS_DIR.iterdir()) if config.RUNS_DIR.is_dir() else []:
        if runs is not None and run_dir.name not in runs:
            continue
        directory = run_dir / "tiles" / class_name
        if directory.is_dir():
            found.extend(sorted(directory.glob("*.png")))
    return found


def compare_class(class_name: str, runs: list[str] | None = None) -> dict:
    """One class, BCSS against BRACS, with the ratios and what they imply."""
    bcss_profile = profile(_bcss_tiles(class_name), f"BCSS {class_name}")
    bracs_profile = profile(_bracs_tiles(class_name, runs), f"BRACS {class_name}")

    result: dict = {
        "class": class_name,
        "bcss": bcss_profile.as_json(),
        "bracs": bracs_profile.as_json(),
        "format_identical": None,
        "quantile_ratios": {},
        "findings": [],
    }
    if bcss_profile.n == 0 or bracs_profile.n == 0:
        result["findings"].append(
            "one side has no tiles, so there is nothing to compare for this class."
        )
        return result

    result["format_identical"] = (
        bcss_profile.shape == bracs_profile.shape
        and bcss_profile.dtype == bracs_profile.dtype
        and bcss_profile.mode == bracs_profile.mode
    )

    ratios = {
        key: (bracs_profile.quantiles[key] / bcss_profile.quantiles[key])
        if bcss_profile.quantiles[key] > 0
        else float("nan")
        for key in bcss_profile.quantiles
    }
    result["quantile_ratios"] = {k: _finite(v, 3) for k, v in ratios.items()}

    finite = [v for v in ratios.values() if np.isfinite(v) and v > 0]
    if not finite:
        return result

    # The class this app exists to harvest is the class BCSS has almost none of, so the
    # BCSS side of its comparison is a handful of tiles from one or two slides. Saying
    # this next to the ratio matters: a reader who takes a p99 built from nine tiles as
    # a population value would conclude something firm about the domain gap from what
    # is really one patient's stain.
    if min(bcss_profile.n, bracs_profile.n) < 30:
        smaller = "BCSS" if bcss_profile.n <= bracs_profile.n else "BRACS"
        result["findings"].append(
            f"only {min(bcss_profile.n, bracs_profile.n)} tiles on the {smaller} side, "
            "so these ratios describe a couple of slides rather than a dataset. Read "
            "them as a direction, not a measurement."
        )

    spread = max(finite) / min(finite)
    result["ratio_spread"] = round(spread, 2)

    # A shape difference and a strength difference need different fixes, and telling
    # them apart is the entire value of reporting a ladder rather than a mean.
    if spread > 1.5:
        result["findings"].append(
            f"the BRACS/BCSS density ratio runs from {min(finite):.2f} to "
            f"{max(finite):.2f} across the quantile ladder - a {spread:.1f}x spread. "
            "That is a difference in the SHAPE of the density distribution, not its "
            "strength, and no monotone correction removes it: not stain normalisation, "
            "not a per-slide white point, not the alpha jitter. Approach 1's `gamma` "
            "term is the knob that moves shape, and its training range of 0.5 to 3.5 "
            "is what would have to cover this."
        )
    else:
        centre = float(np.median(finite))
        inside = JITTER_ALPHA[0] <= centre <= JITTER_ALPHA[1]
        result["findings"].append(
            f"the ratio is close to constant at {centre:.2f}x across the ladder, so "
            "this is a stain-STRENGTH difference and a monotone correction fixes it. "
            + (
                f"It sits inside approach 1's alpha jitter {JITTER_ALPHA}, so the "
                "existing augmentation already covers it and nothing need change."
                if inside
                else f"It sits OUTSIDE approach 1's alpha jitter {JITTER_ALPHA}, so "
                "the model would extrapolate on every BRACS tile. Either widen alpha "
                "or train with `standardise=True`, which divides each tile by its own "
                "p99 and removes a pure scale difference by construction."
            )
        )
    return result


def compare_all(runs: list[str] | None = None) -> dict:
    """Every class, plus the count that is the actual reason this app exists."""
    import bcss

    classes = list(bcss.CLASS_NAMES)
    per_class = [compare_class(name, runs) for name in classes]

    bcss_counts = {name: len(_bcss_tiles(name)) for name in classes}
    bracs_counts = {name: len(_bracs_tiles(name, runs)) for name in classes}

    target = "non_invasive_epithelium"
    return {
        "classes": per_class,
        "tile_counts": {"bcss": bcss_counts, "bracs": bracs_counts},
        "headline": {
            "class": target,
            "bcss_tiles": bcss_counts[target],
            "bracs_tiles": bracs_counts[target],
            "multiple": (
                round(bracs_counts[target] / bcss_counts[target], 1)
                if bcss_counts[target]
                else None
            ),
            # The count is necessary and nowhere near sufficient, and saying so here
            # keeps the caveat attached to the number rather than in a paragraph
            # somewhere the number will be quoted without.
            "caveat": (
                "This is a count of tiles, not of correct tiles. Every BRACS label here "
                "is BEETLE's opinion of a region no pathologist has reviewed, so the "
                "ceiling on their accuracy is BEETLE's own external non-invasive Dice. "
                "A held-out set built from these tiles measures agreement with BEETLE, "
                "not with the truth - the only honest test of the trained model is "
                "BCSS's real class 1 tiles, kept entirely out of training."
            ),
        },
    }
