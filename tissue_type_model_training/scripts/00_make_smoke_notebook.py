"""Generate notebooks/00_smoke_test.ipynb.

The notebook is generated rather than hand-written so the cells stay diffable and so
the gate thresholds live in one place. Re-run this script after changing a threshold;
do not edit the .ipynb by hand.

    python scripts/00_make_smoke_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "00_smoke_test.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS = [
    md(
        """
# Notebook 00 — smoke test: gates G0 and G0b

**Write and pass this before anything else.** Nothing downstream is worth writing until both
gates are green, because both of them test the *plumbing* that every later number depends on.

| Gate | Question | Failure looks like |
| --- | --- | --- |
| **G0** | Does the notebook get the **same H channel** the API would? | a "model problem" that is a transform problem, found a week late |
| **G0b** | Does the H channel of an **IHC** slide sit where the H channel of an **H&E** slide sits? | a model trained on H&E that quietly under-performs on every slide we actually score |

G0 is approach 1's original gate. G0b is new and it serves **all three approaches** — approach 1
trains on BCSS H&E, approach 4a distils from our own H&E, approach 4b trains on BEETLE H&E, and
*all three* are served on IHC. Nothing else in the project measures that step.

No downloads are needed for G0. G0b needs two slides from `images/`, which are already on disk.
"""
    ),
    code(
        """
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd().parent / "src"))

import backend_path  # noqa: F401  - sys.path side effect, must precede app.* imports
import hchannel

from PIL import Image

from app.common.imaging import optical_density
from app.common.stains import RUIFROK_HDAB
from app.ingestion.slide_reader import SlideReader
from app.pipeline.step06_colour_deconvolution.deconvolution import separate

REPO = Path.cwd().parent
IMAGES = REPO.parent / "tissue_scoring_demo" / "images"
REPORTS = REPO / "reports"
REPORTS.mkdir(exist_ok=True)

results: dict[str, object] = {}
print("numpy", np.__version__)
print("OD_CLIP", hchannel.OD_CLIP, "| OD_FLOOR", hchannel.OD_FLOOR)
print("images/ present:", IMAGES.is_dir())
"""
    ),
    md(
        """
---

## G0.1 — the H channel, computed two ways, must agree

`hchannel.haematoxylin_od` is the one definition of the H channel in this codebase. Step 8 calls
it, the exporter calls it, the Dataset calls it. Here it is checked against the same two backend
functions applied by hand — if these ever disagree, one of the three callers is being served a
different input from the one the model was fitted on.

**This is the check that catches a stray `skimage.color.rgb2hed`.**
"""
    ),
    code(
        """
# A synthetic tile with a bit of everything: bright glass, mid stroma, dark nuclei.
tile = np.full((224, 224, 3), 235, dtype=np.uint8)
tile[40:120, 40:120] = (150, 130, 190)   # haematoxylin-ish
tile[130:200, 130:200] = (170, 140, 100) # DAB-ish

white = hchannel.white_point(tile, percentile=99.0)

# through the module
h_module = hchannel.haematoxylin_od(tile, white)

# by hand, through the same two backend functions
od = optical_density(tile, white, floor=hchannel.OD_FLOOR)
h_hand = separate(od, RUIFROK_HDAB).haematoxylin.astype(np.float32)

max_abs = float(np.max(np.abs(h_module - h_hand)))
print(f"max |module - hand| = {max_abs:.3e}")
assert np.allclose(h_module, h_hand, atol=1e-6), "G0.1 FAILED - two paths to the H channel disagree"
results["g0_1_max_abs_diff"] = max_abs
print("G0.1 PASS")
"""
    ),
    md(
        """
## G0.2 — the transform contract: quantise round-trip, and the polarity

Three properties of `to_model_input` that every later notebook assumes:

1. `quantise` → `dequantise` loses less than one OD level (0.0059), so the stored uint8 tile is a
   faithful stand-in for the density.
2. The output is `3xHxW` float32, three identical channels, ImageNet-normalised.
3. **Polarity:** a pure-haematoxylin patch maps near the top of the range and a pure-DAB patch near
   the bottom. Our H channel puts **nuclei bright**.
"""
    ),
    code(
        """
h = hchannel.dequantise(hchannel.quantise(h_module))
round_trip = float(np.max(np.abs(h - np.clip(h_module, 0.0, hchannel.OD_CLIP))))
one_level = hchannel.OD_CLIP / 255.0
print(f"quantise round-trip max err = {round_trip:.5f} OD  (one level = {one_level:.5f})")
assert round_trip <= one_level, "G0.2 FAILED - quantisation loses more than one level"

x = hchannel.to_model_input(h_module)
assert x.shape == (3, 224, 224), f"G0.2 FAILED - shape {x.shape}"
assert x.dtype == np.float32, f"G0.2 FAILED - dtype {x.dtype}"
assert np.allclose(x[0] * hchannel.IMAGENET_STD[0] + hchannel.IMAGENET_MEAN[0],
                   x[1] * hchannel.IMAGENET_STD[1] + hchannel.IMAGENET_MEAN[1],
                   atol=1e-5), "G0.2 FAILED - the three channels are not the same image"

# polarity, on the two synthetic patches
h_stain = float(np.mean(h_module[40:120, 40:120]))
h_dab = float(np.mean(h_module[130:200, 130:200]))
print(f"mean H on haematoxylin patch = {h_stain:.3f} OD")
print(f"mean H on DAB patch          = {h_dab:.3f} OD")
assert h_stain > h_dab, "G0.2 FAILED - polarity inverted: DAB is brighter than haematoxylin"

results["g0_2_round_trip_od"] = round_trip
results["g0_2_h_on_haem"] = h_stain
results["g0_2_h_on_dab"] = h_dab
print("G0.2 PASS")
"""
    ),
    md(
        """
---

# G0b — the H&E → IHC haematoxylin bridge

**The assumption every approach rests on, and the only one nothing else measures.**

The model is trained on a haematoxylin channel derived from **H&E** slides and served on one
derived from **IHC**, where haematoxylin is only the counterstain — applied lightly to show nuclei
while DAB carries the signal. If the IHC H channel is systematically weaker or flatter than the
H&E one, the model meets an input distribution it never saw.

Approach 1 already trains with a stain-strength jitter, `alpha ~ U(0.80, 1.25)`. **That jitter is
the hedge, and this gate is the test of whether the hedge is wide enough.**

The measurement: take one case that has both stains, push tiles from each through the *identical*
production path, and compute the **effective alpha** — the ratio of the IHC H-channel p95 to the
H&E one. That ratio is exactly what the augmentation would have to absorb.
"""
    ),
    code(
        """
CASE = "CAN_00251_26"
PAIRS = {"H&E": IMAGES / f"{CASE}_H&E.svs", "IHC (A = CD44)": IMAGES / f"{CASE}_A.svs"}
TARGET_MPP = 0.5
TILE_PX = 224
N_TILES = 200
JITTER_LO, JITTER_HI = 0.80, 1.25   # approach 1, Part 6

for name, path in PAIRS.items():
    print(f"{name:16s} {'OK' if path.is_file() else 'MISSING'}  {path.name}")
assert all(p.is_file() for p in PAIRS.values()), "G0b needs both slides of one case in images/"
"""
    ),
    code(
        """
def sample_h_tiles(path: Path, *, n: int, seed: int = 0) -> tuple[np.ndarray, dict]:
    \"\"\"n H-channel tiles at TARGET_MPP, sampled from tissue, plus the slide metadata.

    Two things here are not optional, and getting either wrong makes the gate measure
    the wrong thing.

    **The resample happens in intensity space, before the logarithm.** These slides are
    0.2222 um/px with no pyramid level near 0.5, so `best_level_for_mpp` returns level 0
    and the downsample is ours to do - with BOX averaging on the RGB, exactly as step 5
    `read_tile` does. Averaging transmissions is what a coarser sensor physically does;
    averaging densities is the log of a geometric mean, biased low, and no instrument
    records it. An earlier version of this cell skipped the resample and compared at
    0.2222 um/px, which flatters the dark tail.

    **Tissue is found the cheap way** - reject tiles near glass - because this gate is
    about intensity distributions, not segmentation. The white point uses the same
    99th-percentile rule the exporter uses, so both slides are treated identically.
    \"\"\"
    rng = np.random.default_rng(seed)
    with SlideReader(path) as sr:
        mpp = sr.mpp
        assert mpp, f"{path.name} records no mpp - resolve it before trusting this gate"
        level = sr.best_level_for_mpp(TARGET_MPP)
        down = sr.level_downsamples[level]
        span = int(round(TILE_PX * TARGET_MPP / mpp))     # level-0 pixels covered
        read_px = max(1, int(round(span / down)))         # pixels actually read
        w, h = sr.dimensions
        meta = {"mpp": mpp, "level": level, "downsample": float(down),
                "level_mpp": round(mpp * down, 4), "span_l0": span, "read_px": read_px,
                "resampled_to": TARGET_MPP, "dims": [int(w), int(h)], "vendor": sr.vendor}

        kept, tries = [], 0
        while len(kept) < n and tries < n * 60:
            tries += 1
            x = int(rng.integers(0, max(1, w - span)))
            y = int(rng.integers(0, max(1, h - span)))
            pil = sr.read_region_pil((x, y), level, (read_px, read_px))
            # BOX average down to the target spacing, on RGB, before any logarithm
            if pil.size != (TILE_PX, TILE_PX):
                pil = pil.resize((TILE_PX, TILE_PX), Image.BOX)
            rgb = np.asarray(pil)
            if float(rgb.mean()) > 215.0:          # glass
                continue
            if float(rgb.std()) < 6.0:             # featureless
                continue
            white = hchannel.white_point(rgb, percentile=99.0)
            kept.append(hchannel.haematoxylin_od(rgb, white).ravel())
        meta["tiles"] = len(kept)
        meta["accept_rate"] = round(len(kept) / max(1, tries), 3)
    # Return the tiles themselves as well: the gate below has to re-render them
    # through the jitter grid, and re-reading two whole slides to do it would be
    # twenty wasted seconds and a second chance to sample differently.
    return (np.concatenate(kept) if kept else np.zeros(0, np.float32)), kept, meta


PCTS = [50, 75, 90, 95, 99, 99.5]

stats = {}
tile_store: dict[str, list] = {}
for name, path in PAIRS.items():
    values, raw, meta = sample_h_tiles(path, n=N_TILES, seed=1)
    tile_store[name] = [t.reshape(TILE_PX, TILE_PX) for t in raw]
    q = {f"p{p}": float(v) for p, v in zip(PCTS, np.percentile(values, PCTS))}
    stats[name] = {
        **q,
        "mean": float(values.mean()),
        "frac_above_0_15": float((values > 0.15).mean()),   # Macenko beta, step 5 `stained`
        "frac_clipped": float((values > hchannel.OD_CLIP).mean()),
        **meta,
    }
    st = stats[name]
    print(f"{name:16s} tiles={st['tiles']:4d} level={st['level']} "
          f"read={st['read_px']}px @ {st['level_mpp']} -> {TILE_PX}px @ {TARGET_MPP} um/px")
    print(f"{'':16s} " + "  ".join(f"p{p}={q[f'p{p}']:.3f}" for p in PCTS))
    print(f"{'':16s} stained={st['frac_above_0_15']:.3f}  clipped={st['frac_clipped']:.5f}")

he_tiles = tile_store["H&E"]
ihc_tiles = tile_store["IHC (A = CD44)"]
"""
    ),
    code(
        """
he, ihc = stats["H&E"], stats["IHC (A = CD44)"]

# ---- part 1: the diagnosis. Is this a strength difference, or a shape one? -------
ratios = {p: ihc[f"p{p}"] / he[f"p{p}"] for p in PCTS if he[f"p{p}"] > 1e-6}
print("IHC / H&E density ratio, by percentile:")
for p, v in ratios.items():
    print(f"  p{p:<5} {v:6.3f}")

mid = [ratios[p] for p in (75, 90, 95, 99) if p in ratios]
spread = max(mid) / min(mid) if mid else float("inf")
print(f"\\nspread over p75..p99 = {spread:.2f}x   (1.00 would be a pure strength difference)")

if spread > 1.5:
    print("=> SHAPE difference. No alpha, and no per-slide normalisation, can close this:")
    print("   every such correction is monotone, and a monotone map cannot alter the")
    print("   ratio between two quantiles of one image. See test_gamma_and_standardise.py.")

# ---- part 2: the gate. Does the TRAINING envelope CONTAIN the serving data? ------
# Aligning the two stains is not achievable and not the goal. What matters is that the
# model never has to extrapolate: for each percentile, the IHC value must fall inside
# the range the H&E tiles span once the (alpha, gamma) jitter has been applied.
import datasets   # noqa: E402  - for the jitter ranges, so gate and training agree

J = datasets.Jitter()
GRID = [(a, g) for a in J.alpha for g in J.gamma]


def model_view(h_tiles, *, alpha, gamma, standardise):
    vals = []
    for t in h_tiles:
        x = hchannel.to_model_input(t, alpha=alpha, gamma=gamma, standardise=standardise)
        vals.append((x[0] * hchannel.IMAGENET_STD[0] + hchannel.IMAGENET_MEAN[0]).ravel())
    v = np.concatenate(vals)
    return {p: float(q) for p, q in zip(PCTS, np.percentile(v, PCTS))}


coverage = {}
for standardise in (False, True):
    ihc_q = model_view(ihc_tiles, alpha=1.0, gamma=1.0, standardise=standardise)
    views = [model_view(he_tiles, alpha=a, gamma=g, standardise=standardise)
             for a, g in GRID]
    covered = []
    print(f"\\nstandardise={standardise}:")
    print(f"  {'pct':>5} {'IHC (served)':>13} {'H&E envelope (trained)':>26}  inside")
    for p in PCTS:
        lo = min(v[p] for v in views)
        hi = max(v[p] for v in views)
        inside = lo <= ihc_q[p] <= hi
        covered.append(inside)
        print(f"  {p:>5} {ihc_q[p]:13.4f}   [{lo:9.4f}, {hi:9.4f}]  {'yes' if inside else 'NO'}")
    coverage[standardise] = covered
    print(f"  covered {sum(covered)}/{len(PCTS)}")

# The reference point: what the alpha-only jitter used to cover.
ihc_q = model_view(ihc_tiles, alpha=1.0, gamma=1.0, standardise=False)
old = [min(model_view(he_tiles, alpha=a, gamma=1.0, standardise=False)[p] for a in J.alpha)
       <= ihc_q[p] <=
       max(model_view(he_tiles, alpha=a, gamma=1.0, standardise=False)[p] for a in J.alpha)
       for p in PCTS]

best = max(coverage, key=lambda k: sum(coverage[k]))
n_best, n_tot = sum(coverage[best]), len(PCTS)
verdict = "PASS" if n_best == n_tot else "FAIL"

print(f"\\nalpha only (gamma=1, no standardise): {sum(old)}/{n_tot} covered")
print(f"alpha + gamma, standardise={best}:     {n_best}/{n_tot} covered")

results["g0b"] = {
    "verdict": verdict,
    "ratios": {str(k): v for k, v in ratios.items()},
    "spread": spread,
    "diagnosis": "shape" if spread > 1.5 else "strength",
    "coverage": {str(k): v for k, v in coverage.items()},
    "coverage_alpha_only": old,
    "recommended_standardise": bool(best),
    "jitter": {"alpha": list(J.alpha), "beta": list(J.beta), "gamma": list(J.gamma)},
    "stats": stats,
}

print()
if verdict == "PASS":
    print(f"G0b PASS - every percentile of the served IHC distribution falls inside the")
    print(f"envelope the model trains over, with standardise={best}. The model")
    print(f"interpolates rather than extrapolates. Set standardise={best} in the Dataset")
    print(f"and in step 8, and record it in the manifest via hchannel.descriptor().")
else:
    print(f"G0b FAIL - {n_tot - n_best} percentile(s) of the served distribution lie outside")
    print(f"anything the model will see in training, so it must extrapolate there.")
    print(f"In order of preference:")
    print(f"  1. widen Jitter.gamma - it is the only knob that reshapes, and the envelope")
    print(f"     above shows which end needs the room;")
    print(f"  2. turn on standardise if it is not already on (it fixes the scale gap that")
    print(f"     alpha's range cannot reach);")
    print(f"  3. only then consider raising OD_CLIP - it changes the stored tile format.")
"""
    ),
    md(
        """
### What G0b does *not* prove

It compares intensity distributions, not morphology. A pass means the model's input range is the
same on both stains; it does not mean a duct looks the same in an IHC counterstain as it does in
H&E. That is a genuinely open question, and the honest place for it is the report — not this gate.

Two caveats worth writing down beside the numbers:

- One case (`CAN_00251_26`), one IHC marker (`A` = CD44). Widen to a second case and a second
  marker if the ratio lands near a boundary.
- Approach 4a's slide `00267` is our low-contrast case (17.5 mm² of tissue against ~84.8 on its
  siblings). Run this cell against it too before trusting the number as cohort-wide.
"""
    ),
    code(
        """
out = REPORTS / "00_smoke_test.json"
results["gates"] = {"G0": "PASS", "G0b": results["g0b"]["verdict"]}
out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
print(f"written {out}")
print(json.dumps(results["gates"], indent=2))
if results["gates"]["G0b"] != "PASS":
    raise SystemExit("G0b is not green - settle it before Notebook 01.")
"""
    ),
]

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB.parent.mkdir(exist_ok=True)
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"written {NB}  ({len(CELLS)} cells)")
