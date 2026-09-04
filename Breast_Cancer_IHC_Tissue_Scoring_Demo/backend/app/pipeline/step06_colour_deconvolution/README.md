# Step 06 — Colour deconvolution ★ the fork

Implemented. Catalogue entry: `app/data/pipeline_steps.py`, `id="colour-deconvolution"`.
Guide: `docs/demo-pipeline-guide.md`, step 7 there — the guide's step 6 is stain
normalisation, which this pipeline skips (see Rule 2), so everything from here on
is one lower in this codebase than in the guide.

## Input

Step 5's optical density tile, taken from `density_service.transformed_tile()`.
Not the RGB tile, and not a density of this step's own: colour deconvolution is a
*linear* inverse and a mixture of stains is only linear in optical density, so the
requirement is the validity condition of the arithmetic rather than a data
dependency.

## Output

Three channels per basis — haematoxylin, DAB, and the residual that is neither —
plus the numbers that say whether the separation was clean and what the choice of
stain vectors cost.

## Why this is the fork

One deconvolution, two branches. Rule 2:

```
              tile (raw, calibrated)
                       |
              Step 6 — deconvolve
                 /            \
        H channel              DAB channel
             |                      |
    Step 8 — region model   Steps 13-15 — measure
```

- **haematoxylin → the region model**, in place of RGB. The largest single source
  of colour difference between the H&E slide and the five IHC slides is the brown;
  after this step it is gone from what the model sees, which is what lets one
  detector cover all six rather than one per stain.
- **DAB → the measurement**, on an absolute optical density scale. Rule 3:
  separate the stains before you threshold, and threshold the DAB channel only.

Both channels come out of a single `deconvolve()` call in
`deconvolution_service._run`, so the Y is a fact about the code and not a drawing.

## Fixed vectors, and why the demo computes the alternative anyway

The matrix could come from Ruifrok & Johnston's published constants or from an
estimate of this tile's own two stain directions (Macenko — the same estimator
step 5 draws its arms with). This step computes **both, through identical
arithmetic**, and reports what each would score under the same absolute cut.

They differ, and the difference is the argument. A per-image estimate rescales
itself to whatever slide it is given, so "0.4 DAB" would mean a different amount
of dye on every slide in a study and two slides could not be compared. The
measurement branch therefore only ever sees the fixed basis; the estimate is
served so a reader can watch the score move and see why.

If the fixed vectors fit a scanner badly, the answer is to calibrate them once on
control slides and freeze them — not to re-estimate per slide.

## The one rule about where this code lives

`app.common.stains` holds the vectors and the basis arithmetic; `deconvolution.py`
holds the step. Every consumer imports `deconvolve` from this package.
**Nothing is permitted to reach for `skimage.color.rgb2hed` on its own** — two
implementations drift, and when they do the IHC slides score nothing like the H&E.
The guide names this failure explicitly.

## Layout

| File | What it holds |
| --- | --- |
| `deconvolution.py` | The algorithm: both bases, the channels, and the checks |
| `overlay.py` | The four panels — input, haematoxylin, DAB, residual |
| `pipeline.py` | `run(context)` for the end-to-end runner |

The service, schemas and endpoints sit where every other step's do:
`app/services/deconvolution_service.py`, `app/schemas/deconvolution.py`,
`app/api/v1/endpoints/deconvolution.py`.

## What is checked rather than claimed

- **Exactness** — the basis is square, so un-mixing and recomposing must return
  the input. Reported as the worst error in optical density.
- **Negatives** — a pixel outside the cone the two vectors span needs a negative
  amount of one stain. Counted, never clipped.
- **Disentangling** — the three optical density channels of a stained tile
  correlate strongly, because all of them track total dye. The two stain channels
  afterwards should not. Both correlations are measured on the reader's own tile.

## References

Ruifrok AC, Johnston DA. *Quantification of histochemical staining by colour
deconvolution.* Anal Quant Cytol Histol 23(4):291-299 (2001).
Macenko M et al. ISBI 2009:1107-1110 — the estimator, kept for the comparison.
