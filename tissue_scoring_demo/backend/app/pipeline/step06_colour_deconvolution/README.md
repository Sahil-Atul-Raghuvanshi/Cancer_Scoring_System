# Step 06 — Colour deconvolution ★ the fork

Implemented. Catalogue entry: `app/data/pipeline_steps.py`, `id="colour-deconvolution"`.
Guide: `docs/guides/demo-pipeline-guide.md`, step 7 there — the guide's step 6 is stain
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
    Step 8 — region model      Step 14 — measure
```

- **haematoxylin → the region model**, in place of RGB. The largest single source
  of colour difference between the H&E slide and the five IHC slides is the brown;
  after this step it is gone from what the model sees, which is what lets one
  detector cover all six rather than one per stain.
- **DAB → the measurement** (step 14, via `stain_input`), on an absolute optical
  density scale. Rule 3:
  separate the stains before you threshold, and threshold the DAB channel only.

Both channels come out of a single `deconvolve()` call in
`deconvolution_service._run` **for this screen**. That sentence used to end there,
and it was the most load-bearing false claim in this package.

**What is actually shared, and what is not.** Steps 7 and 8 import `separate()` from
this module directly, so the model arm really does un-mix through this code.
The measurement arm does not go through this *service*: steps 11 and 14 call
`step13_nuclei_segmentation.stain_input.concentrations`, a second wrapper over the
same two primitives (`common.imaging.optical_density` and `common.stains.unmix`).
That is defensible - it un-mixes a tile it has just read at the nuclei model's own
resolution, which is not the tile this screen is showing - but it is two wrappers,
and two wrappers drift. Two ways they had already drifted, both now closed:

- `concentrations` took `optical_density`'s default floor of 1.0 while steps 4 and 5
  passed `settings.calibration_od_floor`. They agreed only because the setting
  happens to be 1.0. It now passes the setting.
- step 11's basis is a setting (`nuclei_macenko_per_slide`) and step 14's is fixed,
  so turning that setting on made detection and measurement two different un-mixings
  of one slide, silently. Step 14 now refuses when they disagree.

**Which slide this step runs on.** Both, once each - see `runs_on` in the catalogue.
That is new, and it is what the step was always for: on an H&E the DAB panel is a
picture of nothing, and the guide's own instruction for this step is "one IHC tile,
three panels". Until the screen could be pointed at the immunostained slide, the one
channel the score is actually made of was the one channel it never showed.

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
