# Step 05 - Optical density

`app/data/pipeline_steps.py` (`id="optical-density"`) · implemented.

## Input

One tile at working magnification, plus step 4's white point I0

## Output

The tile in optical density, the point cloud with its two stain arms, and the
Beer-Lambert additivity check

## Contents

- `density.py` - the algorithm. Read the module docstring first: it states what
  this step measures rather than asserts, and where the transform stops being a
  measurement.
- `tiles.py` - which tile, and why that one. Scoring, the two eligibility gates,
  and the pyramid read.
- `overlay.py` - the five panels (map, tile, density, scatter, limits).
- `pipeline.py` - `run(context)`. Adapts `app.services.density_service`, and
  enforces that step 4 ran first.

`optical_density` itself lives in `app/common/imaging.py`, not here, because step
4 needed it before step 5 existed - a white point is only meaningful if you can
say what the glass measures once you divide by it. One implementation means the
noise floor step 4 quotes and the density this step computes cannot disagree.

## What this step is for

`OD = -log10(I / I0)`, per channel. One line, and the reason it is a step rather
than a utility function is the **Beer-Lambert law**: absorbance is proportional
to the concentration of the absorbing substance.

Transmitted light is **multiplicative**. Two stains stacked multiply their
transmissions, so an RGB value is a product of the things you want to separate,
and no matrix can un-mix a product. The logarithm turns it into a **sum**, and a
sum is what linear algebra inverts. That single fact is what makes step 6 valid,
and everything downstream of step 6 rests on it.

This step is also **the fork** (Rule 2). From here the model branch takes step
6's normalised pixels and the measurement branch takes these densities untouched.
They share this step and nothing after it.

## What is measured rather than asserted

Two claims, and both are checkable on the reader's own slide, so both are checked.

**The arms.** In density space one stain at any concentration lies on one ray
from the origin, `OD = c·v`. Two stains make two rays and mixtures fill the wedge
between them. `build_cloud` projects the tile's densities onto the plane holding
most of them - Macenko's method, by eigendecomposition of `OD'OD` rather than of
the covariance, because the cloud emanates from the origin and the origin is a
point the plane has to contain - and reports where the two arms fall.

The step is told nothing about haematoxylin or DAB. On the demo's slide the arms
come back **11-20 degrees from Ruifrok's haematoxylin and 22-26 from his DAB**,
which is the claim "the arms are the stains" as a number.

Those estimated vectors are for looking at, not measuring with. Estimating per
image gives every slide its own scale, so `0.4 DAB` would mean a different amount
of stain on every slide in a study - see step 6's fixed-vector choice. These arms
are the evidence that the fixed ones fit.

**The linearity.** `measure_additivity` sorts one arm's pixels from faintest to
darkest and asks whether the *direction* moves. In density space it must not; in
intensity space it must, because `I = I0·10^(-cv)` curves. On the demo's slide:
**0.85 degrees of drift in density space against 8.1 in intensity space, a factor
of 9.6.**

Two design decisions keep that from proving itself:

- The intensity-space quantity is `I0 - I`, the light removed, not `I`. `I` is
  dominated by the illumination - all tissue is some shade of bright - so angles
  between `I` vectors are small for reasons unrelated to stain.
- The pixels are selected by their direction in **intensity** space, never in
  density space. Selecting on density angle would cap how far the density
  direction could then be found to move, and the conclusion would be baked into
  the method. Selecting in the space being discredited caps the *other* number
  instead, so whatever gap survives is a floor on the real one.

A residual confound is left and stated: darker pixels in a section tend also to
be purer. It is identical in both spaces - same pixels, same bins - so it
inflates both figures and cancels in the ratio, which is why the report carries
both and the verdict turns on the ratio.

## The three decisions worth arguing about

**Two admission tests, not one, and they exclude opposite populations.** A
direction is only meaningful when the vector is long enough to have one and short
enough to be resolved.

| Test | Excludes | Why |
| --- | --- | --- |
| `beta` (Macenko, 0.15 mean OD) | too little stain | dividing a near-zero vector by its own near-zero length turns the last bit of quantisation into an angle |
| `direction_is_stable` (1.5°) | too little light | one 8-bit level at intensity `I` moves a channel's density by `1/(I ln10)` - 0.014 at I=32, **0.43 at I=1** |

The second one is not a safeguard, it is a bug fix, and the bug was visible on
screen. Both populations land at the angular **extremes**, which is exactly where
the arms are read from. Measured on the demo's slide: without the stability test
the pixels beyond the low arm had a **median green channel of 1**, and the arm
they set landed nearest **eosin** - on a haematoxylin-and-DAB slide. With it, that
arm lands on haematoxylin on every tile tried, and the wedge narrows from 68
degrees to 63, for **3% of the tile** given up — about a tenth of what would
otherwise have been plotted.

1.5 degrees is what 8-bit data supports rather than a preference: at the beta
boundary it asks for a weakest channel of about 64 levels, and at 1.5 OD for about
11. Half a degree empties the cloud, which is the honest sign that the limit
belongs to the data.

**The arms are magnitude-weighted percentiles, at Macenko's own alpha.** A pixel's
angular uncertainty is roughly its noise over its length, so a pixel at 0.15 OD
has several times the scatter of one at 0.8 while an unweighted percentile gives
them the same vote. Since the arms *are* the angular extremes, unweighted hands
the two most important numbers in the step to the least reliable pixels in the
tile.

Weighting rather than raising alpha, which looks similar and is not: alpha
discards a fixed *share* regardless of reliability; this discounts each pixel by
how reliable it is, and keeps alpha at the published value so the method is still
Macenko's.

**The tile is chosen, and the choice is not neutral.** Unlike steps 3 and 4 this
step has to pick a field of view, because a density is per pixel. Two thirds of a
stained section is counterstain and stroma, and a tile of that has *one* arm - so
picking at random regularly puts a one-armed scatter under a caption about two
stains, through no fault of the maths.

`rank_tiles` scores every block on the thumbnail steps 3 and 4 already hold:

    score = stain × mixing

`stain` is the block's mean optical density. `mixing` is `sqrt(λ2/λ1)` of its
density cloud - how far the cloud spreads off a single ray, so one stain scores
near zero and two open a wedge. Multiplied, because either alone is a bad tile.

Plus two gates, and one of them was learnt the hard way - see below.

## Measured limits, recorded so nobody re-derives them

**The mixing term is maximised by colour artefacts, not by stains.** The demo's
slide has a patch of lurid purple-and-yellow false colour near the top edge of the
section, which step 2 did not flag. It scored **highest of 400 blocks** on
`stain × mixing`, because two wrong colours are still two colours - and the arms
it produced were 78 degrees apart with DAB 39 degrees off.

`mixing` is a ratio of two eigenvalues, and a ratio means nothing until the cloud
is bigger than the noise around it. Hence `min_stain`, at **3× the optical-density
noise floor step 4 measured on this slide's own glass** rather than a constant: the
artefact sits at 0.11 mean OD against 0.21 for a stained duct and a floor of 0.048,
so the multiple separates them cleanly while leaving every genuinely stained block
eligible. The rejected blocks still report the figure they were rejected on.

**Step 2's map has to be honoured for the whole block, not the scored part.** The
score is computed over clean pixels, but the *tile that gets read* covers the whole
block - so a block that is 98% tissue and a third artefact would be scored on its
good part and transformed including its bad part. Both `tissue_share` and
`considered_share` therefore have to clear the bar.

**The wedge is wider than the published pair on every tile of this slide** - 58 to
63 degrees against 37 for Ruifrok's haematoxylin and DAB. Three things widen a
wedge: a third absorber, extremes reaching into unreliable pixels (which the
stability test addresses and does not eliminate), and DAB itself, whose colour
shifts with concentration because it is a scattering precipitate rather than a
clean chromophore. Only the last is benign. The report says so rather than
presenting a wide wedge as a better result, and none of it changes what step 6
does - it measures with the published vectors either way.

**Resampling happens in intensity space, before the logarithm.** The demo's slide
is 40× at 0.2222 um/px with pyramid downsamples of 1, 4, 8, so nothing sits at
0.5 um/px and the tile is read at level 0 and area-averaged from 1,152 pixels to
512. A coarser sensor averages the light arriving over a larger area, so averaging
*transmissions* is what a coarser scan physically is. Averaging densities instead
takes the mean of logarithms - the logarithm of a geometric mean of transmission -
a different number, biased low, and one no instrument records.

**Nothing is cached on disk, and that is a decision.** Step 5's input is one 512 px
tile, a thousandth of the pixels steps 3 and 4 read, and one `read_region`. The
white point comes from `calibration_service.white_point()` and the candidate scores
from the RGB thumbnail that service already caches. A disk cache here would have to
be keyed on step 3's threshold, step 4's percentile and the tile position at once,
and would save less than invalidating it correctly would cost.

## What this step does not do

It does not repair anything. The transform is a **change of units with an exact
inverse**, and `measure_limits` performs that inverse and reports the residual: on
the demo's slide **100% of the tile returns to within half an intensity level**,
worst error 0.0000. That is the honest evidence that no pixel was enhanced,
smoothed or clipped on the way through.

Three places it stops being a measurement, all reported and none silently fixed:

| | On the demo's tile | What it means |
| --- | --- | --- |
| at the intensity floor | 0.33% | a channel recorded no light; the density is a *lower bound* |
| brighter than I0 | 0.86% | negative density, which cannot happen - I0 is slightly low here |
| below beta | 67% | a magnitude but no reliable direction; out of the cloud |

The negative share is partly structural and worth knowing: step 4 measures I0 at
2 um/px and this tile is read at 0.5, and a percentile of averaged pixels sits
lower than the same percentile of the pixels that were averaged. A *large* share
would mean the glass step 4 sampled is not representative of the light reaching
here.

It also does not rewrite anything. Step 6 un-mixes these densities, once, and
hands the haematoxylin channel to the model branch and the DAB channel to the
measurement branch.

## Settings

All in `app/core/config.py`. The resolution and tile size are step 1's -
`target_mpp` and `tile_size` - deliberately not step 5's own: the working
magnification is one decision for the whole pipeline.

| Setting | Default | What it is |
| --- | --- | --- |
| `density_min_tissue_share` | 0.85 | Share of a block that must be tissue, and clear of step 2's artefacts |
| `density_min_stain_multiple` | 3.0 | Multiple of step 4's noise floor a block's stain must clear |
| `density_beta` | 0.15 | Mean OD below which a pixel has no direction (Macenko) |
| `density_arm_percentile` | 1.0 | Which tail is an arm (Macenko's alpha) |
| `density_angular_tolerance_deg` | 1.5 | How far quantisation may move a direction before exclusion |
| `density_candidates` | 12 | Candidate tiles returned with the report |

## Papers

Primer items 74, 76 · [Ruifrok & Johnston
2001](https://pubmed.ncbi.nlm.nih.gov/11531144/) ·
[Macenko et al., ISBI 2009](https://ieeexplore.ieee.org/document/5193250) ·
Beer 1852
