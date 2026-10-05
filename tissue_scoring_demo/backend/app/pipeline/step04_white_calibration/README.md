# Step 04 - White calibration

`app/data/pipeline_steps.py` (`id="white-calibration"`) · implemented.

## Input

Binary tissue mask, as step 3 left it, plus the slide

## Output

Per-channel white point I0 - and the optical-density noise floor it implies

## Contents

- `calibration.py` - the algorithm. Read the module docstring first: it states
  why I0 cannot be a constant, why it cannot be a percentile of the whole image
  either, and why a percentile rather than a maximum or a mean.
- `overlay.py` - the four panels (thumbnail, glass, field, corrected) and the
  I0 swatch.
- `pipeline.py` - `run(context)`. Adapts `app.services.calibration_service`, and
  enforces that step 3 ran first.

`channel_percentile` and `optical_density` live in `app/common/imaging.py`, not
here, because step 5 is entirely the second of those and step 4 needs it to state
its own noise floor. One implementation means the floor this step quotes and the
density step 5 computes cannot disagree.

## What this step is for

Optical density is defined *relative to a reference*: `OD = -log10(I / I0)`. Step
5 cannot compute a density at all until something says what `I0` is. This step
says it, by measuring the one part of the slide known to hold no stain - the
empty glass beside the section.

The obvious alternative is `I0 = 255`, or one triple measured once from a
reference slide. Both are wrong the same way: **I0 is a property of an
acquisition, not of a stain.** A lamp dims as it ages, a white balance is set per
batch, mounting medium yellows, and a slide scanned twice on two machines comes
back with two different whites. All of that lands in a denominator, and none of
it is biology. Two slides with identical tissue would report two different
H-scores and the difference would be the lamp.

This is also the step that makes an absolute stain scale possible (primer items
79-80). Without it you are back on per-slide percentile normalisation, which -
see Rule 2 - removes the very differences being measured.

## The five decisions worth arguing about

**Glass is defined by exclusion, five times over.** The guide's version is
"invert the tissue mask and drop the outermost border". That is the first two.
The other three exist because the guide's version assumes the only thing on the
glass is glass, and on a real slide it is not:

| Exclusion | Why it is not glass |
| --- | --- |
| not tissue | step 3's mask, inverted - the basis of the whole step |
| not the outer frame | vignetting is worst there, and so are the coverslip and slide edges |
| not step 2's artefacts | pen ink sits *on* the glass, and is far darker than it |
| not the scanner's background fill | nothing was imaged there at all |
| not the seam around that fill | a downsample blends the two, so the blend is neither |
| not the halo around the tissue | mounting medium, section edge, mask boundary error |

Every one of the last four is *darker* than glass, so each biases I0 downward -
and because I0 is the denominator of `I / I0`, a low I0 *compresses* every optical
density downstream: faint stain and no stain measure closer together, and glass
brighter than its own reference measures negative. The same damage a clipped
channel does, self-inflicted. The ladder in the report shows what each exclusion
cost.

**The background fill was not anticipated and it dominates.** On the demo's
Aperio scans **68% of the frame is exactly `(146, 146, 146)`** - the part of the
slide canvas the scanner never imaged. It is darker than the real glass at 193,
so left in the sample it drags I0 down; left in the noise floor it reports
`log10(193/146) = 0.12` OD of apparent stain over a region that does not exist.
That is a third of a real DAB signal, on nothing.

It is told from real glass by the one property a digital constant cannot fake.
Real glass carries sensor noise, so colours sit either side of its commonest
colour; a constant has no such neighbours. Measured on the demo slide:

| Colour | Share of frame | Neighbour ratio | Verdict |
| --- | --- | --- | --- |
| `(146,146,146)` | 68.3% | **0.0017** | digital fill, excluded |
| `(193,193,193)` | 8.3% | **0.24** | real glass, kept |

A 145-fold gap, and the cutoff of 0.05 sits thirty times above the fill and five
times below the glass. On an uncompressed scan the glass scores 7 and up, so the
margin is wider still. Excluding the fill moved this slide's noise floor from
**0.121 OD to 0.048 OD**, and the *median* glass from 0.121 to **0.0007 OD** -
which is what empty glass should measure, and did not before.

**And the fill's own edge outlived the fill.** The exclusion above matches an
exact triple, which is what keeps it from ever swallowing real glass - and is why
its boundary survived it. These exclusions run on a *downsampled* thumbnail, and a
downsample averages whatever falls under one output pixel, so along the fill's
edge the result is part fill and part glass: a short ramp between 146 and 193, not
one pixel of which is the fill's colour.

That is a few pixels wide and it lands nowhere near I0, which is a *high*
percentile. It lands squarely in the noise floor, which is a low one. The tell was
in the numbers above and went unread for a while: a floor of 0.048 OD over a
median of 0.0007 is a factor of seventy between the typical glass pixel and the
dim tail, and clean glass does not do that.

Measured on `CAN_00270_26_H&E`, whose scan is tight enough that the glass is a
narrow band: 7% of the sampled glass sat at 174-178 against a body at 190-196, and
that 7% landed exactly on the 5th percentile. Those pixels are a median of 1.4 px
from the fill and 17 px from the tissue, so they are the fill's seam and not the
section's halo. Standing 60 um back from the fill's edge:

| Slide | Floor before | Floor after | I0 |
| --- | --- | --- | --- |
| `CAN_00270_26_H&E` | 0.039 | **0.018** | 196, unmoved |
| `CAN_00270_26_A` | 0.018 | **0.009** | 193, unmoved |
| `CAN_00270_26_F` | 0.026 | **0.002** | 193, unmoved |
| `CAN_00270_26_R` | 0.018 | **0.005** | 193, unmoved |

I0 does not move on any of them, which is the point: the seam was only ever in the
tail. What does move is every threshold downstream that is stated as a multiple of
the floor - step 5's tile gate first, which was rejecting the H&E slide outright
for want of 1.5% more stain than a contaminated floor was asking for.

The ring costs 48% of the *candidate* glass on that slide, and that is not a
warning sign: where a scan is tight the glass is a narrow band between the fill
and the section, so a ring from one side of it takes much of the band. 0.36 mm2 of
clean glass is a better sample than 0.50 mm2 with the seam in it. What would be a
warning sign is too few pixels left to read a percentile from, so that is what the
guard tests - a count, not a share.

**A percentile, not the maximum and not the mean.** I0 sits in a denominator, so
an overestimate darkens the whole slide and an underestimate can drive densities
negative. The maximum is set by one hot pixel or one glint off the coverslip; the
mean is dragged down by whatever leaked past step 3's mask. The report carries the
whole percentile ladder, because the argument for the 95th is that its neighbours
agree with it - on clean glass the value is flat from the 90th to the 99th, and
where it climbs the sample is contaminated. That spread is reported as `plateau`,
and 3% is the line: the demo's slides sit at 1% clean and at 32% with the
background fill left in.

**Flat or varying, decided from the data.** The guide mentions the robust variant
in one line - fit a smooth surface to also correct uneven illumination. Both are
always computed and always reported; which is used turns on two tests, in the
same shape as step 3's threshold-rule choice:

- **swing over residual** (`snr`, default 4). The fitted field must bend across
  the slide by at least four times as much as it typically misses a patch sample
  by. Below that the bend *is* the scatter, and a white point that curves to
  follow noise is worse than a flat one that does not - it stamps a spurious
  spatial gradient onto every measurement, where nothing later can tell it from
  biology. The two errors are not symmetric, which is why the bar is shy.
- **extrapolation leverage over the tissue** (default 1.0). Glass samples
  necessarily form a *ring*, because the middle of the slide is the section.
  Usually the ring encloses the tissue and the field is pinned down across it;
  when the section runs off the edge of the scan the ring is open and the fit is
  guessing across exactly the part every measurement comes from.

  Leverage is measured over the **tissue**, not the frame, and that is the point.
  A ring fit is wildly unconstrained at the frame's corners - on the demo slide
  it reaches 9.6 there - and it does not matter at all, because no density is
  ever computed in a corner. Over the tissue the same fit scores 0.88, so the
  ring does enclose it. Judging over the frame would reject a sound field for
  being uncertain about a place nobody reads.

**Vignetting is reported in optical density, not as a percentage.** "6% of I0" is
not a number anyone can act on in a pipeline whose output is a density. `od_error`
converts it: a swing of a fraction *f* costs `log10(1/(1-f))` OD. On the demo
slide 1.6% is **0.007 OD** - nothing beside a DAB signal of 0.3, and worth
knowing beside the gap between a 1+ and a 2+ call. Stating the percentage alone
leaves the reader unable to tell which case they are in.

## Measured limits, recorded so nobody re-derives them

**The surface removes most of a vignette, not all of it.** On a synthetic bowl of
known depth the fit recovers the swing to within 3% and, applied, removes about
80% of the centre-to-corner difference. What is left is a systematic bias: each
patch's value is the 95th percentile of the glass *inside that patch*, which sits
above the field at the patch centre, and by more where the gradient is steeper.
So the fitted surface is "the local 95th percentile of glass as a function of
position" - the spatially-varying counterpart of the flat white point, and
consistently defined with it. The residual is about 2% of the level, or 0.01 OD.
Smaller patches reduce it and make each percentile noisier; 1000 um was the
better trade.

**Two of the demo's slides are the same file.** `compare` between them correctly
reports a 0.000 OD shift. The endpoint's argument needs two genuinely different
acquisitions to make its point on screen.

## What this step does not do

It does not rewrite a single pixel, and that is the whole of Rule 2. Calibration
*records* what zero stain means; rescaling a slide's colours to match a reference
slide *rewrites* them. The first preserves the measurement and makes it
comparable across slides; the second destroys the quantity being measured. This
pipeline does the first and never the second - the model branch is handed step
6's haematoxylin channel instead, which removes the colour nuisance without
touching what is measured.

It also does not repair a clipped white. If a channel is pinned at 255 the sensor
ran out of range before the glass did, so I0 is an underestimate and every density
is compressed - faint stain and no stain measure closer together than they should.
That is a scanning fault; the report says so and no arithmetic here recovers it.

## Settings

All in `app/core/config.py`. Every distance is physical, so none changes meaning
on a different scanner.

| Setting | Default | What it is |
| --- | --- | --- |
| `calibration_mpp` | 2.0 | Resolution asked for; step 3's grid is used, so its cap is the one that binds |
| `calibration_percentile` | 95.0 | Which percentile of the glass is I0 |
| `calibration_border_um` | 500 | Outer frame discarded |
| `calibration_tissue_clearance_um` | 60 | Ring outside the tissue discarded |
| `calibration_fill_min_share` | 0.05 | Share a colour must hold to be tested as a fill |
| `calibration_fill_max_shoulder` | 0.05 | Neighbour ratio at or below which it is a digital constant |
| `calibration_fill_clearance_um` | 60 | Ring inside the fill's edge discarded - its seam with the glass |
| `calibration_patch_um` | 1000 | Side of one sampling patch |
| `calibration_patch_min_glass` | 0.35 | Share of a patch that must be glass to be sampled |
| `calibration_min_patches` | 12 | Fewest usable patches before a surface is permitted |
| `calibration_vignette_snr` | 4.0 | Swing over residual a surface must clear |
| `calibration_max_leverage` | 1.0 | Extrapolation leverage over the tissue a surface must stay under |
| `calibration_od_floor` | 1.0 | Intensity floor, so black is a finite density |

## Papers

Primer items 76, 79-80 · [Ruifrok & Johnston
2001](https://pubmed.ncbi.nlm.nih.gov/11531144/) · Beer 1852
