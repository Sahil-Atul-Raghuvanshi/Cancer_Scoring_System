# Step 11 - Nuclei segmentation

## Input

The three invasive regions step 10 carried onto the IHC slide, in that slide's
level-0 pixels, plus the slide itself.

## Output

Every nucleus in a sample of fields inside each region: an outline, a centroid,
an area, a perimeter, a circularity, an eccentricity, and how much haematoxylin
it holds. Plus the density per mm2 that the next three steps' denominator rests
on.

## Changed 6 October 2026 (P-03): Cellpose, and density per mm2 of tissue

Two of the decisions below were reversed on measurement. The full benchmark - 240
production fields from 10 pairs across all five markers, plus labelled IHC from two
public sets - is in `p03_nuclei/` and `storage/v1_data/data/p03_nuclei/REPORT.md`.

* **The detector is Cellpose's published `nuclei` model, zero-shot**
  (`settings.nuclei_engine = "cellpose"`, `app/nuclei/cellpose_model.py`), reading
  each field as inverted grey. InstanSeg on the DAB-removed haematoxylin render -
  decision 2 below - missed 58% of the H&E's nuclei per mm2 of tissue, with 1 of 10
  pairs within 20%; Cellpose missed 20%, 6 of 10 within 20%. On breast IHC cells
  labelled from immunofluorescence it scored F1 0.77 against 0.70. Fine-tuning it on
  public IHC (lymphoma or breast) made our slides worse, as did other inputs.
  InstanSeg stays available as `"instanseg"` and still counts the H&E reference.
* **Decision 2's rule no longer holds by construction.** Cellpose sees the brown.
  Whether stained cells are found more readily is now answered by measurement: the
  H&E check runs on every pair, and on the breast set Cellpose found 109% of the
  labelled cells, stained and unstained.
* **The H&E check is per mm2 of tissue** on both sides (decision 3). Two of the
  benchmark's worst pairs had fields on bare glass and on the scanner-fill rectangle,
  and a per-field density read both as ~95% missing nuclei. Tissue = optical-density
  sum over 0.25 **and** 9x9 grey variation over 1 (fill and glass are perfectly
  flat), measured over each field's counted interior. `densityShortfall` is now this
  figure; the old one is `areaShortfall`.
* **Not solved:** on heavily stained sections tumour nuclei show only as pale holes
  in the brown, with no counterstain, and no detector tested finds them. That is P-22.
* Cellpose installs with `--no-deps` (`requirements-nodeps.txt`): its metadata pins
  numpy<2.1, but it reproduces the benchmark byte for byte on this venv's numpy 2.5,
  and it is parity-checked at every load. Its checkpoint must keep a name ending in
  `nucleitorch_0`, or Cellpose loads it at base diameter 30 instead of 17 and every
  field comes out different.

## The three decisions that shape this step

### 1. It samples. It does not exhaust.

The three regions total about 26 mm2. Measured on this machine, InstanSeg runs
at roughly 60 s/mm2 on four CPU threads, so segmenting them whole is over half an
hour. Twelve 256 um fields per region is about two minutes and answers the
question the screen actually asks.

So **every count here is an estimate from a stated sample**, and the report
prints the sampled area beside every number derived from it - `12 of 361 fields,
0.51 of 22.07 mm2`. A census would be a different claim and this step does not
make it.

The fields are taken at a **uniform stride** through the candidates, not ranked.
Ranking by stain content would pick the densest fields and report their density
as the region's, which would destroy the one free QC check this step has (see 3).

### 2. It detects on the counterstain, with the DAB removed. *(InstanSeg path only since 6 October 2026 - see above.)*

The guide is emphatic and it is right. On an IHC slide the brown is what is being
measured, so it must not decide where cells are thought to be - otherwise a
strongly stained cell becomes a more findable cell and the percentage inflates
itself through a route no later step can see or correct.

Implemented as: un-mix the field, take the haematoxylin amount, and **draw it
back** as a blue-on-white brightfield image. Not as a replicated density map -
InstanSeg's declared input is three-channel brightfield RGB, and a stack of
identical density planes is out of distribution in a way nobody has measured.

The screen keeps the other path available and shows both counts, because the size
of the difference is the argument for the rule.

### 3. Density per mm2 is a QC metric, not a statistic.

The six slides of a case are serial sections of one block, so their nuclei per
mm2 should land in the same neighbourhood. A marker 30% below its siblings is a
segmentation failure rather than biology. The report carries every other marker
of the same case that has already been segmented, keyed by letter, so the check
costs nothing.

This is also why the border band exists. A nucleus whose centroid falls within
24 px of a field edge was seen truncated, so it is drawn but not counted - and
the *area* it sat in is removed from the denominator with it. Counting over the
whole field while dropping its edge objects would undercount density by about
18% at the shipped geometry, which is far larger than the difference this check
is trying to detect.

## Two measurements that reverse the guide's advice

Both were made on `CAN_00270`'s CD44 slide over six fields, and both are recorded
here because the guide says otherwise and a reader should see why.

### Per-slide Macenko is *worse* here, not better

| basis | nuclei found |
| --- | --- |
| **Ruifrok, fixed** | **695** |
| Macenko, per slide | 581 |
| hybrid: DAB from Macenko, H from Ruifrok | 574 |

The guide recommends the per-slide estimate precisely because a fixed matrix is
at its worst when one stain dominates - which is this panel. The estimated
vectors say what actually happens. Macenko recovers **DAB** well, 3.3 degrees
from the published direction, because DAB is what dominates the cloud. Its second
arm comes back **achromatic** - (0.577, 0.577, 0.577), 18.7 degrees off
haematoxylin - because with a weak counterstain under heavy DAB there is no
second *colour* in the cloud to find, so the method returns the darkness axis.

The condition that is supposed to justify estimating is the condition that breaks
the estimate of the stain we need. So the default is the fixed basis, and the
Macenko estimate is still computed and reported, because its drift is the
evidence. A marker with a stronger counterstain may well flip this back.

### 0.5 um/px, not 40x

The guide says to zoom to 40x. The *viewer* does. The model does not: its own
metadata declares an input scale of 0.5 um/px (about 20x), and feeding it 0.25
would be out of distribution. Segmenting and looking are different resolutions
for different purposes, and conflating them would degrade the segmentation to
make a caption true.

## Why InstanSeg, and why there is no new dependency *(the detector until 6 October 2026; still the H&E reference)*

Apache-2.0, **and so is its training data** - tnbc_2018, lynsec, nuinsseg and
ihc_tma are CC BY 4.0 and consep is Apache-2.0. Two of those five are IHC rather
than H&E, which is the real evidence behind "it already handles IHC". HoVer-Net
would be the nicer model and cannot ship: PanNuke is CC BY-NC-SA.

`instanseg-torch` itself is **not installed and is not needed**. It declares
Python 3.9-3.11 and publishes no 3.13 wheel, and this backend is 3.13 on numpy
2.x - the same collision that pushed VALIS into its own interpreter. But the
published model is a TorchScript archive with its post-processing compiled in, so
`torch.jit.load` runs it against the torch already installed for step 2.

**The model is proved on load, not trusted.** The release ships its own test pair
and `app/nuclei/model.py` requires an exact reproduction of it before serving
anything. That is not ceremony: fed the raw 0-255 test input without the
`rdf.yaml` percentile stretch, this model returns **zero** nuclei - silently, no
error. A step whose job is to produce a denominator cannot afford a failure mode
that reads as "this tissue has no cells".

## The comparison panels

Two, and both are arguments rather than decoration.

* **InstanSeg against a naive watershed**, same field, same input. The watershed
  is a fair implementation - Otsu, distance transform, seeded flood - and its
  failures are the point: it merges nuclei sitting shoulder to shoulder and
  shatters one large irregular nucleus into two. Every merge removes a cell from
  the denominator.
* **Haematoxylin against raw RGB**, same field, same model. This is decision 2
  made visible, with both counts on screen.

## Status

Built. Runs on CPU with no GPU and no new Python dependency. Refuses to start
until step 10's alignment is `ready` **and confirmed by a person** - step 10
declines to approve its own result, and this is the first step downstream, so
this is where that stops being advice.
