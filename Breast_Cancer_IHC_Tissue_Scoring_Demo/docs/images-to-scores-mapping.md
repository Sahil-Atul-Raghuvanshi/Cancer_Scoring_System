# From slide images to scores: what we have, what we must compute, and where it lands

**Plain-English companion to** [demo-pipeline-guide.md](demo-pipeline-guide.md) (the *how*) and
[breast-cancer-pathology-primer.md](breast-cancer-pathology-primer.md) (the *what things mean*).

**What this document is for.** Three questions get asked repeatedly and are answered in three
different places today:

1. Which image file belongs to which antibody, and what is actually inside each file?
2. Which numbers are we supposed to produce from each image, and how are they computed?
3. Where do those numbers land in `6Slide Reports2.xlsx`, and how does a risk score come out?

This document answers all three in one place, in ordinary language, with the real numbers from
the six cases we have on disk.

---

## Part 1 — The end goal, in one paragraph

A patient has breast cancer. The surgeon removed the tumour, and now someone has to decide:
**is chemotherapy worth it for this person, or would it be harm without benefit?** OncoStem's
CanAssist Breast test answers that. It takes one block of the tumour, cuts six thin slices from
it, stains five of them brown — each brown revealing one specific protein — reads how dark and
how widespread each brown is, and feeds those five readings plus a few clinical facts into a
prediction model that outputs **low risk** or **high risk** of the cancer coming back.

Today four pathologists read those five slides by eye, per case, and their numbers are averaged.
**Our system's job is to replace the eye with the camera** — to produce, automatically and
reproducibly, the same five pairs of numbers that a pathologist would write down, so they can be
fed to OncoStem's existing prediction algorithm.

```
  6 slide images  →  our pipeline  →  5 pairs of numbers  →  OncoStem's model  →  low / high risk
  (what we have)     (what we build)   (the deliverable)      (their black box)     (the answer)
```

**The single most important thing to understand about scope:** our deliverable is the *five pairs
of numbers*, not the risk score. The risk model is OncoStem's proprietary property and they have
been explicit that intermediate values stay compartmentalised — only they see inside it. Part 7
covers exactly what that means for us.

---

## Part 2 — What is on disk

Location: `Breast_Cancer_IHC_Tissue_Scoring_Demo/images/`

**36 files, 6 cases × 6 slides each, about 35 GB total.** Every file is a `.svs` whole-slide
image — a single huge scanned photograph of one glass slide, stored as a pyramid of
progressively smaller copies so software can open a zoomed-out view without loading the
whole thing.

| Case | Block ID (from the Excel) | Slides present |
| --- | --- | --- |
| `CAN_00251_26` | H26-364-B1 | H&E, A, F, R, U, W |
| `CAN_00259_26` | H2600058-A1 | H&E, A, F, R, U, W |
| `CAN_00267_26` | 25H - 19421-A2 | H&E, A, F, R, U, W |
| `CAN_00270_26` | H26-154-C6 | H&E, A, F, R, U, W |
| `CAN_00303_26` | B2011-T | H&E, A, F, R, U, W |
| `CAN_00865_26` | 2460/GL-A8 | H&E, A, F, R, U, W |

All six sets are complete. This is a change from the first batch (`AlgoDev1.zip`), where
`CAN_00865_26` was missing `F`, `U` and its H&E — if you are reading older analysis notes that
say the set is incomplete, that note is now out of date.

### The filename trap

Five cases use underscores. `CAN_00865_26` uses hyphens, and its H&E has *both*:

```
CAN_00251_26_A.svs          CAN_00865_26-A.svs
CAN_00251_26_H&E.svs        CAN_00865_26-_H&E.svs      <- note the "-_"
```

Any ingestion script must accept `_` or `-` as the separator before the letter code, and must
survive the literal `&` in `H&E`. Parse with a tolerant regex, not by splitting on `_`.

---

## Part 3 — The image ↔ antibody mapping

This is the core lookup. The single trailing letter in the filename **is** the antibody.

| Letter | Antibody / marker | What it is, in plain terms | Where in the cell the brown must be | Expected % range |
| --- | --- | --- | --- | --- |
| **A** | **CD44** | A stem-cell-like behaviour marker. Tumours rich in it tend to behave more aggressively | **Membrane** — the cell's outer skin | 0–85 % |
| **F** | **ABCC4** | A pump that pushes drugs out of the cell — relevant to whether chemo works | **Membrane** | 35–70 % |
| **R** | **ABCC11** | A related pump, also called MRP8 | **Membrane** | 30–70 % |
| **U** | **N-cadherin** (CDH2) | A cell-to-cell glue molecule; changes when cells start to spread | **Cytoplasm** — the body of the cell | 70–85 % |
| **W** | **Pan-cadherin** | Overall adhesion molecules, all cadherins together | **Cytoplasm** | 70–85 % |
| **H&E** | *none* | The plain pink-and-purple structural stain. No antibody, no brown | n/a — used to find the tumour, not to score | n/a |

Confirmed three ways: OncoStem's CAB Technical Approach deck slide 5, the 28 July call, and the
physical slide labels themselves (slide A's label reads `Ab-A`, slide U's reads `Ab-U`). It also
matches `backend/app/panel.py` in the client repo exactly.

**Two corrections worth knowing**, because both were wrong in earlier drafts of our own code:

- N-cadherin and pan-cadherin are **cytoplasmic, not membranous**. They take a completely
  different code path from the three membrane markers, because the brown sits somewhere else.
- The expected ranges are **descriptive, not thresholds**. They are the spread OncoStem has seen
  over years of cases. A result outside the range is a flag for human review, not an error and
  not a positive/negative call. There is *no* positivity cut-off at the scoring stage.

### What a single image actually contains

Every one of the 36 files is one slide, scanned at 40×, **0.2222 µm per pixel**, on a Morphle
scanner writing an Aperio-compatible SVS. Inside a brown (IHC) slide you will see:

- **Blue-purple nuclei** everywhere. That is the counterstain (haematoxylin), applied to every
  slide so you can see the tissue behind the brown. It is not the signal.
- **Brown, in varying darkness and patchiness.** That *is* the signal — one protein, this slide only.
- **Tissue that is not tumour**: fat, stroma (the supporting scaffolding), normal milk ducts,
  DCIS (cancer still trapped inside a duct), and sometimes necrosis (dead tissue). None of it
  should be scored. Sorting this out is the hardest part of the whole system.
- **Normal ducts as a built-in reference.** Pathologists judge "how brown is brown" against the
  normal ducts *on the same slide*. This matters enormously — see Part 8, question Q7.

Three structural quirks of these particular files that will bite you:

1. **The canvas is padded, not cropped.** The IHC slides are 63488 × 63488 pixels — 14 × 14 mm —
   but the tissue occupies roughly one quadrant. The rest is flat grey (about RGB 150,150,150)
   where the scanner wrote nothing. Any statistic computed over the whole canvas is mostly
   measuring padding.
2. **The zoom pyramid skips 2×.** Levels are 1, 4, 8, 16, 32, 64, 128. Code that assumes a
   standard 1/4/16/64 ladder picks the wrong level and silently scores at the wrong resolution.
3. **The "thumbnail" is not a thumbnail.** It is 31744 × 31744 — a half-resolution copy of the
   entire slide. Decoding it costs about 8 GB of RAM and has already killed one probe run. Never
   decode associated images blindly.

### The H&E is a different section — you cannot copy regions across

The H&E has a different canvas shape (63488 × 31744) and is a physically different slice of
tissue, cut seconds later. **The cells on the H&E are not the same cells as on the CD44 slide.**
There is no shared coordinate frame. So a tumour outline drawn on the H&E cannot simply be
pasted onto the IHC slides.

This is exactly what pathologists deal with, and their answer is to re-identify the tumour by
eye on each slide, using the normal ducts as landmarks. Our system does the same: **find the
region on each IHC slide directly** rather than aligning slides to each other. Registration error
at a region boundary turns straight into scoring error, so we avoid registration on the main path.

---

## Part 4 — What scores must be calculated

### The contract: two numbers per marker, five markers, one case

For each of the five brown slides, produce exactly two numbers:

| # | Number | Meaning in plain terms | Scale | Step size seen in real data |
| --- | --- | --- | --- | --- |
| 1 | **Percent positive** | Out of all the viable invasive tumour cells, what share show brown in the right compartment? | 0–100 % | multiples of **5** |
| 2 | **Intensity** | How dark is that brown, on average, across the whole slide? | **0–2** | 0, 0.5, 1, 1.5, 1.75, 2 |

That is the whole deliverable: **ten numbers per case.** Nothing else crosses the boundary to
OncoStem.

### Important: OncoStem does not want an H-score

The pipeline guide teaches H-score, Allred and the HER2 rules because they are the standard
vocabulary of this field, and you need them to understand what percent-and-intensity even means.
But **OncoStem's actual report format keeps percent and intensity separate** — look at
`6Slide Reports2.xlsx`: ten columns, five percent-and-intensity pairs, no combined score anywhere.

So: compute H-score if you like, for the demo screens and for internal sanity checks. It is a
useful one-number summary and it is `1×(% at 1+) + 2×(% at 2+) + 3×(% at 3+)`, range 0–300. But do
not present it as the deliverable, and never let it replace the pair. **Collapsing two numbers
into one throws away information OncoStem's model is using.**

### The intensity scale conflict — read this before implementing

OncoStem's own material describes intensity two incompatible ways:

- Deck pages 6 and 7: intensity runs **1 to 3** (1 lightest, 3 darkest).
- Deck page 9: reporting bands are **Negative 0, Weak 0.5–1.0, Moderate 1.5, Strong 1.75–2.0**.

These are different scales — one tops out at 3, the other at 2, and an average of numbers
between 1 and 3 can never be 0.

**All 120 intensity readings in the Excel sit on the 0–2 scale.** So we implement 0–2 and treat
the 1–3 scale as describing something else, probably a per-field observation before averaging.
This is a live open question (Q2, Part 8) — flag it, do not bury it.

One more measured observation: **119 of the 120 readings land exactly on a band value** (0.5, 1,
1.5, 1.75, 2). That is what you would see if a pathologist picks *one band per slide* rather than
computing a continuous average. The single exception is reader SA on case `CAN_00270_26`,
pan-cadherin, recorded as **1.25** — a value no band permits. Our software naturally produces a
continuous number, so we will need a banding step to match how they actually report.

---

## Part 5 — How to calculate them

### The one idea underneath everything

Every pathology score is a counting problem wrapped in three filtering problems:

```
score = (how many of the RIGHT cells are stained) / (how many of the RIGHT cells there are)
```

Each phrase in that sentence is a problem the pipeline has to solve:

| The phrase | The question it raises | What solves it |
| --- | --- | --- |
| "the RIGHT cells" | Which part of the slide counts at all? | Find the invasive tumour region |
| "cells" | Where does one cell end and the next begin? | Nuclei segmentation |
| "stained" | How brown is brown enough? | Stain separation + calibration + thresholding |
| "how many" | plain arithmetic | Aggregation rules |

Get the *order* wrong and you compute a beautifully precise number about the wrong population of
cells. That is the most common failure in this field.

### The funnel: big and dumb → small and smart

```
Whole slide (4 billion pixels, mostly grey padding and glass)
   |  throw away glass and padding                     <- cheap, classical
Tissue only
   |  throw away unusable tissue (blur, folds, pen)    <- cheap, pre-trained QC model
Usable tissue
   |  throw away the wrong TYPES of tissue             <- expensive, must be learned
   |  (fat, stroma, DCIS, normal duct, necrosis)
Invasive tumour only          <- everything from here on is "the RIGHT cells"
   |  outline every individual cell                    <- expensive, pre-trained model
Cells
   |  measure brown in each cell's correct compartment <- cheap, colour arithmetic
Per-cell measurements
   |  apply OncoStem's rulebook                        <- cheap, pure logic
Percent + intensity
```

**Every expensive step runs on a smaller area than the step before it.** That is the whole reason
for the order, not an accident. Nuclei segmentation on a full slide is a GPU-day; on a small
invasive focus it is seconds.

### The steps, in order, with the arithmetic

**Steps 1–5, preparation (all classical, no models).** Open the slide and pick a zoom level.
Reject blurry, folded or pen-marked areas. Separate tissue from glass and padding. Sample the
empty glass to learn what "no stain" looks like *on this slide* — this is **calibration**, and it
is what makes numbers comparable between slides. Convert colour to optical density (how much
stain is present) via Beer–Lambert.

**Step 6/7, the fork — the single most important design decision.** Two things are both called
"normalisation" and confusing them destroys your numbers:

| | **Stain normalisation** | **Stain calibration** |
| --- | --- | --- |
| What it does | Rewrites colours to match a reference slide | Records what "zero stain" means, in physical units |
| Feed it to | Deep learning models only | The measurement path |
| Effect on your numbers | **Destroys them** — you overwrote the intensity you wanted | Preserves them, makes them comparable |

If you rescale every slide so its darkest brown becomes "maximum brown", a genuinely weak slide
and a genuinely strong slide both come out looking strong. **You have normalised away the
diagnosis.** So: the region-finding model gets normalised pixels; the measurement path gets raw,
calibrated pixels. Same tile, two branches, no shared preprocessing.

The measurement branch instead does **colour deconvolution** — un-mixing the image into a blue
(nuclei) channel and a brown (DAB) channel. You must un-mix *before* you threshold. A dark blue
nucleus under faint brown looks, in raw RGB, a lot like a moderately brown nucleus; you cannot
threshold a mixture.

**Steps 8–10, find the invasive tumour.** Cut the slide into tiles, classify each one
(tumour / DCIS / stroma / fat / necrosis / normal duct), and stitch the `tumour` tiles into one
clean region. **This is the only step in the whole pipeline that requires training a model from
scratch, and it is the hardest and riskiest part of the project.**

Two rules that are easy to get wrong here:

- **Remove fat here, not earlier.** Fat looks white and empty, so it is tempting to threshold it
  out during tissue detection. But fat *is* tissue, and if you drop it there you also drop pale
  tumour and washed-out areas, with no record of what you dropped. Make `fat` an explicit output
  class so you can show a pathologist exactly what was excluded and why.
- **`DCIS` and `tumour` must be separate output classes.** DCIS is cancer still trapped inside a
  duct; invasive carcinoma has broken out. Both look like dense sheets of abnormal nuclei, and a
  density-based rule cannot tell them apart. But the score is *defined on the invasive component*
  — including DCIS silently changes the answer. OncoStem stressed this repeatedly on the call.

**Steps 11–13, find the cells and their compartments.** Outline every nucleus *inside the region
only*, using an off-the-shelf model (HoVer-Net, StarDist, Cellpose — no training needed). Keep the
tumour cells and drop lymphocytes and stromal cells. Then build the compartment that matters for
each marker:

- **Membrane markers (A, R, F)** — grow a ring 3–5 µm outward from each nucleus, subtract the
  nucleus, and subtract any neighbouring nucleus so two adjacent cells never claim the same
  pixels. In crowded tissue, expand outward but stop at the midline between neighbours, so a
  strongly stained cell cannot bleed its signal into a negative neighbour.
- **Cytoplasmic markers (U, W)** — the band of cell body between nucleus and outer edge.

Measuring CD44 as "average brown inside the nucleus" produces a number that is not so much wrong
as *meaningless*. The marker is not there.

**Step 14, measure each cell.** One number per cell: the mean brown optical density in that
cell's correct compartment, taken from the **un-normalised, calibrated** channel. For the three
membrane markers, a second number too: **how complete the ring is.** Walk around the ring in 36
angular bins of 10°, mark each bin stained or not, report the fraction. A cell with 34/36 bins
stained has genuine complete membrane staining; a cell with 6/36 has non-specific speckle that
only looks positive if you average it away.

Completeness is not decoration. OncoStem's deck explicitly distinguishes **complete** membrane
staining (entire circumference) from **partial** (only part). It is what separates a real membrane
marker from brown noise. How partial staining should enter the percentage is unanswered — Q1,
Part 8, and it is the open question with the largest effect on our numbers.

**Step 15, bin the intensity.** Turn the continuous number into bands. *Where the thresholds come
from is the entire question:*

| Approach | Thresholds from | Comparable across slides? |
| --- | --- | --- |
| Per-slide percentiles | this slide's own 1st/99th percentile | **No** — 37 % here ≠ 37 % there |
| **Absolute calibrated OD** | fixed cutoffs on the calibrated scale | **Yes** |
| Control-slide anchored | cutoffs set once on known 0/1+/2+/3+ controls | **Yes**, and clinically strongest |

Use absolute cutoffs, anchored on control slides if we can get them. The client repo already made
this transition in EPIC-010, moving *away* from per-slide percentiles, for exactly this reason.

**Step 16, aggregate.** No image processing at all here — arithmetic and `if` statements, in a
small, heavily tested, separately reviewable module. Do not let scoring logic reach back into
pixels.

```
percent positive  = 100 x (positive tumour cells) / (total tumour cells)
                    then round to the nearest 5

intensity         = mean brown OD over the stained tumour cells
                    then map to the nearest band: 0 / 0.5 / 1 / 1.5 / 1.75 / 2
```

Then reconcile with how pathologists actually work: they divide the tumour into roughly three or
four parts, assess each, and average across them — more finely when staining is patchy, skipped
entirely when it is uniform. Our software naturally produces an **area-weighted** average, which
differs from a plain average when the parts are unequal. Which one OncoStem uses is Q3, Part 8.

**Step 17, validate.** Covered in Part 9.

---

## Part 6 — Where the numbers land: decoding `6Slide Reports2.xlsx`

### Row structure

Each row is **one reader's complete read of one case**. The row label packs three things:

```
PS_CAN/00251/26_H26-364-B1
|   |             |
|   |             +- block ID — should match the slide label in the image file
|   +- case ID — matches the image filename, but with / instead of _
+- reader initials: PS, SA, DP, NK — four pathologists
```

Every case has **five rows: four readers plus an `AVG_` row.** 6 cases × 4 readers = 24 reads ×
10 values = **240 numbers**, which is the whole dataset we have to calibrate against.

Note the case ID punctuation is inconsistent here too (`CAN/00251/26` on some rows,
`CAN00259/26` on others). Normalise on both sides before joining to filenames.

### Column decoding

**The columns are not in panel order.** They run A, W, U, R, F — not A, F, R, U, W. Mis-joining
here silently swaps ABCC4 with ABCC11, and both are plausible pump markers with overlapping
ranges, so the error would not announce itself. Be careful.

| Col | Header | Marker | Number | Compartment |
| --- | --- | --- | --- | --- |
| A | `ID` | — | reader + case + block | — |
| B | `A%-M` | **CD44** | percent positive | membrane |
| C | `AI-C` | **CD44** | intensity | " |
| D | `W%-C` | **pan-cadherin** | percent positive | cytoplasm |
| E | `WI-C` | **pan-cadherin** | intensity | " |
| F | `U%-C` | **N-cadherin** | percent positive | cytoplasm |
| G | `UI-C` | **N-cadherin** | intensity | " |
| H | `R%-M` | **ABCC11** | percent positive | membrane |
| I | `RI-C` | **ABCC11** | intensity | " |
| J | `F%-M` | **ABCC4** | percent positive | membrane |
| K | `FI-C` | **ABCC4** | intensity | " |

Reading the header: the letter is the marker, `%` vs `I` is percent vs intensity. On the percent
columns the suffix is the compartment and it matches the panel exactly — `M` for the three
membrane markers, `C` for the two cytoplasmic ones. **On the intensity columns the suffix is
always `-C`, including for the membrane markers**, so it cannot mean compartment there. It most
likely abbreviates something like "complete", or is simply a labelling artefact. Worth confirming
with OncoStem, but it does not change the mapping — the leading letter and the `%`/`I` are
unambiguous.

This mapping is independently confirmed by the data itself: every column's values fall inside
that marker's expected range from the deck, and the odd 1.25 intensity in column E lands on
exactly the reader and case that OncoStem's own SOP notes as the pan-cadherin exception.

### The `AVG_` row is a plain arithmetic mean

Verified on all 60 averages. No weighting, no trimming, no dropping of outliers:

```
Case CAN_00251_26, CD44 percent (column B):
    PS 60,  SA 50,  DP 50,  NK 40   ->  (60+50+50+40)/4 = 50.00   OK, matches AVG row

Case CAN_00251_26, CD44 intensity (column C):
    PS 1.5, SA 1.0, DP 1.5, NK 1.0  ->  (1.5+1+1.5+1)/4 = 1.25    OK, matches AVG row
```

Some averages are rounded to two decimals in the sheet (`1.94` is really 1.9375, `1.13` is
1.125, `1.81` is 1.8125). Do not read those roundings as data.

**This average is what feeds OncoStem's prediction algorithm** — not any individual reader's
numbers. So it is also what our system is being measured against.

### The ± 10 rule

A reading more than 10 from the group average is referred back to that reader to re-examine. From
the data, that 10 is **absolute percentage points, not a relative 10 %**: no reading in the 120
percent values exceeds 10 points from its case mean, and four sit at exactly 10.00 — for example
CD44 on `CAN_00251_26`, where 60 and 40 both sit exactly 10 from the mean of 50.

The rule appears **not** to apply to intensity: reader disagreement on intensity reaches 0.625 on
a 0–2 scale, which no 10 % rule would tolerate. That is Q4, Part 8.

**This gives us our accuracy target.** If four trained pathologists agree within ±10 points, then
our system landing within ±10 of their consensus puts it inside the human spread. That is the
result to aim for and the honest way to report it.

### What the 240 numbers actually look like

| Marker | Mean of 24 reads | Min | Max | Case averages, low → high |
| --- | --- | --- | --- | --- |
| **A** CD44 | 48.8 % | 5 | 85 | 11.25, 20, 50, 60, 71.25, 80 |
| **F** ABCC4 | 49.6 % | 35 | 65 | 38.75, 42.5, 45, 48.75, 60, 62.5 |
| **R** ABCC11 | 48.8 % | 35 | 65 | 35, 37.5, 50, 53.75, 55, 61.25 |
| **U** N-cadherin | 80.0 % | 80 | 80 | 80, 80, 80, 80, 80, 80 |
| **W** pan-cadherin | 79.6 % | 75 | 80 | 77.5, 80, 80, 80, 80, 80 |

Two things jump out, and both shape how we should build and demo this:

**CD44 is the marker that separates cases.** It ranges from 11 % to 80 % across the six cases —
a genuine seven-fold spread. If our system gets CD44 right, we have shown the thing that matters.
Make CD44 the demo's hero marker.

**N-cadherin and pan-cadherin are nearly constant.** N-cadherin was recorded as exactly 80 % by
all four pathologists on all six cases; pan-cadherin as 80 % in 22 of 24 reads. Either these two
markers genuinely behave this way in almost every breast tumour, or 80 % is a working convention
where the precise figure matters less. **We need to know which before we tune against them** — a
model that "achieves 100 % agreement" on a constant has demonstrated nothing. That is Q6, Part 8.
Neither answer is a criticism; we simply need to know what we are looking at.

---

## Part 7 — How the risk score is calculated

### The honest answer: we do not calculate it, and that is by design

The risk model is **OncoStem's proprietary intellectual property**. They were explicit on the
28 July call: the per-biomarker numbers flow into *their* existing prediction algorithm, which
converts them into the risk score and generates the patient report. The pipeline must be
compartmentalised — intermediate values stay inaccessible, and only OncoStem has visibility into
the model. We hand over ten numbers per case. What happens next is theirs.

Coditas confirmed on that call that **the automation and reporting layer is not the hard part —
the core challenge is the scoring.** That framing is correct and worth holding on to.

### What their report contains

From the CanAssist Breast report Dr. Savitha showed on the call: patient demographics, sample
details, the **CanAssist risk score** with a low-risk vs high-risk-of-recurrence category and
where this patient falls on the scale, a **graph of probability of distant recurrence** under
prescribed local treatment plus radiotherapy, representative images (the H&E plus the five
biomarkers), tumour content examined, fixation comments, and signatures.

So our ten numbers are not the only thing they need from us — the representative images and the
tumour-content figure also come out of our pipeline. Worth remembering when scoping outputs.

### What the published literature says the model does

For orientation only, not for implementation. The CanAssist Breast literature describes a model
that takes the **five biomarker readings plus three clinical facts** — tumour size, tumour grade,
and lymph node status — and runs them through a machine-learning classifier (described as an
SVM-family model) to produce a score on a 0–100 scale, with a cutoff separating low from high
risk of distant recurrence.

Treat the specific numbers in that description as literature context, not as specification. **We
have not been given the model, the coefficients, or the cutoff, and we should not attempt to
reverse-engineer them** — that is both technically futile with six cases and commercially wrong.

### What the client repo does instead, and why

`backend/app/algorithm/literature_model.py` implements a **deliberate design-around**: the
**Nottingham Prognostic Index**, a fully public, non-proprietary, non-SVM prognostic index from
Galea 1992, with a small CD44 adjustment layered on:

```
NPI          = 0.2 x tumour_size_cm  +  grade_score(1-3)  +  node_score(1-3)

CD44 nudge   = 0.5 x (CD44 H-score / 300)        capped so the NPI always dominates

adjusted NPI = NPI + CD44 nudge

bands          good     <= 3.4      -> reported "low" risk
               moderate <= 5.4      -> reported "high" risk
               poor      > 5.4      -> reported "high" risk

probability  logistic curve fitted to published NPI 5-year outcomes
             (good ~6 %, moderate ~20 %, poor ~45 % recurrence risk)
```

This exists so the platform can demonstrate an end-to-end flow — images in, a risk category out —
**without touching OncoStem's patent family and without pretending to be their test.** The module
carries its own disclaimer: research/demo only, not clinically validated, not a diagnostic device,
must not drive treatment, and not a legal freedom-to-operate opinion.

**Keep the two firmly separate in every demo, every screen, and every conversation:**

| | Our deliverable | The demo risk number |
| --- | --- | --- |
| What it is | 5 × (percent, intensity) per case | An NPI-based illustration |
| Whose model | none — raw measurements | public literature (Galea 1992) |
| Validated? | against 4 pathologists' consensus | **no** |
| Goes to a patient? | via OncoStem's model, yes | **never** |

Anyone who leaves a demo believing our NPI number is a CanAssist score has been misled. Label it
on screen, not just in the docs.

---

## Part 8 — The open questions that change our numbers

These are the points where we are currently guessing. They are ordered by how much they change
what the software computes. All were sent to OncoStem in the SOP reconstruction dated 13 Aug 2026
(`docs/client/OncoStem/CanAssist-scoring-SOP-reconstruction.md` in the client repo).

| # | Question | Why it matters to the code |
| --- | --- | --- |
| **Q1** | When only part of a cell's membrane is stained, does that cell count as positive — with the partial staining pulling intensity down — or is it left out of the percentage? | Largest single effect on our output. The two readings give materially different results on intermediate cases. We have built both behaviours and can switch |
| **Q2** | Which intensity scale is authoritative: 1–3, or the 0–2 bands? | We implement 0–2 because the data does. If 1–3 is a per-field observation that gets converted, we need the conversion |
| **Q3** | When the tumour is split into three or four parts, are those parts weighted by area, by cell count, or averaged plainly? | Our pipeline naturally produces an area-weighted average, which differs whenever the parts are unequal |
| **Q7** | Is intensity judged relative to the internal control ducts on the same slide? | Our own analysis suggests that for four of five markers, a threshold set relative to each slide's own staining level behaves far more consistently than an absolute one — exactly what you would expect if pathologists normalise against an internal reference and we do not. This may be the difference between matching them and not |
| **Q6** | Are N-cadherin and pan-cadherin genuinely near-constant at 80 %, or is 80 % a working convention? | Determines whether agreement on U and W means anything at all |
| **Q5** | Is intensity one judgement per slide, or an average of the sub-areas? | 119 of 120 readings land exactly on a band, which points to one-per-slide. Changes whether we band before or after averaging |
| **Q4** | Does the ±10 re-review rule apply to intensity, or to percent only? | Sets the tolerance we hold ourselves to on intensity |
| **Q9** | Were the readings we have independent first reads, or the values *after* the re-review cycle? | Changes what counts as acceptable spread between pathologists, and therefore the standard we hold the software to |
| **Q10** | How is "viable" defined for partially degenerate but not frankly necrotic tumour? | Affects the denominator; may be an irreducible judgement call |

---

## Part 9 — How we prove the numbers are right

Without this the system is a rendering engine, not a measurement tool.

**Use the right metric for the right thing.** Using the wrong one is a classic mistake here:

| What you are validating | Metric | The trap |
| --- | --- | --- |
| Tumour region mask | Dice, IoU | Report per tumour-content band, never pooled |
| Nuclei detection | F1 at a matching distance | Accuracy is meaningless — the classes are wildly imbalanced |
| **Percent positive** | **Lin's concordance correlation, Bland–Altman** | **Not plain R²** — R² is blind to systematic bias |
| Intensity band | Weighted Cohen's kappa | Unweighted kappa treats 0-vs-2 as no worse than 1.75-vs-2 |

**Three things people skip and should not:**

1. **Report against the four-reader consensus, not one reader.** Two pathologists routinely
   disagree with each other on IHC scores. Our disagreement with a single reader is not
   necessarily an error. Report inter-reader agreement first, then ours in that context. **If our
   system sits inside the human spread, that is the actual result** — say it that way.
2. **Stratify by difficulty.** Overall correlation hides everything. Bin by tumour content and
   report each bin separately.
3. **Say what the sample size buys you.** Six cases and 240 readings give an enormous confidence
   interval on any correlation. State the interval rather than quoting a bare point estimate.

**And be honest about what this batch does not contain.** All six cases are 70–90 % tumour, which
OncoStem themselves called the easy half. The hard case they described — a 2–3 mm tumour focus
inside a 2–3 cm block — is not represented here at all. We should ask for some before claiming
the region-finding step works.

---

## Part 10 — Known problems in the current implementation

From running the platform's own code against these real files. Recorded here so nobody
rediscovers them the hard way.

| Problem | What is happening | Severity |
| --- | --- | --- |
| **Scoring at the wrong zoom** | The app picks the coarsest level under 2048 px — 7.11 µm/px. A 12 µm tumour cell is 1.7 pixels across. Percent-positive computed there is a texture statistic, not a cell measurement | **Blocking.** Nothing downstream is trustworthy until fixed |
| **Speed** | Cell detection takes 12.1 s per tile; a slide's tumour area is ~300 tiles. That is ~1 hour per slide, 6 hours per case, months for the 350 cases OncoStem mentioned | **Serious, but ordinary engineering.** Three levers: parallelise across cores, move to a GPU detector (the code anticipates this), or sample representative tiles and publish the sampling error |
| **Tumour content over 100 %** | One slide reported "tumour content 103.14 %" — more than all the tissue is tumour. A smoothing step expands the region past the tissue outline and nothing clamps it | Ten-minute fix, but a pathologist seeing 103 % on screen rightly loses confidence in every other number on that screen |
| **DCIS vs invasive** | The current detector is a density-based placeholder. At 7.11 µm/px it physically cannot resolve the duct shapes that distinguish DCIS from invasive tumour | **The real research risk.** Needs a trained classifier and pathologist-outlined ground truth. Interim honest option: let the pathologist draw the region — the API already supports it |
| **Identifiers inside the image files** | Each `.svs` carries a photo of the slide's paper label showing case number, block number, prep code and QR code. Our de-identification step does not touch these | Lab reference numbers rather than patient names, so not severe — but we told OncoStem we strip identifiers at ingestion, and right now we do not. Blank the label and slide-overview images on ingest |

---

## Appendix — The five things to remember

1. **The letter in the filename is the antibody.** A = CD44, F = ABCC4, R = ABCC11 (all
   membrane); U = N-cadherin, W = pan-cadherin (both cytoplasm); H&E = structure only, never
   scored.
2. **Ten numbers per case is the whole deliverable.** Five markers × (percent positive 0–100,
   intensity 0–2). Not an H-score. Not a risk score.
3. **Score only viable invasive tumour.** Not DCIS, not normal ducts, not stroma, not fat, not
   necrosis. Getting this region right is the hard problem; everything else is arithmetic.
4. **Calibrate for measuring, normalise only for models.** Confusing the two normalises away the
   diagnosis, and it does so silently.
5. **The risk model is OncoStem's.** We produce measurements and hand them over. The NPI number in
   our repo is a public-literature demo, clearly labelled, and must never be presented as a
   CanAssist score.
