# Aligning our numbers with OncoStem's: what is wrong, what is fixable, what is not

Written 2026-09-15 after reading `data/oncostem_requiremnet_docs/`, in particular
`client/OncoStem/CanAssist-scoring-SOP-reconstruction.md` and the real
`client/OncoStem/6Slide Reports2.xlsx`.

The question this answers: **our CAN_00270 numbers were over-called by +14.5 points on
average. Why, and what in the code is actually wrong?**

The short version: **the percentage is one parameter away from correct, and the
intensity is a different measurement we are not making at all.** Those two need
completely different responses, and treating them as one problem is why the bias looked
mysterious.

---

## 1. The diagnosis: one fixable fault and one that is not the same fault

My first reading of this was that over-calling the percentage while under-calling the
intensity was **one** fault seen from two sides — a positivity cut set too low, which
admits weak cells, raising the count *and* dragging down the mean density of the cells
called positive. One parameter, both directions.

**Half of that is right, and the test that confirmed it also refuted the other half.**

Sweeping only the positivity cut, with the percentage as the sole target and the
intensity left to fall out wherever it lands:

| | Shipped cuts | Cut fitted to percent only |
|---|---|---|
| Percent — mean absolute error | 15.5 points | **3.5 points** |
| Percent — bias | +9.5 | **−2.5** |
| Intensity — mean absolute error | 0.362 | **0.412** *(worse)* |
| Intensity — bias | −0.312 | −0.263 |

The percentage collapses to near-exact. **The intensity does not follow it**, and gets
slightly worse. So the cut is the percentage's whole problem and almost none of the
intensity's.

The reason is the band table. Our positive-cell mean densities sit at 0.28–0.50 OD and
the reporting bands are wide, so moving the cut rarely moves a marker across a band
boundary — only ABCC4 shifted (1.5 → 1.75), and in the wrong direction. Intensity error
is dominated by the *mapping*, not by which cells are averaged, which is section 3.

## 2. The percentage is a one-parameter fix, and it lands

Sweeping only the positivity cut over the already-stored per-cell rows, per marker:

| Marker | Shipped cut | Fitted cut | Our % | Readers' % | Error |
|---|---|---|---|---|---|
| A CD44 | 0.150 | 0.093 | 65 | 80 | **−15** |
| F ABCC4 | 0.150 | **0.369** | 65 | 62.5 | +2.5 |
| R ABCC11 | 0.150 | 0.170 | 55 | 55 | 0 |
| U N-cadherin | 0.120 | **0.263** | 80 | 80 | 0 |
| W Pan-cadherin | 0.120 | 0.138 | 80 | 80 | 0 |

Four of five land exactly or within the rounding step. **CD44 is the whole remaining
error**, and it is not a threshold problem — no cut reaches 80, because the shortfall is
the region-heterogeneity question in section 4.4.

**So the measurement machinery is right and the thresholds are wrong.** Everything
upstream — segmentation, compartments, deconvolution, the ring-completeness test — is
producing a per-cell signal from which the readers' own answer is recoverable. That is
the most encouraging result in this analysis and it is worth stating plainly.

Note also how far apart the fitted cuts are: 0.142 for CD44 against 0.422 for ABCC4, a
factor of three. This is the strongest evidence yet for five independent cut-point sets
— one shared table could not straddle that.

### Why the cuts have NOT been changed

Fitting five parameters to five numbers on **one case** reproduces the target by
construction and demonstrates nothing. `scripts/calibrate_marker_cuts.py` does the fit
properly — leave-one-case-out, fit and held-out error reported separately — and
**refuses to write below four cases**. Today it refuses, and prints the table above as a
diagnostic.

**To actually fix the percentage: score the other five cases.** Only CAN_00270 has its
five IHC slides on disk; the rest of `data/original/oncostem_slides/` holds H&E only.
With four or more cases the fit becomes calibration and the script will write
`marker_cuts.v2.json`.

## 3. The intensity is a different measurement, and no threshold fixes it

The obvious idea — normalise the optical density against something on the same slide —
was tested against every reference available without new segmentation:

| Reference | Spearman with readers' intensity |
|---|---|
| positive cells' mean OD | +0.15 |
| negative cells' mean OD | −0.36 |
| all cells' median OD | +0.10 |
| positive − negative | +0.36 |
| positive ÷ negative | +0.10 |

None of them orders the five markers the way the readers do. At n=5 even +0.36 is noise.
The ordering is genuinely different: ABCC4 has the **highest** optical density of the
five and the **fourth** reader intensity; CD44 is middling on density and joint-highest
on intensity.

**The SOP says why.** Clause 5.7: *"Intensity is graded against the internal control
ducts, or against the batch control when internal control is absent."* The readers are
not measuring absolute brownness. They are measuring brownness **relative to normal
ducts on the same slide**, which normalises away section thickness, DAB development time
and staining batch in one step. We measure absolute brownness and have never looked at a
duct.

This is open question Q7 in the client register, where our own earlier analysis had
already noticed that relative thresholds behave more consistently than absolute ones for
four of five markers. The SOP confirms the mechanism.

### What implementing it would take, and why it was not bolted on

1. Step 10 carries `non_invasive_epithelium` regions as well as invasive ones.
2. Steps 11–14 measure DAB in those regions the same way they do in tumour.
3. Step 16 reports the tumour's intensity relative to that control population.

Step 1 is the problem. Step 8's `non_invasive_epithelium` class contains **normal ducts
and DCIS together**, and DCIS expresses these markers. Using that class as an internal
control would normalise the tumour against other tumour — a plausible-looking number
that is wrong in a way nothing downstream could detect. Separating normal duct from DCIS
is a real modelling task, not a config change.

So: **scoped, not guessed.** The honest position today is that our intensity is an
absolute optical-density band and theirs is a ratio to an internal control, and those are
not the same quantity.

### One more thing about the intensity ground truth

It is much softer than the percentage. On CAN_00270 the four readers gave CD44 intensity
1.5, 2, 2, 1.75 — a spread of 0.5 on a 0–2 scale, a quarter of the range. The SOP itself
notes (8.6, Q4) that the ±10 re-review rule appears not to apply to intensity at all.
Intensity should be held to a looser standard than percentage, and any fit against it
needs far more than six cases.

---

## 4. Sampling: "instead of top 3 regions, which works best"

**Neither. The region count was the wrong knob.** The SOP is explicit, clause 4.2:

> *"Percentage may vary field to field; the entire slide is scanned and the average of
> all fields is the final value."*

Two things were wrong against that, and the second is worse than the first.

### 4.1 Coverage: three regions was 63% of the tumour

On CAN_00270, step 9 finds **104 invasive regions totalling 41.4 mm²**. The three
largest are 26.2 mm². So 37% of the invasive carcinoma was never looked at, and the
report said only "3 regions carried".

| Regions carried | Invasive area covered |
|---|---|
| top 3 | 63.2 % |
| top 5 | 69.4 % |
| top 10 | 76.4 % |
| top 20 | 84.5 % |
| top 50 | 93.5 % |

**Changed to: select by area covered, not by count.** Regions are taken in step 9's
ranking order until 90% of the invasive area is reached, subject to a 0.25 mm² minimum
region size and a cap of 24. On this case that selects **19 regions covering 84%**, and
the alignment report now states the coverage it actually reached rather than a region
count that says nothing about how much tumour was read.

The registration is cached per pair, so carrying more regions costs almost nothing —
only the extra rings are warped.

### 4.2 Allocation: the real problem, and it was distorting the answer

Every region got **twelve fields regardless of size**. A 22 mm² region and a 2 mm² region
each contributed twelve. So the measured cells were a sample of the *regions*, not of the
*tumour*: the largest region held 84% of the invasive area and supplied 42% of the cells.

That is exactly why pooling the cells and weighting the regions by area disagreed by more
than twenty points on CD44 — 60% against 81%. They were two different attempts to undo
the same bad allocation.

**Changed to: a 96-field budget split in proportion to region area.**

| | Before | After |
|---|---|---|
| Regions carried | 3 | 19 |
| Invasive area covered | 63 % | 84 % |
| Fields | 36 | 96 |
| Tumour actually measured | **4.7 %** | **12.5 %** |
| Largest region's share of the sample | 33 % | 52 % (its area share is 63 %) |

This is a correctness fix rather than a tuning knob, and the reason is worth being
precise about: **with a proportional sample, the plain mean over fields and the
area-weighted mean of the regions are the same number.** Open question Q3 — "are the
sub-areas weighted by area, by cell count, or averaged plainly?" — stops changing the
answer instead of being resolved by fiat. The numbers become consistent because the
sampling stopped making them inconsistent, which is the only legitimate way to make two
numbers agree.

### 4.3 Why the per-region floor is one field, not four

A first attempt gave every carried region a floor of four fields so its own percentage
would be readable. That silently undid the fix: 19 regions × 4 = 76 of the 96 fields
went to floors, leaving 20 to distribute, and the region holding 63% of the area got
**18%** of the sample — worse than before.

The floor is one because **a region does not need a readable percentage of its own.**
Step 16 combines regions by area, so a 0.25 mm² region carries 0.7% of the weight however
noisily it was measured. The allocation decides precision, not the answer; spending
fields to tidy a negligible region's percentage takes them from the region that
determines the score.

---

## 5. Also corrected against the SOP

- **CD44's expected range is 0–85 %, not 5–85 %** (SOP 9.1). Ours said 5.
- The `expected_percent` field now records that these ranges are explicitly *descriptive*
  and never thresholds — SOP 9.1 says a case outside them is not by itself an error, and
  9.2 says no positivity cut-off is applied at the scoring stage at all.

## 6. Requirements we do NOT meet, stated rather than quietly skipped

**SOP 2.6 — "areas unstained across all markers are treated as non-representative and
counted out."** I first recorded this as unimplementable, on the grounds that each
marker's slide is sampled at its own coordinates so there is no correspondence between
markers. **That was wrong at the level that matters.** The regions are traced once on the
H&E and warped onto each marker's slide, so *region rank N is the same anatomy on all
five*. The correspondence exists one level up from where I looked for it.

`scripts/check_region_consistency.py` prints the grid this clause needs — every carried
region against every marker — and what each marker would report with the unstained-
everywhere regions counted out. It is a diagnostic; nothing is excluded automatically,
because whether a 10% floor is what OncoStem mean by "unstained" is a question for them.

This matters more than it first looked: see section 4.4.

**SOP 5.7 — intensity against the internal control.** Section 3 above.

**SOP 2.7 — "slides are not registered to one another; normal ducts serve as
landmarks."** We *do* register, with VALIS. This is a deliberate divergence: registration
is a defensible engineering substitute for a human re-identifying the tumour by eye, and
it is measured and gated rather than assumed. Worth confirming with OncoStem that a
registered region is acceptable in place of a re-identified one.

**Step 11's nuclei shortfall — 68–79% fewer nuclei per mm² than the case's own H&E.**
Unchanged by any of this and still the largest single threat to the numbers. Missed
nuclei skew toward the strongly stained, so it inflates the percentage independently of
the cut point.

---

## 6b. What the approach study settled (2026-09-15, evening)

Nine variants were built and measured. [SCORING_APPROACHES.md](SCORING_APPROACHES.md) has
the full grid; the three results that matter:

**The internal control does not work as implemented, and I no longer claim it is the
answer.** Section 3 above proposed it as the likely fix for intensity. It has now been
measured on every slide (`measure_internal_control.py`) and the tumour-to-control ratio
orders the markers *backwards*. The measurement is untrustworthy for a visible reason —
CD44's control reads 0.036 OD, essentially blank, for a marker that should stain normal
ducts — and the likely cause is that the non-invasive epithelium is 141 small regions
where registration error is worst. So SOP 5.7 may still be right; we cannot yet measure
what it describes. See F3.

**A registration bug had been dropping two markers.** Step 10 refused CD44 and ABCC4 with
"VALIS recorded no registration summary" while the summary CSV sat on disk — the
in-memory attribute is empty on a reused registration. Fixed. This is why the earlier
comparison ran on three markers. See F4.

**Intensity's best-scoring variant was rejected on principle.** A 90th-percentile
statistic gave the lowest error of anything tried, and SOP 5.3 specifies an average.
Shipping it would have been choosing what fit three data points over what the procedure
says. What shipped instead is a caveat: this pipeline's densities occupy two of the band
table's six bands, so the reported intensity has far less resolution than the scale
implies.

---

## 7. What to do next, in order

1. **Get the other five cases' IHC slides.** Everything in section 2 is blocked on n=1.
   With four or more cases the cut points become calibrated rather than provisional, and
   the +14.5 point bias is a solved problem.
2. **Ask OncoStem Q7 directly** (intensity vs internal control). If confirmed, the duct /
   DCIS separation in section 3 becomes the next real piece of modelling work.
3. **Fix step 11's nuclei shortfall on IHC.** It biases the percentage in the same
   direction as the cut point, so fixing the cut without fixing this would transfer error
   rather than remove it.
4. **Ask Q1** (partial membrane staining). On this case all three readings give the same
   answer, so it costs nothing today — but that is a fact about CAN_00270, not a general
   result.
