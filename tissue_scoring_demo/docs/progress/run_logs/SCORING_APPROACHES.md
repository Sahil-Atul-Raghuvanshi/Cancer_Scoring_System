# Scoring approaches: what we tried, what we measured, what we shipped

Started 2026-09-15 15:15. Live status at the bottom; the runner appends to
`SCORING_REBUILD_LOG.md`.

**The target is *similar*, not identical.** Four pathologists reading the same slide
disagree by up to 10 percentage points and 0.5 intensity bands, so there is no single
correct number to hit. The standard is landing inside that spread. A system that matched
one consensus value exactly would be fitting noise, not measuring tissue.

**The rule for every approach below.** Ship it only if it is right on the evidence or on
OncoStem's own written procedure. Never because it moves one case's number in a pleasing
direction. With one scored case, a five-parameter fit reproduces the target by
construction and demonstrates nothing — so fitting is used to *reject* approaches and to
*measure the residual*, not to choose thresholds.

---

## The two numbers, and why they need different answers

| | Where the error is | Fixable now? |
|---|---|---|
| **Percent positive** | The positivity cut per marker | Yes in principle — blocked on cases |
| **Intensity** | The optical-density-to-band mapping | Needs the internal control (SOP 5.7) |

Measured on CAN_00270 against the four readers (from the real `6Slide Reports2.xlsx`):

```
percent    MAE 15.5 pts   bias +9.5    2 of 5 inside the human spread
intensity  MAE 0.362      bias -0.31   3 of 5 inside the readers' own spread
```

Sweeping only the positivity cut, with percent as the sole target, collapses percent MAE
to **3.5** — and leaves intensity *slightly worse* (0.412). That single experiment is
what splits the problem in two, and it is why the approaches below are grouped by number
rather than by mechanism.

---

## Part 1 — Percent positive

### P0 · Baseline: absolute cuts, area-weighted over 90 % coverage *(shipped)*

What runs today. Five independent cut tables on the calibrated DAB optical-density scale;
regions selected until 90 % of invasive area is covered; sampling fields allocated in
proportion to region area; regions combined by area.

| | |
|---|---|
| Justification | SOP 4.2 (entire slide scanned, all fields averaged) |
| Status | **shipped** |
| Result | MAE 15.5, bias +9.5 |

### P1 · Region selection

Which connected components of invasive carcinoma enter the score.

| Variant | Rationale | Verdict |
|---|---|---|
| top 3 regions | the original | 63 % of invasive area unmeasured-by-design; contradicts SOP 4.2 |
| **90 % area coverage** | SOP 4.2 | **shipped** |
| dominant mass only | SOP 1.4 — block chosen for *most invasive tumour content* | see finding F1 |
| all 104 regions | literal reading of 4.2 | tail is single-tile speckle; rings cost more than the area adds |

### P2 · Field allocation

| Variant | Verdict |
|---|---|
| fixed 12 per region | **rejected** — makes the sample a sample of the *regions*, not the tumour |
| **proportional to area** | **shipped** — and it makes P3's variants agree (see below) |

### P3 · Combining regions — open question Q3

| Variant | CD44 | Note |
|---|---|---|
| **area-weighted** | 64.2 % | shipped; what SOP 4.2 implies |
| pooled over all cells | 53.5 % | equals area-weighted *only* under proportional sampling |
| plain mean of regions | 27.6 % | gives a 0.25 mm² fragment the same say as a 22 mm² mass |

Proportional allocation is what makes the first two converge. They still differ by ~10
points here because the one-field-per-region floor lifts small regions above their share.

### P4 · Cut points

| Variant | Verdict |
|---|---|
| **absolute, per antibody** | **shipped** — SOP-neutral, comparable across slides |
| per-slide percentiles | **rejected** — a weak and a strong slide score alike |
| one shared table | **rejected** — fitted cuts span 0.09–0.37, a factor of four |
| fitted per antibody | **correct, and blocked**: needs ≥ 4 cases; `calibrate_marker_cuts.py` refuses at n=1 |

### P5 · Partial membrane staining — open question Q1

count / half / exclude. On CAN_00270 all three give the identical answer for all five
markers, so it costs nothing today. That is a fact about this case, not a general result.

---

## Part 2 — Intensity

### I0 · Baseline: absolute OD → fixed band table *(shipped)*

| | |
|---|---|
| Result | MAE 0.362, bias −0.31, 3 of 5 inside the readers' spread |

### I1 · The band-range mismatch — a measurable defect, not a preference

Our positive-cell mean densities land at **0.28–0.50 OD**. The shipped band table needs
≥ 0.60 for band 1.75 and ≥ 0.80 for band 2.0. **So this pipeline can essentially never
emit 1.75 or 2.0**, while the readers use those bands constantly — 13 of the 30 case
averages in the real sheet are ≥ 1.75.

That is a defect in the table rather than a tuning preference: the breakpoints were
written from the abstract OD scale without reference to the densities this pipeline
actually produces. Fixing it by fitting to one case is overfitting; fixing it by
rescaling to the *observed range* is a different and more defensible operation. Both are
tested in Part 3.

### I2 · Internal control — the SOP's own answer

> SOP 5.7: *"Intensity is graded against the internal control ducts, or against the batch
> control when internal control is absent."*
> SOP 2.4: *"Normal ducts are identified and used as the internal control."*

The readers do not measure absolute brownness. They measure brownness **relative to
normal ducts on the same slide**, which cancels section thickness, DAB development time
and staining batch at a stroke. We measure absolute brownness and have never looked at a
duct — and our absolute densities correlate with their intensities at Spearman **+0.15**,
which is nothing.

`scripts/measure_internal_control.py` measures the DAB in step 8's
`non_invasive_epithelium` on each IHC slide, reusing the cached registration.

**The known weakness, stated up front:** that class holds normal ducts *and* DCIS
together, and DCIS expresses these markers. A contaminated control is pulled toward the
tumour, compressing the ratio. That biases the test *against* finding an effect — which
is the safe direction: an effect seen through the contamination is real, a null result
could still be the contamination.

### I3 · A different statistic over the positive cells

Mean is shipped. Median and the 75th percentile are cheap to test from the stored rows,
and a reader's visual impression of "how brown is this slide" may be closer to a high
percentile than to a mean.

---

## Part 3 — How each approach is judged

With one case there are five (marker, value) pairs per number. That admits exactly two
honest tests:

1. **Ordering.** Does the approach rank the five markers as the readers did? A perfect
   ordering of five has a 1-in-120 chance of arising by luck, so a clean result is
   evidence and a null is informative.
2. **Residual after fitting.** Fit the approach's free parameters to this case, and see
   what error *remains*. An approach that cannot match even when fitted is wrong in
   structure, not merely mis-tuned. This rejects; it never selects.

What is explicitly **not** a test: "this variant moved CD44 closer to 80". That is the
tuning this project is organised against.

---

## Findings

### F1 · The small regions are real tumour, and CD44 is genuinely heterogeneous

I suspected the 18 small regions newly included by the 90 % coverage rule were step 8
false positives or badly registered stroma. **They are not.** Every one is 78–100 %
positive on ABCC4 and N-cadherin. `check_region_consistency.py`: **0 of 19 regions are
unstained on every marker**, so SOP 2.6 excludes nothing and changes every marker by
+0.0.

So CD44's dominant mass is 89 % positive and its scattered foci are near-zero — real
biological heterogeneity, exactly what SOP 6.2 anticipates. The readers' 80 % matches the
dominant mass; area-weighting over all 19 regions gives 64 %.

**This is a question for OncoStem, not a bug to fix:** when a tumour has a dominant mass
plus scattered foci that stain differently, is the reported percentage over all of it or
over the main mass? On CD44 it is a 25-point swing — larger than any threshold effect.

### F2 · The cut fixes percent and not intensity

Confirmed by direct sweep. See the table at the top. This is why Parts 1 and 2 are
separate.

### F3 · The internal control was measured, and it does not work as implemented

I2 was the approach with the best prior justification — it is what OncoStem's own SOP
says their readers do. It has now been measured on all five slides, and it fails.

| Marker | Control OD | Tumour OD | Ratio | Readers' intensity |
|---|---|---|---|---|
| A CD44 | **0.036** | 0.39 | ~10.8 | 1.81 |
| F ABCC4 | **0.608** | 0.50 | 0.82 | 1.375 |
| R ABCC11 | 0.268 | 0.300 | 1.12 | 1.125 |
| U N-cadherin | 0.499 | 0.440 | 0.88 | **1.81** |
| W Pan-cadherin | 0.305 | 0.314 | 1.03 | 1.44 |

**Corrected once all five markers were available.** On the three markers measured first,
the ratio ordered them exactly backwards (Spearman −1.00). With CD44 and ABCC4 restored it
is **+0.54 — the best ordering of any intensity variant tried**, against +0.15 for the
shipped mean and +0.27 for a 75th percentile. That is still not significant at n=5 (a
perfect +1.00 would be p≈0.008; +0.54 is well inside chance), but it is the opposite of
the reading I recorded from three points, and it deserves saying plainly: the earlier
"exactly backwards" was an artefact of the two missing markers.

Its *absolute* mapping is nonetheless hopeless — MAE 1.188, the worst of any variant —
and the reason is one bad measurement rather than the mechanism. CD44's ratio is 10.8
where every other marker sits between 0.8 and 1.1, because its control reads 0.036 OD.
Rescaling a range with a 10× outlier in it puts everything else in the bottom band.

**The measurement is not trustworthy, and the reason is visible in it — and it is the
same measurement that drives the outlier above.** CD44's control reads 0.036 OD,
essentially unstained. CD44 is a basal marker that should stain normal ducts, so a blank
control there is far more likely to mean the control regions landed on something that is
not duct than that the ducts are negative. The non-invasive epithelium
on this case is 16.9 mm² spread over **141 regions**, so the control regions are small —
and small regions are exactly where registration error is worst, which is the same effect
F1 examined from the other side.

So the SOP's mechanism may well be right — it has the best ordering on the board, from
the one variant with a principled reason to work — and this implementation of it is too
noisy to build on. **Not shipped, and not closed either.** What it would take to test properly:

1. separating normal duct from DCIS, which step 8's class does not do;
2. control regions large enough that a 174 µm round-trip error does not move them off
   the duct;
3. more than one case, so the ordering test has power.

A null result from a measurement this shaky is not a refutation of clause 5.7 — it is a
statement that we cannot yet measure what clause 5.7 describes.

---

## The decision rule, written before the results

Fixed in advance so the outcome cannot pick the rule that flatters it.

**Ship an approach if, and only if, one of these holds:**

1. **It follows OncoStem's written procedure.** The SOP is the specification; a
   clause we are not implementing is a defect whether or not fixing it improves the
   numbers. This is how P1 (coverage), P2 (proportional fields) and I2 (internal
   control) got built — none of them was chosen because it moved a number.
2. **It removes an inconsistency of our own making.** P2 qualifies: proportional
   sampling makes the pooled and area-weighted readings agree, so Q3 stops changing
   the answer. The numbers become consistent because the cause of the inconsistency
   was removed, not because a convention was imposed.
3. **It fixes a measurable defect in our own machinery.** I1 qualifies if the band
   table genuinely cannot reach 1.75 and 2.0 — that is a range error, independent
   of any target.

**Do not ship an approach because:**

- it lowers the error on CAN_00270 (one case, five points — that is fitting);
- it makes CD44 land on 80 (that is the F1 open question, not a bug);
- it produces "nicer" numbers.

**What the two tests decide.** Ordering (Spearman) is evidence *for* a mechanism —
a perfect ordering of five markers is p≈0.008. Residual-after-fitting is evidence
*against* one: an approach that cannot match even when fitted is structurally wrong.
Neither selects a threshold. Thresholds come from `calibrate_marker_cuts.py` once
there are four or more cases, and not before.

---

## F4 · A registration bug that silently dropped two markers from the study

CD44 and ABCC4 refused at step 10 with *"VALIS recorded no registration summary. Its
output directory may be too deep for the filesystem to write into"* — a real failure mode,
and not this one. `pair_summary.csv` was sitting in the output directory with every number
the gate needed.

`registrar.summary_df` is empty on a **reused** registration: the pickle is loaded rather
than recomputed and the attribute does not survive. The reader only looked in memory, so
a successful registration was reported as an unmeasurable one, and the message named the
wrong cause with complete confidence.

Fixed in `valis_service/register.py`: fall back to the CSV VALIS itself wrote. That is
not inventing a measurement — it is the same measurement from the file instead of from an
attribute that did not survive pickling. The path-limit case still refuses, because there
the file genuinely is absent, and the message now distinguishes the two.

Worth noting for what it cost: this is why the first study ran on three markers instead of
five, and three points is too few for the ordering test to say anything.

---

## The decision, taken against the rule written above

### Percent positive — keep P0, fix the cuts when cases arrive

Every region-combining variant lands within 1.6 points of the others on this case
(MAE 11.7–13.3), so the metric does not choose between them and, by the rule, nor do I.
**Area-weighted stays**, because SOP 4.2 implies it: the entire slide is scanned and every
field averaged, and weighting by area is what that means when the sample is proportional.

The bias is **+11.7 points and systematic** — every marker over-called. That is the cut
points, and sweeping them collapses the error to 3.5. It is not shipped, because five
parameters fitted to five numbers on one case is not calibration. `calibrate_marker_cuts.py`
refuses below four cases and will write `marker_cuts.v2.json` when they exist.

### Intensity — ship nothing, report the limit

| Variant | Verdict |
|---|---|
| I0 mean, shipped bands | **kept** |
| I3 p90 (MAE 0.167, 3/3 inside) | **rejected** — best metric, no mechanism |
| I1 rescale to observed range | rejected — two free parameters, one case |
| I2 internal control | rejected — measured; see F3 |

The p90 rejection is the one worth explaining, because it had the best numbers on the
board. There is a story for it — a reader judging "how brown is this slide" may attend to
the strongest convincing staining rather than the average — but SOP 5.3 says *"the average
intensity observed over the entire section is the recorded value"*, which says average.
Shipping p90 would be choosing the variant that fit three points over the one the
procedure specifies. That is precisely the move the rule forbids.

**What did ship is a caveat.** This pipeline's positive-cell densities span 0.30–0.44 OD,
and the band table's breakpoints put that entire range inside two of six bands. So the
reported intensity has a six-value scale and about two values of resolution — measurable
without reference to any target, and something a reader comparing our intensity against a
pathologist's needs to know before reading the gap. Step 16 now says so when a slide sits
at either edge of the table.

**And the calibration machinery was extended.** The intensity is the mean density of the
cells the *cut* admitted, so the two reported numbers are coupled through one parameter:
fitting the cut while holding the band table fixed calibrates half a model.
`calibrate_marker_cuts.py` now fits the breakpoints alongside the cut, on the densities
that cut produces — still refusing below four cases. No shipped number changes today; the
path is correct for when the data arrives.

---

## Status

- [x] Ground truth wired in — the real `6Slide Reports2.xlsx` found in the requirement docs
- [x] P1, P2, P3 implemented and measured
- [x] F1 tested and settled — small regions are real tumour, CD44 is genuinely heterogeneous
- [x] F2 tested and settled — the cut fixes percent, not intensity
- [x] F3 internal control measured; verdict corrected once all five markers were in
- [x] F4 VALIS `summary_df` bug found and fixed — it had been dropping two markers
- [x] Rebuild after the artefact wipe, all five markers
- [x] I1 band-range mismatch quantified, and reported as a caveat by step 16
- [x] I3 statistic variants measured; p90 rejected despite the best numbers
- [x] Final decision taken and implemented
- [x] UI updated — step 10 shows area coverage, step 11 explains proportional sampling
- [x] Calibration extended to fit the band table alongside the cut
- [x] All five pairs re-derive independently (`verify_case_scores.py`)

### Final numbers, CAN_00270, five markers

| Marker | Ours | Readers | Diff | Inside their spread? |
|---|---|---|---|---|
| A CD44 | 65 % / 1 | 80 % / 1.81 | −15 | no |
| F ABCC4 | 95 % / 1.5 | 62.5 % / 1.375 | +32.5 | no |
| R ABCC11 | 65 % / 1 | 55 % / 1.125 | +10 | **yes** |
| U N-cadherin | 100 % / 1.5 | 80 % / 1.81 | +20 | no |
| W Pan-cadherin | 85 % / 1 | 80 % / 1.44 | +5 | **yes** |

percent MAE 16.5, bias **+10.5** · intensity MAE 0.362, bias −0.31, 3 of 5 inside

The bias is one parameter from fixed: sweeping the cut collapses percent MAE to **3.5**.
It is not fitted, because one case cannot calibrate five thresholds.

### The artefact wipe, 2026-09-15 ~15:10

Every derived artefact for CAN_00270 was deleted while the demo app was open: the upload
records went (a UI release or an explicit cleanup), and the backend's start-up orphan
sweep then correctly removed the derived data that had become unreachable. The sweep is
not at fault — it only removes data whose upload is gone.

Cost: the full pipeline is rebuilding, roughly an hour and a half. Nothing is lost
permanently; the slides are untouched.

**Protection for next time.** The sweep is correct and stays. What is worth knowing is
that opening the demo app while a study is running can release the slide records the
study depends on. `SCORING_STUDY_CHECKPOINT.json` now records which phases completed, so
an interrupted study resumes rather than restarts.

### Concurrent edits in this tree

Files are being changed by something other than this session — the frontend panels were
reformatted at 11:49, and `settings.tiling_branch` changed under two step-7 tests and
then changed back. Nothing in the scoring path was affected, but it is worth knowing when
reading timestamps or a test that fails once and passes next run.
