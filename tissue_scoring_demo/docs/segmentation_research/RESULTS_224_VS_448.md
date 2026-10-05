# Fix 1 + Fix 2 — results

> **Superseded for the cross-field-of-view question.** This document compares two
> haematoxylin arms, 224 µm and 448 µm, and reports z = 1.92, p = 0.055 — suggestive
> and not established. All four fields of view now have a head, and each of them has an
> **H&E** counterpart on the same squares, so the comparison to read is
> [`HE_VS_HCHANNEL.md`](HE_VS_HCHANNEL.md): eight cells, paired per tile within a field
> of view, with the selection effects stated above the tables. What is still only here
> is the Fix 1 accounting below, which nothing supersedes.
>
> Tile stores moved: the `data/224_um` style paths named here are now
> `data/h_channel/224um`, with the underscore normalised.

Two changes, measured together. Every number below is read out of a file the run wrote; the file is named so it can be checked.

- **Fix 1 — the consensus decides the lesion.** Both of BEETLE's epithelium classes are written as one code inside a region of interest, and which code that is comes from the three-pathologist ROI consensus rather than from the model. A region the teacher misreads is now relabelled instead of deleted.
- **Fix 2 — the field of view.** A 112 µm window cannot contain a duct wall, so solid and comedo DCIS is undecidable at that scale. Two exports, at **224 µm** and **448 µm**, one head each.

## Fix 1 — what the rule change recovered

`rejected` used to be where the hard morphology went. Under the new rule a region is only rejected when the teacher found almost no epithelium at all — a resolution or stain fault — never for disagreeing about the lesion.

| tree | regions kept | rejected | patients | epithelium written as | consensus overruled the teacher | median share overruled |
| --- | --- | --- | --- | --- | --- | --- |
| `dcis` | 239 | 0 | 84 | `dcis` | 183 regions | 16% |
| `ic` | 202 | 3 | 111 | `tumor` | 94 regions | 11% |
| `normal` | 213 | 0 | 130 | `normal_acinus_or_duct` | 158 regions | 0% |
| `bach_insitu` | 100 | 0 | 100 | `dcis` | 100 regions | 47% |
| `bach_invasive` | 99 | 1 | 99 | `tumor` | 19 regions | 4% |
| `bach_normal` | 100 | 0 | 100 | `normal_acinus_or_duct` | 90 regions | 13% |

**The BACH in-situ row is the headline.** Under the old rule BEETLE disagreed with BACH's two pathologists on 47 of those 100 images, every one by calling invasive where they said in-situ, and all 47 were dropped. A 'second laboratory' filtered down to what the teacher already found familiar is not a second laboratory.

## Fix 2 — the two exports

| | 224 µm | 448 µm |
| --- | --- | --- |
| geometry | 224 px @ 1.0 µm/px | 224 px @ 2.0 µm/px |
| tiles kept | 6,907 | 1,157 |
| non_epithelium | 2009 | 275 |
| non_invasive_epithelium | 1772 | 277 |
| invasive_epithelium | 3126 | 605 |
| — of which `bcss/` | 2253 | 424 |
| — of which `beetle/` | 3332 | 556 |
| — of which `bach/` | 1322 | 177 |

**Read the `bach` row with the geometry in mind.** A BACH image is 2048×1536 at 0.42 µm/px, about 860×645 µm of tissue, so a 448 µm window fits **once** in it and a 224 µm window fits six times. The 448 µm export is therefore far more BCSS-weighted than the 224 µm one, by arithmetic rather than by choice, and that is a real limit on what its in-situ number can mean.

## The two heads, on held-out data

`dcis_called_invasive` is the number the whole exercise is about: the share of held-out in-situ tiles the model calls invasive. The served checkpoint is in the first column for reference — it was fitted at **112 µm** under the old label rule, so it differs from these two in both changes at once and is a reference point rather than a controlled comparison.

| metric | served v3 (112 µm, old rule) | 224 µm / imagenet | 224 µm / simclr | 448 µm / imagenet | 448 µm / simclr | 672 µm / imagenet | 672 µm / simclr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **dcis_called_invasive** | 22.4% | — | — | — | — | — | — |
| recall in-situ | 0.827 | — | — | — | — | — | — |
| recall invasive | 0.839 | — | — | — | — | — | — |
| recall stroma | 0.915 | — | — | — | — | — | — |
| macro recall | 0.861 | — | — | — | — | — | — |
| min recall | 0.827 | — | — | — | — | — | — |
| Dice invasive vs in-situ | 0.892 | — | — | — | — | — | — |
| held-out tiles | 7473 | 1848 | 1848 | 320 | 320 | — | — |

## The comparison that answers the question

| model | in-situ → invasive | recall in-situ | recall invasive | recall stroma | Dice inv vs in-situ | held-out tiles |
| --- | --- | --- | --- | --- | --- | --- |
| served v3 · 112 µm · old rule | 12.6% | 0.827 | 0.839 | 0.915 | 0.892 | 7473 |
| **224 µm · Fix 1 + Fix 2** | 15.3% | 0.816 | 0.849 | 0.905 | 0.900 | 1848 |
| **448 µm · Fix 1 + Fix 2** | 6.8% | 0.890 | 0.877 | 0.883 | 0.962 | 320 |
| **672 µm · Fix 1 + Fix 2** | 9.1% | 0.818 | 0.926 | 0.727 | 0.980 | 76 |

All three rows are the **same architecture** (two frozen ResNet18 bodies concatenated into an MLP) scored at **argmax**, so architecture is held fixed and only the label rule and the field of view move.

**What is not controlled, and it matters:** the three models are scored on *different held-out sets*. The Fix 1 sets contain the solid and comedo DCIS the old rule deleted, plus BACH in-situ from a laboratory v3 barely saw — so they are harder by construction, and a lower score on them is not the same kind of number as a lower score on v3's set. The 224 µm vs 448 µm row pair is the one genuinely controlled comparison in this table: same rule, same sources, same split, only the geometry differs.

## Verdict, with error bars

| model | in-situ called invasive | 95% interval |
| --- | --- | --- |
| served v3 · 112 µm | **157/1246 = 12.6%** | 10.9% – 14.6% |
| **224 µm** | **72/472 = 15.3%** | 12.3% – 18.8% |
| **448 µm** | **5/73 = 6.8%** | 3.0% – 15.1% |
| **672 µm** | **1/11 = 9.1%** | 1.6% – 37.7%  ⚠ too few tiles to interpret |

****224 µm** vs **448 µm**: z = 1.92, two-sided p = 0.055.**

So the 448 µm result is **suggestive and not established**. It is the best point estimate on every metric — in-situ recall, invasive recall, macro recall and Dice all move the same way, which is more persuasive than any single number — but at p = 0.055 it does not clear the conventional bar, and its interval overlaps both of the others.

**And there is a selection effect on top of the sample size.** A 448 µm window needs 896 × 896 pixels of source, so every region smaller than that contributes *nothing* to the 448 µm arm. That is not a random subsample — it systematically drops the small regions, and nothing here measures whether small regions are harder or easier. The 448 µm arm is therefore scored on a different and self-selected population.

**What would settle it**, in order of cost:

1. **Score `CAN_00270_26_H&E` with both heads.** The failure was seen on a slide, not on a held-out tile, and that slide has a pathologist's DCIS contour to compare against. ~30 min of step-8 inference per head, and it is worth more than any of the numbers above.
2. **Give 448 µm more tiles.** Overlapping windows at training time would raise its count several-fold. The exporter uses stride = tile size on purpose — overlapping training tiles are near-duplicates and inflate the apparent dataset — but for a starved class that trade may be worth making deliberately, with the overlap recorded.
3. **The two-branch model.** 224 µm for detail and 448 µm for context, fed to the concat architecture that is already in the codebase. That uses the wide field without needing many independent wide windows, which is the thing the corpus cannot supply — and it is the option the numbers above most support.

## What this does and does not establish

- The two heads here differ from the served one in **both** the label rule and the field of view, so a difference cannot be attributed to either alone. The 224 µm vs 448 µm column pair *is* controlled — same rule, same sources, only the geometry differs.
- Every borrowed label remains BEETLE's placement of epithelium. Fix 1 changes who names the lesion, not who finds the tissue, so the ceiling is still BEETLE's epithelium-versus-stroma accuracy.
- A BRACS region of interest is labelled by its *predominant* lesion, so a DCIS region holding a true focus of invasion now teaches that focus as in-situ. That error is the accepted cost of not deleting the region; it is bounded by how often the consensus is incomplete, and nothing here measures it.
- None of this has been run on `CAN_00270_26_H&E`, the slide the failure was seen on. Step 8 scoring that slide with one of these heads is the next measurement, and it is the one that answers the original question.

### Artefacts

```
original_data/                      the sources, read-only, 28 GB
  bcss/ bracs/{dcis,ic,normal}/ bach/{InSitu,Invasive,Normal}/
tissue_label_generation/data/
  runs/<roi_id>/                    BEETLE's segmentation + manifest.pre-collapse.json
  regions/<tree>/{images,masks}/    region pairs in BCSS label codes
tissue_type_model_training/data/
  224_um/{bcss,beetle,bach}/<class>/*.png   tiles at 224 µm
  224_um/{features,reports}/                vectors and the fitted head
  448um/{bcss,beetle,bach}/<class>/*.png   tiles at 448 µm
  448um/{features,reports}/                vectors and the fitted head
  672um/{bcss,beetle,bach}/<class>/*.png   tiles at 672 µm
  672um/{features,reports}/                vectors and the fitted head
```

