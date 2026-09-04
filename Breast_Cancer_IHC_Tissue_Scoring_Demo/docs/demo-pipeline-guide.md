# Building the Demo: A Step-by-Step Pipeline for IHC Scoring

**Companion to** [breast-cancer-pathology-primer.md](breast-cancer-pathology-primer.md) (the 100 concepts) and *Breast Cancer & Digital Pathology.pdf* (your deep-dive research).

**Purpose of this document.** The primer tells you *what* every term means. This document tells you *in what order to do things*, *why that order and no other*, and *for each step, whether to write classical code, train a model, or download a model someone already trained*. Every claim is tied to a paper you can open.

**What the demo is.** A separate web app — not this client repo — that walks a viewer through one slide, one step at a time. Each step is a screen: the picture that went in, the picture that came out, the numbers, and the knobs. By the last screen the viewer understands exactly how a raw glass-slide scan became the pair `CD44: 50 % positive, intensity 1.5`.

**What this document is scoped to.** The **OncoStem five-marker panel** — CD44, ABCC4, ABCC11, N-cadherin, pan-cadherin — plus an H&E. Not ER/PR/HER2/Ki67. Those appear here only as worked examples of published scoring machinery, never as the deliverable. See *The panel this pipeline serves* at the end of Part 0, and [images-to-scores-mapping.md](images-to-scores-mapping.md) for the data contract.

> **Revised 2 Sep 2026** to match the build plan in
> [`docs/architecture/segmentation-pipeline-plan.md`](docs/architecture/segmentation-pipeline-plan.md).
> Five things changed, and each one changed for a reason worth knowing:
>
> 1. **A licence gate now runs through the whole document.** This ships to a paying client, so
>    non-commercial models and datasets were moved out of the recommendations and into a benchmark
>    tree. That removed GrandQC, DeepLIIF, HoVer-Net/PanNuke, UNI, CONCH, Virchow2 and Phikon-v2 —
>    several of which the earlier draft recommended as defaults.
> 2. **The Rule 2 fork moved from step 6 to step 7.** Deconvolve once; H channel to the model, DAB
>    channel to the measurement. Stain normalisation is now a fallback, not a step.
> 3. **A new ⚙ build block sits between steps 8 and 9** — the one place anything is trained.
> 4. **Step 9 is three classes, not eight**, and runs a ResNet18 head rather than a foundation
>    encoder with a linear probe.
> 5. **Step 3 uses optical density, not HSV saturation.** Measured on this cohort, saturation
>    captured 4 % of the tissue on a pale slide. The numbers are in the step.

---

## Part 0 — The one idea that makes everything else click

A pathology score is a **counting problem wrapped in three filtering problems**.

The final number is always some version of:

```
score = (how many of the RIGHT cells are stained) / (how many of the RIGHT cells there are)
```

Everything in the pipeline exists to make each word in that sentence trustworthy:

| Word in the formula | The problem it creates | The pipeline step that solves it |
| --- | --- | --- |
| "the RIGHT cells" | Which part of the slide counts? | Region segmentation (find invasive tumour) |
| "cells" | Where does one cell end and the next begin? | Nuclei instance segmentation |
| "stained" | How brown is brown enough? | Stain separation + calibration + thresholding |
| "how many" | Simple arithmetic | Aggregation and scoring rules |

If you get the order wrong, you compute a perfectly precise number about the wrong population of cells. That is the single most common failure in this field, and it is why ordering is the subject of this whole document.

**A useful mental model:** the pipeline is a funnel that goes from *big and dumb* to *small and smart*.

```
Whole slide (billions of pixels, mostly glass)
   |  throw away glass                              <- cheap, classical
Tissue only
   |  throw away bad tissue (blur, folds, pen)      <- cheap, classical + small model
Usable tissue
   |  throw away wrong tissue types                 <- expensive, learned
   |  (fat, stroma, DCIS, normal duct, necrosis)
Invasive tumour region only
   |  find individual cells                         <- expensive, pre-trained model
Cells
   |  measure stain per cell                        <- cheap, classical colour maths
Per-cell measurements
   |  apply the pathologist's rulebook              <- cheap, pure logic
One score
```

Notice: **every expensive step runs on a smaller area than the step before it.** That is not an accident, it is the reason for the order. Running nuclei segmentation on a whole slide before you have narrowed the region is how you burn a GPU-day producing an answer you then throw away.

### The panel this pipeline serves

Every step below is written against these five antibodies and no others. The single trailing
letter in a slide's filename **is** the antibody.

| Letter | Marker | Compartment the brown must be in | Code path | Observed % range |
| --- | --- | --- | --- | --- |
| **A** | **CD44** | **Membrane** | membrane ring + completeness | 5–85 % |
| **F** | **ABCC4** | **Membrane** | membrane ring + completeness | 35–65 % |
| **R** | **ABCC11** (MRP8) | **Membrane** | membrane ring + completeness | 35–65 % |
| **U** | **N-cadherin** (CDH2) | **Cytoplasm** | cytoplasm band, no completeness | 80 % (near-constant) |
| **W** | **Pan-cadherin** | **Cytoplasm** | cytoplasm band, no completeness | 75–80 % |
| H&E | *none* | n/a | not scored; orientation only | n/a |

**Two of the five take a different path from the other three.** N-cadherin and pan-cadherin are
**cytoplasmic, not membranous** — this was wrong in earlier drafts of the code and it is the
easiest mistake to re-introduce, because almost every worked example in the literature is a
membrane marker. It changes step 13 (which compartment to grow), step 14 (whether completeness
means anything) and step 15 (which cut points apply).

**The deliverable is two numbers per marker, ten per case:** percent positive (0–100, reported in
multiples of 5) and intensity on OncoStem's **0–2** band scale (0, 0.5, 1, 1.5, 1.75, 2).
**Not an H-score.** See step 16.

**The % ranges above are descriptive, not thresholds.** They are the spread OncoStem has seen over
years of cases. A result outside a range is a flag for human review, not an error, and no
positivity cut-off is applied at the scoring stage.

---

## Part 1 — The pipeline at a glance

Read this table first. The rest of the document is one section per row. The **Per-marker?** column
is the one earlier drafts did not have: it says whether a step is identical code for all five
antibodies, or forks by marker.

| # | Step | Plain-English job | Approach | Train? | Per-marker? |
| --- | --- | --- | --- | --- | --- |
| 1 | **Read the slide** | Open the giant scan, pick a zoom level | Library (OpenSlide / tiffslide) | No | Same for all 5 |
| 2 | **Quality control** | Reject blurry / pen-marked / folded areas | Classical, **in-house** (`artefacts.py`) — GrandQC is non-commercial | No | Same for all 5 |
| 3 | **Tissue mask** | Separate tissue from empty glass | Classical (**optical density** — *not* saturation) | No | Same for all 5 |
| 4 | **White calibration** | Learn what "no stain" looks like on *this* slide | Classical (sample the glass) | No | Same code, **per-slide values** |
| 5 | **Optical density** | Convert colour into "how much stain" | Classical (Beer–Lambert) | No | Same for all 5 |
| 6 | **Stain normalisation** *(fallback only — normally skipped)* | Make slides look alike so models generalise | Classical (Macenko) | No | Same for all 5 |
| 7 | **Colour deconvolution** ★ *(the fork — feeds **both** branches)* | Split into a blue channel and a brown channel | Classical (Ruifrok & Johnston, **fixed vectors**) | No | Same code, same vectors |
| 8 | **Tiling** | Cut the slide into bite-sized patches | Plumbing | No | Same for all 5 |
| — | **⚙ Build block — train the region model** | *One-time, offline. Not part of the per-slide run* | ResNet18 head on BCSS + AICAN | **Yes — one small head** | Built once, serves all 5 |
| 9 | **Tissue-type segmentation** | Label every patch: **invasive** / **non-invasive epithelium** / non-epithelium | The head from the build block | No — inference only | Same model, **fed the H channel** so all 5 stains look alike |
| 10 | **Build the ROI mask** | Stitch patches into one clean "invasive tumour" region | Classical post-processing | No | Same for all 5 |
| 11 | **Nuclei segmentation** | Outline every individual nucleus *inside the ROI* | **Pre-trained** (InstanSeg / Cellpose — both permissive) | No | Same code; **hardest on U/W** |
| 12 | **Cell typing** | Tumour cell vs lymphocyte vs stromal cell | Classical on the H channel + region class | No | Same for all 5 |
| 13 | **Compartments** | Grow the compartment the marker lives in | Classical morphology | No | **Forks: A/F/R ring, U/W band** |
| 14 | **Per-cell measurement** | How much brown is in *this cell's* compartment | Classical | No | **Forks: completeness only for A/F/R** |
| 15 | **Intensity binning** | Turn OD into intensity levels | Classical, calibrated thresholds | No | **Five cut-point sets, one per antibody** |
| 16 | **Aggregate** | Percent positive + intensity band | Pure logic | No | **Same formula, per-marker banding** |
| 17 | **Validate** | Prove it agrees with pathologists | Statistics | No | **Report stratified by marker** |

**The headline, corrected.** Out of 17 steps, **zero require training a model from scratch.** There
is exactly **one** training block, it is offline, it fits a three-class head on top of an ImageNet
ResNet18, and it takes minutes on a CPU. One step uses somebody else's pretrained model (11), one
uses a pretrained model as an offline *teacher* (the build block). The remaining fifteen are
classical image processing and arithmetic.

That reframing matters for the schedule as much as for the demo. The riskiest question becomes
*"does a head fitted on public breast labels transfer to our IHC slides?"*, which you can answer in
a couple of days, instead of *"can we annotate and train a segmentation network?"*, which is a
six-week bet.

> **Licence gate — read before choosing anything.** This ships to a paying client, so a model or
> dataset that forbids commercial use cannot enter `backend/app/`. That rules out a lot of the
> best-known work in this field: GrandQC, DeepLIIF, HoVer-Net/PanNuke, TIGER, BEETLE, UNI, CONCH,
> Virchow2 and Phikon-v2 are all non-commercial. What survives, and what this guide now
> recommends throughout: **BCSS** (CC0), **AICAN breast-epithelium-segmentation** (MIT),
> **torchvision ImageNet weights** (BSD-3), **InstanSeg** (Apache-2.0), **Cellpose** (BSD-3) and
> **VALIS** (MIT). The non-commercial tools remain useful as internal benchmarks in a separate
> tree that the shipped package cannot import.

**Of the 17 steps, only 5 ever need to know which antibody they are looking at** — 13, 14, 15, 16
and 17. Steps 1–12 are marker-blind. Build them once, run them five times.

---

## Part 2 — The six ordering rules (read before writing any code)

You asked specifically *when* to do each thing. Here are the six decisions people get wrong, stated as rules.

### Rule 1 — Remove fat at the **tissue-type** step, not at the tissue-mask step

**The tempting mistake.** Fat looks white and empty, so during tissue detection you threshold out pale regions and call it "removing fat".

**Why it is wrong.** At the tissue-mask stage you only know "tissue vs glass". Fat *is* tissue. If you drop it there, you also drop pale tumour, glandular secretions, and any washed-out area — and you keep no record of what you dropped. Worse, adipose tissue is not always pale: at low zoom, a fatty region and a torn tissue edge look identical.

**The rule.** The tissue mask keeps **everything that is tissue**, fat included. Fat gets removed later, at step 9, as an explicit *class* — the model outputs `fat` and you exclude that class. Now it is auditable: you can show the viewer a fat overlay and say "we removed exactly this, and here is why".

Public data backs this up. The **BCSS** dataset annotates `fat` as one of 16 tissue classes alongside `tumor`, `stroma`, `dcis`, `necrosis` — precisely because fat is a semantic class, not a brightness threshold.
→ [BCSS, Grand Challenge](https://bcsegmentation.grand-challenge.org/) · [NuCLS & BCSS, GigaScience 2022](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac037/6586817)

*The honest fallback.* If you have no trained model yet, "large pale round vacuoles with almost no nuclei" is a reasonable placeholder — that is what the client repo does today, and it correctly reports confidence `0.0` to flag itself as a placeholder (primer item 86). Show that as the "before" screen and the trained model as the "after". The contrast is one of the best teaching moments in the demo.

### Rule 2 — Deconvolve once, then fork: **H to the model, DAB to the measurement**

This is the subtlest point in the pipeline and the one most demos get wrong.

Two completely different things are both called "normalisation":

| | **Stain normalisation** | **Stain calibration** |
| --- | --- | --- |
| What it does | Rewrites the image's colours to match a reference slide | Records what "zero stain" and "full stain" mean in physical units |
| When | Before feeding a **deep learning model** | Before **measuring** anything |
| Effect on your numbers | **Destroys** them — you overwrote the intensity you wanted to measure | Preserves them, and makes them comparable across slides |
| Methods | Macenko, Vahadane, Reinhard | White-point estimation from glass, fixed reference vectors |

**The rule, in the shape this pipeline actually uses.** The fork is **step 7**, not step 6.
Deconvolve once, then send **the haematoxylin channel to the model** and **the DAB channel to the
measurement**. Both halves come out of the same un-normalised, calibrated deconvolution, so nothing
downstream can accidentally measure a rewritten pixel.

```
              tile (raw, calibrated)
                       |
              Step 7 — deconvolve
                 /            \
        H channel              DAB channel
             |                      |
    Step 9 — region model    Steps 14-15 — measure
```

Draw this fork as a Y-shape in the demo. It is the single most instructive picture in the whole app.

**Why normalisation then mostly disappears.** Step 6 existed to make five differently-stained
slides look alike to a model. Once the model is fed the H channel only, the largest source of
that variation — the brown itself — is gone before normalisation could act on it, and the residual
haematoxylin variation is handled by **HED colour augmentation at training time**, which costs
nothing at inference. Keep Macenko on the shelf as a fallback: if the region behaves differently on
an H&E slide than on its serial IHC sections, add normalisation **on the H channel only** and
re-measure before blaming the model.

*Why normalisation helps models.* Slides stained on different days, in different labs, on different scanners look visibly different. A CNN trained on one lab's pink underperforms on another's. Normalisation removes that nuisance variation.
→ [Macenko et al., ISBI 2009](https://ieeexplore.ieee.org/document/5193250) · [Vahadane et al., IEEE TMI 2016](https://ieeexplore.ieee.org/document/7460968) · [Reinhard et al., IEEE CG&A 2001](https://ieeexplore.ieee.org/document/946629) · [Multi-centre benchmark, Sci Rep](https://www.nature.com/articles/s41598-026-40943-3) · [arXiv 2506.19106](https://arxiv.org/abs/2506.19106) · [Effect on IDC grading, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10665422/) · [StainNet, a fast learned alternative](https://pmc.ncbi.nlm.nih.gov/articles/PMC8602577/)

*Why it destroys measurement.* If you rescale every slide so its darkest brown becomes "maximum brown", a genuinely weak 1+ slide and a genuinely strong 3+ slide both come out looking 3+. You have normalised away the diagnosis. This is exactly the per-slide-percentile problem the client repo's EPIC-010 moved away from — primer item 79.

### Rule 3 — Separate the stains **before** you threshold, never after

**The mistake.** Thresholding on "brownness" in RGB, or on the R−B difference.

**Why it is wrong.** Hematoxylin (blue) and DAB (brown) are *mixed* in every pixel. A dark blue nucleus under faint brown looks, in RGB, a lot like a moderately brown nucleus. You cannot threshold a mixture — you must un-mix it first.

**The rule.** RGB → optical density → colour deconvolution → **then** threshold, and threshold the DAB channel only.
→ [Ruifrok & Johnston, Anal Quant Cytol Histol 2001](https://pubmed.ncbi.nlm.nih.gov/11531144/) · [full PDF](https://helios2.mi.parisdescartes.fr/~lomn/Cours/CV/BME/HistoPatho/Quantification_of_histochemical_staining.pdf)

### Rule 4 — Segment the **region** before you segment the **cells**

**The mistake.** Run nuclei segmentation over the whole slide, then filter cells by region afterwards.

**Why it is wrong.** Two reasons. *Cost* — nuclei segmentation on a full 40× whole-slide image is a GPU-hours job; on a 5% invasive focus it is seconds. *Correctness* — nuclei models are trained on cellular tissue and behave unpredictably on fat and necrosis, producing junk detections that pollute your denominator the moment any filtering step leaks.

**The rule.** Region mask first (step 10), then nuclei **inside the mask only** (step 11).

### Rule 5 — Gate scoring to **invasive** tumour, not to "tumour"

DCIS (cancer still trapped inside the duct) and invasive carcinoma both look like "dense sheets of abnormal nuclei". A density-based heuristic cannot tell them apart — the primer flags this at item 87, and this client repo's [invasive.py](../../backend/app/scoring/invasive.py) says so in its own docstring.

But the clinical meaning differs completely, and the biomarker score is *defined on the invasive component*. Including DCIS silently changes the answer.

**The rule.** The segmentation model in step 9 must have `dcis` and `tumor` as **separate output classes**, and only invasive `tumor` enters the denominator.

> ⚠ **Separate classes are necessary but not sufficient — and this was measured, not assumed.** BCSS
> raw *does* keep `dcis` separate from `tumor`, and it is still only **0.050 % of pixels, present in
> 1 of 150 regions**, all in a training institution. A class the training data does not contain
> cannot be learned however cleanly the ontology separates it. Census:
> [`approach-1-training-plan.md`](approach-1-training-plan.md) Part 2. Fix:
> [`approach-4a-training-plan.md`](approach-4a-training-plan.md).
→ [Cruz-Roa et al., invasive tumour extent, Sci Rep 2017](https://www.nature.com/articles/srep46450) · [DCIS vs IDC classification, PubMed 2022](https://pubmed.ncbi.nlm.nih.gov/35076741/) · [Invasive carcinoma segmentation on IHC WSIs, Cancers 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10778369/) · [MS-ResMTUNet, PMC 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11636800/)

### Rule 6 — Measure in the compartment the marker actually lives in

A marker sits somewhere specific inside the cell, and the scoring rule follows the location
(primer items 44–45). **This is the rule that forks the pipeline**, so the table below is the real
panel, not the textbook one.

| Letter | Marker | Lives in | You must measure | Extra rule |
| --- | --- | --- | --- | --- |
| **A** | **CD44** | Cell membrane | A ring around the nucleus | Ring **completeness** is a second measurement |
| **F** | **ABCC4** | Cell membrane | A ring around the nucleus | Ring completeness; efflux pumps are membrane-dominant |
| **R** | **ABCC11** | Cell membrane | A ring around the nucleus | Ring completeness |
| **U** | **N-cadherin** | **Cytoplasm** | The band between nucleus and cell edge | **No completeness** — see step 14 |
| **W** | **Pan-cadherin** | **Cytoplasm** | The band between nucleus and cell edge | **No completeness** |

Scoring CD44 as "average brown inside the nucleus" produces a number that is not so much wrong as
*meaningless* — the protein is not there. The reverse error is just as real and much less obvious:
applying a 36-bin membrane **completeness** test to N-cadherin measures the circularity of
something that has no circumference to be complete around, and it will quietly suppress genuinely
positive cells.

**Two textbook markers, kept as worked examples only.** HER2 (membrane, where ASCO/CAP wrote
completeness into the guideline) and ER / PR / Ki67 (nuclear, the source of Allred and the
proliferation index) appear later in this document because they are the vocabulary this field
speaks and steps 14–16 borrow their machinery. **Neither is in the panel.** If you find yourself
emitting a HER2 0/1+/2+/3+ call or an Allred score as an *output*, you have drifted.

→ [ASCO/CAP HER2 rules, summarised](https://spj.science.org/doi/10.34133/bmef.0048) · [CD44/CD24 membrane scoring, PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165253) · [CD44 IHC in breast carcinoma, PMC 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12382596/) · [N-cadherin and EMT in breast cancer, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5340734/) · [ABCC11 / MRP8 in breast cancer, PubMed](https://pubmed.ncbi.nlm.nih.gov/16467088/)

---

## Part 3 — The steps in detail

Each step below has the same five parts: **What** (plain English) · **Why here** (ordering justification) · **How** (approach + whether to train) · **Show in the demo** (the screen) · **Papers**.

---

### Step 1 — Read the slide

**What.** A whole-slide image is not a photo, it is a pyramid: the same slide stored at many zoom levels, cut into tiles, so software can pull just the piece it needs. A single slide is 1–10 GB and can be 100,000 × 100,000 pixels. You never load it all at once.

**Why here.** Everything downstream depends on one decision made here: *what magnification do I work at?* This choice is physical, not arbitrary. It is expressed in **microns per pixel (mpp)**:

| mpp | Common name | Use it for |
| --- | --- | --- |
| ~1.0–2.0 | 5–10× | Tissue mask, QC, region context |
| ~0.5 | 20× | Tissue-type segmentation, most models |
| ~0.25 | 40× | Nuclei segmentation, membrane rings |

Always convert between levels using mpp, never using "level 2". Different scanners have different native mpp, so hard-coding a level number silently changes the physical scale of everything.

**How.** Classical, library work. `openslide-python` for `.svs`/`.ndpi`/`.mrxs`; `tifffile` for pyramidal TIFF; `DeepZoom` for serving tiles to a browser. No training.

**Show in the demo.** A pan-and-zoom viewer (OpenSeadragon) with a corner readout: current mpp, current level, tile count. Let the viewer scrub the zoom slider and watch the tile grid change. This makes "gigapixel" concrete instead of abstract.

**Papers / tools.** [OpenSlide, J Pathol Inform 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3815078/) · primer items 58–60

---

### Step 2 — Quality control

**What.** Slides have defects: out-of-focus areas, tissue folds, air bubbles, pen marks the pathologist drew, dust, coverslip edges. These produce confident, wrong answers downstream.

**Why here — first, before anything else.** QC is a filter on *input*. If you run it later you will have already wasted compute on garbage regions, and — much worse — a blurred region can produce a plausible-looking tumour prediction that quietly enters your denominator. QC before tissue masking, because pen marks are dark and saturated and will otherwise be *included* as "tissue".

**How — and this one is ours to write.** GrandQC is the best tool here and it is **CC BY-NC-SA**,
so it cannot ship. Build `backend/app/ingestion/artefacts.py` instead. Every detector is classical,
each is a few lines, and together they cover what actually appears on these 36 slides:

| Artefact | Detector | Measured on this cohort |
| --- | --- | --- |
| **Saturated chromogen** | DAB optical density above 1.0 | Present on **12 of 30** IHC slides |
| **Tissue folds** | Very dark **and** low local variance in the H channel | Case 00303 worst affected |
| **Pen marks** | High saturation, non-tissue hue, large connected components | Suspected on the 00865 H&E |
| **Defocus** | Low local gradient energy (Tenengrad) per tile | — |

Run **GrandQC in the benchmark tree** to audit this in-house mask — that is a legitimate
non-commercial use and it tells you what you are missing. Use **HistoQC**'s feature maps for the
demo explanation.

**Show in the demo.** Thumbnail with a red artefact overlay, plus a bar chart of "% of tissue rejected, by artefact type". Add a toggle: *"score with QC on / off"* and display the two final scores side by side. Watching the number move because a blurred corner was included is the most persuasive single thing in the demo.

**Papers.** [GrandQC, Nature Communications 2024](https://www.nature.com/articles/s41467-024-54769-y) · [HistoQC vs PathProfiler comparison](https://www.sciencedirect.com/science/article/pii/S1120179726000323) · [Explicit artefact augmentation, Sci Rep 2024](https://www.nature.com/articles/s41598-024-68667-2) · [Human-in-the-loop QC, arXiv 2409.19587](https://arxiv.org/pdf/2409.19587)

---

### Step 3 — Tissue mask

**What.** Most of a slide is empty glass. Find the tissue.

**Why here.** After QC (so pen marks are already excluded), before everything else (so nothing downstream wastes time on glass). And **before** any tissue-type work, because "is this tissue at all" is a strictly easier question than "what kind of tissue is this", and answering the easy question first shrinks the input to the hard one by 60–90%.

**How.** Pure classical, no training, and it works essentially always:

0. **Remove the canvas padding first.** These Morphle files are a fixed 63488 or 126976 px grid
   with tissue in one quadrant and the rest written by nobody. Padding is *not* background — it is
   absence of data. Separate it from glass by **flatness**, not brightness: padding is memset to a
   single value with zero local variance, glass carries sensor noise. Measured here, padding sits
   at grey 146 and glass at 191–195, so a brightness cut at 200 eats the glass.
1. Downsample to ~2 mpp (a thumbnail is enough).
2. Convert to **optical density** and threshold that. **Do not use HSV saturation.**
3. Apply **Otsu's method** on the OD histogram to pick the threshold automatically.
4. Clean up with **morphology**: closing to fill small holes, opening to drop specks, then keep large connected components.

> **Why saturation is wrong here, with numbers.** Saturation measures how *stained* a slide is, not
> whether tissue is present. Against a plain optical-density rule on the same pixels:
>
> | Slide | Saturation rule | OD rule | Captured |
> | --- | --- | --- | --- |
> | 00865 / R (well stained) | 82.6 mm² | 84.8 mm² | 97 % |
> | 00270 / A | 45.9 mm² | 101.2 mm² | 45 % |
> | 00267 / A | 5.0 mm² | 39.3 mm² | 13 % |
> | 00865 / A | 0.7 mm² | 17.5 mm² | **4 %** |
>
> They agree on well-stained slides and collapse on pale ones. That figure is the **denominator of
> percent-positive**, so the error inflates the score exactly where expression is lowest — the case
> least likely to look wrong. This is the single most instructive "we got it wrong and fixed it"
> screen available to the demo.

**Do not remove fat here** (Rule 1). Fat is tissue and stays in the mask.

**Show in the demo.** Four panels: original thumbnail → saturation channel (greyscale) → histogram with the Otsu line drawn on it → binary mask overlay. Add a manual threshold slider next to the Otsu value so the viewer can see that Otsu lands where a human would put it.

**Papers.** [Otsu, IEEE Trans SMC 1979](https://ieeexplore.ieee.org/document/4310076) · primer items 61, 66–71

---

### Step 4 — White calibration

**What.** Sample the empty glass to learn what "zero stain" looks like on *this* slide, on *this* scanner, on *this* day. That value is called **I₀** (incident light).

**Why here.** This must come before optical density (step 5), because OD is defined *relative* to I₀. And it must be **per slide** — that is the entire point. Scanners drift, bulbs age, staining batches vary. Two slides with identical biology can come off two scanners with different baseline whites.

This step is what makes "absolute stain scale" possible (primer items 79–80). Without it you are forced back onto per-slide percentile normalisation, which — see Rule 2 — destroys cross-slide comparability.

**How.** Classical, no training. Take the tissue mask from step 3, invert it to get pure glass, drop the outermost border, and take a high percentile (e.g. the 95th) of each RGB channel. Store `I₀ = (R₀, G₀, B₀)` in the slide's metadata. A robust variant fits a smooth low-order surface across the slide to also correct uneven illumination (vignetting).

**Show in the demo.** Thumbnail with the sampled glass patches marked in green, plus the resulting I₀ swatch. Then show two different slides side by side with their two different I₀ values — that is the "why per slide" argument in one picture.

**Papers.** Primer items 76, 79–80 · [Ruifrok & Johnston 2001](https://pubmed.ncbi.nlm.nih.gov/11531144/)

---

### Step 5 — Optical density

**What.** Convert colour into a physical quantity: *how much stain is in this pixel*.

**Why here.** Immediately after calibration, and before both the model branch and the measurement branch, because both are cleaner in OD space.

The reason this is not just a colour trick is the **Beer–Lambert law**: light absorbance is proportional to the concentration of the absorbing substance. RGB values are *multiplicative* in transmitted light — two stains stacked multiply their transmissions. Taking the logarithm turns that into *addition*, and addition is something linear algebra can un-mix. This is the single mathematical fact that makes colour deconvolution valid.

**How.** Classical, one line:

```
OD = -log10( I / I0 )
```

per channel, where `I` is the observed pixel and `I₀` is your calibrated white from step 4. Clip `I` to a small floor first so pure black does not produce infinity.

**Show in the demo.** Side-by-side: RGB tile → OD tile (rendered as a heatmap) → a scatter plot of pixels in OD space showing two visible arms. Those two arms *are* the two stains — seeing them makes the next step obvious rather than magical.

**Papers.** Primer items 74, 76 · [Ruifrok & Johnston 2001](https://pubmed.ncbi.nlm.nih.gov/11531144/)

---

### Step 6 — Stain normalisation *(fallback only — you will normally skip this)*

> **⚠ Numbering: this step is not built, so the codebase is one lower from here on.**
> This section is skipped by design (see the status note below and Rule 2), and the
> implementation does not reserve a slot for it. So the guide's step 7 is the code's
> **step 6**, the guide's step 8 is the code's **step 7**, and so on to the guide's
> step 17 / the code's step 16 — which is why the pipeline is sixteen steps in
> `app/data/pipeline_steps.py` and seventeen headings here. When reading a step
> package's README, trust the `id` (`colour-deconvolution`, `tiling`, …) rather than
> the number.

**What.** Rewrite the tile's colours so it looks like it was stained in the same lab as your training data.

> **Status: not in the default pipeline.** The fork moved to step 7 (see Rule 2). Because the region
> model is fed the **haematoxylin channel** rather than RGB, the brown that made the five antibodies
> look different is already gone, and the remaining variation is absorbed by **HED augmentation
> during the build block**. Keep this step implemented and switchable, and turn it on only if
> validation check V2 shows the region behaving differently on an H&E slide than on its serial IHC
> sections. If you do turn it on, normalise **the H channel only**, never the DAB channel and never
> anything on the measurement branch.

**Why it is a branch, not a step.** Everything that feeds a **learned model** may pass through here. Everything that feeds a **measurement** must not. If you draw one linear pipeline you will get this wrong.

**How.** Classical, no training, three choices in increasing order of quality and cost:

| Method | Idea | Trade-off |
| --- | --- | --- |
| **Reinhard** | Match mean and standard deviation of colour in LAB space | Fastest, crudest; poor on the weaker stain |
| **Macenko** | Estimate each image's stain vectors from the OD point cloud (SVD), map to a reference | Good balance, the common default; fails if the stain-vector estimate fails |
| **Vahadane** | Sparse non-negative matrix factorisation of the OD image | Best structure preservation, slowest |

There are also learned normalisers (StainNet, CycleGAN-based). Skip them for a demo — they add a second black box for little gain and are much harder to explain on a screen.

**A cheaper alternative worth knowing:** instead of normalising, augment. Randomly perturb stain vectors during training so the model learns to ignore colour. In practice, strong stain augmentation often beats normalisation, and it costs nothing at inference time.

**Show in the demo.** A 3×3 grid: three slides from different "labs" (simulate by colour-shifting) down the rows, and raw / Macenko / Vahadane across the columns. The rows start visibly different and end visibly similar. Then a caption: *"and this is exactly why we never do this before measuring"*.

**Papers.** [Macenko, ISBI 2009](https://ieeexplore.ieee.org/document/5193250) · [Vahadane, IEEE TMI 2016](https://ieeexplore.ieee.org/document/7460968) · [Reinhard, 2001](https://ieeexplore.ieee.org/document/946629) · [Comprehensive review and experimental comparison, ResearchGate](https://www.researchgate.net/publication/373610633_Stain_normalization_methods_for_histopathology_image_analysis_A_comprehensive_review_and_experimental_comparison) · [Multi-centre benchmark, arXiv 2506.19106](https://arxiv.org/abs/2506.19106) · [StainNet, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8602577/) · [Slideflow, arXiv 2304.04142](https://arxiv.org/pdf/2304.04142)

---

### Step 7 — Colour deconvolution ★ *(the fork — feeds **both** branches)*

**What.** Split the mixed image into one channel per stain: a "how much hematoxylin" image and a "how much DAB" image.

**Why here, and why it is now the most important classical step in the pipeline.** After OD (you
need the log-transform for the maths to be linear), before any thresholding (Rule 3), and on
**un-normalised** pixels (Rule 2). This one step does double duty:

- the **DAB channel** goes to the measurement branch (steps 14–15), which is the classical use;
- the **H channel** goes to the region model (step 9), which is what makes **one** model work on
  the H&E and all five IHC slides. Without this, you need either a separate IHC model or a
  registration step, and both are worse.

Everything downstream depends on train and inference calling **the same deconvolution function**.
Put it in `backend/app/scoring/stains.py` and have the tile exporter and the runtime both import it
— do not let a training script reach for `skimage.color.rgb2hed` on its own, or the two will drift
and the IHC slides will score nothing like the H&E.

**How.** Classical, no training. You have a 3×3 stain matrix whose rows are the reference OD colour vectors for each stain. For H-DAB, the Ruifrok & Johnston reference vectors are the standard, and `skimage.color.rgb2hed` implements exactly this. Invert the matrix, multiply the OD pixel by it, and you get per-stain concentrations.

**The critical choice — fixed vectors or estimated vectors?**

| | Fixed reference vectors | Per-image estimated vectors (Macenko-style) |
| --- | --- | --- |
| Cross-slide comparability | **Yes** — "0.4 DAB" means the same thing everywhere | No — each slide gets its own scale |
| Robust to odd staining | Less | More |
| Right for **measurement** | **Use this** | Do not use this |
| Right for visual display | Either | Either |

Since your goal is a comparable score, use **fixed vectors**. If you want to be careful, calibrate the vectors once per scanner/protocol on control slides and then freeze them — that is the best of both.

**Show in the demo.** One IHC tile → three panels: hematoxylin channel, DAB channel, residual. Then a toggle between "fixed vectors" and "estimated per-image vectors" showing that the DAB channel changes — and therefore so does the score. Put the two scores on screen.

**Papers.** [Ruifrok & Johnston 2001](https://pubmed.ncbi.nlm.nih.gov/11531144/) · [PDF](https://helios2.mi.parisdescartes.fr/~lomn/Cours/CV/BME/HistoPatho/Quantification_of_histochemical_staining.pdf) · [The maths, explained on image.sc](https://forum.image.sc/t/on-the-math-behind-colour-deconvolution-ruifrok-and-johnston-2001/66325) · primer items 75–78

---

### Step 8 — Tiling

**What.** Cut the tissue into patches the model can eat.

**Why here.** After the tissue mask, so you only generate tiles that contain tissue — typically a 5–10× reduction in tile count and therefore in compute. After QC, so you skip artefact tiles.

**How.** Plumbing, but three parameters matter and each has a right answer:

- **Size**: 256×256 or 512×512 pixels. Model-dependent, not biology-dependent.
- **Magnification**: 20× (~0.5 mpp) for tissue-type work. At 20× a 256-pixel tile covers ~128 µm — enough to see gland architecture, which is what distinguishes DCIS from invasive. At 40× you see cells but lose the architecture; at 5× you see architecture but lose the cells.
- **Overlap**: 25–50% for segmentation. Predictions are unreliable at tile edges because the model lacks context there. Overlap lets you average edges away and prevents visible seams in the stitched mask.

**Show in the demo.** The tile grid drawn on the slide, with the tissue-only tiles highlighted and a counter: "1,240,000 possible tiles → 96,000 tissue tiles → 91,200 after QC". Big numbers dropping fast is a satisfying screen.

---

## ⚙ Build block — Train the region model

> **This is the one place in the whole pipeline where you train anything.** It is **offline** and
> **one-time**: it does not run when a slide is scored. It sits here because it needs tiles
> (step 8) and produces the model step 9 uses. Run it once, pin the weights, and the per-slide
> pipeline never touches it again.
>
> **Cost:** two to three days of work, minutes of CPU compute, no GPU.
> **Full task list:** `docs/architecture/segmentation-pipeline-plan.md`, sections 3–5.

### What you are training, and what you are not

You are **not** training a segmentation network. You are fitting a **three-class head** on top of an
ImageNet ResNet18 whose body stays frozen. That is a convex-ish fit with a handful of
hyper-parameters, it runs on a laptop CPU in minutes, and you can retrain it live in front of an
audience.

| Class | Plain meaning | Fate |
| --- | --- | --- |
| `2` **invasive epithelium** | Cancer that has broken out of the ducts | **The only thing scored** |
| `1` **non-invasive epithelium** | DCIS, LCIS, normal ducts and lobules | Excluded (Rule 5) |
| `0` **non-epithelium** | Stroma, fat, inflammation, necrosis | Excluded (Rule 1) |

Three classes, not eight. Fat, stroma and necrosis do not need to be told apart from each other —
they all get excluded. Collapsing them puts every training example you have behind the **one**
boundary that matters: invasive versus in-situ.

### Input B1 — BCSS raw labels *(CC0, ships)*

151 breast regions from TCGA-BRCA with pixel labels, released **CC0**, so anything fitted on them
is clean to ship.

> **The trap.** There is a widely-used "5-class" version of BCSS — tumour / stroma / inflammatory /
> necrosis / other — and it **merges DCIS into tumour**. That is the one merge this project cannot
> accept, and it is the default in most tutorials and in TIAToolbox's `fcn_resnet50_unet-bcss`. Use
> the **raw** codes from `meta/gtruth_codes.tsv`, where `dcis` is its own label. Read that file
> before you write the mapping.

| Our class | BCSS labels |
| --- | --- |
| `2` invasive | `tumor`, `angioinvasion` |
| `1` non-invasive | `dcis`, `normal_acinus_or_duct` |
| `0` non-epithelium | `stroma`, `lymphocytic_infiltrate`, `necrosis_or_debris`, `fat`, `blood`, `plasma_cells`, `other_immune_infiltrate`, `mucoid_material`, `blood_vessel`, `lymphatics`, `nerve`, `skin_adnexa`, `glandular_secretions`, `metaplasia_NOS`, `other` |
| ignore | `exclude`, `undetermined` |

### Input B2 — AICAN pseudo-labels *(MIT, ships)*

BCSS teaches the model what breast cancer looks like. It does **not** teach it what the Morphle
scanner and OncoStem's staining look like. **AICAN breast-epithelium-segmentation** closes that gap:
a pretrained, MIT-licensed model that already outputs benign / in-situ / invasive epithelium.

```powershell
pip install pyfast
runPipeline --datahub breast-epithelium-segmentation --file <your H&E slide>
```

Run it on ~40 sampled 2048² regions per case (about 25 min/case), treat its output as ground truth,
and tile it exactly like BCSS. This is **not circular**: AICAN's own labels came from cytokeratin
restaining, an objective source, on a different cohort.

Two things to know before relying on it. First, **check the weights licence, not just the repo
badge** — one hour, and the whole step rests on it. Second, its published weak spot is in-situ and
benign classification, which is our problem area, so judge its output on whether it **withholds**
DCIS, not on whether it finds tumour.

> **If pyFAST will not run on your CPU** (FAST needs an OpenCL runtime): try Intel's CPU OpenCL,
> then stop. **This input is optional.** Train on BCSS alone and carry on — you lose the scanner
> adaptation, not the pipeline. Do not spend three days on an installer.

### Making tiles

Both inputs go through the identical exporter, so nothing can drift:

1. Resample to **0.5 µm/px** (BCSS ships at 0.25, so downsample 2×).
2. Convert to the **H channel** using `stains.py` — the same function step 7 calls at inference.
3. Clip OD to `[0, 1.5]`, divide by 1.5, replicate to 3 channels.
4. Cut **224 × 224** tiles, stride 224.
5. Label by majority vote, and **throw away the ambiguous ones**:

```
usable = fraction of pixels with a non-ignored label
if   usable < 0.70:              drop
elif invasive_frac    >= 0.50:   label = 2
elif noninvasive_frac >= 0.50:   label = 1
elif nonepi_frac      >= 0.80:   label = 0
else:                            drop      # too mixed to learn from
```

**Why 224 px at 0.5 µm/px:** that is 112 µm, roughly nine cells across — enough to see whether
epithelium sits inside a duct or is infiltrating, which is the only signal that separates DCIS from
invasive. Smaller tiles cannot see it, and that is why nuclear density never worked.

### The fit

```python
import torchvision, torch
m = torchvision.models.resnet18(weights="IMAGENET1K_V1")   # BSD-3, safe to ship
m.fc = torch.nn.Linear(512, 3)
for p in m.parameters():      p.requires_grad = False
for p in m.fc.parameters():   p.requires_grad = True
```

- **Class weights** in the cross-entropy loss, inversely proportional to frequency. Without them the
  model answers "non-epithelium" every time, looks 80 % accurate, and is useless.
- **HED colour augmentation**, plus flips and 90° rotations. The HED jitter is the one that matters
  — it is what buys stain tolerance and lets you skip step 6.
- **Leave-one-source-slide-out**, never a random tile split. Tiles from one slide are highly
  correlated; a random split reports a beautiful number that means nothing.
- Frozen body first (minutes). Unfreeze `layer4` only if the frozen head falls short (1–2 hours).

### Done when

- [ ] `models/invasive_tile_v1.pt` exists and its **SHA-256 is recorded**.
- [ ] A 3×3 confusion matrix is printed on held-out slides.
- [ ] The **invasive ↔ non-invasive cell is reported on its own**, never folded into an average.
      That one number is the project.
- [ ] Weights are **vendored and pinned** — never fetched from a model hub at runtime.

**Target:** Dice ≥ 0.65 on invasive vs non-invasive. The anchor is BEETLE's own 5-model nnU-Net,
which scores **0.65 on non-invasive epithelium** across unseen scanners — 0.83 on its own data, 0.56
at its worst centre — on a far bigger, purpose-built, multi-centre dataset.

> ⚠ **Earlier drafts of this guide quoted ~0.92 here. That was the same model's *overall* Dice,
> inflated by its `other` class at 0.97, which is most of the pixels. Do not quote it.** The
> per-class table is in [`approach-4a-training-plan.md`](approach-4a-training-plan.md) Part 2.

**Show in the demo.** The confusion matrix with the DCIS-vs-invasive cell highlighted, next to a
button that refits the head live. Watching a model retrain in eleven seconds is the cheapest way to
make "this is not a six-week training project" believable.

---

### Step 9 — Tissue-type segmentation ★ *the hard step — and it is inference only*

**What.** For every tile, answer: what kind of tissue is this? Three classes, from the build block
above:

`invasive epithelium` · `non-invasive epithelium (DCIS, LCIS, normal ducts)` · `non-epithelium (stroma, fat, inflammation, necrosis)`

**Why here.** After tiling and deconvolution, before nuclei segmentation (Rule 4). This is where
**fat is removed** (Rule 1) and where **DCIS is separated from invasive** (Rule 5). It is the
hardest and highest-risk step in the pipeline.

**This step trains nothing.** It loads `models/invasive_tile_v1.pt` from the build block and runs
inference: 224 px tiles at 0.5 µm/px, stride 112, batched. Roughly 45 minutes for six H&E slides on
a CPU. Dense pixel segmentation at 0.25 µm/px would be about **60 hours per slide** on the same
machine, which is precisely why this is a tile classifier and not a U-Net.

**Why it cannot be classical.** Nuclear density — the classical proxy — cannot separate DCIS from
invasive, because both are dense. The difference is *architectural*: DCIS nuclei sit inside an
intact duct with a myoepithelial layer around it; invasive nuclei have broken out into the stroma.
That is a spatial-pattern judgement, which is what learned features are for and what thresholds
are not. Primer item 87 says the same thing.

#### Start from the honest position

**There is no downloadable model that outputs `invasive / non-invasive / non-epithelium` on breast
IHC slides.** Anyone who tells you otherwise is describing an H&E model. What you *can* assemble
entirely from permissively-licensed parts is the build block above: **AICAN (MIT) as an offline
teacher, BCSS (CC0) as the label source, and a ResNet18 head** run on the **haematoxylin channel**
of whatever slide you point it at. No annotation, no GPU, no six-week training project.

#### The stain problem, and the fix that makes one model serve all five

This is the risk the earlier draft of this guide never named. The five antibodies do not look
alike. N-cadherin and pan-cadherin sit at **75-85 % positive**: dense cytoplasmic DAB floods the
tile, drowns the counterstain and erases the architectural cues a region model reads. A model
fitted on CD44 tiles - which range from 5 % to 85 % - will not transfer to them, and neither will
anything fitted on RGB H&E.

**The fix: do not show the region model RGB.** Step 7 already un-mixes every IHC tile into a
haematoxylin channel and a DAB channel. Feed the region model the **H channel only**, and:

- all five antibodies become the *same* input distribution, because the DAB has been removed;
- the input resembles the H&E that every public dataset was labelled on, which closes most of the
  domain gap for free;
- the U/W flooding problem disappears, because the flood was entirely in the channel you dropped.

Build the H-channel/RGB toggle as a demo screen. Two class maps side by side on an N-cadherin
slide makes the argument better than any paragraph can.

**Do not synthesise a fake eosin channel.** It is tempting to rebuild a pseudo-H&E so an
RGB-trained model will accept the tile, but the eosin information is not present in an IHC slide -
you would be inventing morphological evidence and then letting a model treat it as real. Training
on the H channel from the start is strictly better: nothing is fabricated, and the model is
stain-agnostic by construction rather than by disguise.

#### What ships, and what only benchmarks

The licence gate decides this, not accuracy. Every row below is real and useful; only the top four
can enter `backend/app/`.

| Asset | What it gives you | Licence | Ships |
| --- | --- | --- | --- |
| **BCSS** | 151 TCGA-BRCA regions, raw codes keep `dcis` separate from `tumor` | CC0 | **Yes** |
| **AICAN breast-epithelium** | Pretrained benign / in-situ / invasive epithelium. The offline teacher | MIT | **Yes** |
| **torchvision ResNet18** | ImageNet initialisation for the head | BSD-3 | **Yes** |
| **VALIS** | Serial-section registration, for the consistency check | MIT | **Yes** |
| TIGER | 195 WSIs, labels `1 invasive` / `3 in-situ` natively. Excellent, and the right ontology | CC BY-NC 4.0 | Benchmark only |
| BEETLE | 587 slides, 527 patients, **7 scanners**; non-invasive epithelium is its own class with **~225 mm²** of annotation; released 5-model nnU-Net scoring **0.65 Dice on non-invasive** across unseen scanners | CC BY-NC-SA | Benchmark only — but see [`approach-4a-training-plan.md`](approach-4a-training-plan.md) |
| BRACS | 4,539 ROI crops, three-pathologist consensus, 7 lesion types — **3,890 of them non-invasive epithelium** across 151 patients. ROI-level labels only, no tissue masks | **Non-commercial** *(verified 3 Sep 2026 at the download page; the paper's CC0 does not hold)* | **No** — `benchmarks/` only |
| TIAToolbox `fcn_resnet50_unet-bcss` | Dense 5-class breast segmentation at 0.25 mpp | CC0 weights | Yes, **but** it merges DCIS into tumour - see the build block trap |
| GrandQC | Tissue + artefact segmentation | CC BY-NC-SA | Benchmark only |
| HoVer-Net + PanNuke | Per-nucleus class | CC BY-NC-SA | No |
| DeepLIIF | Virtual multiplex from IHC. Also trained on **nuclear** markers, and our panel is membrane/cytoplasmic | Commons Clause | No |
| UNI, CONCH, Virchow2, Phikon-v2 | Frozen foundation encoders | NC / ND | No |
| Hibou-B | Frozen encoder, ViT-B/14, ~86M params | Apache-2.0 | Yes - the only permissive *and* CPU-sized encoder, if you want to try one |

**A note on foundation encoders.** An earlier draft of this guide recommended UNI2-h, Virchow2,
Phikon-v2 or CONCH with a logistic probe on top. The idea was sound and the licences are not: all
four are non-commercial. If you want to try the frozen-encoder route anyway, **Hibou-B** is the one
that is both permissively licensed and small enough to run on a CPU. The ResNet18 head in the build
block is the default because it is faster, smaller and has no licence question at all.

#### Two things you must do that most people skip

1. **Report accuracy stratified by tumour content.** A slide that is 80–90 % tumour is easy; a
   2–3 mm invasive focus inside a 2–3 cm block, mixed with DCIS and normal ducts, is the hard case
   and the clinically common one. Bin your test set by tumour percentage and publish a number per
   bin. A model that averages well can be useless on exactly the cases that matter.
2. **Report accuracy stratified by antibody.** One probe now serves five stains. Show its
   confusion matrix on an **N-cadherin** slide next to a **CD44** slide. If the H-channel trick is
   working, the two matrices look alike; if it is not, you will see it immediately rather than
   discovering it in the final numbers.

**Show in the demo.** Slide thumbnail with a class-colour overlay and a per-class opacity toggle,
so the viewer can switch fat on and off and *watch the denominator change*. Beside it, a UMAP of
tile embeddings coloured by class — and colour the *same* plot by antibody, to show the five stains
overlapping rather than forming five separate clouds. Below, the confusion matrix with the
DCIS-vs-invasive cell highlighted, because that is where the errors live.

**Papers.** [UNI, Nature Medicine 2024](https://github.com/mahmoodlab/UNI) · [CONCH, Nature Medicine 2024](https://github.com/mahmoodlab/CONCH) · [Virchow, Nature Medicine 2024](https://www.nature.com/articles/s41591-024-03141-0) · [Benchmarking foundation models as feature extractors, arXiv 2408.15823](https://arxiv.org/pdf/2408.15823) · [Survey of computational pathology foundation models, arXiv 2501.15724](https://arxiv.org/pdf/2501.15724) · [TIAToolbox, Communications Medicine 2022](https://www.nature.com/articles/s43856-022-00186-5) · [Cerberus multi-task, Medical Image Analysis 2023](https://arxiv.org/abs/2203.00077) · [Cruz-Roa invasive extent, Sci Rep 2017](https://www.nature.com/articles/srep46450) · [DCIS classification, PubMed 2022](https://pubmed.ncbi.nlm.nih.gov/35076741/) · [Invasive segmentation on IHC WSIs, Cancers 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10778369/) · [Multiclass segmentation of immunostained breast tissue, medRxiv](https://www.medrxiv.org/content/10.1101/2022.08.17.22278889v1.full)

---

### Step 10 — Build the ROI mask

**What.** Turn a grid of per-tile predictions into one smooth, closed region you can call "the invasive tumour".

**Why here.** Between the model and the cells. Raw model output is noisy and blocky; nuclei segmentation should run on a clean region, and the pathologist reviewing your ROI should see something that looks like an anatomical region rather than a QR code.

**How.** Classical post-processing, no training:

1. **Stitch** overlapping tile predictions by averaging probabilities in the overlap zones (this is why you used overlap in step 8).
2. **Smooth** the probability map with a Gaussian blur — enforces spatial coherence, because tissue types come in patches, not speckles.
3. **Threshold** to binary.
4. **Morphological closing** to fill small holes (a blood vessel inside a tumour should not punch a hole in the region).
5. **Drop small components** below a physical area — express the cutoff in **mm², not pixels**, so it survives a change of scanner.
6. **Optional: keep the largest N components**, matching how a pathologist circles one or two foci.

**Show in the demo.** Five panels marching left to right: raw blocky prediction → smoothed heatmap → binary → morphology-cleaned → final outline drawn on the H&E. Show the tumour area in mm² and the tumour-content percentage. Then add a **pathologist-edit mode** where the viewer can drag the boundary and watch the final score update live — that closes the loop and shows the human is still in charge.

**Papers.** Primer items 62–63, 70–72

---

### Step 11 — Nuclei segmentation

**What.** Find every individual nucleus inside the ROI and give each one its own outline and ID. Not "which pixels are nuclei" (semantic) but "which pixels belong to nucleus #4,182" (**instance**). The distinction matters because nuclei touch, and if two touching nuclei merge into one blob, you have just lost a cell from your count.

**Why here.** Inside the ROI, after step 10 (Rule 4) — cheap, and safe from junk detections in fat and necrosis.

**How.** Use a **pre-trained model**. Do not train this yourself; the public models are strong, and the annotated datasets they were trained on (PanNuke: 189,744 nuclei across 19 tissue types) are far larger than anything you will annotate.

| Model | Licence | How it works | Verdict |
| --- | --- | --- | --- |
| **InstanSeg** | **Apache-2.0** | Embedding-based instance segmentation, brightfield H&E *and* IHC, TorchScript, ~60 % faster than the alternatives | **The default choice.** Permissive, CPU-fast, and it already handles IHC |
| **Cellpose** | **BSD-3** | Generalist "flow field" model; works across modalities without retraining | Solid fallback. You already have `cellpose_count.py` |
| **StarDist** | BSD-3 code; **check the H&E weights** | Each nucleus as a star-convex polygon; excellent on round, densely packed objects | Fast and fine for H&E; confirm the `2D_versatile_he` weights terms before shipping |
| ~~HoVer-Net + PanNuke~~ | **CC BY-NC-SA** | Predicts each nuclear pixel's offset to its centre; also classifies cell type in the same pass | **Cannot ship.** Technically the nicest option and the reason this row hurts. Benchmark tree only |
| ~~CellViT~~ | inherits PanNuke terms | Transformer successor to HoVer-Net | Cannot ship, and needs a GPU anyway |

**Consequence for step 12.** Losing HoVer-Net means losing free cell typing, because typing came
bundled with its segmentation. Replace it with something classical: you already know each tile's
region class from step 9, and nuclear size, shape and haematoxylin density separate lymphocytes
(small, round, very dark) from tumour nuclei (large, irregular) well enough for a denominator.
That is a small loss and an honest one.

**For IHC slides: run nuclei detection on the haematoxylin channel from step 7, never on RGB.**
The counterstain is what marks nuclei; the DAB is the thing you are trying to measure and must not
influence where you think cells are. Mixing them creates a feedback loop — strongly stained cells
become more likely to be detected, which biases your percentage upward. This is a subtle, real bug
and worth a demo screen of its own.

**And it bites hardest on U and W.** N-cadherin and pan-cadherin sit at 75–85 % positive with dense
cytoplasmic staining. At that DAB load, deconvolution residual leaks into the H channel, nuclear
boundaries blur, and touching nuclei merge — every one of which *removes cells from the
denominator*, which inflates the percentage. Three defences, in order of how much they buy you:

1. Estimate stain vectors **per slide** (Macenko on that slide's own pixels) rather than using the
   textbook H-DAB matrix. A fixed matrix is at its worst exactly when one stain dominates.
2. Watch **cell count per mm² inside the ROI** as a QC metric, and compare it across the six slides
   of the same case. Those slides are serial sections of one block, so their densities should be in
   the same neighbourhood. If the N-cadherin slide yields 30 % fewer cells per mm² than the CD44
   slide, that is a segmentation failure, not biology — and it is a check that costs nothing.
3. Prefer **InstanSeg**, which was trained on brightfield IHC as well as H&E, over a model that has
   only ever seen H&E. (DeepLIIF is the other IHC-native option and is the obvious suggestion here —
   but it is **Commons Clause**, and it was trained on *nuclear* markers while our panel is
   membrane and cytoplasmic. Two independent reasons not to build on it.)

**Show in the demo.** Zoom to 40×, show outlines over the raw tile, with a live count. Add a "touching nuclei" close-up comparing a naive watershed against InstanSeg so the viewer sees *why* instance segmentation is a separate problem. Add the H-channel-vs-RGB toggle and show the two different cell counts.

**Papers.** [HoVer-Net, Medical Image Analysis 2019](https://arxiv.org/abs/1812.06499) · [code](https://github.com/vqdang/hover_net) · [StarDist, MICCAI 2018](https://arxiv.org/abs/1806.03535) · [Cellpose, Nature Methods 2021](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v2.full) · [PanNuke dataset](https://arxiv.org/abs/2003.10778) · [NuLite, a fast lightweight alternative](https://arxiv.org/html/2408.01797v2) · [Cellpose vs StarDist, practical comparison](https://www.technologynetworks.com/informatics/articles/cellpose-stardist-and-the-deep-learning-cell-segmentation-revolution-414419)

---

### Step 12 — Cell typing

**What.** Not every cell inside the tumour region is a tumour cell. Lymphocytes, fibroblasts, and endothelial cells are mixed in. If they land in your denominator, your percentage is diluted.

**Why here.** Right after segmentation, before measurement. This is the last filter on *which cells count*.

**How.** This used to be free: HoVer-Net outputs a class per nucleus when run with PanNuke weights.
Those weights are **CC BY-NC-SA**, so on the shipped path you write it yourself — and it is genuinely
easy. Fit a small classifier on morphology features (area, eccentricity, haematoxylin intensity,
local density) plus the tile's region class from step 9. Lymphocytes are small, dark and round;
tumour nuclei are large and irregular. A few hundred hand-checked nuclei are enough, and unlike the
region problem this one is easy for a non-expert to label.

**Show in the demo.** The same tile with nuclei coloured by class, plus a stacked bar of the cell-type mix. Then the punchline screen: **"% positive of all cells" vs "% positive of tumour cells only"** — usually a difference of 5–15 points. That is a number an audience remembers.

**Papers.** [HoVer-Net classification](https://arxiv.org/abs/1812.06499) · [NuCLS, GigaScience 2022](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac037/6586817) · [Panoptic segmentation for TILs scoring, PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213912/)

---

### Step 13 — Build compartments ❗ *forks by marker*

**What.** Nuclei segmentation gives you the nucleus. The marker does not live there. This step
builds the region of each cell where the brown is *supposed* to be — and which region that is
depends on the antibody.

**Why here.** After you know which cells count, before you measure. Compartments are geometry, not
biology.

**The fork.** This is the first step in the pipeline that reads the antibody letter.

| Markers | Compartment | How it is built |
| --- | --- | --- |
| **A** CD44, **F** ABCC4, **R** ABCC11 | **Membrane ring** | Dilate the nucleus by ~3–5 µm, subtract the nucleus, subtract every neighbouring nucleus |
| **U** N-cadherin, **W** pan-cadherin | **Cytoplasm band** | The cell body between the nuclear boundary and the outer cell edge — a wider expansion, ~5–8 µm, nucleus subtracted, **not** thinned to a ring |
| *(reference only)* | Nucleus | The segmentation mask as-is. Ki67, ER, PR — not in this panel |

The two are not the same shape and not the same size, and running U/W through the membrane path is
the specific regression this section exists to prevent: a thin 3 µm ring samples the outer edge of
the cytoplasm, catches the neighbouring cell's membrane, and reports a number that correlates with
cell packing rather than with N-cadherin.

**How.** Classical morphology. All distances are in **microns, converted to pixels via mpp** —
never hard-coded pixel counts. At this project's 0.2222 µm/px, a 4 µm ring is 18 px; on another
scanner it is not.

For crowded epithelium, a **Voronoi-constrained expansion** beats plain dilation: expand each
nucleus outward but stop at the midline between neighbours. This is what QuPath's "cell expansion"
does, and it prevents a strongly stained cell from bleeding its signal into a negative neighbour.
It matters more for U/W than for A/F/R, because the wider the expansion the more neighbours you
collide with.

**Parameters, and the honest status of each.**

| Parameter | Applies to | Suggested start | Status |
| --- | --- | --- | --- |
| `ring_um` | A / F / R | 4.0 µm | Sweep it; report the sensitivity |
| `band_um` | U / W | 6.0 µm | Sweep it; report the sensitivity |
| `voronoi_constrained` | all | on | Off only to demo why it should be on |

Neither default is established fact. Both come from the QuPath convention plus the physical size of
a breast epithelial cell; **publish the sensitivity of the final score to each**, because a score
that swings 15 points between a 3 µm and a 5 µm ring is a score with a hidden parameter in it.

**Show in the demo.** A single cell blown up huge, with the overlays toggled: nucleus / ring / band
/ whole cell. Beside it, a slider for the compartment width in microns, with the resulting percent
and intensity updating live. Then the punchline: put a CD44 cell and an N-cadherin cell side by
side with **both** compartments drawn on each, so the viewer sees that picking the wrong one is not
a small error.

**Papers.** [QuPath, Scientific Reports 2017](https://www.nature.com/articles/s41598-017-17204-5) · primer items 44–45, 88–89

---

### Step 14 — Per-cell measurement ❗ *forks by marker*

**What.** For each cell, one number always: the DAB optical density in that cell's compartment.
Plus, **for the three membrane markers only**, a second number: how *complete* the ring is.

**Why here.** This is the first moment you actually measure anything. Everything before it was
deciding *what* to measure.

**How.** Classical arithmetic on the DAB channel from step 7 — the **un-normalised, calibrated**
one. This is the step Rule 2 exists to protect.

**Intensity (all five markers).** Mean DAB OD across the compartment's pixels. Mean is more stable
than max; max is what QuPath uses by default and it is more sensitive but noisier. Report which you
chose and do not change it silently between markers.

**Membrane completeness (A, F, R only).** Walk around the ring in 36 angular bins of 10°, mark a
bin as stained if its DAB OD exceeds the positivity threshold, and report the fraction of stained
bins. A cell with 34/36 bins stained has genuine complete membrane staining; a cell with 6/36 has
non-specific speckle that only looks positive once you average it away. OncoStem's own deck
explicitly distinguishes **complete** membrane staining from **partial**, and ASCO/CAP wrote the
same distinction into the HER2 guideline, where "complete, intense membrane staining" *is* the
definition of 3+.

**For U and W, completeness is not a harder measurement — it is a meaningless one.** Cytoplasmic
staining has no circumference. Do not compute it, and do not let a shared code path compute it
anyway and quietly feed a near-zero value into the positivity rule. The cytoplasmic markers get a
different second number, or none:

| Marker group | Number 1 | Number 2 | Positivity rule |
| --- | --- | --- | --- |
| **A / F / R** membrane | mean DAB OD in the ring | **ring completeness**, 0–1 over 36 bins | OD above cut **and** completeness above cut |
| **U / W** cytoplasmic | mean DAB OD in the band | **stained fraction** of the band's pixels, 0–1 | OD above cut **and** stained fraction above cut |

The cytoplasmic second number is the honest analogue: it answers "is the brown spread through this
cell's body, or is it one speckle?" without pretending the brown has a shape it does not have.

**The open question, and its true scope.** How partial staining should enter the percentage — does
a cell at 12/36 bins count as positive, as half-positive, or not at all? — is unanswered, and it is
the open question with the largest effect on our numbers (Q1 in
[images-to-scores-mapping.md](images-to-scores-mapping.md)). Note that **it only applies to three
of the five markers.** Whatever is decided for CD44, ABCC4 and ABCC11 does not automatically
transfer to N-cadherin and pan-cadherin, and the two decisions should be recorded separately.

**Show in the demo.** A scatter plot: every dot is a cell, x = mean DAB OD, y = completeness (or
stained fraction). Draw the decision boundary on it. Let the viewer click a dot and see that cell's
image crop. Switch the marker and watch the y-axis change meaning — that transition is the clearest
way to show the audience that "positive" is defined per marker, not universally.

**Papers.** [ASCO/CAP HER2 completeness criteria](https://spj.science.org/doi/10.34133/bmef.0048) · [QuPath IHC cell classification](https://www.nature.com/articles/s41598-017-17204-5) · [IHC Profiler, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4011881/) · [DeepLIIF, Nature Machine Intelligence 2022](https://www.nature.com/articles/s42256-022-00471-x)

---

### Step 15 — Intensity binning ❗ *five cut-point sets, one per antibody*

**What.** Convert a continuous OD number into levels. Two things happen here, and conflating them
is a common bug:

1. **Per-cell binning** — each cell gets 0 / 1+ / 2+ / 3+. This is internal machinery, used to
   count positives and, if you want it, to compute an H-score for the demo screens.
2. **Slide-level banding** — the case's reported intensity, on **OncoStem's 0–2 scale**. This is
   the deliverable and it happens in step 16.

**Why here.** After per-cell measurement, before aggregation. This is the translation layer
between the machine's world and the pathologist's.

**The scale you report is not 0/1+/2+/3+.** All 120 intensity readings in `6Slide Reports2.xlsx`
sit on OncoStem's **0–2** scale, with permitted values **0, 0.5, 1, 1.5, 1.75, 2** (Negative 0;
Weak 0.5–1.0; Moderate 1.5; Strong 1.75–2.0). OncoStem's deck also describes a 1–3 scale in
places; the data says 0–2, so implement 0–2 and treat the conflict as a live open question rather
than burying it. **119 of the 120 readings land exactly on a band value**, which is what you see
when a reader picks one band per slide rather than averaging a continuum — so a banding step is
not optional polish, it is how you match the target.

**Where thresholds come from is the entire question.**

| Approach | Thresholds derived from | Comparable across slides? |
| --- | --- | --- |
| Per-slide percentiles | This slide's own 1st/99th percentile | **No** — 37 % here ≠ 37 % there |
| **Absolute calibrated OD** | Fixed OD cutoffs on the I₀-calibrated scale | **Yes** |
| Control-slide anchored | Cutoffs set once on known 0/1+/2+/3+ control slides | **Yes**, and clinically strongest |

Use absolute cutoffs, anchored on controls if you can get them. This is exactly the transition the
client repo made in EPIC-010, and primer item 79 explains why.

**One cut-point table cannot serve five antibodies.** Each has its own antibody concentration,
incubation time, detection chemistry and dynamic range. The OD that means "moderate" for CD44 is
not the OD that means "moderate" for pan-cadherin — and the observed data says so loudly: CD44
spans 5–85 % across six cases while N-cadherin is 80 % on every single read. Sharing one threshold
set across the panel guarantees that at least three of the five markers are cut in the wrong place.

So the configuration is **five independent sets**:

```
CUTS = {
  "A": {"od": (t1, t2, t3), "completeness_min": c},   # CD44
  "F": {...},                                          # ABCC4
  "R": {...},                                          # ABCC11
  "U": {"od": (t1, t2, t3), "stained_fraction_min": f},# N-cadherin
  "W": {...},                                          # pan-cadherin
}
```

Keep it in one versioned config file, never inline in code, and record which slides each set was
anchored on.

**You already own a calibration set — use it.** `6Slide Reports2.xlsx` holds **120 reader-scored
(percent, intensity) pairs**: four pathologists × six cases × five markers. That is enough to *fit*
the per-antibody OD-to-band mapping rather than guess it: run steps 1–14 on the six cases, then
choose each marker's cut points to minimise disagreement with the reader consensus for that marker.
This is legitimate calibration, not overfitting to the test set — provided you say so plainly, hold
out cases, and report the fit and the held-out result separately. With only six cases the honest
statement is "cut points were calibrated on n=6; they are provisional".

**Show in the demo.** The DAB intensity histogram with the cut lines drawn on it, draggable. Two
humps should be visible — unstained and stained populations — and the viewer can see whether the
cut lands in the valley between them. Then two switches: **per-slide vs absolute** across three
slides (per-slide scores come out wrongly similar, absolute correctly different), and **shared vs
per-antibody cuts** across CD44 and N-cadherin, which shows the shared setting driving one of them
into nonsense.

**Papers.** Primer items 49, 79–82 · [Ruifrok & Johnston 2001](https://pubmed.ncbi.nlm.nih.gov/11531144/) · [images-to-scores-mapping.md](images-to-scores-mapping.md), Parts 4 and 6

---

### Step 16 — Aggregate into a score ❗ *this is the deliverable — get the contract right*

**What.** Apply the rulebook. This step contains no image processing at all — it is arithmetic and
`if` statements, and it should be a small, heavily tested, separately reviewable module.

**Why here.** Last. By now every hard decision has been made; do not let scoring logic reach back
into pixels.

#### The contract: two numbers per marker, ten numbers per case

```
percent positive  = 100 x (positive tumour cells) / (total tumour cells)
                    then round to the nearest 5

intensity         = mean DAB OD over the positive tumour cells
                    then map through that antibody's OD-to-band table
                    to the nearest permitted band: 0 / 0.5 / 1 / 1.5 / 1.75 / 2
```

That is the whole deliverable. Five markers, two numbers each. Nothing else crosses the boundary
to OncoStem — see [images-to-scores-mapping.md](images-to-scores-mapping.md), Part 4.

Both roundings are load-bearing, not cosmetic. Real reported percentages are **multiples of 5**,
and 119 of 120 real intensities land **exactly on a band value**. Emitting `61.7 %` and `1.34`
does not look more precise to the reader receiving it — it looks like a different measurement than
the one they asked for, and it cannot be compared against their sheet.

#### OncoStem does not want an H-score

**Compute it if you like; never present it as the deliverable.** H-score is
`1x(% at 1+) + 2x(% at 2+) + 3x(% at 3+)`, range 0–300, and it is a genuinely useful one-number
summary for demo screens and internal sanity checks. But `6Slide Reports2.xlsx` has ten columns —
five percent-and-intensity pairs — and no combined score anywhere. **Collapsing the two numbers
into one throws away information their model is using.**

The same goes for **Allred** (proportion + intensity, 0–8), the **ASCO/CAP HER2** 0/1+/2+/3+ rules
and the **Ki67** hotspot-vs-average debate. All three are worth understanding, all three are worth a
demo screen showing the field's standard vocabulary, and **none of them is an output of this
system.** They are reference material; the panel is CD44, ABCC4, ABCC11, N-cadherin and
pan-cadherin.

#### The averaging question, which is still open

Pathologists divide the tumour into roughly three or four parts, assess each, and average — more
finely when staining is patchy, skipped entirely when it is uniform. Our software naturally
produces an **area-weighted** average over the whole ROI, which differs from a plain mean of
regions when the regions are unequal in size. Which one OncoStem uses is Q3 in the mapping doc.
Implement both, label them, default to area-weighted, and show the gap on screen.

#### What the module signature should look like

```
score(marker_letter, cells, cuts) -> {percent: int, intensity: float, ...}
```

The marker letter is a parameter, not a global. Every per-marker rule — which compartment, which
second number, which cut points, which banding table — resolves from it in one place, so adding a
sixth antibody later is a config entry rather than a code change.

**Show in the demo.** One screen, the whole cascade visible at once: cell counts by bin → the
formula written out with the actual numbers substituted in → the rounding step shown explicitly,
before and after → the final pair. No hidden arithmetic. Beside it, the case's ten numbers as the
grid OncoStem actually receives, with the pathologists' own values alongside where you have them.
Add a heterogeneity map — the ROI tiled by local % positive — so the viewer sees that "50 %" is an
average over a very non-uniform field.

**Papers.** [images-to-scores-mapping.md](images-to-scores-mapping.md), Parts 4 and 6 · [ASCO/CAP HER2 guideline, summarised](https://spj.science.org/doi/10.34133/bmef.0048) · [QuPath TMA scoring validation, PubMed](https://pubmed.ncbi.nlm.nih.gov/29575153/) · [Software comparison for breast IHC scoring](https://www.sciencedirect.com/science/article/pii/S215335392200712X) · [CD44/CD24 cutoffs, PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165253)

---

### Step 17 — Validation ❗ *report per marker, never pooled*

**What.** Evidence that the number is right. Without this the demo is a rendering engine, not a
measurement tool.

**Why here.** Continuously, on a held-out set — never on the slides you tuned thresholds on.

**How.** Different metrics for different steps, and using the wrong one is a classic mistake:

| What you are validating | Metric | Watch out for |
| --- | --- | --- |
| Tissue / ROI masks | Dice, IoU | Report per tumour-content band **and per antibody**, not pooled |
| Nuclei detection | F1 at a matching distance, panoptic quality | Accuracy is meaningless — the classes are wildly imbalanced |
| Cell classification | Per-class F1, confusion matrix | Macro-average, not micro |
| **Percent positive** | **Bland–Altman, and the ±10 rule below** | **Not plain R²** — R² is blind to systematic bias |
| **Intensity band** | **Weighted Cohen's kappa** on 0 / 0.5 / 1 / 1.5 / 1.75 / 2 | Unweighted kappa treats 0-vs-2 as no worse than 1.5-vs-1.75 |

#### Your accuracy target is already written down

OncoStem refers a reading back to its reader when it sits **more than 10 from the group average**,
and the data confirms that 10 is **absolute percentage points**: no reading among the 120 percent
values exceeds 10 points from its case mean. **So the target is explicit — land within ±10 points
of the four-reader consensus and the system sits inside the human spread.** That is the result to
aim for and the honest way to report it. (The rule appears not to apply to intensity, where reader
disagreement reaches 0.625 on a 0–2 scale — Q4 in the mapping doc.)

#### Pooling the five markers would hide everything that matters

The panel's markers are not equally informative, and averaging them produces a number that means
nothing:

| Marker | Spread across the six cases | What agreement on it proves |
| --- | --- | --- |
| **A** CD44 | 11 % to 80 % — a genuine seven-fold spread | **Almost everything.** This is the marker that separates cases |
| **F** ABCC4 | 38.75 % to 62.5 % | A real, moderate spread |
| **R** ABCC11 | 35 % to 61.25 % | A real, moderate spread |
| **U** N-cadherin | **80 % on every read, all four readers, all six cases** | **Nothing.** See below |
| **W** pan-cadherin | 77.5 % to 80 % | Nearly nothing |

**A model that returns a constant 80 scores perfectly on N-cadherin and has demonstrated
nothing.** Any pooled metric across all five markers is therefore inflated by two near-constants,
and any headline "we agree with pathologists 90 % of the time" that includes them is misleading —
including to ourselves. So:

1. **Report a table with five rows, one per marker.** Never a single pooled figure.
2. **Make CD44 the headline.** It has the range, so it carries the evidence.
3. **Flag U/W explicitly as uninformative under the current data**, and resolve the open question
   before tuning against them: do N-cadherin and pan-cadherin genuinely behave this way in nearly
   every breast tumour, or is 80 % a working convention where the exact figure is not the point?
   (Q6 in the mapping doc.) Neither answer is a criticism — we simply need to know which one we
   are looking at before we claim a result.

#### The three things people skip, and shouldn't

1. **Report against consensus, not one reader.** Two pathologists disagree with each other on IHC
   scores routinely (primer items 53–54). Your model's disagreement with a single reader is not
   necessarily an error. You have four readers per case; report inter-reader agreement first, then
   show the model's agreement in that context. If the model sits inside the human spread, that is
   the actual result.
2. **Stratify by difficulty as well as by marker.** Overall correlation hides everything. Bin by
   tumour content and report each bin.
3. **Say what the sample size buys you.** Six cases and 24 reads per marker is a very small n, and
   a confidence interval on a correlation at that size is enormous. State it rather than quoting a
   bare point estimate (primer items 97–98). "Provisional, n=6" is a stronger claim than a
   confident number nobody can reproduce.

**Show in the demo.** A five-row table — one marker per row — with model, consensus, and the delta,
colour-coded against the ±10 rule. A Bland–Altman plot for CD44 specifically, with the bias line
and limits of agreement drawn. A weighted-kappa confusion matrix for the intensity bands. And a bar
comparing inter-pathologist agreement against model-vs-pathologist agreement, per marker, so the
audience sees the human spread the model is being measured inside.

**Papers.** [images-to-scores-mapping.md](images-to-scores-mapping.md), Parts 6 and 9 · [QuPath validation vs manual scoring, PubMed](https://pubmed.ncbi.nlm.nih.gov/29575153/) · [Three-software comparison study](https://www.sciencedirect.com/science/article/pii/S215335392200712X) · [Ki67 automated vs manual, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10572449/) · [Cruz-Roa, reproducibility framing, Sci Rep 2017](https://www.nature.com/articles/srep46450) · primer items 96–99

---

## Part 4 — Two tracks: H&E and IHC

Real cases involve two slides of the same tissue: an H&E slide (for finding the tumour) and one or more IHC slides (for measuring the marker). They are cut from the same block, seconds apart, but they are **different physical sections** — the cells are not the same cells.

```
H&E slide                            IHC slide (CD44 + haematoxylin)
   |                                    |
Steps 1-5  QC, tissue, calibrate     Steps 1-5  QC, tissue, calibrate
   |                                    |
Step 7  deconvolve  -> H             Step 7  deconvolve -> H and DAB
   |                                    |            |
Steps 8-10  invasive region          Steps 8-10  invasive region
   |                                    |            |
   |                                    |            +--> Steps 14-16 measure DAB
   +---- consistency check only --------+
```

**The same code runs down both columns.** That is the point of feeding the region model the H
channel: the H&E track and the IHC track are not two pipelines, they are one pipeline run six
times per case. The H&E is no longer a *prerequisite* for the IHC slides - it is a **cross-check**.

**Option A — register the H&E ROI onto the IHC slide.** Align the two images (rigid, then elastic) and warp the mask across. Attractive, and it lets you use the H&E — where tumour is easiest to identify — to gate the IHC. But serial sections deform, tear, and shift, and registration error at the boundary directly becomes scoring error.

**Option B — find the region on the IHC slide directly.** No registration. Requires a segmentation model that works on IHC-stained tissue, where the counterstain is weaker and the DAB confounds appearance. This is harder to train but has no alignment error, and it matches what pathologists actually do — they re-identify the region by eye on each slide, using normal ducts as landmarks.

**Recommendation.** Build **Option B** as the main path, exactly as this client repo's `invasive.py` argues. Then build **Option A** as a demo screen showing the two masks overlaid and their disagreement in mm² — that visualisation is itself a good argument for why you chose B. Note there is published work on invasive-region segmentation trained directly on IHC WSIs ([Cancers 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10778369/)), so Option B is well-supported.

**But keep Option A, because it is your only label-free accuracy metric.** Six serial sections of
one block must yield a similar region area and shape. Register with **VALIS** (MIT), transfer the
H&E region onto each IHC slide, and compare it against what Option B found independently. Agreement
between two routes that share no code is real evidence, and it costs you nothing in annotation.
Bar to clear: **area CV ≤ 0.20 per case.**

One warning from this cohort: CD44 is near-blank on cases 00267 and 00865 (17.5 mm² against 84.8 on
its serial siblings), so intensity-feature registration **will** fail there. Fall back to matching
the tissue outline, and report registration quality per pair — a failed alignment must announce
itself rather than quietly producing a bad number.

---

## Part 5 — Build order for the demo app

Do not build the pipeline in pipeline order. Build it in **risk order**: the thing most likely to fail, first.

### Phase 1 — Skeleton (week 1)
Viewer + tiling + tissue mask + a hard-coded score. Nothing intelligent, but the whole app runs end to end and the screen-per-step structure exists. Ship this before anything else — it is the thing everything else plugs into.

### Phase 2 — The measurement branch (week 2)
Calibration, OD, deconvolution, thresholding, aggregation. All classical, all deterministic, no GPU. At the end of this phase you can already produce a real (if region-naive) score, and you have the most explainable screens in the app.

### Phase 3 — Cells (week 3)
Drop in **InstanSeg** (Apache-2.0, CPU-fast, handles IHC) or **Cellpose**. Pre-trained, so no training time. Now the score is per-cell instead of per-pixel, which is a visible, defensible upgrade — make the before/after a demo screen.

### Phase 4 — The region step (weeks 4–6) ★
The hard one, but no longer a training project. Work the **build block** above in order: AICAN
licence check and first run (day 1), BCSS download and remap (day 2), tile export (day 3), AICAN
pseudo-labels (day 4), fit the head (day 5). Then wire `HChannelTileDetector` into
`backend/app/scoring/invasive.py` and run all 36 slides.

Budget the time for iteration and evaluation, not for training — the fit itself takes minutes. Keep
the heuristic nuclear-density detector alive as the "before" comparison, and evaluate on a **CD44
slide and an N-cadherin slide separately** — that pair is your transfer test, and if the H-channel
trick is working their confusion matrices look alike.

### Phase 5 — QC and calibration polish (week 7)
In-house `artefacts.py`, absolute stain scale, control-slide anchoring, and the tissue-mask fix from
OD instead of saturation. These are the steps that turn a working demo into a *trustworthy* one.

### Phase 6 — Validation and story (week 8)
Ground-truth comparison, Bland–Altman, stratified tables, the narrated walkthrough that ties all screens together.

**The demo app shape.** Left rail = the 17 steps as a vertical stepper with tick marks. Main area = input | output | controls for the current step. Bottom bar = the score so far, updating live as the viewer changes anything upstream. That live-updating bottom bar is what makes the whole thing land: every knob visibly moves the answer, so the audience internalises that these are *choices*, not physics.

---

## Part 6 — Answering your original questions directly

| Your question | Short answer |
| --- | --- |
| **When do I remove fat tissue?** | Step 9, as part of the `non-epithelium` class. **Not** during tissue masking. |
| **When do I segment?** | Twice. Region segmentation at step 9–10 (which part of the slide). Cell segmentation at step 11 (which pixels are one cell). Region always first. |
| **What do I segment?** | Region level: three classes — invasive epithelium, non-invasive epithelium, non-epithelium. Cell level: individual nuclei, then grow compartments. |
| **When do I normalise?** | **Normally you don't.** The fork is step 7, not step 6: deconvolve once, H channel to the model, DAB channel to the measurement. Keep Macenko as a fallback on the H channel only, if validation shows H&E and IHC diverging. |
| **When do I create a mask?** | Four masks, in order: canvas padding vs glass (step 3.0), tissue-vs-glass (step 3), invasive ROI (step 10), per-cell compartments (step 13). |
| **When do I separate the stains?** | Step 7, after OD conversion, before any thresholding, on un-normalised pixels — and it feeds **both** branches. |
| **Do I need supervised learning?** | One small piece of it, once, offline: the **build block** between steps 8 and 9 fits a three-class head on a frozen ImageNet ResNet18, using BCSS labels and AICAN pseudo-labels. Minutes on a CPU. Every other step is classical or uses somebody else's weights. |
| **Can I use a pre-existing model?** | Yes, but the licence decides which. Ships: **AICAN** (MIT) as the offline teacher, **InstanSeg** / **Cellpose** for nuclei, **torchvision** ImageNet weights, **VALIS** for registration. Cannot ship: GrandQC, DeepLIIF, HoVer-Net/PanNuke, UNI, CONCH, Virchow2, Phikon-v2, TIGER, BEETLE — all non-commercial, all fine in the benchmark tree. |
| **Do I need a GPU?** | No, anywhere. The build block fits in minutes on CPU and step 9 does six H&E slides in ~45 minutes. That is a consequence of classifying tiles rather than segmenting pixels — the dense equivalent is ~60 hours per slide. |
| **Are these steps the same for all five markers?** | Steps 1–12 are identical — build once, run five times. Steps 13–17 fork: A/F/R are membrane markers (ring + completeness), U/W are cytoplasmic (band, no completeness), and every marker needs its own intensity cut points. |
| **Do I process the six slides together?** | No. There is no cross-slide dependency — Option B finds the region on each IHC slide directly — and calibration is per-slide by design. Run five independent slide jobs and collect ten numbers. Load the *models* once and stream slides through them; that is the only thing worth sharing. |

---

## Part 7 — Mapping the 100 primer points to the pipeline

If someone asks "where does concept N appear in the code", this is the table.

| Pipeline step | Primer items it makes concrete |
| --- | --- |
| Background reading — why any of this matters | 1–28 (biology, breast cancer, diagnosis workflow) |
| What the stains mean | 29–38 (histology, H&E, IHC, DAB, counterstain) |
| Which marker, measured where | 39–46 (ER, PR, HER2, Ki67, CD44; nuclear vs membrane vs cytoplasmic) |
| Step 16 — aggregation rulebook | 47–52 (percent positivity, intensity grading, H-score, Allred, ASCO/CAP) |
| Step 17 — validation | 53–56, 96–99 (inter-observer variability, consensus, ground truth, concordance, sample size, SaMD) |
| Steps 1, 3, 10 | 57–64 (WSI, pyramidal formats, tissue mask, ROI, why invasive matters) |
| Steps 3, 5, 10, 13 | 65–74 (pixels, RGB, HSV, thresholding, Otsu, morphology, connected components, Gaussian, density maps, optical density) |
| Steps 4, 5, 7, 15 | 75–82 (colour deconvolution, Beer–Lambert, Ruifrok & Johnston, reference vectors, per-slide vs absolute scale, white balance, histograms, percent positive) |
| Steps 3, 9, 10, 14 — the "before" state | 83–90 (heuristics, nuclear density proxy, fat exclusion, confidence=0, DCIS limitation, membrane completeness, geometric counting, invasive gating) |
| Steps 9, 11, 12 — the "after" state | 91–95 (CNNs, patch classification, Cellpose/StarDist, transfer learning, annotated data) |
| The whole document | 100 (how the pipeline maps to the pathologist's manual workflow) |

---

## Part 8 — Reference library

### Foundations (classical)
- Otsu — automatic thresholding · [IEEE Trans SMC 1979](https://ieeexplore.ieee.org/document/4310076)
- Ruifrok & Johnston — colour deconvolution · [PubMed](https://pubmed.ncbi.nlm.nih.gov/11531144/) · [PDF](https://helios2.mi.parisdescartes.fr/~lomn/Cours/CV/BME/HistoPatho/Quantification_of_histochemical_staining.pdf) · [maths explained](https://forum.image.sc/t/on-the-math-behind-colour-deconvolution-ruifrok-and-johnston-2001/66325)
- Reinhard — colour transfer · [IEEE CG&A 2001](https://ieeexplore.ieee.org/document/946629)
- Macenko — stain normalisation · [ISBI 2009](https://ieeexplore.ieee.org/document/5193250)
- Vahadane — structure-preserving normalisation · [IEEE TMI 2016](https://ieeexplore.ieee.org/document/7460968)
- OpenSlide — WSI reading · [J Pathol Inform 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3815078/)

### Stain normalisation, compared
- [Multi-centre benchmark, arXiv 2506.19106](https://arxiv.org/abs/2506.19106) · [Sci Rep version](https://www.nature.com/articles/s41598-026-40943-3)
- [Comprehensive review and experimental comparison](https://www.researchgate.net/publication/373610633_Stain_normalization_methods_for_histopathology_image_analysis_A_comprehensive_review_and_experimental_comparison)
- [Effect on IDC grading, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10665422/)
- [StainNet — fast learned normalisation, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8602577/)
- [Slideflow — practical WSI toolkit, arXiv](https://arxiv.org/pdf/2304.04142)

### Quality control
- [GrandQC, Nature Communications 2024](https://www.nature.com/articles/s41467-024-54769-y)
- [HistoQC vs PathProfiler](https://www.sciencedirect.com/science/article/pii/S1120179726000323)
- [Artefact augmentation, Sci Rep 2024](https://www.nature.com/articles/s41598-024-68667-2)
- [Human-in-the-loop QC, arXiv 2409.19587](https://arxiv.org/pdf/2409.19587)

### Region / tissue segmentation
- [Cruz-Roa — invasive tumour extent, Sci Rep 2017](https://www.nature.com/articles/srep46450) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5394452/)
- [DCIS vs IDC deep learning classification, PubMed 2022](https://pubmed.ncbi.nlm.nih.gov/35076741/)
- [Invasive carcinoma segmentation on IHC WSIs, Cancers 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10778369/) · [DOI](https://doi.org/10.3390/cancers16010167)
- [MS-ResMTUNet, PMC 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11636800/)
- [Multiclass segmentation of immunostained breast tissue, medRxiv](https://www.medrxiv.org/content/10.1101/2022.08.17.22278889v1.full)

### Nuclei / cell segmentation
- [HoVer-Net, arXiv 1812.06499](https://arxiv.org/abs/1812.06499) · [Medical Image Analysis](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045) · [code](https://github.com/vqdang/hover_net)
- [StarDist, arXiv 1806.03535](https://arxiv.org/abs/1806.03535)
- [Cellpose, bioRxiv / Nature Methods](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v2.full)
- [NuLite — lightweight successor, arXiv](https://arxiv.org/html/2408.01797v2)
- [Practical Cellpose vs StarDist comparison](https://www.technologynetworks.com/informatics/articles/cellpose-stardist-and-the-deep-learning-cell-segmentation-revolution-414419)
- [3D usability study, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667290122000420)

### Foundation models
- [UNI, Nature Medicine 2024](https://github.com/mahmoodlab/UNI)
- [CONCH, Nature Medicine 2024](https://github.com/mahmoodlab/CONCH)
- [Virchow, Nature Medicine 2024](https://www.nature.com/articles/s41591-024-03141-0)
- [Benchmarking foundation models as feature extractors, arXiv 2408.15823](https://arxiv.org/pdf/2408.15823)
- [Survey of computational pathology foundation models, arXiv 2501.15724](https://arxiv.org/pdf/2501.15724)
- [Representational similarity analysis across models, arXiv 2509.15482](https://arxiv.org/html/2509.15482v1)

### IHC scoring and quantification
- [QuPath, Scientific Reports 2017](https://www.nature.com/articles/s41598-017-17204-5) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5715110/)
- [QuPath TMA scoring validation, PubMed](https://pubmed.ncbi.nlm.nih.gov/29575153/)
- [Three-software comparison for breast biomarkers](https://www.sciencedirect.com/science/article/pii/S215335392200712X)
- [IHC Profiler, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4011881/)
- [DeepLIIF, Nature Machine Intelligence 2022](https://www.nature.com/articles/s42256-022-00471-x) · [code](https://github.com/nadeemlab/DeepLIIF) · [platform paper, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9494834/)
- [Automated HER2 scoring with pyramid sampling, BME Frontiers 2024](https://spj.science.org/doi/10.34133/bmef.0048) · [arXiv 2404.00837](https://arxiv.org/abs/2404.00837)
- [HER2-IHC-40x dataset, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12545830/)
- [piNET Ki67 framework, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7792768/)
- [Ki67 hotspot algorithm comparison, PMC 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11649514/)
- [Multi-scale Ki67 quantification, Sci Rep](https://www.nature.com/articles/s41598-025-28734-8)
- [Ki67 manual vs automated comparative study, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10572449/)

### The five markers of the panel

**A — CD44** (membrane; the marker with the range, and therefore the one that carries the evidence)
- [CD44/CD24 vs ALDH1 as prognostic indicators, PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165253)
- [CD44−/CD24+ in early invasive breast cancer, Springer](https://link.springer.com/article/10.1007/s10549-011-1865-8)
- [CD44/CD24 with EGFR in IDC, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4246130/)
- [CD44 IHC expression in breast carcinoma, PMC 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12382596/)

**F — ABCC4 / MRP4** (membrane; drug-efflux pump)
- [ABCC4 in breast cancer, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4359292/)
- [ABC transporters and chemoresistance, review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7345170/)

**R — ABCC11 / MRP8** (membrane; drug-efflux pump)
- [ABCC11/MRP8 expression and breast cancer, PubMed](https://pubmed.ncbi.nlm.nih.gov/16467088/)
- [ABCC11 as a prognostic marker, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3033490/)

**U — N-cadherin / CDH2** (**cytoplasmic**; EMT / adhesion switch)
- [N-cadherin and EMT in breast cancer, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5340734/)
- [Cadherin switching in tumour progression, J Cell Sci](https://journals.biologists.com/jcs/article/121/6/727/30346)

**W — Pan-cadherin** (**cytoplasmic**; all cadherins together)
- [Cadherins in breast cancer, review PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3095145/)

**Note on the evidence base.** CD44 has a substantial breast-IHC scoring literature; the other four
have far less, and none has an ASCO/CAP-style consensus scoring guideline. That is precisely why
step 15's cut points must be calibrated against OncoStem's own 120 reader scores rather than lifted
from a paper — for four of the five markers there is no paper to lift them from.

### Model zoo (pretrained weights you can actually download)

**Split by licence, because that is the first question, not the last.**

*Ships — permissive, safe in `backend/app/`:*

- **Region teacher** — [AICAN breast-epithelium-segmentation](https://github.com/AICAN-Research/breast-epithelium-segmentation) (MIT) · [PLOS One paper](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0328033)
- **Labels** — [BCSS](https://github.com/PathologyDataScience/BCSS) (CC0, dataset) · [Grand Challenge page](https://bcsegmentation.grand-challenge.org/)
- **Nuclei** — [InstanSeg](https://github.com/instanseg/instanseg) (Apache-2.0) · [Cellpose](https://github.com/MouseLand/cellpose) (BSD-3) · [StarDist](https://github.com/stardist/stardist) (`2D_versatile_he` — verify weights terms)
- **Encoder, if you want one** — [Hibou-B](https://huggingface.co/histai/hibou-b) (Apache-2.0, ViT-B/14, CPU-sized)
- **Registration** — [VALIS](https://github.com/MathOnco/valis) (MIT)
- **Toolkit** — [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox) (BSD-3) — but check each *model's* licence separately; `fcn_resnet50_unet-bcss` is CC0, `fcn-tissue_mask` and the HoVer-Net weights are not

*Benchmark tree only — non-commercial, never imported by the shipped package:*

- **QC** — [GrandQC](https://www.nature.com/articles/s41467-024-54769-y) (CC BY-NC-SA)
- **Datasets** — [TIGER](https://tiger.grand-challenge.org/Data/) (CC BY-NC) · [BEETLE](https://arxiv.org/html/2510.02037v1) (CC BY-NC-SA)
- **Encoders** — [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) · [Virchow2](https://huggingface.co/paige-ai/Virchow2) · [Phikon-v2](https://huggingface.co/owkin/phikon-v2) · [CONCH](https://huggingface.co/MahmoodLab/CONCH) · [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath) · [H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) (Apache-2.0 but gated and 1.1B params — CPU-infeasible)
- **Nuclei** — [HoVer-Net](https://github.com/vqdang/hover_net) + [PanNuke](https://arxiv.org/abs/2003.10778) · [CellViT](https://github.com/TIO-IKIM/CellViT)
- **IHC-native** — [DeepLIIF](https://github.com/nadeemlab/DeepLIIF) (Apache-2.0 **with Commons Clause**)
- **Multi-task** — [Cerberus](https://github.com/TissueImageAnalytics/cerberus) (non-commercial, colon-centric)

**Verify before you rely on any registry name.** Model-zoo keys and checkpoint URLs change between
releases of these toolkits. Every name above is a starting point to check against the version you
install, not a string to paste into production.

### Datasets
- [BCSS — breast tissue semantic segmentation, 16 classes incl. fat](https://bcsegmentation.grand-challenge.org/)
- [NuCLS — nucleus classification and segmentation, GigaScience 2022](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac037/6586817) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9112766/)
- [PanNuke — pan-cancer nuclei, 189,744 annotations](https://arxiv.org/abs/2003.10778)
- [BACH / ICIAR2018 — normal / benign / in-situ / invasive](https://iciar2018-challenge.grand-challenge.org/)
- [CAMELYON16 / 17](https://camelyon17.grand-challenge.org/)
- [Janowczyk IDC patches](https://andrewjanowczyk.com/deep-learning/)
- [TCGA-BRCA](https://portal.gdc.cancer.gov/)
- [TILs panoptic segmentation dataset, PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213912/)

---

## Closing note

The demo's real subject is not the algorithm. It is the **chain of decisions**: which pixels are tissue, which tissue is tumour, which tumour is invasive, which cells are tumour cells, which compartment of those cells to measure, and what counts as brown. Six decisions, each of which moves the final number, and each of which a pathologist makes silently in about two seconds.

Build the app so that every one of those six decisions is a visible screen with a knob on it and a
live score at the bottom. Then the demo does not merely explain the pipeline — it shows the
audience how much *judgement* is buried inside a number that looks like a fact.

And there is a seventh decision the earlier draft of this document did not name: **which marker am
I looking at?** It sets the compartment, the second measurement, the cut points and the banding.
Make it a visible control in the app rather than a constant in the code — switching from CD44 to
N-cadherin and watching four downstream steps change their behaviour is, on its own, one of the
better screens in the demo.
