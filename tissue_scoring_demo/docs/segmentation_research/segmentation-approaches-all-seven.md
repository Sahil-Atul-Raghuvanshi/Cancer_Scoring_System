# All Seven Approaches — Full Reference, with Time and Storage

> Created: 2 Sep 2026 | Audience: whoever has to budget this — time, disk, and someone else's hours.
> **This is the single-file reference.** Approaches 1–4 are argued in
> [`segmentation-approaches-comparison.md`](segmentation-approaches-comparison.md); approaches 5–7 in
> [`segmentation-approaches-ranked.md`](segmentation-approaches-ranked.md); the build orders are in
> [`approach-1-training-plan.md`](approach-1-training-plan.md),
> [`approach-3-training-plan.md`](approach-3-training-plan.md),
> [`approach-4a-training-plan.md`](approach-4a-training-plan.md) and
> [`approach-4b-training-plan.md`](approach-4b-training-plan.md).
>
> What this document adds that none of those have: **time and storage computed the same way for all
> seven**, from one set of measured constants, so the columns are actually comparable. Where a number
> is measured it says so; where it is arithmetic it shows the arithmetic; where it is a guess it is
> labelled a guess.
>
> **Step-by-step implementation for all seven, with code against the real `backend/app/` APIs, is in
> [`segmentation-approaches-implementation.md`](segmentation-approaches-implementation.md).**
> Feeds: guide step 9 = **code step 8**, `backend/app/pipeline/step08_tissue_type_segmentation/`.

---

## Part 0 — The measured baseline

Every estimate below is built from these. They were measured on this machine, not assumed.

### What is on disk today *(measured 2 Sep 2026)*

| Location | Size | What it is |
| --- | --- | --- |
| `images/` | **35 GB** | The cohort: 36 `.svs` slides, 6 cases × (1 H&E + 5 IHC) |
| `data/slides/` | **13 GB** | ⚠ **16 byte-identical copies of one slide** — see below |
| `data/uploads/` | 793 MB | Chunked upload staging |
| `data/calibration/` | 90 MB | Step 4 per-slide `I₀` cache |
| `data/qc/` | 65 MB | Step 2 artefact masks |
| `data/tissue/` | 37 MB | Step 3 tissue masks |
| `models/` | 187 MB | `grandqc/` + `tissue_type/pretrained/` |
| **Free on `C:`** | **27 GB** | 476 GB total, 95 % used |

> ⚠️ **The free-space figure in the build plans is stale, and the direction is the wrong one.**
> [`approach-1-training-plan.md`](approach-1-training-plan.md) and
> [`approach-4b-training-plan.md`](approach-4b-training-plan.md) both record **33 GB free**. It is now
> **27 GB**. Six gigabytes went somewhere between writing those plans and today, and approach 1's own
> budget is 10–20 GB against that. **Re-measure before every download, not once per document.**

### The 12.5 GB sitting in `data/slides/` for free

All 16 `.svs` files in `data/slides/` are the **same slide**, `CAN_00251_26_A.svs`, at
830,790,030 bytes each — 16 upload-test runs that each kept their payload. Their sidecar JSONs all
report the same `filename` and `state: "ready"`.

```
16 × 830,790,030 B  =  13.3 GB
keep 1, delete 15   =  12.5 GB reclaimed
27 GB free  →  ~39 GB free
```

**That is more headroom than any approach on this page needs for itself, and it costs nothing.** It
is also more than the 6 GB that has quietly disappeared since the plans were written. Nothing here
deletes it — the files are yours to remove, and the upload path may still reference the IDs — but no
storage plan should be made without knowing it is there. Worth checking whether the upload handler
is meant to de-duplicate by `sha256` (the field exists in the sidecar and is `null` in all 16).

### The constants every estimate below uses

| Constant | Value | Where it comes from |
| --- | --- | --- |
| One slide | **830 MB** | Measured: `CAN_00251_26_A.svs` = 830,790,030 B |
| The cohort | **36 slides, 35 GB** | Measured: `images/` |
| Tile geometry | **224 px at 0.5 µm/px = 112 µm square = 0.0125 mm²** | [`approach-1-training-plan.md`](approach-1-training-plan.md) Part 1 |
| **CPU feature extraction** | **0.39 s/tile**, un-batched, PNG read from disk | **Measured**: approach 1 cached ~14,000 tiles in ~1.5 h |
| Tile on disk | **~17.8 KB** as uint8 PNG | Derived: 250 MB ÷ 14,000 tiles |
| Feature vector | **2 KB** (512 × float32) | Arithmetic |
| Tissue area per slide | 17.5 mm² (worst: 00267 CD44) to 315 mm² (largest: 00259); **~100 mm² typical** | [`segmentation-approaches-comparison.md`](segmentation-approaches-comparison.md) |

### The tile census, which drives every compute estimate

```
tiles per slide  = 100 mm² ÷ 0.0125 mm²        ≈   8,000   (stride 224, no overlap)
all 36 slides                                  ≈ 287,000
all 36 slides at stride 112 (4× overlap)       ≈ 1,150,000
```

> ⚠ **The number that reorders the cheap approaches.** At the measured 0.39 s/tile, embedding *every*
> tile on all 36 slides costs **287,000 × 0.39 s ≈ 31 hours**. Any plan that says "embed our slides"
> without subsampling has quietly signed up for two overnight runs. Approach 5 is affordable
> *because* it subsamples — see Part 6.

> ⚠ **And one honest inconsistency, flagged rather than smoothed over.** 0.39 s/tile implies ~52 min
> per slide, but [`segmentation-approaches-comparison.md`](segmentation-approaches-comparison.md)
> Part 5 quotes 5–8 min per slide for inference. Both are plausible: 0.39 s/tile is un-batched with a
> PNG decode per tile, while inference reads from the WSI pyramid in batches. **Batching is the
> difference, and nobody has measured the batched rate.** Where the two disagree below, the range
> spans both. Measure it on one slide before committing to any schedule built on it.

---

## Part 1 — The two master tables

### Time

| # | Approach | Download | Build compute | Human time | **Wall clock to first number** | GPU? |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | BCSS + AICAN | 30–60 min | ~5 h | — | **~5–6 h** | No |
| **2** | TIGER + BEETLE U-Net | 2–4 h *(+147 GB)* | **months on CPU** / 3–7 d GPU | — | **not viable on CPU** | **Mandatory** |
| **3** | MoCo init | +1 min (46 MB) | same as 1 | — | **~5–6 h** | No |
| **4a** | BEETLE teacher | ~1 h (1.9 GB) | 6–18 h overnight + ~3 h | 2 h pathologist | **~2 days + 1 overnight** | No |
| **4b** | BEETLE direct | **3.5–13 h** (147 GB) | ~6–10 h export + ~11.5 h features | — | **~6 working days** | No |
| **5** | Click own tiles | **none** | **~4 h** (subsampled) | **60–90 min clicking** + 30 min pathologist | **~1 working day** | No |
| **6** | Register & propagate | none | **1–2 h** | ~2 h outlining (optional) | **~half a day** | No |
| **7** | Manual + client data | client transfer | ~1 week after data lands | 3 d ours + pathologist hours | **4–12 weeks** | No |

### Storage

| # | Approach | Peak transient | Permanent kept | Deletable after | Fits in 27 GB free? |
| --- | --- | --- | --- | --- | --- |
| **1** | BCSS + AICAN | **10–20 GB** | ~500 MB | BCSS raw, after Notebook 02 checksums pass | ⚠️ **Tight** |
| **2** | TIGER + BEETLE U-Net | **~300 GB** | — | — | ❌ **No** |
| **3** | MoCo init | +46 MB over 1 | +46 MB | never (it is the encoder) | ✅ Yes |
| **4a** | BEETLE teacher | **~8.5 GB** | ~1.5 GB | `model.zip` + folds, after N03 | ⚠️ Yes, but not alongside 1 |
| **4b** | BEETLE direct | **~400 GB** | ~2.2 GB | images, after tiling | ❌ **No — external disk only** |
| **5** | Click own tiles | **< 1 GB** | ~700 MB | sampled tiles, after features cached | ✅ **Yes, easily** |
| **6** | Register & propagate | ~5 GB *(estimate)* | ~750 MB of masks | VALIS intermediates | ✅ Yes |
| **7** | Manual + client data | **40–170 GB of client slides** | all of it | never — it is the dataset | ❌ **No — needs a plan now** |

> **The two lines nobody has budgeted.** Approach 4b's 400 GB is documented and gated (Gate D0).
> **Approach 7's 40–170 GB is not documented anywhere**, and it is the approach with the highest
> ceiling and the longest lead time. 50 cases × 1 H&E × 830 MB = **41.5 GB**; 100 cases × 2 slides =
> **166 GB**. If the OncoStem request goes out this week, the disk to receive the answer has to be
> ordered in the same week. See Part 8.

---

## Part 2 — Approach 1: BCSS + AICAN → H-channel ResNet18 *(the committed route)*

**What it is.** Train a 3-class head on a frozen ResNet18 reading 224 px haematoxylin tiles, learning
what cancer looks like from BCSS (151 TCGA slides, CC0) and what our scanner looks like from AICAN's
opinion on our own H&E slides. Full build order in
[`approach-1-training-plan.md`](approach-1-training-plan.md).

**Classes:** `2` invasive / `1` non-invasive / `0` non-epithelium. Use BCSS's **raw** codes — the
popular 5-class version merges DCIS into tumour, which is the one merge this project cannot accept.

### Time

| Task | CPU | GPU |
| --- | --- | --- |
| BCSS download (girder route, ~4 GB) | 30–60 min, network-bound | same |
| AICAN on 6 H&E slides *(optional — skip if pyFAST will not run)* | ~2.5 h | ~15 min |
| Tile export, both sources | ~1 h, disk-bound | same |
| Feature extraction, frozen body, once (~14,000 tiles) | **~1.5 h** *(measured)* | ~3 min |
| Fit the head on cached features | **seconds to 2 min** | seconds |
| **Total to first model** | **~5–6 h** | ~30 min |
| *Optional* unfreeze `layer4`, 10 epochs | ~15 h, overnight | ~20 min |
| Inference, 6 H&E, stride 224 | 30–45 min | ~1 min |
| Inference, all 36 slides, stride 112 | ~12–14 h, overnight | ~25 min |

### Storage

| What | Size | Deletable when |
| --- | --- | --- |
| BCSS images + masks at 0.25 µm/px | **~5–10 GB** | after Notebook 02 exports tiles **and its manifest checksums pass** |
| Working copies during export | doubles the above → **10–20 GB peak** | transient |
| `gtruth_codes.tsv` | 1 KB | keep |
| ResNet18 ImageNet weights | 45 MB | keep — vendored |
| Exported H-channel tiles (~14,000 × 224², PNG) | ~250 MB | **keep — it is the training set** |
| Cached frozen features (14,000 × 512 f32) | ~30 MB | keep — refitting reads only this |
| Checkpoints + reports | < 200 MB | keep |
| **Peak / permanent** | **10–20 GB / ~500 MB** | |

**Against 27 GB free this is tight but workable** — provided nothing else downloads concurrently.
Sequence it: finish Notebook 02, verify checksums, delete the BCSS raw, *then* start anything else.

**Binding constraint:** not compute — **disk, and class `1`.** BCSS's non-invasive epithelium is
0.129 % of pixels with DCIS from a single patient, a 237 : 1 imbalance. The ontology separates DCIS;
the data cannot train it.

---

## Part 3 — Approach 2: TIGER + BEETLE → U-Net → register onto IHC

**What it is.** Train a proper dense pixel segmentation network on the two largest public breast
datasets, run it on the H&E, then register the mask onto each IHC slide.

### Time

| Task | CPU | GPU |
| --- | --- | --- |
| Download TIGER (2.6 GB) + BEETLE (150.9 GB) | 2–4 h+ | same |
| Train nnU-Net, 5-fold | **months — not viable** | 3–7 days |
| Dense inference, largest slide (315 mm²) | **~60–70 h — not viable** | 40–80 min |
| Dense inference, 6 H&E | ~200 h | ~4 h |
| VALIS registration, per case | 5–20 min | same (CPU-bound) |
| Train the second IHC-specific model | not viable | 1–2 days |

### Storage

| What | Size |
| --- | --- |
| TIGER | 2.6 GB |
| BEETLE `images.zip` | 147.2 GB |
| **Peak during extraction** | **~300 GB** |
| nnU-Net preprocessed cache | tens of GB more *(estimate)* |
| **Total** | **~300–400 GB, external disk only** |

**Why it is dead, in one line:** the licences are CC BY-NC and CC BY-NC-SA, so it cannot ship — and
even if they were permissive, dense CPU inference at 60–70 h per slide fails on hardware grounds
independently. **Benchmark tree only**, to learn where the ceiling is.

---

## Part 4 — Approach 3: MoCo initialisation *(a one-line change to approach 1)*

**What it is.** Approach 1 with the ResNet18 starting from Ciga & Martel's self-supervised
histopathology weights (MIT) instead of ImageNet. Same architecture, same parameter count, same
forward cost. Build plan: [`approach-3-training-plan.md`](approach-3-training-plan.md).

### Time and storage

| | Cost |
| --- | --- |
| **Extra download** | **46 MB** — one `.ckpt` |
| **Extra compute** | **zero.** Identical architecture |
| **Extra storage** | **46 MB** |
| **Extra wall clock** | one more feature pass (~1.5 h) if you A/B both inits, which you should |

This is not really a separate approach — it is ablation **A1** in approach 1's own list, ranked first
there precisely because it is free. **Train both inits, keep whichever wins on held-out BCSS.** The
commonly-cited "ResNet50" is wrong; the released checkpoint is ResNet18.

**Expected gain:** a few points, not a transformation. There is no reason not to do it.

---

## Part 5 — Approach 4: BEETLE as the class-`1` source

Two routes to the same deliverable. Route A downloads 1.9 GB of **weights**; Route B downloads 147 GB
of **data**. Plans: [`approach-4a-training-plan.md`](approach-4a-training-plan.md) and
[`approach-4b-training-plan.md`](approach-4b-training-plan.md).

**Why BEETLE at all:** its four annotation classes map 1:1 onto ours, and its non-invasive epithelium
is ~225 mm² across 527 patients and 7 scanners — a 3.1 : 1 imbalance instead of BCSS's 237 : 1.
**3.1 : 1 is a `class_weight` argument; 237 : 1 is a class you cannot train.**

> ⚠️ **Licence, unresolved for both routes.** CC BY-NC-SA 4.0. ShareAlike is the clause that matters:
> a student distilled from these weights is arguably an adaptation, which would have to be released
> CC BY-NC-SA too — so it follows our own weights rather than merely blocking redistribution.
> **Deprioritised by decision, not resolved.** Must be closed before commercial release.

### 4a — teacher and distil

| Task | CPU |
| --- | --- |
| Notebook 00 CPU smoke test *(hard gate — nnU-Net on CPU or stop)* | 1–2 h |
| Download + pin `model.zip` (1.9 GB) | ~1 h |
| Sample 240 regions of 2048² from 6 H&E slides | ~1 h |
| **Teacher inference, 6 slides** | **6–18 h, overnight** *(estimate — measure on 3 regions first)* |
| Pathologist review of pseudo-labels on 30 regions | **2 h of a pathologist** |
| Export tiles | ~1 h |
| Fit head + A/B against approach 1 | minutes |
| **Total** | **~2 working days plus one overnight** |

| Storage | Size | Deletable when |
| --- | --- | --- |
| `model.zip` + extracted folds | ~4 GB *(estimate)* | after Notebook 03 writes pseudo-labels and checksums pass |
| Sampled H&E regions, 2048² at 0.5 µm/px, 6 × ~40 | ~3 GB *(estimate)* | after Notebook 05 exports tiles |
| Pseudo-label masks (uint8, same geometry) | ~1 GB *(estimate)* | **keep** — they are the labels, and the review is against them |
| Exported tiles + features | ~280 MB | keep |
| `data_overview.csv` | 176 kB | keep |
| **Peak / permanent** | **~8.5 GB / ~1.5 GB** | |

> ⚠ **4a and approach 1 compete for the same 27 GB.** 8.5 GB + 10–20 GB does not fit. Sequence them,
> and delete approach 1's BCSS raw before starting 4a's download.

### 4b — train directly on BEETLE data, no BCSS

| Task | CPU |
| --- | --- |
| Download `images.zip` (147.2 GB) | **3.5 h at 100 Mbps / ~13 h at 25 Mbps** |
| Extraction + per-slide checksums | hours |
| Notebook 01, annotations and splits | 2–3 h |
| Notebook 02, tile export | **6–10 h, disk-bound** *(estimate)* |
| Notebook 03, cache features (~106,000 surviving tiles) | **~11.5 h, one overnight** |
| Fit head + the R1–R4 reports | minutes |
| **Total** | **~6 working days** |

| Storage | Size | Where |
| --- | --- | --- |
| `images.zip` | **147.2 GB** | external disk |
| Extracted WSIs | ~150 GB *(estimate — TIFF, already compressed)* | external disk |
| **Peak during extraction** | **~300 GB** | external disk |
| `annotations.zip` + extracted | 1.8 GB + ~5 GB *(estimate)* | external disk |
| Exported H-channel tiles | ~2 GB | project drive |
| Cached frozen features | ~220 MB | project drive |
| **Gate D0** | **≥ 400 GB free on an external disk** | **hard gate — there is no partial version** |

**Two things make 4b the worst-value route despite the best data.** It is 400 GB and six days against
Route A's 1.9 GB and two — and dropping BCSS removes the **clean-licence fallback**. Under Route A the
BCSS-only arm of the ablation is a shippable model that exists whatever happens to the licence
question. Under 4b every checkpoint produced is non-commercial. **Written, not scheduled.**

---

## Part 6 — Approach 5: label ~900 of our own tiles

**What it is.** No public dataset at all. Tile our own 36 slides through steps 1–7 (already built),
embed a subsample with a frozen encoder, cluster so that clicking is fast, click ~900 tiles into three
buckets, fit a logistic regression on the cached vectors. Detail in
[`segmentation-approaches-ranked.md`](segmentation-approaches-ranked.md) Part 1.

**Why it ranks first:** it is the only one of the seven that produces a test set on OncoStem tissue.
Approaches 1–4 can only be evaluated on public H&E, and held-out BCSS provably cannot measure the
invasive-vs-DCIS boundary because it contains no DCIS.

### The subsampling arithmetic, which is the whole plan

```
Embed every tile, 36 slides:  287,000 × 0.39 s  =  31 h     ← do NOT do this at build time
Embed 1,000 tiles per slide:   36,000 × 0.39 s  =  3.9 h    ← one evening
```

1,000 tiles per slide is ample: k-means into ~60 clusters, then draw ~900 exemplars across clusters
for clicking. The other 251,000 tiles are never needed until inference, which is a separate cost
shared with every approach on this page.

### Time

| Task | CPU | Notes |
| --- | --- | --- |
| Sample + export 1,000 H-channel tiles × 36 slides | ~1 h | disk-bound, reuses step 7's index |
| **Embed 36,000 tiles, frozen encoder** | **~3.9 h** | 36,000 × 0.39 s *(measured rate)* |
| k-means to 60 clusters + build contact sheets | ~5 min | |
| **Clicking ~900 tiles** | **60–90 min** | the only significant human cost |
| Pathologist review of the ~40 boundary tiles | **30 min of their time** | the one thing we cannot self-serve |
| Fit logistic regression on cached vectors | **seconds** | 36,000 × 512 floats is 74 MB — full batch |
| Leave-one-case-out, 6 folds | ~1 min | |
| **Total to first measured number** | **~1 working day** | |

### Storage

| What | Size | Deletable when |
| --- | --- | --- |
| Encoder checkpoint (ResNet18 45 MB / MoCo 46 MB / Hibou-B ~330 MB) | 45–330 MB | keep |
| Sampled tiles, 36,000 × 224² uint8 PNG at ~17.8 KB | **~640 MB** | after features are cached |
| Cached features, 36,000 × 512 f32 | **~74 MB** | **keep — refitting reads only this** |
| Contact sheets (JPEG) | ~20 MB | after clicking |
| Label file (tile id → class) | < 1 MB | **keep — this is the irreplaceable artefact** |
| Head checkpoint + reports | < 100 MB | keep |
| **Peak / permanent** | **< 1 GB / ~700 MB** | |

**The cheapest approach on this page by both measures**, and by a wide margin: about 1/20th of
approach 1's disk and 1/400th of 4b's.

> **Back up the label file the moment it exists.** Everything else here can be regenerated in an
> evening. Those 900 human decisions cannot, and they are worth more than every model in this
> document.

**Binding constraint:** **gate zero.** If our six cases contain little or no DCIS, no amount of
clicking creates class-`1` examples — approach 5 then learns a two-class problem wearing a three-class
head, and the class-`1` prior still has to come from BCSS, BEETLE or BRACS. **Approach 5 does not
replace approaches 1/4; it measures them.**

---

## Part 7 — Approach 6: register once, propagate the mask

**What it is.** Segment the one H&E per case, then VALIS-register it to each of the five IHC serial
sections and warp the mask across. Five of six slides never need a model. Detail in
[`segmentation-approaches-ranked.md`](segmentation-approaches-ranked.md) Part 2.

### Time

| Task | CPU |
| --- | --- |
| VALIS install + Bio-Formats/JVM setup | 1–2 h, once *(and it is fiddly)* |
| Hand-outline the 6 H&E slides in QuPath *(if that is the mask source)* | ~2 h total |
| Registration, per case (5 pairs) | 5–20 min |
| **All 6 cases** | **~1–2 h** |
| **Total to a mask on all 36 slides** | **~half a day** |

### Storage

| What | Size | Notes |
| --- | --- | --- |
| VALIS + dependencies (incl. JVM, Bio-Formats) | ~1–2 GB *(estimate)* | one-off install |
| Downsampled working copies VALIS writes per pair | ~2–4 GB *(estimate — verify on case 1)* | transient, delete per case |
| Warped masks, 30 IHC slides at 2 µm/px uint8 | **~750 MB** | 100 mm² ÷ (2 µm)² ≈ 25 M px = 25 MB × 30 |
| Transform parameters (JSON/matrices) | < 1 MB | **keep — cheap to store, expensive to recompute** |
| **Peak / permanent** | **~5 GB / ~750 MB** *(estimate)* | |

**Where it earns its place:** not as the product — as a **QA cross-check requiring zero annotation.**
Run the tissue model on the H&E and on the five IHC slides of one case, register, and measure mask
agreement. That directly tests the bet the whole H-channel design rests on — *does one model really
behave the same on H&E and on IHC?* — and we can run it today on six cases for free.

**Two hard limits.** It needs a matched H&E in the same block, so it is not a standalone-IHC product.
And on 00267's near-blank CD44 (17.5 mm² against 84.8 on its siblings) VALIS will return a transform
anyway, and it will be wrong **silently**. Gate on matched-keypoint count, residual error and
tissue-area ratio, and **refuse rather than emit** below threshold — about 30 lines, and it converts
the worst failure mode into a visible one.

---

## Part 8 — Approach 7: annotate it ourselves, then ask OncoStem for real data

**What it is.** Phase A: outline the 6 H&E slides ourselves in QuPath, propagate to the 30 IHC slides
via approach 6, have a pathologist *review* rather than draw, and train on the result. Phase B: ask
OncoStem for annotated cases and train properly. Detail in
[`segmentation-approaches-ranked.md`](segmentation-approaches-ranked.md) Part 3.

**Last to finish, first to start** — its clock is controlled by someone else.

### Time

| Phase | Task | Elapsed |
| --- | --- | --- |
| **A** | Outline 6 H&E ourselves in QuPath | ~1 day |
| **A** | Propagate via approach 6 + export tiles + fit head | ~1 day |
| **A** | Pathologist review | 2 h of their time, ~1 week of calendar |
| **A** | **Phase A total** | **~3 days of ours + 2 pathologist hours** |
| **B** | Draft and send the annotation spec | **half a day** |
| **B** | **Client turnaround** | **4–12 weeks — the real cost** |
| **B** | Transfer 40–170 GB | 1–2 days, network-dependent |
| **B** | Ingest, QC the annotations, retrain, re-measure | ~1 week once data lands |

### Storage — the line item nobody has budgeted

| What | Size | Notes |
| --- | --- | --- |
| Our own QuPath project + GeoJSON for 6 slides | < 50 MB | trivial |
| **Client slides, 30 cases × 1 H&E** | **~25 GB** | at the measured 830 MB/slide |
| **Client slides, 50 cases × 1 H&E** | **~42 GB** | the realistic ask |
| **Client slides, 100 cases × 2 slides** | **~166 GB** | the ideal ask |
| Client annotations (GeoJSON / ImageScope XML) | < 500 MB | tiny, and the most valuable bytes in the project |
| Exported tiles from 50 cases *(scaling approach 1's 250 MB / 151 ROIs)* | ~2–4 GB *(estimate)* | |
| Cached features | ~500 MB *(estimate)* | |
| **Total** | **40–170 GB, none of it deletable** | **it is the dataset** |

> ⚠️ **Order the disk in the same week the request goes out.** This is the largest storage commitment
> in the project after 4b's 400 GB, it is the one with the highest payoff, and it appears in **no**
> existing plan. 27 GB free — 39 GB after reclaiming the duplicates — does not receive 50 annotated
> cases. And unlike BEETLE, this data cannot be re-downloaded from Zenodo if it is lost: budget for a
> backup copy too, so **plan roughly 2× the raw figure.**

### The three lines in the request that matter more than the case count

1. **Sparse fully-labelled ~2048² boxes, 10–20 per slide — not whole-slide outlines.** 20 min/slide
   instead of 3 h/slide. For tile-classifier training it is worth the same. This framing is the
   difference between a request that gets fulfilled and one that does not.
2. **`in-situ / DCIS` as its own class.** Without it we have bought nothing — that is the exact
   failure that makes BCSS unusable for class `1`.
3. **10 cases double-read by two independent pathologists.** Two pathologists agree at roughly 0.85
   Dice on this boundary. **A model at 0.82 against a single reader may already be at human level,
   and today we have no way to know.** Cheapest item on the list, and it is what makes every other
   number in these documents interpretable.

Full spec table: [`segmentation-approaches-ranked.md`](segmentation-approaches-ranked.md) Part 3.
De-identification must be confirmed before any transfer —
[`../handbook/admin-guide/de-identification.md`](../handbook/admin-guide/de-identification.md).

---

## Part 9 — The storage plan that follows from all of this

### Right now, in order

| Step | Action | Frees / costs | Result |
| --- | --- | --- | --- |
| 1 | **Re-measure.** `df -h /c` before any download | — | 27 GB is already 6 GB below what the plans assume |
| 2 | **Reclaim `data/slides/`** — 15 of 16 identical copies | **+12.5 GB** | ~39 GB free |
| 3 | Check whether the upload path should de-duplicate on `sha256` | — | stops it recurring |
| 4 | Run **approach 5** (< 1 GB) and **approach 6** (~5 GB) | −6 GB | ~33 GB free, two approaches done |
| 5 | Run **approach 1 + 3** (10–20 GB peak), delete BCSS raw after N02 | −20 GB peak, −500 MB kept | fits, but only alone |
| 6 | **Then** approach 4a (~8.5 GB) — never concurrently with step 5 | −8.5 GB peak | |
| 7 | **Order external storage** for approach 7: **≥ 2 × 170 GB**, i.e. a 500 GB–1 TB disk | costs money, not disk | also unblocks 4b's Gate D0 |

### What each approach needs, one line each

```
5   < 1 GB    ─┐
6   ~5 GB      │  all fit in current free space, today
3   +46 MB     │
1   10-20 GB  ─┘  fits alone, not alongside 4a
4a  ~8.5 GB       fits alone, after 1's raw data is deleted
7   40-170 GB     external disk, and buy it now
4b  ~400 GB       external disk, Gate D0, not scheduled
2   ~300-400 GB   plus a GPU; not viable regardless
```

**One 1 TB external disk unblocks approach 7, satisfies 4b's Gate D0, and leaves room for backups.**
It is the single cheapest thing on this page that changes what is possible.

---

## Part 10 — Provenance of every number

Because these documents are strict about this distinction, and should stay strict.

| Number | Status | Source |
| --- | --- | --- |
| 27 GB free, 476 GB total, 95 % used | **Measured** | `df -h /c`, 2 Sep 2026 |
| 35 GB cohort, 36 slides; 830,790,030 B per slide | **Measured** | `du`, `ls -la` on `images/` |
| 16 identical copies of `CAN_00251_26_A.svs` = 13.3 GB | **Measured** | `data/slides/*.json` filenames + byte sizes |
| 0.39 s/tile CPU feature extraction | **Measured** | approach 1: ~14,000 tiles in ~1.5 h |
| ~17.8 KB per uint8 PNG tile | Derived | 250 MB ÷ 14,000 tiles |
| BCSS ~4 GB download, 5–10 GB with masks | From plan | [`approach-1-training-plan.md`](approach-1-training-plan.md) Part 2 |
| BEETLE 147.2 GB / 1.9 GB / ~300 GB peak / 400 GB gate | From plan | [`approach-4b-training-plan.md`](approach-4b-training-plan.md) Parts 1–2 |
| Tissue areas 17.5 / 84.8 / 315 mm² | From comparison doc | [`segmentation-approaches-comparison.md`](segmentation-approaches-comparison.md) |
| 287,000 tile census; 31 h to embed all | **Arithmetic** on the above | shown in Part 0 |
| Approach 5 timings and sizes | **Arithmetic** on measured constants | shown in Part 6 |
| Approach 6 VALIS install and intermediate sizes | ⚠ **Estimate — unverified** | verify on case 1 before trusting |
| Approach 7 client-slide volumes | **Arithmetic** on 830 MB/slide | case count is the unknown, not the size |
| Dense-inference 60–70 h/slide; 5–8 min/slide student | From comparison doc | ⚠ inconsistent with 0.39 s/tile — see Part 0 |
| BRACS DCIS ROI set: **790 files, 10.13 GB** (train 665 / val 40 / test 85) | **Measured** | FTP listing, 3 Sep 2026 |
| BRACS licence: **non-commercial**, not the paper's CC0 | **Verified** | its download page, 3 Sep 2026 |
| All Dice figures, every approach | **Estimates, not measurements** | [`segmentation-approaches-comparison.md`](segmentation-approaches-comparison.md) Part 6 — and approach 5 is the thing that would replace them with measurements |

### The three numbers to measure before believing any schedule here

1. **The batched inference rate**, on one slide. It decides whether per-slide scoring is 8 minutes or
   52, and the two published figures in this repo disagree by 6×.
2. **Teacher inference on 3 regions**, before committing to 4a's "6–18 h" overnight.
3. **VALIS on one case**, before believing approach 6's half-day or its ~5 GB.
