# Fix 1 + Fix 2 — plan

**The failure being fixed.** On `CAN_00270_26_H&E` the served tile model filled a
pathologist's DCIS contour with invasive. That is not a surprise: the checkpoint's own
manifest records `dcis_called_invasive = 0.224` on held-out data. Two causes, and this
plan addresses both.

**Live state:** [../progress/PROGRESS.md](../progress/PROGRESS.md) · **Results:** `RESULTS_224_VS_448.md`
(written by the last stage)

---

## Fix 1 — the consensus decides the lesion, the teacher only finds the tissue

### What was wrong

Class 1 was taught from BRACS and BACH regions whose pixels BEETLE labelled. Where
BEETLE's verdict contradicted the human ROI consensus, **the whole region was deleted**:

| tree | rejected under the old rule |
| --- | --- |
| BRACS `dcis` | 33 of 153 (22%) |
| BRACS `ic` | ~10% |
| BRACS `normal` | ~30% |
| **BACH in-situ** | **47 of 100** |

That filter is not neutral. A large solid or comedo duct read 112 µm at a time shows no
duct wall, so the teacher calls it invasive, so the region was rejected — **the filter
selected against exactly the morphology the student then gets wrong.** BEETLE's own bias
runs one way and was already measured: it disagreed with BACH's two pathologists on 47
of 100 in-situ images, every one by calling invasive.

### What it is now

```
BRACS/BACH region of interest
        │
        ├── BEETLE:  invasive ∪ in-situ  ────►  epithelium is HERE
        │
        └── the ROI consensus (3 pathologists) ────►  and it is THIS lesion
                    │
        DCIS ROI  ──┼──►  class 1     (code `dcis`)
        IC ROI    ──┼──►  class 2     (code `tumor`)
        normal ROI ─┴──►  class 1     (code `normal_acinus_or_duct`)
```

A region the teacher misreads is **relabelled, not deleted**. A region is still rejected
when the teacher found almost no epithelium at all — a resolution or stain fault, which
is a different kind of problem — or when no tile survived the vote.

### The cost, stated rather than buried

BRACS labels a region by its **predominant** lesion, so a DCIS region holding a true
focus of invasion now teaches that focus as in-situ. That error is real and bounded by
how often the consensus is incomplete. It is the smaller of the two on offer: the
alternative was deleting the region, which biased against one morphology systematically.
Each region records `teacher_overruled_share`, so the regions where the two sources
disagree are a **sorted review queue** rather than a silent decision.

### Where it lives

| file | change |
| --- | --- |
| `bracs_app/config.py` | `RoiTree.epithelium_code`, set on all six trees |
| `bracs_app/labels.py` | `targets_for(epithelium_target=…)`, `recode_epithelium` |
| `bracs_app/pipeline.py` | `run_region` applies it; `verdict` records the disagreement instead of rejecting on it; `relabel_run` |
| `scripts/export_to_approach1.py` | cache invalidated on the rule; `--split-epithelium` restores the old behaviour |

Four guarantees, each enforced rather than documented:

1. **The collapse target must remap into `teaches`** — asserted per region in
   `run_region`, and over the whole table in `tests/test_labels.py`. A tree collapsing
   onto a class its consensus cannot vouch for would manufacture the unadjudicated label
   `teaches` exists to exclude.
2. **`by_source_authority` is unchanged** — a DCIS tree still teaches class 1 only.
3. **The cache cannot silently defeat the change.** A run written under the old rule is
   refused by `cached_manifest`. Without that check the change would appear to have been
   made and would have changed nothing.
4. **It is reversible.** `relabel_run` preserves `manifest.pre-collapse.json`, so
   `--split-epithelium` reproduces the earlier export from cache rather than
   re-segmenting.

**Re-running is minutes, not a day.** The collapse is a pure relabelling of an argmax
already on disk, so cached regions are re-coded at ~0.2 min per 100 rather than
re-segmented at 25 s each.

---

## Fix 2 — a field of view that can contain a duct

### What was wrong

The served window is 224 px at 0.5 µm/px = **112 µm**. A high-grade solid or comedo duct
is 300–1500 µm across. A window centred inside one contains a sheet of tumour cells with
no basement membrane, no periductal stroma and no rim — physically indistinguishable from
invasive. The model only gets DCIS right where a duct edge happens to fall inside the
window, which is exactly the speckled pattern seen on the slide.

The repo's own sweep already said so, using **a fifth of the data**:

| | train tiles | in-situ recall | invasive recall | Dice inv |
| --- | --- | --- | --- | --- |
| 112 µm (served) | 13,683 | 0.788 | 0.788 | 0.861 |
| 224 µm | 2,906 | **0.818** | **0.810** | **0.882** |

### What it is now

Two exports, `tile_px` fixed at 224 so the backbone is untouched and only the resolution
moves — the same convention the earlier 224 µm experiment used:

| export | geometry | field of view |
| --- | --- | --- |
| `data/224_um/` | 224 px @ 1.0 µm/px | 224 µm |
| `data/448um/` | 224 px @ 2.0 µm/px | 448 µm |

**A known limit, by arithmetic — and it is bigger than it first looks.** A 448 µm
window is 896×896 px at 0.5 µm/px, and much of the corpus is smaller than that. Measured
over the region trees on disk:

| tree | tiles at 224 µm | tiles at 448 µm | regions yielding **zero** at 448 µm |
| --- | --- | --- | --- |
| `dcis` | 734 | 127 | **64 of 131** |
| `ic` | 1,867 | 354 | 34 of 163 |
| `normal` | 714 | 112 | **82 of 169** |
| each BACH tree | 600 | 100 | 0 (one window per image, exactly) |

So **half the BRACS regions are physically too small to yield a single 448 µm window**,
and class 1 at 448 µm lands at roughly 400 tiles against ~2,500 at 224 µm — before the
70% usable vote drops more.

**This confounds the very comparison being run.** If 448 µm scores worse, it will be
hard to separate "the field of view is too coarse" from "the class had 400 tiles". The
comparison is still worth having — it is the cheapest way to find out whether more
context helps at all — but it is a sample-size contest as much as a geometry one, and
the results document says so beside the numbers.

**The better use of 448 µm is as a second branch, not a second export.** The served
architecture is already `concat_resnet18_mlp` — two backbones concatenated — so a coarse
*context* branch centred on the same point as a 224 µm detail branch would use the wide
field without needing many independent coarse windows, which is exactly what the corpus
cannot supply. That is the follow-up this stage's numbers should inform.

### Where it lives

| file | change |
| --- | --- |
| `scripts/02_export.py` | the G2 class-0/2 count gate now **scales with the field of view** — 1,000 tiles is a mapping fault at 112 µm and simple arithmetic at 448 µm |
| `scripts/07_split_export_by_source.py` | new: re-files one export into `{bcss,beetle,bach}/` |
| `scripts/03_features.py`, `04_train.py` | `--tiles` / `--features`, so each geometry keeps its own vectors. Explicit arguments, not environment variables, which this project does not use |

---

## Layout

The reorganisation that made "clear the exports" a safe sentence: sources, derived
artefacts and exports are now three separate places instead of one directory holding all
three.

```
original_data/                       ← the sources. 28 GB. Read-only. Nothing regenerates these.
├── bcss/{images,masks,meta}/            151 TCGA regions, human pixel labels, CC0
├── bracs/{dcis,ic,normal}/              772 ROIs, flat (BRACS's own splits in .bracs_splits/)
└── bach/{InSitu,Invasive,Normal}/       300 images, Porto

tissue_label_generation/data/ ← derived. Re-derivable: runs/ at 25 s a region, regions/ in minutes.
├── runs/<roi_id>/                       BEETLE's segmentation + manifest.pre-collapse.json
└── regions/<tree>/{images,masks}/       region pairs in BCSS's 22 label codes

tissue_type_model_training/data/   ← exports only. All of it re-cuttable.
├── 224_um/{bach,bcss,beetle}/
└── 448um/{bach,bcss,beetle}/
```

BRACS's own train/val/test split is **not** a loss: nothing here ever divided on it —
`bcss.bracs_test_cases` holds out 25% of *patients* by hashing patient ids, because a
BRACS split puts regions of one patient on both sides and a model scored across that
boundary is scored on its own memory. The assignment is preserved in each tree's
`.bracs_splits/` regardless.

Both path changes use "first candidate that exists wins", so a clone with the old layout
keeps working and neither location is asserted as the only one.

---

## Stages

| # | stage | what it does |
| --- | --- | --- |
| S1 | region trees | segment the uncached regions; re-code the cached ones |
| S2 | 224 µm export | BCSS + six borrowed trees, then split by source |
| S3 | 448 µm export | the same at 2.0 µm/px |
| S4 | features | imagenet + simclr, per field of view |
| S5 | train | one head per field of view |
| S6 | results | read the reports, write the comparison |

### Re-planned at 15:02 against a 17:00 deadline

The straight sequential order finished S1 at ~17:00 and S2 at ~18:15 — too late. Three
changes, and the first two are the ones worth remembering.

**1. Two things run in parallel, because they are genuinely independent.** S2's expensive
half is cutting the 151 BCSS regions, and those tiles come straight from `original_data`
— no region tree, no teacher. So it runs *alongside* S1's DCIS segmentation, and
`--reuse-bcss` picks the rows up afterwards. Measured while both ran: **~2.9 of 8 logical
cores**, no contention penalty, because one is nnU-Net threads and the other is numpy and
disk. Two *teachers* in parallel would have thrashed; a teacher and a tiler do not.

**2. `ic`'s remaining 131 segmentations are deferred**, via a new `--cached-only`. `ic`
supplies class 2, which BCSS already has in abundance; class 1 is the starved one. An
hour of teacher spent on invasive regions buys far less than the same hour spent on DCIS,
so `ic` is assembled from its 189 cached or re-codable regions and the rest is a
follow-up run. They are recorded as `skipped`, deliberately kept apart from `rejected`,
so a later reader cannot mistake "never looked at" for "judged and thrown out".

**3. The order is a priority order.** 224 µm is carried all the way to a fitted head
before 448 µm starts, and within 224 µm the imagenet head is fitted and its results
written *before* the simclr feature pass begins. Each pass is ~25 min, so doing both
first would have put the first real number past the deadline. imagenet is the documented
G3→G4 path and stands alone; simclr earns its 25 minutes only as the second half of the
concat pair the served v3 uses. If the machine stops at any point, what survives is the
most useful thing that was finishable.

A stage that fails stops the chain and says so in `PROGRESS.md`. The one exception is a
G2 gate complaint: the gate is a statement *about the data*, and this run is what
produces the evidence for it, so it is recorded as a warning and the chain continues.

**Live timings are in [../progress/PROGRESS.md](../progress/PROGRESS.md).** Expected: S1 ~15:54, S2 ~16:20, a
fitted 224 µm head ~16:50, the concat re-fit ~17:40, the 448 µm arm ~18:30.

---

## What this will and will not have established

- The two heads differ from the served one in **both** the label rule and the field of
  view, so a difference cannot be attributed to either alone. The 224 µm vs 448 µm
  comparison **is** controlled — same rule, same sources, only geometry differs.
- Every borrowed label is still BEETLE's placement of epithelium. Fix 1 changes who
  names the lesion, not who finds the tissue, so the ceiling remains BEETLE's
  epithelium-versus-stroma accuracy.
- **Nothing here scores `CAN_00270_26_H&E`.** Running step 8 over that slide with one of
  these heads is the measurement that answers the original question, and it is the
  obvious next step once a head exists.
- Fix 3 (component-level DCIS-vs-invasive) is untouched and remains the fix that
  addresses the cause rather than the representation. If DCIS is still called invasive
  after these two, that is where to go next.
