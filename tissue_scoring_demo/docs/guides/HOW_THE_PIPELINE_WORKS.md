# How tissue segmentation works here, end to end

Plain-language walkthrough: from the pictures you downloaded, to a trained model, for
both the 224 µm and 448 µm versions.

**The one-sentence version.** There are two folders of code. One is a **label factory** —
it looks at a picture and works out which pixels are epithelium. The other is a **tile
factory and trainer** — it chops those labelled pictures into small squares and fits a
model on them.

```
original_data/            tissue_label_generation/       tissue_type_model_training/
(pictures)          →     (makes the masks)                →    (cuts tiles, trains)
                          "the label factory"                   "the tile factory"
```

---

## Stage 0 — `original_data/`: what you actually start with

Three sources, and **they are not the same kind of thing**. This difference drives
everything that follows.

| folder | what it is | does it come with labels? |
| --- | --- | --- |
| `bcss/` | 151 pictures from TCGA + **masks drawn by pathologists** | **Yes — per pixel** |
| `bracs/{dcis,ic,normal}/` | 772 pictures from an Italian centre | **No.** Only a folder name |
| `bach/{InSitu,Invasive,Normal}/` | 300 pictures from Porto | **No.** Only a folder name |

That's the whole problem in one table.

For BCSS, somebody has already coloured in every pixel: *this bit is tumour, this bit is
stroma, this bit is fat.* You can use it immediately.

For BRACS and BACH you have a picture and **one word**. The word comes from the folder it
sits in — `bracs/dcis/BRACS_1247_DCIS_1.png` means *"three pathologists looked at this
whole region and said the main thing in it is DCIS."* Nobody said **where** in the picture
the DCIS is. A typical region is only about half epithelium; the rest is stroma, fat and
inflammation.

So you cannot train on it yet. **You need a mask, and nobody drew one.** Making that mask
is Stage 1.

> `original_data/` is read-only. Nothing in the project ever writes to it. It's 28 GB and
> nothing can regenerate it, which is why it lives outside both code folders.

---

## Stage 1 — `tissue_label_generation/`: the label factory

**This is the only place any segmentation happens during data preparation.**

It takes one picture with no labels and produces one mask. It does this with **BEETLE**, a
released nnU-Net (a segmentation neural network) that was trained by someone else on
breast tissue. We use it as a **teacher**: it looks at the picture and colours in every
pixel with one of five answers.

### What happens to one region

```
original_data/bracs/dcis/BRACS_1247_DCIS_1.png     a picture, no labels
        │
        │  1. resample to 0.5 µm per pixel  (BEETLE was trained at this scale)
        │
        │  2. run BEETLE  →  every pixel gets one of five answers:
        │        background · other tissue · in-situ epithelium
        │        · invasive epithelium · necrosis
        │
        │  3. translate those five into BCSS's own 22-code vocabulary,
        │     so BRACS masks and BCSS masks speak the same language
        │
        │  4. ***Fix 1***  merge the two epithelium answers into ONE,
        │     and let the folder name decide which one it becomes
        │
        ▼
regions/dcis/images/BRACS_1247_DCIS_1.png    the picture, resampled   (~4.5 MB)
regions/dcis/masks/ BRACS_1247_DCIS_1.png    the mask                 (~15 KB)
```

Those two files are the same size and line up pixel for pixel. That pair is what "region"
means.

### Step 4 is the important one — how the two sources split the job

BEETLE gets asked **two different questions**, and it's only good at one of them:

| question | who answers it | why |
| --- | --- | --- |
| **Where is the epithelium?** (vs stroma, fat, inflammation) | **BEETLE** | A texture question at cell scale. It's reliable at this. |
| **Which lesion is it?** (invasive vs in-situ) | **the three pathologists** | Needs to see a whole duct wall. A pixel model reading a small patch structurally can't. |

So we take BEETLE's two epithelium answers, **glue them together** into a single
"epithelium is here" answer, and then the folder name says what that epithelium *is*:

```
inside a  dcis/  region  →  all epithelium becomes  "DCIS"        (class 1)
inside an ic/    region  →  all epithelium becomes  "invasive"    (class 2)
inside a  normal/region  →  all epithelium becomes  "normal duct" (class 1)
```

**Why this matters.** BEETLE tends to call big solid DCIS "invasive", because a solid duct
seen close up looks like a sheet of tumour cells. It disagreed with BACH's pathologists on
**47 of 100** in-situ images, always in that direction. The old rule deleted every region
where BEETLE and the pathologists disagreed — which deleted exactly the hard DCIS the model
most needed to see. Now the pathologists win the argument and the region is kept.

Every region records how much the pathologists overruled BEETLE, so you can go back and
look at the ones where they disagreed most.

### BCSS skips this stage completely

It already has human masks. It goes straight to Stage 2. **BEETLE never touches BCSS.**

### What Stage 1 leaves on disk

```
tissue_label_generation/data/
├── runs/<region_id>/          BEETLE's raw output per region + a record of what was done
│                              THE EXPENSIVE THING. ~18 s of neural network per region.
└── regions/<tree>/
    ├── images/                the resampled pictures
    └── masks/                 the masks
```

`runs/` is the cache. Because it exists, changing the labelling rule is a **re-label**
(seconds) instead of a **re-segmentation** (hours). `regions/` is the same pixels tidied
into one folder per lesion type, which is the format Stage 2 reads.

---

## Stage 2 — `tissue_type_model_training/`: the tile factory and the trainer

A neural network can't eat a 25-megapixel picture. It eats small fixed-size squares. This
folder chops the region pairs into squares, gives each square **one label**, and fits the
model.

### So what do "data → region → export" mean?

| word | what it is | example |
| --- | --- | --- |
| **data** | a raw picture, no labels | `original_data/bracs/dcis/BRACS_1247_DCIS_1.png` |
| **region** | one picture **+ one mask**, aligned | `regions/dcis/{images,masks}/BRACS_1247_DCIS_1.png` |
| **export** | thousands of small squares cut from regions, each with **one class** | `h_channel/224um/beetle/non_invasive_epithelium/BRACS_1247_DCIS_1__r003c004.png` |

One region becomes tens or hundreds of tiles. The name `__r003c004` is just the grid
position: row 3, column 4.

### How one region becomes tiles

```
region (picture + mask, aligned)
    │
    │  1. resample to the target scale
    │        224 µm export  →  1.0 µm per pixel
    │        448 µm export  →  2.0 µm per pixel
    │
    │  2. lay a grid of 224 × 224 pixel squares, no overlap
    │
    │  3. for each square, LOOK AT THE MASK UNDERNEATH and vote  ← masks used here
    │
    │  4. convert the picture square into what the model eats
    │
    ▼
one PNG per surviving square, filed in a folder named after its class
```

### Step 3, the vote — this is where masks are needed

For each square we look at the mask under it and count what's there:

| rule | meaning |
| --- | --- |
| less than **70%** of the square has any label at all | **drop it** — too much unlabelled background |
| **≥ 50%** of the labelled pixels are invasive | label the square **invasive** |
| **≥ 50%** are in-situ | label it **in-situ (DCIS)** |
| **≥ 80%** are non-epithelium | label it **non-epithelium** |
| none of the above | **drop it** — genuinely mixed, no honest answer |

**Why 50% for tumour but 80% for stroma?** A square that's 60% stroma and 40% tumour is a
*tumour square with some stroma in it*. Calling it stroma would teach the model to ignore
the edge of every tumour. So it's harder to earn the "stroma" label than the "tumour" one.

Squares with no majority are **thrown away rather than forced into a class**. A square
that's genuinely half duct and half stroma has no right answer, and inventing one teaches
the model that the boundary is its own thing.

### Step 4 — what the model actually accepts

**Not a normal colour picture.** The model only ever sees the **haematoxylin channel** —
the blue-purple stain that marks nuclei.

```
colour square (RGB)
    │  turn brightness into "how much stain is here"  (optical density)
    │  mathematically separate the two dyes           (colour deconvolution)
    │  KEEP haematoxylin.  THROW AWAY the other dye.
    │  standardise: scale so this tile's 99th-percentile density hits a fixed target
    │  clip anything above 1.5 density, then squash to the range 0–1
    │  copy that single channel three times
    │  subtract the ImageNet mean, divide by the ImageNet standard deviation
    ▼
a 3 × 224 × 224 block of numbers  ← this is what goes into the network
```

**Why throw away colour?** Every slide in a case carries haematoxylin — the H&E one and
all five antibody-stained ones. Only the *second* dye differs (eosin on H&E, brown DAB on
the others), and on some antibodies that brown floods 75–85% of the picture. Keep only
haematoxylin and **all six slides become the same kind of image, so one model serves them
all** and never has to cope with the brown.

On disk each tile is stored as a **single-channel grey PNG** (the density squeezed into
0–255). The three-channel copy and the normalising happen when it's loaded for training.

### What Stage 2 leaves on disk

Eight exports: four fields of view times two input channels, each split by where its
tiles came from.

```
tissue_type_model_training/data/
├── h_channel/                       Ruifrok's haematoxylin channel, as uint8 density
│   ├── 112um/                       224 px at 0.5 µm/px  =  112 µm of tissue per tile
│   │   ├── bcss/<class>/*.png           from BCSS — the only human-drawn labels
│   │   ├── beetle/<class>/*.png         from BRACS — BEETLE found it, pathologists named it
│   │   ├── bach/<class>/*.png           from BACH — same, second laboratory
│   │   ├── tiles_manifest.csv           one row per tile: where it came from, its class
│   │   ├── export_summary.json          counts, geometry, provenance, and the channel
│   │   ├── features/                    the cached 512-number summaries
│   │   └── reports/                     the trained head + its scores
│   ├── 224um/                       224 px at 1.0 µm/px
│   ├── 448um/                       224 px at 2.0 µm/px
│   └── 672um/                       224 px at 3.0 µm/px
└── he/                              the sRGB photograph itself, nothing separated out
    └── 112um/ 224um/ 448um/ 672um/  (identical layout)
```

**The two halves hold the same squares.** The exporter takes its grid and its class vote
from the mask alone — the white point, the deconvolution and the quantisation all happen
*after* the vote — so for one field of view the two stores contain the same `tile_id`s
with the same labels, and only the pixels differ. That is what makes a model fitted on
one comparable, tile by tile, against a model fitted on the other; it is asserted per arm
by `scripts/10c_assert_paired.py` rather than assumed.

`<class>` is one of `non_epithelium`, `non_invasive_epithelium`, `invasive_epithelium`.

**Nothing else lives in `data/`.** The sources are in `original_data/`, the masks are in
the other folder. Everything here can be deleted and re-cut, which is what makes "clear
the exports and start again" a safe thing to say.

---

## 224 µm vs 448 µm — what's actually different

**The tile is always 224 × 224 pixels.** Only how much tissue those pixels cover changes:

| | pixels | µm per pixel | tissue covered | roughly |
| --- | --- | --- | --- | --- |
| the old model | 224 | 0.5 | **112 µm** | ~9 cells across |
| **224 µm export** | 224 | 1.0 | **224 µm** | a small duct fits |
| **448 µm export** | 224 | 2.0 | **448 µm** | a large duct plus its rim fits |

**Why bother?** Telling DCIS from invasive means seeing whether the tumour is still
*inside* a duct. A big solid DCIS duct is 300–1500 µm across. At 112 µm, a square taken
from the middle of one contains nothing but tumour cells — no duct wall, no surrounding
stroma. It is *physically impossible* to tell it from invasive. Wider squares can see the
wall.

**The catch with 448 µm, and it's real.** A 448 µm square is 896 × 896 pixels of original
data, and a lot of the source pictures are smaller than that. About half the BRACS regions
produce **zero** 448 µm tiles, and each BACH image produces exactly **one**. So the 448 µm
export has far fewer DCIS tiles than the 224 µm one (~400 vs ~2,500). If it scores worse,
that could be because the view is too coarse *or* just because it had less data — the two
can't be separated. Worth measuring anyway, but read it with that in mind.

---

## Stage 3 — making the model

Two steps, and the first is the slow one.

**1. Squash every tile into 512 numbers.** A ResNet18 (a standard image network, already
trained on everyday photographs) is run over every tile **once**, and its 512-number
summary is saved. The network itself is **frozen** — never adjusted. This is the ~25-minute
step, and it's done once per export.

**2. Fit a small classifier on those numbers.** A tiny layer that takes 512 numbers and
gives three scores — non-epithelium, in-situ, invasive. Because the heavy lifting is
already cached, this takes **minutes**, so you can try many variations cheaply.

```
tile PNG  →  frozen ResNet18  →  512 numbers  →  small head  →  3 scores
             (never changes)      (cached)       (this is what gets trained)
```

Two versions are fitted per export: one starting from a network trained on everyday
photographs (`imagenet`), one from a network trained on histology slides (`simclr`).

**How it's scored.** Whole patients are held out — never single tiles. Tiles from one
patient look almost identical, so testing on a patient you trained on measures memory, not
skill. The headline number is **`dcis_called_invasive`**: the share of held-out DCIS tiles
the model wrongly calls invasive. The currently served model scores **0.224** — that's the
22% we're trying to fix.

---

## Where masks are needed — the short answer

This trips people up, so plainly:

| where | are masks involved? |
| --- | --- |
| Stage 1, BEETLE | **Yes — this is where they're created** |
| Stage 2, the tile vote | **Yes — read, to decide each tile's single label** |
| Stage 2, training | **No.** The model sees a tile and a class name. Never a mask. |
| Serving a real slide | **No.** No mask anywhere. |

Masks are **scaffolding for producing labels**. Once each tile has its one word, the masks
have done their job and the model never sees one.

That's also why this is a **tile classifier**, not a pixel segmenter. On a real slide it
slides a window across, classifies each position, and the map you see on screen is those
per-window answers stitched together — which is why the window size matters so much, and
why Fix 2 exists.

---

## The whole thing on one page

```
 original_data/                              28 GB, read-only, irreplaceable
 ├── bcss/       151 pics + HUMAN masks ─────────────────────┐
 ├── bracs/      772 pics, no masks ──┐                      │
 └── bach/       300 pics, no masks ──┤                      │
                                      │                      │
                    ┌─────────────────▼──────────────────┐   │
                    │  tissue_label_generation/   │   │  BCSS needs no
                    │                                    │   │  teacher — it
                    │  resample to 0.5 µm/px             │   │  already has
                    │  run BEETLE → 5 classes/pixel      │   │  human masks
                    │  translate to BCSS's 22 codes      │   │
                    │  FIX 1: merge the two epithelium   │   │
                    │  classes; the folder name picks    │   │
                    │  which one it becomes              │   │
                    │                                    │   │
                    │  data/runs/      the cache         │   │
                    │  data/regions/   picture + mask ───┼───┤
                    └────────────────────────────────────┘   │
                                                             │
                    ┌────────────────────────────────────▼───┴──┐
                    │  tissue_type_model_training/            │
                    │                                           │
                    │  resample to 1.0 µm/px   or   2.0 µm/px   │
                    │  cut into 224 × 224 squares               │
                    │  vote each square against the mask        │
                    │  keep haematoxylin only, OR keep the RGB  │
                    │                                           │
                    │  data/h_channel/<fov>um/{bcss,…}/         │
                    │  data/he/<fov>um/{bcss,…}/                │
                    │            │                              │
                    │  frozen ResNet18 → 512 numbers/tile       │
                    │  small head → 3 classes                   │
                    │            │                              │
                    │  two models: one at 224 µm, one at 448 µm │
                    └───────────────────────────────────────────┘
```

---

## Related documents

- [../design/FIX1_FIX2_PLAN.md](../design/FIX1_FIX2_PLAN.md) — why each change was made, and what it costs
- [../progress/PROGRESS.md](../progress/PROGRESS.md) — the live run
- `RESULTS_224_VS_448.md` — the numbers, written when the models are fitted
