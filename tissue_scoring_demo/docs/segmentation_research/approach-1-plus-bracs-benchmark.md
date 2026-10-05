# BCSS-Only Ships; BRACS Measures the Ceiling It Can't Reach

> Created: 3 Sep 2026 | Audience: whoever decides what trains the shipped model.
> **Supersedes an earlier plan in this conversation that combined BCSS + BRACS into one shippable
> model.** That plan assumed BRACS was CC0, per the paper. **It is not** — verified 3 Sep 2026 at the
> BRACS download page: *"The BRACS dataset may be used only for non-commercial research."* The paper's
> licence claim does not hold at the source that actually governs use.
> Companion: [`approach-1-training-plan.md`](approach-1-training-plan.md) (the build plan this document
> does not change), [`approach-4a-training-plan.md`](approach-4a-training-plan.md) (BEETLE, the other
> non-commercial class-`1` source), [`segmentation-approaches-comparison.md`](segmentation-approaches-comparison.md)
> Parts 3 and 7 (where the licence table and the recommendation are now corrected).
> Feeds: guide step 9 = **code step 8**, `backend/app/pipeline/step08_tissue_type_segmentation/`.

---

## Part 0 — What changed, in one sentence

BRACS cannot be pooled into training data for a model that ships, because it is non-commercial —
so the plan becomes **BCSS alone trains the shipped model**, and **BRACS becomes a benchmark**, used
only to measure how far short of BRACS's own three-pathologist-consensus ceiling the shipped model
falls on class `1`. Same role TIGER and BEETLE already have in this project: `benchmarks/`, never
imported by `backend/`.

### Why this matters beyond one dataset

A scoping review of public breast H&E WSI datasets found BRACS is the **only** one that annotates
DCIS as its own class. It is non-commercial. So are BEETLE (CC BY-NC-SA), TIGER (CC BY-NC) and BACH
(CC BY-NC-ND).

**No commercially-usable public source of DCIS annotation exists.** Every route to a *trained and
shippable* class `1` runs through either BCSS (0.129% of pixels, one patient), our own clicked tiles
(approach 5), or client annotation (approach 7). BRACS, BEETLE and TIGER can inform the project — as
teachers, as benchmarks, as ceiling measurements — but none of them can be baked into shipped weights.

---

## Part 1 — What ships: BCSS alone, unchanged from approach 1

**No change to [`approach-1-training-plan.md`](approach-1-training-plan.md).** BCSS is CC0 and stays
the only training source for the shipped checkpoint:

```
class 2  invasive         ← BCSS pixel masks, majority vote
class 1  non-invasive     ← BCSS pixel masks (dcis + normal_acinus_or_duct — thin, one patient's DCIS)
class 0  non-epithelium   ← BCSS pixel masks
```

The binding constraint is exactly what approach 1 Part 2 already measured: class `1` is 237:1 against
class `2`, and its DCIS comes from a single patient in a training institution. That does not change
by anything in this document. What changes is how you find out whether it matters.

---

## Part 2 — What BRACS is for instead: measuring, not training

BRACS gives you the thing approach 1 has never had: **a class-`1` set with real patient spread and
three-pathologist consensus**, to measure the shipped model against.

```
790 DCIS ROIs        ┐
507 ADH               │
756 FEA                ├─ 3,890 non-invasive ROIs, 151 patients, single scanner (Aperio AT2)
517 UDH               │
836 Pathological Benign┘
649 Invasive Carcinoma ROIs  ← not used; BCSS already has pixel-mask invasive
```

### Step 1 — Build an eval-only export, not a training export

Same exporter approach 1 already has (`app.common.stains`, `step06...deconvolution.separate`, the
H-channel — nothing new to write here), but the output goes to a benchmark manifest, never to
`data/tiles/` where training reads from.

1. **Epithelium-filter first.** A DCIS-labelled ROI is 20–50% epithelium; the rest is stroma and fat.
   Run a cheap epithelium-vs-not filter — trained on BCSS's `tumor + dcis + normal_acinus_or_duct`
   pixel masks, which is a task BCSS is good at even though it's bad at separating in-situ from
   invasive — over every BRACS ROI, and keep only tiles it calls epithelium. Skipping this step
   teaches nothing except "stroma looks like whatever the ROI's slide-level label says."
2. Collapse the six non-invasive BRACS subtypes to class `1`; keep the invasive ROIs out entirely —
   don't use them for class `2`, and don't use them at all here. Their ROI-level label risks pulling
   adjacent DCIS into an "invasive" bucket, which is the same contamination this project already
   worries about in BCSS's own `tumor` label.
3. Downsample 0.25 → 0.5 µm/px, same as BCSS.
4. Store the manifest under `benchmarks/bracs/eval_tiles/`, tagged `source=BRACS, split=eval_only`.

### Step 2 — Report class `1` separately, per source

Run the BCSS-trained checkpoint against the BRACS eval tiles and report:

| | BCSS held-out (existing) | BRACS eval (new) |
| --- | --- | --- |
| class `1` patient count | 1 | 151 |
| class `1` recall / precision | (already reported) | **new number** |
| invasive ↔ non-invasive confusion | (already reported) | **new number** |

This is the first time this project has a class-`1` number measured against more than one patient.
It will very likely be worse than the BCSS number — that's expected, and it's the point. A model
that looks fine on BCSS and falls apart on BRACS has been telling you it learned one patient, not
DCIS.

### Step 3 — Run the same dataset-leakage check either way

Even in eval-only use, confirm the model isn't keying on "which dataset/scanner is this" rather than
tissue. Train a throwaway classifier to predict source (BCSS vs BRACS) from the same frozen features.
If that's trivially easy, the H-channel/jitter design isn't fully closing the domain gap, and the
BRACS eval number should be trusted less, not more.

---

## Part 3 — What this buys, and what it doesn't

| | Delivered |
| --- | --- |
| A shippable class `1` | ❌ still just BCSS's one patient — unchanged from before this document |
| A ceiling measurement for class `1` | ✅ new — 151 patients, 3-pathologist consensus |
| Evidence of overfitting to BCSS's one DCIS patient | ✅ new — this is the actual value |
| Anything to feed approach 4a's ablation | ⚠️ partial — a BRACS eval number is a better yardstick than BCSS-held-out, but it's still not our scanner |
| Progress on gate zero / approach 5 / approach 7 | ❌ none — those remain the routes to a real fix |

**If the BRACS eval number is close to the BCSS number:** weak but real signal that class `1` isn't
purely memorising one patient. Worth reporting, not worth trusting alone.

**If the BRACS eval number is much worse:** expected, and it's the argument for approach 5 (our own
clicked tiles, patient-diverse in the sense that matters — our scanner) and approach 7 (client
annotation) over anything this document can produce. BRACS tells you the BCSS-only model is thin; it
cannot make it thicker.

---

## Part 4 — Storage and licence discipline

| What | Where | Why |
| --- | --- | --- |
| BRACS DCIS ROI PNGs (790 files, 10.13 GB) | `benchmarks/bracs/dcis/` | already downloading — non-commercial, so outside `backend/` |
| Eval-tile manifest built from them | `benchmarks/bracs/eval_tiles/` | derived asset, same licence as its source |
| BCSS-trained checkpoint | `models/tissue_type/` | unchanged — still CC0-clean |
| CI rule | unchanged from the project's existing discipline | `backend/` must not import anything under `benchmarks/` |

Nothing here adds disk beyond what's already downloading. No new licence exposure — BRACS never
enters a training loop that produces shipped weights, only an eval loop that reads a checkpoint and
writes numbers.

---

## Part 5 — Order

1. Let the current BRACS DCIS download finish (`benchmarks/bracs/dcis/`, in progress).
2. Build/confirm approach 1's BCSS-only shipped checkpoint — no change from its existing plan.
3. Build the epithelium filter (reuses BCSS pixel masks — no new data).
4. Export the BRACS eval tiles, report class `1` per-source, run the leakage check.
5. Use the BRACS eval number, not the BCSS-held-out number, as the honest headline for class `1` —
   until approach 5's clicked tiles exist, which will be the more honest number still, since it's on
   our own scanner.

This document does not change the recommended build order in
[`segmentation-approaches-implementation.md`](segmentation-approaches-implementation.md) Part 10 —
gate zero, gate 0, approach 5, then approach 1+3 in parallel remain first. It adds one cheap,
non-commercial-safe measurement step alongside them.
MDEOF
echo "written"; wc -l "/c/Users/Coditas/Desktop/Healthcare_Projects/Cancer_Scoring_System/tissue_scoring_demo/docs/segmentation_research/approach-1-plus-bracs-benchmark.md"