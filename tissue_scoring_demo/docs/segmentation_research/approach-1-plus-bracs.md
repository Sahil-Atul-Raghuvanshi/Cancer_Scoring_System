# BCSS + BRACS — the plan, and the licence finding that changed it

> Created: 3 Sep 2026 | Audience: whoever decides what trains the shipped model.
> **This document was requested as "BCSS + BRACS = one shippable model with a trainable class `1`".
> That plan does not survive its own licence check**, and the correction is the most important thing
> on this page. What follows is the plan as proposed, what killed it, and what survives.
> Companion: [`approach-1-training-plan.md`](approach-1-training-plan.md) (unchanged by this),
> [`approach-4a-training-plan.md`](approach-4a-training-plan.md) (BEETLE — the other non-commercial
> class-`1` source), [`segmentation-approaches-comparison.md`](segmentation-approaches-comparison.md)
> Parts 3 and 7 (licence table and recommendation, both corrected 3 Sep 2026).
> Feeds: guide step 9 = **code step 8**, `backend/app/pipeline/step08_tissue_type_segmentation/`.

---

## Part 0 — The finding, first, because it inverts the conclusion

The plan below was built on BRACS being **CC0**, which is what the BRACS paper states and what this
project's own licence table recorded.

**The BRACS download page says otherwise:**

> *"The BRACS dataset may be used only for non-commercial research."*

Verified 3 Sep 2026, after registration. The download page governs use; the paper does not. Coditas
is building this for OncoStem, a paying client, and the product ships — that is commercial use.

**So BRACS cannot train a shipped model.** It joins TIGER, BEETLE and BACH in `benchmarks/`.

### The consequence is larger than one dataset

A [scoping review of public breast H&E WSI datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC10884505/)
finds BRACS is the **only** public dataset that annotates DCIS as its own class. Every dataset that
carries DCIS annotation is non-commercial:

| Dataset with DCIS annotation | Licence | Ships? |
| --- | --- | --- |
| **BRACS** | non-commercial *(verified at download, 3 Sep 2026)* | ❌ |
| **BEETLE** | CC BY-NC-SA 4.0 | ❌ |
| **TIGER** | CC BY-NC 4.0 | ❌ |
| **BACH / ICIAR2018** | CC BY-NC-ND | ❌ |
| **Radboud** (69 DCIS WSIs) | unpublished; RUMC precedent is NC | ❌ |
| **BCSS** | **CC0** ✅ | ✅ — but 0.129 % of pixels, **one patient** |

> **There is no commercially-usable public source of DCIS annotation.** That makes client annotation
> ([`segmentation-approaches-ranked.md`](segmentation-approaches-ranked.md) Part 3, approach 7) not the
> *best* route to a trained-and-shippable class `1` but the **only** one. It reframes approach 7 from
> "highest ceiling, longest lead time" to "the only door out of this room."

One email is still worth sending: **ask ICAR-CNR for commercial terms on BRACS**, exactly as the
BEETLE licence email is already recommended. Both cost nothing and both run on someone else's clock.

---

## Part 1 — The plan as proposed *(recorded, because the reasoning still holds if a licence lands)*

The idea was sound and worth keeping on file. BCSS and BRACS are complementary in exactly the right
way — each one's fatal gap is the other's surplus:

| | class `2` : class `1` | supplies class `0`? | scanner | patients with DCIS |
| --- | --- | --- | --- | --- |
| **BCSS** | **237 : 1** | yes, abundantly | one TCGA era | **1** |
| **BRACS** | **1 : 6** | **no, never** | one Aperio AT2 | **151** |

Combined they are roughly balanced and span two scanner eras instead of one. The configuration:

```
class 2  invasive        ← BCSS pixel masks, majority vote
class 1  non-invasive    ← BRACS ROIs: N, PB, UDH, FEA, ADH, DCIS collapsed to 1
class 0  non-epithelium  ← BCSS pixel masks
```

Three design rules that were part of the plan and remain correct in any use of BRACS:

**1. Never take class `2` from BRACS's 649 invasive ROIs.** Invasive carcinoma frequently has DCIS
adjacent to it, and an ROI-level "invasive" label would drag that DCIS into class `2` — the same
contamination this project already suspects in BCSS's own `tumor` label. BCSS has pixel masks for
invasive; use those.

**2. The epithelium filter is mandatory, not a refinement.** A DCIS-labelled ROI is 20–50 %
epithelium; the rest is stroma and fat. Tile it naively and you teach the model **stroma = DCIS**.
Train a cheap 2-class epithelium-vs-not filter on BCSS pixel masks (`tumor` + `dcis` +
`normal_acinus_or_duct` = epithelium), run it over the BRACS ROIs, and keep only the tiles it calls
epithelium. BCSS is poor at separating in-situ from invasive but perfectly good at *finding*
epithelium — a different task, with abundant data.

**3. Watch for dataset-as-label leakage.** If every class-`1` example comes from BRACS and every
class-`0`/`2` from BCSS, the model can score well by detecting *which scanner and lab it is looking
at* and never learn tissue at all — then collapse on our slides, where every tile is one scanner.
The H-channel input and the `alpha`/`beta`/`gamma` jitter push against this but do not guarantee it.
**Test it directly:** train a throwaway classifier to predict the source dataset from the same frozen
features. If that is easy, the class-`1` number is inflated.

---

## Part 2 — What survives: BCSS ships, BRACS measures

BRACS cannot feed shipped weights. It can still do something this project has never been able to do:
**measure class `1` against more than one patient.**

### What ships — unchanged

[`approach-1-training-plan.md`](approach-1-training-plan.md), untouched. BCSS is CC0 and remains the
only training source for the shipped checkpoint, with class `1` as thin as Part 2 of that plan
measured it. Nothing here improves that.

### What BRACS becomes — an eval-only benchmark

```
790 DCIS ROIs           ┐
507 ADH                 │
756 FEA                 ├─ 3,890 non-invasive ROIs, 151 patients, 3-pathologist consensus
517 UDH                 │
836 Pathological Benign ┘
649 Invasive Carcinoma    ← unused
```

**Step 1 — Export eval tiles, not training tiles.** Same exporter (`app.common.stains`,
`step06…deconvolution.separate` — nothing new to write), epithelium-filtered per rule 2 above, six
non-invasive subtypes collapsed to class `1`, downsampled 0.25 → 0.5 µm/px as BCSS already is. Output
goes to `benchmarks/bracs/eval_tiles/`, tagged `split=eval_only`, **never** to the path training
reads from.

**Step 2 — Report class `1` per source.**

| | BCSS held-out *(existing)* | BRACS eval *(new)* |
| --- | --- | --- |
| class `1` patients | 1 | **151** |
| class `1` recall / precision | already reported | **new** |
| invasive ↔ non-invasive confusion | already reported | **new** |

**Step 3 — Run the leakage check** from rule 3, even in eval-only use.

### How to read the result

- **BRACS number close to the BCSS number** — weak but real evidence that class `1` is not purely
  memorising one patient. Report it; do not lean on it.
- **BRACS number much worse** — the expected outcome, and the *point*. A model that looks fine on
  held-out BCSS and falls apart across 151 patients has been telling you it learned one patient, not
  DCIS. That is the argument for approach 5 and approach 7, made with a number instead of an
  assertion.

Either way this measures *"can we separate DCIS from invasive on Aperio AT2 lesion crops"* — **not**
*"on our Morphle IHC slides."* Only approach 5's clicked tiles answer the second question, and they
remain the more honest headline once they exist.

---

## Part 3 — Does this remove the need for 4a?

**No — it defers it, and the licence finding partly rehabilitates it.**

When BRACS looked CC0, it dominated BEETLE on the axis that mattered (licence). It does not. Both are
non-commercial, so the comparison reverts to quality, where BEETLE wins on every axis:

| | BRACS | BEETLE (4a) |
| --- | --- | --- |
| Label form | ROI-level, no masks | **pixel masks** |
| Scanners | 1 (Aperio AT2) | **7** |
| Patients | 151 | **527** |
| Download | 10.13 GB (DCIS ROIs) | **1.9 GB** (`model.zip`) |
| Runs on *our* slides? | no | **yes** — scanner adaptation |
| Licence | non-commercial | non-commercial (CC BY-NC-SA) |

For a `benchmarks/`-only asset, **4a is cheaper and better.** BRACS keeps one advantage that matters
for evaluation specifically: its labels are **three-pathologist consensus**, where BEETLE's are a
model's guesses corrected by humans. For measuring a ceiling, real consensus labels are the better
yardstick.

4a's blocking problem is unchanged and is not solved by anything here: its ablation compares
with/without pseudo-labels **on held-out BCSS, which contains no DCIS**, so it cannot measure the
thing it exists to improve. Approach 5 remains a prerequisite for 4a, not an alternative to it.

---

## Part 4 — Storage and discipline

| What | Where | Size |
| --- | --- | --- |
| BRACS DCIS ROI PNGs | `tissue_label_generation/data/dcis/{train,val,test}/` | **790 files, 10.13 GB** *(measured from the FTP listing)* |
| Eval-tile manifest derived from them | `benchmarks/bracs/eval_tiles/` | small |
| BCSS-trained checkpoint | `models/tissue_type/` | unchanged, CC0-clean |

**The rule, from [`segmentation-approaches-comparison.md`](segmentation-approaches-comparison.md)
Part 3:** `backend/` may import only CC0/MIT/BSD/Apache assets; non-commercial assets live outside it
and CI fails the build if `backend/` reaches into them. BRACS is now inside that fence. Nothing in
this document puts BRACS into a loop that produces shipped weights — only into one that reads a
checkpoint and writes numbers.

---

## Part 5 — Order

1. Finish the BRACS DCIS download *(in progress)*.
2. **Gate zero** — how much DCIS is in our six cases. An afternoon, and it governs whether any of
   this matters this quarter.
3. **Gate 0** — settle 224 vs 512 tile geometry
   ([`segmentation-approaches-implementation.md`](segmentation-approaches-implementation.md) §0.2).
4. Build/confirm approach 1's **BCSS-only** shipped checkpoint — no change to its plan.
5. Build the epithelium filter (reuses BCSS pixel masks; no new data).
6. Export BRACS eval tiles, report class `1` per source, run the leakage check.
7. **Send two licence emails** — ICAR-CNR (BRACS) and the BEETLE authors — and record both answers.
8. Quote the **BRACS eval number**, not the BCSS held-out number, as the honest class-`1` headline,
   until approach 5's clicked tiles replace it with an in-domain one.

This does not change the build order in
[`segmentation-approaches-implementation.md`](segmentation-approaches-implementation.md) Part 10 —
gate zero, gate 0, approach 5, then approach 1+3 remain first. It adds one cheap,
licence-safe measurement alongside them.
