# Making the app know which marker it is looking at

**Companion to** [../guides/demo-pipeline-guide.md](../guides/demo-pipeline-guide.md) (the corrected 17 steps) and
[../guides/images-to-scores-mapping.md](../guides/images-to-scores-mapping.md) (the data contract).

**What this document is.** A file-by-file specification for teaching the existing backend and
frontend about the five-antibody panel. The pipeline guide says *what* has to change conceptually;
this says *which file, which symbol, which prop*.

**The one-sentence version.** The marker becomes the **entry point**: the home page offers five
antibody cards, choosing one opens that marker's own walkthrough, and the letter travels with the
slide from upload through to its two numbers — read by exactly five of the seventeen steps, and
never guessed silently.

**The navigation model.**

```
  /                      five marker cards + H&E, each showing its own progress
      |  click "A - CD44"
  /demo/A                upload the CD44 slide, then its 17-step walkthrough
      |  finish
  /                      card A now reads "50 %, 1.5"; pick the next marker
```

Five walkthroughs, one per antibody, each with its own slide, its own progress and its own pair of
output numbers. **One set of components serves all five** — the differences resolve from a variant
object, not from five copies of the step views. See § 2.6.

---

## Part 0 — The shape of the change

### What is true today

| Layer | Today | Consequence |
| --- | --- | --- |
| `PipelineContext` | carries `upload_id` only | no step can know which antibody it is scoring |
| `PipelineStage` (catalogue) | no per-marker field | the UI cannot say "this step forks" |
| `SlideSessionProvider` | one slide, no marker | the demo is implicitly a CD44 demo |
| Routing | `/` → `/demo`, one walkthrough | no way to have five walkthroughs in flight |
| Steps 13–17 | stubs raising `StepNotImplementedError` | **nothing to unpick — build them right the first time** |

That last row is the good news. Steps 13–17 are unimplemented stubs, so this is not a refactor of
working code. It is a specification for code that has not been written yet, plus a thin thread of
marker identity through the parts that do exist.

### The rule that keeps this small

> **Marker identity is carried by everything and read by almost nothing.**

`PipelineContext` gains one field. Steps 1–12 never look at it. Steps 13–17 resolve every
marker-dependent decision through a **single lookup table** in one module. Add a sixth antibody
later and you edit one dict, not five step implementations.

Resist the temptation to branch on the marker inside step 3 or step 7 "just in case". Per-slide
calibration and per-slide stain vectors already handle everything those steps need — they vary
*per slide*, which is not the same thing as varying *per marker*, and conflating the two is how you
end up with a CD44-specific tissue mask.

---

## Part 1 — Backend

### 1.1 New: `backend/app/panel.py` — the single source of truth

Everything marker-dependent resolves here. Nothing else in the codebase hard-codes a letter.

```python
"""The five antibodies this system scores, and what each one implies.

The single trailing letter in a slide's filename is the antibody - confirmed by
OncoStem's CAB deck, the 28 July call, and the physical slide labels. Every
per-marker decision in the pipeline resolves through this module, so adding a
sixth antibody is a new PANEL entry rather than a change to five step
implementations.

Compartment is the field that forks the pipeline: three membrane markers take a
ring, two cytoplasmic markers take a band, and the second per-cell measurement
differs accordingly. N-cadherin and pan-cadherin were treated as membranous in
an earlier draft; they are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Marker(str, Enum):
    """The antibody a slide was stained with, keyed by its filename letter."""

    CD44 = "A"
    ABCC4 = "F"
    ABCC11 = "R"
    N_CADHERIN = "U"
    PAN_CADHERIN = "W"
    HE = "HE"  # not scored; used for orientation only


class Compartment(str, Enum):
    """Where in the cell the brown is expected to be (Rule 6)."""

    MEMBRANE = "membrane"
    CYTOPLASM = "cytoplasm"
    NONE = "none"


@dataclass(frozen=True)
class MarkerSpec:
    """One antibody, and every downstream decision that follows from it."""

    letter: str
    name: str
    full_name: str
    compartment: Compartment
    scored: bool

    # Step 13 - how wide to grow, in MICRONS. Never pixels: at 0.2222 um/px a
    # 4 um ring is 18 px, and on another scanner it is not.
    compartment_width_um: float

    # Step 14 - the name of the second per-cell number. Membrane markers get
    # ring completeness over 36 angular bins; cytoplasmic markers get the
    # stained fraction of the band. Completeness on a cytoplasmic marker is not
    # a harder measurement, it is a meaningless one.
    second_measure: str

    # Descriptive spread OncoStem has seen. A result outside it is a flag for
    # human review, NOT a threshold and NOT an error.
    expected_percent: tuple[int, int]


PANEL: dict[str, MarkerSpec] = {
    "A": MarkerSpec("A", "CD44", "CD44", Compartment.MEMBRANE, True, 4.0, "ring_completeness", (5, 85)),
    "F": MarkerSpec("F", "ABCC4", "ABCC4 (MRP4)", Compartment.MEMBRANE, True, 4.0, "ring_completeness", (35, 70)),
    "R": MarkerSpec("R", "ABCC11", "ABCC11 (MRP8)", Compartment.MEMBRANE, True, 4.0, "ring_completeness", (30, 70)),
    "U": MarkerSpec("U", "N-cadherin", "N-cadherin (CDH2)", Compartment.CYTOPLASM, True, 6.0, "stained_fraction", (70, 85)),
    "W": MarkerSpec("W", "Pan-cadherin", "Pan-cadherin", Compartment.CYTOPLASM, True, 6.0, "stained_fraction", (70, 85)),
    "HE": MarkerSpec("HE", "H&E", "Haematoxylin & eosin", Compartment.NONE, False, 0.0, "none", (0, 0)),
}

SCORED_MARKERS = tuple(letter for letter, spec in PANEL.items() if spec.scored)


def spec(letter: str) -> MarkerSpec:
    """The spec for a letter, or a KeyError naming what was asked for."""
    try:
        return PANEL[letter.upper()]
    except KeyError as exc:
        raise KeyError(
            f"unknown marker {letter!r}; expected one of {sorted(PANEL)}"
        ) from exc


def infer_from_filename(filename: str) -> str | None:
    """Guess the antibody from a filename, or None if it cannot be read.

    Returns a *suggestion*, never a decision. The user confirms it in the UI,
    because the filename convention is OncoStem's and a slide that arrives from
    anywhere else will not follow it. Silently mis-detecting a marker produces
    a plausible number measured in the wrong compartment, which is the worst
    kind of wrong: it does not look like an error.
    """
    ...
```

**On `infer_from_filename`.** [../guides/images-to-scores-mapping.md](../guides/images-to-scores-mapping.md) has a
section called *The filename trap* — read it before writing this function, and write the parser
against the real filenames in `data/`, not against a guess. Return `None` freely; an honest "I
don't know, please pick" is better than a confident wrong letter.

### 1.2 New: `backend/app/schemas/panel.py`

```python
class MarkerInfo(APIModel):
    """One antibody, as the UI needs to render it."""

    letter: str
    name: str
    full_name: str
    compartment: Literal["membrane", "cytoplasm", "none"]
    scored: bool
    compartment_width_um: float
    second_measure: str
    expected_percent_min: int
    expected_percent_max: int


class PanelResponse(APIModel):
    markers: list[MarkerInfo]
```

### 1.3 New endpoint: `backend/app/api/v1/endpoints/panel.py`

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/panel` | The five markers plus H&E. Static; the UI fetches it once at boot |

Register it in [router.py](../../backend/app/api/v1/router.py) alongside the others.

### 1.4 Changed: the slide carries a marker, chosen before the bytes move

Because the user picks the antibody **before** uploading, the marker arrives with the upload rather
than being patched on afterwards. The filename inference does not go away — it becomes a
**cross-check**.

**`backend/app/schemas/upload.py`** — `UploadInit` gains the user's choice:

```python
    marker: str | None = Field(
        default=None,
        description="Antibody letter the user selected before uploading",
    )
```

and `UploadStatus` gains three fields:

```python
    marker: str | None = None
    marker_source: Literal["user", "filename", "unknown"] = "unknown"
    # The letter the filename suggests, when it disagrees with `marker`.
    # None when they agree or when the filename is unreadable. This is a
    # QUESTION for the user, never an automatic correction: the file may
    # genuinely be named badly, or the user may genuinely have picked wrong,
    # and the server cannot tell which.
    marker_conflict: str | None = None
```

**`backend/app/services/upload_service.py`** — on reassembly:

1. Run `panel.infer_from_filename(filename)`.
2. If it agrees with the selected marker, or returns `None`, do nothing more.
3. If it **disagrees**, set `marker_conflict` to the inferred letter. Do **not** overwrite the
   user's choice, and do **not** fail the upload.

The conflict is the whole reason inference survives the change of flow. A user who selects CD44 and
uploads the N-cadherin slide gets a number measured in the wrong compartment — a plausible-looking
number that no downstream check can catch. This is the only place it can be caught, and it costs
one comparison.

**`backend/app/api/v1/endpoints/slides.py`** — one route, for corrections:

| Method | Path | Purpose |
| --- | --- | --- |
| `PATCH` | `/api/v1/slides/{uploadId}/marker` | Change the marker; sets `marker_source="user"`, clears `marker_conflict` |

Changing the marker must **invalidate cached results for steps 13–17** and nothing earlier. Steps
1–12 are marker-blind, so their caches stay valid — this is the first place the "carried by
everything, read by almost nothing" rule pays for itself.

### 1.5 Changed: `backend/app/pipeline/contract.py`

```python
@dataclass
class PipelineContext:
    upload_id: str
    # The antibody this slide was stained with, as a PANEL letter.
    #
    # None is legal: steps 1-12 never read it, so an unmarked slide runs the
    # whole preparation pipeline fine. Steps 13-17 call require_marker() and
    # fail with a message naming the problem, rather than defaulting to a
    # compartment and scoring the wrong part of the cell.
    marker: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def require_marker(self) -> MarkerSpec:
        """The spec for this slide's antibody, or a PipelineError."""
        if self.marker is None:
            raise PipelineError(
                "this step scores a specific antibody, but the slide has no "
                "marker set; PATCH /slides/{id}/marker first"
            )
        return panel.spec(self.marker)
```

`require_marker()` deliberately mirrors the existing `require(stage_id)`: same failure mode, same
shape, same reason. A step states what it needs and gets a message naming it.

### 1.6 Changed: `backend/app/data/pipeline_steps.py`

Add one field to `PipelineStage` in `schemas/pipeline.py`:

```python
class PerMarker(str, Enum):
    """Whether a step's behaviour depends on which antibody it is scoring."""

    SHARED = "shared"        # identical for all five - steps 1-12
    FORKS = "forks"          # branches on compartment - steps 13, 14
    PARAMETERISED = "parameterised"  # same code, per-marker constants - 15, 16, 17
```

Then set it on every stage and correct the five entries the current catalogue gets wrong:

| Stage id | Change |
| --- | --- |
| `tissue-type-segmentation` | `approach=PRETRAINED`, `trains_model=False`; tagline drops "the one model you actually train"; `how` becomes the frozen-encoder + linear-probe description; add that it runs on the **haematoxylin channel** so all five stains present alike |
| `compartments` | `per_marker=FORKS`; `what` and `tagline` must name **both** paths, not just the ring; `why_here` currently says "CD44 is a membrane marker" — generalise it |
| `per-cell-measurement` | `per_marker=FORKS`; `what` currently says "inside its membrane ring" — must cover the cytoplasm band and say completeness is membrane-only |
| `intensity-binning` | `per_marker=PARAMETERISED`; `tagline` currently says "0 / 1+ / 2+ / 3+" — the *reported* scale is OncoStem's 0–2 bands; say per-cell bins are internal |
| `aggregate` | `per_marker=PARAMETERISED`; `output_label` currently "Percent positive, H-score, category" → **"Percent positive + intensity band"**; the `rule` field currently states the H-score formula and should state the contract instead |
| `validation` | `per_marker=PARAMETERISED`; `rule` should add "report per marker, never pooled" |

**The `trains_model` flag is user-visible.** `PipelineSummary.trained_steps` currently reports 1;
after this it reports **0**, and the frontend's `violet` "trained" badge disappears from step 9.
That is the intended outcome — see the corrected Part 1 of the pipeline guide.

### 1.7 New: `backend/config/marker_cuts.json` — the per-antibody cut points

One versioned file, never inline constants. Step 15 reads it; nothing else does.

```json
{
  "version": "0.1.0-provisional",
  "anchored_on": "not yet calibrated - placeholder values",
  "cuts": {
    "A": {"od": [0.15, 0.35, 0.60], "ring_completeness_min": 0.50},
    "F": {"od": [0.15, 0.35, 0.60], "ring_completeness_min": 0.50},
    "R": {"od": [0.15, 0.35, 0.60], "ring_completeness_min": 0.50},
    "U": {"od": [0.15, 0.35, 0.60], "stained_fraction_min": 0.30},
    "W": {"od": [0.15, 0.35, 0.60], "stained_fraction_min": 0.30}
  }
}
```

**These numbers are placeholders and the file says so.** Calibrate them against the 120
reader-scored pairs in `6Slide Reports2.xlsx` as described in step 15 of the pipeline guide, then
bump `version` and fill in `anchored_on`. Until then the API should surface the version string
alongside every score, so no screen can display a number without also displaying that its
thresholds are provisional.

Add to `core/config.py`:

```python
    marker_cuts_path: Path = REPO_ROOT / "backend" / "config" / "marker_cuts.json"
```

### 1.8 New: `backend/app/case/` — the case-level runner

The existing [runner.py](../../backend/app/pipeline/runner.py) runs **one slide**. That is correct and
should not change: there is no cross-slide dependency, because Option B finds the region on each
IHC slide directly, and calibration is per-slide by design. What is missing is the layer above it.

> **One cross-slide dependency now exists.** Step 10 (`ihc-alignment`) registers
> the case's H&E to one IHC slide and carries step 9's invasive regions across,
> so that pair is no longer independent. Steps 1-9 are unchanged and still run
> on one slide - the H&E - and the runner still drives one slide at a time; step
> 10 reads the second slide's upload id from `context.artifacts["ihc-upload-id"]`.
> Everything else in this section stands.

```python
"""Runs the five scored slides of one case and assembles the ten numbers.

Five INDEPENDENT slide jobs, not one joint computation: nothing computed on the
CD44 slide is an input to the N-cadherin slide. So this module fans out and
collects; it does not merge state between slides.

Models, however, ARE shared. GrandQC, the region encoder and the nuclei model
are loaded once and slides are streamed through them - loading HoVer-Net five
times costs more than the inference does on an ROI-sized region.
"""

@dataclass
class CaseResult:
    case_id: str
    scores: dict[str, MarkerScore]   # letter -> {percent: int, intensity: float}
    cuts_version: str
    missing: list[str]               # letters with no slide uploaded

def run_case(slides: dict[str, str]) -> CaseResult:
    """slides maps marker letter -> upload_id."""
```

A case with three of five slides returns three scores and names the two it is missing. It does not
fail, and it does not silently return a partial dict that a caller might mistake for complete.

### 1.9 Steps 13–17: what each stub becomes

| Step | Reads the marker for | Must not |
| --- | --- | --- |
| **13** compartments | `spec.compartment` → ring vs band; `spec.compartment_width_um` | grow a 3–5 µm ring on U/W |
| **14** per-cell measurement | `spec.second_measure` → completeness vs stained fraction | compute 36-bin completeness on a cytoplasmic marker |
| **15** intensity binning | `cuts[letter]` from `marker_cuts.json` | share one threshold table across the panel |
| **16** aggregate | banding table per antibody | emit an H-score as the deliverable |
| **17** validation | groups results by marker | pool the five markers into one figure |

Signature convention for all five, so the fork lives in one argument rather than in five
call sites:

```python
def score(marker: MarkerSpec, cells: CellTable, cuts: MarkerCuts) -> MarkerScore: ...
```

---

## Part 2 — Frontend

The frontend change is larger than the backend one, because the marker stops being a property of a
slide and becomes **the way you navigate**. Five cards on the home page, five walkthroughs, five
pairs of numbers.

The thing that keeps it from being five times the work: **the seventeen step views do not fork.**
They render a variant object resolved from the marker, and `useSlideSession()` keeps its current
signature so every existing feature hook — `useQualityControl(slide.uploadId)`,
`useTissueMask(slide.uploadId)`, and the rest — compiles unchanged.

### 2.1 New: `frontend/src/types/panel.ts`

Mirror the backend schema exactly, as the existing type files do — the backend serialises camelCase.

```ts
export type Compartment = 'membrane' | 'cytoplasm' | 'none'
export type MarkerSource = 'user' | 'filename' | 'unknown'

export interface MarkerInfo {
  letter: string
  name: string
  fullName: string
  compartment: Compartment
  scored: boolean
  compartmentWidthUm: number
  secondMeasure: string
  expectedPercentMin: number
  expectedPercentMax: number
}

/** What a marker's card shows on the home page. */
export type MarkerProgress =
  | { state: 'empty' }
  | { state: 'uploading'; fraction: number }
  | { state: 'in-progress'; step: number; total: number }
  | { state: 'scored'; percent: number; intensity: number; outOfRange: boolean }
  | { state: 'conflict'; selected: string; inferred: string }
```

Also extend `PipelineStage` in [types/pipeline.ts](../../frontend/src/types/pipeline.ts) with
`perMarker: 'shared' | 'forks' | 'parameterised'`, and `UploadStatus` in
[types/slide.ts](../../frontend/src/types/slide.ts) with `marker`, `markerSource`, `markerConflict`.

### 2.2 New: `frontend/src/api/panel.ts`

```ts
export function fetchPanel(signal?: AbortSignal): Promise<PanelResponse>
export function setSlideMarker(uploadId: string, letter: string): Promise<UploadStatus>
```

`uploadSlide()` in [api/uploads.ts](../../frontend/src/api/uploads.ts) gains a `marker` argument that
it passes through in the `POST /uploads` init body. Everything else about the chunked-upload
protocol is untouched.

### 2.3 Changed: routing — the marker is in the URL

[App.tsx](../../frontend/src/App.tsx) today routes `/` → `HomePage` and `/demo` → `DemoPage`. The
walkthrough becomes marker-addressed:

```tsx
<Route path="/" element={<HomePage stages={stages} />} />
<Route path="/demo/:marker" element={<DemoPage stages={stages} loading={loading} upload={upload} />} />
{/* an older bookmark, and anything unrecognised, goes back to the chooser */}
<Route path="/demo" element={<Navigate to="/" replace />} />
<Route path="*" element={<Navigate to="/" replace />} />
```

**Put the marker in the URL, not only in state.** Three things fall out of it for free: reload
returns you to the same walkthrough, the browser back button moves between markers the way a user
expects, and a screenshot of the demo carries which marker it was taken on — which matters when
five walkthroughs produce five different-looking step 13 screens.

`DemoPage` reads `useParams()`, validates the letter against the fetched panel, and redirects to
`/` if it is unknown. An unrecognised letter is a bad URL, not an error screen.

### 2.4 Changed: `SlideSessionProvider` → `CaseSessionProvider`

Today [SlideSession.tsx](../../frontend/src/features/upload/SlideSession.tsx) holds exactly one slide
and persists its id under `ihc.uploadId`. It now holds **up to six**, one per marker, so a viewer
can upload CD44, walk it to step 10, go back, start N-cadherin, and return to CD44 without losing
anything.

```
CaseSessionProvider                      // owns slots: Record<letter, SlideSlot>
   |
   +-- MarkerRouteProvider               // reads :marker, exposes ONE slot as the active slide
          |
          +-- useSlideSession()          // UNCHANGED signature - resolves the active slot
          +-- useMarkerVariant()         // the marker's spec + per-step variant
```

**This split is the whole trick.** `useSlideSession()` keeps returning exactly what it returns
today — `stage`, `uploadId`, `status`, `readout`, `progress`, `readSlide`, `adjust`, `clear` — so
[QCView](../../frontend/src/features/qc/QCView.tsx),
[TissueView](../../frontend/src/features/tissue/TissueView.tsx),
[CalibrationView](../../frontend/src/features/calibration/CalibrationView.tsx),
[DensityView](../../frontend/src/features/density/DensityView.tsx) and
[NormalisationView](../frontend/src/features/normalisation/NormalisationView.tsx) need **no
changes at all**. They ask for "the current slide"; the context just resolves that per route now.

New in `CaseSessionValue`:

```ts
export interface CaseSessionValue {
  /** The whole panel, fetched once at boot. */
  panel: MarkerInfo[]
  /** Every marker's slot, whether or not a slide has been uploaded. */
  slots: Record<string, SlideSlot>
  /** What each card renders on the home page. */
  progress: Record<string, MarkerProgress>
  /** The ten numbers, for the case roll-up. Absent markers are absent, not zero. */
  scores: Partial<Record<string, MarkerScore>>
  cutsVersion: string | null
  /** Discard one marker's slide without touching the other four. */
  clearMarker: (letter: string) => void
  /** Discard the whole case. */
  clearCase: () => void
}
```

**Persistence.** Replace the `ihc.uploadId` string with `ihc.case`, a JSON map of letter to upload
id. Keep the same defensive `try/catch` the current file has — private mode and disabled storage
must still leave a working app, they just lose the session across a reload.

**Migration.** There is no stored-data migration to write: `ihc.uploadId` is session storage, so it
dies with the tab. Read the old key once and drop it if you want to be tidy; do not build a
migration path for a value that cannot outlive a browser session.

### 2.5 New: `frontend/src/features/panel/MarkerGrid.tsx` — the home page

The home page becomes the chooser. Six cards, **grouped by compartment**, because the grouping is
the teaching point and a flat row of five buttons throws it away.

```
  MEMBRANE  -  measured in a ring around the nucleus
  +---------------+  +---------------+  +---------------+
  |  A            |  |  F            |  |  R            |
  |  CD44         |  |  ABCC4        |  |  ABCC11       |
  |  ring 4.0 um  |  |  ring 4.0 um  |  |  ring 4.0 um  |
  |  ...........  |  |  ...........  |  |  ...........  |
  |  50 %   1.5   |  |  step 7 of 17 |  |  no slide yet |
  +---------------+  +---------------+  +---------------+

  CYTOPLASM  -  measured in the cell body
  +---------------+  +---------------+
  |  U            |  |  W            |
  |  N-cadherin   |  |  Pan-cadherin |
  |  band 6.0 um  |  |  band 6.0 um  |
  |  ...........  |  |  ...........  |
  |  no slide yet |  |  no slide yet |
  +---------------+  +---------------+

  REFERENCE
  +---------------+
  |  H&E          |
  |  not scored   |     opens a viewer, not a walkthrough
  +---------------+
```

Each card is one `<Card>` from [components/ui/Card.tsx](../../frontend/src/components/ui/Card.tsx),
carrying:

| Element | Content |
| --- | --- |
| Letter | `A`, large — it is how the slides are physically labelled |
| Name | `CD44` |
| Compartment badge | `membrane` / `cytoplasm`, using the existing [Badge](../../frontend/src/components/ui/Badge.tsx) tones |
| Compartment hint | `ring 4.0 µm` / `band 6.0 µm` — shows the fork before the user has run anything |
| Status line | driven by `MarkerProgress`: *no slide yet* / *uploading 62 %* / *step 7 of 17* / **`50 %  ·  1.5`** |
| Range flag | when scored and outside `expectedPercentMin/Max`, a quiet *"outside the usual range"* — never styled as an error |

**States the card must handle.** `empty`, `uploading`, `in-progress`, `scored`, `conflict`. The
conflict state is the one people forget: the user picked `A`, the filename said `U`, and the card
must say so rather than showing a confident score.

**H&E is a card, but not a walkthrough.** Selecting it opens the slide viewer for orientation and
says plainly *"the H&E carries no antibody; it is used to find tumour, not to score it."* Do not
route it into `/demo/HE` and then disable five steps — that teaches the wrong thing. Give it its
own small route.

### 2.6 What actually differs between the five walkthroughs — and how to not write it five times

This is the section to read before touching a step view.

**Steps 1–12: nothing differs.** Same components, same parameters, same defaults. The only thing
that changes is which image is loaded. The marker appears in these screens as **read-only context**
in the header — present, quiet, not interactive. Making it clickable on step 3 would imply the
tissue mask depends on it, which is exactly the wrong lesson.

**Steps 13–17: five differences, all resolvable from one object.**

| Step | What changes | A / F / R | U / W |
| --- | --- | --- | --- |
| **13** compartments | The overlay shape and the slider | Ring drawn around each nucleus; slider labelled `ring width (µm)`, default 4.0 | Band filling the cell body; slider labelled `band width (µm)`, default 6.0 |
| **14** per-cell | The scatter's **y-axis** | `ring completeness (0–1)`, 36 angular bins | `stained fraction of band (0–1)` |
| **15** binning | Which cut points load; where the lines sit on the histogram | `cuts.A/F/R` + `ringCompletenessMin` | `cuts.U/W` + `stainedFractionMin` |
| **16** aggregate | The banding table and the range flag | that marker's OD→band map; range 5–85 for A | that marker's map; range 70–85 |
| **17** validation | Which 24 reader scores to compare against | CD44's spread carries the evidence | U/W are near-constant — label them uninformative |

**The mechanism: one hook, one variant object.** Not `if (letter === 'U')` scattered through five
views.

```ts
// frontend/src/features/panel/useMarkerVariant.ts
//
// Everything a step view needs to know about the current antibody, resolved in
// one place. A step view reads fields off this; it never branches on a letter.
// Adding a sixth antibody is a PANEL entry on the server plus, at most, a new
// case in this file - never an edit to StepThirteenView.

export interface MarkerVariant {
  marker: MarkerInfo

  // Step 13
  compartmentLabel: string        // 'membrane ring' | 'cytoplasm band'
  widthLabel: string              // 'ring width (um)' | 'band width (um)'
  defaultWidthUm: number          // 4.0 | 6.0
  overlayKind: 'ring' | 'band'

  // Step 14
  secondMeasureLabel: string      // 'ring completeness' | 'stained fraction'
  secondMeasureAxis: string       // the scatter's y-axis caption
  showsAngularBins: boolean       // the 36-bin diagram: membrane markers only

  // Steps 15-16
  cuts: MarkerCuts
  expectedRange: [number, number]
}

export function useMarkerVariant(): MarkerVariant
```

Step 13's view then reads `variant.overlayKind` to pick which overlay component to mount, and
`variant.widthLabel` for the slider caption. Two overlay components exist — `RingOverlay` and
`BandOverlay` — because they genuinely draw different geometry; everything around them is shared.

**And the outputs genuinely differ.** The finish screen at the end of each walkthrough shows *that
marker's* pair, its expected range, its own reader-consensus comparison, and the `cutsVersion` the
numbers were produced under. Five walkthroughs, five different final screens — not one screen with
a swapped label.

### 2.7 Changed: `DemoPage` becomes one marker's walkthrough

[DemoPage.tsx](../../frontend/src/pages/DemoPage.tsx) keeps its structure — stepper on the left, step
panel in the middle, live score at the bottom. Four additions:

1. **Marker identity in the header**, beside the filename: `A · CD44 · membrane`. Not a control.
   Switching markers means going back to the chooser, which is what the URL structure already says.
2. **Upload happens here, not on the home page.** The user has already chosen the antibody by
   arriving at this route, so [UploadPanel](../../frontend/src/features/upload/UploadPanel.tsx) is
   rendered inside the walkthrough at step 0 with the marker already fixed. It passes the letter to
   `uploadSlide()`.
3. **The conflict banner.** If `markerConflict` comes back set, the walkthrough stops before step 1
   and asks: *"You chose CD44 (A), but this file looks like Ab-U (N-cadherin). Which is right?"*
   with both options. Never auto-correct — the server cannot tell whether the file is misnamed or
   the user mis-clicked, and neither can the UI.
4. **The bottom score bar shows this marker's pair**, in OncoStem's format (percent, then intensity
   band), plus the range flag. Not an H-score.

### 2.8 Changed: the stepper shows which steps fork

[PipelineTrack.tsx](../../frontend/src/features/pipeline/components/PipelineTrack.tsx) and
[StageDetails.tsx](../../frontend/src/features/pipeline/components/StageDetails.tsx) already render
`approach` as a `Badge` through `APPROACH_TONE` in [DemoPage.tsx](../../frontend/src/pages/DemoPage.tsx).
Add a parallel treatment for `perMarker`:

- `shared` → no badge (the default; badging 12 of 17 steps is noise)
- `forks` → a distinct badge, **"forks by marker"**
- `parameterised` → a quieter badge, **"per-marker cuts"**

Since the user is inside one marker's walkthrough, these badges are doing real work: they tell the
viewer *"from here on, what you are watching is specific to CD44"*, which is the moment the demo
earns the five-card home page.

`APPROACH_TONE` also needs its `trained: 'violet'` entry reviewed: with step 9 reclassified to
`pretrained`, no stage carries `trained` any more. Leave the key — the enum still has the member —
but expect the violet badge never to render.

### 2.9 New: `frontend/src/features/case/CasePanel.tsx` — the ten numbers, on the home page

The deliverable is a case, not a slide, so the roll-up belongs where the cards are. It appears
below the grid once at least one marker is scored:

```
                  % positive    intensity
  A  CD44             50           1.5
  F  ABCC4            45           1.0
  R  ABCC11           -             -      not yet run
  U  N-cadherin       80           1.75
  W  Pan-cadherin     -             -      not yet run

  cut points: 0.1.0-provisional
```

Three requirements, all honesty features rather than polish:

1. **Missing markers show as gaps, never as zeros.** Three of five scored means three rows of
   numbers and two explicit blanks. A `0` in a percent column is a real, meaningful result.
2. **`cutsVersion` on screen**, so no number is displayed without its provenance. While the cut
   points are placeholders the panel says `provisional` in plain sight.
3. **Out-of-range flags** using `expectedPercentMin/Max`, worded as *"outside the usual range —
   worth a human look"* and never as *"error"*. The ranges are descriptive, not thresholds.

Add a **Copy as row** action that emits the ten numbers in the column order of
`6Slide Reports2.xlsx`. That sheet is where these numbers are going; making the last mile a
copy-paste rather than a transcription removes the one error source we control completely.

## Part 3 — The checklist

### Backend

| File | Action |
| --- | --- |
| `app/panel.py` | **new** — `Marker`, `Compartment`, `MarkerSpec`, `PANEL`, `spec()`, `infer_from_filename()` |
| `app/schemas/panel.py` | **new** — `MarkerInfo`, `PanelResponse` |
| `app/api/v1/endpoints/panel.py` | **new** — `GET /panel` |
| `app/api/v1/router.py` | register the panel router |
| `app/schemas/upload.py` | add `marker` to `UploadInit`; `marker`, `marker_source`, `marker_conflict` to `UploadStatus` |
| `app/services/upload_service.py` | store the selected marker; infer from filename as a **cross-check**; set `marker_conflict` on disagreement, never overwrite |
| `app/api/v1/endpoints/slides.py` | add `PATCH /slides/{id}/marker` for corrections; invalidate steps 13–17 only |
| `app/pipeline/contract.py` | add `marker` field and `require_marker()` |
| `app/pipeline/runner.py` | thread `marker` into the context it builds |
| `app/schemas/pipeline.py` | add `PerMarker` enum and the `per_marker` field |
| `app/data/pipeline_steps.py` | correct steps 9, 13, 14, 15, 16, 17 (see § 1.6) |
| `config/marker_cuts.json` | **new** — five cut-point sets, marked provisional |
| `app/core/config.py` | add `marker_cuts_path` |
| `app/case/runner.py` | **new** — fan out five slide jobs, assemble ten numbers |
| `app/pipeline/step13…step17/` | implement against the `MarkerSpec` fork |

### Frontend

| File | Action |
| --- | --- |
| `src/types/panel.ts` | **new** — `MarkerInfo`, `Compartment`, `MarkerSource`, `MarkerProgress` |
| `src/types/pipeline.ts` | add `perMarker` |
| `src/types/slide.ts` | add `marker`, `markerSource`, `markerConflict` to `UploadStatus` |
| `src/api/panel.ts` | **new** — `fetchPanel`, `setSlideMarker` |
| `src/api/uploads.ts` | `uploadSlide()` takes a `marker` and sends it in the init body |
| `src/App.tsx` | route `/demo/:marker`; redirect `/demo` and `*` to `/` |
| `src/features/upload/CaseSession.tsx` | **new** — replaces `SlideSession.tsx`; six slots, persisted under `ihc.case` |
| `src/features/upload/caseSessionContext.ts` | **new** — `CaseSessionValue`, `slots`, `progress`, `scores`, `clearMarker`, `clearCase` |
| `src/features/upload/slideSessionContext.ts` | keep the **same** `useSlideSession()` signature; resolve the active slot from the route |
| `src/features/panel/useMarkerPanel.ts` | **new** — fetch the panel once, cache it |
| `src/features/panel/useMarkerVariant.ts` | **new** — the variant object steps 13–17 read (§ 2.6) |
| `src/features/panel/MarkerGrid.tsx` | **new** — the six home-page cards, grouped by compartment |
| `src/features/panel/MarkerCard.tsx` | **new** — one card and its five states |
| `src/pages/HomePage.tsx` | becomes the chooser: `MarkerGrid` + `CasePanel` |
| `src/pages/DemoPage.tsx` | read `:marker`; marker identity in the header; conflict banner; `perMarker` badge tone map |
| `src/features/upload/UploadPanel.tsx` | rendered inside the walkthrough with the marker fixed |
| `src/features/pipeline/components/StageDetails.tsx` | render the `perMarker` badge |
| `src/features/case/CasePanel.tsx` | **new** — the ten-number grid, with `cutsVersion` and Copy-as-row |
| `src/features/steps/RingOverlay.tsx` / `BandOverlay.tsx` | **new** — the two step-13 geometries |

**Files that need no change at all**, and this is the point of § 2.4: `QCView`, `TissueView`,
`CalibrationView`, `DensityView`, `NormalisationView`, `SlideViewer`, `PipelineTrack`, and every
hook that takes a `uploadId`. They ask the context for "the current slide" and keep working.

---

## Part 4 — Order of work

Marker plumbing first, then navigation, then the steps that fork. Navigation lands early because it
is the change that makes the demo stop being implicitly a CD44 demo, and it is visible before any
scoring exists.

1. **`panel.py` and `GET /panel`.** No UI, no pipeline. Just the table and a route. Half a day.
2. **`types/panel.ts`, `api/panel.ts`, `useMarkerPanel`.** The frontend can now name the five
   antibodies.
3. **`MarkerGrid` + `MarkerCard` on the home page, routing to `/demo/:marker`.** Cards all read
   *no slide yet*; the walkthrough is the one that already exists. **Ship here** — it is the
   smallest change that makes the panel visible, and it costs nothing downstream.
4. **`CaseSessionProvider`, keeping `useSlideSession()`'s signature.** Five slots, persisted. The
   existing views must still compile untouched; if they do not, the split in § 2.4 is wrong.
5. **Marker on the upload.** `UploadInit.marker`, the filename cross-check, `marker_conflict`, and
   the conflict banner. The walkthrough now knows what it is looking at.
6. **Catalogue corrections (§ 1.6) and the `perMarker` badges.** Text and tone maps, no logic.
   Immediately visible in the stepper, and it stops the app telling viewers step 9 needs training.
7. **`marker_cuts.json` and the config path.** Placeholders, clearly labelled.
8. **`useMarkerVariant` + steps 13 and 14.** The first place behaviour actually forks, and the
   first screen where switching markers visibly changes what is drawn.
9. **Steps 15 and 16** — per-marker cuts and the two-number contract. Cards start showing scores.
10. **`CasePanel` and the case runner** — five slides, ten numbers, Copy-as-row.
11. **Step 17** — per-marker validation against the 120 reader scores.
12. **Calibrate `marker_cuts.json`** against the reader consensus and bump its version.

**Where the demo becomes worth showing:** step 8. That is when clicking `A` and clicking `U` produce
visibly different screens from the same code, which is the argument the whole five-card home page
exists to make.

---

## Part 5 — Things to get right, and things not to do

**Do**

- Keep every per-marker decision resolvable from `PANEL`. If a step needs a `if letter == "U"`
  branch, that is a missing `MarkerSpec` field.
- Express every compartment width in **microns**. The scanner is 0.2222 µm/px; the next one will
  not be.
- Render an **inferred** marker differently from a **confirmed** one, everywhere.
- Keep `useSlideSession()`'s signature stable when you introduce `CaseSessionProvider`. If the
  existing step views need edits, the abstraction is in the wrong place.
- Put the marker letter in the **URL**. Reload, back button and screenshots all depend on it.
- Surface `cuts_version` next to every score while the thresholds are provisional.
- Invalidate **only** steps 13–17 when the marker changes.

**Do not**

- Do not thread the marker into steps 1–12. They are marker-blind, and making them look otherwise
  teaches the viewer something false about the pipeline.
- Do not default an unknown marker to `A`. Fail with a message. A plausible number measured in the
  wrong compartment is worse than no number, because it does not look wrong.
- Do not compute membrane completeness for U and W, even if a shared code path makes it free. A
  near-zero completeness value feeding a positivity rule will suppress genuinely positive cells.
- Do not add an H-score column to `CasePanel`. Compute it for the demo screens if you like; the
  deliverable is the pair. Collapsing two numbers into one discards information OncoStem's model
  uses.
- Do not fork the step views per marker. Five copies of `StepThirteenView` is how this becomes five
  times the work and four times the bugs. One view, one variant object (§ 2.6).
- Do not auto-correct a marker conflict. The server cannot tell whether the file is misnamed or the
  user mis-clicked; ask.
- Do not route H&E into `/demo/HE` with five steps disabled. Give it its own viewer route.
- Do not render a missing marker as `0 %` in `CasePanel`. Zero is a real result.
- Do not process the six slides as one job. Five independent slide jobs, shared models. See
  [../guides/demo-pipeline-guide.md](../guides/demo-pipeline-guide.md), Part 6.

---

## Appendix — The five things to remember

1. **The marker is the entry point.** Five cards on the home page, five walkthroughs at
   `/demo/:marker`, five pairs of numbers. A, F, R are membrane; U, W are cytoplasm; H&E is not
   scored.
2. **Only five steps care.** 13, 14, 15, 16, 17. The other twelve are marker-blind, and their views
   should not change at all.
3. **One set of components, one variant object.** `useMarkerVariant()` is what makes five
   walkthroughs cost slightly more than one instead of five times more.
4. **The deliverable is ten numbers.** Percent (multiples of 5) and intensity (0, 0.5, 1, 1.5,
   1.75, 2) per marker. No H-score.
5. **One lookup table, one cuts file.** Every per-marker decision resolves through `PANEL` and
   `marker_cuts.json`, so a sixth antibody is a config entry and a new card — not a code change.
