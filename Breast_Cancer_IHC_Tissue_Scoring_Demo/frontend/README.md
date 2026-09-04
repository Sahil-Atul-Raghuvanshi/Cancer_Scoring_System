# IHC Scoring — Frontend

React 18 + TypeScript 5.5 + Vite. A dark, click-through walkthrough of the
16-step pipeline that turns a breast cancer IHC whole-slide scan into a score.

## Quick start

From the repository root, `start.bat` brings up the backend and then this app,
each in its own terminal window, and opens the browser. `stop.bat` shuts both
down. The steps below are the manual equivalent.

## Requirements

- Node 18+

## Setup

```bash
npm install
cp .env.example .env      # optional; empty means "use the dev proxy"
```

## Run

```bash
npm run dev               # http://localhost:5173
```

The dev server proxies `/api` to `http://127.0.0.1:8000`, so start the backend
alongside it. The written content still renders with the API down — the header
says `api offline` and the bundled catalogue is used — but uploading a slide
and running step 1 both need the backend.

## Scripts

| Script              | Does                              |
| ------------------- | --------------------------------- |
| `npm run dev`       | Vite dev server with API proxy    |
| `npm run build`     | Type-check (`tsc -b`) then bundle |
| `npm run preview`   | Serve the production build        |
| `npm run typecheck` | Types only                        |
| `npm run lint`      | ESLint, zero warnings allowed     |

## What the app does

**Home** — the case for the ordering and all 16 steps at a glance.

**Walkthrough** (`/demo`) — the steps in the order they must run. **Step 1 is
implemented**, and it owns the upload: the drop zone appears at that step,
because getting a slide in is the first thing step 1 does. Once a slide is
loaded, running the step opens it and reports what is actually in it —
dimensions, the pyramid, the resolution of every level, and which level the
pipeline would work at for its target µm/px. Steps 2 to 5 run for real too —
quality control, the tissue mask, white calibration and optical density. Steps
6–16 are marked *not built yet*; they show what the step does, why it belongs at that point in the
order, and the ordering rule it enforces, but no output, because they produce
none.

### Nothing is invented

There is no mock data in this app. Every figure on screen was read from the
file you uploaded. A step that has not been implemented reports nothing rather
than a plausible-looking number — the whole subject of this walkthrough is
trusting a pipeline's output, and a fake figure would undercut that on the
first screen.

## Uploading

A whole-slide image is several gigabytes, so `api/uploads.ts` implements the
client half of a resumable chunked protocol:

```
init -> PUT each chunk (retried with backoff) -> complete -> poll until ready/failed
```

Files under 64 MB are checksummed client-side so the server can verify its
reassembly; past that, hashing would mean holding the whole slide in memory,
and the server's size and openability checks stand on their own. The server
validates the extension and size *before* any bytes move, and after reassembly
proves the file actually opens before reporting `ready`.

A failed chunk is retried with backoff, and a retried upload asks the server
which parts it already holds and sends only the remainder — so a dropped
connection or a backend restart costs seconds, not the whole transfer.

The loaded slide lives in `features/upload/SlideSession.tsx`, so the upload and
the step that reads it share one source of truth. The upload id is kept in
`sessionStorage`, so a reload does not throw away a transfer that already
finished.

Note the split: uploading only proves the file arrived intact and can be
opened. Reading the pyramid is step 1's job and happens when you run that step,
so the button does real work rather than replaying something already fetched.

## Layout

```
src/
  api/                  fetch client, pipeline resources, chunked upload
  components/
    layout/             AppShell: header, ambient background, footer
    ui/                 Button, Badge, Card, StatusDot
  features/
    pipeline/
      components/       rail, step-1 readout, flow, written details
      data/stages.ts    GENERATED bundled fallback catalogue
      hooks/            catalogue fetch + the run state machine
    upload/             drop zone, slide session
  pages/                HomePage, DemoPage
  styles/               tokens.css, animations.css, global.css
  types/                shapes mirroring the FastAPI response models
```

## Styling and motion

Dark-only, by intent: slide imagery reads best against a near-black ground, the
way it does on a pathology workstation. Design tokens live in
`styles/tokens.css`; there are no hard-coded colours in components.

All motion is defined in `styles/animations.css` as named keyframes plus
utility classes, so components stay declarative:

- entrances — `rise-in`, `slide-in-*`, `pop-in`
- ambience — `aurora` background glows, `pulse-glow` on the next action,
  `shimmer` on skeletons
- work in flight — `scan-sweep` while a step runs; the upload bar tracks real
  chunk progress, and switches to an indeterminate cycle for the server-side
  reassembly, which has no measurable fraction
- results — staggered reveals on the figures and the pyramid table

The whole system is muted under `prefers-reduced-motion: reduce`.

## The generated catalogue

`features/pipeline/data/stages.ts` is generated from the backend — do not edit
it by hand. After changing `backend/app/data/pipeline_steps.py`, run:

```bash
cd ../backend && python scripts/sync_frontend_catalogue.py
```
