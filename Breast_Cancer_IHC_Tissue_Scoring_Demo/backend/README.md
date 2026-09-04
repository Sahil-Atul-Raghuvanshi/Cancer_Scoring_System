# IHC Scoring API

FastAPI service behind the Breast Cancer IHC Tissue Scoring app.

**Steps 1 to 5 of the pipeline run for real.** Slides are uploaded in chunks,
reassembled and validated server-side, then opened with `tiffslide` to report
their pyramid, resolution and working level (step 1). Step 2 runs both GrandQC
models over the slide — tissue detection, then multi-class artefact segmentation
— alongside classical HistoQC-style metrics that explain each call. Step 3
thresholds tissue against glass on the saturation channel. Step 4 measures I₀
from the slide's own glass, which is what makes an absolute stain scale possible.
Step 5 turns one tile's colour into optical density and shows the two stain arms
that appear in it. Steps 6–16 are documented in the catalogue but not
implemented, and carry no outputs — a step that does not run has no numbers to
report.

Each implemented step's own folder under `app/pipeline/` carries a README with
the decisions worth arguing about and the limits measured on real slides.

Step 2 needs extra dependencies and two downloaded checkpoints; it degrades
politely without them. See **[docs/step-2-quality-control.md](../docs/step-2-quality-control.md)**.

## Quick start

From the repository root, `start.bat` brings up this service and then the
frontend, each in its own terminal window. `stop.bat` shuts both down. The
steps below are the manual equivalent.

## Requirements

- Python 3.11+

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements-dev.txt
cp .env.example .env
```

Step 2 (quality control) needs PyTorch, which is kept out of the base install
because it is several hundred megabytes and nothing else needs it:

```bash
pip install -r requirements-qc.txt --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/check_qc_models.py     # verifies the stack and the checkpoints
```

The two GrandQC checkpoints are downloaded separately, from Zenodo —
[docs/step-2-quality-control.md](../docs/step-2-quality-control.md) has the
links and the folder layout.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

- API root — <http://127.0.0.1:8000/>
- Swagger UI — <http://127.0.0.1:8000/docs>

## Endpoints

| Method | Path                                  | Purpose                                   |
| ------ | ------------------------------------- | ----------------------------------------- |
| GET    | `/api/v1/health`                      | Liveness                                  |
| GET    | `/api/v1/pipeline/stages`             | All 16 stages, in execution order         |
| GET    | `/api/v1/pipeline/stages/{id}`        | One stage                                 |
| GET    | `/api/v1/pipeline/summary`            | Implemented / trained / classical counts  |
| GET    | `/api/v1/slides/upload-capability`    | Accepted formats, size and chunk limits   |
| POST   | `/api/v1/uploads`                     | Open an upload                            |
| PUT    | `/api/v1/uploads/{id}/chunks/{index}` | Upload one chunk (raw bytes)              |
| GET    | `/api/v1/uploads/{id}`                | Upload status; poll after `complete`      |
| POST   | `/api/v1/uploads/{id}/complete`       | Reassemble and validate in the background |
| DELETE | `/api/v1/uploads/{id}`                | Abort and discard staged chunks           |
| GET    | `/api/v1/slides/{id}/readout`         | **Step 1** — the pyramid and resolution   |
| GET    | `/api/v1/slides/{id}/thumbnail`       | Whole-slide overview PNG                  |
| GET    | `/api/v1/slides/{id}/region`          | One region as PNG                         |
| GET    | `/api/v1/qc/capability`               | Whether **step 2** can run, and what is missing |
| POST   | `/api/v1/qc/{id}/run`                 | **Step 2** — start a QC run (background)  |
| GET    | `/api/v1/qc/{id}/run`                 | Run progress; poll it                     |
| GET    | `/api/v1/qc/{id}`                     | The QC report                             |
| GET    | `/api/v1/qc/{id}/grid`                | Per-patch classical metrics               |
| GET    | `/api/v1/qc/{id}/overlay.png`         | Slide with artefacts tinted               |
| GET    | `/api/v1/qc/{id}/tissue.png`          | Pass 1's tissue map                       |
| GET    | `/api/v1/qc/{id}/mask.png`            | Indexed-colour artefact mask              |
| GET    | `/api/v1/qc/{id}/heatmap/{metric}.png`| One metric over the patch grid            |
| GET    | `/api/v1/qc/{id}/explain`             | One region, re-measured finely            |
| GET    | `/api/v1/tissue/{id}`                 | **Step 3** — the tissue mask report       |
| GET    | `/api/v1/tissue/{id}/panels/{name}.png` | thumbnail / saturation / mask / overlay |
| GET    | `/api/v1/calibration/{id}`            | **Step 4** — I₀ from this slide's glass   |
| GET    | `/api/v1/calibration/{id}/panels/{name}.png` | thumbnail / glass / field / corrected |
| GET    | `/api/v1/calibration/{id}/swatch.png` | I₀ as a flat square                       |
| GET    | `/api/v1/calibration/compare`         | Two or more slides' I₀, and the OD cost of confusing them |
| GET    | `/api/v1/density/{id}`                | **Step 5** — one tile in optical density  |
| GET    | `/api/v1/density/{id}/panels/{name}.png` | map / tile / density / scatter / limits |

Responses are serialised in **camelCase** so the TypeScript client can consume
them without a mapping layer.

## Uploading a slide

A resumable 3-call protocol that mirrors S3 multipart, so local staging can be
swapped for object storage without touching the API:

```
POST   /uploads                     -> validates extension + size, returns an upload_id
PUT    /uploads/{id}/chunks/{index} -> one part, raw bytes, idempotent per index
POST   /uploads/{id}/complete       -> checks every part arrived, finalises in the background
GET    /uploads/{id}                -> poll until state is "ready" or "failed"
```

A whole-slide image is several gigabytes, which is why the transfer is chunked
and the reassembly happens off the request path.

Finalising stream-concatenates the parts, verifies the declared sha256 and the
byte count, and then **proves the result opens** with `open_slide()`. A transfer
can be byte-perfect and the upload still fail, because a file that cannot be
read is not a slide.

Chunks are written tmp-then-rename, so a torn transfer leaves a `.tmp` rather
than a half-written part that the resume scan would mistake for complete. The
set of received indices is re-derived from disk on every status call, so it
cannot drift from what was actually written.

Upload state lives in a JSON sidecar next to the staged parts rather than a
database row — this service has no DB, and an upload's state is already tied to
a directory on disk.

## Reading slides

`app/ingestion/slide_reader.py` is the one place that knows about scanner
formats. It uses **tiffslide** (pure Python, works on Windows, reads pyramidal
TIFF and Aperio SVS); a native OpenSlide backend can replace it without
touching callers. Plain JPEG/PNG go to `image_reader.py`, which synthesises a
pyramid in memory and exposes the identical interface.

Two guards in there are worth knowing about, and both come from real scanner
output:

- **Associated images are never served blindly.** `label` and `macro` are
  photographs of the physical slide carrying the case number, block ID and a
  barcode — `read_associated()` refuses them outright. The series named
  `thumbnail` is often not a thumbnail either; on some scanners it is a
  half-resolution copy of the whole slide, so decoding it in a request handler
  is an outage. Overviews come from `get_thumbnail()`, which reads a pyramid
  level.
- **Levels are chosen by mpp, never by index.** The same level number is a
  different resolution on a different scanner. `best_level_for_mpp()` picks the
  nearest level that is *at-or-finer* than the target, and the readout reports
  the extra software downsample needed to land exactly on target — upsampling a
  coarser level would invent detail that was never scanned.

## Layout

```
app/
  main.py               application factory, CORS, error handling
  core/
    config.py           pydantic-settings, read from .env
    logging.py          logging setup
  api/
    deps.py             Annotated dependency aliases
    v1/
      router.py         aggregates the v1 routers
      endpoints/        health, pipeline, qc, slides, uploads
  ingestion/
    slide_reader.py     tiffslide-backed WSI reader (step 1)
    image_reader.py     plain JPEG/PNG with a synthetic pyramid, same interface
  qc/                   step 2, split by what decides and what explains
    classes.py          GrandQC's 7-class pixel coding and palette
    models.py           checkpoint discovery and loading (torch imported lazily)
    inference.py        the two GrandQC passes, against our own slide reader
    features.py         classical HistoQC-style metrics (numpy + scipy only)
    overlay.py          overlays, indexed masks and feature heatmaps
  schemas/              Pydantic v2 request/response models
  services/
    upload_service.py           chunked upload: stage, reassemble, validate, publish
    slide_reader_service.py     step 1: the pyramid readout, thumbnails, regions
    qc_service.py               step 2: the background job, cache and report
    pipeline_service.py         the stage catalogue
  data/                 the 16-step catalogue
tests/                  pytest + httpx
data/                   uploads/ staging and slides/ published (gitignored)
```

## Checks

```bash
pytest        # 71 tests
ruff check .
```

`tests/test_uploads.py` drives the real protocol end to end against a real
generated image — chunk retries, missing chunks, checksum mismatch, an
unopenable file, path traversal in the upload id — rather than mocking it.

## Implementing the remaining steps

`app/services/` is the seam. Add a service per step, expose it under
`/api/v1/slides/{id}/...`, and flip `implemented=True` on that stage in
`app/data/pipeline_steps.py`. The UI reads that flag: a stage marked
implemented gets a run button, and one that is not says so plainly instead of
showing a result it does not have.

## Keeping the frontend catalogue in sync

`frontend/src/features/pipeline/data/stages.ts` is a **generated** copy of the
catalogue, bundled so the UI's written content still renders with this service
stopped. Regenerate it after changing `app/data/pipeline_steps.py` — see
`scripts/sync_frontend_catalogue.py`.
