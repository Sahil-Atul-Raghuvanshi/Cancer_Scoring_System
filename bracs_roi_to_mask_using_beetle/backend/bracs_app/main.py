"""The API, and the static frontend it serves.

One process, not two. The demo next door runs a Vite dev server beside its backend
because it is a React application under active development; this is a single-purpose
research tool whose frontend is three static files, and a Node toolchain would be a
dependency added for nothing. So FastAPI serves `frontend/` directly and `start.bat`
has one service to wait for.

Every route that does real work returns a job id rather than the work. See `jobs.py`.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import compare, config, jobs, labels, pipeline, qpdata, sixslides, student, teacher, wsi

config.install_approach1_path()
config.ensure_dirs()

app = FastAPI(
    title="BRACS DCIS ROI to mask",
    version="1.0",
    description=(
        "Segment BRACS DCIS regions of interest with BEETLE's released nnU-Net, write "
        "the result as a BCSS-coded mask, and cut approach 1's tiles from it - to find "
        "out whether the class BCSS cannot teach can be taught from somewhere else."
    ),
)


# --- what the app knows about itself ------------------------------------------


@app.get("/api/health")
def health() -> dict:
    """Whether every input this app needs is actually present, named individually.

    A single `{"ok": true}` would be answerable while the 1.9 GB of weights were
    missing, and the failure would then surface four clicks later as a stack trace.
    """
    checks = {
        "bracs_regions": config.DCIS_DIR.is_dir(),
        "beetle_weights": config.MODEL_ZIP.exists(),
        "approach1_src": (config.APPROACH1_SRC / "export.py").exists(),
        "demo_backend": (config.DEMO_BACKEND / "app" / "common" / "imaging.py").exists(),
    }
    # The two the six-slide screen needs. Reported separately and NOT folded into `ok`:
    # the other four screens work without a whole slide or a published checkpoint, and a
    # machine that has only the BRACS regions should not be told it is broken.
    optional = {
        "he_slides": len(list(config.HE_IMAGES_DIR.glob("*H&E.svs"))) if
        config.HE_IMAGES_DIR.is_dir() else 0,
        "student_model": student.default_model() if student.available_models() else None,
    }
    detail = None
    if checks["beetle_weights"]:
        try:
            archive = teacher.read_archive()
            detail = {
                "teacher_classes": list(labels.BEETLE_CODES),
                "teacher_mpp": archive.spacing,
                "codes_verified_against_release": True,
            }
        except Exception as error:  # noqa: BLE001
            checks["beetle_weights"] = False
            detail = {"error": str(error)}

    return {
        "ok": all(checks.values()),
        "checks": checks,
        "optional": optional,
        "teacher": detail,
    }


@app.get("/api/settings")
def settings() -> dict:
    """The numbers the UI shows and the one it lets the user change."""
    return {
        "bracs_mpp": config.BRACS_MPP,
        "teacher_mpp": config.TEACHER_MPP,
        "tile_px": 224,
        "tile_um": 224 * config.TILE_MPP,
        "patch_px": config.PATCH_PX,
        "folds_available": 5,
        "class_names": ["non_epithelium", "non_invasive_epithelium", "invasive_epithelium"],
        "class_colours": {str(k): v for k, v in labels.CLASS_COLOURS.items()},
        "bcss_targets": labels.BCSS_TARGETS,
        # What each colour on the overlay means, computed by pushing one pixel of each
        # BEETLE class through the functions that painted it. See `labels.colour_legend`.
        "colour_legend": labels.colour_legend(),
        "max_upload_pixels": config.MAX_UPLOAD_PIXELS,
    }


# --- the regions --------------------------------------------------------------


@app.get("/api/rois")
def list_rois(
    split: str = Query("train", pattern="^(train|val|test)$"),
    limit: int = Query(2000, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> dict:
    try:
        entries = pipeline.list_rois(split)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    done = {path.name for path in config.RUNS_DIR.iterdir()} if config.RUNS_DIR.is_dir() else set()
    window = entries[offset : offset + limit]
    return {
        "split": split,
        "total": len(entries),
        "cases": len({entry.case_id for entry in entries}),
        "items": [{**entry.as_json(), "processed": entry.roi_id in done} for entry in window],
    }


@app.get("/api/rois/{roi_id}/thumbnail")
def roi_thumbnail(roi_id: str) -> FileResponse:
    """A small preview of the untouched region, for the picker.

    Cached under the run directory on first request. Re-encoding a 20 MB PNG on every
    scroll of a 665-item list is the difference between a usable picker and a slideshow.
    """
    try:
        roi = pipeline.find_roi(roi_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    cache = config.RUNS_DIR / roi.roi_id / "thumbnail.jpg"
    if not cache.exists():
        from PIL import Image

        cache.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(roi.path) as image:
            image = image.convert("RGB")
            image.thumbnail((320, 320), Image.Resampling.LANCZOS)
            image.save(cache, quality=82)
    return FileResponse(cache, media_type="image/jpeg")


# --- running ------------------------------------------------------------------


class RunRequest(BaseModel):
    roi_ids: list[str] = Field(min_length=1, max_length=665)
    #: Which of the five folds to average. One is the default because it is six times
    #: faster and, on a DCIS region of interest, rarely disagrees with the ensemble
    #: about anything but a boundary. Five is the right choice for anything whose
    #: numbers will be quoted, and is the only setting that produces a disagreement map.
    folds: list[int] = Field(default=[0])
    #: BRACS's resolution. Editable because it is an assumption - see `config.BRACS_MPP`.
    source_mpp: float = Field(default=config.BRACS_MPP, gt=0.05, lt=2.0)
    export_tiles: bool = True


@app.post("/api/runs")
def start_run(request: RunRequest) -> dict:
    folds = tuple(sorted({int(f) for f in request.folds}))
    if not folds or not all(0 <= f <= 4 for f in folds):
        raise HTTPException(status_code=400, detail="folds must be between 0 and 4")

    try:
        rois = [pipeline.find_roi(roi_id) for roi_id in request.roi_ids]
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    label = (
        rois[0].roi_id if len(rois) == 1 else f"{len(rois)} regions"
    ) + f" - {len(folds)} fold" + ("s" if len(folds) > 1 else "")

    def work(job: jobs.Job) -> dict:
        manifests, failures = [], []
        for index, roi in enumerate(rois):
            if job.cancelled:
                break
            base = index / len(rois)
            span = 1 / len(rois)

            def progress(message: str, fraction: float, base=base, span=span, roi=roi) -> None:
                job.message = f"{roi.roi_id}: {message}"
                job.fraction = base + span * fraction

            try:
                result = pipeline.run_region(
                    roi,
                    folds=folds,
                    source_mpp=request.source_mpp,
                    export_tiles=request.export_tiles,
                    progress=progress,
                )
                manifests.append(result.manifest)
            except Exception as error:  # noqa: BLE001
                # One bad region must not lose a batch that has already cost an hour.
                failures.append({"roi_id": roi.roi_id, "error": f"{type(error).__name__}: {error}"})

        return {
            "completed": [m["roi_id"] for m in manifests],
            "failed": failures,
            "summary": summarise(manifests),
        }

    job = jobs.RUNNER.submit("run", label, work)
    return job.as_json()


def summarise(manifests: list[dict]) -> dict:
    """What a batch produced, in the terms the question was asked in."""
    import bcss

    if not manifests:
        return {"regions": 0}

    totals = {name: 0 for name in bcss.CLASS_NAMES}
    for manifest in manifests:
        for name, count in manifest.get("tiles", {}).get("by_class", {}).items():
            totals[name] += count

    usable = [m for m in manifests if m["verdict"]["usable"]]
    return {
        "regions": len(manifests),
        "regions_usable": len(usable),
        "cases": len({m["case_id"] for m in manifests}),
        "tiles_by_class": totals,
        "mean_non_invasive_area": round(
            sum(m["class_area_fraction"]["non_invasive_epithelium"] for m in manifests)
            / len(manifests),
            4,
        ),
        "seconds": round(sum(m["seconds"] for m in manifests), 1),
    }


@app.get("/api/jobs")
def list_jobs(limit: int = Query(30, ge=1, le=200)) -> dict:
    return {"items": [job.as_json() for job in jobs.RUNNER.recent(limit)]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.RUNNER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id}")
    return job.as_json()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if not jobs.RUNNER.cancel(job_id):
        raise HTTPException(status_code=409, detail="that job has already finished")
    return {"cancelled": job_id}


# --- results ------------------------------------------------------------------


@app.get("/api/results")
def list_results() -> dict:
    """Every region already processed, newest first, with its verdict."""
    items = []
    if config.RUNS_DIR.is_dir():
        for run_dir in config.RUNS_DIR.iterdir():
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                items.append(json.loads(manifest_path.read_text(encoding="utf-8")))
    items.sort(key=lambda m: m.get("roi_id", ""))
    return {"total": len(items), "items": items, "summary": summarise(items)}


@app.get("/api/results/{roi_id}")
def get_result(roi_id: str) -> dict:
    manifest = config.RUNS_DIR / roi_id / "manifest.json"
    if not manifest.exists():
        raise HTTPException(status_code=404, detail=f"{roi_id} has not been processed")
    return json.loads(manifest.read_text(encoding="utf-8"))


@app.get("/api/results/{roi_id}/image/{which}")
def get_result_image(roi_id: str, which: str) -> FileResponse:
    """`original`, `overlay`, or `mask` - the three pictures a result has."""
    names = {
        "original": "original.png",
        "overlay": "overlay.png",
        "mask": f"{roi_id}_bcss_codes.png",
    }
    if which not in names:
        raise HTTPException(status_code=400, detail=f"no image called {which!r}")
    path = config.RUNS_DIR / roi_id / names[which]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{which} for {roi_id} is not on disk")
    return FileResponse(path, media_type="image/png")


@app.get("/api/results/{roi_id}/tiles")
def get_result_tiles(roi_id: str, limit: int = Query(24, ge=1, le=200)) -> dict:
    """A sample of the exported tiles, per class, for looking at.

    Gate G2 of approach 1 says the most valuable check on a tile export is a human
    opening the folder and reading the labels off the pictures. This is that folder,
    reachable without one.
    """
    import bcss

    run_dir = config.RUNS_DIR / roi_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"{roi_id} has not been processed")

    out = {}
    for class_name in bcss.CLASS_NAMES:
        directory = run_dir / "tiles" / class_name
        paths = sorted(directory.glob("*.png")) if directory.is_dir() else []
        out[class_name] = {"total": len(paths), "names": [p.name for p in paths[:limit]]}
    return out


@app.get("/api/results/{roi_id}/tiles/{class_name}/{tile_name}")
def get_tile(roi_id: str, class_name: str, tile_name: str) -> FileResponse:
    import bcss

    if class_name not in bcss.CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"no class {class_name!r}")
    # `Path(tile_name).name` alone would still admit an absolute path on Windows, so
    # the resolved path is checked to be inside the run directory rather than trusted.
    directory = (config.RUNS_DIR / roi_id / "tiles" / class_name).resolve()
    path = (directory / Path(tile_name).name).resolve()
    if directory not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="no such tile")
    return FileResponse(path, media_type="image/png")


# --- one image, brought by the user -------------------------------------------
#
# Everything above works on the 665 BRACS regions, because the experiment is about
# those. This section exists because the pipeline is worth being able to *see* on an
# image somebody has in front of them, and because "show me what the model does with
# this" is the question every reader of the report asks first.
#
# It is deliberately a side door, not a shortcut into the main flow:
#
#   * output goes to `data/uploads/`, never `data/runs/` - see `config.UPLOADS_DIR`
#   * no tiles are cut, so nothing an upload produces can reach approach 1's training
#     set or the stain comparison on screen 3
#   * the resolution is asked for rather than assumed, because unlike BRACS there is
#     not even a scanner convention to guess from
#
# What comes back is the mask and the overlay, which is what the question was about.

#: Generated server-side, so a path built from one cannot escape `UPLOADS_DIR`. Checked
#: on the way back in anyway - a regex here is cheaper than trusting every caller.
UPLOAD_ID = re.compile(r"^[0-9a-f]{12}$")

#: What Pillow will open and what a pathology crop is plausibly saved as. The decoded
#: image is checked too; this only rejects the obvious mistakes early, with a message
#: that names the formats instead of a stack trace from the worker thread.
UPLOAD_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)


def _upload_dir(upload_id: str) -> Path:
    if not UPLOAD_ID.match(upload_id):
        raise HTTPException(status_code=400, detail="that is not an upload id")
    directory = config.UPLOADS_DIR / upload_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail=f"no upload {upload_id}")
    return directory


def _upload_json(directory: Path) -> dict:
    """The record written when the file arrived, plus the manifest if the run finished.

    Two files rather than one because they are written at different times and a reader
    needs the first before the second exists: `upload.json` is what the user gave us,
    `manifest.json` is what the model made of it, and a page has to be able to say
    "still working" without inventing fields.
    """
    record = json.loads((directory / "upload.json").read_text(encoding="utf-8"))
    manifest_path = directory / "manifest.json"
    record["manifest"] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    record["ready"] = record["manifest"] is not None
    return record


@app.post("/api/uploads")
async def create_upload(
    file: UploadFile = File(...),
    #: The uploaded image's own resolution. The UI pre-fills BRACS's assumption and
    #: makes the field prominent, because getting it wrong shows the teacher tissue at
    #: the wrong physical scale and it will return a confident, wrong picture rather
    #: than an error. See `config.BRACS_MPP` for what that costs.
    source_mpp: float = Form(config.BRACS_MPP),
    folds: str = Form("0"),
) -> dict:
    """Take one image, segment it, and keep the mask beside it. Returns a job id."""
    if not 0.05 < source_mpp < 2.0:
        raise HTTPException(
            status_code=400,
            detail="the resolution must be between 0.05 and 2.0 microns per pixel",
        )

    try:
        wanted = tuple(sorted({int(part) for part in folds.split(",") if part.strip()}))
    except ValueError as error:
        raise HTTPException(status_code=400, detail="folds must be numbers") from error
    if not wanted or not all(0 <= fold <= 4 for fold in wanted):
        raise HTTPException(status_code=400, detail="folds must be between 0 and 4")

    original_name = Path(file.filename or "image.png").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{suffix or 'that file'} is not an image this app can read. Use one of: "
                + ", ".join(sorted(UPLOAD_SUFFIXES))
            ),
        )

    upload_id = uuid.uuid4().hex[:12]
    directory = config.UPLOADS_DIR / upload_id
    directory.mkdir(parents=True, exist_ok=True)
    stored = directory / f"source{suffix}"

    # Streamed in chunks and capped while it is still arriving. Reading the whole body
    # and checking its length afterwards means a 4 GB upload is 4 GB of memory before
    # the refusal.
    size = 0
    try:
        with stored.open("wb") as handle:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "that file is over "
                            f"{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
                        ),
                    )
                handle.write(chunk)
    except HTTPException:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    # Opened before anything is promised, so a renamed .txt or a truncated PNG fails
    # here with a sentence rather than four minutes later inside the worker thread.
    try:
        from PIL import Image

        with Image.open(stored) as image:
            width, height = image.size
            image.verify()
    except Exception as error:  # noqa: BLE001
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"{original_name} could not be read as an image ({type(error).__name__})",
        ) from error

    if width * height > config.MAX_UPLOAD_PIXELS:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(
            status_code=413,
            detail=(
                f"{width} x {height} is {width * height / 1e6:.0f} megapixels, over this "
                f"app's {config.MAX_UPLOAD_PIXELS // 1_000_000} MP limit. Crop it to the "
                "region you care about - the model reads 512 px patches and gains "
                "nothing from the rest of the slide."
            ),
        )

    entry = pipeline.entry_for_upload(stored, f"upload_{upload_id}")

    def work(job: jobs.Job) -> dict:
        def progress(message: str, fraction: float) -> None:
            job.message = message
            job.fraction = fraction

        result = pipeline.run_region(
            entry,
            folds=wanted,
            source_mpp=source_mpp,
            # Never. An uploaded image carries no annotation saying what is in it, so
            # tiles cut from it would be training data labelled by a model's guess about
            # an unknown specimen. See the section comment above.
            export_tiles=False,
            out_dir=directory,
            progress=progress,
        )
        return {"upload_id": upload_id, "manifest": result.manifest}

    job = jobs.RUNNER.submit("upload", original_name, work)

    # Written after the job is queued so that it can carry the job id. The worker only
    # ever writes *into* this directory, never reads this file, so the sub-millisecond
    # window in which the directory exists without it costs nothing but an absence from
    # `list_uploads` - which `_upload_json` is guarded against anyway.
    record = {
        "upload_id": upload_id,
        "job_id": job.id,
        "filename": original_name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_mpp": source_mpp,
        "folds": list(wanted),
        "native_size": [width, height],
        "bytes": size,
        "stored": stored.name,
    }
    (directory / "upload.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {"upload_id": upload_id, "job": job.as_json(), "upload": record}


@app.get("/api/uploads")
def list_uploads(limit: int = Query(50, ge=1, le=500)) -> dict:
    """Every image uploaded so far, newest first."""
    items = []
    if config.UPLOADS_DIR.is_dir():
        for directory in config.UPLOADS_DIR.iterdir():
            if (directory / "upload.json").exists():
                try:
                    items.append(_upload_json(directory))
                except (OSError, json.JSONDecodeError):
                    continue  # a directory caught mid-write; it will be there next poll
    items.sort(key=lambda record: record.get("created", ""), reverse=True)
    return {"total": len(items), "items": items[:limit]}


@app.get("/api/uploads/{upload_id}")
def get_upload(upload_id: str) -> dict:
    return _upload_json(_upload_dir(upload_id))


@app.get("/api/uploads/{upload_id}/image/{which}")
def get_upload_image(upload_id: str, which: str) -> FileResponse:
    """`original`, `overlay`, `mask` - and `input`, which the other three do not cover.

    `original` is the web-sized copy of the *resampled* image, not the file that was
    uploaded, so the two panels on screen are the same pixels at the same scale and the
    colours line up with the tissue under them.

    `input` is that same resampled image at full working resolution, and it exists
    because the mask does. The mask is written at the working size; `original` is capped
    at 1400 px on its longest side for the browser's sake, so for anything bigger than
    that the two do not correspond pixel for pixel. A downloaded label mask with no
    image it lines up against is half an artefact, and reconstructing the other half
    means knowing to resample the upload by exactly `source_mpp / 0.5` with area
    averaging. Serving it is cheaper than documenting it.
    """
    directory = _upload_dir(upload_id)
    names = {
        "original": "original.png",
        "overlay": "overlay.png",
        "mask": f"upload_{upload_id}_bcss_codes.png",
        "input": f"upload_{upload_id}_image.png",
    }
    if which not in names:
        raise HTTPException(status_code=400, detail=f"no image called {which!r}")
    path = directory / names[which]
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{which} is not on disk yet - the run may still be going",
        )
    return FileResponse(path, media_type="image/png")


@app.delete("/api/uploads/{upload_id}")
def delete_upload(upload_id: str) -> dict:
    """Throw one upload and everything made from it away.

    Uploads are the only thing in this app a user creates by accident, and the only
    thing whose accumulation is nobody's experiment. A running job is left alone -
    deleting the directory under the worker would fail the run with an `OSError` raised
    from inside a file write, which is a confusing way to say "cancelled".
    """
    directory = _upload_dir(upload_id)
    job = jobs.RUNNER.get(_upload_json(directory).get("job_id") or "")
    if job is not None and job.status in ("queued", "running"):
        raise HTTPException(status_code=409, detail="that upload is still being segmented")
    shutil.rmtree(directory, ignore_errors=True)
    return {"deleted": upload_id}


# --- the question ---------------------------------------------------------------


@app.get("/api/compare")
def compare_tiles() -> JSONResponse:
    """BRACS tiles against BCSS tiles: same format, and the same distribution?"""
    return JSONResponse(compare.compare_all())


# --- the six clinical slides ----------------------------------------------------
#
# The one screen that is not about BRACS. It runs the model this project just trained and
# the model that labelled its training data over the SAME pixels of our own H&E slides,
# and shows them beside each other. See `sixslides.py` for why agreement here is weaker
# evidence than it looks and disagreement is stronger.


class SixSlidesRequest(BaseModel):
    #: Empty means all six. Named individually so a single row can be re-run without
    #: paying for the other five.
    cases: list[str] = Field(default_factory=list)
    folds: list[int] = Field(default=[0])
    size_px: int = Field(default=config.COMPARE_REGION_PX, ge=448, le=6144)
    model_name: str | None = None


@app.get("/api/sixslides")
def list_six_slides() -> dict:
    """The six cases, whether each has been compared, and what is on disk for it."""
    try:
        names = sixslides.cases()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    items = []
    for case in names:
        entry: dict = {"case": case, "result": sixslides.load_result(case)}
        try:
            entry.update(sixslides.slide_summary(case))
            entry["available"] = True
        except FileNotFoundError as error:
            # A missing slide is reported per row rather than failing the whole screen:
            # five comparable slides are worth showing even if the sixth is absent.
            entry.update({"available": False, "error": str(error)})
        items.append(entry)

    done = [item["result"] for item in items if item["result"]]
    return {
        "items": items,
        "models": student.available_models(),
        "default_model": student.default_model() if student.available_models() else None,
        "region_px": config.COMPARE_REGION_PX,
        "region_um": round(config.COMPARE_REGION_PX * config.TEACHER_MPP, 1),
        "summary": sixslides.summarise(done),
    }


@app.post("/api/sixslides/run")
def run_six_slides(request: SixSlidesRequest) -> dict:
    folds = tuple(sorted({int(f) for f in request.folds}))
    if not folds or not all(0 <= f <= 4 for f in folds):
        raise HTTPException(status_code=400, detail="folds must be between 0 and 4")

    try:
        known = sixslides.cases()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    wanted = tuple(request.cases) if request.cases else known
    unknown = [case for case in wanted if case not in known]
    if unknown:
        raise HTTPException(status_code=404, detail=f"no such case(s): {unknown}")

    label = (wanted[0] if len(wanted) == 1 else f"{len(wanted)} slides") + (
        f" - {len(folds)} fold" + ("s" if len(folds) > 1 else "")
    )

    def work(job: jobs.Job) -> dict:
        # ONE job for all six, not six jobs: there is a single worker thread, and six
        # queued jobs would report six progress bars that can only advance one at a time.
        done, failed = [], []
        for index, case in enumerate(wanted):
            if job.cancelled:
                break
            base, span = index / len(wanted), 1 / len(wanted)

            def progress(message: str, fraction: float, base=base, span=span, case=case) -> None:
                job.message = f"{case}: {message}"
                job.fraction = base + span * fraction

            try:
                done.append(sixslides.compare_slide(
                    case, folds=folds, size_px=request.size_px,
                    model_name=request.model_name, progress=progress,
                ))
            except Exception as error:  # noqa: BLE001 - one bad slide must not lose five
                failed.append({"case": case, "error": f"{type(error).__name__}: {error}"})

        return {
            "completed": [r["case"] for r in done],
            "failed": failed,
            "summary": sixslides.summarise(done),
        }

    return jobs.RUNNER.submit("sixslides", label, work).as_json()


@app.get("/api/sixslides/{case}")
def get_six_slide(case: str) -> dict:
    result = sixslides.load_result(case)
    if result is None:
        raise HTTPException(status_code=404, detail=f"{case} has not been compared yet")
    return result


@app.get("/api/sixslides/{case}/image/{which}")
def get_six_slide_image(case: str, which: str) -> FileResponse:
    """`original`, `ours`, `beetle` or `disagreement` - the row's four pictures."""
    names = {
        "original": "original.png",
        "ours": "ours.png",
        "beetle": "beetle.png",
        "disagreement": "disagreement.png",
    }
    if which not in names:
        raise HTTPException(status_code=400, detail=f"no image called {which!r}")
    path = config.SIXSLIDES_DIR / case / names[which]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{which} for {case} is not on disk")
    return FileResponse(path, media_type="image/png")


# --- the whole slide, against a pathologist -------------------------------------
#
# The only screen in this project where a model is scored against a PERSON. BRACS's test
# slides ship QuPath annotations; `qpdata.py` reads them without QuPath, and `wsi.py`
# segments the whole tissue with both models and scores both inside the boxes. Everything
# else here compares a model with another model.


def _wsi_cases() -> dict[str, dict]:
    """Slides on disk that have BOTH an image and an annotation file.

    One without the other is not listed at all: a slide with no annotations cannot be
    scored, and annotations with no slide cannot be drawn, and offering either would
    promise a comparison that cannot be made.
    """
    root = config.BRACS_WSI_DIR
    if not root.is_dir():
        return {}

    annotations = {path.stem: path for path in root.rglob("*.qpdata")}
    found: dict[str, dict] = {}
    for slide in sorted(root.glob("*.svs")):
        if slide.stem not in annotations:
            continue
        directory = config.WSI_DIR / slide.stem
        manifest = directory / "manifest.json"
        found[slide.stem] = {
            "case": slide.stem,
            "slide": str(slide),
            "annotations": str(annotations[slide.stem]),
            "bytes": slide.stat().st_size,
            "result": (
                json.loads(manifest.read_text(encoding="utf-8"))
                if manifest.exists() else None
            ),
        }
    return found


class WsiRequest(BaseModel):
    case: str
    folds: list[int] = Field(default=[0])
    model_name: str | None = None


@app.get("/api/wsi")
def list_wsi() -> dict:
    cases = _wsi_cases()
    items = []
    for case, entry in cases.items():
        row = {k: v for k, v in entry.items() if k not in ("slide", "annotations")}
        if row["result"] is None:
            # Read the annotations anyway: the boxes are what makes this screen worth
            # opening, and they cost milliseconds against the hour the run costs.
            try:
                found = qpdata.parse(Path(entry["annotations"]))
                row["annotation_preview"] = qpdata.summarise(found, 0.25)
            except (ValueError, OSError) as error:
                row["annotation_error"] = str(error)
        items.append(row)
    return {
        "items": items,
        "models": student.available_models(),
        "default_model": student.default_model() if student.available_models() else None,
        "class_map": wsi.BRACS_TO_OURS,
    }


@app.post("/api/wsi/run")
def run_wsi(request: WsiRequest) -> dict:
    cases = _wsi_cases()
    entry = cases.get(request.case)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"no slide+annotation pair called {request.case!r}; have "
                   f"{sorted(cases)}",
        )
    folds = tuple(sorted({int(f) for f in request.folds}))
    if not folds or not all(0 <= f <= 4 for f in folds):
        raise HTTPException(status_code=400, detail="folds must be between 0 and 4")

    def work(job: jobs.Job) -> dict:
        return wsi.run_slide(
            Path(entry["slide"]), Path(entry["annotations"]),
            config.WSI_DIR / request.case,
            folds=folds, model_name=request.model_name,
            progress=jobs.report(job),
        )

    label = f"{request.case} - whole tissue, {len(folds)} fold" + ("s" if len(folds) > 1 else "")
    return jobs.RUNNER.submit("wsi", label, work).as_json()


@app.get("/api/wsi/{case}")
def get_wsi(case: str) -> dict:
    manifest = config.WSI_DIR / case / "manifest.json"
    if not manifest.exists():
        raise HTTPException(status_code=404, detail=f"{case} has not been run yet")
    return json.loads(manifest.read_text(encoding="utf-8"))


@app.get("/api/wsi/{case}/image/{which}")
def get_wsi_image(case: str, which: str) -> FileResponse:
    """`original`, `truth`, `ours` or `beetle` - the four columns, pixel-aligned."""
    if which not in ("original", "truth", "ours", "beetle"):
        raise HTTPException(status_code=400, detail=f"no image called {which!r}")
    path = config.WSI_DIR / case / f"{which}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{which} for {case} is not on disk")
    return FileResponse(path, media_type="image/png")


# --- the frontend -------------------------------------------------------------

if config.FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
