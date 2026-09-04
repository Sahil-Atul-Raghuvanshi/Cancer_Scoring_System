"""Chunked, resumable slide-upload service.

A whole-slide image is several gigabytes, so it is uploaded in chunks and
reassembled server-side. The flow is a 3-call protocol that mirrors S3
multipart, so swapping local staging for object storage later is a change here,
not in the API:

    init_upload      -> create a record + staging dir, validate extension and
                        size up front, before any bytes move
    save_chunk       -> write one part to staging (atomic tmp-then-rename,
                        idempotent per index, so a retry is free)
    complete_upload  -> mark 'processing'; a background job runs finalize
    finalize_upload  -> stream-concatenate the parts, verify sha256, prove
                        open_slide() can actually read the result, then publish
                        to data/slides/<upload_id><ext> and mark 'ready'

The record is a small JSON sidecar in the staging directory rather than a
database row - this service has no DB, and an upload's state is already tied to
a directory on disk. That also means a restart mid-upload can still resume: the
parts and the record are both already on disk.

Durable data is keyed by the opaque `upload_id`, never the client filename,
which may carry identifiers.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.slide_reader import open_slide

logger = get_logger(__name__)

_LOCK = threading.Lock()

# Terminal and in-flight states an upload can be in.
UploadState = str  # "initiated" | "uploading" | "processing" | "ready" | "failed"


class UploadError(ValueError):
    """A client-correctable upload problem: bad extension, size, state, or a missing chunk."""


@dataclass
class UploadRecord:
    """Everything known about one in-flight or finished upload."""

    upload_id: str
    filename: str
    ext: str
    total_size: int
    chunk_size: int
    num_chunks: int
    state: UploadState = "initiated"
    sha256: str | None = None
    final_path: str | None = None
    error: str | None = None
    received: list[int] = field(default_factory=list)


def _staging_dir(upload_id: str) -> Path:
    return settings.uploads_dir / upload_id


def _record_path(upload_id: str) -> Path:
    return _staging_dir(upload_id) / "upload.json"


def _part_path(upload_id: str, index: int) -> Path:
    return _staging_dir(upload_id) / f"{index:06d}.part"


def _received_indices(upload_id: str) -> list[int]:
    """Chunk indices present on disk.

    Disk is the source of truth for resume: it cannot drift from what was
    actually written, which a separately-tracked counter can.
    """
    directory = _staging_dir(upload_id)
    if not directory.exists():
        return []

    indices: list[int] = []
    for part in directory.glob("*.part"):
        try:
            indices.append(int(part.stem))
        except ValueError:
            continue
    return sorted(indices)


def _write_record(record: UploadRecord) -> None:
    path = _record_path(record.upload_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_record(upload_id: str) -> UploadRecord | None:
    # An upload id is used to build a path, so refuse anything that is not the
    # opaque token this service issues.
    if not upload_id or "/" in upload_id or "\\" in upload_id or ".." in upload_id:
        return None

    path = _record_path(upload_id)
    if not path.exists():
        return None
    try:
        return UploadRecord(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def _finished_record(upload_id: str) -> UploadRecord | None:
    """A published upload keeps its record beside the slide, once staging is gone."""
    path = settings.slides_dir / f"{upload_id}.json"
    if not path.exists():
        return None
    try:
        return UploadRecord(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def _load(upload_id: str) -> UploadRecord:
    record = _read_record(upload_id) or _finished_record(upload_id)
    if record is None:
        raise UploadError(f"upload {upload_id} not found")
    return record


def _status(record: UploadRecord) -> dict:
    """The status payload, with `received` re-derived from disk."""
    received = record.received if record.state == "ready" else _received_indices(record.upload_id)
    return {
        "upload_id": record.upload_id,
        "state": record.state,
        "filename": record.filename,
        "total_size": record.total_size,
        "chunk_size": record.chunk_size,
        "num_chunks": record.num_chunks,
        "received": received,
        "received_count": len(received),
        "final_path": record.final_path,
        "error": record.error,
    }


# --- init -------------------------------------------------------------------


def init_upload(
    *,
    filename: str,
    total_size: int,
    num_chunks: int,
    chunk_size: int | None = None,
    sha256: str | None = None,
) -> dict:
    """Reserve an upload id and validate the request before any bytes move."""
    settings.ensure_dirs()

    ext = Path(filename or "").suffix.lower()
    if ext not in settings.allowed_slide_ext:
        raise UploadError(
            f"unsupported slide type '{ext or '(none)'}'. Allowed: "
            + ", ".join(sorted(settings.allowed_slide_ext))
        )
    if total_size <= 0 or total_size > settings.max_upload_bytes:
        raise UploadError(
            f"total_size {total_size} out of range (1..{settings.max_upload_bytes} bytes)"
        )
    if num_chunks <= 0:
        raise UploadError("num_chunks must be >= 1")

    size = chunk_size or settings.upload_chunk_size
    if size <= 0:
        raise UploadError("chunk_size must be >= 1")

    upload_id = secrets.token_urlsafe(16)
    _staging_dir(upload_id).mkdir(parents=True, exist_ok=True)

    record = UploadRecord(
        upload_id=upload_id,
        filename=filename,
        ext=ext,
        total_size=total_size,
        chunk_size=size,
        num_chunks=num_chunks,
        sha256=sha256,
    )
    _write_record(record)
    logger.info("upload %s initiated: %s chunk(s), %s bytes", upload_id, num_chunks, total_size)
    return _status(record)


# --- chunk ------------------------------------------------------------------


def save_chunk(*, upload_id: str, index: int, data: bytes) -> dict:
    """Write one part. Re-sending an index is safe and simply overwrites it."""
    record = _load(upload_id)

    if record.state in ("processing", "ready"):
        raise UploadError(f"upload {upload_id} is {record.state}; no more chunks accepted")
    if index < 0 or index >= record.num_chunks:
        raise UploadError(f"chunk index {index} out of range (0..{record.num_chunks - 1})")
    if len(data) > record.chunk_size:
        raise UploadError(
            f"chunk {index} is {len(data)} bytes, exceeds chunk_size {record.chunk_size}"
        )

    directory = _staging_dir(upload_id)
    directory.mkdir(parents=True, exist_ok=True)

    # Atomic write: a torn transfer leaves a .tmp, never a half-written .part
    # that would look complete to the resume scan.
    tmp = directory / f"{index:06d}.tmp"
    tmp.write_bytes(data)
    tmp.replace(_part_path(upload_id, index))

    if record.state == "initiated":
        record.state = "uploading"
        _write_record(record)

    return _status(record)


# --- status -----------------------------------------------------------------


def upload_status(*, upload_id: str) -> dict:
    return _status(_load(upload_id))


# --- complete ---------------------------------------------------------------


def complete_upload(*, upload_id: str) -> dict:
    """Verify every chunk arrived, then hand off to the background finalize."""
    record = _load(upload_id)
    if record.state == "ready":
        return _status(record)

    received = set(_received_indices(upload_id))
    missing = [index for index in range(record.num_chunks) if index not in received]
    if missing:
        raise UploadError(
            f"cannot complete: {len(missing)} chunk(s) missing "
            f"(first few: {missing[:5]}). Re-upload them and retry."
        )

    record.state = "processing"
    record.error = None
    _write_record(record)
    return _status(record)


# --- abort ------------------------------------------------------------------


def abort_upload(*, upload_id: str) -> dict:
    record = _load(upload_id)
    record.state = "failed"
    record.error = "aborted by user"
    _write_record(record)
    _cleanup_staging(upload_id, keep_record=True)
    return _status(record)


# --- background finalize ----------------------------------------------------


def _cleanup_staging(upload_id: str, *, keep_record: bool = False) -> None:
    directory = _staging_dir(upload_id)
    if not directory.exists():
        return

    for path in directory.iterdir():
        if keep_record and path.name == "upload.json":
            continue
        with contextlib.suppress(OSError):
            path.unlink()

    if not keep_record:
        with contextlib.suppress(OSError):
            directory.rmdir()


def finalize_upload(upload_id: str) -> None:
    """Reassemble, validate and publish.

    Runs as a background task so a multi-gigabyte reassembly never blocks the
    request. Only acts on 'processing', so a duplicate call is a no-op.
    """
    with _LOCK:
        record = _read_record(upload_id)
        if record is None or record.state != "processing":
            return

    final = (settings.slides_dir / f"{upload_id}{record.ext}").resolve()
    try:
        _reassemble(upload_id, record, final)
        # Prove it is genuinely readable before calling it ready. An upload that
        # transferred perfectly but cannot be opened is still a failed upload.
        open_slide(final).close()
    except Exception as exc:  # noqa: BLE001 - any failure marks the upload failed, cleanly
        if final.exists():
            with contextlib.suppress(OSError):
                final.unlink()
        record.state = "failed"
        record.error = f"{type(exc).__name__}: {exc}"[:500]
        _write_record(record)
        logger.warning("upload %s failed: %s", upload_id, record.error)
        return

    record.state = "ready"
    record.final_path = str(final)
    record.error = None
    record.received = list(range(record.num_chunks))

    # Keep the record beside the published slide, so status survives the
    # staging directory being cleared.
    settings.slides_dir.mkdir(parents=True, exist_ok=True)
    (settings.slides_dir / f"{upload_id}.json").write_text(
        json.dumps(asdict(record), indent=2), encoding="utf-8"
    )
    _write_record(record)
    _cleanup_staging(upload_id)
    logger.info("upload %s ready at %s", upload_id, final)


def _reassemble(upload_id: str, record: UploadRecord, final: Path) -> None:
    """Stream-concatenate the parts in index order, verifying size and checksum.

    Never loads the whole file into memory - the point of chunking is that the
    server does not have to hold a multi-gigabyte slide at once.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()

    with open(final, "wb") as out:
        for index in range(record.num_chunks):
            part = _part_path(upload_id, index)
            if not part.exists():
                raise UploadError(f"chunk {index} missing at finalize")
            with open(part, "rb") as handle:
                while True:
                    buffer = handle.read(1024 * 1024)
                    if not buffer:
                        break
                    out.write(buffer)
                    digest.update(buffer)

    if record.sha256 and digest.hexdigest() != record.sha256.lower():
        raise UploadError("checksum mismatch: reassembled file does not match declared sha256")

    size = final.stat().st_size
    if size != record.total_size:
        raise UploadError(f"size mismatch: assembled {size} bytes, expected {record.total_size}")


# --- lookup for the slide endpoints -----------------------------------------


def resolve_ready_path(*, upload_id: str) -> Path:
    """The published slide path for a finished upload."""
    record = _load(upload_id)
    if record.state != "ready" or not record.final_path:
        raise UploadError(f"upload {upload_id} is not ready (state={record.state})")

    path = Path(record.final_path)
    if not path.exists():
        raise UploadError(f"upload {upload_id} is ready but its file is missing")
    return path


def get_record(*, upload_id: str) -> UploadRecord:
    return _load(upload_id)
