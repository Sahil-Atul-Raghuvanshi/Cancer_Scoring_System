"""Schemas for the chunked slide-upload protocol."""

from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel

UploadState = Literal["initiated", "uploading", "processing", "ready", "failed"]


class UploadInit(APIModel):
    """Opens an upload. Sent before any bytes move, so bad requests fail cheaply."""

    filename: str = Field(min_length=1, max_length=512)
    total_size: int = Field(gt=0, description="Total file size in bytes")
    num_chunks: int = Field(gt=0, description="How many parts the client will send")
    chunk_size: int | None = Field(
        default=None, gt=0, description="Bytes per part; defaults to the server's chunk size"
    )
    sha256: str | None = Field(
        default=None,
        description="Optional whole-file checksum, verified during reassembly",
    )


class UploadStatus(APIModel):
    """The state of one upload. Clients poll this after `complete`."""

    upload_id: str
    state: UploadState
    filename: str
    total_size: int
    chunk_size: int
    num_chunks: int
    received: list[int] = Field(description="Chunk indices the server holds; use this to resume")
    received_count: int
    final_path: str | None = None
    error: str | None = None
