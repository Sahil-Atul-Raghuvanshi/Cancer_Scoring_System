"""Schemas shared across API resources."""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class APIModel(BaseModel):
    """Base model that serialises to camelCase for the TypeScript client."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=lambda field: "".join(
            part if index == 0 else part.capitalize()
            for index, part in enumerate(field.split("_"))
        ),
    )


class HealthResponse(APIModel):
    """Liveness payload."""

    status: str = Field(examples=["ok"])
    service: str
    version: str
    environment: str


class ListResponse(APIModel, Generic[T]):
    """Envelope for collection endpoints."""

    items: list[T]
    total: int


class ErrorDetail(APIModel):
    """Normalised error body returned by the exception handlers."""

    code: str
    message: str
    detail: str | None = None
