"""Schemas for resolving a case folder and loading one marker's H&E + IHC pair."""

from pydantic import Field

from app.schemas.common import APIModel


class CaseResolveRequest(APIModel):
    case_path: str = Field(min_length=1)


class CaseResolveResponse(APIModel):
    """What a case folder actually holds, before anything is loaded."""

    case_id: str
    case_path: str
    found: dict[str, str] = Field(description="Marker letter (or 'HE') -> absolute slide path")
    missing: list[str] = Field(description="Marker letters (or 'HE') with no file found")


class CaseLoadRequest(APIModel):
    case_path: str = Field(min_length=1)
    marker: str = Field(min_length=1, max_length=4, description="Which IHC marker to load, e.g. 'A'")


class CaseLoadResponse(APIModel):
    """The two slides now registered as uploads, ready for step 1."""

    case_id: str
    marker: str
    case_path: str
    he_upload_id: str
    ihc_upload_id: str
