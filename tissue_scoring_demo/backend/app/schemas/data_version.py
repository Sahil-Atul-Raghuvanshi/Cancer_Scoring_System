"""Schemas for listing and switching the versioned data tree (`storage/v1_data/`, ...)."""

from pydantic import Field

from app.schemas.common import APIModel


class DataVersionInfo(APIModel):
    """One `storage/v<N>_data/` folder."""

    version: str = Field(examples=["v1"])
    path: str
    layout: int = Field(description="The storage layout its version.json records")
    openable: bool = Field(description="Whether this code can open it")
    not_openable_reason: str | None = None
    has_own_models: bool = Field(
        description="False when this version scores with an earlier version's models/"
    )
    models_path: str
    summary: str = Field(
        default="", description="The first paragraph under VERSION.md's first section, if any"
    )


class DataVersionsResponse(APIModel):
    """Which version this backend is serving, and which others exist."""

    active: str = Field(description="The version this process reads and writes")
    latest: str | None = Field(description="The newest version this code can open")
    requested: str | None = Field(
        description="The version that was asked for when this process started, if any"
    )
    fallback_reason: str | None = Field(
        description="Why `requested` could not be opened and `active` runs instead"
    )
    pinned_by_env: bool = Field(
        description="CSS_DATA_VERSION is set, so the picker cannot change the version"
    )
    shared_data_path: str
    versions: list[DataVersionInfo]


class DataVersionSelectRequest(APIModel):
    version: str = Field(pattern=r"^v[1-9][0-9]*$")


class DataVersionSelectResponse(APIModel):
    selected: str
    restarting: bool = Field(
        description="True when the backend has been asked to reload onto the new version"
    )
