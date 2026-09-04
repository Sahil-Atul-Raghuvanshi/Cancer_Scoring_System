"""Schemas for storage housekeeping.

The shape here exists to make one fact impossible to miss: **the slides are the storage.**
On this project's own machine `data/` is 1,542 MB and one uploaded slide is 1,524 MB of it,
so a report that lumped "cached data" into a single number would suggest clearing caches
saves space. It saves about one per cent, and it throws away the half hour of CPU behind
step 8's class map.

So the usage report separates slides from derived caches and states the slide share
outright, and every removable item carries what re-making it would cost.
"""

from enum import Enum

from pydantic import Field

from app.schemas.common import APIModel


class CleanupScope(str, Enum):
    """What a sweep is allowed to destroy, in increasing order.

    An enum rather than a free string so a typo is refused at the boundary with the valid
    names, rather than being interpreted as something more destructive than intended.
    """

    #: Data nothing can read any more. Always safe.
    ORPHANS = "orphans"
    #: The caches for a slide that is still here. Costs a re-run.
    DERIVED = "derived"
    #: One upload entirely - image, record and caches. Costs a re-upload.
    SLIDE = "slide"
    #: Every slide and every cache.
    ALL = "all"


class StorageItem(APIModel):
    """One removable thing, and what removing it would cost."""

    upload_id: str
    kind: str = Field(
        description="The step that owns this cache, or 'upload-staging' for leftover parts"
    )
    bytes: int
    cost: str | None = Field(
        default=None,
        description="Roughly what re-making this would take, for a cache. Null for staging",
    )
    orphaned: bool = Field(
        description=(
            "Nothing can read this any more - no upload record refers to it. These are "
            "swept automatically at start-up."
        )
    )


class UploadStorage(APIModel):
    """What one upload occupies, split the way a deletion decision needs it."""

    upload_id: str
    slide_bytes: int
    derived_bytes: int
    derived_by_step: dict[str, int] = Field(default_factory=dict)
    orphaned: bool


class StorageUsage(APIModel):
    """What is on disk under `data/`."""

    total_bytes: int
    slide_bytes: int = Field(description="The uploaded images. Almost always the whole story")
    derived_bytes: int = Field(description="Every per-step cache added together")
    staging_bytes: int = Field(description="Upload parts not yet reassembled")
    orphan_bytes: int = Field(description="Unreachable data, freed at start-up")

    slide_share: float = Field(
        description=(
            "Slides over total. Reported because it is the number that decides what to "
            "do: at 0.99 there is no point clearing caches."
        )
    )

    uploads: list[UploadStorage] = Field(default_factory=list)
    items: list[StorageItem] = Field(default_factory=list)


class CleanupPolicy(APIModel):
    """Whether the browser should bother telling the server it is going away.

    Fetched once on load, so the page does not register a `pagehide` handler for a
    feature that is switched off - and so the grace period can be shown to a viewer
    rather than being folklore.
    """

    cleanup_on_disconnect: bool = Field(
        description=(
            "Whether leaving the page schedules this slide's caches to be freed. Off by "
            "default: a browser cannot distinguish a closed tab from a refresh, so this "
            "is a guess about intent."
        )
    )
    grace_seconds: float = Field(
        description=(
            "How long the slide stays claimable after the browser goes away. Loading the "
            "page again within this window cancels the release, which is what makes a "
            "refresh survivable."
        )
    )
    scope: str = Field(
        description=(
            "What a disconnect frees. Never 'slide' - deleting a gigabyte upload on a "
            "guess is a different order of mistake from costing a re-run."
        )
    )


class ReleaseAck(APIModel):
    """What the server did with a disconnect. Nothing has been deleted yet."""

    scheduled: bool
    grace_seconds: float | None = None
    detail: str


class CleanupResult(APIModel):
    """What a sweep removed, or would remove."""

    scope: CleanupScope
    dry_run: bool = Field(
        description=(
            "True means nothing was deleted and `freedBytes` is what would have been. "
            "The default, for every scope that can destroy something a caller may want."
        )
    )
    freed_bytes: int
    removed: list[str] = Field(default_factory=list)
    kept: list[str] = Field(default_factory=list)
