"""Storage housekeeping.

    GET  /maintenance/storage           what is on disk, and what each part costs to remake
    POST /maintenance/cleanup           remove a scope. Dry run unless told otherwise
    GET  /maintenance/policy            whether leaving the page frees anything
    POST /maintenance/claim             this page is using this slide; cancel any release
    POST /maintenance/release           this page is going away  (sendBeacon target)

**"Clean up on browser close" is off by default, and the reason is worth reading.** A
backend cannot tell a closed tab from a refresh - `pagehide` fires for both, and for
following a link away - nor "closed the last tab" from "closed one of two". So
`/release` never deletes anything: it *schedules* a release, and `/claim` cancels it. A
page that reloads claims its slide within a second, so a refresh survives and a genuine
close does not. Turn it on with `CLEANUP_ON_DISCONNECT=true`.

Two things need no guess about intent at all. Unreachable data is swept at **start-up**,
because a cache whose slide is gone can never be served again. And "use a different
slide" in the UI is an explicit act with the byte count in front of it.

**Every destructive scope is a dry run by default.** `POST` without `confirm=true`
measures and reports; nothing is deleted. That is not ceremony - `slide` destroys an
upload that took minutes to transfer and `derived` destroys a class map that took half an
hour to compute.
"""

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.maintenance import (
    CleanupPolicy,
    CleanupResult,
    CleanupScope,
    StorageItem,
    ReleaseAck,
    StorageUsage,
    UploadStorage,
)
from app.core.config import settings
from app.services import maintenance_service

router = APIRouter(prefix="/maintenance", tags=["maintenance"])

# Plain `def`, not `async`: both handlers walk the data tree, and a multi-gigabyte
# directory walk on the event loop would stall every other request for its duration.

_CONFIRM = Query(
    default=False,
    description=(
        "Actually delete. Without it the call is a dry run that reports what it would "
        "free and removes nothing - which is the right default for a scope that can "
        "destroy an upload or half an hour of computed results."
    ),
)

_UPLOAD_REQUIRED = Query(
    ...,
    alias="uploadId",
    description="The slide this page holds.",
)

_UPLOAD = Query(
    default=None,
    # camelCase on the wire, like every other query parameter in this API (see
    # `modelMpp` on step 2). Without the alias the parameter simply never binds and the
    # endpoint reports "this scope needs an upload id" while the caller is staring at
    # the upload id they sent - which is exactly how this was found.
    alias="uploadId",
    description="Which slide to act on. Required for 'derived' and 'slide'.",
)


@router.get(
    "/policy",
    response_model=CleanupPolicy,
    summary="Whether leaving the page frees anything, and how long the grace is",
)
def policy() -> CleanupPolicy:
    """What the browser needs to know before deciding to register a `pagehide` handler.

    Separate from `/storage` because the page asks this on every load and `/storage`
    walks a multi-gigabyte tree. A policy read must not cost that.
    """
    return CleanupPolicy(
        cleanup_on_disconnect=settings.cleanup_on_disconnect,
        grace_seconds=settings.cleanup_disconnect_grace_seconds,
        scope=settings.cleanup_disconnect_scope,
    )


@router.post(
    "/claim",
    response_model=ReleaseAck,
    summary="This page is using this slide - cancel any scheduled release",
)
def claim(upload_id: str = _UPLOAD_REQUIRED) -> ReleaseAck:
    """Cancel a pending release.

    Called on every page load that has a slide in hand, which is what makes a refresh
    survivable: the page that reloads gets here long before the grace period elapses.
    """
    cancelled = maintenance_service.claim(upload_id)
    return ReleaseAck(
        scheduled=False,
        detail=(
            "a scheduled release was cancelled"
            if cancelled
            else "nothing was scheduled for this slide"
        ),
    )


@router.post(
    "/release",
    response_model=ReleaseAck,
    summary="This page is going away - schedule this slide's caches to be freed",
)
def release(upload_id: str = _UPLOAD_REQUIRED) -> ReleaseAck:
    """Schedule, never delete.

    **Shaped for `navigator.sendBeacon`**, which is the only thing that reliably reaches
    a server from a page that is closing: it is a POST, it carries no custom headers, its
    response is discarded, and it must return fast. So everything is in the query string,
    nothing is read from the body, and the work happens on a timer rather than in the
    handler.

    Returns 200 even when the feature is off, because a beacon cannot read a status code
    and a page closing has nothing useful to do with an error.
    """
    grace = maintenance_service.release(upload_id)
    if grace is None:
        return ReleaseAck(
            scheduled=False,
            detail="cleanup on disconnect is off; nothing was scheduled",
        )
    return ReleaseAck(
        scheduled=True,
        grace_seconds=grace,
        detail=(
            f"this slide's caches will be freed in {grace:.0f}s unless the page comes "
            "back and claims it"
        ),
    )


@router.get(
    "/storage",
    response_model=StorageUsage,
    summary="What is on disk under data/, and what it would cost to remake",
)
def storage() -> StorageUsage:
    """Measure the data tree. Changes nothing.

    Reports the slide share explicitly because it is the number that decides what to do:
    on this project's machine it is 0.99, and at that ratio clearing caches is not a
    storage strategy - deleting a slide is.
    """
    report = maintenance_service.usage()

    return StorageUsage(
        total_bytes=report.total_bytes,
        slide_bytes=report.slide_bytes,
        derived_bytes=report.derived_bytes,
        staging_bytes=report.staging_bytes,
        orphan_bytes=report.orphan_bytes,
        slide_share=round(report.slide_share, 4),
        uploads=[
            UploadStorage(
                upload_id=upload_id,
                slide_bytes=entry["slide"],
                derived_bytes=sum(entry["derived"].values()),
                derived_by_step=entry["derived"],
                orphaned=entry["orphaned"],
            )
            for upload_id, entry in sorted(report.uploads.items())
        ],
        items=[
            StorageItem(
                upload_id=item.upload_id,
                kind=item.kind,
                bytes=item.bytes,
                cost=item.cost,
                orphaned=item.orphaned,
            )
            for item in report.items
        ],
    )


@router.post(
    "/cleanup",
    response_model=CleanupResult,
    summary="Remove a scope of stored data. Dry run unless confirmed",
)
def cleanup(
    scope: CleanupScope = CleanupScope.ORPHANS,
    upload_id: str | None = _UPLOAD,
    confirm: bool = _CONFIRM,
) -> CleanupResult:
    """Sweep, or report what sweeping would free.

    A 409 rather than a 400 when the scope needs an upload id: the request is
    well-formed, it is the combination that cannot be acted on, and pointing the caller
    at their query string would send them looking in the wrong place.
    """
    try:
        result = maintenance_service.sweep(
            scope.value, upload_id=upload_id, dry_run=not confirm
        )
    except maintenance_service.MaintenanceError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    return CleanupResult(
        scope=scope,
        dry_run=result.dry_run,
        freed_bytes=result.freed_bytes,
        removed=result.removed,
        kept=result.kept,
    )
