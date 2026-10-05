"""Previous cases - what has been scored, and getting back into it.

    GET    /history                          every case, with each antibody's state
    GET    /history/{case}/thumbnail.png     the case's H&E overview
    POST   /history/{case}/{marker}/open     bring a run back into the working tree
    POST   /history/{case}/{marker}/archive  file a finished run
    DELETE /history/{case}/{marker}          remove one antibody's run
    DELETE /history/{case}                   remove the whole case's stored work

**Open and archive move bytes, they do not copy them.** Both trees are on one
volume, so each is a directory rename. That is why they are POSTs with no body
and why neither takes a "mode": there is one copy of a run and these two routes
decide which half of `data/` it is sitting in.

**Delete is the only destructive one and it says so in its method.** It removes
the run from *both* trees rather than the one the manifest names, because a
manifest that has drifted from the disk is exactly when somebody reaches for it.
"""

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import FileResponse

from app import panel
from app.services import history_service
from app.services.history_service import HistoryError

router = APIRouter(prefix="/history", tags=["history"])

# Sync handlers throughout, like the other routers here: these rename directory
# trees and read files, and an `async def` doing that stalls the event loop.


def _fail(exc: HistoryError) -> HTTPException:
    missing = "nothing is" in str(exc) or "has no run" in str(exc)
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND if missing else status.HTTP_409_CONFLICT,
        detail=str(exc),
    )


def _check_marker(marker: str) -> str:
    letter = marker.upper()
    if letter not in panel.SCORED_MARKERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"{marker!r} is not a scoreable antibody; expected one of "
                f"{sorted(panel.SCORED_MARKERS)}"
            ),
        )
    return letter


@router.get("", summary="Every case, and what has been scored on it")
def cases() -> dict:
    """Filed runs, open runs and untouched case folders, in one list.

    The state of each antibody is read off the disk - which reports exist - and
    not from a flag written when something finished. A flag is wrong after a
    crash, and this list is what offers to resume.
    """
    return {"cases": history_service.list_cases()}


@router.get(
    "/{case_id}/thumbnail.png",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
    summary="The case's H&E overview",
)
def thumbnail(case_id: str) -> Response:
    """Rendered when the case was first filed, while its slide was still reachable.

    Not rendered on demand: once a case is filed, its slide record has moved out
    of the working tree, so there is no upload id left to read a pyramid from.
    """
    path = history_service.thumbnail_path(case_id)
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no stored overview for case {case_id}",
        )
    return FileResponse(
        path, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"}
    )


@router.post("/{case_id}/{marker}/open", summary="Bring a run back to be replayed")
def open_marker(case_id: str, marker: str) -> dict:
    """Rename this marker's work back into the working tree, shared H&E included.

    Everything the walkthrough reads then finds its cached report exactly where
    it expects it, so replaying a finished marker runs nothing - and resuming a
    half-finished one carries on from the first step with no report.
    """
    letter = _check_marker(marker)
    try:
        manifest = history_service.open_marker(case_id, letter)
    except HistoryError as exc:
        raise _fail(exc) from exc
    return {"case": manifest, "marker": manifest["markers"].get(letter)}


@router.post("/{case_id}/{marker}/archive", summary="File a finished run")
def archive_marker(case_id: str, marker: str) -> dict:
    """Move this marker out of the working tree.

    The shared H&E work goes with it only when no other marker of the case is
    still open - taking it sooner would strand a marker without the invasive
    regions its own were carried from.
    """
    letter = _check_marker(marker)
    try:
        manifest = history_service.archive_marker(case_id, letter)
    except HistoryError as exc:
        raise _fail(exc) from exc
    return {"case": manifest, "marker": manifest["markers"].get(letter)}


@router.delete("/{case_id}/{marker}", summary="Remove one antibody's run")
def delete_marker(case_id: str, marker: str) -> dict:
    letter = _check_marker(marker)
    try:
        manifest = history_service.delete_marker(case_id, letter)
    except HistoryError as exc:
        raise _fail(exc) from exc
    return {"case": manifest}


@router.delete("/{case_id}", summary="Remove everything stored for a case")
def delete_case(case_id: str) -> dict:
    try:
        history_service.delete_case(case_id)
    except HistoryError as exc:
        raise _fail(exc) from exc
    return {"caseId": case_id, "deleted": True}
