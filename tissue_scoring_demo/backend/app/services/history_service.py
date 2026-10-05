"""Finished runs, kept where they can be replayed instead of recomputed.

`data/demo/` is a working buffer: whatever pipeline is being driven right now
writes there, and it is keyed by upload id because that is what every step
already knows about. `data/history/` is the other half of that arrangement - the
runs somebody has finished with, filed by case and by antibody, which is how a
person looks for them.

**One copy, moved, never two copies.** Both trees sit under `data/` on one
volume, so filing a run is a directory rename: instant, atomic, and incapable of
leaving a second copy behind. Opening one renames it back. A marker's artefacts
are 500 MB and a case's five markers are two and a half gigabytes, so an archive
that copied would double the largest thing on this disk in exchange for nothing.
The manifest records where each piece currently is, so the list screen can tell
"filed" from "open right now" without walking fourteen directory trees.

**Shared work is filed at case level, marker work at marker level.** Steps 2 to
9 run on the H&E section, and a case's five antibodies share one H&E: the tissue
mask, the white point, the tile grid, the class map and the invasive regions are
one set of files that all five depend on. Filing CD44 must not take the ROI that
ABCC4 still needs. So the H&E-keyed trees move only when no marker of that case
is open, and they come back the moment one is.

    data/history/<CASE_ID>/
      manifest.json     what was run, when, and where each piece is now
      he_thumb.png      the overview the list screen shows
      shared/           the H&E-keyed trees - steps 2 to 11
      markers/<L>/      one antibody: its IHC-keyed trees and its pair-keyed ones

**Nothing here decides that a run is finished.** `state` is read off the disk -
which reports exist - rather than recorded when somebody clicks something. A
flag written at the end of a run is a flag that is wrong after a crash, and the
screens that read this need to be able to say "step 13 onward is missing" and
offer to carry on from there.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app import panel
from app.core.config import DATA_ROOT, ORIGINAL_ROOT, settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Where finished runs are filed. Beside `data/demo/`, deliberately: the rename
#: that files a run is only instant while the two are on one volume.
HISTORY_ROOT = DATA_ROOT.parent / "history"

#: Where to look for cases nobody has run yet, so the screen is useful on a
#: fresh install rather than empty until something has been scored. Shared by
#: every data version, so it sits under `<workspace>/data/`, not `v<N>_data/`.
SLIDE_LIBRARY = ORIGINAL_ROOT / "oncostem_slides"


class HistoryError(ValueError):
    """A client-correctable problem: unknown case, unknown marker, nothing filed."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


# --- what lives where -------------------------------------------------------
#
# Resolved through `settings` rather than spelled out, so moving a directory in
# the configuration moves it here too. The key is the folder name inside a
# history entry; the value is the live tree it came from.


def _trees() -> dict[str, Path]:
    """Tree name to the directory it lives in. Every name below is used by the tuples.

    `SHARED_TREES`, `IHC_TREES` and `PAIR_TREES` index this by name, so a tree added to
    one of those and not to this raises `KeyError` - and it raises it inside `open_marker`
    or `archive_marker`, which is to say at the moment somebody is moving half a gigabyte
    of somebody else's work around. `test_history_trees_are_all_mapped` is the guard.
    """
    return {
        "qc": settings.qc_dir,
        "tissue": settings.tissue_dir,
        "calibration": settings.calibration_dir,
        "tiling": settings.tiling_dir,
        "tissue_type": settings.tissue_type_dir,
        "roi": settings.roi_dir,
        "roi_selection": settings.roi_selection_dir,
        "roi_refinement": settings.roi_refinement_dir,
        "ihc_alignment": settings.ihc_alignment_dir,
        "nuclei": settings.nuclei_dir,
        "cell_typing": settings.cell_typing_dir,
        "compartments": settings.compartments_dir,
        "per_cell": settings.per_cell_dir,
        "binning": settings.binning_dir,
        "scores": settings.scores_dir,
    }


#: Keyed on the H&E upload id, and therefore shared by every marker of the case.
SHARED_TREES = (
    "qc",
    "tissue",
    "calibration",
    "tiling",
    "tissue_type",
    "roi",
    # Steps 10 and 11 are keyed on the H&E upload like the six above, and they belong
    # here for a stronger reason than tidiness: step 11's output *is* the tumour mask
    # every marker of the case is scored inside. Left out, filing one marker would take
    # the H&E away and leave the refined boundary behind for the start-up sweep to read
    # as an orphan - and re-opening the case would offer to spend BEETLE again on
    # regions somebody had already chosen and segmented.
    "roi_selection",
    "roi_refinement",
)

#: Keyed on the IHC upload id - one antibody's own slide.
IHC_TREES = ("qc", "tissue", "calibration")

#: Keyed on the pair, so entirely one antibody's.
PAIR_TREES = (
    "ihc_alignment",
    "nuclei",
    "cell_typing",
    "compartments",
    "per_cell",
    "binning",
    "scores",
)

#: Which report proves a step ran. The step numbers are the walkthrough's, and they
#: moved by two when the ROI review and the per-pixel refinement were inserted after
#: step 9. `LAST_PAIR_STEP` is read rather than spelled out below for that reason - the
#: number that means "this marker is finished" is a consequence of this table.
PAIR_STEPS: tuple[tuple[int, str], ...] = (
    (12, "ihc_alignment"),
    (13, "nuclei"),
    (14, "cell_typing"),
    (15, "compartments"),
    (16, "per_cell"),
    (17, "binning"),
    (18, "scores"),
)

#: The step whose report means a marker is complete - the score.
LAST_PAIR_STEP: str = str(PAIR_STEPS[-1][0])


def _pair(he_upload_id: str, ihc_upload_id: str) -> str:
    return f"{he_upload_id}__{ihc_upload_id}"


def _cases_dir() -> Path:
    return settings.data_dir / "cases"


@dataclass(frozen=True)
class Piece:
    """One directory or file that moves as a unit, and where it goes."""

    live: Path
    filed: Path


def _shared_pieces(case_id: str, he_upload_id: str) -> list[Piece]:
    trees = _trees()
    shared = HISTORY_ROOT / case_id / "shared"
    pieces = [
        Piece(trees[name] / he_upload_id, shared / name / he_upload_id)
        for name in SHARED_TREES
    ]
    pieces.append(
        Piece(
            settings.slides_dir / f"{he_upload_id}.json",
            shared / "slides" / f"{he_upload_id}.json",
        )
    )
    return pieces


def _marker_pieces(
    case_id: str, marker: str, he_upload_id: str, ihc_upload_id: str
) -> list[Piece]:
    trees = _trees()
    root = HISTORY_ROOT / case_id / "markers" / marker
    key = _pair(he_upload_id, ihc_upload_id)

    pieces = [Piece(trees[name] / key, root / name / key) for name in PAIR_TREES]
    pieces += [
        Piece(trees[name] / ihc_upload_id, root / name / ihc_upload_id)
        for name in IHC_TREES
    ]
    pieces.append(
        Piece(
            settings.slides_dir / f"{ihc_upload_id}.json",
            root / "slides" / f"{ihc_upload_id}.json",
        )
    )
    pieces.append(
        Piece(_cases_dir() / f"{case_id}_{marker}.json", root / "case.json")
    )
    return pieces


# --- moving -----------------------------------------------------------------


def _move(source: Path, destination: Path) -> bool:
    """Rename `source` onto `destination`, returning whether anything moved.

    The destination is cleared first so the rename is a plain one. On Windows a
    replacing rename is not allowed for a directory, and falling back to
    copy-then-delete would quietly do the thing this module exists to avoid -
    write a second copy of half a gigabyte.
    """
    if not source.exists():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    try:
        os.replace(source, destination)
    except OSError as exc:
        # Different volumes, or a file held open by a reader. Neither is
        # recoverable by copying: a copy would leave the run in both trees,
        # which is the one state the manifest cannot describe.
        raise HistoryError(
            f"could not file {source.name}: {exc}. Nothing was copied - the run is "
            "still where it was."
        ) from exc
    return True


# --- the manifest -----------------------------------------------------------


def _manifest_path(case_id: str) -> Path:
    return HISTORY_ROOT / case_id / "manifest.json"


def load_manifest(case_id: str) -> dict | None:
    path = _manifest_path(case_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("history: unreadable manifest for %s", case_id)
        return None


def save_manifest(manifest: dict) -> None:
    path = _manifest_path(manifest["caseId"])
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updatedAt"] = _now()
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _blank(case_id: str, case_path: str, he_upload_id: str) -> dict:
    return {
        "caseId": case_id,
        "casePath": case_path,
        "heUploadId": he_upload_id,
        "sharedLocation": "demo",
        "markers": {},
        "updatedAt": _now(),
    }


# --- reading what is on disk ------------------------------------------------


def _steps_done(root: Path, he_upload_id: str, ihc_upload_id: str) -> dict[str, bool]:
    """Which of steps 10 to 16 have a report, under `root`.

    `root` is either the live `data/demo` layout or one marker's filed copy, and
    the two have the same shape below the tree name - which is the point of
    filing them that way.
    """
    key = _pair(he_upload_id, ihc_upload_id)
    out: dict[str, bool] = {}
    for number, tree in PAIR_STEPS:
        out[str(number)] = (root / tree / key / "report.json").is_file()
    return out


def _score_of(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    score = payload.get("score")
    if not isinstance(score, dict):
        return None
    return {
        "percent": score.get("percent"),
        "intensity": score.get("intensity"),
        "intensityLabel": score.get("intensityLabel"),
        "markerName": score.get("markerName"),
    }


def _state_from(steps: dict[str, bool]) -> str:
    """Complete, partial or not started, from which reports exist.

    **A manifest written before the pipeline gained two steps calls its score step 16.**
    Those manifests are on disk in `data/history` and cannot be rewritten by reading
    them, so the old key is accepted as well as the current one. It costs one `or` and
    it is the difference between a filed case listing as complete and listing as
    "partial" with a score sitting in it.
    """
    if steps.get(LAST_PAIR_STEP) or steps.get("16"):
        return "complete"
    return "partial" if any(steps.values()) else "not-started"


def _live_roots() -> Path:
    """The demo tree, as a root the same shape as a filed marker's."""
    return settings.data_dir


def _marker_entry(
    case_id: str, marker: str, he_upload_id: str, ihc_upload_id: str, location: str
) -> dict:
    root = (
        _live_roots()
        if location == "demo"
        else HISTORY_ROOT / case_id / "markers" / marker
    )
    steps = _steps_done(root, he_upload_id, ihc_upload_id)
    key = _pair(he_upload_id, ihc_upload_id)
    return {
        "marker": marker,
        "name": panel.spec(marker).full_name,
        # Per marker, not only per case (P-19): nothing guarantees one H&E per case,
        # and a marker filed under the case-level id would look for its pair
        # directories under the wrong key and find nothing to move.
        "heUploadId": he_upload_id,
        "ihcUploadId": ihc_upload_id,
        "location": location,
        "state": _state_from(steps),
        "steps": steps,
        "lastStep": max(
            (int(number) for number, done in steps.items() if done), default=0
        ),
        "score": _score_of(root / "scores" / key / "report.json"),
        "updatedAt": _now(),
    }


def _he_of(manifest: dict, entry: dict) -> str:
    """The H&E a marker was run against.

    Manifests written before P-19 record it once per case, so the case-level id is the
    fallback rather than an error.
    """
    return entry.get("heUploadId") or manifest["heUploadId"]


def _required(entry: dict, pieces: list[Piece]) -> list[Piece]:
    """The pieces that must exist for a move to describe this marker truthfully.

    Not every piece: a partial run has no score directory, and that is not a fault. What
    must be there is the slide record, the case sidecar, and the directory of every step
    the manifest says finished. A step marked done whose directory is missing means the
    move is pointed at the wrong place - typically the wrong H&E key - and filing it
    anyway is how a marker came to be recorded as `history` while its run stayed live.
    """
    steps = entry.get("steps") or {}
    done = {tree for number, tree in PAIR_STEPS if steps.get(str(number))}
    trees = _trees()
    needed = []
    for piece in pieces:
        if piece.live.suffix == ".json":
            needed.append(piece)
        elif any(piece.live.parent == trees[tree] for tree in done):
            needed.append(piece)
    return needed


def _refuse_missing(case_id: str, letter: str, missing: list[Path], where: str) -> None:
    """Raise, before anything moves, when a piece the manifest relies on is absent."""
    if not missing:
        return
    names = ", ".join(str(path) for path in missing[:4])
    more = f" and {len(missing) - 4} more" if len(missing) > 4 else ""
    raise HistoryError(
        f"{case_id} {letter}: {len(missing)} piece(s) the manifest says exist are not in "
        f"the {where} tree ({names}{more}). Nothing was moved and the manifest is "
        "unchanged - recording the marker as moved would point it at nothing."
    )


# --- the operations ---------------------------------------------------------


def _sidecars() -> list[dict]:
    """Every case+marker pairing `case_service` has registered in the demo tree.

    The file is `asdict(CaseSession)` - a dataclass dump, so snake_case, not one
    of the camelCase API models. Read with the field names it actually has and
    handed on in the shape the rest of this module uses.
    """
    out: list[dict] = []
    directory = _cases_dir()
    if not directory.is_dir():
        return out

    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        needed = {"case_id", "marker", "he_upload_id", "ihc_upload_id"}
        if not needed <= payload.keys():
            continue
        if payload["marker"] not in panel.SCORED_MARKERS:
            continue

        out.append(
            {
                "caseId": payload["case_id"],
                "marker": payload["marker"],
                "casePath": payload.get("case_path", ""),
                "heUploadId": payload["he_upload_id"],
                "ihcUploadId": payload["ihc_upload_id"],
            }
        )
    return out


def refresh(case_id: str | None = None) -> None:
    """Bring the manifests in step with what is actually in the demo tree.

    Called on start-up and before any listing. A run that was driven through the
    UI and never filed still has to appear on the history screen, or a viewer who
    scored a marker yesterday and comes back today is told it was never run - and
    the 500 MB it left behind is invisible as well as unreachable.
    """
    for sidecar in _sidecars():
        if case_id is not None and sidecar["caseId"] != case_id:
            continue

        manifest = load_manifest(sidecar["caseId"]) or _blank(
            sidecar["caseId"], sidecar.get("casePath", ""), sidecar["heUploadId"]
        )
        manifest["heUploadId"] = sidecar["heUploadId"]
        if sidecar.get("casePath"):
            manifest["casePath"] = sidecar["casePath"]
        manifest["sharedLocation"] = "demo"
        manifest["markers"][sidecar["marker"]] = _marker_entry(
            sidecar["caseId"],
            sidecar["marker"],
            sidecar["heUploadId"],
            sidecar["ihcUploadId"],
            "demo",
        )
        save_manifest(manifest)


def archive_marker(case_id: str, marker: str) -> dict:
    """File one marker's run, and the shared H&E work if nothing else needs it."""
    letter = marker.upper()
    manifest = load_manifest(case_id)
    if manifest is None:
        refresh(case_id)
        manifest = load_manifest(case_id)
    if manifest is None:
        raise HistoryError(f"nothing is recorded for case {case_id}")

    entry = manifest["markers"].get(letter)
    if entry is None:
        raise HistoryError(f"case {case_id} has no run for marker {letter}")
    if entry["location"] == "history":
        return manifest

    he = _he_of(manifest, entry)
    ihc = entry["ihcUploadId"]
    pieces = _marker_pieces(case_id, letter, he, ihc)
    missing = [piece.live for piece in _required(entry, pieces) if not piece.live.exists()]
    _refuse_missing(case_id, letter, missing, "live")

    # The overview the list screen shows, rendered while the slide record is
    # still reachable. After the move there is no upload id to render from.
    _ensure_thumbnail(case_id, he)

    for piece in pieces:
        _move(piece.live, piece.filed)

    entry["heUploadId"] = he
    entry["location"] = "history"
    entry["updatedAt"] = _now()
    manifest["markers"][letter] = entry

    # The H&E work goes only when the last marker using *that H&E* has gone. Taking it
    # while another marker is open would strand that marker without the ROI its own
    # regions were carried from. Asked per H&E, because a case may have more than one.
    still_open = {
        _he_of(manifest, item)
        for item in manifest["markers"].values()
        if item["location"] != "history"
    }
    shared = manifest.setdefault("sharedLocations", {})
    if he not in still_open:
        for piece in _shared_pieces(case_id, he):
            _move(piece.live, piece.filed)
        shared[he] = "history"
    every_filed = all(item["location"] == "history" for item in manifest["markers"].values())
    manifest["sharedLocation"] = "history" if every_filed else "demo"

    save_manifest(manifest)
    logger.info("history: filed %s %s", case_id, letter)
    return manifest


def open_marker(case_id: str, marker: str) -> dict:
    """Bring one marker's run back into the demo tree, shared work included."""
    letter = marker.upper()
    manifest = load_manifest(case_id)
    if manifest is None:
        raise HistoryError(f"nothing is filed for case {case_id}")

    entry = manifest["markers"].get(letter)
    if entry is None:
        raise HistoryError(f"case {case_id} has no run for marker {letter}")

    he = _he_of(manifest, entry)
    ihc = entry["ihcUploadId"]
    pieces = _marker_pieces(case_id, letter, he, ihc)
    shared = manifest.setdefault("sharedLocations", {})
    shared_filed = shared.get(he, manifest.get("sharedLocation")) == "history"

    if entry["location"] == "history":
        missing = [piece.filed for piece in _required(entry, pieces) if not piece.filed.exists()]
        _refuse_missing(case_id, letter, missing, "filed")

    # Shared first: the marker's own artefacts refer to regions that live in it.
    if shared_filed:
        for piece in _shared_pieces(case_id, he):
            _move(piece.filed, piece.live)
        shared[he] = "demo"
    manifest["sharedLocation"] = "demo"

    if entry["location"] == "history":
        for piece in pieces:
            _move(piece.filed, piece.live)
        entry["location"] = "demo"

    entry.update(_marker_entry(case_id, letter, he, ihc, "demo"))
    manifest["markers"][letter] = entry
    save_manifest(manifest)
    logger.info("history: opened %s %s", case_id, letter)
    return manifest


def delete_marker(case_id: str, marker: str) -> dict:
    """Remove one marker's run entirely, wherever it currently is.

    Both trees are cleared, not just the one the manifest names. A manifest that
    has drifted from the disk is exactly when somebody reaches for delete, and a
    delete that trusted the drifted record would leave the bytes behind.
    """
    letter = marker.upper()
    manifest = load_manifest(case_id)
    if manifest is None:
        raise HistoryError(f"nothing is recorded for case {case_id}")

    entry = manifest["markers"].get(letter)
    if entry is None:
        raise HistoryError(f"case {case_id} has no run for marker {letter}")

    he = _he_of(manifest, entry)
    ihc = entry["ihcUploadId"]

    for piece in _marker_pieces(case_id, letter, he, ihc):
        for path in (piece.live, piece.filed):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.is_file():
                path.unlink(missing_ok=True)

    filed_root = HISTORY_ROOT / case_id / "markers" / letter
    if filed_root.is_dir():
        shutil.rmtree(filed_root, ignore_errors=True)

    manifest["markers"].pop(letter, None)
    save_manifest(manifest)
    logger.info("history: deleted %s %s", case_id, letter)
    return manifest


def delete_case(case_id: str) -> None:
    """Remove every marker of a case, and the case's shared H&E work with it."""
    manifest = load_manifest(case_id)
    if manifest is None:
        raise HistoryError(f"nothing is recorded for case {case_id}")

    # Every H&E the case's markers were run against, not only the case-level one.
    hes = {_he_of(manifest, entry) for entry in manifest["markers"].values()}
    hes.add(manifest.get("heUploadId", ""))

    for letter in list(manifest["markers"]):
        try:
            delete_marker(case_id, letter)
        except HistoryError:
            continue

    for he in filter(None, hes):
        for piece in _shared_pieces(case_id, he):
            for path in (piece.live, piece.filed):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.is_file():
                    path.unlink(missing_ok=True)

    case_root = HISTORY_ROOT / case_id
    if case_root.is_dir():
        shutil.rmtree(case_root, ignore_errors=True)
    logger.info("history: deleted case %s", case_id)


# --- the thumbnail ----------------------------------------------------------


def thumbnail_path(case_id: str) -> Path:
    return HISTORY_ROOT / case_id / "he_thumb.png"


def _ensure_thumbnail(case_id: str, he_upload_id: str) -> Path | None:
    """Render the case's H&E overview once, while the slide is still reachable.

    Imported inside the function: the slide reader pulls in the whole imaging
    stack, and the history screen should be able to list what is filed on a
    machine where that stack is not installed.
    """
    path = thumbnail_path(case_id)
    if path.is_file():
        return path

    try:
        from app.pipeline.step01_read_slide.pipeline import slide_reader_service

        png = slide_reader_service.thumbnail_png(he_upload_id, max_size=480)
    except Exception as exc:  # noqa: BLE001 - a missing thumbnail is cosmetic
        logger.info("history: no thumbnail for %s (%s)", case_id, exc)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


# --- listing ----------------------------------------------------------------


def _discovered_cases() -> dict[str, str]:
    """Case folders nobody has run yet, so the screen is useful before first use."""
    out: dict[str, str] = {}
    if not SLIDE_LIBRARY.is_dir():
        return out
    for entry in sorted(SLIDE_LIBRARY.iterdir()):
        if entry.is_dir():
            out[entry.name] = str(entry.resolve())
    return out


def list_cases() -> list[dict]:
    """Every case this machine knows about, with each antibody's state.

    Three sources, unioned: what is filed, what is open in the demo tree, and
    what is sitting in the slide library untouched. The third is what makes this
    a way in rather than only a way back - a case with nothing run against it
    still lists its five antibodies, all of them offering to start.
    """
    refresh()

    cases: dict[str, dict] = {}

    for path in sorted(HISTORY_ROOT.glob("*/manifest.json")) if HISTORY_ROOT.is_dir() else []:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cases[manifest["caseId"]] = manifest

    for case_id, case_path in _discovered_cases().items():
        cases.setdefault(
            case_id,
            {
                "caseId": case_id,
                "casePath": case_path,
                "heUploadId": "",
                "sharedLocation": "demo",
                "markers": {},
                "updatedAt": "",
            },
        )

    out: list[dict] = []
    for manifest in cases.values():
        markers = []
        for letter in panel.SCORED_MARKERS:
            entry = manifest["markers"].get(letter)
            if entry is None:
                markers.append(
                    {
                        "marker": letter,
                        "name": panel.spec(letter).full_name,
                        "ihcUploadId": None,
                        "location": None,
                        "state": "not-started",
                        "steps": {},
                        "lastStep": 0,
                        "score": None,
                        "updatedAt": None,
                    }
                )
            else:
                markers.append(entry)

        done = sum(1 for item in markers if item["state"] == "complete")
        out.append(
            {
                "caseId": manifest["caseId"],
                "casePath": manifest.get("casePath", ""),
                "heUploadId": manifest.get("heUploadId") or None,
                "hasThumbnail": thumbnail_path(manifest["caseId"]).is_file(),
                "markers": markers,
                "completed": done,
                "started": sum(1 for item in markers if item["state"] != "not-started"),
                "updatedAt": manifest.get("updatedAt", ""),
            }
        )

    out.sort(key=lambda case: (-case["started"], case["caseId"]))
    return out


def sweep() -> None:
    """Start-up tidy: record whatever a crash left in the demo tree.

    Deliberately does **not** file anything. A run left open is more likely to be
    one somebody is in the middle of than one they finished, and moving it out
    from under a browser that is polling it would break the screen they are
    looking at. Recording it is enough for it to be findable.
    """
    try:
        refresh()
    except Exception as exc:  # noqa: BLE001 - never block start-up on tidying
        logger.warning("history: start-up sweep failed (%s)", exc)


__all__ = [
    "HISTORY_ROOT",
    "HistoryError",
    "archive_marker",
    "delete_case",
    "delete_marker",
    "list_cases",
    "load_manifest",
    "open_marker",
    "refresh",
    "sweep",
    "thumbnail_path",
]
