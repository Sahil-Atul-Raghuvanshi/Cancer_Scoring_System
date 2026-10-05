"""What is on disk under `data/`, and how to get rid of the parts nobody needs.

**Measure before you tidy, because the intuition is wrong here.** On this machine the
demo's whole `data/` tree is 1,542 MB, and **1,524 MB of it is one uploaded slide**. Every
derived cache put together - quality control, the tissue mask, the white-point basis and
step 8's class map - is 19 MB, a little over 1 %.

So "clear the caches to save space" saves about a hundredth of the space and throws away
the expensive part: step 8's class map alone is half an hour of CPU, while the tissue mask
it sits on is seconds. **The storage problem is the slides**, and it is a real one - they
are 0.8-1.6 GB each, nothing has ever removed one, and sixteen of them accumulated here
before this module existed. What frees space is deleting a slide; what deleting a slide
costs is re-uploading it.

That asymmetry is why this module refuses to have a single "clean up" verb. Each scope
names what it destroys and what that costs:

  `orphans`  derived data whose upload no longer exists, plus staging left by an
             interrupted upload. **Pure waste** - nothing can ever read it again. Safe
             enough to run unattended, and it is, at startup.
  `derived`  the caches for a slide that is still here. Frees little, costs a re-run -
             seconds for steps 3 and 4, minutes for step 2, half an hour for step 8.
  `slide`    one upload: the image, its record and everything derived from it. This is
             the one that frees gigabytes, and the one that needs the file again.
  `all`      every slide and every cache. The "give me my disk back" button.

Every scope reports what it *would* free before it frees anything: `sweep(dry_run=True)`
is the default for everything destructive, and the endpoint makes a caller opt in.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: The scopes, in increasing order of what they destroy.
SCOPES: tuple[str, ...] = ("orphans", "derived", "slide", "all")

#: Scopes that need an explicit `dry_run=False` from the caller. `orphans` is absent
#: deliberately: it deletes only data that nothing can read, so demanding a confirmation
#: for it would train people to confirm without reading.
DESTRUCTIVE: frozenset[str] = frozenset({"derived", "slide", "all"})


def _derived_dirs() -> dict[str, Path]:
    """The per-upload cache directories, by the step that owns each.

    Keyed by step so a report can say *what* it is about to throw away in the terms a
    reader thinks in - "the class map" rather than "data/tissue_type".
    """
    return {
        "quality-control": settings.qc_dir,
        "tissue-mask": settings.tissue_dir,
        "white-calibration": settings.calibration_dir,
        # Step 7 keeps no pixels, but it does keep the viewer's committed choice - and
        # a directory nothing inventories is a directory that outlives its slide.
        "tiling": settings.tiling_dir,
        "tissue-type-segmentation": settings.tissue_type_dir,
        # Step 9 was absent from this table, which meant a region of interest survived
        # the slide it was drawn from. It is cheap to re-make and it still has to be
        # counted, because "clearing caches saves 1%" is only a true claim if the
        # inventory is complete.
        "roi-mask": settings.roi_dir,
        # Steps 10 and 11 are keyed on the H&E upload, like steps 2 to 9. Step 11's is
        # the expensive one to lose - it is minutes of BEETLE per region and it is the
        # tumour mask everything after it measures inside.
        "roi-selection": settings.roi_selection_dir,
        "roi-refinement": settings.roi_refinement_dir,
        # Steps 12 to 18 are keyed on a *pair* of uploads rather than one, so their
        # directories are named "<he>__<ihc>" and a sweep keyed on a single upload id
        # would never match them. Both were missing from this table; the inventory
        # claim "clearing caches saves 1%" is only true if it is complete.
        "ihc-alignment": settings.ihc_alignment_dir,
        "nuclei-segmentation": settings.nuclei_dir,
        "cell-typing": settings.cell_typing_dir,
        "compartments": settings.compartments_dir,
        # Steps 16, 17 and 18 were the last three missing, and they are the ones that
        # hold the actual deliverable: per-cell measurement rows, the binned levels,
        # and the score. They are keyed on the same "<he>__<ihc>" pair as 12-15, so
        # nothing keyed on one upload id would ever have matched them either. An
        # inventory that omits the output of the pipeline is not an inventory.
        "per-cell-measurement": settings.per_cell_dir,
        "intensity-binning": settings.binning_dir,
        "aggregate": settings.scores_dir,
    }


#: Roughly what re-making each cache costs, so a confirmation can say so. Measured on
#: this project's own slides rather than guessed; step 8's is the one that matters and is
#: the reason "just clear the caches" is not free.
REGENERATION_COST: dict[str, str] = {
    "quality-control": "a few minutes of CPU",
    "tissue-mask": "seconds",
    "white-calibration": "seconds",
    # Nothing to recompute - it is the viewer's choice, and clearing it means the next
    # visit starts on step 7's default rather than where they left off.
    "tiling": "nothing, but the chosen approach and scale are forgotten",
    "tissue-type-segmentation": "about half an hour of CPU",
    "roi-mask": "seconds, once the class map above exists",
    "roi-selection": "seconds, but the regions somebody chose are forgotten",
    # The second most expensive cache in the pipeline, and the one whose loss is least
    # visible: what it holds is the tumour mask every later step measures inside.
    "roi-refinement": "minutes of CPU per chosen region, and BEETLE runs again",
    # The registrar pickle is the expensive part and it lives in here, so clearing
    # this turns a 20-second reload back into a full registration.
    "ihc-alignment": "several minutes of CPU, and the registration is done again",
    "nuclei-segmentation": "a minute or two of CPU per marker",
    # Arithmetic over numbers step 11 already stored; it does not reopen a slide.
    "cell-typing": "under a second, once the nuclei above exist",
    "compartments": "a second or two, once the nuclei and the cell types exist",
    # Step 14 reopens the immunostained slide and reads every sampled field again, so
    # it is the only one of these three that costs more than arithmetic.
    "per-cell-measurement": "a minute or so of CPU - it reads the slide again",
    "intensity-binning": "under a second, once the measurements above exist",
    "aggregate": "under a second, once the measurements above exist",
}


def _size_of(path: Path) -> int:
    """Bytes under a path, following no symlinks and tolerating a vanishing tree."""
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            # A file removed between the walk and the stat. Counting it as zero is
            # right: it is no longer occupying anything.
            continue
    return total


def _slide_paths(upload_id: str) -> list[Path]:
    """Everything `slides/` holds for one upload: the image and its record."""
    return [
        path
        for path in settings.slides_dir.glob(f"{upload_id}.*")
        if path.is_file()
    ]


def known_uploads() -> set[str]:
    """Upload ids that still have a record, from either the staging or published side."""
    ids: set[str] = set()
    for path in settings.slides_dir.glob("*.json"):
        ids.add(path.stem)
    if settings.uploads_dir.exists():
        for entry in settings.uploads_dir.iterdir():
            if entry.is_dir():
                ids.add(entry.name)
    return ids


def is_orphaned(name: str, known: set[str]) -> bool:
    """Whether a derived directory called `name` can still be reached.

    **Steps 10 to 16 are keyed on a pair, not on one upload.** Their directories
    are named `<he>__<ihc>`, because what they hold is a property of two slides
    together - a registration, the nuclei inside the carried regions, the cells
    measured on one slide inside regions traced on the other. A single upload id
    cannot name any of that.

    This used to compare the whole directory name against the set of known upload
    ids. A composite name is never in that set, so every pair-keyed directory
    looked unreachable and the start-up sweep deleted it - steps 10 through 16,
    on every restart, for every case. It cost this project marker A's whole
    measurement chain once, which is how it was found.

    A pair is reachable when **both** of its slides still have a record. Either
    one being gone means the result can never be served again, which is the same
    test the single-keyed directories get, applied to each half.

    **Matched against the known ids, never split (P-19).** Upload ids come from
    `token_urlsafe`, whose alphabet includes `_`, so an id can itself contain `__`
    or end in `_` - and then `name.split("__")` cuts it in the wrong place.
    Delivered pair CAN_00270 ABCC4 has the IHC id `TS95__IA99rrDpqK5GLRxQ`: split,
    its pair directory is three parts, two of them unknown, and the sweep would
    have deleted it on the next start. So every `__` in the name is tried as the
    seam, and the name is reachable when any one of them divides it into two ids
    that both have a record.
    """
    if name in known:
        return False
    seam = name.find("__")
    while seam != -1:
        if name[:seam] in known and name[seam + 2 :] in known:
            return False
        seam = name.find("__", seam + 1)
    return True


@dataclass
class Item:
    """One thing that could be removed, and what removing it would cost."""

    upload_id: str
    kind: str
    path: Path
    bytes: int
    #: Present for a derived cache: what re-making it would take.
    cost: str | None = None
    #: True when nothing can read this any more - no upload record refers to it.
    orphaned: bool = False


@dataclass
class Usage:
    """What is on disk, grouped the way someone deciding what to delete needs it."""

    total_bytes: int = 0
    slide_bytes: int = 0
    derived_bytes: int = 0
    staging_bytes: int = 0
    orphan_bytes: int = 0

    uploads: dict[str, dict] = field(default_factory=dict)
    items: list[Item] = field(default_factory=list)

    @property
    def slide_share(self) -> float:
        """How much of the tree is slides. The number that makes the point."""
        return self.slide_bytes / self.total_bytes if self.total_bytes else 0.0


def usage() -> Usage:
    """Measure `data/` without changing anything.

    Walks the tree rather than trusting a running total, because the interesting cases -
    a cache left by a slide that was deleted, staging from an upload that died - are
    exactly the ones no bookkeeping knows about.
    """
    known = known_uploads()
    report = Usage()

    for path in settings.slides_dir.glob("*"):
        if not path.is_file():
            continue
        size = _size_of(path)
        report.slide_bytes += size
        upload_id = path.stem
        entry = report.uploads.setdefault(upload_id, {"slide": 0, "derived": {}, "orphaned": False})
        entry["slide"] += size

    for step, directory in _derived_dirs().items():
        if not directory.exists():
            continue
        for entry_path in directory.iterdir():
            if not entry_path.is_dir():
                continue
            upload_id = entry_path.name
            size = _size_of(entry_path)
            orphaned = is_orphaned(upload_id, known)

            report.derived_bytes += size
            if orphaned:
                report.orphan_bytes += size

            record = report.uploads.setdefault(
                upload_id, {"slide": 0, "derived": {}, "orphaned": orphaned}
            )
            record["derived"][step] = size
            record["orphaned"] = record["orphaned"] or orphaned

            report.items.append(
                Item(
                    upload_id=upload_id,
                    kind=step,
                    path=entry_path,
                    bytes=size,
                    cost=REGENERATION_COST.get(step),
                    orphaned=orphaned,
                )
            )

    if settings.uploads_dir.exists():
        for entry_path in settings.uploads_dir.iterdir():
            if not entry_path.is_dir():
                continue
            size = _size_of(entry_path)
            report.staging_bytes += size
            # Staging that survived a finished upload is waste: the parts have been
            # reassembled and nothing reads them again.
            finished = (settings.slides_dir / f"{entry_path.name}.json").is_file()
            if finished:
                report.orphan_bytes += size
            report.items.append(
                Item(
                    upload_id=entry_path.name,
                    kind="upload-staging",
                    path=entry_path,
                    bytes=size,
                    orphaned=finished,
                )
            )

    report.total_bytes = report.slide_bytes + report.derived_bytes + report.staging_bytes
    return report


@dataclass
class Swept:
    """What a sweep removed, or would have removed."""

    scope: str
    dry_run: bool
    freed_bytes: int = 0
    removed: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)


class MaintenanceError(ValueError):
    """A sweep was asked for something it will not do, and why."""


def _remove(path: Path, *, dry_run: bool) -> int:
    """Delete a file or tree, returning the bytes it held. Never raises on a race."""
    size = _size_of(path)
    if dry_run:
        return size
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        logger.warning("could not remove %s", path, exc_info=True)
        return 0
    return size


def sweep(
    scope: str = "orphans",
    *,
    upload_id: str | None = None,
    dry_run: bool = True,
) -> Swept:
    """Remove what `scope` names, and say what that freed.

    `dry_run` defaults to True and the destructive scopes require it to be turned off
    explicitly. This is not ceremony: `slide` and `all` destroy an upload that took
    minutes to transfer, and `derived` destroys a class map that took half an hour to
    compute. A caller that has not looked at `usage()` first has no business deleting
    either.

    **A running step is not consulted**, and that is a limitation worth stating rather
    than hiding. Sweeping while step 8 is mid-pass corrupts nothing - the pass writes its
    cache only at the end - but the pass will then write into a directory this call just
    emptied, and the result will look like a sweep that did not work. Cancel a run before
    sweeping the slide it is running on.
    """
    if scope not in SCOPES:
        raise MaintenanceError(f"unknown scope {scope!r}; expected one of {list(SCOPES)}")
    if scope in {"derived", "slide"} and not upload_id:
        raise MaintenanceError(
            f"scope {scope!r} needs an upload id - it acts on one slide. Use 'all' to "
            "act on every one."
        )

    report = usage()
    result = Swept(scope=scope, dry_run=dry_run)

    def take(path: Path, label: str) -> None:
        freed = _remove(path, dry_run=dry_run)
        if freed or path.exists() or dry_run:
            result.freed_bytes += freed
            result.removed.append(f"{label} ({freed / 1024 / 1024:.1f} MB)")

    if scope == "orphans":
        for item in report.items:
            if item.orphaned:
                take(item.path, f"{item.kind} for {item.upload_id}")
            else:
                result.kept.append(f"{item.kind} for {item.upload_id}")

    elif scope == "derived":
        for step, directory in _derived_dirs().items():
            path = directory / upload_id  # type: ignore[operator]
            if path.exists():
                take(path, f"{step} for {upload_id}")

    elif scope == "slide":
        for step, directory in _derived_dirs().items():
            path = directory / upload_id  # type: ignore[operator]
            if path.exists():
                take(path, f"{step} for {upload_id}")
        for path in _slide_paths(upload_id):  # type: ignore[arg-type]
            take(path, f"slide {path.name}")
        staging = settings.uploads_dir / str(upload_id)
        if staging.exists():
            take(staging, f"upload staging for {upload_id}")

    elif scope == "all":
        for _step, directory in _derived_dirs().items():
            for path in directory.glob("*"):
                take(path, f"{directory.name}/{path.name}")
        for path in settings.slides_dir.glob("*"):
            take(path, f"slides/{path.name}")
        if settings.uploads_dir.exists():
            for path in settings.uploads_dir.glob("*"):
                take(path, f"uploads/{path.name}")

    if not dry_run:
        settings.ensure_dirs()
        logger.info(
            "maintenance.sweep",
            extra={
                "scope": scope,
                "freed_mb": round(result.freed_bytes / 1024 / 1024, 1),
                "items": len(result.removed),
            },
        )

    return result


def discard_derived(upload_id: str) -> list[str]:
    """Remove every cache computed from one slide, pair-keyed ones included.

    `sweep(scope="derived")` removes the folders named after the upload. This also takes
    the `<he>__<ihc>` folders either half of which is this slide - matched against the
    id, never split, for the reason `is_orphaned` gives. Used when the slide's scale
    changes (step 1's override), which invalidates every measurement made from it.
    """
    removed: list[str] = []
    for step, directory in _derived_dirs().items():
        if not directory.exists():
            continue
        for path in directory.iterdir():
            name = path.name
            if name == upload_id or name.startswith(f"{upload_id}__") or name.endswith(f"__{upload_id}"):
                _remove(path, dry_run=False)
                removed.append(f"{step}/{name}")
    if removed:
        logger.info("maintenance.discard_derived", extra={"upload_id": upload_id, "items": len(removed)})
    return removed


def sweep_orphans_on_start() -> Swept:
    """Delete data nothing can read, at start-up. Never raises.

    Only `orphans`, and only because that scope cannot destroy anything a user could
    still want: a cache whose slide is gone can never be served again, and staging for a
    finished upload has already been reassembled. Anything that requires a judgement
    stays a deliberate act.

    This is the honest answer to "clean up when the browser closes". A backend cannot
    tell a closed tab from a refresh - `beforeunload` fires for both - so wiping on that
    signal would throw away half an hour of step 8 because somebody pressed F5. Sweeping
    what is provably unreachable, on every start, gets the disk back without guessing at
    intent.
    """
    try:
        result = sweep("orphans", dry_run=False)
        if result.freed_bytes:
            logger.info(
                "removed %.1f MB of unreachable data at start-up (%d items)",
                result.freed_bytes / 1024 / 1024,
                len(result.removed),
            )
        return result
    except Exception:  # noqa: BLE001 - start-up must not fail over housekeeping
        logger.warning("start-up orphan sweep failed", exc_info=True)
        return Swept(scope="orphans", dry_run=False)


def stale_uploads(older_than_days: float) -> list[str]:
    """Uploads whose slide file has not been touched in this long.

    Offered for a retention policy rather than run automatically: "old" is a judgement
    about how this machine is used, and a demo slide someone returns to next month is not
    waste. `settings.data_retention_days` turns it on.
    """
    if older_than_days <= 0:
        return []

    cutoff = time.time() - older_than_days * 86400
    stale = []
    for path in settings.slides_dir.glob("*.svs"):
        try:
            if path.stat().st_mtime < cutoff:
                stale.append(path.stem)
        except OSError:
            continue
    return sorted(stale)


__all__ = [
    "DESTRUCTIVE",
    "claim",
    "pending_releases",
    "release",
    "REGENERATION_COST",
    "SCOPES",
    "Item",
    "MaintenanceError",
    "Swept",
    "Usage",
    "is_orphaned",
    "known_uploads",
    "stale_uploads",
    "sweep",
    "sweep_orphans_on_start",
    "usage",
]


# --- releasing a slide when the browser goes away -----------------------------
#
# **This whole section is a guess about intent, and it is built to survive being
# wrong.** `pagehide` fires on a closed tab, on a refresh, and on following a link
# away; a backend cannot tell them apart. So a disconnect never deletes anything -
# it *schedules* a release, and any page that loads and claims the slide cancels it.
# A refresh gets there in about a second; a genuine close never does.
#
# The scope is pinned to derived caches. Deleting a 1.5 GB upload on a guess is a
# different order of mistake from costing a re-run, and no setting should be able to
# turn the first one on by accident.

#: Upload id -> the monotonic time after which its release may run.
_pending: dict[str, float] = {}
_pending_lock = threading.Lock()

#: Set once the sweeper thread is running, so repeated claims do not start more.
_sweeper_started = False


def claim(upload_id: str) -> bool:
    """Cancel any pending release for this slide. Returns whether one was cancelled.

    Called whenever a page loads with a slide in hand. This is the half of the design
    that makes a disconnect safe: the "browser closed" signal is unreliable, so the
    recovery from a wrong guess has to be automatic rather than a support request.
    """
    with _pending_lock:
        return _pending.pop(upload_id, None) is not None


def release(upload_id: str) -> float | None:
    """Schedule this slide's caches to be freed, unless something claims it first.

    Returns the grace period in seconds, or None when the feature is off. Never deletes
    anything itself - `_run_due_releases` does, after the grace has elapsed and only if
    no `claim` has intervened.
    """
    if not settings.cleanup_on_disconnect:
        return None

    grace = float(settings.cleanup_disconnect_grace_seconds)
    with _pending_lock:
        _pending[upload_id] = time.monotonic() + grace
    _start_sweeper()
    logger.info(
        "maintenance.release_scheduled",
        extra={"slide": upload_id, "grace_seconds": grace},
    )
    return grace


def pending_releases() -> dict[str, float]:
    """Seconds left before each pending release runs. For the tests and the endpoint."""
    now = time.monotonic()
    with _pending_lock:
        return {slide: round(due - now, 1) for slide, due in _pending.items()}


def _run_due_releases() -> int:
    """Free the caches of every slide whose grace has elapsed. Returns how many.

    The scope is read from settings but clamped: a disconnect may never delete a slide,
    whatever the configuration says. The trigger is a guess and the blast radius has to
    stay proportionate to that.
    """
    now = time.monotonic()
    with _pending_lock:
        due = [slide for slide, deadline in _pending.items() if deadline <= now]
        for slide in due:
            _pending.pop(slide, None)

    scope = settings.cleanup_disconnect_scope
    if scope not in {"derived", "orphans"}:
        logger.warning(
            "cleanup_disconnect_scope=%r is not allowed for a disconnect; using 'derived'",
            scope,
        )
        scope = "derived"

    for slide in due:
        try:
            result = sweep(scope, upload_id=slide, dry_run=False)
            logger.info(
                "maintenance.released",
                extra={
                    "slide": slide,
                    "freed_mb": round(result.freed_bytes / 1024 / 1024, 1),
                },
            )
        except Exception:  # noqa: BLE001 - housekeeping must not kill the thread
            logger.warning("release of %s failed", slide, exc_info=True)

    return len(due)


def _start_sweeper() -> None:
    """Start the daemon that runs due releases, once per process."""
    global _sweeper_started

    with _pending_lock:
        if _sweeper_started:
            return
        _sweeper_started = True

    def loop() -> None:
        while True:
            time.sleep(5.0)
            try:
                _run_due_releases()
            except Exception:  # noqa: BLE001
                logger.warning("release sweeper tick failed", exc_info=True)

    # A daemon thread: this is housekeeping, and it must never hold the process open at
    # shutdown. A release missed by an exit is picked up by the start-up orphan sweep or
    # by the next disconnect, so nothing leaks permanently.
    threading.Thread(target=loop, name="maintenance-release", daemon=True).start()
