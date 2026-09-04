"""A one-at-a-time background worker, so the UI can watch a four-minute run.

Segmenting a region takes 20 seconds on one fold and around two minutes on five, which
is far too long for a request to sit open on. So the API starts a job and returns its
id, and the browser polls.

**One worker thread, deliberately.** The forward pass already uses every core the
machine has (`config.torch_threads`), so a second concurrent region would not run twice
as fast - it would run each half as fast, double the peak memory to two 370 MB
checkpoints plus two sets of region-sized float32 accumulators, and make the progress
estimate meaningless. A queue is the honest structure for work that is CPU-bound.

State is in memory and dies with the process. That is a real limitation and it is the
right one here: the *outputs* are on disk under `data/runs/`, so a restarted server
loses the progress bars and nothing else. Persisting job state would mean reconciling
it with the filesystem on boot, which is more machinery than a single-user research
tool earns.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Queue
from typing import Any, Callable

#: How many finished jobs to remember. Enough that a batch of 50 regions can be read
#: back in full after it finishes; old entries are evicted oldest-first.
MAX_REMEMBERED = 400


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    """One unit of work, and everything the UI needs to draw it."""

    id: str
    kind: str
    label: str
    #: `queued` -> `running` -> `done` | `failed` | `cancelled`.
    status: str = "queued"
    message: str = "waiting for the worker"
    fraction: float = 0.0
    created: str = field(default_factory=_now)
    finished: str | None = None
    result: Any = None
    error: str | None = None
    #: Set by `cancel`; checked by the work function between regions. A running forward
    #: pass is never interrupted - torch has no safe way to do that - so cancelling a
    #: batch stops it after the current region rather than immediately.
    cancelled: bool = False

    def as_json(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            # Reported separately from `status`, which stays `running` until the current
            # region finishes. Without this the UI cannot distinguish "still working"
            # from "working, but stopping after this one", because the pipeline's own
            # progress messages overwrite anything `cancel` writes into `message`.
            "stopping": self.cancelled and self.status == "running",
            "message": self.message,
            "fraction": round(self.fraction, 4),
            "created": self.created,
            "finished": self.finished,
            "result": self.result,
            "error": self.error,
        }


class JobRunner:
    """A queue, a thread, and a bounded record of what has happened."""

    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: Queue[tuple[Job, Callable[[Job], Any]]] = Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name="bracs-worker", daemon=True)
        self._worker.start()

    # --- public ---------------------------------------------------------------

    def submit(self, kind: str, label: str, work: Callable[[Job], Any]) -> Job:
        """Queue `work`, which is handed the `Job` so it can report progress."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_REMEMBERED:
                # Never evict something still queued or running - only history.
                for key, candidate in list(self._jobs.items()):
                    if candidate.status in ("done", "failed", "cancelled"):
                        del self._jobs[key]
                        break
                else:
                    break
        self._queue.put((job, work))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())[-limit:][::-1]

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop. Queued jobs stop at once; running ones after this region."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in ("done", "failed", "cancelled"):
                return False
            job.cancelled = True
            if job.status == "queued":
                job.status = "cancelled"
                job.message = "cancelled before it started"
                job.finished = _now()
            else:
                job.message = "cancelling after the current region"
            return True

    # --- the worker -----------------------------------------------------------

    def _run(self) -> None:
        while True:
            job, work = self._queue.get()
            if job.cancelled:
                self._queue.task_done()
                continue

            job.status = "running"
            job.message = "starting"
            try:
                job.result = work(job)
                job.status = "cancelled" if job.cancelled else "done"
                job.message = (
                    "stopped early" if job.cancelled else "finished"
                )
                job.fraction = 1.0
            except Exception as error:  # noqa: BLE001 - a worker that dies is worse
                job.status = "failed"
                job.error = f"{type(error).__name__}: {error}"
                # The traceback goes to the server log, not to the browser: it names
                # absolute paths on the host, and the API is the wrong place for them.
                traceback.print_exc()
                job.message = "failed"
            finally:
                job.finished = _now()
                self._queue.task_done()


#: The process-wide runner. One, because there is one CPU to share.
RUNNER = JobRunner()


def report(job: Job) -> Callable[[str, float], None]:
    """A `progress` callable that writes into `job`, for handing to the pipeline."""

    def progress(message: str, fraction: float) -> None:
        job.message = message
        job.fraction = max(0.0, min(1.0, float(fraction)))

    return progress
