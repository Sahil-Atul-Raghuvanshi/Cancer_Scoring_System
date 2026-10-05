"""A bounded in-process memo, for steps that recompute rather than store.

Steps 3 and 4 deliberately keep almost nothing on disk: their results depend on a
threshold the viewer can move, so caching the answer would mean caching something
that is about to be wrong. What they do keep is a memo of the *last* answer, because
a single view of one of those screens fetches the report and up to four panels, and
without it that is five identical runs of the same arithmetic.

**Why that memo has to hold more than one entry now.** It used to hold exactly one,
on the argument that nobody looks at two slides at once. That stopped being true when
steps 2 to 6 became a per-slide prefix: a case has an H&E and an immunostained slide,
both are on screen, and the viewer flips between them. With one slot, every flip
evicts the other slide and re-runs step 3's morphology and step 4's glass selection
from scratch - so the toggle that exists to make the two comparable is also the thing
that makes comparing them slow.

`capacity` is therefore read as "how many slides, times how many settings of the one
control that matters". Two slides and two recent thresholds is four, which is the
default; it is a small number on purpose, because these values hold thumbnails and
mask arrays and the point of the memo is to avoid recomputation, not to become the
storage this step declined to have.

Eviction is least-recently-used rather than insertion-ordered: the entry to throw away
when a viewer is alternating between two slides is the one they have not looked at,
which is not the one that arrived first.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar

T = TypeVar("T")

#: Two slides, two recent settings of the threshold. See the module docstring.
DEFAULT_CAPACITY = 4


class BoundedMemo(Generic[T]):
    """Keeps the last few computed values, keyed by whatever identifies them.

    Carries its own lock rather than borrowing the caller's, so a service does not
    have to hold a mutex across the work it is memoising - `get` returning None is
    the whole of the critical section, and the expensive call happens outside it.
    Two callers racing on the same key both compute it and the second `put` wins,
    which costs one duplicated run and avoids serialising every request behind one
    slide's arithmetic.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._entries: OrderedDict[Hashable, T] = OrderedDict()

    def get(self, key: Hashable) -> T | None:
        """The value stored under `key`, or None - and mark it most recently used."""
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key]

    def put(self, key: Hashable, value: T) -> None:
        """Store `value`, evicting the least recently used entry if full."""
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Forget everything. For tests, and for a slide being released."""
        with self._lock:
            self._entries.clear()


__all__ = ["BoundedMemo", "DEFAULT_CAPACITY"]
