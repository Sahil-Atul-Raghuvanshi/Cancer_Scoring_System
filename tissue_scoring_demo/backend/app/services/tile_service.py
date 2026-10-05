"""Serves Deep Zoom tiles for uploaded slides.

Opening a whole-slide image costs on the order of a second, and a viewer asks
for tiles by the dozen, so open `DeepZoom` generators are cached and reused.
The cache is bounded and evicts least-recently-used, closing the reader as it
goes - each entry holds an open file handle, and an unbounded cache would leak
them one slide at a time.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from io import BytesIO

from app.core.logging import get_logger
from app.ingestion.deepzoom import DeepZoom
from app.ingestion.slide_reader import open_slide
from app.services.upload_service import resolve_ready_path

logger = get_logger(__name__)

#: Slides kept open at once. Small: this is a demo, and each entry is a file
#: handle plus whatever the reader caches internally.
MAX_OPEN_SLIDES = 4

TILE_SIZE = 256
JPEG_QUALITY = 80


class TileService:
    """Deep Zoom descriptors and tiles, over a bounded cache of open slides."""

    def __init__(self) -> None:
        self._cache: OrderedDict[str, DeepZoom] = OrderedDict()
        self._lock = threading.Lock()

    def _generator(self, upload_id: str) -> DeepZoom:
        """The DeepZoom for a slide, opening it only if it is not already cached."""
        with self._lock:
            cached = self._cache.get(upload_id)
            if cached is not None:
                self._cache.move_to_end(upload_id)
                return cached

        # Open outside the lock: this is the slow part, and holding the cache
        # lock through it would serialise every other slide's tiles behind it.
        path = resolve_ready_path(upload_id=upload_id)
        generator = DeepZoom(open_slide(path), tile_size=TILE_SIZE, fmt="jpeg")

        evicted: DeepZoom | None = None
        with self._lock:
            # Another thread may have opened the same slide while we were here.
            existing = self._cache.get(upload_id)
            if existing is not None:
                self._cache.move_to_end(upload_id)
                evicted, generator = generator, existing
            else:
                self._cache[upload_id] = generator
                self._cache.move_to_end(upload_id)
                if len(self._cache) > MAX_OPEN_SLIDES:
                    _, evicted = self._cache.popitem(last=False)

        if evicted is not None:
            try:
                evicted.close()
            except OSError:
                logger.warning("failed to close an evicted slide reader", exc_info=True)

        return generator

    def descriptor(self, upload_id: str) -> str:
        """The DZI XML the viewer reads before requesting any tile."""
        return self._generator(upload_id).descriptor()

    def tile_jpeg(self, upload_id: str, *, level: int, col: int, row: int) -> bytes:
        """One tile as JPEG bytes.

        JPEG rather than PNG: these are photographic, there are thousands of
        them, and the size difference is what makes panning feel immediate.
        """
        image = self._generator(upload_id).get_tile(level, col, row)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        return buffer.getvalue()

    def forget(self, upload_id: str) -> None:
        """Drop a slide from the cache, closing its reader."""
        with self._lock:
            generator = self._cache.pop(upload_id, None)
        if generator is not None:
            try:
                generator.close()
            except OSError:
                logger.warning("failed to close slide reader", exc_info=True)

    def close_all(self) -> None:
        """Close every open slide, at shutdown."""
        with self._lock:
            generators = list(self._cache.values())
            self._cache.clear()
        for generator in generators:
            try:
                generator.close()
            except OSError:
                logger.warning("failed to close slide reader", exc_info=True)


tile_service = TileService()
