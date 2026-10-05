"""Deep Zoom (DZI) tile generator over the slide reader.

tiffslide ships only an Aperio-specific generator, so the standard DZI scheme is
implemented here. Tiles are produced on the fly - nothing is pre-generated - by
reading the most suitable slide pyramid level and finishing the downsample in
memory. That keeps it memory-light and scanner-agnostic, and it is what makes a
126,976 px slide pannable in a browser: the viewer only ever fetches the few
hundred-pixel tiles currently on screen.

OpenSeadragon consumes `descriptor()` for the image dimensions and then requests
tiles by level/column/row.

Note the two different level numberings, which are easy to confuse:

  slide levels  the pyramid actually inside the file, coarsening by whatever
                factors the scanner chose (often 1, 4, 16 - not powers of two)
  DZI levels    a strict power-of-two ladder from 1x1 up to full resolution,
                which is what the DZI format mandates

`get_tile` maps the second onto the first.
"""

from __future__ import annotations

import math
import threading
from typing import Any

from PIL import Image


class DeepZoom:
    """Serves DZI tiles for one open slide."""

    def __init__(self, reader: Any, tile_size: int = 256, fmt: str = "jpeg") -> None:
        self.reader = reader
        self.tile_size = tile_size
        self.fmt = fmt
        self.width, self.height = reader.dimensions

        # DZI level 0 is a single pixel; the top level is full resolution.
        self.max_level = int(math.ceil(math.log2(max(self.width, self.height, 1))))

        # A viewer opens ~10-20 tiles at once. The underlying reader holds file
        # state that is not guaranteed thread-safe, and uvicorn runs sync
        # handlers in a threadpool, so reads are serialised per slide.
        self._lock = threading.Lock()

    def descriptor(self) -> str:
        """The DZI XML descriptor."""
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" '
            f'Format="{self.fmt}" Overlap="0" TileSize="{self.tile_size}">'
            f'<Size Width="{self.width}" Height="{self.height}"/>'
            "</Image>"
        )

    def level_scale(self, level: int) -> float:
        """Downsample from full resolution for a DZI level."""
        return float(2 ** (self.max_level - level))

    def level_dimensions(self, level: int) -> tuple[int, int]:
        scale = self.level_scale(level)
        return (
            max(1, math.ceil(self.width / scale)),
            max(1, math.ceil(self.height / scale)),
        )

    def tile_count(self, level: int) -> tuple[int, int]:
        """Columns and rows at a DZI level."""
        width, height = self.level_dimensions(level)
        return math.ceil(width / self.tile_size), math.ceil(height / self.tile_size)

    def get_tile(self, level: int, col: int, row: int) -> Image.Image:
        """Render one tile, reading the cheapest slide level that can serve it."""
        if level < 0 or level > self.max_level:
            raise ValueError(f"level {level} out of range 0..{self.max_level}")
        if col < 0 or row < 0:
            raise ValueError("tile coordinates must not be negative")

        scale = self.level_scale(level)

        # The region this tile covers, in level-0 coordinates.
        x0 = int(col * self.tile_size * scale)
        y0 = int(row * self.tile_size * scale)
        if x0 >= self.width or y0 >= self.height:
            raise ValueError("tile out of bounds")

        w0 = min(int(self.tile_size * scale), self.width - x0)
        h0 = min(int(self.tile_size * scale), self.height - y0)

        with self._lock:
            # Read from the finest pyramid level that is still coarser-or-equal
            # to what is needed, then shrink the rest of the way. Reading level 0
            # for a zoomed-out tile would pull megabytes to produce 256 px.
            best = self.reader.best_level_for_downsample(scale)
            factor = self.reader.level_downsamples[best]
            read_w = max(1, round(w0 / factor))
            read_h = max(1, round(h0 / factor))
            image = self.reader.read_region_pil((x0, y0), best, (read_w, read_h))

        target_w = max(1, round(w0 / scale))
        target_h = max(1, round(h0 / scale))
        if (image.width, image.height) != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.BILINEAR)
        return image

    def close(self) -> None:
        with self._lock:
            self.reader.close()
