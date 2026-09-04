"""Plain-image-backed slide reader (JPEG/PNG) with an in-memory pyramid.

An ordinary JPEG or PNG is single-resolution - it is not a pyramidal TIFF, so
`tiffslide` cannot open it. This reader loads such an image and synthesises a
2x-downsample pyramid in memory, exposing the SAME interface as `SlideReader`
so callers work unchanged.

It exists so the demo can be driven with a small sample image without needing a
multi-gigabyte scan to hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class ImageSlideReader:
    """A plain image presented as a slide, with a synthetic pyramid."""

    def __init__(self, path: str | Path, min_level_dim: int = 512) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        base = Image.open(self.path).convert("RGB")
        self.width, self.height = base.size
        self._levels: list[Image.Image] = [base]

        width, height = self.width, self.height
        while max(width, height) > min_level_dim:
            width, height = max(1, width // 2), max(1, height // 2)
            self._levels.append(self._levels[-1].resize((width, height), Image.BILINEAR))

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def level_count(self) -> int:
        return len(self._levels)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple(level.size for level in self._levels)

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return tuple(self.width / level.size[0] for level in self._levels)

    def best_level_for_downsample(self, downsample: float) -> int:
        best = 0
        for index, factor in enumerate(self.level_downsamples):
            if factor <= downsample + 1e-6:
                best = index
        return best

    def best_level_for_mpp(self, target_mpp: float) -> int:
        """Without a recorded mpp there is nothing physical to convert against."""
        base = self.mpp
        if not base or target_mpp <= 0:
            return 0
        return self.best_level_for_downsample(target_mpp / base)

    @property
    def mpp(self) -> float | None:
        return None  # a plain image carries no physical scale

    @property
    def objective_power(self) -> float | None:
        return None

    @property
    def vendor(self) -> str | None:
        return None

    def read_region_pil(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Any:
        x0, y0 = location  # level-0 coordinates
        factor = self.level_downsamples[level]
        left, top = int(x0 / factor), int(y0 / factor)
        width, height = size
        return self._levels[level].crop((left, top, left + width, top + height)).convert("RGB")

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        return np.asarray(self.read_region_pil(location, level, size))

    def thumbnail_pil(self, max_size: int = 1024) -> Any:
        image = self._levels[0].copy()
        image.thumbnail((max_size, max_size))
        return image.convert("RGB")

    def thumbnail(self, max_size: int = 1024) -> np.ndarray:
        return np.asarray(self.thumbnail_pil(max_size))

    def associated_image_names(self) -> list[str]:
        return []

    def read_associated(self, key: str, *, allow_identifying: bool = False) -> Any:
        raise ValueError(f"no associated image {key!r}: this is a plain image, not a scan")

    def close(self) -> None:
        for level in self._levels:
            level.close()
        self._levels = []

    def __enter__(self) -> ImageSlideReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
