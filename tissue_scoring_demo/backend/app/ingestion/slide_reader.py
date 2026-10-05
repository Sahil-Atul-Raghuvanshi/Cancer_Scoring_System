"""Format-abstracted whole-slide reader (pipeline step 1).

Isolates scanner and format specifics behind one class so everything downstream
is scanner-agnostic. This build uses `tiffslide` - pure Python, easy on Windows,
reads pyramidal TIFF and Aperio SVS. A native OpenSlide backend slots in here
later without touching callers.

`open_slide()` is the factory the rest of the app should use: pyramidal formats
go to `SlideReader`; plain JPEG/PNG go to `ImageSlideReader`, which exposes the
identical interface.

Step 1 exists to answer one question - *what magnification am I working at?* -
and that answer is physical, expressed in microns per pixel. Different scanners
put different resolutions at the same level index, so callers must convert with
`best_level_for_mpp()` and never hard-code a level number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_PYRAMIDAL_EXTS = {".tif", ".tiff", ".svs", ".scn", ".bif"}


def open_slide(path: str | Path) -> Any:
    """Return the right reader for the file type (both share one interface)."""
    if Path(path).suffix.lower() in _PYRAMIDAL_EXTS:
        return SlideReader(path)
    from .image_reader import ImageSlideReader

    return ImageSlideReader(path)


class SlideReader:
    """A pyramidal whole-slide image, read one region at a time."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        import tiffslide

        self._slide = tiffslide.TiffSlide(str(self.path))

    # --- geometry ---------------------------------------------------------------

    @property
    def dimensions(self) -> tuple[int, int]:
        """Level-0 (width, height) in pixels."""
        return self._slide.dimensions

    @property
    def level_count(self) -> int:
        return int(self._slide.level_count)

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._slide.level_dimensions)

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return tuple(self._slide.level_downsamples)

    def best_level_for_downsample(self, downsample: float) -> int:
        """Highest slide level whose downsample is <= the requested downsample."""
        best = 0
        for index, factor in enumerate(self.level_downsamples):
            if factor <= downsample + 1e-6:
                best = index
        return best

    def best_level_for_mpp(self, target_mpp: float) -> int:
        """The level to work at for a target resolution in microns per pixel.

        This is the conversion step 1 is really about. With no mpp recorded on
        the slide there is nothing physical to convert against, so fall back to
        level 0 rather than guessing.
        """
        base = self.mpp
        if not base or target_mpp <= 0:
            return 0
        return self.best_level_for_downsample(target_mpp / base)

    # --- physical scale ---------------------------------------------------------

    @property
    def mpp(self) -> float | None:
        """Microns per pixel at level 0, or None when the scanner did not record it."""
        for key in ("tiffslide.mpp-x", "openslide.mpp-x", "aperio.MPP"):
            value = self._slide.properties.get(key)
            if value:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    @property
    def objective_power(self) -> float | None:
        """Nominal objective magnification, e.g. 40 for a 40x scan."""
        for key in ("tiffslide.objective-power", "openslide.objective-power", "aperio.AppMag"):
            value = self._slide.properties.get(key)
            if value:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    @property
    def vendor(self) -> str | None:
        for key in ("tiffslide.vendor", "openslide.vendor"):
            value = self._slide.properties.get(key)
            if value:
                return str(value)
        return None

    # --- pixels -----------------------------------------------------------------

    def read_region_pil(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> Any:
        """A PIL RGB image for the requested region (location in level-0 coords)."""
        return self._slide.read_region(location, level, size).convert("RGB")

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        """An HxWx3 uint8 RGB array for the requested region."""
        return np.asarray(self.read_region_pil(location, level, size))

    def thumbnail_pil(self, max_size: int = 1024) -> Any:
        """A whole-slide overview, downsampled from the pyramid.

        Uses `get_thumbnail`, which reads an appropriate pyramid level. It does
        *not* touch `associated_images` - see the note below for why that
        matters.
        """
        return self._slide.get_thumbnail((max_size, max_size)).convert("RGB")

    def thumbnail(self, max_size: int = 1024) -> np.ndarray:
        return np.asarray(self.thumbnail_pil(max_size))

    # --- associated images: never decode one just because it exists --------------
    #
    # Aperio-family files carry extra images alongside the pyramid, conventionally
    # named "thumbnail", "label" and "macro". Two facts about real scanner output
    # make naive access dangerous:
    #
    #   1. The series named "thumbnail" is often not a thumbnail. On scanner output
    #      it can be a half-resolution copy of the whole slide - tens of thousands
    #      of pixels square. Decoding it in a request handler is an outage.
    #   2. "label" and "macro" are photographs of the physical slide, showing the
    #      case number, block ID and a barcode. They are identifiers and must not
    #      be served.
    #
    # So: shapes are cheap and safe to report; pixels require an explicit budget.
    MAX_ASSOCIATED_PIXELS = 64_000_000  # ~8000 x 8000; a genuine overview is far smaller
    IDENTIFYING_ASSOCIATED = frozenset({"label", "macro"})

    def associated_image_names(self) -> list[str]:
        """Names of associated images, without decoding any pixels."""
        return sorted(str(name) for name in self._slide.associated_images)

    def read_associated(self, key: str, *, allow_identifying: bool = False) -> Any:
        """Decode one associated image, refusing identifier-bearing and oversized ones.

        Raises ValueError rather than returning None: a silent skip here is how a
        de-identification boundary quietly stops being one.
        """
        name = key.strip().lower()
        if name in self.IDENTIFYING_ASSOCIATED and not allow_identifying:
            raise ValueError(
                f"associated image {name!r} shows the physical slide label and carries "
                "case and block identifiers; it must not be decoded or served."
            )
        if name not in self._slide.associated_images:
            raise ValueError(f"no associated image {name!r} in {self.path.name}")

        from PIL import Image

        image = self._slide.associated_images[name]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))

        pixels = image.size[0] * image.size[1]
        if pixels > self.MAX_ASSOCIATED_PIXELS:
            raise ValueError(
                f"associated image {name!r} is {image.size} = {pixels:,} pixels, over the "
                f"{self.MAX_ASSOCIATED_PIXELS:,} budget. Read a pyramid level instead."
            )
        return image.convert("RGB")

    def close(self) -> None:
        self._slide.close()

    def __enter__(self) -> SlideReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
