"""Tests for `crop_png`'s level choice, against a fake reader rather than a real slide.

The fake exposes exactly the three calls `crop_png` makes - `best_level_for_downsample`,
`level_downsamples`, `read_region_pil` - so this is a test of the cropping arithmetic,
not of `tiffslide`.
"""

from __future__ import annotations

import io

from PIL import Image

from app.pipeline.step09_roi_mask.crops import crop_png


class FakeReader:
    """A slide with three levels, 1x/4x/16x downsample, solid colour per level."""

    def __init__(self) -> None:
        self.level_downsamples = (1.0, 4.0, 16.0)
        self.requested: tuple[int, int, int] | None = None

    def best_level_for_downsample(self, downsample: float) -> int:
        best = 0
        for index, factor in enumerate(self.level_downsamples):
            if factor <= downsample + 1e-6:
                best = index
        return best

    def read_region_pil(self, location, level, size):
        self.requested = (level, *size)
        return Image.new("RGB", size, (10 * level, 10 * level, 10 * level))


def test_a_small_region_is_read_at_level_0():
    reader = FakeReader()
    crop_png(reader, (0, 0, 400, 400), max_size=1600)
    assert reader.requested[0] == 0
    assert reader.requested[1:] == (400, 400)


def test_a_large_region_is_read_at_a_coarser_level_capped_to_max_size():
    reader = FakeReader()
    # 40,000 px wide at max_size=1000 wants a >=40x downsample - level 2 (16x) is the
    # finest level under that, so the read happens there and is still capped after.
    png = crop_png(reader, (0, 0, 40_000, 20_000), max_size=1000)
    assert reader.requested[0] == 2

    image = Image.open(io.BytesIO(png))
    assert max(image.size) <= 1000


def test_the_crop_keeps_its_aspect_ratio():
    reader = FakeReader()
    png = crop_png(reader, (0, 0, 8_000, 2_000), max_size=1000)
    image = Image.open(io.BytesIO(png))
    assert image.size[0] / image.size[1] == 4
