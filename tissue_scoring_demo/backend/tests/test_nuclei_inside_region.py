"""Step 13 counts only nuclei inside the region (P-10).

A field is accepted when 35 % of it is inside the region, so up to 65 % of it can be
stroma, in-situ disease or normal tissue - and every nucleus in it used to be counted.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.step13_nuclei_segmentation.sampling import Field
from app.pipeline.step13_nuclei_segmentation.segment import region_mask, tissue_share

FIELD = Field(index=0, x=1000, y=2000, span=512, size=256, tissue=0.5)


def test_the_region_is_drawn_on_the_fields_own_grid():
    # The left half of the field, in level-0 pixels (2 level-0 px per field px).
    rings = [[[1000, 2000], [1256, 2000], [1256, 2512], [1000, 2512]]]
    mask = region_mask(rings, FIELD, size=256, level0_scale=2.0)
    assert mask[:, :120].all()
    assert not mask[:, 136:].any()


def test_a_hole_in_the_region_is_outside_it():
    outer = [[900, 1900], [1600, 1900], [1600, 2600], [900, 2600]]
    hole = [[1200, 2200], [1300, 2200], [1300, 2300], [1200, 2300]]
    mask = region_mask([outer, hole], FIELD, size=256, level0_scale=2.0)
    assert mask[10, 10]
    assert not mask[125, 125]  # field px (125, 125) is level-0 (1250, 2250): in the hole


def test_tissue_share_is_measured_over_the_inside_part_only():
    rgb = np.full((256, 256, 3), 245, dtype=np.uint8)  # glass
    rng = np.random.default_rng(0)
    rgb[:, :128] = rng.integers(60, 160, size=(256, 128, 3))  # textured tissue, left half
    within = np.zeros((256, 256), dtype=bool)
    within[:, :128] = True
    white = np.array([250.0, 250.0, 250.0])
    assert tissue_share(rgb, white, border_px=8, within=within) > 0.9
    assert tissue_share(rgb, white, border_px=8) < 0.6
