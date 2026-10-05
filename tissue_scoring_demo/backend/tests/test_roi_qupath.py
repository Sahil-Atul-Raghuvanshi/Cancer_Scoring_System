"""Tests for the GeoJSON export - the one place a ring is closed for the wire.

`trace_rings` deliberately leaves the last vertex unrepeated (see `mask.py`); GeoJSON's
`Polygon` spec requires the first and last positions to be equivalent, so this is the
one boundary where that repeat has to happen, and the one thing worth pinning here.
"""

from __future__ import annotations

from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_COLOURS, SCORED
from app.pipeline.step09_roi_mask.borders import class_regions
from app.pipeline.step09_roi_mask.mask import IN_SITU
from app.pipeline.step09_roi_mask.qupath import to_geojson
from tests.test_roi_mask import field, make_class_map, paint


def test_every_ring_closes():
    grid = field(20, 20)
    paint(grid, slice(4, 12), slice(4, 12), SCORED)
    regions = class_regions(make_class_map(grid))
    payload = to_geojson(regions)

    assert payload["type"] == "FeatureCollection"
    feature = payload["features"][0]
    for ring in feature["geometry"]["coordinates"]:
        assert ring[0] == ring[-1]
        assert len(ring) >= 4  # a closed rectilinear ring has at least 4 distinct points


def test_classification_and_colour_match_step_8s_own_palette():
    grid = field(20, 20)
    paint(grid, slice(4, 12), slice(4, 12), SCORED)
    paint(grid, slice(4, 12), slice(14, 18), IN_SITU)
    regions = class_regions(make_class_map(grid))
    payload = to_geojson(regions)

    by_name = {f["properties"]["classification"]["name"]: f for f in payload["features"]}
    assert by_name["Invasive"]["properties"]["classification"]["color"] == list(
        CLASS_COLOURS[SCORED]
    )
    assert by_name["DCIS"]["properties"]["classification"]["color"] == list(
        CLASS_COLOURS[IN_SITU]
    )


def test_stroma_never_appears_in_the_export():
    grid = field(10, 10)  # all non_epithelium
    regions = class_regions(make_class_map(grid))
    payload = to_geojson(regions)

    assert payload["features"] == []


def test_an_empty_class_map_still_exports_a_valid_empty_collection():
    grid = field(6, 6)
    payload = to_geojson(class_regions(make_class_map(grid)))
    assert payload == {"type": "FeatureCollection", "features": []}
