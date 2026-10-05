"""GeoJSON export of the border regions, in the shape QuPath's own import expects.

QuPath opens a `.geojson` file (drag-and-drop, or File > Object > Import objects) as
one annotation per `Feature`, coloured and named from `properties.classification`. That
is the only QuPath-specific knowledge in this module - everything else is a plain
GeoJSON `FeatureCollection` built from `borders.class_regions()`.

**Rings must close, and `trace_rings` deliberately does not close them.** A ring there
ends the moment its walk returns to the start vertex, which is the vertex list's own
first entry - repeating it would be dead weight on every consumer that does not need a
closed ring. GeoJSON's `Polygon` spec does need one: "the first and last positions
[must] be equivalent". So the repeat happens once, here, at the one boundary that
requires it.
"""

from __future__ import annotations

from typing import Any

from app.pipeline.step08_tissue_type_segmentation.classes import CLASS_COLOURS, SCORED
from app.pipeline.step08_tissue_type_segmentation.uncertainty import UNCERTAIN

from .borders import BORDER_CLASSES, ClassRegion
from .mask import IN_SITU

#: The label and colour QuPath shows for each border class. Names chosen for the
#: annotation list a pathologist reads, not the model's own class names - "DCIS" and
#: "Invasive" are what QuPath users already call these.
CLASSIFICATION: dict[int, tuple[str, tuple[int, int, int]]] = {
    SCORED: ("Invasive", CLASS_COLOURS[SCORED]),
    IN_SITU: ("DCIS", CLASS_COLOURS[IN_SITU]),
    UNCERTAIN: ("Uncertain", CLASS_COLOURS[UNCERTAIN]),
}


def _polygon(region: ClassRegion) -> dict[str, Any]:
    """A GeoJSON `Polygon` geometry: outer ring first, then holes, each closed."""
    return {
        "type": "Polygon",
        "coordinates": [
            [[x, y] for x, y in (*ring, ring[0])] for ring in region.rings
        ],
    }


def _feature(region: ClassRegion) -> dict[str, Any]:
    name, colour = CLASSIFICATION[region.class_id]
    return {
        "type": "Feature",
        "geometry": _polygon(region),
        "properties": {
            "objectType": "annotation",
            "name": f"{name} {region.index + 1}",
            "classification": {"name": name, "color": list(colour)},
            "measurements": [{"name": "Area", "value": region.area_mm2, "unit": "mm^2"}],
        },
    }


def to_geojson(regions_by_class: dict[int, list[ClassRegion]]) -> dict[str, Any]:
    """Every border-class region, as one `FeatureCollection` QuPath can import."""
    features = [
        _feature(region)
        for class_id in BORDER_CLASSES
        for region in regions_by_class.get(class_id, [])
    ]
    return {"type": "FeatureCollection", "features": features}


__all__ = ["CLASSIFICATION", "to_geojson"]
