"""Which regions cross to the IHC slide, and in what shape.

**What crosses is step 11's pixel boundary.** Until step 11 existed this module chose
regions itself - the largest patches of invasive *tile*, taken in rank order until they
covered enough of the tumour - and what it carried was a staircase of 224 um windows. It
no longer chooses and it no longer carries squares: a person ticked the regions on step
10 and BEETLE traced them per pixel on step 11, so this module's job is to take that
boundary across unchanged. `selected_regions` is what reads it.

`invasive_regions` and `coverage` are kept for the tile path, which is now only reachable
when step 11 has not run - and `ihc_alignment_service` refuses that rather than falling
back to it. They remain because the tests exercise them and because deleting the
comparison would make it hard to see what changed.

Two decisions still live here.

**Only invasive carcinoma.** Everything downstream of this step measures brown inside
these regions, and the panel is scored on invasive tumour - in-situ disease is explicitly
excluded from the denominator (Rule 5, and step 9 already carves it out of the scored
ROI, and step 11 traces only BEETLE's invasive code). Carrying DCIS across would invite
it back in.

**Densified before warping, simplified after.** See `app.registration.transform` for why:
a non-rigid warp bends the edges between vertices, so a ring has to be subdivided before
it is warped or the deformation along every edge is lost. That mattered for a tile
staircase, whose vertices sat a whole window apart. It matters less for a pixel boundary,
whose vertices are already a couple of microns apart - `densify_ring` is then close to a
no-op, which is correct rather than wasteful: the rule is about the spacing, not about
where the ring came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.pipeline.step08_tissue_type_segmentation.classes import SCORED
from app.pipeline.step09_roi_mask.borders import ClassRegion
from app.registration.transform import densify_ring, simplify_ring, um_to_px


@dataclass(frozen=True)
class Crossable:
    """Whatever this step is about to carry, in the one shape the warp needs.

    A deliberately small surface - an id, an area and a polygon-with-holes - so that a
    focus traced by BEETLE and a patch of tile traced by step 9 reach the registration
    through exactly the same code. The alternative was a fork at every stage between
    here and `regroup`, which is four places for the two paths to drift.

    `rings` is **one piece**: the outer ring first, then its holes. Never several foci in
    one list - `sampling.rasterise` reads every ring after the first as a hole, so a
    flattened multi-focus region becomes its first focus with the others punched out.
    """

    #: Where this came from, for the report. `ROI-007` for a refined focus; step 9's
    #: rank spelled the same way for a tile patch.
    roi_id: str
    index: int
    cells: int
    area_mm2: float
    rings: list[list[list[float]]]
    #: `beetle_pixel` or `tile_roi`. Carried onto `AlignedRegion` so a stored report says
    #: which kind of mask it holds without anybody having to infer it from vertex counts.
    source: str


@dataclass(frozen=True)
class CrossedRegion:
    """One invasive region, in both slides' coordinates."""

    roi_id: str
    index: int
    cells: int
    area_mm2: float
    source: str
    #: Rings in H&E level-0 pixels, as step 11 traced them (or step 9, on the tile path).
    he_rings: list[list[list[float]]]
    #: The same rings warped into IHC level-0 pixels.
    ihc_rings: list[list[list[float]]]


def selected_regions(refined: list[dict[str, Any]]) -> list[Crossable]:
    """Step 11's refined foci, in the shape this module warps.

    Takes `roi_refinement_service.regions()` output verbatim - one entry per focus, each
    already a polygon-with-holes in H&E level-0 pixels - and does nothing to it but wrap
    it. No selection, no ranking, no coverage rule: those decisions were made on step 10
    by a person and on step 11 by the network, and re-making either here would mean this
    step could quietly carry something other than what was approved.

    `cells` is zero and stays zero. It counted windows of step 8's grid, and a refined
    focus is not made of windows; reporting a window count for it would be inventing a
    number in the units of the thing this step stopped using.
    """
    return [
        Crossable(
            roi_id=str(entry.get("roiId") or f"focus-{position}"),
            index=int(entry.get("index") or 0),
            cells=0,
            area_mm2=float(entry.get("areaMm2") or 0.0),
            rings=[[[float(x), float(y)] for x, y in ring] for ring in entry["rings"]],
            source="beetle_pixel",
        )
        for position, entry in enumerate(refined)
    ]


def crossable(regions: list[ClassRegion]) -> list[Crossable]:
    """The tile path's regions in the same shape. Only reachable without step 11."""
    return [
        Crossable(
            roi_id=f"ROI-{region.index + 1:03d}",
            index=region.index,
            cells=region.cells,
            area_mm2=region.area_mm2,
            rings=[[[float(x), float(y)] for x, y in ring] for ring in region.rings],
            source="tile_roi",
        )
        for region in regions
    ]


def invasive_regions(
    regions: dict[int, list[ClassRegion]], limit: int | None = None
) -> list[ClassRegion]:
    """Enough of the largest invasive regions to cover most of the tumour.

    **Selected by area covered, not by a fixed count.** OncoStem's procedure is
    that the entire slide is scanned and every field averaged (SOP 4.2), so a
    fixed "top three" is a decision about how much tumour to ignore, taken
    without reference to how much tumour there is. On CAN_00270 the three
    largest regions are 26.2 mm2 of 41.4 - better than a third of the invasive
    carcinoma went unread, and the report said only that three regions were
    carried.

    So regions are taken in step 9's ranking order until
    `ihc_alignment_area_coverage` of the invasive area is reached, subject to
    two limits that are about what a region can usefully contribute rather than
    about cost:

    * regions below `ihc_alignment_min_region_mm2` are never taken - at the
      sampling field's own size such a region holds too few fields for its
      percentage to mean anything;
    * no more than `ihc_alignment_region_count` are taken, because every ring is
      warped, drawn and stored, and the tail of the list is speckle.

    `limit` overrides the cap for a caller that knows what it wants - the
    coverage rule still applies underneath it.
    """
    ranked = list(regions.get(SCORED, []))
    if not ranked:
        return []

    cap = settings.ihc_alignment_region_count if limit is None else limit
    total = sum(region.area_mm2 for region in ranked)
    wanted = total * settings.ihc_alignment_area_coverage

    chosen: list[ClassRegion] = []
    covered = 0.0
    for region in ranked:
        if len(chosen) >= cap:
            break
        if region.area_mm2 < settings.ihc_alignment_min_region_mm2 and chosen:
            # `and chosen`: a slide whose every region is small still gets its
            # largest one. Returning nothing here would refuse to score a case
            # for having a finely dispersed tumour, which is a property of the
            # disease rather than a reason not to measure it.
            break
        chosen.append(region)
        covered += region.area_mm2
        if covered >= wanted:
            break
    return chosen


def coverage(
    regions: dict[int, list[ClassRegion]], selected: list[ClassRegion]
) -> float:
    """Share of the invasive area the selection actually reaches.

    Reported rather than assumed: the coverage setting is a target, and the cap
    and the minimum size can both stop the selection short of it. A run that
    reached 0.62 when it was asked for 0.90 has measured a different tumour from
    the one the number implies.
    """
    total = sum(region.area_mm2 for region in regions.get(SCORED, []))
    if total <= 0:
        return 0.0
    return sum(region.area_mm2 for region in selected) / total


def tile_regions(candidates: list) -> list[Crossable]:
    """Step 10's selected tile candidates, in the shape this module warps.

    **The coarse staircase, deliberately, and only when asked for by name.** Step 12
    normally refuses to carry these - see `ihc_alignment_service._crossable` - because a
    tile box and a pixel-traced focus are different denominators and nothing downstream
    could tell which one it had been handed.

    This exists so the two can be *compared*: scoring a cohort both ways is the only way
    to find out what BEETLE's refinement is worth in score units rather than in area. The
    protection against silent substitution is that `source` travels with every region and
    is recorded on the report, so a number produced this way can always be told apart
    from one produced the other way.

    `cells` is the real window count here, unlike the refined path where it is zero,
    because a tile candidate genuinely is made of step 8's windows.
    """
    return [
        Crossable(
            roi_id=str(getattr(one, "roi_id", None) or f"tile-{position}"),
            index=int(getattr(one, "index", position) or 0),
            cells=int(getattr(one, "cells", 0) or 0),
            area_mm2=float(getattr(one, "area_mm2", 0.0) or 0.0),
            rings=[[[float(x), float(y)] for x, y in ring] for ring in one.rings],
            source="step9_tile",
        )
        for position, one in enumerate(candidates)
    ]


def densified_rings(
    selected: list[Crossable], *, he_mpp: float | None
) -> tuple[list[list[list[float]]], list[int]]:
    """Every ring of every selected region, subdivided and flattened for the warp.

    Returns the flat list of rings plus how many rings each region contributed,
    so the warped rings can be put back with the region they came from - the
    worker sees an anonymous list of rings and gives one back in the same order.
    """
    spacing_px = um_to_px(settings.registration_vertex_spacing_um, he_mpp)

    flat: list[list[list[float]]] = []
    counts: list[int] = []
    for region in selected:
        counts.append(len(region.rings))
        for ring in region.rings:
            flat.append(densify_ring(ring, spacing_px))
    return flat, counts


def regroup(
    selected: list[Crossable],
    warped: list[list[list[float]]],
    counts: list[int],
    *,
    ihc_mpp: float | None,
) -> list[CrossedRegion]:
    """Put the warped rings back with their regions, tidied for drawing."""
    # Half a micron: far below anything visible on a panel of a 127,000 px
    # slide, so simplification cannot move a border a viewer is judging.
    tolerance_px = um_to_px(0.5, ihc_mpp)

    crossed: list[CrossedRegion] = []
    cursor = 0
    for region, ring_count in zip(selected, counts, strict=True):
        rings = warped[cursor : cursor + ring_count]
        cursor += ring_count
        crossed.append(
            CrossedRegion(
                roi_id=region.roi_id,
                index=region.index,
                cells=region.cells,
                area_mm2=region.area_mm2,
                source=region.source,
                he_rings=[[[float(x), float(y)] for x, y in ring] for ring in region.rings],
                ihc_rings=[simplify_ring(ring, tolerance_px) for ring in rings],
            )
        )
    return crossed


__all__ = [
    "Crossable",
    "CrossedRegion",
    "coverage",
    "crossable",
    "densified_rings",
    "invasive_regions",
    "regroup",
    "selected_regions",
]
