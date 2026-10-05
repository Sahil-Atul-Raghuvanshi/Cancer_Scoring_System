"""Check that step 10 moved the regions somewhere sensible.

The gate in `app/registration/gate.py` measures the *registration*. This
measures the *result*: what actually happened to the polygons. They are
different questions, and a transform can pass the first while doing something
obviously wrong to the second - collapsing a region to a sliver, inflating it
by a factor of ten, or pushing most of it off the edge of the slide.

    python scripts/check_alignment_geometry.py <he-upload-id> <ihc-upload-id>

Each region is reported with:

    area ratio      warped area over original area. A serial section is not
                    the same tissue, and a non-rigid warp genuinely changes
                    area a little - but not by a factor of two.
    displacement    how far the region's centroid moved, in mm. Large is not
                    wrong on its own: two sections are placed on their slides
                    independently, so millimetres of offset are ordinary.
    off-slide       share of the warped vertices that landed outside the IHC
                    canvas and had to be clamped.
    self-intersect  whether the warp *introduced* a ring that crosses itself.
                    Compared against the source ring rather than judged on its
                    own, because step 9's rings are rectilinear traces on a
                    tile grid and are frequently already non-simple: two
                    diagonally-adjacent cells meet at a point, which shapely
                    calls a ring self-intersection. Every region of CAN_00270
                    is like this before anything is warped, so flagging the
                    warped ring alone reports step 9's tracing as though it
                    were a registration fault.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from shapely.geometry import Polygon  # noqa: E402

from app.services.ihc_alignment_service import ihc_alignment_service  # noqa: E402

#: Area ratios outside this are flagged. Wide on purpose - the point is to
#: catch a collapse or an explosion, not to police ordinary deformation.
AREA_RATIO_LIMITS = (0.5, 2.0)


def _polygon(ring: list[list[float]]) -> Polygon | None:
    if len(ring) < 3:
        return None
    return Polygon([(x, y) for x, y in ring])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("he_upload_id")
    parser.add_argument("ihc_upload_id")
    args = parser.parse_args()

    report = ihc_alignment_service.report(args.he_upload_id, args.ihc_upload_id)
    print(f"state: {report.state}   marker: {report.marker}   regions: {len(report.regions)}")
    if report.state != "ready":
        for reason in report.refusal_reasons:
            print(f"  refused: {reason}")
        return 2

    ihc_mpp = report.diagnostics.ihc_mpp or 0.2222
    he_mpp = report.diagnostics.he_mpp or 0.2222
    problems: list[str] = []

    for region in report.regions:
        source = _polygon(region.he_rings[0])
        target = _polygon(region.ihc_rings[0])
        if source is None or target is None:
            problems.append(f"region {region.rank}: a ring has fewer than 3 vertices")
            continue

        # Areas in mm2, each on its own slide's scale - the two slides do not
        # have to share an mpp, and assuming they do is how a scale error hides.
        source_mm2 = source.area * (he_mpp / 1000) ** 2
        target_mm2 = target.area * (ihc_mpp / 1000) ** 2
        ratio = target_mm2 / source_mm2 if source_mm2 else float("inf")

        source_centre = source.centroid
        target_centre = target.centroid
        moved_mm = (
            ((target_centre.x * ihc_mpp) - (source_centre.x * he_mpp)) ** 2
            + ((target_centre.y * ihc_mpp) - (source_centre.y * he_mpp)) ** 2
        ) ** 0.5 / 1000

        vertices = len(region.ihc_rings[0])
        source_simple, target_simple = source.is_valid, target.is_valid

        print(
            f"\nregion {region.rank}: {source_mm2:.2f} mm2 -> {target_mm2:.2f} mm2  "
            f"(ratio {ratio:.2f})"
        )
        print(f"  centroid moved {moved_mm:.2f} mm")
        print(
            f"  {vertices} vertices, ring "
            + (
                "valid"
                if target_simple
                else (
                    "self-intersecting - but so was the source, so the warp did not "
                    "cause it"
                    if not source_simple
                    else "SELF-INTERSECTING, and the source was not"
                )
            )
        )

        if not AREA_RATIO_LIMITS[0] <= ratio <= AREA_RATIO_LIMITS[1]:
            problems.append(
                f"region {region.rank}: area changed by {ratio:.2f}x, outside "
                f"{AREA_RATIO_LIMITS[0]}-{AREA_RATIO_LIMITS[1]}"
            )
        # Only a *newly* broken ring is the warp's fault.
        if source_simple and not target_simple:
            problems.append(
                f"region {region.rank}: the warp turned a simple ring into a "
                f"self-intersecting one"
            )

    clamped = report.diagnostics.clamped_vertices or 0
    total = report.diagnostics.total_vertices or 0
    if total:
        print(f"\noff-slide vertices: {clamped} of {total} ({clamped / total:.1%})")
        if clamped / total > 0.1:
            problems.append(f"{clamped / total:.0%} of vertices warped off the IHC slide")

    print()
    if problems:
        print("PROBLEMS")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("GEOMETRY OK - every region kept a sane area, shape and position")
    return 0


if __name__ == "__main__":
    sys.exit(main())
