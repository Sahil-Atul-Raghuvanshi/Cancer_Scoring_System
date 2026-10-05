"""Drive one case from a folder path all the way to the aligned IHC regions.

The whole flow the demo performs, without a browser: pick a marker, resolve a
case folder, run steps 2-9 on the H&E, then step 10 to carry the invasive
regions onto that marker's IHC slide.

    python scripts/run_case_alignment.py <case-folder> <marker letter>
    python scripts/run_case_alignment.py ../../storage/data/original/oncostem_slides/CAN_00270 A

Step 8 is tens of minutes of CPU and step 10 is a few more, so this is a
background job rather than something to wait on at a prompt. Everything it
computes is cached exactly where the API would put it, so the UI picks the
results up rather than recomputing them.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services import case_service  # noqa: E402
from app.services.ihc_alignment_service import ihc_alignment_service  # noqa: E402
from app.services.roi_service import roi_service  # noqa: E402
from app.services.tissue_type_service import tissue_type_service  # noqa: E402


def _say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_path", help="folder holding the case's .svs files")
    parser.add_argument("marker", help="marker letter: A, F, R, U or W")
    parser.add_argument(
        "--restart-alignment",
        action="store_true",
        help="register again instead of reusing a cached registration",
    )
    args = parser.parse_args()

    _say(f"resolving {args.case_path}")
    resolution = case_service.resolve_case(args.case_path)
    _say(f"case {resolution.case_id}: found {sorted(resolution.found)}, missing {resolution.missing}")

    session = case_service.load_case(args.case_path, args.marker)
    _say(f"H&E  -> {session.he_upload_id}")
    _say(f"IHC  -> {session.ihc_upload_id}  ({session.marker})")

    # Steps 2-9 all run on the H&E. Step 8 is the long one; asking step 9 for
    # its report drives whatever of the chain is not already cached.
    _say("running steps 2-9 on the H&E (step 8 is the slow one)")
    started = time.monotonic()
    tissue_type_service.start(session.he_upload_id)
    tissue_type_service.execute(session.he_upload_id)
    roi = roi_service.build(session.he_upload_id)
    _say(f"step 9 done in {time.monotonic() - started:.0f}s")

    invasive = roi.class_regions.get("invasive_epithelium", [])
    _say(f"step 9 found {len(invasive)} invasive region(s)")
    if not invasive:
        _say("nothing invasive to carry across; stopping here")
        return 1

    _say("step 10: registering the two slides and warping the regions")
    started = time.monotonic()
    ihc_alignment_service.start(
        session.he_upload_id, session.ihc_upload_id, restart=args.restart_alignment
    )
    ihc_alignment_service.execute(session.he_upload_id, session.ihc_upload_id)
    report = ihc_alignment_service.report(session.he_upload_id, session.ihc_upload_id)
    _say(f"step 10 finished in {time.monotonic() - started:.0f}s with state '{report.state}'")

    print()
    print(json.dumps(report.diagnostics.model_dump(by_alias=True), indent=2))
    print()
    if report.state == "refused":
        print("REFUSED:")
        for reason in report.refusal_reasons:
            print(f"  - {reason}")
        return 2

    for region in report.regions:
        print(
            f"  region {region.rank}: {region.area_mm2:.2f} mm2, "
            f"{len(region.ihc_rings[0])} vertices on the IHC slide"
        )
    for note in report.notes:
        print(f"  note: {note}")

    directory = ihc_alignment_service._dir(session.he_upload_id, session.ihc_upload_id)  # noqa: SLF001
    print(f"\nartefacts: {directory}")
    print("  he_borders.png / ihc_borders.png   the side-by-side comparison")
    print("  invasive1..3.png                   the regions, cropped from the IHC slide")
    print(f"\nconfirmed: {report.confirmed} (a person still has to check the panels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
