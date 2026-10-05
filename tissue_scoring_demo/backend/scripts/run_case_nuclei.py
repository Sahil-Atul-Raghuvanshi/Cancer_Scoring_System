"""Segment the nuclei for one aligned pair, without a browser.

    python scripts/run_case_nuclei.py <he-upload-id> <ihc-upload-id>
    python scripts/run_case_nuclei.py <he-upload-id> <ihc-upload-id> --restart

Step 10 has to have run **and been confirmed** for the pair first - the service
refuses otherwise, and so does this. Confirm it in the UI, or:

    python -c "from app.services.ihc_alignment_service import ihc_alignment_service as s; \\
               s.confirm('<he>', '<ihc>', confirmed=True)"

That is a person's decision about whether two sections line up, so it is not
something this script will do on your behalf.

Everything it computes lands exactly where the API would put it, so the UI picks
the result up rather than recomputing it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.services.nuclei_service import NucleiError, nuclei_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("he_upload_id")
    parser.add_argument("ihc_upload_id")
    parser.add_argument(
        "--restart", action="store_true", help="segment again, ignoring any cached result"
    )
    args = parser.parse_args()

    try:
        state = nuclei_service.start(
            args.he_upload_id, args.ihc_upload_id, restart=args.restart
        )
    except NucleiError as exc:
        print(f"cannot start: {exc}")
        return 2

    if state.state == "ready" and not args.restart:
        print("already segmented; pass --restart to do it again")
    else:
        nuclei_service.execute(args.he_upload_id, args.ihc_upload_id)

    run = nuclei_service.state(args.he_upload_id, args.ihc_upload_id)
    if run.state != "ready":
        print(f"{run.state}: {run.error or run.message}")
        return 1

    report = nuclei_service.report(args.he_upload_id, args.ihc_upload_id)
    print(
        f"{report.marker or '?'}: {report.counted:,} nuclei counted over "
        f"{report.sampled_mm2:.3f} mm2 -> {report.density_per_mm2:,.0f}/mm2"
    )
    for region in report.regions:
        print(
            f"  region {region.rank}: {region.counted:>5,} counted  "
            f"{region.density_per_mm2:>7,.0f}/mm2  "
            f"{len(region.fields)} of {region.fields_available} fields "
            f"({region.sampled_share:.1%} of {region.area_mm2:.2f} mm2)"
        )
    if report.he_density_per_mm2 is not None:
        print(
            f"  H&E reference: {report.he_density_per_mm2:,.0f}/mm2"
            + (
                f"  -> this slide is {report.density_shortfall:.0%} short"
                if report.density_shortfall
                else ""
            )
        )
    for note in report.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
