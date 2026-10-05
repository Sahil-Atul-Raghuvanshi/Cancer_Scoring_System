"""Run many regions without the UI, and write the report the UI cannot.

The app is for looking at one region and deciding whether the segmentation is
believable. This is for the other half of the job: processing enough regions that the
tile counts mean something, overnight, without a browser open.

**Sampling matters more than volume here.** 665 regions come from 60 patients, and 42 of
them come from patient 1247 alone. A run that took the first 100 files in alphabetical
order would be measuring four patients' stain, and every conclusion drawn from it about
"BRACS" would really be a conclusion about those four. `--per-case` takes a fixed number
from each patient instead, which is the sampling the tile counts need in order to be
about the dataset.

    python scripts/batch.py --per-case 2                 # 60 patients, ~120 regions
    python scripts/batch.py --per-case 1 --folds 0,1,2,3,4
    python scripts/batch.py --limit 20 --report reports/pilot.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from bracs_app import compare, config, pipeline  # noqa: E402

config.install_approach1_path()
config.ensure_dirs()


def choose(entries: list, per_case: int | None, limit: int | None, seed: int) -> list:
    """Regions to process, sampled by patient rather than by filename."""
    if per_case is None:
        chosen = entries
    else:
        import random

        rng = random.Random(seed)
        by_case: dict[str, list] = defaultdict(list)
        for entry in entries:
            by_case[entry.case_id].append(entry)
        chosen = []
        for case in sorted(by_case):
            regions = by_case[case]
            chosen.extend(rng.sample(regions, min(per_case, len(regions))))

    # Smallest first, so a run that is cut short has processed the most regions it could
    # rather than the fewest. The tile count is what is being accumulated, and a small
    # region still contributes to it.
    chosen.sort(key=lambda entry: entry.width * entry.height)
    return chosen[:limit] if limit else chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--per-case", type=int, default=None,
                        help="regions per patient; the sampling that makes counts mean something")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--folds", default="0", help="comma-separated, 0-4")
    parser.add_argument("--source-mpp", type=float, default=config.BRACS_MPP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-done", action="store_true",
                        help="leave regions that already have a manifest alone")
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).resolve().parents[1] / "reports" / "batch.json")
    args = parser.parse_args()

    folds = tuple(sorted({int(f) for f in args.folds.split(",") if f.strip() != ""}))
    if not folds or not all(0 <= f <= 4 for f in folds):
        print("folds must be between 0 and 4", file=sys.stderr)
        return 2

    entries = pipeline.list_rois(args.split)
    chosen = choose(entries, args.per_case, args.limit, args.seed)
    if args.skip_done:
        chosen = [e for e in chosen if not (config.RUNS_DIR / e.roi_id / "manifest.json").exists()]

    patches = sum(e.as_json()["patches_per_fold"] for e in chosen) * len(folds)
    print(
        f"{len(chosen)} regions from {len({e.case_id for e in chosen})} patients, "
        f"{len(folds)} fold(s), ~{patches} forward passes, "
        f"~{patches * 0.62 / 60:.0f} min estimated",
        file=sys.stderr,
    )

    manifests, failures = [], []
    started = time.time()
    for index, roi in enumerate(chosen, 1):
        prefix = f"[{index}/{len(chosen)}] {roi.roi_id}"
        try:
            result = pipeline.run_region(
                roi, folds=folds, source_mpp=args.source_mpp,
                progress=lambda message, fraction: None,
            )
            manifest = result.manifest
            manifests.append(manifest)
            counts = manifest.get("tiles", {}).get("by_class", {})
            print(
                f"{prefix}: {manifest['class_area_fraction']['non_invasive_epithelium']:.0%} in-situ, "
                f"{counts.get('non_invasive_epithelium', 0)} class-1 tiles, "
                f"{manifest['seconds']}s"
                + ("" if manifest["verdict"]["usable"] else "   <-- CHECK"),
                file=sys.stderr,
            )
        except Exception as error:  # noqa: BLE001 - one bad region must not lose the run
            failures.append({"roi_id": roi.roi_id, "error": f"{type(error).__name__}: {error}"})
            print(f"{prefix}: FAILED - {error}", file=sys.stderr)

    if not manifests:
        print("nothing completed", file=sys.stderr)
        return 1

    totals: dict[str, int] = defaultdict(int)
    for manifest in manifests:
        for name, count in manifest.get("tiles", {}).get("by_class", {}).items():
            totals[name] += count

    report = {
        "split": args.split,
        "folds": list(folds),
        "source_mpp": args.source_mpp,
        "regions": len(manifests),
        "cases": len({m["case_id"] for m in manifests}),
        "failed": failures,
        "tiles_by_class": dict(totals),
        "regions_flagged": [
            {"roi_id": m["roi_id"], "notes": m["verdict"]["notes"]}
            for m in manifests
            if not m["verdict"]["usable"]
        ],
        "wall_clock_seconds": round(time.time() - started, 1),
        "comparison": compare.compare_all([m["roi_id"] for m in manifests]),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 62}", file=sys.stderr)
    print(f"  {len(manifests)} regions from {report['cases']} patients "
          f"in {report['wall_clock_seconds'] / 60:.1f} min", file=sys.stderr)
    for name, count in sorted(totals.items()):
        print(f"  {name:<26} {count:>6} tiles", file=sys.stderr)
    headline = report["comparison"]["headline"]
    print(f"\n  class 1: BCSS has {headline['bcss_tiles']}, BRACS now has "
          f"{headline['bracs_tiles']}"
          + (f" ({headline['multiple']}x)" if headline["multiple"] else ""), file=sys.stderr)
    if report["regions_flagged"]:
        print(f"  {len(report['regions_flagged'])} region(s) flagged - see the report",
              file=sys.stderr)
    print(f"  report: {args.report}", file=sys.stderr)
    print("=" * 62, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
