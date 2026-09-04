"""Gate G2 from the command line: export haematoxylin tiles, then check them.

    python scripts/02_export.py --limit 4     # smoke run on four regions first
    python scripts/02_export.py               # the real export, ~1 hour
    python scripts/02_export.py --include-dcis   # ...plus the borrowed in-situ class

`--limit` is not a convenience. The export is the one step every later stage trusts
without re-checking, so it is worth running on four regions and looking at the numbers
before committing an hour to 151 of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

import backend_path  # noqa: F401
import bcss
import datasets
import download_bcss
import export

Image.MAX_IMAGE_PIXELS = 500_000_000


def reuse_bcss_rows(
    spec: export.TileSpec,
) -> tuple[list[dict], dict, list[dict], dict[str, float]]:
    """The BCSS half of an existing export, re-read instead of recomputed.

    Re-cutting 151 regions costs an hour and produces byte-identical tiles, so when the
    only thing being added is the borrowed DCIS set there is nothing to gain from it.

    What makes reuse safe is checked rather than assumed: the geometry recorded in
    `export_summary.json` must match the geometry being asked for now. A manifest cut at
    a different tile size or resolution cannot be concatenated with one cut at this one -
    the rows would look fine and the tiles would be different sizes, which surfaces as a
    tensor shape error deep in the feature extractor if you are lucky and as a silently
    mixed dataset if you are not.
    """
    manifest_path = backend_path.TILES_DIR / "tiles_manifest.csv"
    summary_path = backend_path.TILES_DIR / "export_summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        raise SystemExit(
            "--reuse-bcss needs an existing export to reuse, and there is no "
            f"{manifest_path.name} yet. Run the full export once first."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    recorded = summary.get("spec", {})
    for field, want in (("tile_px", spec.tile_px), ("mpp", spec.mpp)):
        got = recorded.get(field)
        if got is not None and got != want:
            raise SystemExit(
                f"the export on disk was cut at {field}={got} and you are asking for "
                f"{want}. Tiles of two geometries cannot share a manifest; re-run "
                "without --reuse-bcss."
            )

    rows = [
        r for r in datasets.read_manifest(manifest_path)
        if r.get("source", "bcss") != bcss.DCIS_SOURCE
    ]
    if not rows:
        raise SystemExit("the existing manifest has no BCSS rows to reuse.")

    per_region = [
        entry for entry in summary.get("per_region", [])
        if entry.get("institution") != bcss.DCIS_INSTITUTION
    ]
    # Carried forward, not recomputed: the determinism gate re-exports one region and
    # needs that region's own measured resolution. Dropping it here would leave a
    # manifest that looks complete and fails G2c with a KeyError.
    measured = summary.get("source_mpp_measured", {})
    if not measured:
        # An empty dict here is worse than a missing key: it survives the reuse, gets
        # written back into the new summary, and fails the determinism gate with a
        # KeyError naming a slide - which reads like a data problem rather than a
        # provenance one. Refuse now, while the cause is still visible.
        raise SystemExit(
            "the export on disk records no per-slide resolutions, so its BCSS half "
            "cannot be reused: the determinism gate needs them. This usually means the "
            "summary was written by an earlier --reuse-bcss run that had none either. "
            "Re-run the full export once without --reuse-bcss."
        )
    print(f"== reusing {len(rows):,} BCSS tiles from the export on disk "
          f"({len(per_region)} regions) ==")
    print("   the tiles themselves are untouched; only the DCIS half is being cut")
    return rows, dict(summary.get("dropped", {})), per_region, measured


def run_export(spec: export.TileSpec, *, limit: int | None, include_dcis: bool,
               reuse_bcss: bool = False) -> None:
    if reuse_bcss:
        rows, drop_counts, per_region, measured = reuse_bcss_rows(spec)
        drops: Counter[str] = Counter(drop_counts)
        dcis_summary = None
        if include_dcis:
            dcis_rows, dcis_drops, dcis_per_region, dcis_summary = export_dcis(spec)
            rows.extend(dcis_rows)
            drops.update(dcis_drops)
            per_region.extend(dcis_per_region)
        manifest = export.write_manifest(
            rows, dict(drops), backend_path.TILES_DIR, spec=spec,
            extra={"per_region": per_region,
                   "regions_exported": len(per_region),
                   "source_mpp_measured": measured,
                   "bcss_reused": True,
                   "dcis": dcis_summary},
        )
        print(f"\n  manifest: {manifest}")
        print(f"  {len(rows):,} tiles total")
        return

    regions = bcss.find_regions(backend_path.BCSS_DIR)
    if limit:
        # Spread the sample across the alphabet rather than taking the first N: the
        # first few regions of BCSS are all from one institution, and a smoke run on
        # one institution's staining says nothing about the rest.
        picks = np.linspace(0, len(regions) - 1, limit).round().astype(int)
        regions = [regions[int(i)] for i in dict.fromkeys(picks.tolist())]

    # The measured resolution per slide, not the 0.2500 every filename claims. See
    # download_bcss.slide_mpp: the first slide is 0.2521 and TCGA scans vary, so a
    # fixed factor here would put every tile at the wrong physical size - and the
    # tile size is only defensible as a claim about microns.
    measured = download_bcss.resolutions(backend_path.BCSS_DIR)
    if not measured:
        raise SystemExit(
            "no slide_mpp.json beside the data. Run scripts/01_download.py --route "
            "girder, which records each slide's own resolution as it downloads. "
            "Exporting without it would mean assuming 0.25 um/px, which is wrong."
        )
    unknown = sorted({r.slide_key for r in regions} - set(measured))
    if unknown:
        raise SystemExit(
            f"{len(unknown)} regions have no recorded resolution, e.g. {unknown[:3]}. "
            "Finish the download rather than guessing at their scale."
        )

    spread = sorted({round(measured[r.slide_key], 4) for r in regions})
    print(f"== export: {len(regions)} regions at {spec.tile_px}px / {spec.mpp} um/px "
          f"({spec.tile_um:.0f} um field) ==")
    print(f"  source resolutions actually measured: {spread}")

    rows: list[dict] = []
    drops: Counter[str] = Counter()
    per_region: list[dict] = []

    for index, region in enumerate(regions, start=1):
        region_rows, region_drops = export.write_region(
            region, backend_path.TILES_DIR, spec=spec,
            source_mpp=measured[region.slide_key],
        )
        rows.extend(region_rows)
        drops.update(region_drops)

        labels = [int(row["label"]) for row in region_rows]
        per_region.append({
            "roi_id": region.roi_id,
            "slide_id": region.slide_id,
            "institution": region.institution,
            "kept": len(region_rows),
            **region_drops,
            # The region's own invasive share, for the tumour-content stratification.
            # A property of the tissue, read off the labels, never off a prediction.
            "invasive_share": (
                float(np.mean([label == bcss.INVASIVE for label in labels]))
                if labels else 0.0
            ),
        })

        print(f"  [{index:>3}/{len(regions)}] {region.roi_id[:34]:<34} "
              f"{measured[region.slide_key]:.4f}um  kept {len(region_rows):>4}  "
              f"drops {dict(region_drops)}  total {len(rows):,}", flush=True)

    dcis_summary = None
    if include_dcis:
        dcis_rows, dcis_drops, dcis_per_region, dcis_summary = export_dcis(spec)
        rows.extend(dcis_rows)
        drops.update(dcis_drops)
        per_region.extend(dcis_per_region)

    manifest = export.write_manifest(
        rows, dict(drops), backend_path.TILES_DIR, spec=spec,
        extra={"per_region": per_region, "regions_exported": len(regions),
               "source_mpp_measured": {r.slide_key: measured[r.slide_key]
                                       for r in regions},
               "dcis": dcis_summary},
    )
    print(f"\n  manifest: {manifest}")
    print(f"  kept {len(rows):,} tiles, dropped {sum(drops.values()):,} {dict(drops)}")


def export_dcis(spec: export.TileSpec) -> tuple[list[dict], Counter, list[dict], dict]:
    """The borrowed in-situ regions, through the same exporter as everything else.

    Three things differ from the BCSS pass and nothing else does:

      * the resolution comes from `dcis_manifest.json` rather than from a per-slide
        measurement, because these regions were resampled to the teacher's spacing when
        they were written and carry one resolution between them. Read rather than
        assumed, for the same reason `download_bcss.resolutions` exists: a wrong
        `source_mpp` puts every tile at the wrong physical size and nothing raises.
      * `source="bracs_dcis"` is stamped on every row. That column is the only thing
        downstream that can tell a human-drawn label from a model's guess.
      * the regions are found by `bcss.find_dcis_regions`, which groups by BRACS patient
        instead of parsing a TCGA barcode.
    """
    root = backend_path.DCIS_DIR
    sidecar_path = root / "dcis_manifest.json"
    if not sidecar_path.exists():
        raise SystemExit(
            f"--include-dcis was given but {sidecar_path} does not exist. Run\n"
            "  bracs_roi_to_mask_using_beetle/scripts/export_to_approach1.py --per-case 4\n"
            "which segments BRACS regions with BEETLE and writes the pairs."
        )

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    source_mpp = float(sidecar["source_mpp"])
    regions = bcss.find_dcis_regions(root)

    print(f"\n== DCIS: {len(regions)} borrowed regions at {source_mpp} um/px ==")
    print(f"  labels: {sidecar['labels_are']}")
    print(f"  {sidecar.get('cases', '?')} patients, "
          f"{len(sidecar.get('rejected', []))} regions rejected before export")

    rows: list[dict] = []
    drops: Counter[str] = Counter()
    per_region: list[dict] = []

    for index, region in enumerate(regions, start=1):
        region_rows, region_drops = export.write_region(
            region, backend_path.TILES_DIR, spec=spec,
            source_mpp=source_mpp, source=bcss.DCIS_SOURCE,
        )
        rows.extend(region_rows)
        drops.update(region_drops)
        labels = [int(row["label"]) for row in region_rows]
        per_region.append({
            "roi_id": region.roi_id,
            "slide_id": region.slide_id,
            "institution": region.institution,
            "kept": len(region_rows),
            **region_drops,
            "invasive_share": (
                float(np.mean([label == bcss.INVASIVE for label in labels]))
                if labels else 0.0
            ),
        })
        if index % 20 == 0 or index == len(regions):
            print(f"  [{index:>3}/{len(regions)}] {region.roi_id[:34]:<34} "
                  f"kept {len(region_rows):>4}  total {len(rows):,}", flush=True)

    counts = Counter(int(row["label"]) for row in rows)
    print(f"  DCIS tiles: {len(rows):,} "
          f"{ {bcss.CLASS_NAMES[k]: v for k, v in sorted(counts.items())} }")

    return rows, drops, per_region, {
        "regions": len(regions),
        "cases": len({r.slide_id for r in regions}),
        "tiles": len(rows),
        "source_mpp": source_mpp,
        "per_class": {bcss.CLASS_NAMES[k]: v for k, v in sorted(counts.items())},
        "provenance": sidecar["labels_are"],
        "licence": sidecar["licence"],
    }


def gate_g2() -> None:
    manifest = backend_path.TILES_DIR / "tiles_manifest.csv"
    rows = datasets.read_manifest(manifest)
    labels = np.array([int(row["label"]) for row in rows])

    print("\n== G2: is this dataset trainable, and is it honest? ==")
    print(f"  tiles        : {len(rows):,}")
    print(f"  slides       : {len({row['slide_id'] for row in rows})}")
    print(f"  institutions : {len({row['institution'] for row in rows})}")
    for cls, name in enumerate(bcss.CLASS_NAMES):
        count = int((labels == cls).sum())
        print(f"    {name:<26} {count:>8,}  {count / len(labels):6.1%}")

    # Two different questions, and an earlier version of this gate conflated them.
    #
    #   is the mapping right?      -> do non-invasive PIXELS exist in the masks at all
    #   is the class learnable?    -> how many TILES the vote actually assigns to it
    #
    # The first is a bug if it fails. The second can be small for an honest reason:
    # in-situ carcinoma is genuinely rare in BCSS, and a tile only earns class 1 if
    # half its labelled pixels are non-invasive epithelium. Failing the run for that
    # would be refusing to look at the dataset we have.
    pixel_share = float(np.mean([float(r["non_invasive_frac"]) for r in rows]))
    tile_share = float((labels == bcss.NON_INVASIVE).mean())
    counts = {cls: int((labels == cls).sum()) for cls in range(len(bcss.CLASS_NAMES))}

    # The class-1 gate below is a statement about BCSS and only about BCSS, so it is
    # computed on the BCSS rows alone. Borrowed DCIS tiles are supposed to be numerous -
    # that is the entire reason they were borrowed - and letting them into this share
    # would turn a gate that catches a mapping fault into one that fires on success.
    bcss_rows = [r for r in rows if r.get("source", "bcss") != bcss.DCIS_SOURCE]
    dcis_rows = [r for r in rows if r.get("source") == bcss.DCIS_SOURCE]
    bcss_labels = np.array([int(r["label"]) for r in bcss_rows])
    bcss_tile_share = float((bcss_labels == bcss.NON_INVASIVE).mean()) if len(bcss_labels) else 0.0
    bcss_pixel_share = (
        float(np.mean([float(r["non_invasive_frac"]) for r in bcss_rows]))
        if bcss_rows else 0.0
    )
    if dcis_rows:
        dcis_labels = np.array([int(r["label"]) for r in dcis_rows])
        print(f"  sources      : BCSS {len(bcss_rows):,} tiles "
              f"(human-drawn) + BRACS/BEETLE {len(dcis_rows):,} tiles (model-generated)")
        for cls, name in enumerate(bcss.CLASS_NAMES):
            print(f"    {name:<26} BCSS {int((bcss_labels == cls).sum()):>7,}"
                  f"   DCIS {int((dcis_labels == cls).sum()):>7,}")

    # A smoke run cannot be held to the full dataset's thresholds, and pretending
    # otherwise is how a correct exporter gets debugged for an afternoon. The census
    # in the plan's Part 2 says class 1 is 0.129% of pixels and appears in 27 of 150
    # regions, so on four regions its expected tile count is ZERO.
    summary_path = backend_path.TILES_DIR / "export_summary.json"
    exported = int(json.loads(summary_path.read_text()).get("regions_exported", 0))
    full_run = exported >= 100

    print(f"  regions      : {exported}"
          f"{'' if full_run else '   (SMOKE RUN - absolute thresholds deferred)'}")

    # **Class 1 is recorded, never asserted.** This is the plan's rule, and it is the
    # rule because BCSS cannot supply the class: one region of 150 carries a `dcis`
    # annotation, in a training institution. Asserting it would fail a correct run.
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}, (
        f"labels outside the three classes: {sorted(set(np.unique(labels).tolist()))}"
    )

    if full_run:
        # What the plan asks for, and what should have caught the class-1 gap at
        # export time rather than by a hand-run pixel census afterwards.
        assert counts[bcss.NON_EPITHELIUM] > 1_000, (
            f"class 0 has {counts[bcss.NON_EPITHELIUM]:,} tiles, expected > 1,000 - "
            "the vote or the mapping is wrong, not the dataset"
        )
        assert counts[bcss.INVASIVE] > 1_000, (
            f"class 2 has {counts[bcss.INVASIVE]:,} tiles, expected > 1,000 - "
            "the vote or the mapping is wrong, not the dataset"
        )
        assert bcss_pixel_share > 0.0, (
            "no BCSS tile contains a single non-invasive epithelium pixel. That is a "
            "mapping fault, not a rare class - codes 20 (dcis) and 13 "
            "(normal_acinus_or_duct) should both land in class 1."
        )
        # Inverted, per the plan: a LARGE class 1 **from BCSS** is the bug, because BCSS
        # cannot supply one. 0.129% of pixels cannot become 2% of tiles. Borrowed DCIS
        # tiles are excluded from this share deliberately - see above.
        assert bcss_tile_share < 0.02, (
            f"class 1 is {bcss_tile_share:.2%} of the BCSS tiles. BCSS is 0.129% "
            "non-invasive by pixel, so this is too much - suspect the mapping put "
            "`tumor` or `stroma` into class 1."
        )
        if dcis_rows:
            # The mirror-image gate for the borrowed data. These regions are DCIS
            # regions of interest, so a class-1 share near zero means the teacher, the
            # resolution or the code mapping is wrong - the same fault as above, seen
            # from the other side.
            dcis_share = float((dcis_labels == bcss.NON_INVASIVE).mean())
            assert dcis_share > 0.20, (
                f"class 1 is only {dcis_share:.2%} of the borrowed DCIS tiles. These "
                "are regions BRACS annotates as DCIS, so this is a fault in the "
                "teacher's label codes or in the resolution, not a rare class."
            )
            print(f"  G2d OK - borrowed DCIS tiles are {dcis_share:.1%} class 1")
        print(f"  G2a OK - classes 0 and 2 asserted; BCSS class 1 recorded at "
              f"{int((bcss_labels == bcss.NON_INVASIVE).sum()):,} tiles "
              f"({bcss_tile_share:.2%} of BCSS tiles, {bcss_pixel_share:.3%} of its "
              "labelled pixels)")
    else:
        print(f"  G2a n/a - class 1 recorded at {counts[bcss.NON_INVASIVE]:,} tiles "
              f"({pixel_share:.3%} of labelled pixels). On {exported} regions this "
              "number carries no information:")
        print( "           BCSS has non-invasive epithelium in 27 of 150 regions, so a "
               "handful of")
        print( "           regions is expected to contain none. Run the full export "
               "before reading it.")

    if tile_share < 0.02:
        print()
        print(f"  ! non-invasive is only {tile_share:.2%} of tiles. The mapping is fine -")
        print( "  ! the pixels are there - so this is the dataset: in-situ carcinoma is")
        print( "  ! rare in BCSS, and a tile needs half its labelled pixels to be")
        print( "  ! non-invasive epithelium before the vote will call it that.")
        print( "  !")
        print( "  ! The consequence is specific and must be reported, not averaged away:")
        print( "  ! the invasive-vs-in-situ Dice - the one number this project is for -")
        print( "  ! will rest on few tiles and have a wide confidence interval. Read it")
        print( "  ! alongside its tile count, and see the plan's ablation A6.")
        print()

    try:
        train, test = datasets.institution_split(rows)
    except ValueError as exc:
        # On the full export this is a real failure and must stop the run. On a smoke
        # run it is expected: the held-out institutions are six of about forty, so a
        # handful of sampled regions can easily miss all six.
        if full_run:
            raise
        print(f"  G2b n/a - {exc}")
        print( "           Expected on a smoke run. Not checked until the full export.")
        return
    datasets.assert_no_leak(train, test)
    folds = datasets.slide_folds(train, folds=5)
    datasets.assert_no_leak(*[[train[i] for i in fold] for fold in folds])
    weights = datasets.class_weights(train)

    print(f"  institution split: {len(train):,} train / {len(test):,} test")
    print(f"  slide folds      : {[len(fold) for fold in folds]}")
    print(f"  class weights    : "
          f"{dict(zip(bcss.CLASS_NAMES, [round(float(w), 3) for w in weights]))}")
    print("  G2b OK - no slide crosses a split boundary")

    # Determinism: re-export one region and compare bytes. If this fails, the feature
    # cache cannot be trusted to correspond to the manifest.
    summary = json.loads((backend_path.TILES_DIR / "export_summary.json").read_text())
    spec = export.TileSpec(tile_px=summary["spec"]["tile_px"], mpp=summary["spec"]["mpp"])
    region = next(
        region for region in bcss.find_regions(backend_path.BCSS_DIR)
        if region.roi_id == summary["per_region"][0]["roi_id"]
    )
    again, _ = export.export_region(
        region, spec=spec,
        source_mpp=summary["source_mpp_measured"][region.slide_key],
    )
    _assert_reproduces(again, "BCSS")

    # The borrowed half goes through the same function with a different reader, a
    # different resolution and a mask a model wrote, so it is the half where a silent
    # difference would actually live. Checking only BCSS would leave the new path
    # unverified by the gate that exists to verify exactly this.
    if summary.get("dcis"):
        dcis_region = bcss.find_dcis_regions(backend_path.DCIS_DIR)[0]
        again_dcis, _ = export.export_region(
            dcis_region, spec=spec,
            source_mpp=float(summary["dcis"]["source_mpp"]),
            source=bcss.DCIS_SOURCE,
        )
        _assert_reproduces(again_dcis, "DCIS")
        print("  G2c OK - the exporter is deterministic on both sources\n")
    else:
        print("  G2c OK - the exporter is deterministic\n")


def _assert_reproduces(tiles, what: str) -> None:
    """Re-cut tiles must be byte-identical to what is on disk, or the cache is a lie."""
    if not tiles:
        raise AssertionError(f"{what}: the re-export produced no tiles at all")
    first = tiles[0]
    on_disk = np.asarray(Image.open(
        backend_path.TILES_DIR / bcss.CLASS_NAMES[int(first.row["label"])]
        / f"{first.row['tile_id']}.png"
    ))
    assert np.array_equal(first.stored, on_disk), (
        f"{what}: re-exporting {first.row['tile_id']} does not reproduce the tile on "
        "disk, so the feature cache cannot be trusted to match the manifest"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="export only N regions, spread across the dataset")
    parser.add_argument("--tile-px", type=int, default=224)
    parser.add_argument("--mpp", type=float, default=0.5)
    parser.add_argument("--check-only", action="store_true",
                        help="skip the export; run G2 against the manifest on disk")
    parser.add_argument("--reuse-bcss", action="store_true",
                        help="do not re-cut the BCSS regions; take their rows from the "
                             "export already on disk and only cut the DCIS half. Saves "
                             "an hour and produces identical tiles. The recorded tile "
                             "geometry is checked against the requested one first.")
    parser.add_argument("--include-dcis", action="store_true",
                        help="also export data/dcis/ - BRACS regions labelled by "
                             "BEETLE's nnU-Net. This is the only source of class 1 "
                             "this project has; BCSS supplies nine tiles in total. "
                             "The labels are MODEL-GENERATED and unreviewed, and every "
                             "row is stamped source=bracs_dcis so the report can keep "
                             "them apart from BCSS's human-drawn ones.")
    args = parser.parse_args()

    backend_path.ensure_dirs()
    if not args.check_only:
        run_export(
            export.TileSpec(tile_px=args.tile_px, mpp=args.mpp),
            limit=args.limit,
            include_dcis=args.include_dcis,
            reuse_bcss=args.reuse_bcss,
        )
    gate_g2()
    print("Next: python scripts/03_features.py --init imagenet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
