"""Gate G2 from the command line: export haematoxylin tiles, then check them.

    python scripts/02_export.py --limit 4     # smoke run on four regions first
    python scripts/02_export.py               # the real export, ~1 hour
    python scripts/02_export.py --include dcis ic normal      # ...plus BRACS
    python scripts/02_export.py --reuse-bcss --include dcis ic normal   # only BRACS

`--limit` is not a convenience. The export is the one step every later stage trusts
without re-checking, so it is worth running on four regions and looking at the numbers
before committing an hour to 151 of them.

`--include` names BRACS lesion trees written by
`tissue_label_generation/scripts/export_to_approach1.py`. There are three and
they are not interchangeable - each is admitted under a different consensus and so
teaches a different set of classes, which is `bcss.BORROWED_TREES` and, downstream,
`datasets.SOURCE_CLASSES`. Every tile cut from one is stamped with that tree's own
`source`, which is the only thing in the tile files that can tell a pathologist's
label from a model's guess.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
import tilestore

Image.MAX_IMAGE_PIXELS = 500_000_000


def reuse_bcss_rows(
    spec: export.TileSpec,
    variant: tilestore.Variant = tilestore.H_CHANNEL,
) -> tuple[list[dict], dict, list[dict], dict[str, float]]:
    """The BCSS half of an existing export, re-read instead of recomputed.

    Re-cutting 151 regions costs an hour and produces byte-identical tiles, so when the
    only thing being added is a borrowed BRACS tree there is nothing to gain from it.

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

    # The same argument one step further out: two *channels* cannot share a manifest
    # either, and this failure is nastier than the geometry one because the tiles are
    # the same size. A density map and a photograph of the same square are both
    # 224x224 uint8 arrays, so nothing downstream would notice a mixed store until the
    # accuracy came out wrong.
    stored_channel = str((summary.get("input") or {}).get("channel", "haematoxylin"))
    if stored_channel != variant.channel:
        raise SystemExit(
            f"the export on disk stores {stored_channel!r} tiles and you are asking "
            f"for {variant.channel!r}. Re-run without --reuse-bcss: the BCSS half has "
            "to be cut in the channel it is going to be trained in."
        )

    # Every borrowed source is dropped, not just `bracs_dcis`: the trees named by
    # `--include` are about to be re-cut, and any tree *not* named is being removed from
    # this export on purpose. Carrying a stale one forward would put tiles in the
    # manifest that the run being asked for did not ask for, and nothing downstream
    # could tell.
    rows = [
        r for r in datasets.read_manifest(manifest_path)
        if not bcss.is_borrowed(r.get("source", "bcss"))
    ]
    if not rows:
        raise SystemExit("the existing manifest has no BCSS rows to reuse.")

    per_region = [
        entry for entry in summary.get("per_region", [])
        if entry.get("institution") != bcss.BRACS_INSTITUTION
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
    print("   the tiles themselves are untouched; only the borrowed trees are cut")
    return rows, dict(summary.get("dropped", {})), per_region, measured


def run_export(spec: export.TileSpec, *, limit: int | None,
               include: tuple[str, ...], reuse_bcss: bool = False,
               variant: tilestore.Variant = tilestore.H_CHANNEL) -> None:
    if reuse_bcss:
        rows, drop_counts, per_region, measured = reuse_bcss_rows(spec, variant)
        drops: Counter[str] = Counter(drop_counts)
        borrowed: dict[str, dict] = {}
        for name in include:
            tree_rows, tree_drops, tree_per_region, tree_summary = export_borrowed(
                spec, bcss.BORROWED_TREES[name], variant=variant
            )
            rows.extend(tree_rows)
            drops.update(tree_drops)
            per_region.extend(tree_per_region)
            borrowed[name] = tree_summary
        manifest = export.write_manifest(
            rows, dict(drops), backend_path.TILES_DIR, spec=spec, variant=variant,
            extra={"per_region": per_region,
                   "regions_exported": len(per_region),
                   "source_mpp_measured": measured,
                   "bcss_reused": True,
                   "borrowed": borrowed},
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
            source_mpp=measured[region.slide_key], variant=variant,
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

    borrowed: dict[str, dict] = {}
    for name in include:
        tree_rows, tree_drops, tree_per_region, tree_summary = export_borrowed(
            spec, bcss.BORROWED_TREES[name], variant=variant
        )
        rows.extend(tree_rows)
        drops.update(tree_drops)
        per_region.extend(tree_per_region)
        borrowed[name] = tree_summary

    manifest = export.write_manifest(
        rows, dict(drops), backend_path.TILES_DIR, spec=spec, variant=variant,
        extra={"per_region": per_region, "regions_exported": len(regions),
               "source_mpp_measured": {r.slide_key: measured[r.slide_key]
                                       for r in regions},
               "borrowed": borrowed},
    )
    print(f"\n  manifest: {manifest}")
    print(f"  kept {len(rows):,} tiles, dropped {sum(drops.values()):,} {dict(drops)}")


def export_borrowed(
    spec: export.TileSpec, tree: bcss.BorrowedTree,
    *, variant: tilestore.Variant = tilestore.H_CHANNEL,
) -> tuple[list[dict], Counter, list[dict], dict]:
    """One borrowed BRACS tree, through the same exporter as everything else.

    Three things differ from the BCSS pass and nothing else does:

      * the resolution comes from `<name>_manifest.json` rather than from a per-slide
        measurement, because these regions were resampled to the teacher's spacing when
        they were written and carry one resolution between them. Read rather than
        assumed, for the same reason `download_bcss.resolutions` exists: a wrong
        `source_mpp` puts every tile at the wrong physical size and nothing raises.
      * `tree.source` is stamped on every row. That column is the only thing downstream
        that can tell a human-drawn label from a model's guess, and the only thing that
        can tell one lesion consensus from another.
      * the regions are found by `bcss.find_borrowed_regions`, which groups by BRACS
        patient instead of parsing a TCGA barcode.

    **`teaches_classes` is checked, not trusted or ignored.** The producing script
    records which classes the region's consensus corroborates, and `bcss.BORROWED_TREES`
    states the same thing on this side because `datasets.by_source_authority` needs it
    without reading a manifest. Two statements of one fact drift, so the two are
    compared here and a mismatch stops the export - the alternative is a training set
    whose tiles vote on a class their consensus never covered, which no later check
    would see.
    """
    root = backend_path.borrowed_dir(tree.name)
    sidecar_path = root / f"{tree.name}_manifest.json"
    if not sidecar_path.exists():
        raise SystemExit(
            f"--include {tree.name} was given but {sidecar_path} does not exist. Run\n"
            f"  tissue_label_generation/scripts/export_to_approach1.py "
            f"--roi-type {tree.name}\n"
            "which segments BRACS regions with BEETLE and writes the pairs."
        )

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    source_mpp = float(sidecar["source_mpp"])
    regions = bcss.find_borrowed_regions(root)

    # Absent on the `dcis` manifest, which predates the field: that tree was written
    # when it was the only one and there was nothing to distinguish it from. Missing is
    # therefore "not recorded" and falls through to the table; present and *different*
    # is a fault, and the difference is the whole message.
    recorded = sidecar.get("teaches_classes")
    if recorded is not None and frozenset(recorded) != tree.teaches:
        raise SystemExit(
            f"{sidecar_path.name} says these regions corroborate classes "
            f"{sorted(recorded)}, and bcss.BORROWED_TREES[{tree.name!r}].teaches says "
            f"{sorted(tree.teaches)}. One of the two is wrong and the export cannot "
            "choose: a tile admitted to a class its region's consensus never covered "
            "is exactly what by_source_authority exists to prevent."
        )
    if sidecar.get("source") not in (None, tree.source):
        raise SystemExit(
            f"{sidecar_path.name} was written with source={sidecar['source']!r} but "
            f"this tree stamps {tree.source!r}. The directory and the manifest in it "
            "disagree about which lesion these regions are."
        )

    print(f"\n== {tree.name}: {len(regions)} borrowed {tree.lesion} regions "
          f"at {source_mpp} um/px ==")
    print(f"  teaches: {sorted(tree.teaches)} "
          f"({', '.join(bcss.CLASS_NAMES[c] for c in sorted(tree.teaches))})"
          + ("" if recorded is not None else "   [not recorded in the manifest]"))
    print(f"  labels: {sidecar['labels_are']}")
    print(f"  {sidecar.get('cases', '?')} patients, "
          f"{len(sidecar.get('rejected', []))} regions rejected before export")

    rows: list[dict] = []
    drops: Counter[str] = Counter()
    per_region: list[dict] = []

    for index, region in enumerate(regions, start=1):
        region_rows, region_drops = export.write_region(
            region, backend_path.TILES_DIR, spec=spec,
            source_mpp=source_mpp, source=tree.source, variant=variant,
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
    print(f"  {tree.name} tiles: {len(rows):,} "
          f"{ {bcss.CLASS_NAMES[k]: v for k, v in sorted(counts.items())} }")

    return rows, drops, per_region, {
        "source": tree.source,
        "regions": len(regions),
        "cases": len({r.slide_id for r in regions}),
        "tiles": len(rows),
        "source_mpp": source_mpp,
        # Recorded on the tile side too, so a manifest read years from now says which
        # classes these rows were admitted to vote on without needing the tree table.
        "teaches_classes": sorted(tree.teaches),
        "per_class": {bcss.CLASS_NAMES[k]: v for k, v in sorted(counts.items())},
        "provenance": sidecar["labels_are"],
        "licence": sidecar["licence"],
    }



def gate_g2(variant: tilestore.Variant = tilestore.H_CHANNEL) -> None:
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
    bcss_rows = [r for r in rows if not bcss.is_borrowed(r.get("source", "bcss"))]
    borrowed_rows = {
        name: [r for r in rows if r.get("source") == tree.source]
        for name, tree in bcss.BORROWED_TREES.items()
    }
    borrowed_rows = {name: rs for name, rs in borrowed_rows.items() if rs}
    bcss_labels = np.array([int(r["label"]) for r in bcss_rows])
    bcss_tile_share = float((bcss_labels == bcss.NON_INVASIVE).mean()) if len(bcss_labels) else 0.0
    bcss_pixel_share = (
        float(np.mean([float(r["non_invasive_frac"]) for r in bcss_rows]))
        if bcss_rows else 0.0
    )
    borrowed_labels = {
        name: np.array([int(r["label"]) for r in rs])
        for name, rs in borrowed_rows.items()
    }
    if borrowed_rows:
        total_borrowed = sum(len(rs) for rs in borrowed_rows.values())
        print(f"  sources      : BCSS {len(bcss_rows):,} tiles (human-drawn) + "
              f"BRACS/BEETLE {total_borrowed:,} tiles (model-generated)")
        header = "".join(f"{name.upper():>10}" for name in borrowed_rows)
        print(f"    {'':<26} {'BCSS':>9}{header}")
        for cls, name in enumerate(bcss.CLASS_NAMES):
            counts_row = "".join(
                f"{int((labels == cls).sum()):>10,}"
                for labels in borrowed_labels.values()
            )
            print(f"    {name:<26} {int((bcss_labels == cls).sum()):>9,}{counts_row}")
        # The confound `06_leakage_check.py` measures, printed where it is created and
        # in the units it is created in. With `dcis` alone, class 1 was 1,843 BRACS
        # tiles against BCSS's 9 - 99.5% - so `source` was nearly a synonym for the
        # class and a head could score on in-situ by recognising the dataset. `ic` and
        # `normal` exist to move these three numbers apart from each other; a run where
        # they are all similar is a run where `source` predicts nothing about `label`.
        # Broken out by *laboratory*, not by "borrowed", because that is the question.
        # A line reading "99.8% borrowed" was true before BACH and is still true after
        # it, while the thing that actually changed - who those borrowed tiles came
        # from - moved from one laboratory to two. Lumping them would hide the only
        # number this whole exercise was trying to move.
        print("  source of each class (the source/label confound, per class):")
        print(f"    {'':<26}{'BCSS':>8}{'BRACS':>9}{'BACH':>9}")
        for cls, name in enumerate(bcss.CLASS_NAMES):
            per_lab = {"BRACS": 0, "BACH": 0}
            for tree_name, labels in borrowed_labels.items():
                lab = "BACH" if tree_name.startswith("bach") else "BRACS"
                per_lab[lab] += int((labels == cls).sum())
            from_bcss = int((bcss_labels == cls).sum())
            in_class = from_bcss + sum(per_lab.values())
            if in_class:
                print(f"    {name:<26}{from_bcss / in_class:>7.1%}"
                      f"{per_lab['BRACS'] / in_class:>9.1%}"
                      f"{per_lab['BACH'] / in_class:>9.1%}")

    # A smoke run cannot be held to the full dataset's thresholds, and pretending
    # otherwise is how a correct exporter gets debugged for an afternoon. The census
    # in the plan's Part 2 says class 1 is 0.129% of pixels and appears in 27 of 150
    # regions, so on four regions its expected tile count is ZERO.
    summary_path = backend_path.TILES_DIR / "export_summary.json"
    summary = json.loads(summary_path.read_text())
    exported = int(summary.get("regions_exported", 0))
    full_run = exported >= 100

    # **The absolute thresholds below are a statement about tissue, but they are
    # counted in tiles - so they have to be scaled by the field of view, or they stop
    # meaning what they say.** A 224 px tile at 1.0 um/px covers four times the tissue
    # of one at 0.5, and at 2.0 um/px sixteen times, so the same slides yield roughly a
    # quarter and a sixteenth of the tiles. "Class 2 has fewer than 1,000 tiles" is a
    # mapping fault at 112 um and simple arithmetic at 448 um; asserting the flat
    # number would fail a correct 448 um export and teach whoever hit it to delete the
    # gate. Scaled, the check keeps its intent - "far less than this dataset can
    # possibly yield means the vote or the mapping is broken" - at every geometry.
    mpp = float(summary.get("spec", {}).get("mpp", export.DEFAULT_SPEC.mpp))
    tile_px = int(summary.get("spec", {}).get("tile_px", export.DEFAULT_SPEC.tile_px))
    yield_ratio = (export.DEFAULT_SPEC.mpp / mpp) ** 2 * (
        export.DEFAULT_SPEC.tile_px / tile_px
    ) ** 2
    floor = max(1, int(1_000 * yield_ratio))

    print(f"  regions      : {exported}"
          f"{'' if full_run else '   (SMOKE RUN - absolute thresholds deferred)'}")
    print(f"  field of view: {tile_px} px at {mpp} um/px = {tile_px * mpp:.0f} um"
          f"   (class 0/2 floor scaled to {floor:,} tiles)")

    # **Class 1 is recorded, never asserted.** This is the plan's rule, and it is the
    # rule because BCSS cannot supply the class: one region of 150 carries a `dcis`
    # annotation, in a training institution. Asserting it would fail a correct run.
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}, (
        f"labels outside the three classes: {sorted(set(np.unique(labels).tolist()))}"
    )

    if full_run:
        # What the plan asks for, and what should have caught the class-1 gap at
        # export time rather than by a hand-run pixel census afterwards.
        assert counts[bcss.NON_EPITHELIUM] > floor, (
            f"class 0 has {counts[bcss.NON_EPITHELIUM]:,} tiles, expected > {floor:,} "
            f"at a {tile_px * mpp:.0f} um field of view - the vote or the mapping is "
            "wrong, not the dataset"
        )
        assert counts[bcss.INVASIVE] > floor, (
            f"class 2 has {counts[bcss.INVASIVE]:,} tiles, expected > {floor:,} "
            f"at a {tile_px * mpp:.0f} um field of view - the vote or the mapping is "
            "wrong, not the dataset"
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
        for name, labels in borrowed_labels.items():
            # The mirror-image gate for the borrowed data, one per tree. These are
            # regions BRACS annotates as this lesion by three-pathologist consensus, so
            # a corroborated share near zero means the teacher, the resolution or the
            # code mapping is wrong - the same fault as above, seen from the other side.
            #
            # Each tree needs its own floor because each is a different claim about the
            # tissue: an in-situ share for `dcis`, an invasive share for `ic`, and for
            # `normal` the two non-carcinoma classes together, since a consensus of "no
            # carcinoma here" says nothing about how the rest divides between stroma
            # and duct. `bcss.BORROWED_TREES` carries the measured numbers behind each.
            tree = bcss.BORROWED_TREES[name]
            share = float(np.isin(labels, sorted(tree.gate_classes)).mean())
            assert share > tree.gate_min_share, (
                f"{tree.gate_reads} is only {share:.2%} of the borrowed {name} tiles, "
                f"and the floor is {tree.gate_min_share:.0%}. These are regions BRACS "
                f"annotates as {tree.lesion}, so this is a fault in the teacher's "
                "label codes or in the resolution, not a rare class."
            )
            print(f"  G2d OK - borrowed {name} tiles are {share:.1%} "
                  f"{tree.gate_reads}")
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
    # The determinism re-cut has to use the variant the tiles on disk were written
    # with, or G2c compares a photograph against a density map and fails for the
    # wrong reason. The summary records it, so read it rather than trusting the
    # caller's flag - a `--check-only` run against someone else's export is exactly
    # the case where the two disagree.
    recorded = str((summary.get("input") or {}).get("channel", variant.channel))
    variant = tilestore.VARIANTS.get(recorded, variant)
    print(f"  G2c re-cutting as {variant.channel} ({variant.stored_as})")
    # A region that actually produced tiles, not simply the first one exported.
    #
    # **This is a real constraint at the larger fields of view, not a defensive
    # nicety.** A 672 um window needs 1344 source pixels, so at 3.0 um/px only 189 of
    # 1,104 BCSS regions yield a single tile - and the first region in the summary
    # yields none. Re-cutting that one and asserting it reproduces "the tile on disk"
    # asks whether zero tiles equal zero tiles, gets an empty list, and fails with
    # "the re-export produced no tiles at all" - a determinism failure reported for a
    # region whose determinism was never in question. The gate wants a region with
    # something to compare.
    def first_productive(entries: list[dict]) -> dict | None:
        return next((entry for entry in entries if int(entry.get("kept", 0)) > 0), None)

    bcss_entries = [
        entry for entry in summary["per_region"]
        if entry.get("institution") != bcss.BRACS_INSTITUTION
    ]
    productive = first_productive(bcss_entries)
    if productive is None:
        print("  G2c SKIP for BCSS - no BCSS region yielded a tile at this field of "
              "view, so there is nothing to re-cut")
    else:
        region = next(
            region for region in bcss.find_regions(backend_path.BCSS_DIR)
            if region.roi_id == productive["roi_id"]
        )
        again, _ = export.export_region(
            region, spec=spec,
            source_mpp=summary["source_mpp_measured"][region.slide_key],
            variant=variant,
        )
        _assert_reproduces(again, "BCSS")

    # The borrowed half goes through the same function with a different reader, a
    # different resolution and a mask a model wrote, so it is the half where a silent
    # difference would actually live. Checking only BCSS would leave the new path
    # unverified by the gate that exists to verify exactly this.
    checked = datasets.borrowed_summaries(summary)
    verified: list[str] = []
    for name, tree_summary in checked.items():
        tree = bcss.BORROWED_TREES[name]
        regions = bcss.find_borrowed_regions(backend_path.borrowed_dir(name))

        # Same selection rule as BCSS above, and it bites harder here: most `ic`
        # regions are under 900 px in both axes, so at 448 um and beyond that whole
        # tree contributes nothing and its first region certainly does not.
        kept_ids = {
            str(entry["roi_id"]) for entry in summary["per_region"]
            if int(entry.get("kept", 0)) > 0
        }
        region = next((r for r in regions if r.roi_id in kept_ids), None)
        if region is None:
            print(f"  G2c SKIP for {name} - no region in this tree yielded a tile at "
                  "this field of view")
            continue

        again_borrowed, _ = export.export_region(
            region, spec=spec,
            source_mpp=float(tree_summary["source_mpp"]),
            source=tree.source,
            variant=variant,
        )
        _assert_reproduces(again_borrowed, name)
        verified.append(name)
    checked = {name: checked[name] for name in verified}
    if checked:
        print(f"  G2c OK - the exporter is deterministic on BCSS and on "
              f"{', '.join(checked)}\n")
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
    parser.add_argument("--channel", default="haematoxylin",
                        choices=["haematoxylin", "he"],
                        help="what the stored tiles are. `haematoxylin` deconvolves "
                             "and stores Ruifrok's H channel as one uint8 plane - the "
                             "input every checkpoint before the H&E branch was fitted "
                             "on. `he` stores the resampled sRGB itself, for a model "
                             "that is shown the photograph. The kept tiles and their "
                             "labels are IDENTICAL either way, because the vote reads "
                             "the mask and never the pixels - which is what makes the "
                             "two stores comparable tile by tile.")
    parser.add_argument("--check-only", action="store_true",
                        help="skip the export; run G2 against the manifest on disk")
    parser.add_argument("--reuse-bcss", action="store_true",
                        help="do not re-cut the BCSS regions; take their rows from the "
                             "export already on disk and only cut the borrowed trees. "
                             "Saves an hour and produces identical tiles. The recorded "
                             "tile geometry is checked against the requested one first.")
    parser.add_argument("--include", nargs="*", default=[],
                        choices=sorted(bcss.BORROWED_TREES),
                        metavar="TREE",
                        help="also export these BRACS lesion trees from data/<tree>/ - "
                             "regions labelled by BEETLE's nnU-Net. `dcis` is the only "
                             "real source of class 1 this project has (BCSS supplies "
                             "nine tiles in total); `ic` and `normal` are what stop "
                             "`source` being a synonym for class 1, which is the "
                             "confound 06_leakage_check.py reports as INCONCLUSIVE. "
                             "The labels are MODEL-GENERATED and unreviewed, and each "
                             "row is stamped with its tree's own source so the report "
                             "can keep them apart from BCSS's human-drawn ones, and "
                             "from each other. Choices: "
                             f"{', '.join(sorted(bcss.BORROWED_TREES))}.")
    parser.add_argument("--include-dcis", action="store_true",
                        help="shorthand for `--include dcis`, kept because it is in "
                             "the run logs of every export before there were three "
                             "trees.")
    args = parser.parse_args()

    include = tuple(dict.fromkeys(
        list(args.include) + (["dcis"] if args.include_dcis else [])
    ))

    variant = tilestore.resolve(args.channel)

    backend_path.ensure_dirs()
    if not args.check_only:
        if not args.reuse_bcss:
            # Clear the staging tree first.
            #
            # Tile ids are `<roi>__r<row>c<col>`, and the row and column are indices
            # into *this* geometry's grid - so the same id names a different square at
            # a different field of view. A previous export's PNGs therefore sit at the
            # exact filenames this one is about to write, and any that this run does
            # not happen to overwrite stay behind wearing a name that now means
            # something else. The manifest would not list them, so nothing downstream
            # would read them - but G2c re-cuts a tile and compares it against the
            # file on disk, and that comparison is only meaningful if the file came
            # from this run. Observed exactly once, as a determinism failure on a
            # tile whose determinism was never in question.
            shutil.rmtree(backend_path.TILES_DIR, ignore_errors=True)
            backend_path.TILES_DIR.mkdir(parents=True, exist_ok=True)

        run_export(
            export.TileSpec(tile_px=args.tile_px, mpp=args.mpp),
            limit=args.limit,
            include=include,
            reuse_bcss=args.reuse_bcss,
            variant=variant,
        )
    gate_g2(variant)
    print("Next: python scripts/03_features.py --init imagenet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
