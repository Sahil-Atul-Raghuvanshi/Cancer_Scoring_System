"""Build approach 1's borrowed-region trees from BRACS, one lesion type per run.

Writes exactly what approach 1's exporter consumes, in the layout it already knows:

    tissue_type_model_training/data/dcis/       --roi-type dcis  (the default)
        images/BRACS_1247_DCIS_1.png        RGB, 0.5 um/px
        masks/ BRACS_1247_DCIS_1.png        uint16 I;16, BCSS's own 22 label codes
        dcis_manifest.json                  provenance, resolution, per-region verdict

    tissue_type_model_training/data/ic/         --roi-type ic
        images/BRACS_1003667_IC_1.png       same three artefacts, same format
        masks/ BRACS_1003667_IC_1.png
        ic_manifest.json

**Why two trees rather than one.** `datasets.SOURCE_CLASSES` decides which classes a
source is allowed to teach, and it keys off the `source` column stamped on every tile
row. Pooling both lesion types into one directory would give them one source string and
one authority, which is precisely the distinction being drawn: a DCIS consensus
corroborates BEETLE's in-situ and contradicts its invasive, and an IC consensus does the
opposite. Two trees, two source strings, two rows in the table.

**What the ROI label buys, and what it does not.** BRACS labels a region by its
predominant lesion, by three-pathologist consensus - a human label, and the strongest
thing in this pipeline. It does not say every pixel is that lesion: an IC region can
contain adjacent DCIS, and a DCIS region can contain a focus of invasion.

**How that disagreement is settled, and this changed.** It used to be settled by
deletion: where BEETLE's per-pixel verdict contradicted the consensus the whole region
was dropped. That removed 33 of 153 DCIS regions and 47 of 100 BACH in-situ images, and
it removed them non-randomly - a solid or comedo duct read 112 um at a time shows no
duct wall, so the teacher calls it invasive, so the region was rejected. The filter was
selecting against precisely the morphology the student then gets wrong, and it shows:
the served checkpoint calls 22.4 % of held-out DCIS invasive.

It is now settled in the consensus's favour. `config.RoiTree.epithelium_code` writes
both of BEETLE's epithelium classes as one code - the union says where the epithelium
is, the three pathologists say which lesion it is - so a region the teacher misreads is
relabelled rather than deleted. `verdict` still blocks a region whose teacher found
almost no epithelium at all, which is a resolution or stain fault rather than a
disagreement, and records the disagreement itself as `teacher_overruled_share`.
`--split-epithelium` reinstates the old rule for reproducing an earlier export.

**Resolution.** These are written at **0.5 um/px**, not BRACS's native 0.25. That is the
teacher's spacing and it is also approach 1's tile spacing, so the region is resampled
once here and approach 1's `export.resample` becomes a no-op. Writing at 0.25 would mean
upsampling the mask - inventing label boundaries the teacher never drew - or resampling
twice. `dcis_manifest.json` records `source_mpp: 0.5` and `02_export.py` reads it rather
than assuming, which is the same discipline `download_bcss.resolutions` enforces for
BCSS's own regions.

**Re-running this after the rule change costs minutes, not a day.** Every region ever
processed has its teacher output in `data/runs/`, and the collapse is a pure relabelling
of it, so a run cached under the old rule is re-coded by `pipeline.relabel_run` instead
of being segmented again - seconds a region rather than 25. The cache check refuses to
reuse a pre-collapse run *as though* it were a new one, which is the failure that would
otherwise make this whole change a no-op with a convincing log.

    python scripts/export_to_approach1.py --per-case 2                  # the DCIS 120
    python scripts/export_to_approach1.py --roi-type ic --per-case 2    # the IC set
    python scripts/export_to_approach1.py --roi-type ic --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from bracs_app import config, pipeline  # noqa: E402

config.install_approach1_path()
config.ensure_dirs()

#: Where the region trees are written. The per-lesion subdirectory is named by the tree,
#: so `--roi-type ic` writes `regions/ic/` and nothing has to be told twice.
#:
#: **This stage's `regions/` (`v<N>_data/data/tissue_label_generation/regions/`), not approach 1's.** These pairs are
#: derived - an image resampled from `ORIGINAL_DATA` and a mask predicted by BEETLE -
#: and they are produced here, so they live here. Approach 1's `data/` holds only the
#: tile exports cut from them, one directory per field of view, which is what makes
#: "delete the exports and cut them again" a safe sentence there.
APPROACH1_DATA = config.REGIONS_DIR


def cached_manifest(
    roi,
    folds: tuple[int, ...],
    source_mpp: float,
    non_invasive_target: str = "dcis",
    epithelium_target: str | None = None,
) -> dict | None:
    """A previous run of this exact region, if there is one that still applies.

    Segmenting a region is the whole cost of this script, and the app writes every
    result to `data/runs/<roi_id>/` already - so re-running the same region with the
    same settings is pure waste. That is what makes it cheap to start with a narrow
    sample and widen it later.

    **Reused only when the settings match.** A run computed on one fold is not a run on
    five, and a run at one resolution is not a run at another; silently reusing either
    would put two different kinds of label in one training set with nothing on disk
    recording which was which. Anything that does not match is recomputed.
    """
    manifest_path = config.RUNS_DIR / roi.roi_id / "manifest.json"
    # Asking for the pre-collapse rule on a run that has since been re-coded: the
    # original is beside it, saved by `pipeline.relabel_run` precisely so that
    # `--split-epithelium` costs a file read rather than a re-segmentation.
    if epithelium_target is None:
        preserved = manifest_path.parent / "manifest.pre-collapse.json"
        if preserved.exists():
            manifest_path = preserved
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # a truncated file from an interrupted run - just redo it

    if tuple(manifest.get("teacher", {}).get("folds", ())) != tuple(folds):
        return None
    if abs(float(manifest.get("source_mpp", -1)) - source_mpp) > 1e-9:
        return None
    # A run whose masks wrote BEETLE's non-invasive class as a different BCSS code is
    # not this run. It would be reusable for the *tiles* - `bcss.remap` collapses 13 and
    # 20 into class 1 - but the mask copied into approach 1's tree would carry the other
    # tree's claim about the tissue, which is exactly the error this field exists to
    # stop. Anything that does not match is recomputed.
    if manifest.get("non_invasive_target", "dcis") != non_invasive_target:
        return None
    # **The check that decides whether a rule change actually happens.** Every one of
    # the 120 DCIS runs already on disk was written before `epithelium_code` existed
    # and carries the teacher's own invasive/in-situ split. Reusing them under the new
    # rule would copy the old masks into the training tree while every manifest, log
    # line and report claimed the new one - the change would appear to have been made
    # and would have changed nothing, which is the worst of the available failures.
    # A manifest with no key at all means `None`, which is what it meant.
    if manifest.get("epithelium_target") != epithelium_target:
        return None
    # The pair itself has to still be on disk; the manifest alone is not the artefact.
    run_dir = manifest_path.parent
    for key in ("image_path", "mask_path"):
        if not manifest.get(key) or not (run_dir / manifest[key]).exists():
            return None
    return manifest


def choose(entries: list, per_case: int | None, limit: int | None, seed: int) -> list:
    """Regions to hand over. `per_case=None` means all of them, which is the default.

    Whole-corpus is the right default and the sampling is the thing that needs a reason.
    790 regions cost about two hours on this machine, the labels are free, and the
    borrowed class is the only class the model is short of - so there is nothing to buy
    by leaving regions out.

    `--per-case N` exists for a quick pass. When it is used, two things are balanced:
    sampling per patient stops the set being a study of the handful of patients with the
    most regions (42 of the 665 training regions are case 1247's alone), and preferring
    large regions within a patient makes the shorter run worth its wall clock, because
    tile yield scales with area while the per-region overhead does not.
    """
    if per_case is None:
        chosen = sorted(entries, key=lambda e: (e.case_id, e.roi_id))
        return chosen[:limit] if limit else chosen

    by_case: dict[str, list] = defaultdict(list)
    for entry in entries:
        by_case[entry.case_id].append(entry)

    rng = random.Random(seed)
    chosen = []
    for case in sorted(by_case):
        regions = sorted(by_case[case], key=lambda e: -(e.width * e.height))
        # The largest few, then a random one from the tail so the set is not purely the
        # biggest fields - which would be a systematically different kind of image.
        head = regions[:per_case]
        if len(regions) > per_case and per_case > 1:
            head[-1] = rng.choice(regions[per_case - 1 :])
        chosen.extend(head)

    chosen.sort(key=lambda e: (e.case_id, e.roi_id))
    return chosen[:limit] if limit else chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-type", default="dcis", choices=sorted(config.ROI_TREES),
                        help="which BRACS lesion tree to harvest. `dcis` supplies the "
                             "in-situ class BCSS has nine tiles of; `ic` supplies "
                             "invasive carcinoma from a second laboratory, which is "
                             "what breaks the `source == class 1` confound the "
                             "leakage check currently reports as inconclusive.")
    parser.add_argument("--split", default="all",
                        choices=("all", "train", "val", "test"),
                        help="BRACS's own splits. `all` is the default and is what you "
                             "want: these are not being used as BRACS intended, only as "
                             "a source of in-situ epithelium, and the three splits "
                             "together carry 87 patients against train's 60. This "
                             "pipeline makes its own patient-level split later.")
    parser.add_argument("--per-case", type=int, default=None,
                        help="take only N regions per patient instead of all of them. "
                             "For a quick pass; the full corpus is the default.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-done", action="store_true", default=True,
                        help="reuse a region's cached segmentation from data/runs/ "
                             "instead of recomputing it. On by default; the cached "
                             "result is only reused when its folds and resolution "
                             "match what is being asked for now.")
    parser.add_argument("--no-skip-done", dest="skip_done", action="store_false",
                        help="recompute every region even if a cached run exists")
    parser.add_argument("--keep-flagged", action="store_true",
                        help="include regions whose verdict still blocks them - the "
                             "teacher found almost no epithelium, or no tile survived "
                             "the vote. Off by default. This no longer covers a teacher "
                             "that disagrees about the *lesion*: under the epithelium "
                             "union those regions are kept and relabelled, not flagged.")
    parser.add_argument(
        "--cached-only", action="store_true",
        help="never run the teacher: take only the regions whose segmentation is "
             "already in data/runs/, and skip the rest. **A scheduling tool, not a "
             "cheaper export** - it produces a tree with fewer regions than the source "
             "has, and the manifest records how many were skipped so the difference is "
             "never mistaken for the corpus. Its use is to put an hour of GPU where it "
             "buys the most: class 2 is abundant, so segmenting 131 more invasive "
             "regions is worth far less than segmenting the DCIS ones, and this is what "
             "lets the invasive tree be assembled from cache while that happens.")
    parser.add_argument("--split-epithelium", action="store_true",
                        help="write BEETLE's invasive and in-situ as separate codes and "
                             "reject a region where its verdict contradicts BRACS's "
                             "label - the rule every export before this flag existed "
                             "ran under. Kept so those exports stay reproducible; it "
                             "reinstates the filter that removed one DCIS region in "
                             "five, and it removed the solid and comedo ones.")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be processed and stop")
    args = parser.parse_args()

    folds = tuple(sorted({int(f) for f in args.folds.split(",") if f.strip()}))
    if not folds or not all(0 <= f <= 4 for f in folds):
        print("folds must be between 0 and 4", file=sys.stderr)
        return 2

    tree = config.ROI_TREES[args.roi_type]
    if args.split_epithelium:
        # `RoiTree` is frozen, so the override is a new tree rather than a mutation -
        # which also means every place downstream that reads `tree.epithelium_code`
        # sees one answer for the whole run.
        tree = dataclasses.replace(tree, epithelium_code=None)
    dest = APPROACH1_DATA / tree.name
    # BRACS ships no resolution anywhere on disk, so `BRACS_MPP` is this project's
    # stated assumption (see its docstring). BACH publishes 0.42 um/px in its challenge
    # paper, which is a fact rather than an inference - so the tree carries its own and
    # only falls back to the assumption when it has none.
    source_mpp = tree.source_mpp or config.BRACS_MPP

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    entries = [
        entry
        for split in splits
        for entry in pipeline.list_rois(split, roi_type=args.roi_type)
    ]
    chosen = choose(entries, args.per_case, args.limit, args.seed)
    patches = sum(e.as_json()["patches_per_fold"] for e in chosen) * len(folds)
    print(f"{tree.ftp_dir}: {len(chosen)} regions from "
          f"{len({e.case_id for e in chosen})} patients, teaching "
          f"{'/'.join(str(c) for c in sorted(tree.teaches))} "
          f"({', '.join(tree.teaches_names)}), ~{patches} forward passes, "
          f"~{patches * 0.62 / 60:.0f} min", file=sys.stderr)

    if args.dry_run:
        for entry in chosen[:40]:
            print(f"  {entry.roi_id:<26} {entry.width * entry.height / 1e6:6.2f} MP",
                  file=sys.stderr)
        return 0

    images_dir, masks_dir = dest / "images", dest / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    kept: list[dict] = []
    rejected: list[dict] = []
    # Kept apart from `rejected`: a skipped region is a scheduling decision, and a
    # rejected one is a statement about the tissue. Pooling them would make the next
    # reader of this manifest think the teacher had an opinion about regions it never
    # saw.
    skipped: list[dict] = []
    started = time.time()

    reused = 0
    relabelled = 0
    for index, roi in enumerate(chosen, 1):
        prefix = f"[{index}/{len(chosen)}] {roi.roi_id}"
        cached = (
            cached_manifest(roi, folds, source_mpp, tree.non_invasive_code,
                            tree.epithelium_code)
            if args.skip_done
            else None
        )
        # A run cached under the *previous* rule. Its teacher output is still valid -
        # only the translation of it changed - so it is re-coded in place instead of
        # being re-segmented. This is what turns "re-export the 120 DCIS regions" from
        # an hour of GPU into a few seconds, and "re-export all 790" from five and a
        # half hours into about a minute.
        stale = (
            cached_manifest(roi, folds, source_mpp, tree.non_invasive_code, None)
            if (args.skip_done and cached is None and tree.epithelium_code is not None)
            else None
        )

        if cached is None and stale is None and args.cached_only:
            skipped.append({"roi_id": roi.roi_id, "case_id": roi.case_id,
                            "reason": "--cached-only and no cached segmentation"})
            continue

        if cached is not None:
            manifest = cached
            run_dir = config.RUNS_DIR / roi.roi_id
            reused += 1
        elif stale is not None:
            run_dir = config.RUNS_DIR / roi.roi_id
            manifest = pipeline.relabel_run(
                roi, run_dir, stale, epithelium_target=tree.epithelium_code
            )
            relabelled += 1
        else:
            try:
                # `export_tiles=False`: approach 1's own exporter cuts the tiles, from
                # the pair written below. Cutting them twice would put two tile sets on
                # disk with no way to tell which the manifest describes.
                result = pipeline.run_region(
                    roi, folds=folds, source_mpp=source_mpp,
                    export_tiles=False, split_epithelium=args.split_epithelium,
                    progress=lambda m, f: None,
                )
            except Exception as error:  # noqa: BLE001
                rejected.append({"roi_id": roi.roi_id, "reason": f"failed: {error}"})
                print(f"{prefix}: FAILED - {error}", file=sys.stderr)
                continue
            manifest = result.manifest
            run_dir = result.run_dir
        verdict = manifest["verdict"]
        share = manifest["class_area_fraction"][tree.compare_name]

        if not verdict["usable"] and not args.keep_flagged:
            rejected.append({
                "roi_id": roi.roi_id, "case_id": roi.case_id,
                "corroborated_classes": list(tree.teaches_names),
                "corroborated_area": share,
                # Under its historic name too, so the DCIS manifest keeps the field a
                # reader of the previous run already knows.
                "non_invasive_area": manifest["class_area_fraction"][
                    "non_invasive_epithelium"
                ],
                "reason": "; ".join(verdict["notes"]),
            })
            print(f"{prefix}: {share:.0%} {tree.teaches_short}  REJECTED",
                  file=sys.stderr)
            continue

        shutil.copyfile(run_dir / manifest["image_path"], images_dir / f"{roi.roi_id}.png")
        shutil.copyfile(run_dir / manifest["mask_path"], masks_dir / f"{roi.roi_id}.png")

        kept.append({
            "roi_id": roi.roi_id,
            "case_id": roi.case_id,
            "bracs_split": roi.split,
            "source_mpp": manifest["working_mpp"],
            "native_size": manifest["native_size"],
            "size": manifest["working_size"],
            "teacher_folds": manifest["teacher"]["folds"],
            "teacher_confidence": manifest["teacher"]["mean_confidence"],
            "class_area_fraction": manifest["class_area_fraction"],
            "beetle_class_area": manifest["teacher"]["beetle_class_area"],
            # Under the old rule this region's fate; under the new one, a number. Sort
            # the manifest by it to get the review queue a pathologist should look at
            # first - these are the regions where the consensus and the teacher say
            # different things and only one of them is now reaching the labels.
            "teacher_overruled_share": verdict.get("teacher_overruled_share"),
        })
        overruled = verdict.get("teacher_overruled_share")
        print(f"{prefix}: {share:.0%} {tree.teaches_short}  kept"
              + ("  (cached)" if cached is not None else "")
              + ("  (re-coded)" if stale is not None else "")
              # The number worth seeing in the log, because these are the regions the
              # old rule deleted: a high share is solid or comedo architecture the
              # teacher read as invasive, or genuine microinvasion.
              + (f"  [teacher said {overruled:.0%} {tree.contradicts_short}]"
                 if overruled else ""),
              file=sys.stderr)

    if not kept:
        print("nothing kept - refusing to write an empty manifest", file=sys.stderr)
        return 1

    sidecar = {
        "source": tree.source,
        "roi_type": tree.name,
        "bracs_roi_class": tree.ftp_dir,
        # The one number `datasets.SOURCE_CLASSES` has to agree with. Recorded as data
        # so the authority rule and the tiles it applies to cannot drift apart.
        "teaches_classes": sorted(tree.teaches),
        "teaches_class_names": list(tree.teaches_names),
        "bracs_splits_used": list(splits),
        # Derived, not spelled out: this folder has been renamed once already and a
        # manifest naming a directory that no longer exists is worse than no provenance.
        "produced_by": f"{config.PROJECT_ROOT.name}/scripts/export_to_approach1.py",
        "native_mpp": source_mpp,
        # The one number approach 1's exporter must not guess. Every region here is at
        # the teacher's spacing, which is also the tile spacing, so its resample is a
        # no-op - but it has to be told, not left to infer from a filename.
        "source_mpp": config.TEACHER_MPP,
        "mask_codes": "bcss.GT_CODES (the raw 22-code BCSS vocabulary)",
        "labels_are": (
            "BEETLE's released nnU-Net's prediction on BRACS regions, remapped to BCSS "
            "codes. Model-generated, reviewed by nobody. Not ground truth."
        ),
        "label_authority": (
            f"BRACS's own {tree.lesion} consensus (three pathologists) says what the "
            f"region contains; BEETLE says where. Only class "
            f"{'/'.join(str(c) for c in sorted(tree.teaches))} "
            f"({', '.join(tree.teaches_names)}) is corroborated here, which "
            f"is why `datasets.SOURCE_CLASSES` must map {tree.source!r} to "
            f"{set(sorted(tree.teaches))} and nothing else."
            + (
                f" Both of BEETLE's epithelium classes are written as "
                f"{tree.epithelium_code!r}: the union of its invasive and in-situ "
                "predictions is taken as 'epithelium is here', and the consensus "
                "decides which lesion that is. Where the two disagreed about the "
                "lesion the consensus wins and the region is kept - see "
                "`items[].teacher_overruled_share` for how often, and by how much. "
                "BRACS labels the predominant lesion, so a region holding a true "
                "focus of the other one teaches it as this one; that is the accepted "
                "cost, and it replaces a filter that deleted such regions entirely "
                "and so deleted one morphology preferentially."
                if tree.epithelium_code is not None
                else " BEETLE also decides which lesion, and regions where the two "
                     "disagree are in `rejected` rather than in the training set - the "
                     "pre-`epithelium_code` rule, reproduced by --split-epithelium."
            )
        ),
        # Recorded as data, not prose: `verdict` branches on it, the cache check
        # compares it, and a reader of two manifests must be able to tell which rule
        # wrote which without parsing a sentence.
        "epithelium_target": tree.epithelium_code,
        "licence": "CC BY-NC-SA 4.0 (BEETLE) + BRACS non-commercial. Research only.",
        "teacher_folds": list(folds),
        "regions": len(kept),
        "reused_from_cache": reused,
        "cases": len({row["case_id"] for row in kept}),
        "rejected": rejected,
        "skipped": skipped,
        "skipped_rule": (
            "--cached-only was set and these regions had no segmentation in "
            "data/runs/, so the teacher was never run on them. They are absent from "
            "this tree, not judged by it."
            if skipped else None
        ),
        "rejection_rule": (
            (
                "verdict.usable == False - the teacher found almost no epithelium at "
                f"all in a region BRACS annotates {tree.lesion} (a resolution or stain "
                "fault, not a disagreement about the lesion), or no tile survived the "
                "vote. A teacher that disagrees about the *lesion* no longer rejects "
                "anything: the consensus overrules it and the tiles are kept."
                if tree.epithelium_code is not None
                else f"verdict.usable == False - the teacher found more "
                     f"{tree.contradicts_short} than {tree.teaches_short} epithelium, "
                     f"or almost no {tree.teaches_short} at all, in a region BRACS "
                     f"annotates {tree.lesion}"
            )
        ),
        "wall_clock_seconds": round(time.time() - started, 1),
        "items": kept,
    }
    (dest / f"{tree.name}_manifest.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 64}", file=sys.stderr)
    print(f"  kept {len(kept)} regions from {sidecar['cases']} patients", file=sys.stderr)
    if skipped:
        print(f"  SKIPPED {len(skipped)} regions with no cached segmentation "
              f"(--cached-only)", file=sys.stderr)
    print(f"  rejected {len(rejected)}  |  reused from cache {reused}"
          f"  |  re-coded from a pre-collapse run {relabelled}", file=sys.stderr)
    if tree.epithelium_code is not None:
        overruled = [
            row["teacher_overruled_share"] for row in kept
            if row.get("teacher_overruled_share")
        ]
        # The headline of this rule change: how many regions the old filter would have
        # thrown away, and how much epithelium the consensus overruled the teacher on.
        print(f"  the consensus overruled the teacher on {len(overruled)} of "
              f"{len(kept)} regions"
              + (f", median {sorted(overruled)[len(overruled) // 2]:.0%} of their "
                 "epithelium" if overruled else ""), file=sys.stderr)
    print(f"  -> {dest}", file=sys.stderr)
    print(f"  {(time.time() - started) / 60:.1f} min", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
