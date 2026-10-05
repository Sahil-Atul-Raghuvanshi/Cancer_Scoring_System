"""One BRACS region, end to end: image in, mask out, tiles counted, verdict recorded.

The order of operations is the whole argument, so it is stated once here and the code
below follows it exactly:

  1. read the region at BRACS's native resolution
  2. resample **once**, to 0.5 um/px, with area averaging on the intensities. This is
     both the teacher's training spacing and approach 1's tile spacing, which is a
     coincidence worth exploiting: resampling once means the mask and the image the
     exporter votes on are the same array, guaranteed in register.
  3. run the teacher, get a 5-class argmax
  4. translate it into BCSS's own codes (`labels.to_bcss_codes`) and write it as the
     same kind of uint16 PNG the BCSS release ships
  5. hand the pair to **approach 1's exporter, unmodified**, and count what comes out

Step 5 is the answer to the question this app was built to ask. Anything less than
approach 1's own `export_region` - a reimplementation, a simplified vote, a different
white point - would produce tiles that look like approach 1's and are not, and the
comparison would be worthless. So `export.export_region` is imported and called, and
this module's job is only to build the `Region` it wants.

**What the numbers here do and do not establish.** A large class 1 tile count proves
the pipeline yields tiles. It does not prove they are correct: the labels are a model's
opinion of a second dataset, two steps removed from a pathologist, and BEETLE's own
external non-invasive Dice is the ceiling on how right they can be. `verdict` says so
in the manifest rather than leaving it to be inferred.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import config, labels, teacher

Image.MAX_IMAGE_PIXELS = None  # BRACS regions run to 40 MPx; none of them is a decompression bomb

Progress = Callable[[str, float], None]


def _noop(message: str, fraction: float) -> None:
    return None


# --- listing ------------------------------------------------------------------


@dataclass(frozen=True)
class RoiEntry:
    roi_id: str
    split: str
    path: Path
    width: int
    height: int
    bytes: int

    #: Which BRACS lesion tree this region came from - `"dcis"` or `"ic"`. Last, with a
    #: default, so the six positional fields above keep working. It decides the
    #: manifest's `source` and which way round `verdict` argues, so it travels with the
    #: entry rather than being re-derived from the filename at each use.
    roi_type: str = "dcis"

    @property
    def tree(self):
        """The `config.RoiTree` this region belongs to."""
        return config.ROI_TREES[self.roi_type]

    #: `BRACS_1247_DCIS_1` -> `BRACS_1247`. The case, and the unit any split must group
    #: by: 42 of the 665 training regions come from case 1247 alone, and tiles from one
    #: case are near-duplicates. Approach 1 groups by TCGA participant for the same
    #: reason, and a BRACS-augmented training set that grouped by region instead would
    #: report a generalisation number it had not earned.
    @property
    def case_id(self) -> str:
        parts = self.roi_id.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else self.roi_id

    def as_json(self) -> dict:
        return {
            "roi_id": self.roi_id,
            "case_id": self.case_id,
            "split": self.split,
            "width": self.width,
            "height": self.height,
            "megapixels": round(self.width * self.height / 1e6, 2),
            "bytes": self.bytes,
            "patches_per_fold": teacher.patch_count(
                max(1, int(round(self.height * config.BRACS_MPP / config.TEACHER_MPP))),
                max(1, int(round(self.width * config.BRACS_MPP / config.TEACHER_MPP))),
            ),
        }


def list_rois(split: str = "train", roi_type: str = "dcis") -> list[RoiEntry]:
    """Every BRACS region of one lesion type in one split - the header, not the pixels."""
    tree = config.ROI_TREES[roi_type]
    # BRACS ships `<tree>/<split>/*.png`; BACH ships one flat directory of TIFFs per
    # class and has no splits of its own. The layout is a property of the tree rather
    # than an assumption here, so a third source can be added without another branch.
    directory = tree.directory / split if tree.has_splits else tree.directory
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no {tree.ftp_dir} regions at {directory}. "
            f"Run data/fetch_{roi_type}.sh to fetch them."
        )
    if not tree.has_splits and split not in ("train", "all"):
        # A flat tree has every image in one place; asking for `val` would silently
        # return the whole set a second time and double-count it.
        return []

    entries = []
    for path in sorted(directory.glob(f"*{tree.suffix}")):
        with Image.open(path) as image:  # lazy - reads the header, not the raster
            width, height = image.size
        entries.append(
            RoiEntry(
                roi_id=path.stem,
                split=split,
                path=path,
                width=width,
                height=height,
                bytes=path.stat().st_size,
                roi_type=roi_type,
            )
        )
    return entries


@dataclass(frozen=True)
class _UploadEntry(RoiEntry):
    """An `RoiEntry` whose `case_id` is not guessed from its name. See `entry_for_upload`."""

    @property
    def case_id(self) -> str:
        return self.roi_id


def entry_for_upload(path: Path, roi_id: str) -> RoiEntry:
    """An `RoiEntry` for an image a user uploaded, so `run_region` needs no second form.

    `run_region` reads six fields off an `RoiEntry` and knows nothing about where the
    file came from, so an upload is describable in the same terms rather than needing a
    parallel code path. Two fields are stated rather than derived and both matter:

      * `split` is `"upload"`, which is not one of BRACS's three. Nothing selects on it,
        but it lands in the manifest, and a manifest that claimed `train` would be a
        file on disk asserting this image is part of a dataset it is not part of.
      * `case_id` is the whole `roi_id`. `RoiEntry.case_id` truncates to the first two
        underscore-separated parts because BRACS names encode the patient there; an
        uploaded filename encodes nothing, and grouping two unrelated uploads under a
        shared prefix would be an invented claim that they came from one patient.
    """
    with Image.open(path) as image:
        width, height = image.size
    return _UploadEntry(
        roi_id=roi_id,
        split="upload",
        path=path,
        width=width,
        height=height,
        bytes=path.stat().st_size,
    )


def find_roi(roi_id: str) -> RoiEntry:
    """One region by name, from whichever lesion tree holds it.

    The name says which tree to look in - BRACS puts the lesion type in the third field
    - but the search falls back to every tree, because a region that is on disk under a
    name this parser does not recognise should still be findable rather than reported
    missing.
    """
    named = config.tree_for_roi_id(roi_id)
    order = [named.name] if named else []
    order += [name for name in config.ROI_TREES if name not in order]

    for roi_type in order:
        tree = config.ROI_TREES[roi_type]
        for split in ("train", "val", "test"):
            candidate = tree.directory / split / f"{roi_id}.png"
            if candidate.exists():
                with Image.open(candidate) as image:
                    width, height = image.size
                return RoiEntry(
                    roi_id=roi_id,
                    split=split,
                    path=candidate,
                    width=width,
                    height=height,
                    bytes=candidate.stat().st_size,
                    roi_type=roi_type,
                )
    raise FileNotFoundError(f"no BRACS region called {roi_id!r} in any split")


# --- resampling ---------------------------------------------------------------


def prepare_region(path: Path, source_mpp: float) -> np.ndarray:
    """The region at the teacher's spacing, resampled in intensity space.

    `Image.Resampling.BOX` is area averaging, and it is the correct filter for the same
    reason approach 1's `export.resample` gives: a coarser scan physically *is* the
    average of the light arriving over a larger sensor area. It also has to happen here,
    before any optical-density transform, because averaging densities takes the mean of
    logarithms - a number no instrument records.
    """
    factor = config.TEACHER_MPP / source_mpp
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if abs(factor - 1.0) < 1e-6:
            return np.asarray(rgb)
        size = (max(1, round(rgb.width / factor)), max(1, round(rgb.height / factor)))
        return np.asarray(rgb.resize(size, Image.Resampling.BOX))


# --- pictures -----------------------------------------------------------------


def overlay(rgb: np.ndarray, our_classes: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """The region with its three classes painted over it.

    Alpha-blended rather than drawn as a flat colour map, because the question a
    pathologist asks of this picture is "did it draw round the ducts", and that cannot
    be answered from a mask alone with the tissue hidden underneath it.
    """
    import bcss

    painted = np.zeros_like(rgb)
    painted[:] = labels.IGNORE_COLOUR
    for cls, colour in labels.CLASS_COLOURS.items():
        painted[our_classes == cls] = colour

    blended = (1 - alpha) * rgb.astype(np.float32) + alpha * painted.astype(np.float32)
    # Unlabelled pixels keep the tissue unmodified: tinting them would imply the model
    # said something there, and `IGNORE` is the absence of a statement.
    unlabelled = our_classes == bcss.IGNORE
    blended[unlabelled] = rgb[unlabelled]
    return blended.clip(0, 255).astype(np.uint8)


def _downscale_for_web(array: np.ndarray, longest: int = 1400) -> Image.Image:
    """A preview that fits down a socket. Full-resolution PNGs run to 20 MB each."""
    image = Image.fromarray(array)
    scale = longest / max(image.size)
    if scale < 1:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


# --- the run ------------------------------------------------------------------


@dataclass
class RunResult:
    roi_id: str
    run_dir: Path
    manifest: dict = field(default_factory=dict)


def run_region(
    roi: RoiEntry,
    *,
    folds: tuple[int, ...] = (0,),
    source_mpp: float | None = None,
    export_tiles: bool = True,
    out_dir: Path | None = None,
    split_epithelium: bool = False,
    write_preview: bool = False,
    progress: Progress = _noop,
) -> RunResult:
    """Segment one region, write its mask, and cut approach 1's tiles from it.

    `out_dir` defaults to this region's directory under `data/runs/`. Uploads pass their
    own so their output never lands in the tree that `compare.py` and
    `scripts/export_to_approach1.py` read as harvested BRACS data - see
    `config.UPLOADS_DIR` for why that separation is not cosmetic.

    `write_preview` additionally writes `overlay.png` and `original.png` - a colour
    rendering of the mask over the region, downscaled. Off by default: they cost about
    4.4 MB a region and nothing in this pipeline reads them, so they are for a person
    looking at a mask, which is worth doing on the regions where the ROI consensus
    overruled the teacher and worth nothing on the rest.

    `split_epithelium` writes BEETLE's invasive and in-situ as separate codes, ignoring
    this tree's `epithelium_code` - the rule every run before that field existed used.
    It is here rather than only on the command line because `run_region` is what decides
    what the mask on disk claims, and a flag that could not reach this far would be a
    flag that did nothing.
    """
    import bcss
    import export as approach1_export

    source_mpp = float(source_mpp if source_mpp is not None else config.BRACS_MPP)
    run_dir = out_dir if out_dir is not None else config.RUNS_DIR / roi.roi_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    progress("reading and resampling the region", 0.01)
    rgb = prepare_region(roi.path, source_mpp)

    archive = teacher.read_archive()

    def teacher_progress(message: str, fraction: float) -> None:
        # The teacher is the overwhelming majority of the wall clock, so it owns the
        # bulk of the bar. A progress bar that jumps from 5 % to 95 % is a spinner.
        progress(message, 0.05 + 0.75 * fraction)

    prediction = teacher.predict(
        rgb, archive, folds=folds, progress=teacher_progress
    )

    progress("translating BEETLE's classes into BCSS codes", 0.82)
    # Per-tree, because the epithelium of a consensus-normal region is normal duct and
    # not carcinoma in-situ. `bcss.remap` sends both to class 1, so no tile label moves
    # - what changes is whether the mask on disk tells the truth about the tissue.
    non_invasive_code = (
        roi.tree.non_invasive_code
        if roi.split != "upload"
        else labels.DEFAULT_NON_INVASIVE_TARGET
    )
    # The epithelium-union rule: BEETLE says where the epithelium is, the ROI consensus
    # says which lesion it is. See `config.RoiTree.epithelium_code`. An upload has no
    # consensus to defer to, so it gets no collapse and the teacher's own classes stand.
    epithelium_code = (
        None
        if (split_epithelium or roi.split == "upload")
        else roi.tree.epithelium_code
    )

    # **Asserted, not trusted.** A tree that collapsed its epithelium onto a class its
    # consensus has no authority for would manufacture the unadjudicated label
    # `teaches` exists to exclude, and `datasets.by_source_authority` would then drop
    # every tile of it one repository downstream - a silent empty class, not an error.
    # `tests/test_config.py` makes the same check over the whole table; this one covers
    # the region actually being written.
    if epithelium_code is not None:
        collapsed_class = int(bcss.remap(
            np.full((1, 1), bcss.GT_CODES[epithelium_code], dtype=np.uint16)
        )[0, 0])
        if collapsed_class not in roi.tree.teaches:
            raise ValueError(
                f"{roi.tree.name}: epithelium_code {epithelium_code!r} remaps to class "
                f"{collapsed_class}, which is not in teaches={sorted(roi.tree.teaches)}. "
                "Collapsing epithelium onto a class this lesion's consensus does not "
                "corroborate would invent the disagreement `teaches` exists to keep out."
            )

    bcss_mask = labels.to_bcss_codes(
        prediction.mask,
        non_invasive_target=non_invasive_code,
        epithelium_target=epithelium_code,
    )
    our_classes = bcss.remap(bcss_mask)

    # **The pair must be in register, and this is where that is guaranteed.** Both
    # arrays are (H, W) derived from one resample, so a mismatch here is not a rounding
    # difference - it is an axis swap, and an axis swap is silent: a transposed mask on
    # a near-square region still saves, still tiles, and teaches every duct at the wrong
    # place. Approach 1's exporter pairs them by filename and would never notice. So the
    # shapes are asserted rather than trusted, in the numpy convention (rows, cols),
    # before either file exists.
    if bcss_mask.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"{roi.roi_id}: the mask is {bcss_mask.shape[:2]} and the image is "
            f"{rgb.shape[:2]} (rows, cols). These must be identical - a transposed or "
            "differently-resampled mask puts every label in the wrong place and nothing "
            "downstream can detect it."
        )

    # The mask, in the BCSS release's own format: single channel, uint16, code-valued.
    # This is the artefact the question "can a mask like TCGA-A1-A0SK-... be created"
    # is asking about, and it is written before the tiles so it exists even if the
    # export is turned off.
    mask_path = run_dir / f"{roi.roi_id}_bcss_codes.png"
    Image.fromarray(bcss_mask).save(mask_path)

    # The resampled RGB, saved alongside. The exporter reads its image from disk, and
    # handing it the original would silently reintroduce the 2x scale difference.
    image_path = run_dir / f"{roi.roi_id}_image.png"
    Image.fromarray(rgb).save(image_path, optimize=True)

    # **Off by default, and the reason is measured.** These two downscaled PNGs existed
    # for the web screen that rendered them, and nothing reads them now: over 725 runs
    # they had reached **3.1 GB of a 5.9 GB cache**, against 25 MB for the masks that
    # carry the actual labels. So the capability is kept and the cost is not paid unless
    # somebody asks - which they should, when reviewing the regions where the consensus
    # overruled the teacher, because that queue is looked at rather than computed on.
    if write_preview:
        progress("drawing the overlay", 0.85)
        _downscale_for_web(overlay(rgb, our_classes)).save(
            run_dir / "overlay.png", optimize=True
        )
        _downscale_for_web(rgb).save(run_dir / "original.png", optimize=True)

    class_areas = {
        name: float((our_classes == index).mean())
        for index, name in enumerate(bcss.CLASS_NAMES)
    }
    class_areas["unlabelled"] = float((our_classes == bcss.IGNORE).mean())

    manifest: dict = {
        "roi_id": roi.roi_id,
        "case_id": roi.case_id,
        "split": roi.split,
        # Read by `verdict`, which has BRACS's own annotation to argue with on a BRACS
        # region and nothing at all to argue with on an image somebody uploaded. On a
        # BRACS region it also says *which* annotation: a DCIS consensus and an IC
        # consensus disagree with the teacher in opposite directions.
        "source": "upload" if roi.split == "upload" else roi.tree.source,
        "roi_type": None if roi.split == "upload" else roi.roi_type,
        # Read by the exporter's cache check: a run whose masks were written under a
        # different in-situ code is not a run that can be reused for this one.
        "non_invasive_target": non_invasive_code,
        # The same, for the rule that matters more. A mask written before
        # `epithelium_code` existed carries the teacher's own invasive/in-situ split and
        # is a different claim about the tissue from one written after it, so the two
        # must never be pooled - and `verdict` reads this to know which of its checks
        # can still fire. `None` means the pre-collapse rule, which is what every
        # manifest already on disk means by not carrying the key at all.
        "epithelium_target": epithelium_code,
        "source_mpp": source_mpp,
        "working_mpp": config.TEACHER_MPP,
        "native_size": [roi.width, roi.height],
        "working_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "teacher": {
            "folds": list(prediction.folds),
            "mean_confidence": round(float(prediction.confidence.mean()), 4),
            "mean_fold_disagreement": (
                round(float(prediction.disagreement.mean()), 4)
                if len(prediction.folds) > 1
                else None
            ),
            "beetle_class_area": {
                name: round(float((prediction.mask == code).mean()), 4)
                for name, code in labels.BEETLE_CODES.items()
            },
        },
        "class_area_fraction": {k: round(v, 4) for k, v in class_areas.items()},
        "mask_path": mask_path.name,
        "image_path": image_path.name,
    }

    if export_tiles:
        progress("cutting approach 1's tiles", 0.9)
        region = bcss.Region(
            roi_id=roi.roi_id,
            slide_id=roi.case_id,
            # Not a TCGA tissue source site, and deliberately not made to look like
            # one. `is_test` reads this, and a two-letter code here would put BRACS
            # tiles into BCSS's held-out institutions by accident.
            institution="BRACS",
            image=image_path,
            mask=mask_path,
        )
        tiles_dir = run_dir / "tiles"
        rows, drops = approach1_export.write_region(
            region, tiles_dir, source_mpp=config.TEACHER_MPP
        )

        counts = {name: 0 for name in bcss.CLASS_NAMES}
        for row in rows:
            counts[str(row["label_name"])] += 1

        manifest["tiles"] = {
            "kept": len(rows),
            "by_class": counts,
            "dropped": drops,
            "spec": approach1_export.DEFAULT_SPEC.descriptor(),
            "exporter": "tissue_type_model_training/src/export.py::write_region",
        }
        (run_dir / "tile_manifest.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

    manifest["seconds"] = round(time.time() - started, 1)
    manifest["verdict"] = verdict(manifest)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    progress("done", 1.0)
    return RunResult(roi_id=roi.roi_id, run_dir=run_dir, manifest=manifest)


def relabel_run(
    roi: RoiEntry,
    run_dir: Path,
    manifest: dict,
    *,
    epithelium_target: str,
) -> dict:
    """Re-code one cached run's mask under the epithelium-union rule. No teacher.

    **This is what makes the rule change affordable.** BEETLE is the entire cost of this
    pipeline - about 25 s a region, so 790 regions is five and a half hours - and the
    collapse is a pure relabelling of an argmax that is already on disk in
    `data/runs/<roi_id>/`. Re-segmenting to obtain it would be recomputing a number
    nobody has lost. `labels.recode_epithelium` carries the argument for why reading the
    two epithelium codes back out of a written mask is exact rather than approximate.

    The original mask is **left on disk** and the re-coded one written beside it under
    its own name, because the two are different claims about the same tissue and a
    directory that quietly overwrote one with the other would make the previous export
    unreproducible. `manifest.json` describes the current state and says which mask it
    means.

    **The collapse is not invertible** - once two codes are one, nothing in the new mask
    can say which pixels were which - so the pre-collapse manifest is copied aside to
    `manifest.pre-collapse.json` before it is replaced. That file plus the original mask
    is the whole of the old run, which is what lets `--split-epithelium` reproduce the
    earlier export from cache instead of spending 25 s a region re-deriving an answer
    that is still sitting on the disk.
    """
    import bcss

    old_mask_path = run_dir / manifest["mask_path"]
    old = np.asarray(Image.open(old_mask_path))
    recoded = labels.recode_epithelium(
        old,
        epithelium_target=epithelium_target,
        non_invasive_target=manifest.get(
            "non_invasive_target", labels.DEFAULT_NON_INVASIVE_TARGET
        ),
    )

    mask_path = run_dir / f"{roi.roi_id}_bcss_codes_epi-{epithelium_target}.png"
    Image.fromarray(recoded).save(mask_path)

    our_classes = bcss.remap(recoded)
    class_areas = {
        name: float((our_classes == index).mean())
        for index, name in enumerate(bcss.CLASS_NAMES)
    }
    class_areas["unlabelled"] = float((our_classes == bcss.IGNORE).mean())

    updated = dict(manifest)
    updated["epithelium_target"] = epithelium_target
    updated["mask_path"] = mask_path.name
    updated["class_area_fraction"] = {k: round(v, 4) for k, v in class_areas.items()}
    # Provenance, because a run that was relabelled and a run that was segmented under
    # the new rule are the same masks by construction but not the same event, and the
    # only place that can be recorded is here.
    updated["relabelled_from"] = {
        "mask_path": old_mask_path.name,
        "rule": "teacher's own invasive/in-situ split",
        "by": "pipeline.relabel_run",
    }
    # The tile counts described tiles cut under the old codes, so they are now a
    # statement about a mask this manifest no longer points at. Dropped rather than
    # recomputed: approach 1's exporter cuts the tiles that matter, from the pair on
    # disk, and a stale count is worse than an absent one. `verdict` reads a missing
    # block as "not measured" rather than as zero.
    updated.pop("tiles", None)
    updated["verdict"] = verdict(updated)

    # The old run, kept whole before it is replaced. Written only once: relabelling a
    # run that has already been relabelled must not overwrite the pre-collapse copy
    # with a post-collapse one, which would lose the only record of the original rule.
    preserved = run_dir / "manifest.pre-collapse.json"
    if not preserved.exists():
        preserved.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (run_dir / "manifest.json").write_text(
        json.dumps(updated, indent=2), encoding="utf-8"
    )
    return updated


#: Below this share of the region, a region of interest whose consensus lesion the
#: teacher barely finds is more likely a preprocessing failure - the wrong resolution, a
#: stain the teacher has not seen - than a genuinely sparse region. 5 % is a low bar on
#: purpose: it is a smoke alarm, not a quality threshold. Per-tree, via
#: `config.RoiTree.flag_when_below`; this is the default both trees currently take.
MIN_PLAUSIBLE_NON_INVASIVE_AREA = 0.05


def verdict(manifest: dict) -> dict:
    """The honest reading of one region's numbers, computed rather than narrated.

    Kept in code so the UI, the batch summary and the report cannot each decide for
    themselves what "worked" means.

    **Every note here is an argument with BRACS's own annotation**, and that is why they
    are conditional on the source. "Only 3 % in-situ epithelium" is a red flag on a
    region BRACS labelled DCIS, because two things that should agree do not. On an image
    a user uploaded there is nothing to disagree with: 3 % may be exactly right, and the
    same sentence would be an invented finding. So an upload gets the numbers and no
    verdict, which is the honest answer to "is this good" when nobody has said what the
    picture contains.

    **And the argument runs in opposite directions on the two lesion trees.** On a DCIS
    region, invasive epithelium outweighing in-situ is the contradiction. On an IC
    region it is the expected result, and *in-situ* outweighing invasive is the
    contradiction. The asymmetry is not a special case bolted on for IC - it is the same
    single rule read off `config.RoiTree`, which is what stops a future third tree from
    quietly inheriting DCIS's polarity.
    """
    areas = manifest["class_area_fraction"]
    tree = config.ROI_TREE_BY_SOURCE.get(manifest.get("source", ""))

    if tree is None:
        # An upload, or a source nobody has ruled on. Numbers, no verdict.
        return {
            "usable": None,
            "checked_against_annotation": False,
            "roi_type": None,
            "corroborated_class": None,
            "corroborated_area": None,
            "corroborated_tiles": None,
            # Kept under their historic names so a reader of an old manifest and a
            # reader of a new one are looking at the same field.
            "non_invasive_area": areas["non_invasive_epithelium"],
            "non_invasive_tiles": (
                manifest.get("tiles", {}).get("by_class", {}).get("non_invasive_epithelium")
            ),
            "notes": [],
            # Present and empty rather than absent, so one shape of verdict reaches
            # every reader. An upload has no consensus, so there is nothing for the
            # collapse to defer to and nothing it could overrule.
            "informational": [],
            "epithelium_union": False,
            "teacher_overruled_share": None,
        }

    area = areas[tree.compare_name]
    against = areas[tree.contradicts_name]
    by_class = manifest.get("tiles", {}).get("by_class", {})
    # Summed over every class this tree corroborates, because a region that yields
    # nothing usable is the thing being tested - and a normal region corroborates two.
    counts = [by_class.get(name) for name in tree.teaches_names]
    tiles = None if all(c is None for c in counts) else sum(c or 0 for c in counts)
    lesion = tree.lesion

    # Which rule wrote this run's mask. A manifest from before `epithelium_code`
    # existed carries no such key, and `None` is exactly what it meant: the teacher's
    # own invasive/in-situ split, with disagreement resolved by deletion.
    collapsed = manifest.get("epithelium_target") is not None

    notes: list[str] = []
    # Kept apart from `notes` on purpose: `usable` is driven by `notes` alone, and a
    # remark that does not block must not be able to reject a region by being appended
    # to the wrong list. That is the mistake the old rule made in the large.
    informational: list[str] = []
    if tree.flag_when_below > 0.0 and area < tree.flag_when_below:
        notes.append(
            f"the teacher found {'epithelium' if collapsed else tree.teaches_short + ' epithelium'} "
            f"on only {area:.1%} of a region BRACS annotates as {lesion}. Check the "
            "resolution assumption before trusting these tiles."
        )
    if against > area:
        # Unreachable under the collapse - `class_area_fraction` is measured after it,
        # so the contradicting class is empty by construction - and kept rather than
        # made conditional, because a mask written under the old rule still gates
        # through here and must still be read the way it was written.
        notes.append(
            f"more {tree.contradicts_short} ({against:.1%}) than {tree.teaches_short} "
            f"({area:.1%}) epithelium in {tree.article} {lesion} region. Either the "
            f"region genuinely contains {tree.contradicts_noun}, or the class codes are "
            "the wrong way round - the second would invert the whole training set and "
            "must be excluded first."
        )
    if tiles == 0:
        notes.append(
            "no tile of "
            + " or ".join(f"class {c}" for c in sorted(tree.teaches))
            + " survived the vote, so this region adds nothing to the training set "
            "even though it segmented."
        )

    # **The disagreement itself, kept as a measurement now that it is no longer a
    # rejection.** Under the collapse the teacher's opinion about *which* lesion this is
    # no longer reaches any label, so it cannot make a region unusable - but it is the
    # most interesting number on the region and losing it would be a real loss. A high
    # share here on a DCIS region is the signature of solid or comedo architecture the
    # teacher read as invasive, or of genuine microinvasion the ROI label does not
    # separate; those are the two possibilities and this pipeline cannot tell them
    # apart. Read off the teacher's raw class areas rather than recomputed, because
    # `class_area_fraction` no longer holds it.
    beetle_areas = (manifest.get("teacher") or {}).get("beetle_class_area") or {}
    contradicting = beetle_areas.get(tree.contradicts_name)
    corroborated_by_teacher = beetle_areas.get(tree.compare_name)
    overruled = None
    if collapsed and contradicting is not None:
        epithelium = contradicting + (corroborated_by_teacher or 0.0)
        overruled = round(contradicting / epithelium, 4) if epithelium > 0 else 0.0
        if corroborated_by_teacher is not None and contradicting > corroborated_by_teacher:
            informational.append(
                f"the teacher called {overruled:.0%} of the epithelium here "
                f"{tree.contradicts_short} in {tree.article} {lesion} region, more than "
                f"it called {tree.teaches_short}. The consensus overrules it and these "
                f"tiles are kept as {tree.teaches_short} - under the pre-collapse rule "
                "this region was deleted, and regions like it are the missing "
                "morphology. Either the architecture is one the teacher misreads, or "
                f"the region genuinely contains {tree.contradicts_noun}; nothing here "
                "can tell those apart, so a pathologist's eye is worth more on this "
                "region than on any other in the tree."
            )

    return {
        # `None` rather than `True` where there is no annotation to check against, so a
        # caller cannot read "we verified this" out of the absence of a complaint.
        #
        # **`notes` only, and that is now the whole of the rule change at this level.**
        # What blocks a region is a teacher that found almost no epithelium at all - a
        # wrong resolution or an unseen stain - or a region that yielded no tile. A
        # teacher that disagreed about the *lesion* is in `informational` and keeps its
        # tiles, because the consensus outranks it on that question.
        "usable": not notes,
        "checked_against_annotation": True,
        "epithelium_union": collapsed,
        "roi_type": tree.name,
        "corroborated_classes": list(tree.teaches_names),
        "compared_class": tree.compare_name,
        "corroborated_area": area,
        "corroborated_tiles": tiles,
        # The historic field names, still meaning exactly what they always meant - the
        # in-situ numbers - so nothing that reads an old manifest changes behaviour.
        # On an IC region these are the *contradicting* class, which is why the
        # corroborated ones above are named rather than left to be inferred.
        "non_invasive_area": areas["non_invasive_epithelium"],
        "non_invasive_tiles": by_class.get("non_invasive_epithelium"),
        "notes": notes,
        # Remarks that do not block. Empty under the pre-collapse rule, where the same
        # finding was a rejection instead.
        "informational": informational,
        # The share of the epithelium the teacher assigned to the contradicting class
        # and the consensus overruled. `None` where there is nothing to measure it from
        # - an old manifest, or a run whose teacher block was not recorded. This is the
        # number to sort by when a pathologist has an hour to review the hard regions.
        "teacher_overruled_share": overruled,
    }
