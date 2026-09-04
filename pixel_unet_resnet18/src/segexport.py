"""Regions in, `(image, mask)` tile pairs out. Idempotent, region by region.

The tile model's exporter reads a pixel mask, majority-votes each 224 px tile into one
label, and discards the pixels. This keeps them. That is the whole difference, and it has
two consequences worth stating.

**Every position is kept, not two thirds of them.** Approach 1 keeps 13,501 of 20,278
positions: it drops 2,671 as "mixed" because no class clears its vote threshold, and 4,106
as "unusable" because under 70 % of pixels carry a label. A tile with no majority is a tile
containing a **boundary** - exactly what a segmentation model is for, and exactly what the
vote exists to reject. So there is no vote here; there is a floor on how much of a tile is
labelled at all (`MIN_LABELLED`), because a tile that is 95 % unlabelled contributes almost
nothing and costs a full forward pass.

**The mask is written beside the image.** Both come from the same `window` of the same two
arrays, so they are pixel-aligned by construction rather than by a later check.

--------------------------------------------------------------------------------
Idempotency
--------------------------------------------------------------------------------

Re-running must be safe and must not redo finished work. An hour-long export that dies at
minute fifty should cost ten minutes, not fifty.

Every region writes **one JSON shard** to `data/seg_tiles/shards/<roi_id>.json` as its last
act, after its tiles are on disk. A shard carries a `settings_fingerprint`. On a later run
a region is skipped when, and only when:

  * its shard exists and parses, and
  * its fingerprint matches the current geometry and floor, and
  * every tile file the shard names is still on disk.

Any of those failing means the region is redone. The fingerprint is the part that matters:
silently reusing tiles cut at a different resolution or a different floor would put two
kinds of training data in one manifest with nothing recording which was which. Tiles are
written with `Image.save` to a temporary name and then `os.replace`d, which is atomic on
Windows and POSIX alike, so a kill mid-write leaves no half-PNG for the next run to trust.

The combined manifest is **rebuilt from the shards** at the end rather than accumulated in
memory, which is what makes a resumed run produce byte-identical output to an
uninterrupted one.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from PIL import Image

import classes
import hstain
import paths

Image.MAX_IMAGE_PIXELS = 500_000_000

Progress = Callable[[str], None]


def _noop(message: str) -> None:
    return None


@dataclass(frozen=True)
class TileSpec:
    """The tile geometry, in the units each number is a statement about.

    224 px at 0.5 um/px is 112 um, about nine cells across - enough to see whether
    epithelium sits inside a duct or has infiltrated. The same geometry the tile model
    uses, deliberately: these two models are meant to be comparable, and a different tile
    size would make any difference between them partly a difference of field of view.

    `stride` equals `tile_px` - no overlap. Overlapping training tiles are near-duplicates:
    they inflate the apparent dataset size, and a split falling between two overlapping
    tiles would leak them into each other.
    """

    tile_px: int = 224
    mpp: float = 0.5
    stride_px: int | None = None

    #: Share of a tile's pixels that must carry any label at all.
    #:
    #: A floor, not a vote. 0.20 rather than approach 1's 0.70 because a segmentation loss
    #: with `ignore_index` learns from the labelled part and ignores the rest, so a tile
    #: that is 40 % annotated is 40 % of a training signal rather than a reject. Below
    #: 20 % the forward pass costs the same and teaches almost nothing.
    min_labelled: float = 0.20

    #: Percentile of each channel taken as `I0` on a region with no glass in it.
    white_percentile: float = 99.0

    @property
    def stride(self) -> int:
        return self.stride_px if self.stride_px is not None else self.tile_px

    @property
    def tile_um(self) -> float:
        return self.tile_px * self.mpp

    def descriptor(self) -> dict[str, object]:
        return {**asdict(self), "stride": self.stride, "tile_um": self.tile_um}

    def fingerprint(self) -> str:
        """A hash of every setting that changes what a tile *is*.

        This is what makes the skip safe. Change the tile size, the resolution, the
        stride, the floor or the white-point rule and every shard becomes stale - which is
        correct, because the tiles on disk are no longer the tiles this spec describes.
        """
        payload = json.dumps(self.descriptor(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


DEFAULT_SPEC = TileSpec()


@dataclass(frozen=True)
class Region:
    """One region of interest: an image, a mask, and where it came from."""

    roi_id: str
    patient: str
    institution: str
    source: str
    image: Path
    mask: Path
    #: This region's own resolution. **No default, deliberately.** Every BCSS filename
    #: ends `MPP-0.2500` and the slides are not 0.25 - the first measures 0.2521 - so a
    #: default here would be an assumption that silently puts tiles at the wrong physical
    #: size. A 224 px tile is meant to cover 112 um, and that is a claim about microns.
    source_mpp: float

    @property
    def is_held_out_institution(self) -> bool:
        return self.institution.upper() in classes.TEST_INSTITUTIONS


# --- finding the regions ------------------------------------------------------


def _slide_resolutions() -> dict[str, float]:
    """Per-slide measured mpp, read from approach 1's download sidecar."""
    if not paths.BCSS_SLIDE_MPP.is_file():
        raise FileNotFoundError(
            f"{paths.BCSS_SLIDE_MPP} is missing, so BCSS's real resolutions are unknown. "
            "Exporting without it would mean assuming 0.25 um/px, which is wrong for "
            "every slide. Run approach 1's scripts/01_download.py --route girder."
        )
    raw = json.loads(paths.BCSS_SLIDE_MPP.read_text(encoding="utf-8"))
    # The sidecar has been written in two shapes across runs: a flat mapping, or one
    # nested under a key. Accept either rather than depending on which ran last.
    if isinstance(raw, dict) and "slides" in raw and isinstance(raw["slides"], dict):
        raw = raw["slides"]
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            value = value.get("mpp") or value.get("mm_x") or value.get("mpp_x")
        if value:
            out[str(key).upper()] = float(value)
    return out


def find_bcss_regions() -> list[Region]:
    """The 150 BCSS regions, image paired with mask, with each slide's measured mpp."""
    resolutions = _slide_resolutions()
    regions: list[Region] = []
    orphans: list[str] = []

    for image in sorted(paths.BCSS_IMAGES.iterdir()):
        if image.suffix.lower() not in {".png", ".tif", ".tiff"}:
            continue
        mask = paths.BCSS_MASKS / image.name
        if not mask.exists():
            orphans.append(image.name)
            continue

        slide_id, institution = classes.parse_bcss(image.name)
        key = classes.slide_key(image.stem)
        if key not in resolutions:
            raise KeyError(
                f"{image.name} belongs to slide {key}, whose resolution is not in "
                f"{paths.BCSS_SLIDE_MPP.name}. Finish the download rather than guessing "
                "at its scale."
            )
        regions.append(Region(
            roi_id=image.stem, patient=slide_id, institution=institution,
            source=classes.SOURCE_BCSS, image=image, mask=mask,
            source_mpp=resolutions[key],
        ))

    if orphans:
        raise FileNotFoundError(
            f"{len(orphans)} BCSS image(s) have no mask, e.g. {orphans[:3]}. A partial "
            "download that silently trains on two thirds of the dataset is worse than "
            "one that stops."
        )
    return regions


def find_dcis_regions() -> list[Region]:
    """The 120 BRACS regions with BEETLE-drawn masks, already at the working resolution.

    Their `source_mpp` comes from the sidecar the exporter that made them wrote, not from
    a constant here - the same discipline BCSS's per-slide resolutions get.
    """
    if not paths.DCIS_MANIFEST.is_file():
        return []

    sidecar = json.loads(paths.DCIS_MANIFEST.read_text(encoding="utf-8"))
    source_mpp = float(sidecar["source_mpp"])

    regions: list[Region] = []
    orphans: list[str] = []
    for image in sorted(paths.DCIS_IMAGES.iterdir()):
        if image.suffix.lower() not in {".png", ".tif", ".tiff"}:
            continue
        mask = paths.DCIS_MASKS / image.name
        if not mask.exists():
            orphans.append(image.name)
            continue
        patient, institution = classes.parse_dcis(image.name)
        regions.append(Region(
            roi_id=image.stem, patient=patient, institution=institution,
            source=classes.SOURCE_DCIS, image=image, mask=mask, source_mpp=source_mpp,
        ))

    if orphans:
        raise FileNotFoundError(
            f"{len(orphans)} DCIS image(s) have no mask, e.g. {orphans[:3]}."
        )
    return regions


def find_regions(*, include_dcis: bool = True) -> list[Region]:
    regions = find_bcss_regions()
    if include_dcis:
        regions += find_dcis_regions()
    return regions


# --- one region ---------------------------------------------------------------


def _atomic_save(image: Image.Image, path: Path) -> None:
    """Write to a temporary name, then rename. Atomic on Windows and POSIX alike.

    Without this, a run killed mid-write leaves a truncated PNG that the next run's
    existence check would accept, and training would fail on a corrupt file hours later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    # `format` explicitly, because PIL infers it from the extension and `.png.partial`
    # has none it knows. Passing it is also the honest thing: the format is a decision
    # here, not something to be read back off a filename.
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def shard_path(roi_id: str) -> Path:
    return paths.SHARDS_DIR / f"{roi_id}.json"


def shard_is_current(roi_id: str, spec: TileSpec) -> bool:
    """Whether this region can be skipped: shard present, settings match, files there."""
    path = shard_path(roi_id)
    if not path.is_file():
        return False
    try:
        shard = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False  # truncated by an interrupted run: redo it

    if shard.get("settings_fingerprint") != spec.fingerprint():
        return False
    for row in shard.get("rows", []):
        for key in ("image_path", "mask_path"):
            if not (paths.TILES_DIR / row[key]).is_file():
                return False
    return True


def export_region(region: Region, *, spec: TileSpec = DEFAULT_SPEC) -> dict:
    """Cut one region into `(image, mask)` pairs and write its shard. Returns the shard.

    Order of operations, none of it interchangeable:

      1. resample to the working resolution **in intensity space, before the logarithm**,
         area-averaged, with the mask by nearest neighbour;
      2. remap the 22 codes onto our three classes plus `IGNORE`;
      3. one white point over the whole region, then one deconvolution pass;
      4. cut the grid, keep what is labelled enough, write both arrays.
    """
    rgb = np.asarray(Image.open(region.image).convert("RGB"))
    mask_raw = np.asarray(Image.open(region.mask))

    if mask_raw.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"{region.roi_id}: image is {rgb.shape[:2]} and mask is {mask_raw.shape[:2]}. "
            "A mask that does not cover its image would be read off the wrong pixels."
        )

    rgb, mask_raw = hstain.resample(
        rgb, mask_raw, source_mpp=region.source_mpp, target_mpp=spec.mpp
    )
    remapped = classes.remap(mask_raw)

    # One white point per region and one deconvolution pass over the whole thing, so
    # neighbouring tiles share an `I0` - which they must, being the same physical section.
    white = hstain.white_point(rgb, percentile=spec.white_percentile)
    h_od = hstain.haematoxylin_od(rgb, white)

    height, width = h_od.shape
    rows: list[dict] = []
    skipped_sparse = 0

    for row_index, y in enumerate(
        range(0, max(1, height - spec.tile_px + 1), spec.stride)
    ):
        for col_index, x in enumerate(
            range(0, max(1, width - spec.tile_px + 1), spec.stride)
        ):
            window = (slice(y, y + spec.tile_px), slice(x, x + spec.tile_px))
            mask_tile = remapped[window]
            if mask_tile.shape != (spec.tile_px, spec.tile_px):
                # A partial tile at the right or bottom edge. Dropped rather than padded:
                # padding invents pixels and the loss would train on them.
                continue

            labelled = mask_tile != classes.IGNORE
            labelled_share = float(labelled.mean())
            if labelled_share < spec.min_labelled:
                skipped_sparse += 1
                continue

            tile_id = f"{region.roi_id}__r{row_index:03d}c{col_index:03d}"
            image_rel = Path("images") / region.source / f"{tile_id}.png"
            mask_rel = Path("masks") / region.source / f"{tile_id}.png"

            stored = hstain.quantise(h_od[window])
            _atomic_save(Image.fromarray(stored, mode="L"), paths.TILES_DIR / image_rel)
            _atomic_save(Image.fromarray(mask_tile, mode="L"), paths.TILES_DIR / mask_rel)

            counts = {
                name: int((mask_tile == index).sum())
                for index, name in enumerate(classes.CLASS_NAMES)
            }
            rows.append({
                "tile_id": tile_id,
                "roi_id": region.roi_id,
                "patient": region.patient,
                "institution": region.institution,
                "source": region.source,
                "is_held_out_institution": region.is_held_out_institution,
                "image_path": str(image_rel).replace("\\", "/"),
                "mask_path": str(mask_rel).replace("\\", "/"),
                "labelled_share": round(labelled_share, 4),
                **{f"px_{name}": count for name, count in counts.items()},
                "px_unlabelled": int((~labelled).sum()),
                "x": int(x),
                "y": int(y),
                "tile_px": spec.tile_px,
                "mpp": spec.mpp,
                "source_mpp": round(region.source_mpp, 5),
                "i0_rule": f"p{spec.white_percentile:g}_per_channel_over_region",
            })

    shard = {
        "roi_id": region.roi_id,
        "patient": region.patient,
        "source": region.source,
        "settings_fingerprint": spec.fingerprint(),
        "spec": spec.descriptor(),
        "kept": len(rows),
        "skipped_too_sparse": skipped_sparse,
        "rows": rows,
    }
    # Written last and atomically: the shard is the record that this region is done, so it
    # must not exist until its tiles do.
    temporary = shard_path(region.roi_id).with_suffix(".json.partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(shard, indent=1), encoding="utf-8")
    os.replace(temporary, shard_path(region.roi_id))
    return shard


def export_all(
    regions: list[Region],
    *,
    spec: TileSpec = DEFAULT_SPEC,
    force: bool = False,
    progress: Progress = _noop,
) -> Iterator[tuple[Region, dict, bool]]:
    """Export every region, skipping those already current. Yields `(region, shard, redone)`."""
    for index, region in enumerate(regions, 1):
        if not force and shard_is_current(region.roi_id, spec):
            shard = json.loads(shard_path(region.roi_id).read_text(encoding="utf-8"))
            progress(f"[{index}/{len(regions)}] {region.roi_id}: cached ({shard['kept']})")
            yield region, shard, False
            continue

        shard = export_region(region, spec=spec)
        progress(
            f"[{index}/{len(regions)}] {region.roi_id}: {shard['kept']} tiles"
            f" ({shard['skipped_too_sparse']} too sparse)"
        )
        yield region, shard, True


# --- the combined manifest ----------------------------------------------------


def read_manifest() -> list[dict]:
    path = paths.TILES_DIR / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/01_export_seg_tiles.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def rebuild_manifest(spec: TileSpec = DEFAULT_SPEC) -> dict:
    """Combine every current shard into one manifest, sorted so the order is stable.

    Rebuilt from the shards rather than accumulated during the run, which is what makes a
    resumed export produce byte-identical output to an uninterrupted one. Shards whose
    fingerprint does not match the current spec are ignored rather than mixed in.
    """
    rows: list[dict] = []
    regions = 0
    stale = 0
    for path in sorted(paths.SHARDS_DIR.glob("*.json")):
        try:
            shard = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stale += 1
            continue
        if shard.get("settings_fingerprint") != spec.fingerprint():
            stale += 1
            continue
        rows.extend(shard["rows"])
        regions += 1

    rows.sort(key=lambda row: row["tile_id"])
    if not rows:
        raise ValueError(
            "no tiles in any current shard. Either the export has not run, or every "
            "shard was cut under different settings than the spec now asks for."
        )

    per_class = {
        name: sum(int(row[f"px_{name}"]) for row in rows)
        for name in classes.CLASS_NAMES
    }
    by_source: dict[str, dict] = {}
    for row in rows:
        entry = by_source.setdefault(row["source"], {"tiles": 0, **{n: 0 for n in classes.CLASS_NAMES}})
        entry["tiles"] += 1
        for name in classes.CLASS_NAMES:
            entry[name] += int(row[f"px_{name}"])

    manifest = {
        "spec": spec.descriptor(),
        "settings_fingerprint": spec.fingerprint(),
        "regions": regions,
        "stale_shards_ignored": stale,
        "tiles": len(rows),
        "patients": len({row["patient"] for row in rows}),
        "pixels_per_class": per_class,
        "pixels_unlabelled": sum(int(row["px_unlabelled"]) for row in rows),
        "by_source": by_source,
        "labels_drawn_by": {
            classes.SOURCE_BCSS: "pathologists",
            classes.SOURCE_DCIS: "BEETLE's nnU-Net, reviewed by nobody",
        },
        "rows": rows,
    }
    path = paths.TILES_DIR / "manifest.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(temporary, path)
    return manifest


def fingerprint_rows(rows: list[dict]) -> str:
    """A hash of the tile order, so nothing downstream can pair with the wrong manifest."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["tile_id"].encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()
