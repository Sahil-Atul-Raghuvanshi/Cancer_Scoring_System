"""The tile exporter: annotated regions in, labelled haematoxylin tiles out.

One exporter, and every source goes through it - BCSS now, AICAN pseudo-labels when
an H&E slide exists, hand-clicked tiles if anyone ever draws them. That is the point
of it being one function: two exporters would be two definitions of what a training
tile is, and the second one would be discovered by accident.

The order of operations is not negotiable, and each step is here because doing it in
a different order is wrong rather than merely different:

  1  resample to the working resolution **in intensity space, before the logarithm**,
     with area averaging. A coarser sensor averages the light arriving over a larger
     area, so averaging transmissions is what a coarser scan physically *is*.
     Averaging densities instead takes the mean of logarithms - the log of a geometric
     mean - which is biased low and is a number no instrument records. Step 5's
     `read_tile` says the same thing about slides; this says it about regions.
  2  resample the **mask by nearest neighbour**. A label is not an intensity.
     Interpolating one produces classes that were never annotated, and between
     `tumor` (1) and `stroma` (2) the interpolated value is `dcis`-adjacent nonsense.
  3  the white point, then optical density, then deconvolution - `hchannel`'s job,
     not this module's.
  4  cut the grid, vote, and drop what is too mixed to learn from.

Every tile that survives gets a row in the manifest carrying its provenance: which
region, which slide, which hospital, which vote fractions, which white-point rule.
A question about a tile three notebooks later is then answerable without re-running
anything, which is the difference between a dataset and a directory of PNGs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

import bcss
import hchannel
import tilestore

# --- the geometry -------------------------------------------------------------


@dataclass(frozen=True)
class TileSpec:
    """The tile geometry, in the units each number is a statement about.

    `tile_px` and `mpp` together are the field of view, and the field of view is the
    whole argument for this design: 224 px at 0.5 um/px is 112 um, about nine cells
    across, which is enough to see whether epithelium sits inside a duct or has
    infiltrated. Smaller tiles literally cannot contain that information, which is
    why nuclear-density heuristics never separated DCIS from invasive carcinoma.

    `stride` is `tile_px` - no overlap at training time. Overlapping training tiles
    are near-duplicates: they inflate the apparent dataset size, and if a split ever
    fell between two overlapping tiles they would leak into each other. Inference
    overlaps (step 7's `tiling_overlap`) because there the goal is a smooth class map,
    which is a different goal.
    """

    tile_px: int = 224
    mpp: float = 0.5
    stride_px: int | None = None

    #: Vote thresholds, exactly as the plan states them.
    min_usable: float = 0.70
    min_invasive: float = 0.50
    min_non_invasive: float = 0.50
    min_non_epithelium: float = 0.80

    #: Percentile of each channel taken as `I0` on a region with no glass in it.
    white_percentile: float = 99.0

    @property
    def stride(self) -> int:
        return self.stride_px if self.stride_px is not None else self.tile_px

    @property
    def tile_um(self) -> float:
        return self.tile_px * self.mpp

    def descriptor(self) -> dict[str, object]:
        """The spec as JSON, for the export manifest."""
        return {**asdict(self), "stride": self.stride, "tile_um": self.tile_um}


DEFAULT_SPEC = TileSpec()


# --- the vote -----------------------------------------------------------------

#: Why a tile was thrown away. Counted by reason and reported, because "14,000 tiles
#: kept" without "and 9,000 dropped, 6,000 of them unlabelled background" is not a
#: number anyone can check.
DROP_UNUSABLE = "unusable"
DROP_MIXED = "mixed"
DROP_REASONS = (DROP_UNUSABLE, DROP_MIXED)


@dataclass(frozen=True)
class Vote:
    """The label a tile's mask voted for, and the fractions behind it."""

    label: int | None
    #: Share of the tile's pixels carrying any label at all.
    usable: float
    #: Shares **of the usable pixels**, not of the tile. A tile that is 80% outside
    #: the annotated region and 20% tumour is 100% tumour where it is annotated at
    #: all, and treating it as 20% tumour would drop exactly the tiles at the edge of
    #: an annotation - which is where the interesting boundaries are.
    invasive: float
    non_invasive: float
    non_epithelium: float
    dropped_by: str | None


def vote(remapped: np.ndarray) -> Vote:
    """Majority vote over a remapped mask tile, with the ambiguous discarded.

    The thresholds are asymmetric on purpose. Half the usable pixels is enough to
    call a tile epithelial, because epithelium in a tile is what the model needs to
    find; but 80% is required to call a tile non-epithelium, because a tile that is
    60% stroma and 40% tumour is a tumour tile with stroma in it, and teaching the
    model to call it stroma would erode every tumour boundary.

    Anything that clears none of the three is dropped rather than forced into a
    class. A tile that is genuinely half duct and half stroma has no majority answer,
    and inventing one is how a model learns that the boundary is its own class.
    """
    total = int(remapped.size)
    labelled = remapped != bcss.IGNORE
    usable_count = int(labelled.sum())
    usable = usable_count / total if total else 0.0

    if usable < DEFAULT_SPEC.min_usable or usable_count == 0:
        return Vote(None, usable, 0.0, 0.0, 0.0, DROP_UNUSABLE)

    labels = remapped[labelled]
    invasive = float(np.mean(labels == bcss.INVASIVE))
    non_invasive = float(np.mean(labels == bcss.NON_INVASIVE))
    non_epithelium = float(np.mean(labels == bcss.NON_EPITHELIUM))

    if invasive >= DEFAULT_SPEC.min_invasive:
        label: int | None = bcss.INVASIVE
    elif non_invasive >= DEFAULT_SPEC.min_non_invasive:
        label = bcss.NON_INVASIVE
    elif non_epithelium >= DEFAULT_SPEC.min_non_epithelium:
        label = bcss.NON_EPITHELIUM
    else:
        label = None

    return Vote(
        label=label,
        usable=usable,
        invasive=invasive,
        non_invasive=non_invasive,
        non_epithelium=non_epithelium,
        dropped_by=None if label is not None else DROP_MIXED,
    )


# --- resampling ---------------------------------------------------------------


def resample(
    rgb: np.ndarray, mask: np.ndarray, *, source_mpp: float, target_mpp: float
) -> tuple[np.ndarray, np.ndarray]:
    """Region and mask to the working resolution, each by its correct filter.

    BCSS ships at 0.25 um/px and the pipeline works at 0.5, so this is normally a 2x
    downsample. The factor is computed from the two resolutions rather than assumed,
    because a mirror that reshipped the regions at a different scale would otherwise
    produce tiles at the wrong field of view - and a 224 px tile at 0.25 um/px is 56
    um, half the cells and none of the architecture.
    """
    if source_mpp <= 0 or target_mpp <= 0:
        raise ValueError("resolutions are microns per pixel and must be positive")

    factor = target_mpp / source_mpp
    if abs(factor - 1.0) < 1e-6:
        return np.asarray(rgb), np.asarray(mask)

    height, width = rgb.shape[:2]
    new_size = (max(1, int(round(width / factor))), max(1, int(round(height / factor))))

    # BOX = area averaging, on the intensities, before any logarithm. See the module
    # docstring, and step 5's `read_tile` for the same decision on slide pixels.
    rgb_small = np.asarray(
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize(new_size, Image.Resampling.BOX)
    )
    # NEAREST on the labels. Never BOX, never BILINEAR: an averaged label is a class
    # nobody annotated.
    mask_small = np.asarray(
        Image.fromarray(np.asarray(mask, dtype=np.uint8)).resize(new_size, Image.Resampling.NEAREST)
    )
    return rgb_small, mask_small


# --- one region ---------------------------------------------------------------


@dataclass
class ExportedTile:
    """One kept tile: the pixels to write, and the row to record."""

    stored: np.ndarray
    row: dict[str, object]


def export_region(
    region: bcss.Region,
    *,
    spec: TileSpec = DEFAULT_SPEC,
    source_mpp: float,
    source: str = "bcss",
    variant: tilestore.Variant = tilestore.H_CHANNEL,
) -> tuple[list[ExportedTile], dict[str, int]]:
    """Cut one BCSS region into labelled tiles, stored as `variant` says.

    `variant` decides only what the stored pixels are - haematoxylin density, or the
    colour photograph. **It cannot change which tiles are kept or what they are
    labelled**, because the grid and the vote below read `remapped` and nothing else:
    the white point, the deconvolution and the quantisation all happen after the vote
    has been taken. That is deliberate and it is load-bearing. It means the two stores
    for one field of view hold the same `tile_id`s with the same labels, so a model
    fitted on one can be compared against a model fitted on the other tile by tile,
    with no alignment step and nothing to get wrong.

    `source_mpp` is this **region's own** resolution and has no default, deliberately.
    Every filename in this dataset ends `MPP-0.2500` and the slides are not 0.25 - the
    first one measures 0.2521, and TCGA scans vary - so a default here would be an
    assumption that silently puts tiles at the wrong physical size. A 224 px tile is
    meant to cover 112 um because that is nine cells across, and nine cells is a claim
    about microns. `download_bcss.resolutions` is where the measured values come from.

    Returns the kept tiles and a count of drops by reason. Nothing is written here -
    the caller writes, so this function is testable on a synthetic region without a
    filesystem.
    """
    rgb = np.asarray(Image.open(region.image).convert("RGB"))
    mask_raw = np.asarray(Image.open(region.mask))

    if mask_raw.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"{region.roi_id}: image is {rgb.shape[:2]} and mask is {mask_raw.shape[:2]}. "
            "A mask that does not cover its image cannot be voted on - the labels "
            "would be read off the wrong pixels."
        )

    rgb, mask_raw = resample(rgb, mask_raw, source_mpp=source_mpp, target_mpp=spec.mpp)
    remapped = bcss.remap(mask_raw)

    # One pass over the whole region rather than one per tile: the transform is per
    # pixel, so doing it once is the same arithmetic done fewer times, and on the
    # haematoxylin variant it guarantees neighbouring tiles share a white point -
    # which they must, being the same physical section. On the RGB variant this is the
    # identity, and the cost of calling it anyway is one `asarray`.
    plane = variant.prepare(rgb, spec)

    height, width = plane.shape[:2]
    kept: list[ExportedTile] = []
    drops = {reason: 0 for reason in DROP_REASONS}

    for row_index, y in enumerate(range(0, max(1, height - spec.tile_px + 1), spec.stride)):
        for col_index, x in enumerate(range(0, max(1, width - spec.tile_px + 1), spec.stride)):
            window = (slice(y, y + spec.tile_px), slice(x, x + spec.tile_px))
            label_tile = remapped[window]
            if label_tile.shape != (spec.tile_px, spec.tile_px):
                # A partial tile at the right or bottom edge. Dropped rather than
                # padded: padding invents pixels and the vote would count them.
                continue

            decision = vote(label_tile)
            if decision.label is None:
                drops[decision.dropped_by or DROP_MIXED] += 1
                continue

            pixels = plane[window]
            kept.append(
                ExportedTile(
                    stored=variant.store(pixels),
                    row={
                        "tile_id": f"{region.roi_id}__r{row_index:03d}c{col_index:03d}",
                        "roi_id": region.roi_id,
                        "slide_id": region.slide_id,
                        "institution": region.institution,
                        "is_test_institution": region.is_test,
                        "label": int(decision.label),
                        "label_name": bcss.CLASS_NAMES[decision.label],
                        "usable": round(decision.usable, 4),
                        "invasive_frac": round(decision.invasive, 4),
                        "non_invasive_frac": round(decision.non_invasive, 4),
                        "non_epithelium_frac": round(decision.non_epithelium, 4),
                        **variant.stats(pixels),
                        "x": int(x),
                        "y": int(y),
                        # Which dataset, and therefore whether a person drew this
                        # label or a model did. Every split and every report keeps the
                        # two apart on this column alone - the tile files themselves
                        # are indistinguishable.
                        "source": source,
                        # Which store this row belongs to, twice over: the contract's
                        # name and the encoding's. Redundant on purpose - a manifest
                        # read years from now has to answer without its siblings.
                        "channel": variant.channel,
                        "variant": variant.stored_as,
                        "i0_rule": variant.i0_rule.format(**asdict(spec)),
                        "tile_px": spec.tile_px,
                        "mpp": spec.mpp,
                        "source_mpp": round(source_mpp, 5),
                    },
                )
            )

    return kept, drops


def write_region(
    region: bcss.Region,
    out_dir: Path,
    *,
    spec: TileSpec = DEFAULT_SPEC,
    source_mpp: float,
    source: str = "bcss",
    variant: tilestore.Variant = tilestore.H_CHANNEL,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """`export_region`, with the tiles written as 8-bit PNGs. Returns the rows."""
    tiles, drops = export_region(
        region, spec=spec, source_mpp=source_mpp, source=source, variant=variant
    )

    # One directory per class, so a human can open a folder and check the labels by
    # eye - which is gate G2's most valuable test and the one a manifest cannot do.
    rows: list[dict[str, object]] = []
    for tile in tiles:
        label_dir = Path(out_dir) / bcss.CLASS_NAMES[int(tile.row["label"])]
        label_dir.mkdir(parents=True, exist_ok=True)
        path = label_dir / f"{tile.row['tile_id']}.png"
        image = (
            Image.fromarray(tile.stored)
            if variant.pil_mode is None
            else Image.fromarray(tile.stored, mode=variant.pil_mode)
        )
        image.save(path, optimize=True)
        rows.append({**tile.row, "tile_path": str(path.relative_to(Path(out_dir)))})

    return rows, drops


def iter_regions(
    regions: list[bcss.Region],
    out_dir: Path,
    *,
    spec: TileSpec = DEFAULT_SPEC,
    source_mpp: float | dict[str, float],
    source: str = "bcss",
    variant: tilestore.Variant = tilestore.H_CHANNEL,
) -> Iterator[tuple[bcss.Region, list[dict[str, object]], dict[str, int]]]:
    """Export every region, one at a time, yielding progress as it goes.

    A generator rather than a list because the export is an hour long and a notebook
    that shows nothing for an hour is indistinguishable from a hung one.
    """
    for region in regions:
        mpp = (source_mpp[region.slide_key]
               if isinstance(source_mpp, dict) else source_mpp)
        rows, drops = write_region(region, out_dir, spec=spec, source_mpp=mpp,
                                   source=source, variant=variant)
        yield region, rows, drops


def write_manifest(
    rows: list[dict[str, object]],
    drops: dict[str, int],
    out_dir: Path,
    *,
    spec: TileSpec = DEFAULT_SPEC,
    extra: dict[str, object] | None = None,
    variant: tilestore.Variant = tilestore.H_CHANNEL,
) -> Path:
    """The manifest: every kept tile's row, plus what was dropped and the geometry.

    CSV, and a JSON sidecar for the summary. CSV because it is the format that opens
    in anything and cannot bit-rot behind a library version, which matters for a file
    whose whole job is to still be readable when someone asks where a tile came from.
    """
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = out_dir / "tiles_manifest.csv"
    if not rows:
        raise ValueError(
            "no tiles were kept from any region. Either the mask mapping is wrong or "
            "the vote thresholds are - check gate G1 before believing the exporter."
        )

    fields = list(rows[0].keys())
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    labels = [int(row["label"]) for row in rows]
    summary = {
        "tiles_kept": len(rows),
        "dropped": drops,
        "per_class": {
            bcss.CLASS_NAMES[cls]: int(sum(1 for label in labels if label == cls))
            for cls in (bcss.NON_EPITHELIUM, bcss.NON_INVASIVE, bcss.INVASIVE)
        },
        "slides": len({row["slide_id"] for row in rows}),
        "institutions": sorted({str(row["institution"]) for row in rows}),
        "spec": spec.descriptor(),
        "input": variant.descriptor(
            tile_px=spec.tile_px, mpp=spec.mpp, invert=False
        ),
        **(extra or {}),
    }
    (out_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
