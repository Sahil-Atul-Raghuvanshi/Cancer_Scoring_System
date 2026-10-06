"""Turn BC-DeepLIIF into a labelled breast-IHC test set (all detectors) and a train set.

BC-DeepLIIF is breast-cancer Ki-67 IHC whose cell masks come from co-registered
multiplex immunofluorescence of the same section, not from a person drawing on the IHC -
so a nucleus buried under brown is still labelled. That is exactly what LyNSeC (lymphoma)
could not show us, and what our own slides have no labels for.

Each file is one 3072x512 strip of six 512 px panels, left to right: IHC, haematoxylin,
DAPI, Lap2, marker, Seg. In Seg, red = positive cell, blue = negative cell, green = the
outline drawn round each, black = background. An instance is one red or blue core plus the
green ring round it (green pixels within `RING_PX` of a core go to the nearest core).

Files `N_k.png` are flipped/rotated (and lightly warped) copies of `N.png` - measured: each
is within a few grey levels of a flip or rotation of the base - so the set is only 41
distinct tiles (6 validation, 35 training), ~170 nuclei each. A group never straddles the
split. Test = every validation group plus training groups up to `TEST_GROUPS`, base image
only; train = the remaining groups with all their variants (augmentation the trainer
would otherwise do itself). The pixel size is measured from the median nucleus area, as for LyNSeC,
then everything is brought to 0.5 um/px.
"""

from __future__ import annotations

import io
import re
import sys
import zipfile

import numpy as np
from scipy import ndimage

import common

LOG = common.LOGS / "bcdl_prep.log"
PANEL = 512
RING_PX = 6
MIN_PX = 20
NUCLEUS_UM2 = 40.0  # breast epithelial nucleus, order of magnitude only: chooses 0.25 vs 0.5
TEST_GROUPS = 16


def instances(seg: np.ndarray) -> np.ndarray:
    r, g, b = (seg[..., k].astype(np.int16) for k in range(3))
    core = ((r > 128) & (g < 100) & (b < 100)) | ((b > 128) & (r < 100) & (g < 100))
    ring = (g > 128) & (r < 100) & (b < 100)
    labels, _ = ndimage.label(core)
    distance, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    grow = ring & (distance <= RING_PX)
    labels[grow] = labels[iy[grow], ix[grow]]
    sizes = np.bincount(labels.ravel())
    labels[sizes[labels] < MIN_PX] = 0
    _, labels = np.unique(labels, return_inverse=True)
    return labels.reshape(seg.shape[:2]).astype(np.int32)


def groups(archive: zipfile.ZipFile) -> dict[str, list[str]]:
    """Group id -> its member names, base image (`N.png`) first."""
    found: dict[str, list[tuple[int, str]]] = {}
    for member in archive.namelist():
        match = re.search(r"(?:^|/)(\d+)(?:_(\d+))?\.png$", member)
        if match:
            found.setdefault(match.group(1), []).append((-1 if match.group(2) is None else int(match.group(2)), member))
    return {gid: [m for _, m in sorted(members)] for gid, members in found.items()}


def main() -> int:
    from PIL import Image

    common.ensure_dirs()
    if not all((common.DOWNLOADS / f"{n}.ok").exists() for n in ("bcdl_val", "bcdl_train")):
        common.say("BC-DeepLIIF not downloaded yet", LOG)
        return 2
    if (common.BCDL / "meta.json").exists():
        return 0

    val = zipfile.ZipFile(common.DOWNLOADS / "BC-DeepLIIF_Validation_Set.zip")
    train = zipfile.ZipFile(common.DOWNLOADS / "BC-DeepLIIF_Training_Set.zip")
    val_groups, train_groups = groups(val), groups(train)

    def load(archive, member):
        strip = np.asarray(Image.open(io.BytesIO(archive.read(member))).convert("RGB"))
        if strip.shape[1] != 6 * PANEL or strip.shape[0] != PANEL:
            raise RuntimeError(f"{member}: expected a 6-panel {6 * PANEL}x{PANEL} strip, got {strip.shape}")
        return strip[:, :PANEL], instances(strip[:, 5 * PANEL:])

    # Pixel size from the validation set's nucleus areas.
    areas = []
    for member in [members[0] for members in val_groups.values()]:
        _, inst = load(val, member)
        counts = np.bincount(inst.ravel())[1:]
        areas.extend(counts[counts > 0].tolist())
    area_px = float(np.median(areas))
    raw_mpp = (NUCLEUS_UM2 / area_px) ** 0.5
    native = 0.25 if raw_mpp < 0.375 else 0.5
    scale = native / 0.5
    common.say(f"{len(val_groups)} validation groups, {len(train_groups)} training groups; median nucleus "
               f"{area_px:.0f} px -> ~{raw_mpp:.3f} um/px, native {native}, rescaled to 0.5 um/px", LOG)

    rng = np.random.default_rng(common.SEED)
    train_ids = sorted(train_groups, key=int)
    order = [train_ids[i] for i in rng.permutation(len(train_ids))]
    extra = max(0, TEST_GROUPS - len(val_groups))
    split = {
        "test": [(val, f"v{g}", val_groups[g][0]) for g in sorted(val_groups, key=int)]
        + [(train, f"t{g}", train_groups[g][0]) for g in order[:extra]],
        "train": [(train, f"t{g}_{k}", member) for g in order[extra:]
                  for k, member in enumerate(train_groups[g])],
    }
    meta = {"native_mpp": native, "estimated_mpp": round(raw_mpp, 3), "mpp": 0.5,
            "median_area_px_native": area_px, "train": [], "test": []}
    for part, items in split.items():
        (common.BCDL / part).mkdir(parents=True, exist_ok=True)
        for k, (archive, name, member) in enumerate(items):
            rgb, inst = load(archive, member)
            if scale != 1.0:
                size = (int(round(PANEL * scale)),) * 2
                rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BOX))
                inst = np.asarray(Image.fromarray(inst).resize(size, Image.Resampling.NEAREST))
            Image.fromarray(rgb).save(common.BCDL / part / f"{name}.png")
            np.savez_compressed(common.BCDL / part / f"{name}_inst.npz", labels=inst.astype(np.int32))
            meta[part].append({"name": name, "source": member, "nuclei": int(len(np.unique(inst)) - 1)})
            if k % 25 == 0:
                common.heartbeat("bcdl_prep", f"{part} {k}/{len(items)}")
    common.write_json(common.BCDL / "meta.json", meta)
    common.say(f"train {len(meta['train'])}, test {len(meta['test'])}", LOG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
