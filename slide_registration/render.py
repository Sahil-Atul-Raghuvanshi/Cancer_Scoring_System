"""Phase 1 - render every slide of a case onto one common physical grid.

    python render.py CAN_00303 [--mpp 8.0]

**Why this exists.** VALIS reads slides through their own pyramids and then normalises
every image in a run down to the weakest one. `Valis.check_img_max_dims()` silently sets
`max_image_dim_px` to the long edge of the smallest image it could read, and then drags
`max_processed_image_dim_px` down to match. Our pyramid ladder is 1, 4, 8, 16, 32, 64,
128, 256 with no 2x step and a depth that varies from 6 to 8 across slides, so the
smallest readable level on some slide lands near 496 px - and every registration this
project has ever run has therefore happened at 512 px, about 27.6 um per processed pixel,
while `registration_max_dim_px = 2048` sat in the config being described as the single
biggest accuracy lever. Rendering all six slides ourselves, to a resolution we choose,
takes that decision away from VALIS entirely.

**The three things a render has to get right:**

*   *One physical scale.* Every slide of a case lands at exactly the same um/px, so the
    matcher never has to absorb a scale difference. Pairwise VALIS never guaranteed this,
    and two of six cases have the H&E on a different canvas from the IHC slides.
*   *Tissue, found by absorbance.* The pipeline's own mask keys off colour saturation,
    which measures how *stained* a slide is rather than whether tissue is present. On
    this cohort it captured 4% of the tissue on 00865/A and 13% on 00267/A - the pale
    CD44 sections, which are exactly the ones registration fails on. Optical density is a
    physical quantity a near-blank section still has.
*   *An invertible transform.* Everything downstream is expressed in the slide's own
    level-0 pixels, so the exact map from level-0 to render pixels is written beside the
    image and never recomputed by eye.

The canvas padding is separated from glass by *flatness*, not brightness: scanner padding
is memset to a single value and carries zero local variance, while glass carries sensor
noise. Measured on these files padding sits at grey 146 and glass at 191-195, so a
brightness cut that clears the glass also swallows the padding - and on a pale slide, the
tissue with it.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tissue_scoring_demo" / "backend"))

import common  # noqa: E402

#: Microns per pixel every slide is rendered to. 8 um/px puts these specimens
#: (35-315 mm2) at roughly 2000-4000 px on the long edge - about 3.5x finer than the
#: 27.6 um/px everything has actually run at so far, and small enough that a non-rigid
#: pass over six of them is minutes rather than hours. It is a parameter because 4 um/px
#: is the stretch target if the night has room for it.
DEFAULT_MPP = 8.0

#: Local standard deviation below which a pixel is scanner padding rather than glass.
#: Padding is memset to one value; glass is noisy. Measured separation is large - padding
#: sits at essentially 0 and glass at 1-3 grey levels - so this needs no tuning.
PADDING_STD = 0.35

#: Optical density bounds for the tissue cut. Otsu chooses within them.
#:
#: **The floor only has to clear glass, not padding**, because padding is already gone by
#: the time the cut is chosen - it is removed by flatness, above. Getting that wrong cost
#: an hour: a floor of 0.28, set to clear padding at OD 0.24, measured CAN_00865's
#: near-blank CD44 section at 0.05 mm2 against a documented 17.5, and CAN_00267's pale
#: H&E at 4.6 mm2 against 163.5. Otsu was choosing a sensible cut and being clamped up
#: past the tissue. Glass measures OD 0.12 on these files, so 0.14 clears it with room to
#: spare and leaves every pale section in play.
#:
#: The ceiling stops a strongly stained slide from cutting so high that it keeps only the
#: darkest tissue and calls the rest glass.
OD_FLOOR, OD_CEILING = 0.14, 0.60

#: Optical density below which a pixel is bare glass rather than anything at all.
#: Measured at 0.12 on these files. Used only by the permissive `visible` mask that
#: decides what the feature matcher is shown - never to measure tissue.
GLASS_OD = 0.06

#: Smallest tissue island kept, in mm2. Below this it is debris, a bubble edge or a
#: fragment of the label, none of which should move the bounding box.
#:
#: 0.5 was too aggressive on fragmented specimens: CAN_00259, the largest and most broken-
#: up case in the cohort, measured 0.41 to 0.64 of its published area on all six slides
#: with that value - a systematic loss, not a stain problem, because the specimen is many
#: genuine small pieces rather than one mass.
MIN_ISLAND_MM2 = 0.1

#: How far the mask is grown before everything outside it is whitened, in microns.
#:
#: **This is deliberately far larger than the mask needs to be**, and the reason is that
#: the mask has two jobs with opposite error costs. For placing the section - bounding
#: box and centroid - a tight mask is right, and missing a pale fringe barely moves a
#: centroid. For deciding what the feature matcher is allowed to see, a tight mask is
#: dangerous: whitening a pale tumour edge because the optical-density cut did not quite
#: reach it destroys exactly the tissue the registration needs, and it does so silently.
#: 200 um is comfortably wider than any fringe measured here and still nowhere near the
#: scanner-padding staircase this is removing.
WHITEN_MARGIN_UM = 200.0

#: Margin left around the tissue in the common canvas, as a share of the tissue's long
#: edge. Non-rigid registration pushes pixels outwards near the border, and a warp that
#: runs into the edge of the canvas is clipped rather than wrong-but-visible.
MARGIN = 0.06


def absorbance(rgb: np.ndarray) -> np.ndarray:
    """How much light the tissue took out, from an RGB image.

    Deliberately not saturation. H&E is pink and purple and saturated; a near-blank IHC
    section is a pale counterstain and almost nothing else, and a saturation rule scores
    it as empty glass. Absorbance is a physical quantity both carry.
    """
    grey = np.asarray(rgb[..., :3], dtype=np.float32).mean(axis=2) / 255.0
    return -np.log10(np.clip(grey, 1e-3, 1.0))


def _otsu(values: np.ndarray, lo: float, hi: float) -> float:
    """Otsu's cut on `values`, clamped into [lo, hi]."""
    if values.size == 0:
        return lo
    histogram, edges = np.histogram(values, bins=256, range=(0.0, max(hi * 2, 1.0)))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return lo
    probability = histogram / total
    centres = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(probability)
    mean = np.cumsum(probability * centres)
    grand = mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (grand * weight - mean) ** 2 / (weight * (1 - weight))
    between[~np.isfinite(between)] = 0
    return float(np.clip(centres[int(np.argmax(between))], lo, hi))


def tissue_mask(rgb: np.ndarray, mpp: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tissue by optical density, with scanner padding excluded by flatness first.

    Returns three things: the strict mask, a permissive `visible` mask, and the numbers
    behind them. Two masks because they answer different questions with opposite error
    costs - the strict one places and measures the section, the permissive one decides
    what the feature matcher is allowed to see. See the comment beside `visible`.

    The numbers are returned because a mask that silently came out empty on a pale
    slide is the failure this whole file exists to avoid.
    """
    optical = absorbance(rgb)

    # Padding first. A uniform filter of the squared signal minus the square of the
    # filtered signal is the local variance, and padding is the only thing on a slide
    # with none at all.
    smooth = ndimage.uniform_filter(optical, size=5)
    variance = ndimage.uniform_filter(optical * optical, size=5) - smooth * smooth
    local_std = np.sqrt(np.clip(variance, 0, None)) * 255.0
    considered = local_std > PADDING_STD

    if considered.sum() < optical.size * 0.01:
        # Nothing looked noisy - almost certainly a render with no padding at all rather
        # than a slide that is entirely padding. Fall back to using everything.
        considered = np.ones_like(considered)

    cut = _otsu(optical[considered], OD_FLOOR, OD_CEILING)
    mask = (optical > cut) & considered

    px_mm2 = (mpp / 1000.0) ** 2
    radius = max(1, int(round(30.0 / mpp)))  # 30 um of closing, in render pixels
    mask = ndimage.binary_closing(mask, np.ones((3, 3)), iterations=radius)
    mask = ndimage.binary_opening(mask, np.ones((3, 3)), iterations=max(1, radius // 2))

    labelled, count = ndimage.label(mask)
    if count:
        sizes = np.asarray(ndimage.sum(mask, labelled, range(1, count + 1)))
        keep = np.where(sizes * px_mm2 >= MIN_ISLAND_MM2)[0] + 1
        mask = np.isin(labelled, keep) if keep.size else (labelled == (sizes.argmax() + 1))
    mask = ndimage.binary_fill_holes(mask)

    # A second, deliberately permissive mask: everything that is not scanner padding and
    # not bare glass. It is not a measurement and must never be used as one - it is only
    # the answer to "what may the feature matcher see".
    #
    # The two have opposite error costs and the strict mask cannot serve both. On a
    # near-negative section the strict mask is *correct* and still almost empty:
    # CAN_00865's CD44 slide measures 14 mm2 here against a published 17.5 while its
    # siblings measure 85-96, because the tissue is plainly there and barely absorbs any
    # light. Whitening everything outside that strict mask would erase the faint tissue
    # the registration needs, and would do it silently - the render would simply look
    # blank and no number would say why.
    visible = (optical > GLASS_OD) & considered
    visible = ndimage.binary_closing(visible, np.ones((3, 3)), iterations=radius)
    visible = ndimage.binary_fill_holes(visible)

    return mask, visible, {
        "odCut": round(cut, 4),
        "odMedianTissue": round(float(np.median(optical[mask])) if mask.any() else 0.0, 4),
        "paddingShare": round(1.0 - float(considered.mean()), 4),
        "tissueMm2": round(float(mask.sum()) * px_mm2, 3),
        "islands": int(count),
    }


def read_at(path: pathlib.Path, target_mpp: float) -> tuple[np.ndarray, float, float]:
    """The whole slide as RGB at approximately `target_mpp`, plus its scale.

    Returns (rgb, base_mpp, actual_mpp_of_this_array). The level is chosen by measured
    downsample, never by index - the ladder has no 2x step and depth varies from 6 to 8
    across these slides, so an index means a different resolution on different files.
    """
    from app.ingestion.slide_reader import open_slide

    reader = open_slide(path)
    base = reader.mpp
    if not base:
        raise ValueError(f"{path.name} records no physical scale, so it cannot be registered")

    level = reader.best_level_for_downsample(target_mpp / base)
    width, height = reader.level_dimensions[level]
    factor = float(reader.level_downsamples[level])
    rgb = np.asarray(reader.read_region_pil((0, 0), level, (width, height)).convert("RGB"))
    return rgb, float(base), float(base) * factor


def render_slide(path: pathlib.Path, target_mpp: float) -> dict:
    """One slide, cropped to its tissue and resampled to exactly `target_mpp`."""
    rgb, base_mpp, level_mpp = read_at(path, target_mpp)
    mask, visible, stats = tissue_mask(rgb, level_mpp)

    if not mask.any():
        raise ValueError(f"{path.name}: no tissue found (od cut {stats['odCut']})")

    # Resample to exactly the target. The level is at or finer than the target by
    # construction, so this only ever downsamples.
    scale = level_mpp / target_mpp
    out_w = max(1, int(round(rgb.shape[1] * scale)))
    out_h = max(1, int(round(rgb.shape[0] * scale)))
    image = Image.fromarray(rgb).resize((out_w, out_h), Image.BILINEAR)
    small = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((out_w, out_h), Image.NEAREST)
    ) > 127
    small_visible = np.asarray(
        Image.fromarray(visible.astype(np.uint8) * 255).resize((out_w, out_h), Image.NEAREST)
    ) > 127

    rows, cols = np.nonzero(small)
    box = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
    centroid = (float(cols.mean()), float(rows.mean()))

    return {
        "rgb": np.asarray(image),
        "mask": small,
        "visible": small_visible,
        "box": box,
        "centroid": centroid,
        "baseMpp": base_mpp,
        "levelMpp": level_mpp,
        # level-0 x -> this array's x is `x * level0ToRender`
        "level0ToRender": base_mpp / target_mpp,
        "stats": stats,
    }


def render_case(case: str, target_mpp: float = DEFAULT_MPP, log=None) -> dict:
    """Every slide of one case onto one common canvas, written to disk.

    The canvas is sized by the largest tissue bounding box across the six slides plus a
    margin, and each slide is placed with its tissue *centroid* at the canvas centre.
    Centroid rather than bounding-box centre because a torn corner or a detached fragment
    moves a bounding box a long way and barely moves a centroid, and these sections do
    both.
    """
    slides = common.slides_for(case)
    if "HE" not in slides:
        raise ValueError(f"{case}: no H&E slide found among {sorted(slides)}")

    out = common.case_dir(case) / "render"
    out.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, dict] = {}
    for code, path in slides.items():
        common.say(f"{case}: rendering {code} ({path.name})", log)
        rendered[code] = render_slide(path, target_mpp)
        common.say(
            f"{case}:   {code} tissue {rendered[code]['stats']['tissueMm2']} mm2, "
            f"od cut {rendered[code]['stats']['odCut']}, "
            f"{rendered[code]['stats']['islands']} island(s)",
            log,
        )

    widest = max(one["box"][2] - one["box"][0] for one in rendered.values())
    tallest = max(one["box"][3] - one["box"][1] for one in rendered.values())
    side = int(math.ceil(max(widest, tallest) * (1 + 2 * MARGIN)))
    canvas = (side, side)

    manifest = {
        "case": case,
        "targetMpp": target_mpp,
        "canvas": list(canvas),
        "slides": {},
    }

    for code, one in rendered.items():
        cx, cy = one["centroid"]
        # Where this slide's array is pasted so its centroid lands at the canvas centre.
        offset_x = side / 2.0 - cx
        offset_y = side / 2.0 - cy

        placed = np.full((side, side, 3), 255, dtype=np.uint8)
        placed_mask = np.zeros((side, side), dtype=bool)
        placed_visible = np.zeros((side, side), dtype=bool)
        _paste(placed, one["rgb"], offset_x, offset_y)
        _paste_mask(placed_mask, one["mask"], offset_x, offset_y)
        _paste_mask(placed_visible, one["visible"], offset_x, offset_y)

        # Everything outside the tissue goes to white, and this is not cosmetic. Each
        # slide carries its own blocky staircase of scanner padding at its own grey
        # level, plus its own area of bare glass; none of that is tissue, all of it is
        # high-contrast structure, and it is *different on every slide*. Left in, a
        # feature detector spends its budget on edges that cannot possibly correspond
        # between two sections. Dilated by a few pixels first so the cut never eats the
        # tissue border the registration is being asked to follow.
        grow = max(1, int(round(WHITEN_MARGIN_UM / target_mpp)))
        keep = ndimage.binary_dilation(placed_mask, np.ones((3, 3)), iterations=grow)
        # The union, not the strict mask alone. `placed_visible` is what saves a
        # near-negative section: its strict mask is a few specks, but everything that is
        # not padding and not bare glass is still real tissue and still has structure a
        # matcher can use.
        placed[~(keep | placed_visible)] = 255

        Image.fromarray(placed).save(out / f"{code}.png")
        Image.fromarray(placed_mask.astype(np.uint8) * 255).save(out / f"{code}_mask.png")
        Image.fromarray(placed_visible.astype(np.uint8) * 255).save(
            out / f"{code}_visible.png"
        )

        manifest["slides"][code] = {
            "file": f"{code}.png",
            "maskFile": f"{code}_mask.png",
            # The permissive mask, and the one shape comparisons must use. Measuring
            # how alike two sections are on the strict mask asks how alike their
            # STAINING is: CAN_00865's near-negative CD44 section scored 0.36 against
            # its siblings and was handed a 253 degree rotation derived from a handful
            # of specks - a rotation the stack registration would then have applied.
            "visibleFile": f"{code}_visible.png",
            "source": str(slides[code]),
            "marker": common.PANEL.get(code),
            "baseMpp": one["baseMpp"],
            # render_xy = level0_xy * scale + offset. Written out rather than
            # recomputed anywhere else, because every coordinate downstream depends on it.
            "scale": one["level0ToRender"],
            "offset": [offset_x, offset_y],
            "tissueMm2": one["stats"]["tissueMm2"],
            "odCut": one["stats"]["odCut"],
            "paddingShare": one["stats"]["paddingShare"],
        }

    common.write_json(common.case_dir(case) / "render.json", manifest)
    common.say(f"{case}: rendered {len(rendered)} slides at {target_mpp} um/px, canvas {side} px", log)
    return manifest


def _paste(canvas: np.ndarray, patch: np.ndarray, ox: float, oy: float) -> None:
    """Paste `patch` into `canvas` at a (possibly negative) integer offset, clipped."""
    ox, oy = int(round(ox)), int(round(oy))
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(cw, ox + pw), min(ch, oy + ph)
    if x1 <= x0 or y1 <= y0:
        return
    canvas[y0:y1, x0:x1] = patch[y0 - oy : y1 - oy, x0 - ox : x1 - ox]


def _paste_mask(canvas: np.ndarray, patch: np.ndarray, ox: float, oy: float) -> None:
    ox, oy = int(round(ox)), int(round(oy))
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(cw, ox + pw), min(ch, oy + ph)
    if x1 <= x0 or y1 <= y0:
        return
    canvas[y0:y1, x0:x1] = patch[y0 - oy : y1 - oy, x0 - ox : x1 - ox]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("--mpp", type=float, default=DEFAULT_MPP)
    args = parser.parse_args()
    common.ensure_dirs()
    render_case(args.case, args.mpp, log=common.RUN_LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
