"""Step 11 - nuclei segmentation.

Split into three groups, because they need three different things:

  * the geometry and sampling maths, which need nothing and always run
  * the stain arithmetic, which needs numpy and always runs
  * the model itself, which needs the 15 MB checkpoint and skips without it

The last group is where the parity test lives, and it is the important one: this
model returns an empty label map rather than an error when its input is not
normalised the way its metadata specifies, so "no nuclei" and "broken" look
identical from the outside unless something checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.common import stains
from app.core.config import settings
from app.pipeline.step13_nuclei_segmentation import instances, sampling, watershed
from app.pipeline.step13_nuclei_segmentation.stain_input import (
    estimate_basis,
    haematoxylin_only_rgb,
    ruifrok_basis,
)

# --- sampling ---------------------------------------------------------------


def square(x0: float, y0: float, side: float) -> list[list[float]]:
    return [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]]


def test_a_region_is_rasterised_with_its_holes_cut_out() -> None:
    """Outer ring fills, every later ring cuts back - step 9 and 10's convention."""
    outer = square(0, 0, 1000)
    hole = square(300, 300, 400)

    footprint = sampling.rasterise([outer, hole], mpp=1.0, target_mpp=10.0)

    assert footprint.mask.any()
    # The centre sits inside the hole and must be empty.
    height, width = footprint.mask.shape
    assert not footprint.mask[height // 2, width // 2]
    # A point in the outer ring but outside the hole is filled.
    assert footprint.mask[2, 2]


def test_fields_are_spread_across_the_region_not_taken_from_one_corner() -> None:
    """The sample must cover the region, or the density it reports is one corner's.

    This is the property that keeps nuclei-per-mm2 usable as the cross-slide QC
    check: a sample drawn from the top-left would report that corner's density as
    the region's, and the check compares densities.
    """
    region = [square(0, 0, 20_000)]

    fields, available = sampling.fields_for_region(
        region, mpp=0.25, slide_width=50_000, slide_height=50_000, count=8, tile_px=512
    )

    assert len(fields) == 8
    assert available > 8
    ys = [f.y for f in fields]
    # The sample must reach past the first row of candidates.
    assert max(ys) - min(ys) > 0.4 * 20_000


def test_sampling_is_deterministic() -> None:
    """Two runs disagree about segmentation or not at all - never about which pixels."""
    region = [square(0, 0, 8_000)]
    kwargs = dict(mpp=0.25, slide_width=20_000, slide_height=20_000, count=5, tile_px=512)

    first, _ = sampling.fields_for_region(region, **kwargs)
    second, _ = sampling.fields_for_region(region, **kwargs)

    assert [(f.x, f.y) for f in first] == [(f.x, f.y) for f in second]


def test_a_field_is_sized_in_microns_not_pixels() -> None:
    """The model is scale-specific, so the read spans whatever 512 of its pixels are."""
    region = [square(0, 0, 40_000)]

    fields, _ = sampling.fields_for_region(
        region, mpp=0.25, slide_width=60_000, slide_height=60_000, count=1,
        tile_px=512, model_mpp=0.5,
    )

    # 512 model pixels at 0.5 um/px is 256 um, which on a 0.25 um/px slide is 1024 px.
    assert fields[0].span == 1024


def test_fields_never_run_off_the_slide() -> None:
    region = [square(9_000, 9_000, 4_000)]

    fields, _ = sampling.fields_for_region(
        region, mpp=1.0, slide_width=10_000, slide_height=10_000, count=20, tile_px=512
    )

    for field in fields:
        assert field.x + field.span <= 10_000
        assert field.y + field.span <= 10_000


# --- measuring one nucleus --------------------------------------------------


def disc(size: int, radius: float) -> np.ndarray:
    span = np.arange(size) - (size - 1) / 2.0
    ys, xs = np.meshgrid(span, span, indexing="ij")
    return (xs * xs + ys * ys) <= radius * radius


def test_a_disc_measures_as_a_disc() -> None:
    """Area from pixel count, and a circularity that actually reaches 1."""
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[disc(64, 10)] = 1

    found = instances.extract(
        labels, haematoxylin=None, mpp=0.5, x0=0, y0=0, level0_scale=1.0,
        border_px=0, min_area_um2=0.0,
    )

    assert len(found) == 1
    nucleus = found[0]
    expected = int(disc(64, 10).sum()) * 0.25
    assert nucleus.area_um2 == pytest.approx(expected)
    assert nucleus.circularity > 0.9
    assert nucleus.eccentricity < 0.2
    # Centred, in the frame the caller asked for.
    assert nucleus.x == pytest.approx(31.5, abs=1.0)


def test_an_ellipse_is_eccentric_and_a_disc_is_not() -> None:
    """The shape terms have to separate a lymphocyte from a fibroblast nucleus."""
    labels = np.zeros((64, 64), dtype=np.int32)
    span = np.arange(64) - 31.5
    ys, xs = np.meshgrid(span, span, indexing="ij")
    labels[(xs / 20.0) ** 2 + (ys / 4.0) ** 2 <= 1.0] = 1

    nucleus = instances.extract(
        labels, haematoxylin=None, mpp=0.5, x0=0, y0=0, level0_scale=1.0,
        border_px=0, min_area_um2=0.0,
    )[0]

    assert nucleus.eccentricity > 0.9
    assert nucleus.circularity < 0.8


def test_a_nucleus_in_the_border_band_is_found_but_not_counted() -> None:
    """It is real, so it is drawn; it was seen truncated, so it is not counted."""
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[0:6, 0:6] = 1          # in the corner, inside the band
    labels[28:36, 28:36] = 2      # in the middle

    found = instances.extract(
        labels, haematoxylin=None, mpp=0.5, x0=0, y0=0, level0_scale=1.0,
        border_px=10, min_area_um2=0.0,
    )

    by_label = {n.label: n for n in found}
    assert by_label[1].counted is False
    assert by_label[2].counted is True


def test_the_denominator_shrinks_with_the_border_band() -> None:
    """Dropping edge objects while counting over the whole field would bias density.

    At the shipped geometry the band is about a fifth of the field, which is far
    larger than any difference the cross-slide check is trying to detect.
    """
    whole = instances.counted_area_mm2(size=512, border_px=0, mpp=0.5)
    inner = instances.counted_area_mm2(size=512, border_px=24, mpp=0.5)

    assert inner < whole
    assert inner / whole == pytest.approx(((512 - 48) / 512) ** 2)


def test_objects_below_the_physical_minimum_are_dropped() -> None:
    """In um2, so the cutoff survives a change of scanner."""
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[10, 10] = 1                 # one pixel: 0.25 um2 at 0.5 um/px
    labels[30:40, 30:40] = 2           # 100 px: 25 um2

    found = instances.extract(
        labels, haematoxylin=None, mpp=0.5, x0=0, y0=0, level0_scale=1.0,
        border_px=0, min_area_um2=10.0,
    )

    assert [n.label for n in found] == [2]


def test_outlines_come_back_in_slide_coordinates() -> None:
    """The caller's frame, not the crop's - one conversion, at the boundary."""
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[10:20, 10:20] = 1

    nucleus = instances.extract(
        labels, haematoxylin=None, mpp=0.5, x0=5_000, y0=7_000, level0_scale=2.0,
        border_px=0, min_area_um2=0.0,
    )[0]

    xs = [x for ring in nucleus.rings for x, _ in ring]
    ys = [y for ring in nucleus.rings for _, y in ring]
    assert min(xs) == pytest.approx(5_000 + 10 * 2.0)
    assert min(ys) == pytest.approx(7_000 + 10 * 2.0)


# --- the stain arithmetic ---------------------------------------------------


def test_removing_the_dab_leaves_the_haematoxylin_and_takes_the_brown() -> None:
    """The rule the step exists to enforce, as arithmetic.

    A field painted with both stains must come back with the blue intact and the
    brown gone - not merely dimmer.
    """
    basis = ruifrok_basis()
    white = np.float32(255.0)

    h_vector = np.asarray(stains.REFERENCE_BY_NAME["haematoxylin"])
    dab_vector = np.asarray(stains.REFERENCE_BY_NAME["dab"])

    density = np.zeros((8, 8, 3), dtype=np.float64)
    density[:4] += 0.8 * h_vector          # top half: haematoxylin
    density[4:] += 0.8 * dab_vector        # bottom half: DAB
    rgb = np.clip(255.0 * np.power(10.0, -density), 0, 255).astype(np.uint8)

    stripped = haematoxylin_only_rgb(rgb, white, basis, gain=1.0)

    # The DAB half comes back as blank white; the haematoxylin half stays dark.
    assert stripped[4:].min() > 240
    assert stripped[:4].mean() < stripped[4:].mean()


def test_the_rendering_direction_is_fixed_even_when_the_basis_is_not() -> None:
    """The bug that cost 68 nuclei on a field: drawing along an achromatic arm.

    Macenko on a DAB-dominated slide returns a grey "counterstain" direction.
    Whatever basis measures the amount, the amount is drawn in Ruifrok's blue -
    so a haematoxylin-only render is always blue-on-white, never grey-on-white.
    """
    grey = np.asarray([0.577, 0.577, 0.577])
    dab = np.asarray(stains.REFERENCE_BY_NAME["dab"])
    matrix = stains.complete_basis(np.stack([grey, dab], axis=1))

    from app.pipeline.step13_nuclei_segmentation.stain_input import StainBasis

    achromatic = StainBasis(
        matrix=matrix, inverse=np.linalg.inv(matrix), source="macenko"
    )

    rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
    rendered = haematoxylin_only_rgb(rgb, np.float32(255.0), achromatic, gain=1.0)

    # Blue survives where red and green are absorbed: a haematoxylin render is
    # never neutral, whatever basis measured it.
    assert rendered[..., 2].mean() > rendered[..., 0].mean()


def test_macenko_falls_back_to_the_published_basis_rather_than_guessing() -> None:
    """And says so, so a report can show the estimate was not used."""
    assert estimate_basis(np.zeros((10, 3))).source == "ruifrok"
    assert estimate_basis(np.zeros((0, 3))).source == "ruifrok"


# --- the classical baseline -------------------------------------------------


def test_the_watershed_baseline_separates_two_touching_discs() -> None:
    """A fair implementation, so the comparison on screen is a fair comparison."""
    stain = np.zeros((64, 96), dtype=np.float32)
    span = np.arange(96)
    rows = np.arange(64)
    ys, xs = np.meshgrid(rows, span, indexing="ij")
    stain[((xs - 34) ** 2 + (ys - 32) ** 2) <= 15**2] = 1.0
    stain[((xs - 62) ** 2 + (ys - 32) ** 2) <= 15**2] = 1.0

    labels = watershed.segment(stain, mpp=0.5)

    assert len(np.unique(labels)) - 1 == 2


def test_the_watershed_baseline_returns_nothing_on_blank_tissue() -> None:
    assert watershed.segment(np.zeros((32, 32), dtype=np.float32), mpp=0.5).max() == 0


# --- the model --------------------------------------------------------------


def model_available() -> bool:
    from app.nuclei import model as nuclei_model

    try:
        nuclei_model.load()
    except nuclei_model.ModelUnavailable:
        return False
    return True


needs_model = pytest.mark.skipif(
    not model_available(), reason="the InstanSeg checkpoint is not installed"
)


@needs_model
def test_the_model_reproduces_its_published_output_exactly() -> None:
    """The load-time gate, asserted here too so a change is a test failure.

    Exact equality rather than a tolerance: the model is deterministic on CPU and
    this pair is the upstream's own regression fixture, so anything less is a
    change worth stopping for.
    """
    from app.nuclei.model import _run, load, models_dir

    loaded = load()
    produced = _run(
        loaded.module,
        np.load(models_dir() / "parity-input.npy").astype(np.float32),
        loaded.manifest["input"]["scale_range"],
    )
    expected = np.load(models_dir() / "parity-output.npy")
    if expected.ndim == 4:
        expected = expected[:, 0]

    assert np.array_equal(produced, expected.astype(np.int32))
    # Instance *count*, not the largest id: the model does not emit dense labels
    # - this pair's ids run to 338 for 336 objects.
    assert len(np.unique(produced)) - 1 == loaded.manifest["parity"]["labels"]


@needs_model
def test_without_the_declared_normalisation_the_model_finds_nothing() -> None:
    """The silent-zero failure the parity gate exists to catch.

    This is the reason loading runs the model instead of merely checking a hash:
    the failure mode is not an exception, it is an empty answer that reads as
    "this tissue has no cells".
    """
    import torch

    from app.nuclei.model import load, models_dir

    loaded = load()
    raw = np.load(models_dir() / "parity-input.npy").astype(np.float32)
    with torch.no_grad():
        out = loaded.module(torch.from_numpy(raw))
    tensor = out[0] if isinstance(out, list | tuple) else out

    assert int(tensor.detach().cpu().numpy().max()) == 0


@needs_model
def test_the_checkpoint_matches_the_sha256_its_manifest_declares() -> None:
    from app.nuclei.model import CHECKPOINT, _sha256, manifest, models_dir

    assert _sha256(models_dir() / CHECKPOINT) == manifest()["sha256"]


@needs_model
def test_the_capability_report_names_the_model_and_its_licence() -> None:
    """The screen says what is running before anyone waits for it."""
    from app.nuclei import capability

    report = capability()
    assert report.available is True
    assert report.licence == "Apache-2.0"
    assert report.mpp == 0.5


@needs_model
def test_segment_array_refuses_anything_but_one_rgb_field() -> None:
    from app.nuclei.model import segment_array

    with pytest.raises(ValueError):
        segment_array(np.zeros((64, 64), dtype=np.uint8))


def test_the_shipped_settings_are_the_ones_that_were_measured() -> None:
    """Pins the defaults the README's measurements justify.

    Not a style check: each of these came out of a sweep recorded in
    `config.py`, and changing one silently would leave the prose describing a
    configuration that is no longer running.
    """
    assert settings.nuclei_tile_px == 512
    assert settings.nuclei_border_margin_px == 24
    assert settings.nuclei_haematoxylin_gain == 2.5
    assert settings.nuclei_macenko_per_slide is False
    assert settings.nuclei_remove_dab is True


# --- staleness against step 10 -----------------------------------------------


def _report(**overrides):
    """A minimal `ready` step 11 report, for the staleness checks below."""
    from app.schemas.nuclei import NucleiReport, StainReport

    return NucleiReport(
        he_upload_id="he",
        ihc_upload_id="ihc",
        state="ready",
        generated_at="2026-09-14T20:38:47Z",
        regions=[],
        stain=StainReport(basis="ruifrok", gain=1.0),
        **overrides,
    )


def test_nuclei_from_a_previous_alignment_are_not_reused():
    """A re-run alignment is a different set of regions, not the same ones again.

    This is the bug the rule exists for, and it was real: step 10 was re-run with
    a wider region selection, step 11 handed back its cached report, and every
    later step scored the old three-region geometry while the pipeline reported
    success throughout. The frontend already keyed on the alignment stamp, so the
    screen would have cleared - but the batch runner is not the screen, and a
    staleness rule only one caller obeys is not a rule.
    """
    from types import SimpleNamespace

    from app.services.nuclei_service import nuclei_service

    alignment = SimpleNamespace(generated_at="2026-09-15T11:05:28Z")

    same = _report(alignment_generated_at="2026-09-15T11:05:28Z")
    stale = _report(alignment_generated_at="2026-09-10T15:59:49Z")

    assert nuclei_service._matches_alignment(same, alignment) is True
    assert nuclei_service._matches_alignment(stale, alignment) is False


def test_an_unstamped_nuclei_report_is_treated_as_stale():
    """Written before the stamp existed, so it cannot prove it is current.

    One wasted re-segmentation, which is visible, against measuring the wrong
    regions, which is not.
    """
    from types import SimpleNamespace

    from app.services.nuclei_service import nuclei_service

    unstamped = _report()
    assert unstamped.alignment_generated_at is None
    assert (
        nuclei_service._matches_alignment(
            unstamped, SimpleNamespace(generated_at="anything")
        )
        is False
    )
