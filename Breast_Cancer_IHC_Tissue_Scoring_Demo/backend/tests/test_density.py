"""Step 5 tests.

Split the way steps 2, 3 and 4's are: the maths needs nothing but numpy, so it is
the bulk of this file and runs everywhere. The endpoints are tested for the states
that need no slide on disk.

Most assertions are about a *property* rather than a value, because the numbers
depend on the slide and the properties are what the pipeline relies on. Three
tests are different, and deliberately: `test_two_synthetic_stains_are_recovered`,
`test_a_single_stain_is_not_reported_as_two` and
`test_density_space_is_linear_where_intensity_space_is_not` build slides whose
stain vectors are known by construction and check that the step finds them. Those
are the step's headline claims, so they are pinned against ground truth rather
than against themselves.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.common.imaging import optical_density
from app.common.stains import weighted_percentile
from app.pipeline.step05_optical_density import density as od
from app.pipeline.step05_optical_density import overlay, tiles
from app.pipeline.step05_optical_density.density import (
    DensityError,
    build_cloud,
    degrees_between,
    direction_is_stable,
    measure_additivity,
    measure_limits,
    measure_stats,
    transform,
    unit,
)
from app.pipeline.step05_optical_density.tiles import TileError, rank_tiles

WHITE = (240.0, 238.0, 232.0)

HAEMATOXYLIN = np.array(od.REFERENCE_VECTORS[0][1], dtype=np.float64)
DAB = np.array(od.REFERENCE_VECTORS[1][1], dtype=np.float64)


# --- synthetic slides --------------------------------------------------------


def _compose(
    concentrations: np.ndarray,
    vectors: np.ndarray,
    *,
    white: tuple[float, float, float] = WHITE,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """An RGB tile built forwards through Beer-Lambert from known stain amounts.

    `concentrations` is HxWxK, `vectors` is Kx3. The optical density of the result
    is `concentrations @ vectors` by construction, so any test that recovers those
    vectors is recovering something that is genuinely there rather than something
    the estimator and the generator agreed on.

    The composition is a *sum in density space*, which is the whole point: it is
    the same statement as multiplying transmissions, and building the image this
    way is why these tests are a check on the step and not on the arithmetic that
    generated them.
    """
    density = concentrations @ vectors
    reference = np.asarray(white, dtype=np.float64)
    intensity = reference * np.power(10.0, -density)

    if noise:
        intensity = intensity + np.random.default_rng(seed).normal(
            0.0, noise, size=intensity.shape
        )

    return np.clip(np.round(intensity), 0, 255).astype(np.uint8)


def _two_stain_tile(size: int = 160, seed: int = 1) -> np.ndarray:
    """Half haematoxylin, half DAB, with a band of mixtures between them.

    Concentrations span a wide range on purpose. A tile where every pixel carries
    the same amount of stain would still have two arms, but it could not test the
    claim that the arms do not move as concentration changes.
    """
    rng = np.random.default_rng(seed)
    amounts = np.zeros((size, size, 2))

    third = size // 3
    amounts[:, :third, 0] = rng.uniform(0.2, 1.4, (size, third))
    amounts[:, -third:, 1] = rng.uniform(0.2, 1.4, (size, third))

    middle = size - 2 * third
    amounts[:, third : third + middle, 0] = rng.uniform(0.15, 0.7, (size, middle))
    amounts[:, third : third + middle, 1] = rng.uniform(0.15, 0.7, (size, middle))

    return _compose(amounts, np.stack([HAEMATOXYLIN, DAB]), noise=0.6, seed=seed)


# --- the primitives ----------------------------------------------------------


def test_a_unit_vector_keeps_its_direction_and_loses_its_length() -> None:
    scaled = unit(np.array([0.3, 0.6, 0.9]))
    assert np.linalg.norm(scaled) == pytest.approx(1.0)
    assert degrees_between(scaled, np.array([0.3, 0.6, 0.9])) == pytest.approx(0.0, abs=1e-9)


def test_a_zero_vector_is_left_alone_rather_than_dividing_by_zero() -> None:
    assert np.allclose(unit(np.zeros(3)), np.zeros(3))


def test_the_published_stain_vectors_are_unit_length_as_published() -> None:
    """A sanity check on the constants, not on the code.

    Ruifrok's vectors are quoted normalised, and every angle in this step is a dot
    product against them. A typo in one digit would tilt an arm comparison by
    degrees while everything still ran, so it is worth one assertion.
    """
    for _, triple in od.REFERENCE_VECTORS:
        assert np.linalg.norm(triple) == pytest.approx(1.0, abs=0.005)


def test_haematoxylin_and_dab_are_far_enough_apart_to_be_separable() -> None:
    """The reference pair's own separation, which the report quotes for comparison."""
    assert 30.0 < degrees_between(HAEMATOXYLIN, DAB) < 50.0


def test_a_weighted_percentile_matches_the_plain_one_when_weights_are_equal() -> None:
    values = np.linspace(0.0, 1.0, 101)
    weights = np.ones_like(values)
    for q in (1.0, 25.0, 50.0, 99.0):
        assert weighted_percentile(values, weights, q) == pytest.approx(
            float(np.percentile(values, q)), abs=0.01
        )


def test_a_weighted_percentile_is_pulled_towards_the_heavy_pixels() -> None:
    """The property the arms depend on: a reliable pixel outvotes an unreliable one."""
    values = np.array([0.0, 1.0])
    assert weighted_percentile(values, np.array([1.0, 99.0]), 50.0) > 0.5
    assert weighted_percentile(values, np.array([99.0, 1.0]), 50.0) < 0.5


def test_a_weighted_percentile_falls_back_rather_than_dividing_by_zero() -> None:
    values = np.linspace(0.0, 1.0, 11)
    assert weighted_percentile(values, np.zeros_like(values), 50.0) == pytest.approx(0.5)


# --- the two admission tests -------------------------------------------------


def test_a_pixel_with_a_nearly_black_channel_has_no_stable_direction() -> None:
    """The failure this test exists to prevent, in its simplest form.

    One 8-bit level at an intensity of 1 is 0.43 of optical density, so such a
    pixel's direction is set by which way the last level rounded. It is also
    *long*, which puts it at the angular extreme - exactly where the arms are read
    from. That is not a small effect politely ignored; on a real slide it moved an
    arm onto the wrong stain.
    """
    density = np.array([[[1.0, 2.4, 0.4]]])
    dark = np.array([[[24.0, 1.0, 96.0]]])
    bright = np.array([[[110.0, 60.0, 140.0]]])

    assert not direction_is_stable(density, dark, tolerance_deg=1.5)[0, 0]
    assert direction_is_stable(density, bright, tolerance_deg=1.5)[0, 0]


def test_a_faint_pixel_needs_more_light_than_a_dark_one_for_the_same_precision() -> None:
    """The test scales with the vector's length, which is why it is one test and not two.

    Angular error is roughly the quantisation step over the length, so a short
    vector has to be measured more precisely than a long one to point as reliably.
    """
    faint = np.array([[[0.05, 0.06, 0.03]]])
    strong = np.array([[[1.0, 1.2, 0.6]]])
    intensity = np.array([[[40.0, 40.0, 40.0]]])

    assert direction_is_stable(strong, intensity, tolerance_deg=1.5)[0, 0]
    assert not direction_is_stable(faint, intensity, tolerance_deg=1.5)[0, 0]


def test_a_tighter_tolerance_never_admits_more_pixels() -> None:
    rng = np.random.default_rng(3)
    density = rng.uniform(0.0, 2.0, (40, 40, 3))
    intensity = rng.uniform(1.0, 250.0, (40, 40, 3))

    admitted = [
        int(direction_is_stable(density, intensity, tolerance_deg=tol).sum())
        for tol in (0.5, 1.0, 2.0, 5.0)
    ]
    assert admitted == sorted(admitted)


# --- the point cloud ---------------------------------------------------------


def test_two_synthetic_stains_are_recovered() -> None:
    """The step's headline claim, against ground truth.

    A tile is composed forwards from Ruifrok's haematoxylin and DAB vectors at
    known, widely varying concentrations. Nothing tells `build_cloud` what went in.
    If the arms come back pointing at the two vectors, the geometry step 6 inverts
    is really there - and if they ever stop, this test is the one that says so.
    """
    rgb = _two_stain_tile()
    result = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0)

    cloud = result.cloud
    assert cloud is not None
    assert cloud.two_armed

    # Each arm within a few degrees of a *different* reference, and between them
    # they account for both stains rather than both landing on the stronger one.
    assert {arm.nearest for arm in cloud.arms} == {"haematoxylin", "dab"}
    for arm in cloud.arms:
        assert arm.degrees_from_nearest < 6.0

    # The plane really is a plane: two absorbers span one, and a cloud that needed
    # three dimensions would make the drawing a projection to distrust.
    assert cloud.explained > 0.99


def test_a_single_stain_is_not_reported_as_two() -> None:
    """The honest negative, and the state the demo is most likely to hit.

    A tile of counterstain alone has one arm. The step must say so rather than
    reporting the two tails of one lobe as two stains - which is what an
    unguarded percentile pair would do, since it always returns two numbers.
    """
    rng = np.random.default_rng(7)
    amounts = rng.uniform(0.2, 1.4, (160, 160, 1))
    rgb = _compose(amounts, HAEMATOXYLIN[None, :], noise=0.6)

    cloud = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0).cloud
    assert cloud is not None
    assert not cloud.two_armed
    assert cloud.separation < od.TWO_ARM_SEPARATION


def test_a_tile_with_no_stain_has_no_cloud_rather_than_two_noisy_arms() -> None:
    """Returning None is the point: a confident answer here would be fabricated."""
    rgb = np.full((80, 80, 3), 239, dtype=np.uint8)
    result = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0)

    assert result.cloud is None
    assert result.additivity is None


def test_the_cloud_is_reproducible_rather_than_mirroring_between_runs() -> None:
    """An eigenvector is defined up to sign, so without orienting them the scatter
    would flip between runs on one slide and the arms would swap places."""
    rgb = _two_stain_tile()
    first = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0).cloud
    second = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0).cloud

    assert first is not None and second is not None
    assert first.basis == second.basis
    assert [arm.angle for arm in first.arms] == [arm.angle for arm in second.arms]


def test_the_plotted_box_contains_the_origin() -> None:
    """Zero stain is a point the picture has to contain: every ray starts there,
    and a wedge floating in a box without it is not the geometry step 6 inverts."""
    cloud = transform(rgb=_two_stain_tile(), white=WHITE, mpp=0.5, floor=1.0).cloud
    assert cloud is not None
    assert cloud.x_low <= 0.0 <= cloud.x_high
    assert cloud.y_low <= 0.0 <= cloud.y_high


def test_the_plot_axes_carry_the_same_scale() -> None:
    """Otherwise every angle on screen is wrong, and the angles are the content."""
    cloud = transform(rgb=_two_stain_tile(), white=WHITE, mpp=0.5, floor=1.0).cloud
    assert cloud is not None
    assert (cloud.x_high - cloud.x_low) == pytest.approx(cloud.y_high - cloud.y_low)


def test_a_reference_vector_in_the_plane_reports_no_out_of_plane_part() -> None:
    """The honesty term on the projection. A reference drawn near an arm but
    standing out of the page is a shadow, not a match."""
    cloud = transform(rgb=_two_stain_tile(), white=WHITE, mpp=0.5, floor=1.0).cloud
    assert cloud is not None

    by_name = {reference.name: reference for reference in cloud.references}
    # Both stains that composed the tile span the plane, so both lie in it.
    assert by_name["haematoxylin"].out_of_plane < 0.1
    assert by_name["dab"].out_of_plane < 0.1
    # Eosin was never in the tile, so it has no reason to lie in that plane.
    assert by_name["eosin"].out_of_plane > 0.1


# --- Beer-Lambert, the claim the whole step rests on -------------------------


def test_density_space_is_linear_where_intensity_space_is_not() -> None:
    """The reason for the logarithm, on a slide whose stain is known exactly.

    One stain at a wide range of concentrations. In density space the direction
    cannot move, because concentration only scales the ray. In intensity space it
    must, because `I = I0 * 10^(-cv)` curves. The check is built to be
    conservative - its pixels are selected by their direction in *intensity*
    space, so the intensity drift is the constrained one - and the gap still has
    to appear.
    """
    rng = np.random.default_rng(11)
    amounts = rng.uniform(0.1, 1.8, (200, 200, 1))
    rgb = _compose(amounts, HAEMATOXYLIN[None, :], noise=0.4)

    result = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0)
    additivity = result.additivity
    assert additivity is not None

    assert additivity.od_drift < 2.0
    assert additivity.intensity_drift > 2.0 * max(additivity.od_drift, 0.05)
    # The bands have to span a real range of concentration or the test is vacuous:
    # every direction is trivially stable across pixels that all carry the same
    # amount of stain.
    assert additivity.bins[-1].density > 1.5 * additivity.bins[0].density


def test_the_faintest_band_is_the_reference_both_drifts_are_measured_from() -> None:
    rgb = _two_stain_tile()
    additivity = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0).additivity
    assert additivity is not None

    assert additivity.bins[0].od_degrees == 0.0
    assert additivity.bins[0].intensity_degrees == 0.0
    assert additivity.od_drift == additivity.bins[-1].od_degrees


# --- where the transform stops being a measurement --------------------------


def test_the_transform_is_reversible_to_the_precision_of_the_arithmetic() -> None:
    """A change of units, not a filter. If this ever fails, step 5 is destroying data."""
    rgb = _two_stain_tile()
    limits = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0).limits

    assert limits.roundtrip_max < 0.05
    assert limits.roundtrip_exact_share == pytest.approx(1.0)


def test_a_pixel_brighter_than_the_white_point_reports_negative_density() -> None:
    """Not clamped. Negative stain does not exist, so the number is a statement
    about I0 being low here - and clamping would make it look like faint stain."""
    rgb = np.full((20, 20, 3), 250, dtype=np.uint8)
    result = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0)

    assert result.limits.negative_share == pytest.approx(1.0)
    assert result.limits.negative_worst < 0.0
    assert result.stats.mean_minimum < 0.0


def test_a_black_pixel_is_counted_against_the_floor_rather_than_left_infinite() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    result = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0)

    assert np.isfinite(result.od).all()
    assert result.limits.floor_share == pytest.approx(1.0)


def test_the_admission_shares_account_for_the_whole_tile() -> None:
    """Admitted plus faint plus unstable is everything, so nothing is silently lost."""
    cloud = transform(rgb=_two_stain_tile(), white=WHITE, mpp=0.5, floor=1.0).cloud
    assert cloud is not None

    total = cloud.admitted_share + cloud.faint_share + cloud.unstable_share
    assert total == pytest.approx(1.0, abs=1e-6)


def test_a_histogram_covers_the_range_it_reports() -> None:
    stats = measure_stats(
        optical_density(_two_stain_tile().astype(np.float32), WHITE, floor=1.0)
    )
    assert stats.histogram_high > stats.histogram_low
    assert len(stats.histogram) == od.DENSITY_BINS
    assert sum(stats.histogram) > 0


def test_percentiles_are_ordered_as_percentiles(
) -> None:
    stats = measure_stats(
        optical_density(_two_stain_tile().astype(np.float32), WHITE, floor=1.0)
    )
    assert stats.mean_median <= stats.mean_p99 <= stats.mean_maximum
    for channel in range(3):
        assert stats.median[channel] <= stats.p99[channel] <= stats.maximum[channel]


# --- guards ------------------------------------------------------------------


def test_a_greyscale_tile_is_refused_with_a_reason() -> None:
    with pytest.raises(DensityError, match="RGB tile"):
        transform(rgb=np.zeros((8, 8), dtype=np.uint8), white=WHITE, mpp=0.5, floor=1.0)


def test_a_white_field_of_the_wrong_shape_is_refused() -> None:
    """A position-dependent I0 has to be evaluated on the tile's own grid, and
    silently broadcasting a mismatched one would divide by the wrong pixels."""
    with pytest.raises(DensityError, match="white field"):
        transform(
            rgb=np.full((8, 8, 3), 200, dtype=np.uint8),
            white=np.full((4, 4, 3), 240.0),
            mpp=0.5,
            floor=1.0,
        )


def test_a_white_point_with_no_glass_behind_it_is_refused() -> None:
    """Step 4 returns zeros when it found no glass to sample. Dividing by that is
    not a density, and the message says whose problem it is."""
    with pytest.raises(DensityError, match="zero or negative"):
        transform(
            rgb=np.full((8, 8, 3), 200, dtype=np.uint8),
            white=(0.0, 0.0, 0.0),
            mpp=0.5,
            floor=1.0,
        )


def test_a_field_shaped_white_point_is_accepted_as_readily_as_a_triple() -> None:
    """The two shapes `optical_density` broadcasts, so no caller branches on which
    field step 4 justified."""
    rgb = _two_stain_tile(size=80)
    flat = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0)
    field = transform(
        rgb=rgb,
        white=np.broadcast_to(np.asarray(WHITE), (80, 80, 3)).copy(),
        mpp=0.5,
        floor=1.0,
    )
    assert np.allclose(flat.od, field.od)


# --- choosing a tile ---------------------------------------------------------


def _screening_slide(size: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """A thumbnail with three blocks: two stains, one stain, and bare glass."""
    rgb = np.full((size, size, 3), 238, dtype=np.uint8)
    tissue = np.zeros((size, size), dtype=bool)

    block = size // 3
    rng = np.random.default_rng(5)

    # Block 0: both stains, strongly.
    both = rng.uniform(0.3, 1.2, (block, block, 2))
    rgb[:block, :block] = _compose(both, np.stack([HAEMATOXYLIN, DAB]), noise=0.5)
    tissue[:block, :block] = True

    # Block 1: one stain, just as strongly - nothing to un-mix.
    single = rng.uniform(0.3, 1.2, (block, block, 1))
    rgb[:block, block : 2 * block] = _compose(single, HAEMATOXYLIN[None, :], noise=0.5)
    tissue[:block, block : 2 * block] = True

    return rgb, tissue


def test_the_chooser_prefers_the_block_that_holds_two_stains() -> None:
    """The reason the chooser exists. A block of pure counterstain is stained just
    as strongly and has one arm, so ranking on stain alone would pick it half the
    time and put a one-armed scatter under a caption about two stains."""
    rgb, tissue = _screening_slide()
    ranked = rank_tiles(
        rgb=rgb,
        tissue=tissue,
        considered=None,
        white=WHITE,
        mpp=2.0,
        base_mpp=0.5,
        slide_size=(960, 960),
        tile_um=160.0,
        min_tissue_share=0.85,
        min_stain=0.05,
        od_floor=1.0,
    )

    assert ranked
    best = ranked[0]
    # The two-stain block is the top-left one.
    assert best.col == 0 and best.row == 0
    assert best.mixing > 0.0


def test_the_chooser_refuses_a_block_that_step_two_flagged() -> None:
    """The score is measured over clean pixels but the *tile* covers the whole
    block, so a block that is a third artefact would be scored on its good part
    and transformed including its bad part."""
    rgb, tissue = _screening_slide()
    # Everything in play except the two-stain block, which is the top-left one. So
    # the block that would otherwise win is the only one step 2 has ruled out.
    considered = np.ones_like(tissue)
    considered[:80, :80] = False

    ranked = rank_tiles(
        rgb=rgb,
        tissue=tissue,
        considered=considered,
        white=WHITE,
        mpp=2.0,
        base_mpp=0.5,
        slide_size=(960, 960),
        tile_um=160.0,
        min_tissue_share=0.85,
        min_stain=0.05,
        od_floor=1.0,
    )
    assert all(not (entry.col == 0 and entry.row == 0) for entry in ranked)


def test_the_chooser_refuses_a_block_too_faint_for_its_mixing_to_mean_anything() -> None:
    """`mixing` is a ratio of two eigenvalues of a point cloud, and a ratio means
    nothing until the cloud is bigger than the noise around it. Without this gate a
    faint patch of false colour outscores real stain, because two wrong colours are
    still two colours - which is what happened on the demo's own slide."""
    rgb, tissue = _screening_slide()
    with pytest.raises(TileError, match="mean optical density"):
        rank_tiles(
            rgb=rgb,
            tissue=tissue,
            considered=None,
            white=WHITE,
            mpp=2.0,
            base_mpp=0.5,
            slide_size=(960, 960),
            tile_um=160.0,
            min_tissue_share=0.85,
            min_stain=5.0,  # nothing on any slide reaches this
            od_floor=1.0,
        )


def test_every_scored_block_reports_the_figures_it_was_judged_on() -> None:
    """An audited list rather than a filtered one: a block that lost still says why."""
    rgb, tissue = _screening_slide()
    ranked = rank_tiles(
        rgb=rgb,
        tissue=tissue,
        considered=None,
        white=WHITE,
        mpp=2.0,
        base_mpp=0.5,
        slide_size=(960, 960),
        tile_um=160.0,
        min_tissue_share=0.85,
        min_stain=0.05,
        od_floor=1.0,
    )
    scores = [entry.score for entry in ranked]
    assert scores == sorted(scores, reverse=True)
    for entry in ranked:
        assert entry.score == pytest.approx(entry.stain * entry.mixing)
        assert 0.0 <= entry.fx <= 1.0
        assert 0.0 <= entry.fy <= 1.0


def test_a_requested_position_snaps_to_a_block_that_was_actually_scored() -> None:
    """A viewer clicking the slide map does not land on a block boundary, and
    reading an unscored tile would report a score belonging to its neighbour."""
    rgb, tissue = _screening_slide()
    ranked = rank_tiles(
        rgb=rgb,
        tissue=tissue,
        considered=None,
        white=WHITE,
        mpp=2.0,
        base_mpp=0.5,
        slide_size=(960, 960),
        tile_um=160.0,
        min_tissue_share=0.85,
        min_stain=0.05,
        od_floor=1.0,
    )
    picked = tiles.nearest(ranked, x=10, y=10)
    assert picked in ranked


def test_a_slide_smaller_than_one_tile_is_refused_with_a_reason() -> None:
    with pytest.raises(TileError):
        rank_tiles(
            rgb=np.full((8, 8, 3), 238, dtype=np.uint8),
            tissue=np.ones((8, 8), dtype=bool),
            considered=None,
            white=WHITE,
            mpp=2.0,
            base_mpp=0.5,
            slide_size=(32, 32),
            tile_um=4000.0,
            min_tissue_share=0.85,
            min_stain=0.01,
            od_floor=1.0,
        )


# --- the panels --------------------------------------------------------------


def test_every_panel_encodes_a_png() -> None:
    result = transform(rgb=_two_stain_tile(size=96), white=WHITE, mpp=0.5, floor=1.0)
    candidate = tiles.Candidate(
        col=0,
        row=0,
        x=0,
        y=0,
        tissue_share=1.0,
        considered_share=1.0,
        stain=0.4,
        mixing=0.2,
        score=0.08,
        usable=True,
        fx=0.0,
        fy=0.0,
        fw=0.5,
        fh=0.5,
    )
    thumbnail = np.full((64, 64, 3), 238, dtype=np.uint8)

    panels = [
        overlay.map_png(thumbnail, candidates=[candidate], chosen=candidate),
        overlay.tile_png(result.rgb),
        overlay.density_png(result),
        overlay.scatter_png(result),
        overlay.limits_png(result),
    ]
    for png in panels:
        assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_the_scatter_serves_a_ground_rather_than_failing_when_there_is_no_cloud() -> None:
    """A reported fact should not look like an error. The report already says the
    tile carries no stain; a panel that 404s would contradict it."""
    flat = np.full((32, 32, 3), 239, dtype=np.uint8)
    result = transform(rgb=flat, white=WHITE, mpp=0.5, floor=1.0)

    assert result.cloud is None
    assert overlay.scatter_png(result).startswith(b"\x89PNG\r\n\x1a\n")


# --- the endpoints -----------------------------------------------------------


def test_an_unknown_slide_is_a_404(client: TestClient) -> None:
    assert client.get("/api/v1/density/not-a-slide").status_code == 404


def test_an_unknown_panel_is_rejected_at_the_boundary(client: TestClient) -> None:
    """422 and the valid names, because a typo is not a conflict."""
    response = client.get("/api/v1/density/not-a-slide/panels/wedge.png")
    assert response.status_code == 422


def test_a_negative_tile_position_is_rejected_at_the_boundary(client: TestClient) -> None:
    assert client.get("/api/v1/density/not-a-slide?x=-1&y=0").status_code == 422


def test_a_percentile_out_of_range_is_rejected_at_the_boundary(client: TestClient) -> None:
    assert client.get("/api/v1/density/not-a-slide?percentile=140").status_code == 422


def test_measure_limits_refuses_an_empty_tile() -> None:
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(DensityError):
        measure_limits(empty, empty.astype(np.float32), WHITE, floor=1.0)


def test_additivity_is_skipped_rather_than_guessed_when_there_are_too_few_pixels() -> None:
    """Five bands of forty pixels is the floor. Below it the check would report a
    drift measured from a handful of pixels as if it were a property of the stain."""
    rng = np.random.default_rng(13)
    amounts = rng.uniform(0.4, 1.2, (26, 26, 1))
    rgb = _compose(amounts, HAEMATOXYLIN[None, :], noise=0.4)

    result = transform(rgb=rgb, white=WHITE, mpp=0.5, floor=1.0)
    if result.cloud is not None:
        assert measure_additivity(result.cloud, rgb, WHITE) is None


def test_a_cloud_is_not_built_from_too_few_points() -> None:
    """`MIN_CLOUD_PIXELS` exists because two arms can always be produced; the
    question is whether they mean anything."""
    rng = np.random.default_rng(17)
    amounts = rng.uniform(0.4, 1.2, (18, 18, 2))
    rgb = _compose(amounts, np.stack([HAEMATOXYLIN, DAB]), noise=0.4)

    assert build_cloud(
        optical_density(rgb.astype(np.float32), WHITE, floor=1.0),
        np.maximum(rgb.astype(np.float32), 1.0),
    ) is None
