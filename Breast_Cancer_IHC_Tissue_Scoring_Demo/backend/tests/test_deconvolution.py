"""Step 6 tests.

Split the way steps 2 to 5's are: the maths needs nothing but numpy, so it is the
bulk of this file and runs everywhere; the endpoints are tested for the states
that need no slide on disk.

Most of this file is pinned against **ground truth** rather than against itself,
and it can be because step 6 is invertible. A synthetic tile is built forwards -
known amounts of the published haematoxylin and DAB vectors, summed in optical
density, which is what Beer-Lambert says a real section does - and the step is
then asked to recover those amounts. A test that gets them back has recovered
something genuinely there, not something the estimator and the generator agreed
on.

Three tests carry the step's headline claims and are worth naming:
`test_known_stain_amounts_are_recovered` is that the separation is correct,
`test_a_pure_stain_leaves_nothing_in_the_other_channel` is Rule 3 - that a
threshold on the DAB channel is a threshold on one dye - and
`test_an_off_nominal_slide_scores_differently_under_an_estimated_basis` is the
reason the measurement branch is given fixed vectors and never an estimate.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.common.stains import (
    REFERENCE_BY_NAME,
    RUIFROK_HDAB,
    complete_basis,
    degrees_between,
    recompose,
    unit,
    unmix,
)
from app.pipeline.step06_colour_deconvolution import deconvolution as dc
from app.pipeline.step06_colour_deconvolution import overlay
from app.pipeline.step06_colour_deconvolution.deconvolution import (
    DeconvolutionError,
    deconvolve,
    separate,
)

HAEMATOXYLIN = unit(np.asarray(REFERENCE_BY_NAME["haematoxylin"], dtype=np.float64))
DAB = unit(np.asarray(REFERENCE_BY_NAME["dab"], dtype=np.float64))
EOSIN = unit(np.asarray(REFERENCE_BY_NAME["eosin"], dtype=np.float64))

WHITE = np.array([240.0, 238.0, 232.0])


# --- synthetic tiles ---------------------------------------------------------


def _tile(
    amounts: np.ndarray, vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """An optical density tile built forwards from known stain amounts.

    `amounts` is HxWx2 and `vectors` is 2x3, so the tile's optical density is
    `amounts @ vectors` *by construction* - the same statement as multiplying two
    transmissions, which is what two stacked dyes physically do. Any test that
    recovers those amounts is checking the step rather than the arithmetic that
    generated the fixture.

    Returns the density, an RGB rendering of it, and step 5's two masks computed
    the way step 5 computes them - so the fixtures carry the same notion of "this
    pixel has stain in it" the real hand-off does.
    """
    od = (amounts @ vectors).astype(np.float32)
    rgb = np.clip(np.round(WHITE * np.power(10.0, -od.astype(np.float64))), 0, 255).astype(
        np.uint8
    )

    mean_od = od.mean(axis=-1)
    stained = mean_od >= 0.15
    admitted = stained.reshape(-1)
    return od, rgb, stained, admitted


def _two_stain_amounts(size: int = 96, seed: int = 7) -> np.ndarray:
    """Half the tile mostly haematoxylin, half mostly DAB, a mixed band between.

    Concentrations span a wide range on purpose: a tile where every pixel carries
    the same amount would still separate correctly, but could not show that the
    separation is independent of concentration - which is the property the
    logarithm bought and the whole reason this step is valid.
    """
    rng = np.random.default_rng(seed)
    amounts = np.zeros((size, size, 2))

    half = size // 2
    amounts[:, :half, 0] = rng.uniform(0.3, 1.6, size=(size, half))
    amounts[:, :half, 1] = rng.uniform(0.0, 0.12, size=(size, half))

    amounts[:, half:, 0] = rng.uniform(0.0, 0.12, size=(size, size - half))
    amounts[:, half:, 1] = rng.uniform(0.3, 1.6, size=(size, size - half))

    # A band of genuine mixtures, so the tile is not two separable halves.
    band = slice(half - 8, half + 8)
    amounts[:, band, :] = rng.uniform(0.3, 1.1, size=(size, 16, 2))
    return amounts


def _graded_amounts(size: int = 96, seed: int = 11) -> np.ndarray:
    """DAB spread continuously across the positivity cut, counterstain behind it.

    Deliberately unlike `_two_stain_amounts`, which is bimodal: most of that
    tile's DAB sits either well above the cut or well below it, so rescaling the
    channel moves every *reading* while moving almost no pixel across the line.
    That is a property of the fixture and not of real tissue - a section's DAB is
    graded, which is why a 1+ and a 3+ are different diagnoses rather than
    different sides of a gap - so the claim "changing the basis changes the score"
    has to be tested on a distribution where a change of scale can change it.
    """
    rng = np.random.default_rng(seed)
    amounts = np.zeros((size, size, 2))
    amounts[..., 1] = rng.uniform(0.02, 0.9, size=(size, size))
    amounts[..., 0] = rng.uniform(0.1, 0.6, size=(size, size))
    return amounts


def _run(amounts: np.ndarray, vectors: np.ndarray, *, cut: float = 0.25) -> dc.Separation:
    od, rgb, stained, admitted = _tile(amounts, vectors)
    return deconvolve(
        od=od, rgb=rgb, stained=stained, admitted=admitted, cut=cut, alpha=1.0
    )


# --- the basis itself --------------------------------------------------------


def test_completing_a_stain_pair_gives_unit_columns() -> None:
    matrix = complete_basis(np.stack([HAEMATOXYLIN, DAB], axis=1))
    lengths = np.linalg.norm(matrix, axis=0)
    assert np.allclose(lengths, 1.0, atol=1e-9)


def test_the_third_column_is_the_direction_neither_stain_occupies() -> None:
    """Which is what makes the matrix invertible, and the separation exact."""
    matrix = complete_basis(np.stack([HAEMATOXYLIN, DAB], axis=1))
    residual = matrix[:, 2]

    assert abs(float(residual @ HAEMATOXYLIN)) < 1e-9
    assert abs(float(residual @ DAB)) < 1e-9
    assert abs(float(np.linalg.det(matrix))) > 1e-6


def test_two_parallel_stains_are_refused_rather_than_inverted() -> None:
    """One stain twice is one stain, and there is no plane for a third axis in."""
    with pytest.raises(ValueError, match="parallel"):
        complete_basis(np.stack([HAEMATOXYLIN, HAEMATOXYLIN * 2.0], axis=1))


def test_the_published_basis_is_what_the_step_actually_uses() -> None:
    """The constant step 5 draws its arms against is the one step 6 un-mixes with.

    Two copies would be two chances for the arms on one screen and the matrix on
    the next to disagree about where a stain points, which is the drift the guide
    names by name.
    """
    assert np.allclose(RUIFROK_HDAB[:, 0], HAEMATOXYLIN)
    assert np.allclose(RUIFROK_HDAB[:, 1], DAB)


def test_unmixing_and_recomposing_returns_the_input() -> None:
    rng = np.random.default_rng(3)
    od = rng.uniform(-0.2, 2.5, size=(500, 3))

    coefficients = unmix(od, np.linalg.inv(RUIFROK_HDAB))
    assert np.allclose(recompose(coefficients, RUIFROK_HDAB), od, atol=1e-9)


# --- the separation, against ground truth ------------------------------------


def test_known_stain_amounts_are_recovered() -> None:
    """The headline claim: put in known amounts of two dyes, get them back."""
    amounts = _two_stain_amounts()
    od, _, _, _ = _tile(amounts, np.stack([HAEMATOXYLIN, DAB]))

    channels = separate(od, RUIFROK_HDAB)

    assert np.allclose(channels.haematoxylin, amounts[..., 0], atol=1e-4)
    assert np.allclose(channels.dab, amounts[..., 1], atol=1e-4)


def test_a_pure_stain_leaves_nothing_in_the_other_channel() -> None:
    """Rule 3: a threshold on the DAB channel is a threshold on one dye.

    A tile of haematoxylin alone, at every concentration from faint to dark, must
    read as zero DAB - because in RGB it does not. A dark blue nucleus is dark in
    every channel, so any 'brownness' measure on the mixture would call the darkest
    of these positive.
    """
    amounts = np.zeros((64, 64, 2))
    amounts[..., 0] = np.linspace(0.2, 2.0, 64)[None, :]

    od, _, stained, _ = _tile(amounts, np.stack([HAEMATOXYLIN, DAB]))
    channels = separate(od, RUIFROK_HDAB)

    assert float(np.abs(channels.dab[stained]).max()) < 1e-3
    # And the counterstain itself is recovered across the whole concentration
    # range, not just where it is strong - which is what the logarithm bought.
    assert np.allclose(channels.haematoxylin, amounts[..., 0], atol=1e-4)


def test_the_residual_is_empty_when_two_stains_explain_the_tile() -> None:
    separation = _run(_two_stain_amounts(), np.stack([HAEMATOXYLIN, DAB]))
    assert separation.fixed.residual_share < 0.01


def test_a_third_absorber_shows_up_in_the_residual() -> None:
    """The honest test of the two-stain assumption, and it has to be able to fail."""
    amounts = _two_stain_amounts()
    od, rgb, stained, admitted = _tile(amounts, np.stack([HAEMATOXYLIN, DAB]))

    # Eosin is neither of this assay's dyes, so it has to land in the third column.
    contaminated = od + (0.6 * EOSIN).astype(np.float32)
    dirty = deconvolve(
        od=contaminated,
        rgb=rgb,
        stained=stained,
        admitted=admitted,
        cut=0.25,
        alpha=1.0,
    )

    clean = deconvolve(
        od=od, rgb=rgb, stained=stained, admitted=admitted, cut=0.25, alpha=1.0
    )
    assert dirty.fixed.residual_share > clean.fixed.residual_share


def test_the_separation_is_exact_whatever_the_basis() -> None:
    """A square system loses nothing, so this holds for a wrong basis too.

    Worth pinning separately from the residual: 'exact' and 'a good fit' are
    different properties, and a reader is owed the distinction. A basis pointing
    entirely the wrong way still reconstructs the tile perfectly - it just puts the
    tile's density in the wrong columns.
    """
    separation = _run(_two_stain_amounts(), np.stack([HAEMATOXYLIN, DAB]))

    assert separation.fixed.exactness_max < 1e-5
    if separation.estimated is not None:
        assert separation.estimated.exactness_max < 1e-5


def test_negative_concentrations_are_counted_rather_than_clipped() -> None:
    """A negative amount of dye is impossible, so it is reported, not repaired.

    Clipping would break the reconstruction every other check on this step depends
    on, so the count is the honest response.
    """
    amounts = _two_stain_amounts()
    od, rgb, stained, admitted = _tile(amounts, np.stack([HAEMATOXYLIN, DAB]))

    # Push part of the tile outside the cone the two published vectors span.
    outside = od.copy()
    outside[:, :20, :] = (outside[:, :20, :] + 0.5 * EOSIN).astype(np.float32)

    separation = deconvolve(
        od=outside, rgb=rgb, stained=stained, admitted=admitted, cut=0.25, alpha=1.0
    )
    shares = [channel.negative_share for channel in separation.fixed.channels]
    assert max(shares) > 0.0
    assert separation.fixed.exactness_max < 1e-5


# --- fixed versus estimated, which is the point of the step ------------------


def test_a_nominal_slide_estimates_back_onto_the_published_vectors() -> None:
    """Built from the published pair, the estimate should land on the published pair.

    This is what makes the comparison trustworthy in the other direction: when the
    estimator disagrees with the constants on a real slide, it is the slide
    speaking rather than the estimator being unreliable.
    """
    separation = _run(_two_stain_amounts(), np.stack([HAEMATOXYLIN, DAB]))

    assert separation.estimated is not None
    assert max(separation.estimated.degrees_from_published) < 6.0


def test_an_off_nominal_slide_scores_differently_under_an_estimated_basis() -> None:
    """The argument for fixed vectors, as a number.

    Two slides stained a little differently must not be given two different
    scales, because a score is only comparable if the units are. Here the *same*
    tile is un-mixed twice under the *same* absolute cut, and only the vectors
    change - so any movement in the score is caused by the basis alone.
    """
    # Skewed far enough to clear `BASIS_DEPARTURE_DEG`, which is the point at which
    # the step calls the two bases visibly different scales. 0.22 of eosin only
    # moves a vector about 7 degrees, so the fixture has to lean harder than looks
    # necessary for the departure to be the one the step reports on.
    skewed = np.stack([unit(HAEMATOXYLIN + 0.45 * EOSIN), unit(DAB + 0.40 * EOSIN)])
    separation = _run(_graded_amounts(), skewed)

    assert separation.estimated is not None
    assert max(separation.estimated.degrees_from_published) > dc.BASIS_DEPARTURE_DEG

    fixed = separation.fixed.preview
    estimated = separation.estimated.preview
    assert abs(estimated.positive_share - fixed.positive_share) > 0.02
    # And the DAB channel is on a different scale, which is the mechanism behind
    # the score moving - "0.4 DAB" would mean two different amounts of dye.
    assert abs(estimated.mean_dab / max(fixed.mean_dab, 1e-9) - 1.0) > 0.05


def test_the_fixed_basis_does_not_depend_on_the_tile() -> None:
    """The property the whole measurement branch is built on.

    Two differently stained tiles must get the same matrix, or a study's slides
    cannot be compared. The estimated basis is expected to differ between them -
    that is exactly what disqualifies it from measuring.
    """
    nominal = _run(_two_stain_amounts(seed=1), np.stack([HAEMATOXYLIN, DAB]))
    skewed = _run(
        _two_stain_amounts(seed=2),
        np.stack([unit(HAEMATOXYLIN + 0.2 * EOSIN), unit(DAB + 0.2 * EOSIN)]),
    )

    assert nominal.fixed.matrix == skewed.fixed.matrix
    assert nominal.fixed.degrees_from_published == (0.0, 0.0)

    assert nominal.estimated is not None and skewed.estimated is not None
    assert nominal.estimated.matrix != skewed.estimated.matrix


def test_both_bases_are_counted_over_the_same_range() -> None:
    """Otherwise the comparison would redraw its own axis and hide the shift."""
    separation = _run(
        _two_stain_amounts(),
        np.stack([unit(HAEMATOXYLIN + 0.2 * EOSIN), unit(DAB + 0.2 * EOSIN)]),
    )
    assert separation.estimated is not None

    for fixed, estimated in zip(
        separation.fixed.channels, separation.estimated.channels, strict=True
    ):
        assert fixed.histogram_low == estimated.histogram_low
        assert fixed.histogram_high == estimated.histogram_high


def test_an_estimate_is_refused_rather_than_guessed_from_too_few_pixels() -> None:
    """And the step still answers, because the fixed basis does not need the tile."""
    amounts = _two_stain_amounts()
    od, rgb, stained, _ = _tile(amounts, np.stack([HAEMATOXYLIN, DAB]))

    starved = np.zeros(stained.size, dtype=bool)
    starved[: dc.MIN_ESTIMATE_PIXELS - 1] = True

    separation = deconvolve(
        od=od, rgb=rgb, stained=stained, admitted=starved, cut=0.25, alpha=1.0
    )

    assert separation.estimated is None
    assert separation.estimated_refusal is not None
    assert separation.fixed.preview.positive_share >= 0.0


# --- the preview score -------------------------------------------------------


def test_the_preview_counts_only_stained_pixels() -> None:
    """Including the empty background would drag every basis towards the same answer."""
    amounts = np.zeros((64, 64, 2))
    # A quarter of the tile strongly DAB-positive, the rest empty glass.
    amounts[:16, :, 1] = 1.2

    separation = _run(amounts, np.stack([HAEMATOXYLIN, DAB]), cut=0.25)

    # Over the whole tile the positive share would be a quarter; over the stained
    # pixels it is all of them, because the stained pixels *are* the positive ones.
    assert separation.fixed.preview.positive_share > 0.9
    assert float(np.mean(separation.stained)) == pytest.approx(0.25, abs=0.02)


def test_a_higher_cut_never_calls_more_pixels_positive() -> None:
    amounts = _two_stain_amounts()
    low = _run(amounts, np.stack([HAEMATOXYLIN, DAB]), cut=0.2)
    high = _run(amounts, np.stack([HAEMATOXYLIN, DAB]), cut=0.8)

    assert high.fixed.preview.positive_share <= low.fixed.preview.positive_share


# --- refusals ----------------------------------------------------------------


def test_a_tile_with_no_stain_is_refused_with_a_reason() -> None:
    od = np.zeros((32, 32, 3), dtype=np.float32)
    rgb = np.full((32, 32, 3), 240, dtype=np.uint8)
    stained = np.zeros((32, 32), dtype=bool)

    with pytest.raises(DeconvolutionError, match="carries enough stain"):
        deconvolve(
            od=od,
            rgb=rgb,
            stained=stained,
            admitted=stained.reshape(-1),
            cut=0.25,
            alpha=1.0,
        )


def test_a_single_channel_tile_is_refused() -> None:
    with pytest.raises(DeconvolutionError, match="HxWx3"):
        deconvolve(
            od=np.zeros((32, 32), dtype=np.float32),
            rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            stained=np.ones((32, 32), dtype=bool),
            admitted=np.ones(32 * 32, dtype=bool),
            cut=0.25,
            alpha=1.0,
        )


def test_a_mask_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(DeconvolutionError, match="different sizes"):
        deconvolve(
            od=np.zeros((32, 32, 3), dtype=np.float32),
            rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            stained=np.ones((16, 16), dtype=bool),
            admitted=np.ones(16 * 16, dtype=bool),
            cut=0.25,
            alpha=1.0,
        )


def test_an_unknown_channel_is_refused_with_the_valid_names() -> None:
    separation = _run(_two_stain_amounts(), np.stack([HAEMATOXYLIN, DAB]))
    with pytest.raises(DeconvolutionError, match="unknown channel"):
        separation.fixed_channels.by_name("eosin")


def test_an_unknown_basis_is_refused_with_the_valid_names() -> None:
    separation = _run(_two_stain_amounts(), np.stack([HAEMATOXYLIN, DAB]))
    with pytest.raises(DeconvolutionError, match="unknown basis"):
        separation.channels_for("macenko")


# --- the panels --------------------------------------------------------------


def test_every_panel_encodes_a_png() -> None:
    separation = _run(_two_stain_amounts(), np.stack([HAEMATOXYLIN, DAB]))

    for name in overlay.PANELS:
        png = overlay.panel_png(separation, name, basis=dc.FIXED)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_the_two_bases_draw_different_pictures() -> None:
    """Which is the toggle's whole content, so it is worth pinning.

    Drawn against the same scale - see `overlay` - so a difference here is a
    difference in the channels rather than in how they were stretched.
    """
    separation = _run(
        _two_stain_amounts(),
        np.stack([unit(HAEMATOXYLIN + 0.2 * EOSIN), unit(DAB + 0.2 * EOSIN)]),
    )
    assert separation.estimated is not None

    fixed = overlay.panel_png(separation, "dab", basis=dc.FIXED)
    estimated = overlay.panel_png(separation, "dab", basis=dc.ESTIMATED)
    assert fixed != estimated


def test_the_input_panel_is_the_same_under_either_basis() -> None:
    """Both bases un-mix the same pixels, and the picture must not suggest otherwise."""
    separation = _run(_two_stain_amounts(), np.stack([HAEMATOXYLIN, DAB]))
    assert overlay.panel_png(separation, "tile", basis=dc.FIXED) == overlay.panel_png(
        separation, "tile", basis=dc.ESTIMATED
    )


def test_an_unknown_panel_is_refused() -> None:
    separation = _run(_two_stain_amounts(), np.stack([HAEMATOXYLIN, DAB]))
    with pytest.raises(ValueError, match="unknown panel"):
        overlay.panel_png(separation, "scatter", basis=dc.FIXED)


# --- the geometry step 5 and step 6 share ------------------------------------


def test_the_two_stains_are_far_enough_apart_to_be_separable() -> None:
    """43 degrees on the published pair. Nothing here would work if they were 2."""
    assert degrees_between(HAEMATOXYLIN, DAB) > 30.0


# --- the endpoints -----------------------------------------------------------


def test_an_unknown_slide_is_a_404(client: TestClient) -> None:
    assert client.get("/api/v1/deconvolution/nope").status_code == 404


def test_an_unknown_panel_is_rejected_at_the_boundary(client: TestClient) -> None:
    response = client.get("/api/v1/deconvolution/nope/panels/scatter.png")
    assert response.status_code == 422


def test_an_unknown_basis_is_rejected_at_the_boundary(client: TestClient) -> None:
    response = client.get(
        "/api/v1/deconvolution/nope/panels/dab.png", params={"basis": "vahadane"}
    )
    assert response.status_code == 422


def test_a_negative_tile_position_is_rejected_at_the_boundary(client: TestClient) -> None:
    response = client.get("/api/v1/deconvolution/nope", params={"x": -1, "y": 0})
    assert response.status_code == 422
