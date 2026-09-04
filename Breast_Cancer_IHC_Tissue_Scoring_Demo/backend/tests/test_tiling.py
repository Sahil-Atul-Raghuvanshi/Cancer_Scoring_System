"""Step 7 tests.

Split the way every earlier step's are: the geometry needs nothing but numpy, so
it is the bulk of this file, and the endpoints are tested for the states that need
no slide on disk.

Tiling is plumbing, so most of these are about *arithmetic that has to hold*
rather than about a method being correct - the grid covering the slide, the funnel
counts adding up, a coordinate landing where it says it does. Three are about
something less obvious and are worth naming:
`test_covered_area_does_not_shrink_when_the_overlap_grows` pins the measurement
artefact that a stride-sized grid would introduce,
`test_the_clean_gate_is_measured_over_tissue_and_not_over_the_tile` pins why a
half-glass tile is not credited for the glass being clean, and
`test_an_edge_tile_survives_the_tissue_gate` pins the reason this step's tissue
gate is far lower than step 5's.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.pipeline.step07_tiling import index as tiling
from app.pipeline.step07_tiling import overlay
from app.pipeline.step07_tiling.index import TilingError, build_index, sample

# A mask grid at 2 um/px over a slide whose native resolution is 0.5 um/px, which
# is the demo's usual arrangement: one mask pixel to sixteen slide pixels.
MASK_MPP = 2.0
BASE_MPP = 0.5
TARGET_MPP = 0.5
SIZE = 64


def _slide_size(mask: np.ndarray) -> tuple[int, int]:
    """The level-0 dimensions a mask of this shape implies."""
    height, width = mask.shape
    return (int(width * MASK_MPP / BASE_MPP), int(height * MASK_MPP / BASE_MPP))


def _index(
    mask: np.ndarray,
    considered: np.ndarray | None = None,
    *,
    overlap: float = 0.0,
    min_tissue_share: float = 0.1,
    min_clean_share: float = 0.5,
    size: int = SIZE,
    max_tiles: int = 400_000,
) -> tiling.TileIndex:
    return build_index(
        tissue=mask,
        considered=considered,
        mask_mpp=MASK_MPP,
        base_mpp=BASE_MPP,
        slide_size=_slide_size(mask),
        target_mpp=TARGET_MPP,
        size=size,
        overlap=overlap,
        min_tissue_share=min_tissue_share,
        min_clean_share=min_clean_share,
        max_tiles=max_tiles,
    )


def _solid(shape: tuple[int, int] = (128, 128), fill: float = 0.5) -> np.ndarray:
    """A mask with a solid square of tissue in the middle, `fill` of each axis.

    Deliberately tile-aligned, which makes every tile either wholly on the square
    or wholly off it. That is the right fixture for counting and for coordinates -
    the expected numbers are exact - and the wrong one for anything about partial
    tiles, which is what `_disc` is for.
    """
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    h0, w0 = int(height * (1 - fill) / 2), int(width * (1 - fill) / 2)
    mask[h0 : height - h0, w0 : width - w0] = True
    return mask


def _disc(shape: tuple[int, int] = (160, 160), radius: float = 0.35) -> np.ndarray:
    """A round island of tissue, so the grid produces genuinely partial tiles.

    A section is not a rectangle and does not line up with anybody's grid, so the
    tiles around its perimeter hold some fraction of tissue rather than all or
    none. Those tiles are the whole subject of this step's tissue gate - they are
    where the invasive front tends to sit - so the tests about that gate need a
    fixture that actually has some.
    """
    height, width = shape
    ys, xs = np.ogrid[:height, :width]
    centre_y, centre_x = height / 2.0, width / 2.0
    limit = min(height, width) * radius
    return ((ys - centre_y) ** 2 + (xs - centre_x) ** 2) <= limit**2


# --- the grid ----------------------------------------------------------------


def test_the_grid_covers_the_whole_slide() -> None:
    """Ceiling division, so the last partial tile at each edge is still a tile.

    A floor would drop a strip up to one tile wide down two sides of every slide,
    and on a small biopsy that strip is a real part of the specimen.
    """
    mask = _solid()
    index = _index(mask)
    width, height = _slide_size(mask)

    assert (index.cols - 1) * index.stride + index.span >= width
    assert (index.rows - 1) * index.stride + index.span >= height


def test_every_tile_lies_inside_the_slide() -> None:
    mask = _solid()
    index = _index(mask)
    width, height = _slide_size(mask)

    for tile in index.tiles:
        assert 0 <= tile.x <= width
        assert 0 <= tile.y <= height
        assert tile.x + tile.span <= width + index.stride
        assert tile.y + tile.span <= height + index.stride


def test_the_funnel_is_a_nested_sequence() -> None:
    """Each gate can only remove, never add. If this ever fails the funnel is a lie."""
    index = _index(_solid(), overlap=0.25)
    funnel = index.funnel

    assert funnel.every >= funnel.on_tissue >= funnel.clean >= 1
    assert funnel.every == index.cols * index.rows
    assert funnel.every == len(index.tiles)


def test_the_funnel_counts_agree_with_the_tiles_it_reports() -> None:
    """A filtered list and an audited one are different things, so they must match."""
    index = _index(_solid(), overlap=0.25)

    assert sum(1 for tile in index.tiles if tile.kept) == index.funnel.clean
    assert (
        sum(1 for tile in index.tiles if tile.rejected_by is None) == index.funnel.clean
    )


def test_a_rejected_tile_names_the_gate_that_rejected_it() -> None:
    mask = _solid()
    considered = mask.copy()
    considered[40:60, :] = False  # a horizontal band of artefact across the tissue

    index = _index(mask, considered)

    for tile in index.tiles:
        if tile.kept:
            assert tile.rejected_by is None
        else:
            assert tile.rejected_by in {"tissue", "clean"}

    # Both gates fired on this slide, which is what makes the naming meaningful.
    reasons = {tile.rejected_by for tile in index.tiles if not tile.kept}
    assert reasons == {"tissue", "clean"}


def test_more_overlap_means_more_tiles_over_the_same_tissue() -> None:
    """The step's one compute dial, and the direction it turns."""
    mask = _solid()
    none = _index(mask, overlap=0.0)
    quarter = _index(mask, overlap=0.25)
    half = _index(mask, overlap=0.5)

    assert none.funnel.every < quarter.funnel.every < half.funnel.every
    assert none.funnel.clean < half.funnel.clean


def test_covered_area_does_not_shrink_when_the_overlap_grows() -> None:
    """Pins the measurement artefact a stride-sized grid would introduce.

    Overlapping more cannot cover *less* of the slide - the finer lattice is a
    superset of the coarse one in reach, not a replacement for it. An earlier
    version of `_covered_mm2` painted onto a grid at stride resolution, so the cell
    size moved with the overlap and 50% reported less ground than 25%. That is a
    fact about the ruler and not about the slide, and this is the test that says so.
    """
    mask = _solid()
    areas = [_index(mask, overlap=share).covered_mm2 for share in (0.0, 0.25, 0.5)]

    assert areas == sorted(areas)


def test_covered_area_counts_overlap_once() -> None:
    """Not `count x tile area`, which is 4x the truth at 50% overlap.

    A reader comparing overlaps would otherwise see the area quadruple while the
    section stayed the same size, and conclude that overlapping tiles look at more
    of the slide. They look at the same tissue more often.
    """
    mask = _solid()
    index = _index(mask, overlap=0.5)

    naive = index.funnel.clean * (index.span * BASE_MPP / 1000.0) ** 2
    assert index.covered_mm2 < naive / 2


def test_covered_area_is_in_the_same_ballpark_as_the_tissue() -> None:
    """Tiles are squares and a section is not, so a little over is expected."""
    mask = _solid()
    index = _index(mask, overlap=0.25)

    assert index.tissue_mm2 > 0
    assert 0.9 < index.covered_mm2 / index.tissue_mm2 < 2.0


# --- the two gates -----------------------------------------------------------


def test_a_tile_of_pure_glass_is_dropped() -> None:
    index = _index(_solid(fill=0.25))

    corner = next(tile for tile in index.tiles if tile.col == 0 and tile.row == 0)
    assert corner.tissue_share == 0.0
    assert not corner.kept
    assert corner.rejected_by == "tissue"


def test_an_edge_tile_survives_the_tissue_gate() -> None:
    """The reason this step's gate is 10% where step 5's was 85%.

    Step 8 needs *all* the tissue: a tile it never sees is a region it cannot
    classify, and the invasive front - the part the whole score is gated on - often
    sits at the section's edge, where tiles are half glass. So a partial tile has to
    survive here even though step 5 would refuse to stand on it.
    """
    mask = _disc()
    index = _index(mask, min_tissue_share=0.1)

    partial = [tile for tile in index.tiles if 0.1 <= tile.tissue_share < 0.85]
    assert partial, "the fixture should produce edge tiles to test"
    assert all(tile.kept for tile in partial)

    # And the same tiles would be refused under step 5's much stricter gate.
    strict = _index(mask, min_tissue_share=0.85)
    strict_partial = [
        tile for tile in strict.tiles if 0.1 <= tile.tissue_share < 0.85
    ]
    assert not any(tile.kept for tile in strict_partial)


def test_the_clean_gate_is_measured_over_tissue_and_not_over_the_tile() -> None:
    """Otherwise a half-glass tile is credited for the glass being clean.

    Glass is always clean - it has nothing on it to be wrong - so a share taken
    over the whole tile would let a tile whose *tissue* is entirely folded pass,
    on the strength of the empty half beside it.
    """
    mask = np.zeros((128, 128), dtype=bool)
    # Tissue in the left half of the top-left tile only, and all of it flagged.
    mask[0:16, 0:8] = True
    # Plus a clean block well away from it, so the index has something to keep -
    # a slide with no usable tile at all raises, which would test the wrong thing.
    mask[64:96, 64:96] = True

    considered = np.ones_like(mask)
    considered[0:16, 0:8] = False

    index = _index(mask, considered, min_tissue_share=0.05)
    corner = next(tile for tile in index.tiles if tile.col == 0 and tile.row == 0)

    assert corner.tissue_share > 0.05
    assert corner.clean_share == 0.0
    assert corner.rejected_by == "clean"


def test_no_quality_control_passes_the_artefact_gate() -> None:
    """Step 2 not having run must not silently drop tiles."""
    mask = _solid()
    index = _index(mask, None)

    assert not index.qc_gated
    assert index.funnel.on_tissue == index.funnel.clean
    assert all(tile.clean_share == 1.0 for tile in index.tiles if tile.tissue_share > 0)


def test_a_stricter_gate_never_keeps_more_tiles() -> None:
    mask = _solid()
    loose = _index(mask, min_tissue_share=0.1)
    tight = _index(mask, min_tissue_share=0.6)

    assert tight.funnel.clean <= loose.funnel.clean


# --- coordinates -------------------------------------------------------------


def test_a_tile_origin_lands_where_its_fractional_bounds_say() -> None:
    """The browser draws from the fractions and the reader clicks through to the
    pixels, so the two have to describe the same square."""
    mask = _solid()
    index = _index(mask, overlap=0.25)
    width, height = _slide_size(mask)

    for tile in index.tiles[::7]:
        assert tile.fx * width == pytest.approx(tile.x, abs=1.0)
        assert tile.fy * height == pytest.approx(tile.y, abs=1.0)


def test_the_span_follows_from_the_resolutions_rather_than_the_level() -> None:
    """A tile is a physical extent, and the level it is read from is a consequence.

    At 0.5 um/px target on a 0.5 um/px slide, a 64 px tile spans 64 level-0 pixels;
    on a 0.25 um/px slide the same tile spans 128. Converting through microns is
    what keeps that right across scanners.
    """
    mask = _solid()
    index = build_index(
        tissue=mask,
        considered=None,
        mask_mpp=MASK_MPP,
        base_mpp=0.25,
        slide_size=(int(mask.shape[1] * MASK_MPP / 0.25), int(mask.shape[0] * MASK_MPP / 0.25)),
        target_mpp=0.5,
        size=64,
        overlap=0.0,
        min_tissue_share=0.1,
        min_clean_share=0.5,
        max_tiles=400_000,
    )
    assert index.span == 128
    assert index.tile_um == pytest.approx(32.0)


def test_the_tile_um_is_the_field_of_view_and_not_the_pixel_count() -> None:
    """The number that decides whether gland architecture fits in one tile."""
    index = _index(_solid(), size=512)
    assert index.tile_um == pytest.approx(512 * TARGET_MPP)


# --- the sampled listing -----------------------------------------------------


def test_the_sample_spans_the_index_rather_than_its_first_rows() -> None:
    """The grid is generated in row order, so the head of the list is one strip.

    A reader shown that would be looking at the top edge of the section and being
    told it was a sample of the slide.
    """
    # Big enough, and overlapped enough, that the kept tiles outnumber what the
    # listing will carry - which is the situation the spacing rule exists for.
    index = _index(_disc(shape=(256, 256)), overlap=0.5)
    kept = [tile for tile in index.tiles if tile.kept]
    assert len(kept) > tiling.MAX_LISTED

    listed = sample(index, limit=20)
    rows = {tile.row for tile in listed}

    assert len(listed) <= 20
    assert all(tile.kept for tile in listed)
    # Spread across the section's rows rather than clustered at its top.
    assert max(rows) - min(rows) > (max(t.row for t in kept) - min(t.row for t in kept)) / 2


def test_a_small_index_is_listed_whole() -> None:
    index = _index(_solid(fill=0.2))
    kept = [tile for tile in index.tiles if tile.kept]
    assert len(sample(index)) == len(kept)


# --- refusals ----------------------------------------------------------------


def test_a_slide_with_no_usable_tile_is_refused_with_a_reason() -> None:
    mask = np.zeros((128, 128), dtype=bool)
    with pytest.raises(TilingError, match="no tile of this slide"):
        _index(mask)


def test_a_slide_whose_tissue_is_all_flagged_is_refused() -> None:
    mask = _solid()
    with pytest.raises(TilingError, match="no tile of this slide"):
        _index(mask, np.zeros_like(mask))


def test_a_grid_larger_than_the_cap_is_refused_rather_than_truncated() -> None:
    """Truncating would hand step 8 an arbitrary two thirds of a section."""
    with pytest.raises(TilingError, match="past the"):
        _index(_solid(), max_tiles=10)


def test_an_overlap_of_one_is_refused() -> None:
    """At 1 the stride would be zero and the grid would never advance."""
    with pytest.raises(TilingError, match="overlap"):
        _index(_solid(), overlap=1.0)


def test_a_negative_overlap_is_refused() -> None:
    with pytest.raises(TilingError, match="overlap"):
        _index(_solid(), overlap=-0.25)


def test_a_tile_smaller_than_a_patch_is_refused() -> None:
    with pytest.raises(TilingError, match="not a patch"):
        _index(_solid(), size=8)


def test_a_slide_with_no_dimensions_is_refused() -> None:
    with pytest.raises(TilingError, match="no dimensions"):
        build_index(
            tissue=_solid(),
            considered=None,
            mask_mpp=MASK_MPP,
            base_mpp=BASE_MPP,
            slide_size=(0, 0),
            target_mpp=TARGET_MPP,
            size=SIZE,
            overlap=0.0,
            min_tissue_share=0.1,
            min_clean_share=0.5,
            max_tiles=400_000,
        )


# --- the panels --------------------------------------------------------------


def test_the_grid_panel_encodes_a_png() -> None:
    mask = _solid()
    index = _index(mask, overlap=0.25)

    thumbnail = np.full((*mask.shape, 3), 220, dtype=np.uint8)
    png = overlay.grid_png(thumbnail, index)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_a_dense_grid_still_encodes_a_png() -> None:
    """Past `MAX_DRAWN` the cells are filled rather than outlined - a different path."""
    mask = _solid(shape=(900, 900))
    index = _index(mask, overlap=0.0, size=16)
    assert max(index.cols, index.rows) > tiling.MAX_DRAWN

    thumbnail = np.full((*mask.shape, 3), 220, dtype=np.uint8)
    assert overlay.grid_png(thumbnail, index).startswith(b"\x89PNG\r\n\x1a\n")


def test_the_sample_panel_encodes_a_png() -> None:
    channel = np.random.default_rng(0).uniform(0.0, 1.5, size=(64, 64)).astype(np.float32)
    assert overlay.sample_png(channel, high=1.2).startswith(b"\x89PNG\r\n\x1a\n")


def test_the_sample_panel_serves_a_ground_rather_than_failing_when_there_is_no_tile() -> None:
    """A real state, and the report says so in words. A 404 would make a reported
    fact look like a failure."""
    assert overlay.empty_png(64).startswith(b"\x89PNG\r\n\x1a\n")


# --- the endpoints -----------------------------------------------------------


def test_an_unknown_slide_is_a_404(client: TestClient) -> None:
    assert client.get("/api/v1/tiling/nope").status_code == 404


def test_an_unknown_panel_is_rejected_at_the_boundary(client: TestClient) -> None:
    assert client.get("/api/v1/tiling/nope/panels/scatter.png").status_code == 422


def test_an_overlap_of_one_is_rejected_at_the_boundary(client: TestClient) -> None:
    """Refused by the API rather than reaching the service: at 1 the stride is zero."""
    assert client.get("/api/v1/tiling/nope", params={"overlap": 1.0}).status_code == 422


def test_a_negative_overlap_is_rejected_at_the_boundary(client: TestClient) -> None:
    assert client.get("/api/v1/tiling/nope", params={"overlap": -0.1}).status_code == 422


def test_a_threshold_out_of_range_is_rejected_at_the_boundary(client: TestClient) -> None:
    assert client.get("/api/v1/tiling/nope", params={"threshold": 300}).status_code == 422
