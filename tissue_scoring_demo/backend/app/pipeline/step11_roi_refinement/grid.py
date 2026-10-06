"""BEETLE's own window grid for a slide, built once and shared by steps 10 and 11.

**Why it cannot be step 8's grid.** Both steps narrow "where a model looks" down to
windows, and both use `inference.build_grid` to do it, but the geometry is a property of
the *model*: step 8's tile head reads a 224 px window at 2 um/px, BEETLE reads a 448 px
one at its fixed 0.5. The two land on different numbers of windows over the same tissue,
so pricing a refinement against step 8's grid would report a cost from the wrong model,
and restricting against it would hand `pixels.segment` a grid whose `size` and `mpp` do
not describe the network it is about to run.

**Why it is not step 7's choice either.** Step 7 commits a field of view because that
choice selects which *trained checkpoint* step 8 loads. BEETLE's weights are identical at
all four fields of view - only the extent of the array changes - so there is no
checkpoint to match, and inheriting step 7's number would tie this step to a decision
that was made about something else. `roi_refinement_fov_um` is its own setting.

**What it does inherit, and must.** Step 7's audited tile list and step 3's tissue mask,
through the same two gates `build_grid` applies for step 8: a window runs when its centre
lands on a kept tile and enough of the window is tissue. Those are decisions about the
slide, not about the model, and re-deciding them here would let step 11 look at tissue
the pipeline had already excluded.

Both steps go through this function, which is the only reason step 10's cost estimate and
step 11's actual run can be trusted to agree - they are the same grid.
"""

from __future__ import annotations

from app.core.config import settings
from app.pipeline.step08_tissue_type_segmentation import beetle
from app.pipeline.step08_tissue_type_segmentation.inference import WindowGrid, build_grid


class RefinementGridError(ValueError):
    """The grid cannot be laid - a missing step, or a slide with no physical scale."""


def build_for(upload_id: str, *, slide_size: tuple[int, int], base_mpp: float) -> WindowGrid:
    """BEETLE's grid over one slide's kept tissue.

    `slide_size` and `base_mpp` are passed in rather than read here because every caller
    already has a slide open - opening it again to ask for two numbers is a second read
    of a pyramid header and a second chance for the two to disagree.

    Zero overlap, and that is a real choice rather than a default left alone. Overlapping
    windows cost forward passes in proportion to the window count and buy a smoothed
    per-window summary; this step does not use the per-window summary at all - it traces
    the *pixel* mask - and `pixels.segment` writes each window's core, so overlap changes
    nothing about which pixels are written. It would be a pure multiplication of the
    step's cost.
    """
    # Imported inside the function: this module lives in the pipeline package and the
    # services import it, so importing them at module scope would be a cycle. It is the
    # same lazy-import pattern every `pipeline.py` in this tree uses.
    from app.services.tiling_service import tiling_service
    from app.services.tissue_service import tissue_service

    if not base_mpp:
        raise RefinementGridError(
            "this slide records no microns-per-pixel, so there is no physical scale to "
            "read BEETLE's window at. Supply one on step 1 first."
        )

    selection = tiling_service.selection(upload_id)
    index = tiling_service.tile_index(
        upload_id,
        branch=selection.branch,
        fov=selection.field_of_view_um,
        overlap=selection.overlap,
        threshold=selection.tissue_threshold,
    )
    footprint = tissue_service.footprint(upload_id, threshold=selection.tissue_threshold)

    return build_grid(
        index,
        base_mpp=float(base_mpp),
        slide_size=slide_size,
        size=beetle.window_px(settings.roi_refinement_fov_um),
        mpp=beetle.SPACING,
        overlap=settings.roi_refinement_overlap,
        max_windows=settings.tissue_type_max_windows,
        tissue=footprint.mask,
        mask_mpp=footprint.mpp,
        min_tissue_share=settings.roi_refinement_min_tissue_share,
    )


__all__ = ["RefinementGridError", "build_for"]
