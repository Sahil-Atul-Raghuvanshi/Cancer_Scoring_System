"""Aggregates every v1 endpoint router."""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    binning,
    calibration,
    cases,
    cell_typing,
    compartments,
    data_versions,
    deconvolution,
    density,
    health,
    history,
    ihc_alignment,
    maintenance,
    nuclei,
    panel,
    per_cell,
    pipeline,
    qc,
    roi,
    roi_refinement,
    roi_selection,
    scores,
    slides,
    tiling,
    tissue,
    tissue_type,
    uploads,
    validation,
)

api_router = APIRouter()
api_router.include_router(binning.router)
api_router.include_router(calibration.router)
api_router.include_router(cases.router)
api_router.include_router(cell_typing.router)
api_router.include_router(compartments.router)
api_router.include_router(data_versions.router)
api_router.include_router(deconvolution.router)
api_router.include_router(density.router)
api_router.include_router(health.router)
api_router.include_router(history.router)
api_router.include_router(ihc_alignment.router)
api_router.include_router(maintenance.router)
api_router.include_router(panel.router)
api_router.include_router(nuclei.router)
api_router.include_router(per_cell.router)
api_router.include_router(pipeline.router)
api_router.include_router(qc.router)
api_router.include_router(roi.router)
api_router.include_router(roi_selection.router)
api_router.include_router(roi_refinement.router)
api_router.include_router(scores.router)
api_router.include_router(slides.router)
api_router.include_router(tiling.router)
api_router.include_router(tissue.router)
api_router.include_router(tissue_type.router)
api_router.include_router(uploads.router)
api_router.include_router(validation.router)
