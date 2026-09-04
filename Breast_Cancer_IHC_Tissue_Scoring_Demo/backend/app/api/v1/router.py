"""Aggregates every v1 endpoint router."""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    calibration,
    deconvolution,
    density,
    health,
    maintenance,
    pipeline,
    qc,
    slides,
    tiling,
    tissue,
    tissue_type,
    uploads,
)

api_router = APIRouter()
api_router.include_router(calibration.router)
api_router.include_router(deconvolution.router)
api_router.include_router(density.router)
api_router.include_router(health.router)
api_router.include_router(maintenance.router)
api_router.include_router(pipeline.router)
api_router.include_router(qc.router)
api_router.include_router(slides.router)
api_router.include_router(tiling.router)
api_router.include_router(tissue.router)
api_router.include_router(tissue_type.router)
api_router.include_router(uploads.router)
