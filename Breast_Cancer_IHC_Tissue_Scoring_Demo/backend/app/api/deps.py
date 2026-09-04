"""Shared FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends

from app.services.pipeline_service import PipelineService, pipeline_service
from app.services.slide_service import SlideService, slide_service


def get_pipeline_service() -> PipelineService:
    """Provide the pipeline catalogue service."""
    return pipeline_service


def get_slide_service() -> SlideService:
    """Provide the template slide service."""
    return slide_service


PipelineServiceDep = Annotated[PipelineService, Depends(get_pipeline_service)]
SlideServiceDep = Annotated[SlideService, Depends(get_slide_service)]
