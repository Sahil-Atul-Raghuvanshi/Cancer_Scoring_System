"""Endpoints exposing the 17-step pipeline catalogue."""

from fastapi import APIRouter, HTTPException, status

from app.api.deps import PipelineServiceDep
from app.schemas.common import ListResponse
from app.schemas.pipeline import PipelineStage, PipelineSummary

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.get(
    "/stages",
    response_model=ListResponse[PipelineStage],
    summary="List every pipeline stage in execution order",
)
async def list_stages(
    service: PipelineServiceDep,
) -> ListResponse[PipelineStage]:
    """Return the ordered stage catalogue the demo walks through."""
    stages = service.list_stages()
    return ListResponse[PipelineStage](items=stages, total=len(stages))


@router.get(
    "/summary",
    response_model=PipelineSummary,
    summary="Headline counts for the pipeline overview",
)
async def pipeline_summary(
    service: PipelineServiceDep,
) -> PipelineSummary:
    """Return how many steps are trained, pre-trained and classical."""
    return service.summary()


@router.get(
    "/stages/{stage_id}",
    response_model=PipelineStage,
    summary="Fetch a single stage by id",
)
async def get_stage(
    stage_id: str,
    service: PipelineServiceDep,
) -> PipelineStage:
    """Return one stage, or 404 when the identifier is unknown."""
    stage = service.get_stage(stage_id)
    if stage is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown pipeline stage: {stage_id}",
        )
    return stage
