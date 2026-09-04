"""Read-only service layer over the static pipeline catalogue.

There is deliberately no image processing here. When the real pipeline is wired
up, these methods become the seam the API keeps talking to.
"""

from app.data.pipeline_steps import PIPELINE_STAGES, STAGES_BY_ID
from app.schemas.pipeline import Approach, PipelineStage, PipelineSummary


class PipelineService:
    """Serves the pipeline stage catalogue and its headline counts."""

    def list_stages(self) -> list[PipelineStage]:
        """Return every stage, in execution order."""
        return sorted(PIPELINE_STAGES, key=lambda stage: stage.index)

    def get_stage(self, stage_id: str) -> PipelineStage | None:
        """Return a single stage by its identifier, or ``None`` if unknown."""
        return STAGES_BY_ID.get(stage_id)

    def summary(self) -> PipelineSummary:
        """Return the counts used by the pipeline overview panel."""
        stages = self.list_stages()
        return PipelineSummary(
            total_steps=len(stages),
            implemented_steps=sum(1 for stage in stages if stage.implemented),
            trained_steps=sum(1 for stage in stages if stage.trains_model),
            pretrained_steps=sum(
                1 for stage in stages if stage.approach is Approach.PRETRAINED
            ),
            classical_steps=sum(
                1
                for stage in stages
                if stage.approach
                in {Approach.CLASSICAL, Approach.LIBRARY, Approach.LOGIC, Approach.PLUMBING}
            ),
        )


pipeline_service = PipelineService()
