"""Schemas describing the 16-step IHC scoring pipeline.

A stage carries only what is true: what the step does, why it must run at that
point in the order, how it would be built, and whether it is implemented yet.
There are no example outputs here - a step that has not been built has no
numbers, and inventing them would make the walkthrough describe software that
does not exist.
"""

from enum import Enum

from pydantic import Field

from app.schemas.common import APIModel


class Approach(str, Enum):
    """How a step is implemented."""

    CLASSICAL = "classical"
    LIBRARY = "library"
    PRETRAINED = "pretrained"
    TRAINED = "trained"
    LOGIC = "logic"
    PLUMBING = "plumbing"


class Branch(str, Enum):
    """Which fork of the pipeline a step belongs to.

    The pipeline forks at colour deconvolution: one deconvolution feeds
    both arms, handing the haematoxylin channel to the model and the DAB
    channel to the measurement. A step marked shared runs before that fork,
    or - like deconvolution itself - serves both sides of it.
    """

    SHARED = "shared"
    MODEL = "model"
    MEASUREMENT = "measurement"


class PipelineStage(APIModel):
    """One step of the pipeline."""

    id: str
    index: int = Field(ge=1, le=16)
    title: str
    tagline: str
    what: str
    why_here: str
    how: str
    approach: Approach
    branch: Branch = Branch.SHARED
    trains_model: bool = False
    input_label: str
    output_label: str
    action_label: str
    rule: str | None = None
    references: list[str] = Field(default_factory=list)
    implemented: bool = Field(
        default=False,
        description="Whether this step actually runs. Steps 1 and 2 do today.",
    )


class PipelineSummary(APIModel):
    """Headline counts for the pipeline overview."""

    total_steps: int
    implemented_steps: int
    trained_steps: int
    pretrained_steps: int
    classical_steps: int
