"""Schemas describing the 19-step IHC scoring pipeline.

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


class SlideRole(str, Enum):
    """Which slide of a case a step reads.

    A case is two physical sections cut from one block: an H&E, where structure is
    legible enough to find the tumour, and one immunostained slide per antibody,
    where the brown that actually gets scored lives. Which of those a step reads is
    a property of the step, not of whatever the viewer happened to load - so it is
    declared here, once, rather than inferred at each of the nineteen call sites.

    `CASE` is the third answer and not a synonym for "both": step 19 compares five
    markers of one block against four readers of the same block, so it is keyed on
    the case and a single slide is not a thing it can be run on.
    """

    HE = "he"
    IHC = "ihc"
    CASE = "case"


class PipelineStage(APIModel):
    """One step of the pipeline."""

    id: str
    index: int = Field(ge=1, le=19)
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
        description="Whether this step actually runs. All seventeen do today.",
    )
    runs_on: list[SlideRole] = Field(
        default_factory=lambda: [SlideRole.HE],
        description=(
            "Which slide or slides of a case this step reads. Two entries does NOT by "
            "itself mean the step runs twice - see `reads_slides_together`."
        ),
    )
    reads_slides_together: bool = Field(
        default=False,
        description=(
            "True when a step needs both slides in hand at the same moment, rather "
            "than running once per slide. "
            "**This is the difference between two kinds of two-slide step, and "
            "collapsing them puts a dead control on screen.** Steps 2 to 6 are a "
            "per-slide prefix: each has a separate answer for each slide, so the UI "
            "offers a switch between them. Step 1 reads both pyramids so the two "
            "readouts land side by side, and step 10 registers one slide onto the "
            "other - neither has a 'which slide' to choose, and offering the switch "
            "there would be a toggle that changes nothing."
        ),
    )


class PipelineSummary(APIModel):
    """Headline counts for the pipeline overview."""

    total_steps: int
    implemented_steps: int
    trained_steps: int
    pretrained_steps: int
    classical_steps: int
