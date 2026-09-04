"""The step contract every pipeline stage speaks.

A step receives the `PipelineContext` built by everything that ran before it
and returns a `StepResult` naming what it added. The runner is the only thing
that sees the whole sequence; a step only ever sees the context, never
another step's module - so steps stay reorderable in principle even though
the pipeline defines one fixed order, and a step's dependency on an earlier
one is a lookup by stage id, not an import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PipelineError(RuntimeError):
    """A step's precondition - an earlier step's output - is missing."""


class StepNotImplementedError(NotImplementedError):
    """Raised by a stub step: documented in the catalogue, not built yet.

    Matches `PipelineStage.implemented=False` for the same stage id - a stub
    module and a catalogue entry must agree on whether the step runs.
    """

    def __init__(self, stage_id: str, title: str) -> None:
        super().__init__(f"step '{stage_id}' ({title}) is not implemented")
        self.stage_id = stage_id
        self.title = title


class RunCancelled(RuntimeError):
    """A long-running step was asked to stop, and did.

    Shared between the two steps whose work outlives a request - quality control and
    tissue-type segmentation - because it is one concept and two names for it would
    end up meaning two different things.

    **Distinct from a failure, deliberately.** A cancelled run is a decision someone
    made; a failed one is a problem to report. Collapsing them would make the UI
    apologise for something the viewer just asked for, and would bury real errors
    among cancellations in the log.

    The mechanism is a raise from the progress callback rather than a flag the worker
    polls at the top of its loop. That is what makes a cancel land promptly on a step
    whose inner loop is a forward pass: the callback is already called after every
    batch, so there is exactly one place that has to check, and no path where a
    cancelled run keeps going because nobody looked.
    """


@dataclass
class PipelineContext:
    """Everything produced by the steps that have already run.

    Slides are addressed by `upload_id` everywhere else in this codebase -
    `upload_service` owns turning that id into a path once the upload is
    verified - so the context does the same rather than inventing a second
    way to name a slide.

    `artifacts` is keyed by stage id (e.g. "quality-control"), not by
    position, so a step states what it needs by name via `require()` rather
    than by trusting it is the Nth thing in a list.
    """

    upload_id: str
    artifacts: dict[str, Any] = field(default_factory=dict)

    def require(self, stage_id: str) -> Any:
        """The output of an earlier step, or a `PipelineError` naming it."""
        try:
            return self.artifacts[stage_id]
        except KeyError as exc:
            raise PipelineError(
                f"step requires the output of '{stage_id}', which has not run yet"
            ) from exc


@dataclass
class StepResult:
    """What a step hands back to the runner."""

    stage_id: str
    output: Any
