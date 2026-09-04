# Step 11 - Cell typing

Not implemented yet. Documented here so the shape of the step is fixed before
the code is, per `app/data/pipeline_steps.py` (`id="cell-typing"`) and
`docs/demo-pipeline-guide.md`.

## Input

Nucleus instances

## Output

Typed cells

## Status

Not built. `pipeline.run()` raises `StepNotImplemented` - the same thing the
API reports for this stage via `PipelineStage.implemented=False`. When this
step is built, its algorithmic core goes here; anything it shares with other
steps (image, colour or geometry math, WSI access) belongs in `app.ingestion`
/ a future `app.common`, not copied into this folder.
