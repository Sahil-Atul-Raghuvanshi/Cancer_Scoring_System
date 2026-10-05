# Step 14 - Per-cell measurement

Not implemented yet. Documented here so the shape of the step is fixed before
the code is, per `app/data/pipeline_steps.py` (`id="per-cell-measurement"`) and
`docs/guides/demo-pipeline-guide.md`.

## Input

Compartments + DAB channel

## Output

One measurement row per cell

## Status

Not built. `pipeline.run()` raises `StepNotImplemented` - the same thing the
API reports for this stage via `PipelineStage.implemented=False`. When this
step is built, its algorithmic core goes here; anything it shares with other
steps (image, colour or geometry math, WSI access) belongs in `app.ingestion`
/ a future `app.common`, not copied into this folder.
