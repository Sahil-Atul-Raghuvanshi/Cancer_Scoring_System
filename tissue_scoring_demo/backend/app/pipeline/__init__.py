"""The 16-step IHC scoring pipeline.

Each `stepNN_<name>/` package is one stage, named and numbered to match
`app.data.pipeline_steps.PIPELINE_STAGES` - that catalogue is generated into
the frontend, so a step's folder name and its id must never drift apart.

A step package owns *how* its stage works. It does not own *when* it runs -
that is `runner.py` - and it does not duplicate slide reading, image math or
config; those live in `app.ingestion`, `app.core` and are imported, not
copied. `contract.py` defines the shape a step consumes and returns, so a
step never imports another step's module - only the context the runner hands
it.

Steps 1 to 5 are implemented and run for real against an uploaded slide.
Steps 6-16 are stubs: each documents its input, output and the plan for
building it, and raises `StepNotImplementedError` if actually invoked. This
mirrors `PipelineStage.implemented` in the API - a stub here is exactly the
steps the catalogue marks unimplemented, and no others.
"""
