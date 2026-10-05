# Step 02 - Quality control

`app/data/pipeline_steps.py` (`id="quality-control"`) · implemented.

## Input

Tiled pyramid

## Output

Artefact mask, usable tissue only

## Contents

Two independent things live in this package, and the split is the whole point:

- `models.py`, `inference.py` - GrandQC, a pair of pre-trained segmentation
  models that decide *what* is wrong with the slide (tissue detection, then
  multi-class artefact segmentation). Accurate, and a black box.
- `features.py` - classical, per-tile image statistics (sharpness, contrast,
  texture) that explain *why* a region looks wrong. Transparent, and far too
  naive to be trusted as the decision.
- `classes.py` - the shared class table both of the above are defined against.
- `overlay.py` - renders the artefact mask and class map for the UI.
- `pipeline.py` - `run(context)`, the orchestrator-facing entry point. It
  adapts `app.services.qc_service` (the job-managed, disk-cached run the API
  drives) to the pipeline's `run(context) -> StepResult` contract, rather than
  duplicating that logic here.

GrandQC decides, the classical metrics explain. Nothing measured for the
explanation is allowed back into the decision - see the module docstrings for
the citation and licence notes.
