# Step 01 - Read the slide

`app/data/pipeline_steps.py` (`id="read-slide"`) · implemented.

## Input

Glass slide scan

## Output

Tiled pyramid, level selected

## Contents

- `pipeline.py` - `build_readout()` (metadata only, no pixels) and
  `SlideReaderService` (readout, thumbnail, region reads for the tile
  viewer). `slide_reader_service` is the singleton the API and the
  orchestrator both call - there is one implementation of step 1.

Slide I/O itself (OpenSlide/tiffslide, DeepZoom) is shared infrastructure and
lives in `app.ingestion`, not here - multiple steps and services read a slide,
so that code is imported, not duplicated per step.
