"""Step 1 - read the slide.

Opens an uploaded whole-slide image and reports what the rest of the pipeline
needs to know before it can do anything: how big it is, what zoom levels exist,
and - the part that actually matters - what physical resolution each level
represents.

Every downstream step is defined in microns per pixel, not level indices,
because different scanners put different resolutions at the same index. So the
readout is built around mpp and the level chosen for a target mpp.

Two things here are adjustable, and they are not the same thing:

  target_mpp    the resolution the pipeline wants to work at. Changing it
                changes which level gets read.
  mpp_override  the slide's own scale, for files that record none (a plain
                image) or record it wrongly. This is an assertion by whoever
                set it, not a measurement, so the readout reports its source.
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

from app import panel
from app.core.config import settings
from app.ingestion.slide_reader import open_slide
from app.pipeline.contract import PipelineContext, StepResult
from app.schemas.slide import PyramidLevel, SlideReadout
from app.services.upload_service import resolve_ready_path


def _magnification_label(mpp: float | None, objective: float | None) -> str:
    """A human-facing magnification, preferring what the scanner recorded."""
    if objective:
        return f"{objective:g}x"
    if not mpp:
        return "unknown"
    # Fall back to the usual correspondence between resolution and objective.
    for limit, label in ((0.15, "80x"), (0.3, "40x"), (0.6, "20x"), (1.2, "10x"), (2.5, "5x")):
        if mpp <= limit:
            return label
    return "<5x"


def _identity(filename: str) -> tuple[str | None, str | None, str]:
    """Which slide of a case this file looks like, read off its name.

    A guess, and the return type says so by admitting None. `infer_from_filename`
    is explicit that the convention is OncoStem's and a slide from anywhere else
    will not follow it, so this reports what the name suggests and leaves the
    deciding to a person. The role is the part the UI acts on: it is what lets a
    panel notice that the step it is drawing declares `runs_on=[IHC]` while the
    slide underneath it looks like an H&E.
    """
    letter = panel.infer_from_filename(filename)
    if letter is None:
        return None, None, "unknown"
    return letter, panel.spec(letter).name, "he" if letter == "HE" else "ihc"


def _tile_count(width: int, height: int, tile: int) -> int:
    return math.ceil(width / tile) * math.ceil(height / tile)


def _best_level_for_mpp(
    downsamples: tuple[float, ...], base_mpp: float | None, target: float
) -> int:
    """The level to work at for a target resolution.

    Picks the *finest-or-equal* level: the highest index whose downsample is
    still at or below what the target asks for. With no physical scale there is
    nothing to convert against, so fall back to level 0 rather than guessing.

    This mirrors `SlideReader.best_level_for_mpp`, but takes the base mpp as an
    argument so a caller-supplied override can drive it.
    """
    if not base_mpp or target <= 0:
        return 0

    wanted = target / base_mpp
    best = 0
    for index, factor in enumerate(downsamples):
        if factor <= wanted + 1e-6:
            best = index
    return best


def build_readout(
    path: Path,
    *,
    target_mpp: float,
    tile_size: int,
    mpp_override: float | None = None,
) -> SlideReadout:
    """Open the slide and describe its pyramid. Reads metadata only, no pixels."""
    with open_slide(path) as reader:
        scanner_mpp = reader.mpp

        if mpp_override and mpp_override > 0:
            base_mpp: float | None = mpp_override
            mpp_source = "override"
        elif scanner_mpp:
            base_mpp = scanner_mpp
            mpp_source = "scanner"
        else:
            base_mpp = None
            mpp_source = "unknown"

        working_level = _best_level_for_mpp(reader.level_downsamples, base_mpp, target_mpp)

        levels: list[PyramidLevel] = []
        for index, (dimensions, downsample) in enumerate(
            zip(reader.level_dimensions, reader.level_downsamples, strict=True)
        ):
            width, height = int(dimensions[0]), int(dimensions[1])
            levels.append(
                PyramidLevel(
                    level=index,
                    width=width,
                    height=height,
                    downsample=round(float(downsample), 4),
                    mpp=round(base_mpp * float(downsample), 4) if base_mpp else None,
                    tiles=_tile_count(width, height, tile_size),
                    is_working_level=index == working_level,
                )
            )

        width, height = reader.dimensions
        base = levels[0]

        # A pyramid rarely has a level at the resolution you actually want. The
        # reader takes the nearest level that is at-or-finer than the target and
        # shrinks it the rest of the way; going the other direction would
        # upsample, inventing detail that was never scanned.
        working_mpp = levels[working_level].mpp
        working_downsample = (
            round(target_mpp / working_mpp, 4) if working_mpp and target_mpp > 0 else 1.0
        )
        exact = working_mpp is not None and abs(working_downsample - 1.0) < 0.01

        # An overridden scale makes the file's own objective power meaningless,
        # so derive the label from the asserted mpp rather than contradict it.
        objective = reader.objective_power
        label = _magnification_label(base_mpp, None if mpp_source == "override" else objective)

        marker_letter, marker_name, slide_role = _identity(path.name)

        return SlideReadout(
            upload_id="",  # filled in by the caller, which knows the id
            filename=path.name,
            marker_letter=marker_letter,
            marker=marker_name,
            slide_role=slide_role,
            width_px=int(width),
            height_px=int(height),
            megapixels=round(int(width) * int(height) / 1_000_000, 1),
            mpp=round(base_mpp, 4) if base_mpp else None,
            mpp_source=mpp_source,
            scanner_mpp=round(scanner_mpp, 4) if scanner_mpp else None,
            magnification=label,
            objective_power=objective,
            vendor=reader.vendor,
            level_count=reader.level_count,
            levels=levels,
            tile_size=tile_size,
            tiles_at_level_0=base.tiles,
            target_mpp=target_mpp,
            working_level=working_level,
            working_mpp=working_mpp,
            working_downsample=working_downsample,
            exact_level_match=exact,
            file_size_mb=round(path.stat().st_size / (1024 * 1024), 1),
            associated_images=reader.associated_image_names(),
        )


class SlideReaderService:
    """Reads uploaded slides for step 1 of the pipeline."""

    def readout(
        self,
        upload_id: str,
        *,
        target_mpp: float | None = None,
        mpp_override: float | None = None,
    ) -> SlideReadout:
        """The step-1 readout for a finished upload.

        `target_mpp` defaults to the configured working resolution;
        `mpp_override` supplies a scale for files that record none.
        """
        path = resolve_ready_path(upload_id=upload_id)
        readout = build_readout(
            path,
            target_mpp=target_mpp if target_mpp and target_mpp > 0 else settings.target_mpp,
            tile_size=settings.tile_size,
            mpp_override=mpp_override,
        )
        readout.upload_id = upload_id
        return readout

    def thumbnail_png(self, upload_id: str, *, max_size: int = 1024) -> bytes:
        """A whole-slide overview as PNG bytes.

        Read from the pyramid, never from the file's associated images - on
        scanner output those can be a half-resolution copy of the whole slide,
        or a photograph of the label carrying case identifiers.
        """
        path = resolve_ready_path(upload_id=upload_id)
        max_size = max(64, min(max_size, 4096))

        with open_slide(path) as reader:
            image = reader.thumbnail_pil(max_size)

        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def region_png(
        self,
        upload_id: str,
        *,
        x: int,
        y: int,
        level: int,
        width: int,
        height: int,
    ) -> bytes:
        """One region of the slide as PNG bytes, for the tile viewer.

        `x` and `y` are level-0 coordinates, as OpenSlide and tiffslide define
        them; `width` and `height` are in pixels at the requested level.
        """
        path = resolve_ready_path(upload_id=upload_id)
        width = max(1, min(width, 2048))
        height = max(1, min(height, 2048))

        with open_slide(path) as reader:
            if level < 0 or level >= reader.level_count:
                raise ValueError(f"level {level} out of range (0..{reader.level_count - 1})")
            image = reader.read_region_pil((x, y), level, (width, height))

        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


slide_reader_service = SlideReaderService()


STAGE_ID = "read-slide"


def run(context: PipelineContext) -> StepResult:
    """The orchestrator-facing entry point: read the slide named by the context.

    Thin on purpose - it is the same `SlideReaderService.readout` the API
    calls, just handed a `PipelineContext` instead of query parameters, so
    there is exactly one implementation of step 1 to keep correct.
    """
    readout = slide_reader_service.readout(context.upload_id)
    return StepResult(stage_id=STAGE_ID, output=readout)
