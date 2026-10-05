"""Schemas for slide upload policy and the step-1 readout."""

from pydantic import Field

from app.schemas.common import APIModel


class UploadCapability(APIModel):
    """Advertises whether slide upload is currently accepted."""

    enabled: bool
    reason: str
    accepted_formats: list[str]
    max_file_size_mb: int
    chunk_size_bytes: int


class PyramidLevel(APIModel):
    """One zoom level of a whole-slide image."""

    level: int
    width: int
    height: int
    downsample: float = Field(description="Level-0 pixels per pixel at this level")
    mpp: float | None = Field(default=None, description="Microns per pixel at this level")
    tiles: int = Field(description="Tiles this level covers at the configured tile size")
    is_working_level: bool = Field(
        default=False, description="Whether this is the level chosen for the target mpp"
    )


class SlideReadout(APIModel):
    """Step 1 output: what the pipeline needs to know before it can start.

    The important field is `mpp`. Every downstream step is defined in microns
    per pixel, and the working level is derived from it - never from a
    hard-coded level index, because the same index means different resolutions
    on different scanners.
    """

    upload_id: str
    filename: str

    marker_letter: str | None = Field(
        default=None,
        description=(
            "The antibody letter read off the filename - 'HE', or one of AFRUW - or "
            "null when the name does not follow OncoStem's convention. A SUGGESTION, "
            "never a decision: see `app.panel.infer_from_filename`. A silently "
            "mis-detected marker measures the right cells in the wrong compartment, "
            "which is the kind of wrong that does not look like an error, so the UI "
            "must present this as 'we think' and let a person correct it."
        ),
    )
    marker: str | None = Field(
        default=None,
        description="Display name for `marker_letter` - 'H&E', 'CD44' - or null",
    )
    slide_role: str = Field(
        default="unknown",
        description=(
            "Which half of a case this slide looks like: 'he', 'ihc', or 'unknown'. "
            "Matches `SlideRole` in the pipeline catalogue, so a panel can check that "
            "the slide it is showing is the one its step declares it runs on."
        ),
    )

    width_px: int
    height_px: int
    megapixels: float

    mpp: float | None = Field(
        description="Microns per pixel at level 0, from the scanner or supplied by the caller"
    )
    mpp_source: str = Field(
        default="unknown",
        description=(
            "Where `mpp` came from: 'scanner' if the file recorded it, 'override' if the "
            "caller supplied it, 'unknown' if neither. An overridden scale is an assertion "
            "by whoever set it, not a measurement - the UI must say so."
        ),
    )
    scanner_mpp: float | None = Field(
        default=None,
        description="What the file itself recorded, kept even when overridden",
    )
    magnification: str
    objective_power: float | None = None
    vendor: str | None = None

    level_count: int
    levels: list[PyramidLevel]

    tile_size: int
    tiles_at_level_0: int

    target_mpp: float = Field(description="The resolution the pipeline wants to work at")
    working_level: int = Field(description="The level chosen for `target_mpp`")
    working_mpp: float | None = None
    working_downsample: float = Field(
        default=1.0,
        description=(
            "Extra downsample applied in software after reading the working level, to land "
            "exactly on target_mpp. A pyramid rarely has a level at the resolution you want, "
            "so the reader takes the nearest level that is at-or-finer and shrinks it - "
            "upsampling a coarser level would invent detail that was never scanned."
        ),
    )
    exact_level_match: bool = Field(
        default=False,
        description="Whether a pyramid level sits on target_mpp exactly (no software resample)",
    )

    file_size_mb: float
    associated_images: list[str] = Field(
        default_factory=list,
        description=(
            "Names only. Label and macro images carry case identifiers and are never served."
        ),
    )
