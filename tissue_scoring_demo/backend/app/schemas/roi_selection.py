"""Schemas for step 10 - reviewing and selecting the candidate regions.

Step 10 reports three things:

  the candidates    every invasive patch step 8 drew that is worth offering, largest
                    first, each with its box, its boundary, the windows it is made of
                    and how sure the model was. The product.
  the selection     which of them are ticked, and whether that is still the default
                    the pipeline chose or a person's own answer.
  the omissions     how many patches were not offered and how much tissue that was.
                    Reported rather than absorbed, because a screen headed "the
                    invasive regions" that silently shows 18 of 47 is making a
                    different claim than it looks like it is making.

**Nothing here is a measurement.** Every number on a candidate came out of step 9 and
is echoed, not recomputed - the area is step 9's cell count, the confidence is the tile
model's own probability. This step's only output that did not exist before is the set of
ticked ids.

See `pipeline/step10_roi_selection/`.
"""

from pydantic import Field

from app.schemas.common import APIModel


class RoiCandidateModel(APIModel):
    """One coarse invasive region, offered for review."""

    roi_id: str = Field(
        description=(
            "Stable id within this class map, `ROI-001` upward, ranked by area - "
            "`ROI-001` is the largest invasive patch on the slide. A class map rebuilt "
            "at different parameters is a different set of regions and the ids do not "
            "carry across it; `classMapKey` on the report is what pins that down."
        )
    )
    index: int = Field(description="Step 9's own 0-based rank for this region, largest first")

    x: int = Field(description="Bounding box origin in H&E level-0 pixels")
    y: int
    width: int
    height: int

    candidate_class: str = Field(description="What step 8 called it. Always `invasive` today.")
    confidence: float = Field(
        description=(
            "Mean probability the tile model gave this region's own class, over the "
            "windows it is made of. A description of the model's call, not of the "
            "tissue, and not a gate anywhere."
        )
    )
    cells: int = Field(description="Windows of step 8's grid this region covers")
    area_mm2: float

    #: Level-0 polygon rings, outer first then holes - the same convention and the same
    #: frame as `RoiRegionModel.rings`, so a client that can draw one can draw the other.
    rings: list[list[tuple[int, int]]]

    #: `(row, col)` of every window of step 8's grid inside this region, and the same
    #: list spelled `r<row>c<col>`. Carried so a candidate can be reproduced from the
    #: class map alone rather than from a picture of it.
    tile_cells: list[tuple[int, int]] = Field(default_factory=list)
    tile_ids: list[str] = Field(default_factory=list)

    #: Windows BEETLE would run on this region once its box is padded for context. The
    #: step's cost, per candidate, before any of it is spent.
    windows: int = 0


class RoiSelectionReport(APIModel):
    """Step 10's payload: what may be refined, and what is currently ticked."""

    upload_id: str
    generated_at: str

    slide_width: int = Field(description="H&E width in level-0 pixels - the frame `rings` is in")
    slide_height: int

    candidates: list[RoiCandidateModel]

    #: Ids currently ticked. Ordered the way the candidates are, so a client can zip the
    #: two without sorting.
    selected: list[str] = Field(default_factory=list)
    #: What the pipeline would tick on its own: enough of the largest regions to cover
    #: `roi_selection_default_coverage` of the offered area. Kept beside `selected` so a
    #: screen can show a person how far their answer is from the pipeline's.
    default_selection: list[str] = Field(default_factory=list)
    #: False until somebody has actually chosen. An unattended run refines the default
    #: and says so here rather than claiming a review that never happened - the same
    #: distinction step 12's `confirmedBy` draws.
    chosen_by_person: bool = False
    chosen_at: str | None = None

    #: Which class map these ids belong to. A selection is only meaningful against the
    #: class map it was made on, and this is what lets step 11 refuse a stale one
    #: instead of refining ids that now point at different tissue.
    class_map_key: str | None = None

    invasive_mm2: float = Field(description="All invasive carcinoma step 9 found")
    offered_mm2: float = Field(description="How much of it is on this screen")
    selected_mm2: float = Field(description="How much of it is ticked")
    selected_windows: int = Field(description="Forward-pass windows the ticked regions cost")

    dropped_small: int = Field(description="Patches below the minimum size, not offered")
    dropped_small_mm2: float
    dropped_capped: int = Field(description="Patches past the cap, not offered")
    dropped_capped_mm2: float

    notes: list[str] = Field(
        default_factory=list,
        description="What this selection commits to, in the words the UI shows",
    )


class RoiSelectionRequest(APIModel):
    """A person's answer: exactly which candidates go to BEETLE."""

    selected: list[str] = Field(
        description=(
            "The ids to refine. Replaces the current selection outright rather than "
            "merging into it - a partial update would make 'deselect all' impossible to "
            "express, and would leave a client unable to tell an empty list from a "
            "missing one."
        )
    )
