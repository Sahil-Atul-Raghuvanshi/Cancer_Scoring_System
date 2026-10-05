"""Schemas for step 13 - where in each cell the brown is supposed to be."""

from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel

#: The fork. Which one a marker gets is `app.panel`'s decision, not a request's.
CompartmentKind = Literal["membrane", "cytoplasm"]


class CompartmentParams(APIModel):
    """The widths, in microns. Neither default is established fact.

    Both come from QuPath's convention plus the physical size of a breast
    epithelial cell, which is exactly why the sensitivity sweep below travels
    with them: a score that swings fifteen points between a 3 um and a 5 um ring
    is a score with a hidden parameter in it.
    """

    #: Total distance the cell is grown outward from its nucleus.
    expansion_um: float
    #: For a membrane marker, how thick the band at the outer edge is. Ignored
    #: for a cytoplasmic marker, where the whole cell body is measured.
    ring_um: float | None = None
    #: Off only to demonstrate why it should be on.
    voronoi_constrained: bool = True


class FieldCompartments(APIModel):
    """One sampled field's compartments, as areas rather than pixels."""

    region_rank: int
    index: int
    cells: int
    #: Mean measured-compartment area per cell, in square microns.
    mean_measured_um2: float
    mean_cell_um2: float
    mean_nucleus_um2: float
    #: Pixels a nearer neighbour claimed - what the Voronoi constraint prevented
    #: from being counted twice.
    contested_px: int


class RegionCompartments(APIModel):
    rank: int
    cells: int
    mean_measured_um2: float
    mean_cell_um2: float
    mean_nucleus_um2: float
    #: Share of the expanded area that two cells would have fought over under
    #: plain dilation. The size of the error the constraint is preventing.
    contested_share: float
    fields: list[FieldCompartments]


class WidthSensitivityPoint(APIModel):
    width_um: float
    mean_measured_um2: float
    contested_share: float


class CompartmentsReport(APIModel):
    """Step 13's result: the geometry the measurement will happen in."""

    he_upload_id: str
    ihc_upload_id: str
    marker: str | None = None
    marker_name: str | None = None
    #: The fork, resolved. Read from `app.panel`, never from the request.
    compartment: CompartmentKind
    #: What step 14 will compute for this marker: ring completeness for a
    #: membrane marker, stained fraction for a cytoplasmic one.
    second_measure: str | None = None

    generated_at: str
    nuclei_generated_at: str | None = None

    params: CompartmentParams
    regions: list[RegionCompartments]

    cells: int = 0
    mean_measured_um2: float = 0.0
    mean_cell_um2: float = 0.0

    #: How the compartment area moves with the width. Published because neither
    #: default is established fact.
    width_sensitivity: list[WidthSensitivityPoint] = Field(default_factory=list)

    #: Only tumour cells get compartments, per step 12. Carried so the two
    #: screens cannot disagree about the denominator.
    tumour_only: bool = True
    typed_cells: int | None = None

    notes: list[str] = Field(default_factory=list)
