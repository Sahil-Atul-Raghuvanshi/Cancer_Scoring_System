"""Schemas for the five-antibody panel."""

from typing import Literal

from app.schemas.common import APIModel


class MarkerInfo(APIModel):
    """One antibody, as the UI needs to render it."""

    letter: str
    name: str
    full_name: str
    compartment: Literal["membrane", "cytoplasm", "none"]
    scored: bool
    compartment_width_um: float
    second_measure: str
    expected_percent_min: int
    expected_percent_max: int


class PanelResponse(APIModel):
    markers: list[MarkerInfo]
