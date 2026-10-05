"""The five-antibody panel, static for the UI's biomarker picker."""

from fastapi import APIRouter

from app.panel import PANEL
from app.schemas.panel import MarkerInfo, PanelResponse

router = APIRouter(prefix="/panel", tags=["panel"])


@router.get("", response_model=PanelResponse, summary="The five antibodies plus H&E")
async def get_panel() -> PanelResponse:
    markers = [
        MarkerInfo(
            letter=marker_spec.letter,
            name=marker_spec.name,
            full_name=marker_spec.full_name,
            compartment=marker_spec.compartment.value,
            scored=marker_spec.scored,
            compartment_width_um=marker_spec.compartment_width_um,
            second_measure=marker_spec.second_measure,
            expected_percent_min=marker_spec.expected_percent[0],
            expected_percent_max=marker_spec.expected_percent[1],
        )
        for marker_spec in PANEL.values()
    ]
    return PanelResponse(markers=markers)
