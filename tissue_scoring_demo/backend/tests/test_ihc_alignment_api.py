"""Step 10's HTTP surface: what it says before there is anything to say.

The registration itself is not exercised here - it is fitted beforehand by
`slide_registration/` on two gigabyte scans. What is pinned is
the contract around it, and in particular that every route fails *informatively*
on a pair that has not been aligned, rather than 500ing or inventing a result.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

HE = "he-upload-id"
IHC = "ihc-upload-id"


def test_capability_says_whether_registration_can_run_here(client: TestClient) -> None:
    body = client.get("/api/v1/ihc-alignment/capability").json()
    assert isinstance(body["available"], bool)
    assert body["serviceDir"]
    # A machine without the environment has to say why, not just "false".
    if not body["available"]:
        assert body["reason"]


def test_reading_a_report_that_does_not_exist_is_a_409(client: TestClient) -> None:
    response = client.get(f"/api/v1/ihc-alignment/{HE}?ihcUploadId={IHC}")
    assert response.status_code == 409
    assert "aligned" in response.json()["detail"]


def test_progress_for_a_pair_nobody_started_is_a_409(client: TestClient) -> None:
    response = client.get(f"/api/v1/ihc-alignment/{HE}/run?ihcUploadId={IHC}")
    assert response.status_code == 409


def test_panels_before_an_alignment_are_a_409(client: TestClient) -> None:
    response = client.get(
        f"/api/v1/ihc-alignment/{HE}/panels/ihc_borders.png?ihcUploadId={IHC}"
    )
    assert response.status_code == 409


def test_an_unknown_panel_name_is_rejected(client: TestClient) -> None:
    response = client.get(f"/api/v1/ihc-alignment/{HE}/panels/nonsense.png?ihcUploadId={IHC}")
    assert response.status_code == 409
    assert "he_borders" in response.json()["detail"]


def test_a_crop_rank_outside_the_carried_regions_is_rejected(client: TestClient) -> None:
    """A rank is now checked against what this pair actually carried.

    How many regions cross depends on how much tumour they cover, so there is no
    fixed number of ranks to validate against - and for a pair with no alignment
    at all the honest answer is that there is no alignment, not that rank 9 is
    out of a range nobody has established.
    """
    response = client.get(f"/api/v1/ihc-alignment/{HE}/crops/9.png?ihcUploadId={IHC}")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "rank" in detail or "aligned" in detail


def test_the_ihc_slide_is_required(client: TestClient) -> None:
    """Without it there is only one slide, and nothing to align it to."""
    assert client.get(f"/api/v1/ihc-alignment/{HE}").status_code == 422


def test_running_against_an_upload_that_does_not_exist_is_a_404(client: TestClient) -> None:
    response = client.post(f"/api/v1/ihc-alignment/{HE}/run?ihcUploadId={IHC}")
    assert response.status_code == 404


def test_a_slide_cannot_be_aligned_to_itself(client: TestClient) -> None:
    response = client.post(f"/api/v1/ihc-alignment/{HE}/run?ihcUploadId={HE}")
    assert response.status_code == 409
    assert "same upload" in response.json()["detail"]


def test_confirming_an_alignment_that_does_not_exist_is_a_409(client: TestClient) -> None:
    response = client.post(f"/api/v1/ihc-alignment/{HE}/confirm?ihcUploadId={IHC}")
    assert response.status_code == 409


def test_the_step_is_in_the_catalogue_between_roi_and_nuclei(client: TestClient) -> None:
    items = client.get("/api/v1/pipeline/stages").json()["items"]
    order = [stage["id"] for stage in items]
    # Two steps now sit between the coarse ROI and the registration: a person picks
    # which candidates matter, then BEETLE traces them per pixel. What crosses here is
    # that boundary, which is why the gap exists.
    assert order.index("roi-mask") + 3 == order.index("ihc-alignment")
    assert order.index("ihc-alignment") + 1 == order.index("nuclei-segmentation")


def test_the_step_declares_itself_built(client: TestClient) -> None:
    stage = client.get("/api/v1/pipeline/stages/ihc-alignment").json()
    assert stage["implemented"] is True
    assert stage["index"] == 12
