"""Pipeline catalogue tests."""

from fastapi.testclient import TestClient


def test_lists_sixteen_stages_in_order(client: TestClient) -> None:
    response = client.get("/api/v1/pipeline/stages")
    assert response.status_code == 200

    body = response.json()
    assert body["total"] == 16
    assert [stage["index"] for stage in body["items"]] == list(range(1, 17))


def test_exactly_one_stage_trains_a_model(client: TestClient) -> None:
    body = client.get("/api/v1/pipeline/summary").json()
    assert body["trainedSteps"] == 1


def test_only_the_first_eight_steps_are_implemented(client: TestClient) -> None:
    """Every other step is documented, not built. Saying otherwise would be a lie.

    Also pins the *order*: the built steps have to be a prefix of the pipeline,
    because each one consumes the previous one's output. A gap in this list would
    mean a step claiming to run on an input nothing produces.
    """
    items = client.get("/api/v1/pipeline/stages").json()["items"]
    assert [s["id"] for s in items if s["implemented"]] == [
        "read-slide",
        "quality-control",
        "tissue-mask",
        "white-calibration",
        "optical-density",
        "colour-deconvolution",
        "tiling",
        "tissue-type-segmentation",
    ]


def test_the_summary_agrees_with_the_catalogue_about_what_is_built(
    client: TestClient,
) -> None:
    items = client.get("/api/v1/pipeline/stages").json()["items"]
    built = sum(1 for stage in items if stage["implemented"])
    assert client.get("/api/v1/pipeline/summary").json()["implementedSteps"] == built


def test_stages_carry_no_example_outputs(client: TestClient) -> None:
    """A step that does not run must not ship numbers that look like results."""
    items = client.get("/api/v1/pipeline/stages").json()["items"]
    for stage in items:
        assert "metrics" not in stage
        assert "layers" not in stage


def test_the_trained_stage_is_tissue_type_segmentation(client: TestClient) -> None:
    items = client.get("/api/v1/pipeline/stages").json()["items"]
    trained = [stage["id"] for stage in items if stage["trainsModel"]]
    assert trained == ["tissue-type-segmentation"]


def test_unknown_stage_returns_404(client: TestClient) -> None:
    assert client.get("/api/v1/pipeline/stages/not-a-stage").status_code == 404
