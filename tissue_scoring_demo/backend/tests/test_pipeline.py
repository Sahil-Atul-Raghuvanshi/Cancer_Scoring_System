"""Pipeline catalogue tests."""

from fastapi.testclient import TestClient


def test_lists_nineteen_stages_in_order(client: TestClient) -> None:
    response = client.get("/api/v1/pipeline/stages")
    assert response.status_code == 200

    body = response.json()
    assert body["total"] == 19
    assert [stage["index"] for stage in body["items"]] == list(range(1, 20))


def test_exactly_one_stage_trains_a_model(client: TestClient) -> None:
    body = client.get("/api/v1/pipeline/summary").json()
    assert body["trainedSteps"] == 1


def test_every_step_is_implemented_and_in_order(client: TestClient) -> None:
    """All nineteen run. Saying otherwise - in either direction - would be a lie.

    Also pins the *order*: the built steps have to be a prefix of the pipeline,
    because each one consumes the previous one's output. A gap in this list would
    mean a step claiming to run on an input nothing produces.

    Step 19 is on this list and is a special case worth stating. It runs, and
    what it reports today is that there are no pathologist readings on disk to
    compare against. That is a result of the step rather than a gap in it - the
    alternative, leaving it unbuilt, would make "we have not validated this"
    something a reader has to infer from an absence instead of something the
    pipeline says.
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
        "roi-mask",
        "roi-selection",
        "roi-refinement",
        "ihc-alignment",
        "nuclei-segmentation",
        "cell-typing",
        "compartments",
        "per-cell-measurement",
        "intensity-binning",
        "aggregate",
        "validation",
    ]


def test_the_runner_holds_every_step_in_catalogue_order(client: TestClient) -> None:
    """The runner and the catalogue are two lists of the same seventeen steps.

    They are written down separately - one is what the API describes, the other
    is what actually executes - so nothing but a test stops them drifting. A
    step present in one and missing from the other is a step that either never
    runs or never appears.
    """
    from app.pipeline.runner import STEP_FUNCTIONS

    items = client.get("/api/v1/pipeline/stages").json()["items"]
    catalogue = [stage["id"] for stage in items]

    import importlib

    runner_ids = [
        importlib.import_module(step.__module__).STAGE_ID for step in STEP_FUNCTIONS
    ]
    assert runner_ids == catalogue


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
