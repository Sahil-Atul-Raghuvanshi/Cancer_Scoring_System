"""Upload capability tests."""

from fastapi.testclient import TestClient


def test_upload_is_enabled(client: TestClient) -> None:
    body = client.get("/api/v1/slides/upload-capability").json()
    assert body["enabled"] is True
    assert ".svs" in body["acceptedFormats"]


def test_capability_reports_the_chunk_size_the_client_should_use(client: TestClient) -> None:
    body = client.get("/api/v1/slides/upload-capability").json()
    assert body["chunkSizeBytes"] > 0
    assert body["maxFileSizeMb"] > 0
