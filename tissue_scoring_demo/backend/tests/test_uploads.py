"""Chunked upload protocol + step 1 slide reading.

Exercises the real 3-call flow against a real generated image, so the atomic
chunk writes, the reassembly, the checksum check and the "prove it opens"
validation are all covered rather than mocked.
"""

import hashlib
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _sample_png(width: int = 1400, height: int = 1100) -> bytes:
    """A plain PNG, which the image reader opens with a synthetic pyramid."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    assert pixels is not None
    for y in range(height):
        for x in range(0, width, 7):
            pixels[x, y] = ((x * 3) % 256, (y * 5) % 256, ((x + y) * 2) % 256)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _upload(client: TestClient, data: bytes, filename: str, chunk_size: int) -> str:
    """Run the full init -> chunks -> complete flow, returning the upload id."""
    num_chunks = max(1, -(-len(data) // chunk_size))
    init = client.post(
        "/api/v1/uploads",
        json={
            "filename": filename,
            "totalSize": len(data),
            "numChunks": num_chunks,
            "chunkSize": chunk_size,
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    )
    assert init.status_code == 201, init.text
    upload_id = init.json()["uploadId"]

    for index in range(num_chunks):
        part = data[index * chunk_size : (index + 1) * chunk_size]
        chunk = client.put(f"/api/v1/uploads/{upload_id}/chunks/{index}", content=part)
        assert chunk.status_code == 200, chunk.text

    done = client.post(f"/api/v1/uploads/{upload_id}/complete")
    assert done.status_code == 200, done.text
    return upload_id


@pytest.fixture(scope="module")
def ready_upload(client: TestClient) -> str:
    """One slide, uploaded in several chunks and finalised."""
    upload_id = _upload(client, _sample_png(), "sample-slide.png", chunk_size=8 * 1024)
    status = client.get(f"/api/v1/uploads/{upload_id}").json()
    assert status["state"] == "ready", status
    return upload_id


# --- protocol ---------------------------------------------------------------


def test_upload_is_enabled(client: TestClient) -> None:
    body = client.get("/api/v1/slides/upload-capability").json()
    assert body["enabled"] is True
    assert ".svs" in body["acceptedFormats"]
    assert body["chunkSizeBytes"] > 0


def test_rejects_unsupported_extension_before_any_bytes_move(client: TestClient) -> None:
    response = client.post(
        "/api/v1/uploads",
        json={"filename": "notes.pdf", "totalSize": 1024, "numChunks": 1},
    )
    assert response.status_code == 400
    assert "unsupported slide type" in response.json()["detail"]


def test_rejects_oversized_declaration(client: TestClient) -> None:
    response = client.post(
        "/api/v1/uploads",
        json={"filename": "huge.svs", "totalSize": 99 * 1024**3, "numChunks": 1},
    )
    assert response.status_code == 400
    assert "out of range" in response.json()["detail"]


def test_multi_chunk_upload_becomes_ready(ready_upload: str, client: TestClient) -> None:
    status = client.get(f"/api/v1/uploads/{ready_upload}").json()
    assert status["state"] == "ready"
    assert status["receivedCount"] == status["numChunks"]
    assert status["numChunks"] > 1, "the fixture should exercise more than one chunk"


def test_resending_a_chunk_is_idempotent(client: TestClient) -> None:
    data = _sample_png(400, 300)
    init = client.post(
        "/api/v1/uploads",
        json={"filename": "retry.png", "totalSize": len(data), "numChunks": 2, "chunkSize": 8192},
    ).json()
    upload_id = init["uploadId"]

    first = data[:8192]
    client.put(f"/api/v1/uploads/{upload_id}/chunks/0", content=first)
    again = client.put(f"/api/v1/uploads/{upload_id}/chunks/0", content=first)

    assert again.status_code == 200
    assert again.json()["received"] == [0], "a retry must not double-count"


def test_complete_reports_which_chunks_are_missing(client: TestClient) -> None:
    data = _sample_png(400, 300)
    init = client.post(
        "/api/v1/uploads",
        json={"filename": "gap.png", "totalSize": len(data), "numChunks": 4, "chunkSize": 8192},
    ).json()
    upload_id = init["uploadId"]
    client.put(f"/api/v1/uploads/{upload_id}/chunks/0", content=data[:8192])

    response = client.post(f"/api/v1/uploads/{upload_id}/complete")
    assert response.status_code == 400
    assert "chunk(s) missing" in response.json()["detail"]


def test_checksum_mismatch_fails_the_upload(client: TestClient) -> None:
    data = _sample_png(300, 200)
    init = client.post(
        "/api/v1/uploads",
        json={
            "filename": "corrupt.png",
            "totalSize": len(data),
            "numChunks": 1,
            "chunkSize": len(data),
            "sha256": "0" * 64,
        },
    ).json()
    upload_id = init["uploadId"]
    client.put(f"/api/v1/uploads/{upload_id}/chunks/0", content=data)
    client.post(f"/api/v1/uploads/{upload_id}/complete")

    status = client.get(f"/api/v1/uploads/{upload_id}").json()
    assert status["state"] == "failed"
    assert "checksum mismatch" in (status["error"] or "")


def test_unreadable_file_fails_validation(client: TestClient) -> None:
    """A transfer can be perfect and the upload still fail: it must be openable."""
    data = b"this is not an image" * 40
    init = client.post(
        "/api/v1/uploads",
        json={
            "filename": "bogus.tiff",
            "totalSize": len(data),
            "numChunks": 1,
            "chunkSize": len(data),
        },
    ).json()
    upload_id = init["uploadId"]
    client.put(f"/api/v1/uploads/{upload_id}/chunks/0", content=data)
    client.post(f"/api/v1/uploads/{upload_id}/complete")

    status = client.get(f"/api/v1/uploads/{upload_id}").json()
    assert status["state"] == "failed"
    assert status["error"]


def test_abort_discards_the_upload(client: TestClient) -> None:
    data = _sample_png(300, 200)
    init = client.post(
        "/api/v1/uploads",
        json={"filename": "abort.png", "totalSize": len(data), "numChunks": 2, "chunkSize": 8192},
    ).json()
    upload_id = init["uploadId"]
    client.put(f"/api/v1/uploads/{upload_id}/chunks/0", content=data[:8192])

    assert client.delete(f"/api/v1/uploads/{upload_id}").json()["state"] == "failed"


def test_unknown_upload_is_404(client: TestClient) -> None:
    assert client.get("/api/v1/uploads/does-not-exist").status_code == 404


def test_traversal_in_the_upload_id_is_rejected(client: TestClient) -> None:
    """The id becomes a path, so it must not be able to escape the staging dir."""
    assert client.get("/api/v1/uploads/..%2F..%2Fetc").status_code == 404


# --- step 1 -----------------------------------------------------------------


def test_readout_reports_the_pyramid(ready_upload: str, client: TestClient) -> None:
    body = client.get(f"/api/v1/slides/{ready_upload}/readout").json()

    assert body["uploadId"] == ready_upload
    assert body["widthPx"] == 1400
    assert body["heightPx"] == 1100
    assert body["levelCount"] >= 2
    assert body["levels"][0]["level"] == 0
    assert body["levels"][0]["downsample"] == 1.0
    # Downsample must increase monotonically down the pyramid.
    factors = [level["downsample"] for level in body["levels"]]
    assert factors == sorted(factors)


def test_readout_marks_exactly_one_working_level(ready_upload: str, client: TestClient) -> None:
    body = client.get(f"/api/v1/slides/{ready_upload}/readout").json()
    flagged = [level for level in body["levels"] if level["isWorkingLevel"]]
    assert len(flagged) == 1
    assert flagged[0]["level"] == body["workingLevel"]


def test_readout_has_no_mpp_for_a_plain_image(ready_upload: str, client: TestClient) -> None:
    """A plain PNG carries no physical scale, so mpp must be null, not a guess."""
    body = client.get(f"/api/v1/slides/{ready_upload}/readout").json()
    assert body["mpp"] is None
    assert body["magnification"] == "unknown"
    assert body["workingLevel"] == 0, "with no mpp there is nothing to convert against"


def test_tile_counts_are_reported(ready_upload: str, client: TestClient) -> None:
    body = client.get(f"/api/v1/slides/{ready_upload}/readout").json()
    assert body["tileSize"] > 0
    # 1400x1100 at 512 px tiles = 3 x 3
    assert body["tilesAtLevel0"] == 9


def test_thumbnail_is_a_png(ready_upload: str, client: TestClient) -> None:
    response = client.get(f"/api/v1/slides/{ready_upload}/thumbnail?maxSize=256")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    image = Image.open(io.BytesIO(response.content))
    assert max(image.size) <= 256


def test_region_is_a_png_of_the_requested_size(ready_upload: str, client: TestClient) -> None:
    response = client.get(
        f"/api/v1/slides/{ready_upload}/region?x=64&y=64&level=0&width=128&height=96"
    )
    assert response.status_code == 200
    assert Image.open(io.BytesIO(response.content)).size == (128, 96)


def test_region_rejects_a_level_that_does_not_exist(ready_upload: str, client: TestClient) -> None:
    response = client.get(f"/api/v1/slides/{ready_upload}/region?level=99")
    assert response.status_code == 422


def test_readout_on_an_unfinished_upload_is_rejected(client: TestClient) -> None:
    data = _sample_png(300, 200)
    init = client.post(
        "/api/v1/uploads",
        json={
            "filename": "pending.png",
            "totalSize": len(data),
            "numChunks": 2,
            "chunkSize": 8192,
        },
    ).json()

    response = client.get(f"/api/v1/slides/{init['uploadId']}/readout")
    assert response.status_code == 409
    assert "not ready" in response.json()["detail"]


# --- adjustable resolution --------------------------------------------------


def test_scanner_mpp_is_labelled_as_such(ready_upload: str, client: TestClient) -> None:
    """A plain PNG records no scale, so nothing may be presented as measured."""
    body = client.get(f"/api/v1/slides/{ready_upload}/readout").json()
    assert body["mppSource"] == "unknown"
    assert body["mpp"] is None
    assert body["scannerMpp"] is None


def test_mpp_override_supplies_a_scale(ready_upload: str, client: TestClient) -> None:
    body = client.get(f"/api/v1/slides/{ready_upload}/readout?mppOverride=0.25").json()

    assert body["mpp"] == 0.25
    assert body["mppSource"] == "override", "an asserted scale must not look measured"
    assert body["scannerMpp"] is None, "the file still records nothing"
    # Every level's mpp now derives from the override.
    assert body["levels"][0]["mpp"] == 0.25
    assert body["levels"][1]["mpp"] == 0.5


def test_override_makes_the_working_level_meaningful(ready_upload: str, client: TestClient) -> None:
    """Without a scale the reader falls back to level 0; with one it can choose."""
    plain = client.get(f"/api/v1/slides/{ready_upload}/readout?targetMpp=1.0").json()
    assert plain["workingLevel"] == 0, "no scale, nothing to convert against"

    scaled = client.get(
        f"/api/v1/slides/{ready_upload}/readout?targetMpp=1.0&mppOverride=0.25"
    ).json()
    assert scaled["workingLevel"] > 0, "0.25 -> 1.0 um/px is two levels down a 2x pyramid"
    assert scaled["levels"][scaled["workingLevel"]]["mpp"] <= 1.0


def test_target_mpp_moves_the_working_level(ready_upload: str, client: TestClient) -> None:
    """Asking for a coarser target selects a coarser level, never a finer one."""
    levels = []
    for target in (0.25, 0.5, 1.0, 2.0, 4.0):
        body = client.get(
            f"/api/v1/slides/{ready_upload}/readout?targetMpp={target}&mppOverride=0.25"
        ).json()
        levels.append(body["workingLevel"])

    assert levels == sorted(levels), f"working level must not go backwards: {levels}"
    assert levels[0] < levels[-1], "a 16x coarser target should land on a different level"


def test_working_level_is_never_coarser_than_the_target(
    ready_upload: str, client: TestClient
) -> None:
    """Reading a coarser level then upsampling would invent detail."""
    for target in (0.3, 0.5, 0.9, 1.5, 3.0):
        body = client.get(
            f"/api/v1/slides/{ready_upload}/readout?targetMpp={target}&mppOverride=0.25"
        ).json()
        working = body["levels"][body["workingLevel"]]
        assert working["mpp"] <= target + 1e-6, (
            f"target {target}: chose level {working['level']} at {working['mpp']} um/px"
        )
        assert body["workingDownsample"] >= 1.0, "software resample must only ever shrink"


def test_software_downsample_lands_on_the_target(ready_upload: str, client: TestClient) -> None:
    body = client.get(
        f"/api/v1/slides/{ready_upload}/readout?targetMpp=0.6&mppOverride=0.25"
    ).json()
    working = body["levels"][body["workingLevel"]]
    assert abs(working["mpp"] * body["workingDownsample"] - 0.6) < 1e-3


def test_exact_match_needs_no_resample(ready_upload: str, client: TestClient) -> None:
    body = client.get(
        f"/api/v1/slides/{ready_upload}/readout?targetMpp=0.5&mppOverride=0.25"
    ).json()
    assert body["exactLevelMatch"] is True
    assert body["workingDownsample"] == 1.0


def test_target_mpp_is_range_checked(ready_upload: str, client: TestClient) -> None:
    assert client.get(f"/api/v1/slides/{ready_upload}/readout?targetMpp=0").status_code == 422
    assert client.get(f"/api/v1/slides/{ready_upload}/readout?targetMpp=-1").status_code == 422
    assert client.get(f"/api/v1/slides/{ready_upload}/readout?mppOverride=0").status_code == 422


# --- Deep Zoom tiles --------------------------------------------------------


def test_dzi_descriptor_matches_the_slide(ready_upload: str, client: TestClient) -> None:
    from xml.etree import ElementTree

    response = client.get(f"/api/v1/slides/{ready_upload}.dzi")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")

    root = ElementTree.fromstring(response.text)
    size = root.find("{http://schemas.microsoft.com/deepzoom/2008}Size")
    assert size is not None
    assert int(size.get("Width") or 0) == 1400
    assert int(size.get("Height") or 0) == 1100
    assert root.get("Format") == "jpeg"
    assert int(root.get("TileSize") or 0) > 0


def test_tiles_are_jpeg(ready_upload: str, client: TestClient) -> None:
    response = client.get(f"/api/v1/slides/{ready_upload}_files/8/0_0.jpeg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8", "not a JPEG"


def test_tiles_are_cacheable(ready_upload: str, client: TestClient) -> None:
    """Tiles are immutable for an upload id, so they must not be re-fetched."""
    response = client.get(f"/api/v1/slides/{ready_upload}_files/8/0_0.jpeg")
    assert "immutable" in response.headers.get("cache-control", "")


def test_zooming_in_yields_more_tiles(ready_upload: str, client: TestClient) -> None:
    """Each DZI level doubles, so a deeper level must have tiles where a
    shallower one does not."""
    shallow = client.get(f"/api/v1/slides/{ready_upload}_files/6/1_0.jpeg")
    deep = client.get(f"/api/v1/slides/{ready_upload}_files/10/1_0.jpeg")
    assert deep.status_code == 200
    assert shallow.status_code == 404


def test_out_of_range_tiles_are_404_not_500(ready_upload: str, client: TestClient) -> None:
    """A viewport routinely asks past the edge; that is not a server fault."""
    for path in (
        f"/api/v1/slides/{ready_upload}_files/8/999_0.jpeg",
        f"/api/v1/slides/{ready_upload}_files/8/0_999.jpeg",
        f"/api/v1/slides/{ready_upload}_files/99/0_0.jpeg",
    ):
        assert client.get(path).status_code == 404, path


def test_dzi_for_an_unknown_slide_is_404(client: TestClient) -> None:
    assert client.get("/api/v1/slides/no-such-slide.dzi").status_code == 404


def test_dzi_route_does_not_shadow_the_readout_route(
    ready_upload: str, client: TestClient
) -> None:
    """`/{id}.dzi` and `/{id}/readout` must stay distinguishable."""
    assert client.get(f"/api/v1/slides/{ready_upload}/readout").status_code == 200
    assert client.get(f"/api/v1/slides/{ready_upload}.dzi").status_code == 200


def test_tile_cache_survives_concurrent_requests(ready_upload: str, client: TestClient) -> None:
    """A viewer opens a screenful at once; the shared reader must hold up."""
    from concurrent.futures import ThreadPoolExecutor

    coords = [(10, c, r) for r in range(3) for c in range(4)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(
                lambda t: client.get(
                    f"/api/v1/slides/{ready_upload}_files/{t[0]}/{t[1]}_{t[2]}.jpeg"
                ),
                coords,
            )
        )

    ok = [r for r in responses if r.status_code == 200]
    assert ok, "no tiles were served"
    assert all(r.content[:2] == b"\xff\xd8" for r in ok), "a concurrent read returned corrupt data"


# --- P-20: the override is the slide's scale everywhere, not only on step 1 ---------


def test_the_override_reaches_every_later_reader(ready_upload: str, client: TestClient) -> None:
    """Every step after 1 opens the slide itself; it must see the scale step 1 was given."""
    from app.ingestion.slide_reader import open_slide
    from app.services.upload_service import resolve_ready_path

    path = resolve_ready_path(upload_id=ready_upload)
    with open_slide(path) as reader:
        assert reader.mpp is None, "a plain PNG records no scale"

    client.get(f"/api/v1/slides/{ready_upload}/readout?mppOverride=0.25")
    with open_slide(path) as reader:
        assert reader.mpp == 0.25

    # The upload is still readable with the new field in its record.
    assert client.get(f"/api/v1/uploads/{ready_upload}").json()["state"] == "ready"

    client.get(f"/api/v1/slides/{ready_upload}/readout")  # the screen cleared it
    with open_slide(path) as reader:
        assert reader.mpp is None


def test_a_changed_scale_discards_what_was_measured_at_the_old_one(
    ready_upload: str, client: TestClient
) -> None:
    from app.core.config import settings

    own = settings.tissue_dir / ready_upload
    pair = settings.per_cell_dir / f"otherHe__{ready_upload}"
    unrelated = settings.tissue_dir / "someoneElse"
    for directory in (own, pair, unrelated):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text("{}", encoding="utf-8")

    client.get(f"/api/v1/slides/{ready_upload}/readout?mppOverride=0.25")
    assert not own.exists() and not pair.exists()
    assert unrelated.exists()

    # The same scale again changes nothing, so nothing is discarded.
    own.mkdir(parents=True)
    client.get(f"/api/v1/slides/{ready_upload}/readout?mppOverride=0.25")
    assert own.exists()


def test_non_square_pixels_are_refused_rather_than_measured_as_square(tmp_path) -> None:
    from app.ingestion.slide_reader import SlideReader, SlideScaleError

    reader = SlideReader.__new__(SlideReader)
    reader.path = tmp_path / "slide.svs"

    class _Slide:
        properties = {"tiffslide.mpp-x": "0.25", "tiffslide.mpp-y": "0.30"}

    reader._slide = _Slide()
    with pytest.raises(SlideScaleError, match="not square"):
        _ = reader.mpp

    _Slide.properties = {"tiffslide.mpp-x": "0.2500", "tiffslide.mpp-y": "0.2501"}
    assert reader.mpp == pytest.approx(0.25)
