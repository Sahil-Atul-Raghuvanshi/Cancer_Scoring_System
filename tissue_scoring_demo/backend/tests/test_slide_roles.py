"""Which slide each step runs on, and that the catalogue keeps saying so.

The pipeline is not a line. Steps 1-6 are a per-slide prefix that runs once on the
H&E and once on the immunostained slide; 7-11 find and refine the region on the H&E;
12 carries it across; 13-18 measure on the IHC slide; 19 is keyed on the case. Before
`runs_on` existed that shape lived only in prose, and the UI drew an H&E tile under a
step whose output is the DAB the score is made of.

These tests pin the shape rather than the wording, so a step that quietly changes
which slide it reads has to change this file too.
"""

from fastapi.testclient import TestClient

from app.data.pipeline_steps import PIPELINE_STAGES

#: The per-slide prefix. Every one of these measures a property of one piece of
#: glass - an artefact map, a tissue footprint, a white point, a density, an
#: un-mixing - so borrowing the other slide's answer is never right.
PER_SLIDE_PREFIX = (
    "read-slide",
    "quality-control",
    "tissue-mask",
    "white-calibration",
    "optical-density",
    "colour-deconvolution",
)

#: The region arm. H&E only, on evidence: the h_channel model returns 0% invasive
#: on real IHC at 0.96+ confidence, which is why step 12 exists at all. The two
#: refinement steps join it for the same reason one model further on - BEETLE traces
#: the boundary where structure is legible, and step 12 carries it across.
HE_ONLY = (
    "tiling",
    "tissue-type-segmentation",
    "roi-mask",
    "roi-selection",
    "roi-refinement",
)

#: The measurement arm. The brown being scored is only on this slide.
IHC_ONLY = (
    "nuclei-segmentation",
    "cell-typing",
    "compartments",
    "per-cell-measurement",
    "intensity-binning",
    "aggregate",
)


def _roles(stage_id: str) -> list[str]:
    stage = next(s for s in PIPELINE_STAGES if s.id == stage_id)
    return [role.value for role in stage.runs_on]


def test_every_stage_declares_at_least_one_slide() -> None:
    """An empty `runs_on` would let a panel fall back to "whatever was loaded"."""
    for stage in PIPELINE_STAGES:
        assert stage.runs_on, f"{stage.id} declares no slide"


def test_the_prefix_runs_on_both_slides() -> None:
    for stage_id in PER_SLIDE_PREFIX:
        assert _roles(stage_id) == ["he", "ihc"], stage_id


def test_the_region_arm_is_he_only() -> None:
    for stage_id in HE_ONLY:
        assert _roles(stage_id) == ["he"], stage_id


def test_the_measurement_arm_is_ihc_only() -> None:
    for stage_id in IHC_ONLY:
        assert _roles(stage_id) == ["ihc"], stage_id


def test_only_the_prefix_is_offered_as_a_choice() -> None:
    """Two slides is not always a question, and treating it as one puts a dead
    control on screen.

    Steps 2-6 have a separate answer per slide, so the screen offers a switch. Step 1
    reads both pyramids so the two readouts land side by side, and step 10 registers
    one onto the other - neither has a "which slide" to pick, and a toggle there
    would change nothing while implying it could.
    """
    together = {s.id for s in PIPELINE_STAGES if s.reads_slides_together}
    assert together == {"read-slide", "ihc-alignment"}

    for stage_id in PER_SLIDE_PREFIX:
        stage = next(s for s in PIPELINE_STAGES if s.id == stage_id)
        if stage_id == "read-slide":
            continue
        assert not stage.reads_slides_together, stage_id


def test_a_single_slide_stage_never_claims_to_read_two_together() -> None:
    for stage in PIPELINE_STAGES:
        if stage.reads_slides_together:
            assert len(stage.runs_on) > 1, stage.id


def test_alignment_needs_both_and_validation_needs_neither() -> None:
    """The two steps that are not simply "one slide, maybe twice".

    Step 10 reads both at one moment, because registering a pair is the one
    operation that cannot be done a slide at a time. Step 17 reads neither: it
    compares five markers of one block against four readers of the same block, so
    a single slide is not a thing it can be run on.
    """
    assert _roles("ihc-alignment") == ["he", "ihc"]
    assert _roles("validation") == ["case"]


def test_every_stage_in_the_catalogue_is_covered_by_this_file() -> None:
    """A new step must state its slide here, not inherit the default silently."""
    named = set(PER_SLIDE_PREFIX) | set(HE_ONLY) | set(IHC_ONLY)
    named |= {"ihc-alignment", "validation"}
    assert {stage.id for stage in PIPELINE_STAGES} == named


def test_the_roles_reach_the_api(client: TestClient) -> None:
    """The frontend draws the badge from this field, so it has to be on the wire."""
    items = client.get("/api/v1/pipeline/stages").json()["items"]
    by_id = {stage["id"]: stage["runsOn"] for stage in items}

    assert by_id["colour-deconvolution"] == ["he", "ihc"]
    assert by_id["per-cell-measurement"] == ["ihc"]
    assert by_id["validation"] == ["case"]

    together = {
        stage["id"]: stage["readsSlidesTogether"]
        for stage in items
    }
    assert together["read-slide"] is True
    assert together["ihc-alignment"] is True
    assert together["colour-deconvolution"] is False
