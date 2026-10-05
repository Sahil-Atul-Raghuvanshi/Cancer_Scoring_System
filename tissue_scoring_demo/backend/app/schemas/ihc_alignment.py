"""Schemas for step 10 - carrying the ROI onto the IHC slide."""

from typing import Literal

from pydantic import Field

from app.schemas.common import APIModel

AlignmentState = Literal["queued", "running", "ready", "refused", "failed"]


class AlignmentRun(APIModel):
    """Live state of one registration. Polled while it runs."""

    he_upload_id: str
    ihc_upload_id: str
    state: AlignmentState
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration: float | None = None
    error: str | None = None


class AlignmentDiagnostics(APIModel):
    """Everything measured about the registration, whether or not it passed.

    Reported in full even when the gate refuses, because the numbers are the
    explanation - "refused" on its own tells a viewer nothing they can act on.
    """

    #: How the transform was found: `intensity` for the mutual-information fit this
    #: pipeline now uses, absent on a report written by the older feature-matching path.
    #:
    #: **It has to be a declared field, not just a key in a dict.** This model drops
    #: anything it does not declare, so a diagnostic added only at the call site is
    #: silently absent from the stored JSON - which is exactly how the cache guard that
    #: reads this ended up discarding every report it was meant to accept.
    registration_kind: str | None = None
    #: Which set of gate rules decided this report. Any report made by an older version,
    #: ready or refused, is re-decided rather than reused (P-15).
    gate_version: int | None = None
    #: `transform_warp.fingerprint` of what the warp read - the transform file, the
    #: placement matrices and the working scale. A report whose stored transform has
    #: since changed is re-decided. Null where no transform was stored.
    transform_sha256: str | None = None
    #: Which rule measured the tissue areas: `optical_density`, or the older saturation
    #: mask when no render exists for the case.
    tissue_area_source: str | None = None
    he_tissue_saturation_mm2: float | None = None
    ihc_tissue_saturation_mm2: float | None = None
    #: Gate reasons that were overridden for this pair rather than refused on.
    #:
    #: Non-empty means the registration did **not** pass the gate and was allowed through
    #: anyway, by explicit per-pair instruction. The reasons are kept in full and become a
    #: caveat on every score from the pair, because a result carried past a failed check is
    #: only defensible while the failed check travels with it.
    gate_overridden: list[str] = Field(default_factory=list)
    #: Which registration method produced the transform, e.g. `mattes`.
    method: str | None = None
    #: Points at which the transform folds tissue through itself. Must be 0: a negative
    #: Jacobian determinant is geometrically impossible, and a mask carried through such
    #: a warp lands on unrelated cells.
    folded_points: int | None = None

    matched_keypoints: int | None = None
    #: Median distance between matched features after warping, in microns.
    residual_error_um: float | None = None
    rigid_error_um: float | None = None
    original_error_um: float | None = None
    #: Warp the region across and back: how far it comes back from where it started.
    round_trip_median_um: float | None = None
    round_trip_max_um: float | None = None
    #: Mutual information between the two sections' absorbance at corresponding
    #: points, and the same figure with no registration at all - which is what
    #: it has to beat. 1.0 would mean the two tell you nothing about each other.
    alignment_nmi: float | None = None
    alignment_nmi_unregistered: float | None = None
    #: Share of the probe lattice that landed on the IHC slide at all.
    probe_points_on_ihc_slide: float | None = None
    he_tissue_mm2: float | None = None
    ihc_tissue_mm2: float | None = None
    tissue_area_ratio: float | None = None
    he_mpp: float | None = None
    ihc_mpp: float | None = None
    clamped_vertices: int | None = None
    total_vertices: int | None = None
    reused_registration: bool | None = None
    seconds: float | None = None


class AlignedRegion(APIModel):
    """One invasive region, in both slides' coordinates."""

    index: int = Field(description="Step 9's own 0-based rank for this region, largest first")
    rank: int = Field(description="1-based position among the regions carried across")
    roi_id: str | None = Field(
        default=None,
        description=(
            "Which step 10 candidate this came from. Several carried regions can share "
            "one id: a coarse box routinely refines into more than one focus, and each "
            "focus crosses as its own polygon."
        ),
    )
    source: str = Field(
        default="tile_roi",
        description=(
            "`beetle_pixel` when this is step 11's per-pixel boundary, `tile_roi` when "
            "it is step 9's window staircase. Stored rather than inferred, so a report "
            "written before step 11 existed still says what it holds."
        ),
    )
    cells: int
    area_mm2: float
    #: Level-0 polygon rings. Outer ring first, then holes.
    he_rings: list[list[list[float]]]
    ihc_rings: list[list[list[float]]]


class AlignmentReport(APIModel):
    """Step 10's result: where the invasive regions landed, and how much to trust it."""

    he_upload_id: str
    ihc_upload_id: str
    marker: str | None = None
    state: AlignmentState
    generated_at: str

    regions: list[AlignedRegion]
    diagnostics: AlignmentDiagnostics

    #: Share of the invasive carcinoma step 9 found that these regions cover.
    #:
    #: The honest headline for this step. OncoStem's procedure is that the entire
    #: slide is scanned and every field averaged (SOP 4.2), so whatever is not
    #: carried here is tumour the score never sees - and a region *count* says
    #: nothing about how much that is. Three regions were 63% of this project's
    #: own case.
    area_coverage: float = 0.0
    #: Square millimetres of invasive carcinoma carried, and the total available.
    carried_mm2: float = 0.0
    invasive_mm2: float = 0.0

    #: Empty when the gate passed. Populated, and `state` is "refused", when it
    #: did not - never both a refusal and a mask.
    refusal_reasons: list[str] = Field(default_factory=list)

    #: The blocking check: a person has looked at the two panels side by side
    #: and said the regions landed correctly. Nothing downstream should use
    #: these regions until this is set.
    confirmed: bool = False
    confirmed_at: str | None = None
    #: "person" or "machine". A batch run that has to proceed unattended can
    #: confirm this gate, but it must not be able to do so *invisibly* - the
    #: whole point of the gate is that somebody looked at the two panels, and a
    #: machine confirmation is a record that nobody did. Step 16 turns this into
    #: a caveat printed beside the score.
    confirmed_by: str | None = None

    notes: list[str] = Field(default_factory=list)
