"""Step 6 - colour deconvolution, orchestrated.

Not a background job, like steps 3, 4 and 5 and unlike step 2: two inverses of a
3x3 and two matrix multiplies over a quarter of a million pixels, so it answers in
the request that asked for it.

**Its input is step 5's output, not step 5's input.** This service asks
`density_service.transformed_tile()` for the optical density array and takes it as
given. It does not read the pyramid, does not choose a tile and does not compute a
logarithm, and all three of those are refusals rather than omissions - optical
density is *defined* against step 4's white point, so a second `-log10(I / I0)`
here would be a second answer to a question two earlier steps were built to answer
once. The hand-off is nearly free in the common case, because step 5's own memo
holds the tile a reader has just been looking at.

**This is where the pipeline forks, and the code has to make that literal.** Rule
2 draws a Y: one deconvolution, the haematoxylin channel to the region model and
the DAB channel to the measurement. A Y is only meaningful if both arms start from
the same point, so there is exactly one `deconvolve()` call here and both channels
come out of it. When the model branch is built it will import the same function
from `app.pipeline.step06_colour_deconvolution`, not reach for
`skimage.color.rgb2hed` on its own - which is the failure the guide names by name,
because two implementations drift and then the IHC slides score nothing like the
H&E.

**Nothing is cached on disk.** One in-process memo, holding one entry, keyed on
step 5's hand-off identity - which already folds in the slide, the mask, the white
point and the tile. A view of this step fetches the report and four or eight
panels, which without the memo would be nine runs of the same arithmetic.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import numpy as np

from app.common.stains import REFERENCE_BY_NAME, degrees_between, unit
from app.core.config import settings
from app.core.logging import get_logger
from app.pipeline.step06_colour_deconvolution import deconvolution as dc
from app.pipeline.step06_colour_deconvolution import overlay
from app.pipeline.step06_colour_deconvolution.deconvolution import (
    BASIS_DEPARTURE_DEG,
    DeconvolutionError,
    Separation,
)
from app.schemas.calibration import Channels
from app.schemas.deconvolution import (
    ArmAgreementOut,
    BasisOut,
    ChannelOut,
    ComparisonOut,
    DeconvolutionParams,
    DeconvolutionReport,
    DeconvolutionTile,
    PreviewOut,
)
from app.services.density_service import DensityHandoff, density_service
from app.services.upload_service import get_record

logger = get_logger(__name__)

CITATION = (
    "Ruifrok AC, Johnston DA. Quantification of histochemical staining by colour "
    "deconvolution. Analytical and Quantitative Cytology and Histology 23(4):291-299 "
    "(2001). https://pubmed.ncbi.nlm.nih.gov/11531144/ - the fixed reference vectors "
    "this step measures with, and what skimage.color.rgb2hed implements. "
    "Macenko M, Niethammer M, Marron JS, et al. A method for normalizing histology "
    "slides for quantitative analysis. ISBI 2009:1107-1110 - the per-image estimator "
    "shown here for comparison and deliberately not used to measure."
)

#: How the stains are written in prose. "dab" is an acronym and reads as a typo in
#: a sentence; the API keys stay lower case so the frontend can switch on them.
STAIN_LABELS = {"haematoxylin": "haematoxylin", "dab": "DAB", "residual": "residual"}


def _stain(name: str) -> str:
    return STAIN_LABELS.get(name, name)


#: How far the two stain channels' correlation has to fall below the tile's own
#: pre-un-mixing correlation before the separation is called a success.
#:
#: A *relative* test, and deliberately not an absolute one. How much two channels
#: correlate is a property of the field of view before it is a property of the
#: basis: on a tile where both dyes genuinely sit on the same structures they
#: correlate for biological reasons no matrix can remove, and on one where they sit
#: apart even a poor basis separates them. Measured on the demo's slides the same
#: fixed basis lands anywhere from 0.2 to 0.5, tracking the tile rather than the
#: method - so an absolute threshold would report the tile and call it the step.
#: Only the before-and-after on the same pixels says what the un-mixing did.
DISENTANGLED_DROP = 0.1

#: Share of a tile's density the residual channel may absorb before it is worth
#: remarking on. The residual is the direction *no* stain occupies, so on a clean
#: haematoxylin-DAB section it holds sensor noise and little else. Above this
#: there is a third absorber in the tile - or the basis points somewhere the
#: section's dyes do not.
RESIDUAL_LIMIT = 0.15


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _channels(values: tuple[float, float, float], *, digits: int = 4) -> Channels:
    return Channels(
        r=round(values[0], digits), g=round(values[1], digits), b=round(values[2], digits)
    )


def _published(name: str) -> np.ndarray:
    """The published unit vector an arm was matched against, by name.

    Looked up in `REFERENCE_BY_NAME` rather than by column of the deconvolution
    matrix, because step 5 matches its arms against three references and one of
    them - eosin - is deliberately not a column here. An arm nearest eosin on a
    haematoxylin-DAB slide is a finding step 5 reports; measuring its angle against
    the wrong constant would turn that finding into a wrong number.
    """
    return unit(np.asarray(REFERENCE_BY_NAME[name], dtype=np.float64))


class DeconvolutionService:
    """Un-mixes step 5's tile onto both bases, and says what the choice costs.

    One memo entry, keyed on step 5's hand-off identity. Anything less would serve
    channels computed against a white point or a tile nobody is looking at any
    more; anything more - a key of this step's own - would be a second opinion
    about which tile step 6 is standing on, and there is only one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._memo: tuple[str, _Run] | None = None

    # --- running ------------------------------------------------------------

    def _run(
        self,
        upload_id: str,
        *,
        threshold: int | None,
        percentile: float | None,
        x: int | None,
        y: int | None,
    ) -> _Run:
        """Take step 5's density tile and un-mix it onto both bases.

        `threshold`, `percentile`, `x` and `y` all belong to earlier steps and are
        passed straight through. Step 6 owns none of them and interprets none of
        them: the mask is step 3's, the white point step 4's, the tile step 5's,
        and this step's only decision is which stain vectors to project onto -
        which is not a query parameter, because making it one would make the DAB
        scale a caller's choice.
        """
        handoff = density_service.transformed_tile(
            upload_id, threshold=threshold, percentile=percentile, x=x, y=y
        )

        with self._lock:
            if self._memo is not None and self._memo[0] == handoff.key:
                return self._memo[1]

        separation = dc.deconvolve(
            od=handoff.od,
            rgb=handoff.rgb,
            # Step 5's own two masks, carried across rather than re-derived. A step
            # that disagreed with the one above it about which pixels carry stain
            # would be un-mixing a different tile from the one whose arms the
            # reader was just shown.
            stained=handoff.stained,
            admitted=handoff.admitted,
            cut=settings.deconvolution_positive_cut,
            alpha=handoff.alpha,
        )

        logger.info(
            "deconvolution.run",
            extra={
                "upload_id": upload_id,
                "tile": f"{handoff.tile.x},{handoff.tile.y}",
                "estimated": separation.estimated is not None,
                "fixed_positive": separation.fixed.preview.positive_share,
            },
        )

        run = _Run(separation=separation, handoff=handoff)
        with self._lock:
            self._memo = (handoff.key, run)
        return run

    def report(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        percentile: float | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> DeconvolutionReport:
        """Run step 6 and describe the result, both bases at once."""
        run = self._run(
            upload_id, threshold=threshold, percentile=percentile, x=x, y=y
        )
        record = get_record(upload_id=upload_id)
        return self._describe(run, upload_id=upload_id, filename=record.filename)

    def panel(
        self,
        upload_id: str,
        name: str,
        *,
        basis: str = dc.FIXED,
        threshold: int | None = None,
        percentile: float | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> bytes:
        """One panel of one basis as PNG."""
        if name not in overlay.PANELS:
            raise DeconvolutionError(
                f"unknown panel {name!r}; expected one of {sorted(overlay.PANELS)}"
            )

        run = self._run(
            upload_id, threshold=threshold, percentile=percentile, x=x, y=y
        )
        return overlay.panel_png(run.separation, name, basis=basis)

    # --- handing on to steps 7 and 8 ----------------------------------------

    def channels(
        self,
        upload_id: str,
        *,
        threshold: int | None = None,
        percentile: float | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> tuple[Separation, DensityHandoff]:
        """Step 6's output for the steps that consume it rather than display it.

        The fork's hand-off, and the reason it returns the whole `Separation`
        rather than one channel: both arms of the Y leave from here, so a consumer
        naming which channel it wants is the design, and a service that returned
        only one would have to be called twice and could be called twice with
        different arguments.

        Fixed basis only, as far as anything downstream is concerned - the
        estimated one exists on the report to be looked at. Nothing that measures
        is allowed to reach for it, and a consumer that wants it has to say
        `estimated` out loud on a panel request.
        """
        run = self._run(
            upload_id, threshold=threshold, percentile=percentile, x=x, y=y
        )
        return run.separation, run.handoff

    # --- describing ---------------------------------------------------------

    def _describe(
        self, run: _Run, *, upload_id: str, filename: str
    ) -> DeconvolutionReport:
        separation, handoff = run.separation, run.handoff
        tile = handoff.tile

        stained_share = float(np.mean(separation.stained))
        admitted_share = float(np.mean(handoff.admitted))

        return DeconvolutionReport(
            upload_id=upload_id,
            filename=filename,
            generated_at=_now(),
            params=DeconvolutionParams(
                positive_cut=separation.cut,
                arm_percentile=handoff.alpha,
                beta=handoff.beta,
                min_estimate_pixels=dc.MIN_ESTIMATE_PIXELS,
                channel_bins=dc.CHANNEL_BINS,
            ),
            tile=DeconvolutionTile(
                x=tile.x,
                y=tile.y,
                size=tile.size,
                mpp=round(tile.mpp, 5),
                tile_um=round(handoff.tile_um, 2),
                level=tile.level,
                rank=handoff.candidates.index(handoff.chosen) + 1,
                requested=handoff.requested,
                candidates_scored=len(handoff.candidates),
                stained_share=round(stained_share, 5),
                admitted_share=round(admitted_share, 5),
            ),
            fixed=self._describe_basis(separation.fixed),
            estimated=(
                self._describe_basis(separation.estimated)
                if separation.estimated is not None
                else None
            ),
            estimated_refusal=separation.estimated_refusal,
            comparison=self._describe_comparison(separation),
            mixed_correlation=round(separation.mixed_correlation, 4),
            arm_agreement=self._describe_agreement(handoff),
            notes=self._notes(run),
            citation=CITATION,
        )

    @staticmethod
    def _describe_basis(basis: dc.Basis) -> BasisOut:
        return BasisOut(
            kind=basis.kind,
            matrix=[_channels(column) for column in basis.matrix],
            channels=[
                ChannelOut(
                    name=channel.name,
                    vector=_channels(channel.vector),
                    median=round(channel.median, 4),
                    p99=round(channel.p99, 4),
                    maximum=round(channel.maximum, 4),
                    mean=round(channel.mean, 4),
                    negative_share=round(channel.negative_share, 5),
                    histogram=list(channel.histogram),
                    histogram_low=round(channel.histogram_low, 4),
                    histogram_high=round(channel.histogram_high, 4),
                )
                for channel in basis.channels
            ],
            preview=PreviewOut(
                cut=round(basis.preview.cut, 4),
                positive_share=round(basis.preview.positive_share, 5),
                mean_dab=round(basis.preview.mean_dab, 4),
                p99_dab=round(basis.preview.p99_dab, 4),
            ),
            degrees_from_published=[
                round(basis.degrees_from_published[0], 2),
                round(basis.degrees_from_published[1], 2),
            ],
            exactness_max=round(basis.exactness_max, 8),
            exactness_mean=round(basis.exactness_mean, 9),
            residual_share=round(basis.residual_share, 5),
            channel_correlation=round(basis.channel_correlation, 4),
        )

    @staticmethod
    def _describe_comparison(separation: Separation) -> ComparisonOut | None:
        estimated = separation.estimated
        if estimated is None:
            return None

        fixed = separation.fixed
        departure = max(estimated.degrees_from_published)

        return ComparisonOut(
            haematoxylin_degrees=round(estimated.degrees_from_published[0], 2),
            dab_degrees=round(estimated.degrees_from_published[1], 2),
            fixed_positive_share=round(fixed.preview.positive_share, 5),
            estimated_positive_share=round(estimated.preview.positive_share, 5),
            share_shift=round(
                estimated.preview.positive_share - fixed.preview.positive_share, 5
            ),
            dab_scale=round(
                estimated.preview.mean_dab / max(fixed.preview.mean_dab, 1e-9), 4
            ),
            departed=departure >= BASIS_DEPARTURE_DEG,
            departure_deg=round(departure, 2),
        )

    @staticmethod
    def _describe_agreement(handoff: DensityHandoff) -> list[ArmAgreementOut]:
        """Step 5's arms measured against the vectors step 6 un-mixes with.

        The bridge between the two screens, and the evidence that using published
        constants on this slide is not an assumption being imposed: step 5 found
        two directions in the tile without being told what either stain looks
        like, and these are the angles between those and the constants.

        Empty when step 5 built no cloud. That is a real state - a tile of pale
        stroma has no arms - and it does not stop step 6 working, because the fixed
        basis does not depend on the tile.
        """
        return [
            ArmAgreementOut(
                stain=name,
                arm_vector=_channels((float(vector[0]), float(vector[1]), float(vector[2]))),
                nearest=name,
                degrees=round(degrees_between(vector, _published(name)), 2),
            )
            for name, vector in handoff.arms
        ]

    @staticmethod
    def _notes(run: _Run) -> list[str]:
        """The caveats that belong beside the numbers, built from the numbers."""
        separation, handoff = run.separation, run.handoff
        fixed, estimated = separation.fixed, separation.estimated
        notes: list[str] = []

        # --- what this step is ---------------------------------------------
        notes.append(
            "Every pixel of this tile carries both dyes at once, and step 5 put them in the "
            "one space where 'at once' means addition: optical density, where a mixture is a "
            "sum of one direction per stain times how much of it is there. A sum of three "
            "numbers is a 3x3 linear system, and a 3x3 system has an inverse. That inverse "
            "is this whole step - no training, no fitting, no per-slide adjustment."
        )
        notes.append(
            "This is the fork. The haematoxylin channel goes to the region model in place of "
            "RGB, which is what lets one detector work on the H&E slide and on all five IHC "
            "slides - the brown, the largest single source of colour nuisance between them, "
            "is gone before the model sees anything. The DAB channel goes to the measurement "
            "on an absolute optical density scale. Both come out of the one deconvolution "
            "above, so nothing downstream can measure a pixel the other branch rewrote."
        )

        # --- the exactness check -------------------------------------------
        notes.append(
            f"The separation is exact, and that is checkable rather than claimed. The third "
            f"column of the matrix is the direction no stain occupies, which makes the matrix "
            f"invertible - so un-mixing the tile and adding the three channels back together "
            f"has to return what went in. It does, to {fixed.exactness_max:.2e} optical "
            f"density at worst, which is float rounding. Nothing was thresholded, clipped or "
            "smoothed here: this is a change of coordinates."
        )

        # --- did it actually disentangle anything? -------------------------
        #
        # Read as a *comparison* and never against a fixed limit. How much two
        # channels correlate depends on the tile - a field where both dyes
        # genuinely sit on the same structures correlates for biological reasons
        # that no basis can or should remove - so the only meaningful statement is
        # the before-and-after on these same pixels, and the honest report is
        # whichever way it actually went.
        before = separation.mixed_correlation
        after = fixed.channel_correlation
        notes.append(
            f"Before un-mixing, the three optical density channels of this tile correlate at "
            f"{before:.2f} on average - red, green and blue all rise and fall with how much "
            f"total dye is present, which is precisely why none of them is a measurement of "
            f"either stain on its own. After un-mixing, the haematoxylin and DAB channels "
            f"correlate at {after:.2f}. "
            + (
                "That drop is the separation doing its job: a threshold on the DAB channel is "
                "now a threshold on one dye rather than on a mixture, which is what Rule 3 "
                "asks for."
                if after <= before - DISENTANGLED_DROP
                else "That is not a drop, and it is a fact about this tile rather than a "
                "fault in the arithmetic - the reconstruction above is exact either way. Two "
                "things produce it. Both dyes may genuinely sit on the same structures here, "
                "in which case no basis can separate what the biology has put together. Or "
                "the published vectors may fit this section poorly, which is measurable: "
                + (
                    f"the tile's own stain directions sit "
                    f"{max(estimated.degrees_from_published):.0f} degrees from the published "
                    f"pair, and un-mixing on those instead leaves the two channels correlated "
                    f"at {estimated.channel_correlation:.2f}. That is the case for calibrating "
                    "the vectors once on this scanner's control slides and freezing them - not "
                    "for re-estimating them per slide, which would cost comparability to buy "
                    "the fit."
                    if estimated is not None
                    else "no per-image basis could be estimated on this tile, so there is "
                    "nothing to test that against here."
                )
                + " Either way, read a DAB threshold on this tile with care."
            )
        )

        # --- the residual ---------------------------------------------------
        if fixed.residual_share <= RESIDUAL_LIMIT:
            notes.append(
                f"The residual channel - what neither dye explains - absorbs "
                f"{fixed.residual_share:.1%} of this tile's density. That is the honest test "
                "of the two-stain assumption: small means haematoxylin and DAB really do "
                "account for what the scanner recorded, and the near-empty third panel is "
                "what that looks like."
            )
        else:
            notes.append(
                f"The residual channel absorbs {fixed.residual_share:.1%} of this tile's "
                f"density, which is more than a clean haematoxylin-DAB section should leave. "
                "Something here is not one of the two dyes: a third absorber, a scanning "
                "artefact step 2 did not flag, or a counterstain whose colour has shifted far "
                "enough that the published vector no longer describes it. The two stain "
                "channels are still exact - the arithmetic is - but they are describing a "
                "tile the two-stain model does not fully fit."
            )

        # --- negative concentrations ---------------------------------------
        worst = max(
            (channel for channel in fixed.channels if channel.name != "residual"),
            key=lambda channel: channel.negative_share,
        )
        if worst.negative_share > 0.01:
            notes.append(
                f"{worst.negative_share:.1%} of the stained pixels need a *negative* amount of "
                f"{_stain(worst.name)}, which is not a physical concentration. Those are pixels "
                "whose colour sits outside the cone the two published vectors span, and they "
                "are counted rather than clipped: the reconstruction above is only exact if "
                "nothing was repaired, and a clipped channel would quietly stop being the "
                "pixel it came from. The panels clamp them to zero for legibility, and this "
                "line is how you know that happened."
            )

        # --- the comparison, which is the point ----------------------------
        if estimated is None:
            notes.append(
                separation.estimated_refusal
                or "No per-image basis could be estimated on this tile, so there is no "
                "comparison to draw. The fixed basis is unaffected - it does not depend on "
                "the tile, which is the property the comparison exists to demonstrate."
            )
        else:
            shift = estimated.preview.positive_share - fixed.preview.positive_share
            notes.append(
                f"The same tile, the same arithmetic, the same positivity cut of "
                f"{separation.cut:g} optical density - and only the stain vectors changed. "
                f"On Ruifrok & Johnston's published pair, {fixed.preview.positive_share:.1%} "
                f"of the stained pixels are positive. On vectors estimated from this tile's "
                f"own colours - Macenko's method, the same estimator that drew step 5's arms "
                f"- {estimated.preview.positive_share:.1%} are. That is a shift of "
                f"{abs(shift) * 100:.1f} percentage points from a choice that has nothing to "
                "do with the biology."
            )
            notes.append(
                f"Which is the argument for fixed vectors, and it is worth stating plainly. "
                f"The estimated pair sits {estimated.degrees_from_published[0]:.1f} degrees "
                f"from published haematoxylin and {estimated.degrees_from_published[1]:.1f} "
                f"from published DAB, and it scales the mean DAB reading by "
                f"{estimated.preview.mean_dab / max(fixed.preview.mean_dab, 1e-9):.2f}x. Every "
                "slide would get a different scaling, because every slide has different "
                "colours - so '0.4 DAB' would mean a different amount of dye on each one and "
                "two slides in a study could not be compared. The published vectors are the "
                "same numbers on every slide anyone has ever used them on, and that is the "
                "only reason a score built on them means anything. Estimated vectors are "
                "shown here and never measured with."
            )
            if estimated.residual_share < fixed.residual_share:
                notes.append(
                    f"Note that the estimated basis fits *this tile* better - its residual is "
                    f"{estimated.residual_share:.1%} against the fixed basis's "
                    f"{fixed.residual_share:.1%} - and that is exactly the trap. Fitting one "
                    "tile better is what a per-image estimate is for; it buys that fit by "
                    "giving this tile its own scale, which is the one thing a comparable "
                    "measurement cannot afford. The right response to a poor fixed-basis fit "
                    "is to calibrate the vectors once per scanner on control slides and then "
                    "freeze them, not to re-estimate them per slide."
                )

        # --- what upstream did ---------------------------------------------
        notes.append(
            f"Both channels were computed over the whole tile, and every statistic on this "
            f"screen over the {float(np.mean(separation.stained)):.0%} of it carrying at "
            f"least {handoff.beta:g} mean optical density - step 5's own 'this pixel has "
            "stain in it' mask, carried across rather than re-derived. Including the faint "
            "background would drag every percentile towards zero and make two different "
            "bases look alike for a reason that has nothing to do with either."
        )

        if handoff.arms:
            arms = ", ".join(
                f"{_stain(name)} at {degrees_between(vector, _published(name)):.1f} degrees"
                for name, vector in handoff.arms
            )
            notes.append(
                f"Step 5 found two stain directions in this tile without being told what "
                f"either dye looks like, and they land {arms} from the published vectors this "
                "step un-mixed with. That is why using constants here is not an assumption "
                "being imposed on the slide - the slide's own colours already point there."
            )
        else:
            notes.append(
                "Step 5 built no point cloud on this tile, so there are no measured arms to "
                "check the published vectors against. The deconvolution is unaffected - the "
                "fixed basis does not depend on the tile - but the evidence that these "
                "vectors suit this slide is on a tile with more stain in it."
            )

        if not handoff.white.qc_gated:
            notes.append(
                "Step 2 has not run, so no artefacts were excluded from the tile this was "
                "computed on. A fold or a pen mark is dark, and dark un-mixes into a "
                "confident amount of some dye. Run quality control and open this step again."
            )

        return notes


class _Run:
    """One complete run of step 6, memoised as a unit.

    A plain class rather than a dataclass because it holds arrays and exists only
    so the report and every panel look at the same numbers - if any of them
    recomputed, a picture on screen and the figure beside it could disagree.
    """

    __slots__ = ("handoff", "separation")

    def __init__(self, *, separation: Separation, handoff: DensityHandoff) -> None:
        self.separation = separation
        self.handoff = handoff


deconvolution_service = DeconvolutionService()

__all__ = ["DeconvolutionError", "deconvolution_service"]
