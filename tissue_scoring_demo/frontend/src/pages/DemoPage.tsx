import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { CalibrationPanel } from '@/features/calibration/CalibrationPanel'
import { calibrationHint } from '@/features/calibration/calibrationStatus'
import { useWhiteCalibration } from '@/features/calibration/useWhiteCalibration'
import { DeconvolutionPanel } from '@/features/deconvolution/DeconvolutionPanel'
import { deconvolutionHint } from '@/features/deconvolution/deconvolutionStatus'
import { useDeconvolution } from '@/features/deconvolution/useDeconvolution'
import { DensityPanel } from '@/features/density/DensityPanel'
import { densityHint } from '@/features/density/densityStatus'
import { useOpticalDensity } from '@/features/density/useOpticalDensity'
import { PipelineTrack } from '@/features/pipeline/components/PipelineTrack'
import type {
  TrackDecision,
  TrackStatus,
} from '@/features/pipeline/components/PipelineTrack'
import { SlideReadoutView } from '@/features/pipeline/components/SlideReadoutView'
import { StageDetails } from '@/features/pipeline/components/StageDetails'
import { StageFlow } from '@/features/pipeline/components/StageFlow'
import '@/features/pipeline/components/pipeline.css'
import { usePipelineRun } from '@/features/pipeline/hooks/usePipelineRun'
import { useStepKeys } from '@/features/pipeline/hooks/useStepKeys'
import { QCPanel } from '@/features/qc/QCPanel'
import { canRunQC, qcHint } from '@/features/qc/qcStatus'
import { useQualityControl } from '@/features/qc/useQualityControl'
import { RoiPanel } from '@/features/roi/RoiPanel'
import { roiHint } from '@/features/roi/roiStatus'
import { useRoi } from '@/features/roi/useRoi'
import { RoiSelectionPanel } from '@/features/roiSelection/RoiSelectionPanel'
import { roiSelectionHint } from '@/features/roiSelection/roiSelectionStatus'
import { useRoiSelection } from '@/features/roiSelection/useRoiSelection'
import { RoiRefinementPanel } from '@/features/roiRefinement/RoiRefinementPanel'
import { roiRefinementHint } from '@/features/roiRefinement/roiRefinementStatus'
import { useRoiRefinement } from '@/features/roiRefinement/useRoiRefinement'
import { TilingPanel } from '@/features/tiling/TilingPanel'
import { tilingHint } from '@/features/tiling/tilingStatus'
import { useTiling } from '@/features/tiling/useTiling'
import { TissuePanel } from '@/features/tissue/TissuePanel'
import { tissueHint } from '@/features/tissue/tissueStatus'
import { useTissueMask } from '@/features/tissue/useTissueMask'
import { TissueTypePanel } from '@/features/tissueType/TissueTypePanel'
import { canRunTissueType, tissueTypeHint } from '@/features/tissueType/tissueTypeStatus'
import { useTissueType } from '@/features/tissueType/useTissueType'
import { IhcAlignmentPanel } from '@/features/ihcAlignment/IhcAlignmentPanel'
import { useIhcAlignment } from '@/features/ihcAlignment/useIhcAlignment'
import { CellTypingPanel } from '@/features/cellTyping/CellTypingPanel'
import { useCellTyping } from '@/features/cellTyping/useCellTyping'
import { CompartmentsPanel } from '@/features/compartments/CompartmentsPanel'
import { useCompartments } from '@/features/compartments/useCompartments'
import { PerCellPanel } from '@/features/perCell/PerCellPanel'
import { usePerCell } from '@/features/perCell/usePerCell'
import { BinningPanel } from '@/features/binning/BinningPanel'
import { useBinning } from '@/features/binning/useBinning'
import { ScorePanel } from '@/features/scores/ScorePanel'
import { useScores } from '@/features/scores/useScores'
import { ValidationPanel } from '@/features/validation/ValidationPanel'
import { useValidation } from '@/features/validation/useValidation'
import { NucleiPanel } from '@/features/nuclei/NucleiPanel'
import { useNuclei } from '@/features/nuclei/useNuclei'
import { archiveHistoryMarker } from '@/api/history'
import { CaseLoaderPanel } from '@/features/panel/CaseLoaderPanel'
import { SlideBadge } from '@/features/slideRole/SlideBadge'
import { SlidePicker } from '@/features/slideRole/SlidePicker'
import { slideChoiceFor } from '@/features/slideRole/slideChoice'
import { slideHint } from '@/features/slideRole/slideHint'
import { useBothSlides } from '@/features/slideRole/useBothSlides'
import type { SlideOption } from '@/features/slideRole/slideChoice'
import { ReleaseSlide } from '@/features/upload/ReleaseSlide'
import { useSlideSession } from '@/features/upload/slideSessionContext'
import type { UploadCapabilityState } from '@/features/upload/useUploadCapability'
import { padIndex } from '@/lib/format'
import type { PipelineStage } from '@/types/pipeline'

import './demo.css'

interface DemoPageProps {
  stages: PipelineStage[]
  loading: boolean
  /** What the server will accept for an upload, or why we cannot say yet. */
  upload: UploadCapabilityState
}

const APPROACH_TONE = {
  classical: 'neutral',
  library: 'neutral',
  plumbing: 'neutral',
  logic: 'neutral',
  pretrained: 'accent',
  trained: 'violet',
} as const

const STATE_TONE: Record<TrackStatus, 'neutral' | 'accent' | 'success' | 'danger'> = {
  pending: 'neutral',
  active: 'accent',
  complete: 'success',
  error: 'danger',
}

function CompleteMark() {
  return (
    <svg className="finish__mark" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.4" opacity="0.45" />
      <path
        d="m7.4 12.4 3.1 3.1 6.1-6.6"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/** What the previous-cases list hands over when it sends somebody here. */
interface ReplayHandoff {
  caseId?: string
  casePath?: string
  marker?: string
  /** True when every step already has a stored result, so nothing recomputes. */
  replay?: boolean
}

export function DemoPage({ stages, loading, upload }: DemoPageProps) {
  const slide = useSlideSession()
  const run = usePipelineRun(stages)

  /**
   * A case chosen on the previous-cases screen, if that is how we got here.
   *
   * The files were already renamed back into the working tree before this page
   * was navigated to - see `openRun` in `HomePage` - so all that is left is to
   * point the session at them. `loadCase` finds the restored sidecar and reuses
   * the upload ids it records rather than registering the slides again, which is
   * what makes every step find its cached report instead of recomputing one.
   */
  const handoff = (useLocation().state ?? null) as ReplayHandoff | null
  const replaying = handoff?.replay === true
  const wanted = handoff?.casePath && handoff?.marker ? handoff : null
  const { loadCase } = slide
  const requested = useRef<string | null>(null)

  useEffect(() => {
    if (!wanted?.casePath || !wanted.marker) return
    const key = `${wanted.casePath}:${wanted.marker}`
    // Once per handoff. Without this the effect re-runs on every render that
    // touches the session it is about to change.
    if (requested.current === key) return
    requested.current = key
    void loadCase(wanted.casePath, wanted.marker).catch(() => {
      /* the step-0 panel is still on screen and reports the failure itself */
    })
  }, [loadCase, wanted])
  //: Read from `run` rather than from the `stage` destructured further down, because
  //: the prefix hooks above need it and they run before that line.
  const stageId = run.stage?.id ?? null

  // Step 1 reads the H&E slide - the one steps 2-9 all key off - and, when a
  // case (rather than a lone file) was loaded, the IHC slide alongside it, so
  // both readouts land together rather than the IHC one lagging a step behind.
  const readBothSlides = useCallback(async () => {
    await slide.readSlide()
    if (slide.ihcUploadId) await slide.readIhcSlide()
  }, [slide])
  // --- which slide the per-slide prefix is showing -------------------------
  //
  // Steps 2 to 6 each measure a property of ONE piece of glass - an artefact map, a
  // tissue footprint, a white point, a density, an un-mixing - so each has an answer
  // for both slides of a case, and the two answers differ. The catalogue says which
  // steps those are (`runsOn`); this is the viewer's choice between them, held here
  // rather than per panel so that walking 2 -> 3 -> 4 stays on the slide they picked.
  const [prefixRole, setPrefixRole] = useState<'he' | 'ihc'>('he')
  const showingIhcPrefix = prefixRole === 'ihc' && slide.ihcUploadId != null

  // **Both slides are live, not just the one on screen.** This started out gated on
  // the toggle, to avoid fetching a slide nobody was looking at. That was the wrong
  // trade: a step that declares two slides has two answers the pipeline needs, and if
  // the second one only exists while its tab is open then step 4 can never truthfully
  // ask whether step 3 has run, and the IHC arm of the measurement is built on
  // whatever the viewer happened to click. So the toggle now chooses what is
  // DISPLAYED, and the step runs on both regardless.
  const ihcPrefixId = slide.ihcUploadId

  // Two instances of each prefix hook, and the split is deliberate.
  //
  // The H&E instances stay pinned to `slide.uploadId` because steps 7, 8 and 9 read
  // their output: step 7's funnel and step 8's class map are built from the H&E's
  // mask, and if flipping this screen's toggle changed the cut those steps were
  // handed, a viewer inspecting the IHC slide's glass would silently re-tile the
  // H&E. The IHC instances fetch only while the toggle is on them - passing null is
  // how these hooks already express "no slide" - so looking at one slide costs one
  // slide's work.
  const qc = useQualityControl(slide.uploadId)
  const ihcQc = useQualityControl(ihcPrefixId)
  const tissue = useTissueMask(slide.uploadId)
  const ihcTissue = useTissueMask(ihcPrefixId)

  // Step 4's input is step 3's mask, so it is handed step 3's *committed* cut
  // rather than reading one of its own. A manual cut is passed through; an
  // automatic one is passed as null, so the server re-derives it by whichever
  // rule the histogram selects instead of receiving it as a hand-set number.
  const tissueCut =
    tissue.report?.threshold.source === 'manual' ? tissue.report.threshold.value : null
  // The IHC slide's own committed cut. Its glass is not the H&E's glass, so it gets
  // its own threshold for the same reason step 4 is per slide at all.
  const ihcTissueCut =
    ihcTissue.report?.threshold.source === 'manual' ? ihcTissue.report.threshold.value : null
  const calibration = useWhiteCalibration(slide.uploadId, tissueCut)
  const ihcCalibration = useWhiteCalibration(ihcPrefixId, ihcTissueCut)

  // Step 5 is handed the same cut, and for the same reason one step further
  // along: a different mask gives step 4 different glass, and step 4's I0 is the
  // denominator of every density step 5 computes. It is deliberately not handed
  // step 4's *percentile* - that is step 4's own comparison control, and step 5
  // asks for the server's default so its numbers describe the calibration the
  // pipeline would really use.
  const density = useOpticalDensity(slide.uploadId, tissueCut)
  const ihcDensity = useOpticalDensity(ihcPrefixId, ihcTissueCut)

  // Step 6 is handed step 5's *chosen tile*, not just its threshold, and that is
  // the one piece of wiring that carries an argument. Rule 2 draws the pipeline
  // forking here into a Y, and a Y only means anything if both arms leave from the
  // same point - so step 6 has to un-mix the very pixels step 5 has just shown,
  // and it has to follow the viewer if they go back and pick a different tile.
  // The server would choose the same tile by the same score if asked for none;
  // naming it is what makes the two screens demonstrably the same field of view.
  const densityTile = density.report
    ? { x: density.report.tile.x, y: density.report.tile.y }
    : null
  const deconvolution = useDeconvolution(slide.uploadId, tissueCut, densityTile)

  // Step 6 on the IHC slide is the point of all of this. Its DAB channel is what
  // step 14 measures; the H&E's DAB channel is a picture of nothing, because there
  // is no DAB in an H&E section. The guide asks for this step to be demonstrated on
  // an IHC tile, and until the toggle existed it never could be.
  const ihcDensityTile = ihcDensity.report
    ? { x: ihcDensity.report.tile.x, y: ihcDensity.report.tile.y }
    : null
  const ihcDeconvolution = useDeconvolution(ihcPrefixId, ihcTissueCut, ihcDensityTile)

  // Step 7 is handed step 3's cut, and one thing more. Unlike step 6 it does not
  // follow step 5's *tile*: it is not about one field of view, it is about all of
  // them, and its input is the mask rather than the density.
  //
  // What it does follow is step 5's **verdict on the dyes**, and that is not a
  // convenience - it is what makes step 7's full-colour option appear at all. That
  // option is gated on the server having a staining verdict for this slide, the
  // verdict only exists once step 5 has run, and step 7's branch payload is fetched
  // as soon as there is a slide - which is *before* step 5 in every walkthrough. So
  // the payload used to be read once, while the answer was still "step 5 has not
  // measured this slide", and never read again: an H&E slide reached step 7 with its
  // H&E option greyed out and the reason telling the reader to go and run a step they
  // had already run. Handing the verdict in makes it a dependency of that fetch, so
  // the option turns itself on the moment step 5 has something to say.
  //
  // The verdict string and not the whole object: it is what the server's decision
  // turns on, so keying on it refetches exactly when the answer could have changed
  // and not every time the viewer nudges step 5's tile.
  const tiling = useTiling(
    slide.uploadId,
    tissueCut,
    density.report?.staining?.staining ?? null,
  )

  // Step 8 is handed the same cut, and for the same reason as step 7: its input is
  // step 7's index, and a different mask is a different set of tiles and therefore a
  // different class map.
  //
  // It is also handed step 7's *overlap*, which completes the join: step 7 already
  // lays this model's own square, read from the checkpoint's manifest, so with the
  // overlap shared the two steps run one grid. That is what makes step 7's "squares
  // to process" the literal number of forward passes step 8 makes, rather than a
  // proxy for it that was out by a factor of twenty.
  //
  // And it is handed step 7's *model*, which is the same join taken one step further.
  // There used to be a checkpoint dropdown on this step; there is not one now, because
  // the thing a reader was really choosing between was fields of view, and a field of
  // view is a property of the weights. So the choice lives on step 7 where its cost in
  // squares is visible, and what arrives here is its consequence. Null means step 7's
  // field of view has no published head yet - step 8 refuses and names the file.
  //
  // And it is handed step 7's *branch* and *field of view*. Those are assertions
  // rather than instructions - the server runs whatever step 7 committed - but sending
  // them is what makes a stale client visible: if the viewer changed step 7 in another
  // tab, this request is refused instead of silently returning a class map from a
  // different pipeline.
  const tissueType = useTissueType(
    slide.uploadId,
    tissueCut,
    tiling.overlap,
    tiling.model,
    tiling.branch,
    tiling.fieldOfView,
  )

  // Step 9's only input is step 8's stored class map. The server discards its own
  // cache when step 8 reruns, but this screen has to notice too, or a viewer who goes
  // back and reclassifies would return to a region drawn from labels that no longer
  // exist - so `tissueType.report.generatedAt` travels in as the key that clears it.
  const roi = useRoi(slide.uploadId, tissueType.report?.generatedAt ?? null)

  // Step 10 offers step 9's invasive patches as a list somebody ticks. It is keyed
  // on step 8's stamp rather than step 9's, and that is not an oversight: the
  // candidate ids are *area ranks within a class map*, so it is step 8 changing that
  // re-points every id at different tissue. Step 9 changing moves the scored region's
  // denominator without renaming anything here.
  const roiSelection = useRoiSelection(
    slide.uploadId,
    tissueType.report?.generatedAt ?? null,
  )

  // Step 11 runs BEETLE on whatever step 10 left ticked, so it is keyed on the
  // selection itself - the stamp *and* who chose. A viewer who goes back and unticks a
  // region must not return to a refinement that still includes it, because that
  // refinement is the mask every later step measures inside.
  const selectionVersion = roiSelection.report
    ? `${roiSelection.report.generatedAt}:${roiSelection.selected.join(',')}`
    : null
  const roiRefinement = useRoiRefinement(slide.uploadId, selectionVersion)

  // Step 12 is the first step that needs two slides. It takes the H&E - the slide
  // every step so far ran on - and the case's IHC slide, and it is keyed on step 11's
  // stamp rather than step 9's now: what it carries across is the refined per-pixel
  // boundary, so a re-run refinement is a different mask and an alignment of the old
  // one is describing something that no longer exists.
  const ihcAlignment = useIhcAlignment(
    slide.uploadId,
    slide.ihcUploadId,
    roiRefinement.report?.generatedAt ?? null,
  )

  // Step 11 measures inside step 10's carried regions, so a re-run alignment is a
  // different set of regions and this result describes something that no longer
  // exists - the same staleness argument as every step above, one link further on.
  const nuclei = useNuclei(
    slide.uploadId,
    slide.ihcUploadId,
    ihcAlignment.report?.generatedAt ?? null,
  )

  // Step 12 sorts step 11's nuclei and runs on every threshold change, so unlike
  // every step above it there is no job to start - arriving on it is the request.
  const cellTyping = useCellTyping(
    slide.uploadId,
    slide.ihcUploadId,
    nuclei.report?.generatedAt ?? null,
  )

  // Step 13 grows step 11's nuclei into the shape step 12 said belongs in the
  // measurement, so it is keyed on step 11's stamp like step 12 is.
  const compartments = useCompartments(
    slide.uploadId,
    slide.ihcUploadId,
    nuclei.report?.generatedAt ?? null,
  )

  // Step 14 measures inside step 13's shapes, so re-shaping them is a different
  // set of pixels and these rows describe something that no longer exists - the
  // same staleness argument every step above makes, one link further on.
  //
  // Unlike steps 12 and 13 it is not request-shaped: it reopens the slide and
  // reads every sampled field again. That is why it is keyed but not auto-run;
  // the page launches it when the viewer actually reaches the step.
  const perCell = usePerCell(
    slide.uploadId,
    slide.ihcUploadId,
    compartments.report?.generatedAt ?? null,
  )

  // Steps 15 and 16 open nothing and segment nothing - they are five thresholds
  // and some arithmetic over rows step 14 wrote. So arriving on either of them is
  // the request, and each is keyed on the step before it.
  const binning = useBinning(
    slide.uploadId,
    slide.ihcUploadId,
    perCell.report?.generatedAt ?? null,
  )
  const scores = useScores(
    slide.uploadId,
    slide.ihcUploadId,
    binning.report?.generatedAt ?? null,
  )

  // Step 17 is the one step keyed on the CASE rather than on a pair: agreement is
  // measured over five markers of one block against four readers of the same
  // block, so a single marker's score is not a thing that can be validated.
  const validation = useValidation(
    slide.caseId ?? null,
    scores.report?.generatedAt ?? null,
  )

  // The square count at each setting the viewer has actually visited, so step 7's two
  // pickers can put a price on each choice. Accumulated here rather than inside the
  // hook because it must survive the hook clearing its report on a rebuild - and
  // deliberately not pre-filled with an estimate: an estimated count would be
  // indistinguishable on screen from a measured one.
  const tileCounts = useRef(new Map<number, number>())
  if (tiling.report) {
    tileCounts.current.set(tiling.report.params.overlap, tiling.report.funnel.clean)
  }

  const { stage, status, current } = run
  const stageRef = useRef<HTMLDivElement>(null)

  // Step 7's choice moved, so every step after it is describing a pipeline that no
  // longer exists. Without this, step 8 stays in `completed` and the rail reads
  // "Completed" with nothing behind it - the answer it is remembering was deleted on
  // the server the moment the new choice was committed.
  const tilingIndex = stages.findIndex((entry) => entry.id === 'tiling')
  const { invalidateAfter } = run
  useEffect(() => {
    if (tilingIndex < 0) return
    invalidateAfter(tilingIndex)
  }, [
    invalidateAfter,
    tilingIndex,
    tiling.branch,
    tiling.fieldOfView,
    tiling.overlap,
  ])

  // The page itself no longer scrolls: the rail is pinned and this column is
  // the only scroller, so a new step starts at the top of that column.
  useEffect(() => {
    stageRef.current?.scrollTo({ top: 0 })
  }, [current])

  const isComplete = status === 'complete'
  const isRunning = status === 'running'
  const isLast = current === stages.length - 1
  const hasSlide = slide.stage === 'loaded'

  // Step 2 is the quality-control step, and it behaves differently enough from
  // step 1 to be worth naming: its work outlives a request, so it owns its own
  // progress and its own result rather than resolving into `slide`.
  // The headline each side of the toggle carries, so the switch shows a difference
  // rather than merely offering one. Both cards can carry a real number on every
  // per-slide step now, because both slides have really run - step 4's pair is the
  // one the guide asks for by name ("two slides side by side with their two different
  // I0 values, that is the why-per-slide argument in one picture"), and the others
  // get it for free.
  const prefixSummary = useCallback(
    (option: SlideOption): string | null => {
      const isIhc = option.role === 'ihc'
      switch (stageId) {
        case 'white-calibration': {
          const report = (isIhc ? ihcCalibration : calibration).report
          return report ? report.white.hex.toUpperCase() : null
        }
        case 'tissue-mask': {
          const report = (isIhc ? ihcTissue : tissue).report
          return report ? `${(report.tissueShare * 100).toFixed(1)}% tissue` : null
        }
        case 'optical-density': {
          const report = (isIhc ? ihcDensity : density).report
          return report?.staining ? report.staining.staining.replace('_', '-') : null
        }
        case 'quality-control': {
          const report = (isIhc ? ihcQc : qc).report
          return report ? `${(report.gate.removedShare * 100).toFixed(1)}% removed` : null
        }
        case 'colour-deconvolution': {
          // The number that makes the whole per-slide argument visible in one glance:
          // the brown channel is full of signal on the stained slide and close to
          // empty on the H&E, because there is no DAB in an H&E section.
          const report = (isIhc ? ihcDeconvolution : deconvolution).report
          const dab = report?.fixed.channels.find((channel) => channel.name === 'dab')
          return dab ? `DAB peaks at ${dab.p99.toFixed(2)}` : null
        }
        default:
          return null
      }
    },
    [
      stageId,
      calibration,
      ihcCalibration,
      tissue,
      ihcTissue,
      density,
      ihcDensity,
      qc,
      ihcQc,
      deconvolution,
      ihcDeconvolution,
    ],
  )

  // Is the slide in the H&E slot actually an H&E?
  //
  // A lone uploaded file goes into that slot with no check at all - `SlideSession`
  // sets `heUploadId` to whatever arrived - so an immunostained slide uploaded on
  // its own reaches steps 7 to 9 and is treated as the slide the tumour is found on.
  // The pipeline already has the evidence to notice: step 5 fits the tile's density
  // cloud and names which published stain vectors its two arms land nearest, and
  // step 7 already consumes that verdict to decide whether to offer its colour
  // branch. This turns the same verdict into a warning on the three steps that would
  // otherwise act on it silently.
  //
  // The reason is measured, not cautionary: the model those steps run returns 0%
  // invasive on real immunostained slides while holding 0.96-0.98 confidence. It does
  // not fail loudly, it fails plausibly, which is why this has to be said before the
  // run rather than inferred from the result.
  const heSlotStaining = density.report?.staining?.staining ?? null
  const heSlotIsNotHe =
    heSlotStaining === 'haematoxylin_dab' || heSlotStaining === 'single_stain'
  const heSlotFilenameSaysIhc = slide.readout?.slideRole === 'ihc'

  // Where each slide has got to in THIS step. A per-slide step runs on both as one
  // unit of work, so mid-run one of them is finished and the other is not, and the
  // viewer is only looking at one of them.
  const prefixState = useCallback(
    (option: SlideOption): 'done' | 'running' | 'pending' => {
      const pick = <T,>(he: T, ihc: T) => (option.role === 'ihc' ? ihc : he)
      switch (stageId) {
        case 'quality-control': {
          const side = pick(qc, ihcQc)
          if (side.report) return 'done'
          return side.busy ? 'running' : 'pending'
        }
        case 'tissue-mask': {
          const side = pick(tissue, ihcTissue)
          return side.report ? 'done' : side.loading ? 'running' : 'pending'
        }
        case 'white-calibration': {
          const side = pick(calibration, ihcCalibration)
          return side.report ? 'done' : side.loading ? 'running' : 'pending'
        }
        case 'optical-density': {
          const side = pick(density, ihcDensity)
          return side.report ? 'done' : side.loading ? 'running' : 'pending'
        }
        case 'colour-deconvolution': {
          const side = pick(deconvolution, ihcDeconvolution)
          return side.report ? 'done' : side.loading ? 'running' : 'pending'
        }
        default:
          return 'pending'
      }
    },
    [
      stageId,
      qc,
      ihcQc,
      tissue,
      ihcTissue,
      calibration,
      ihcCalibration,
      density,
      ihcDensity,
      deconvolution,
      ihcDeconvolution,
    ],
  )

  // Which slide THIS step declares it reads, resolved against the loaded case. The
  // catalogue is the authority - `runsOn` in `backend/app/data/pipeline_steps.py` -
  // so a step that changes which slide it reads changes the badge and the toggle
  // with it, rather than leaving a screen captioned by hand from memory.
  const slideChoice = useMemo(
    () =>
      slideChoiceFor(
        stage?.runsOn,
        slide,
        prefixRole,
        stage?.readsSlidesTogether ?? false,
      ),
    [stage?.runsOn, stage?.readsSlidesTogether, slide, prefixRole],
  )

  // --- running a per-slide step on every slide it declares -------------------
  //
  // One task each, folding both slides into a single unit of work. The pipeline marks
  // a step complete when its task resolves, so this is what makes "step 3 is done"
  // mean "on both slides" - and therefore what lets step 4 refuse to start until it
  // actually is. See `useBothSlides` for why the order is fixed.
  const hasIhcForPrefix = slide.ihcUploadId != null
  const runQcBoth = useBothSlides(qc.start, ihcQc.start, hasIhcForPrefix)
  const runTissueBoth = useBothSlides(tissue.start, ihcTissue.start, hasIhcForPrefix)
  const runCalibrationBoth = useBothSlides(
    calibration.start,
    ihcCalibration.start,
    hasIhcForPrefix,
  )
  const runDensityBoth = useBothSlides(density.start, ihcDensity.start, hasIhcForPrefix)
  const runDeconvolutionBoth = useBothSlides(
    deconvolution.start,
    ihcDeconvolution.start,
    hasIhcForPrefix,
  )

  // --- what this step is actually showing ----------------------------------
  //
  // One alias per prefix step, resolved once. Everything below - the gates, the
  // task the page launches, the panel, the hint - reads these rather than picking
  // between the two instances at each site, because five steps times four call
  // sites is twenty chances to leave one of them pointed at the wrong slide.
  //: Which slides already have a quality-control report, cached or otherwise. An
  //: object rather than two booleans so the effect below has one stable dependency.
  const shownQcReportsExist = useMemo(
    () => ({ he: qc.report !== null, ihc: ihcQc.report !== null }),
    [qc.report, ihcQc.report],
  )

  const shownQc = showingIhcPrefix ? ihcQc : qc
  const shownTissue = showingIhcPrefix ? ihcTissue : tissue
  const shownCalibration = showingIhcPrefix ? ihcCalibration : calibration
  const shownDensity = showingIhcPrefix ? ihcDensity : density
  const shownDeconvolution = showingIhcPrefix ? ihcDeconvolution : deconvolution

  const isQC = stage?.id === 'quality-control'

  // Step 3 owns its own result too, but for a different reason: the threshold is
  // adjustable *after* the step has run, so the panel keeps re-running as the
  // viewer moves the slider rather than resolving once.
  const isTissue = stage?.id === 'tissue-mask'

  // Step 4 owns its own result for the same reason step 3 does - the percentile
  // is adjustable after the step has run - and it additionally has a precondition
  // no earlier step has: without step 3's mask there is no glass to sample, so
  // there is nothing for it to measure rather than merely nothing to compare.
  const isCalibration = stage?.id === 'white-calibration'
  // **Both slides, not the one on screen.** Step 4 runs on both, so its input is
  // ready only when step 3 has produced both - the H&E's mask says nothing about
  // where the immunostained slide's glass is. Gating on the displayed slide would
  // let a viewer sitting on the H&E tab start a step whose IHC half has no input.
  const bothOrLone = (he: unknown, ihc: unknown) =>
    he !== null && (!hasIhcForPrefix || ihc !== null)
  const hasTissueMask = bothOrLone(tissue.report, ihcTissue.report)

  // Step 5's precondition is the strongest in the pipeline so far, and it is a
  // definition rather than a dependency: optical density *is* -log10(I / I0), so
  // without step 4's white point there is no quantity to compute - not a rougher
  // one, none.
  const isDensity = stage?.id === 'optical-density'
  const hasWhitePoint = bothOrLone(calibration.report, ihcCalibration.report)

  // Step 6's precondition is the same kind as step 5's - a validity condition
  // rather than a dependency. Separating stains is a *linear* inverse, and two
  // stains only add up linearly in optical density; on the raw colour photo they
  // multiply, and the same matrix would return three confident numbers that are
  // not amounts of anything.
  const isDeconvolution = stage?.id === 'colour-deconvolution'
  const hasDensity = bothOrLone(density.report, ihcDensity.report)

  // Step 7's precondition is a promise rather than a computation. The grid is
  // pure geometry and could be laid out from the slide's dimensions alone - but
  // the index it hands on is addresses whose *pixels* are the haematoxylin
  // channel, so requiring step 6 is what keeps the fork honest.
  //: Steps 7-9: the ones that read the H&E and only the H&E.
  const isHeOnlyStep =
    stage?.runsOn?.length === 1 && stage.runsOn[0] === 'he'
  const isTiling = stage?.id === 'tiling'
  // NOT the shown one: this gates step 7, which runs on the H&E whatever the
  // prefix toggle is set to. Following the toggle here would let a viewer
  // inspecting the IHC slide's stains unlock - or block - the H&E's tiling.
  const hasChannels = deconvolution.report !== null

  // Step 8 is the one step that runs a trained model, so it has a precondition no
  // other implemented step has: the checkpoint must be on disk and torch must be
  // importable. Unlike step 2's models that is not a download - this one is trained
  // by this project - so the panel's setup state says so rather than pointing at a
  // release page that does not exist.
  const isTissueType = stage?.id === 'tissue-type-segmentation'
  const hasTiles = tiling.report !== null

  // Step 9's precondition is the same kind as step 8's own - a class map to turn into
  // a region - but step 9 itself is not gated: unlike step 8, building it costs a
  // fraction of a second, so arriving on it is not the kind of commitment that
  // deserves asking first.
  const isRoi = stage?.id === 'roi-mask'
  const hasClassMap = tissueType.report !== null

  // Step 10 is not gated: it opens no model and its cost is one slide read per
  // candidate, so arriving on it lists them. What it produces is a *choice*, and a
  // choice nobody made is still the pipeline's own default - which is exactly what
  // step 12 would have selected for itself before this step existed.
  const isRoiSelection = stage?.id === 'roi-selection'
  const hasRoi = roi.report !== null

  // Step 11 is gated, like steps 2 and 8 and for step 8's reason: it is minutes to
  // tens of minutes of CPU, and committing that because somebody navigated would be
  // worse than asking. Its precondition is a *non-empty* selection - not merely that
  // step 10 ran - because this step runs on what was ticked and on nothing else.
  const isRoiRefinement = stage?.id === 'roi-refinement'
  const selectedRoiCount = roiSelection.selected.length
  const selectedRoiWindows = roiSelection.report?.selectedWindows ?? 0

  // Step 10 is gated like steps 2 and 8, and for the same reason: registering two
  // whole-slide images is minutes of CPU, so arriving on the step offers to do it
  // rather than starting it. It also needs a second slide, which a session started
  // from a lone uploaded file does not have.
  // Step 12's input is no longer step 9's tile regions - it is step 11's refined
  // boundary, which the backend refuses to substitute for. So the gate here is a
  // *completed refinement*, and gating on the old tile regions would offer a run the
  // server would reject.
  const isIhcAlignment = stage?.id === 'ihc-alignment'
  const hasRefinedRegions = (roiRefinement.report?.completed ?? 0) > 0
  const hasIhcSlide = slide.ihcUploadId !== null

  // Step 11 has a precondition no earlier step has: a *person* must have signed off
  // step 10's alignment. The backend refuses without it - measuring inside regions
  // nobody has checked is the failure step 10's confirmation exists to prevent - so
  // the button is disabled here rather than letting the click return a 409.
  const isNuclei = stage?.id === 'nuclei-segmentation'
  const hasConfirmedAlignment =
    ihcAlignment.report?.state === 'ready' && ihcAlignment.report.confirmed === true

  // Step 12 opens no slide and runs no model, so it is neither gated nor
  // job-shaped: it needs only that step 11 has left it some nuclei to sort.
  const isCellTyping = stage?.id === 'cell-typing'
  const hasNuclei = (nuclei.report?.counted ?? 0) > 0

  // Step 13 needs step 12's classes, because compartments are built for tumour
  // cells only - a lymphocyte's membrane would put a cell in the numerator that
  // is not in the denominator.
  const isCompartments = stage?.id === 'compartments'
  const hasTypes = (cellTyping.report?.counted ?? 0) > 0

  // Step 14 is the first step that measures anything, and the last one that
  // touches a pixel. It needs step 13's geometry; everything after it is
  // arithmetic over what it wrote.
  const isPerCell = stage?.id === 'per-cell-measurement'
  const hasCompartments = (compartments.report?.cells ?? 0) > 0

  // Steps 15 and 16 need only their predecessor's result. Neither is gated -
  // both are milliseconds - so arriving on them runs them.
  const isBinning = stage?.id === 'intensity-binning'
  const hasMeasurements = (perCell.report?.cells ?? 0) > 0

  const isAggregate = stage?.id === 'aggregate'
  const hasBins = (binning.report?.cells ?? 0) > 0

  // Step 17 reports agreement against pathologist readings. It runs, and with no
  // reader sheet on disk what it reports is that there is nothing to compare
  // against - which is a result of the step, not a gap in it.
  const isValidation = stage?.id === 'validation'
  const hasScore = scores.report !== null

  /**
   * Steps 11 to 15 all draw step 11's outlines on the slide, so the outlines are
   * fetched once for whichever of them is on screen rather than by each panel.
   *
   * Per panel it would be five copies of the same request and, worse, a viewer
   * who walked into step 12 without having stood on step 11 - which is every
   * replay of a finished marker - would find a slide with nothing drawn on it
   * and read that as "this step found no cells".
   */
  const showsCells = isNuclei || isCellTyping || isCompartments || isPerCell || isBinning
  const { loadAllRegions } = nuclei
  useEffect(() => {
    if (showsCells && nuclei.report) void loadAllRegions()
  }, [loadAllRegions, nuclei.report, showsCells])

  // Two steps are *gated*: arriving on one does not start it, because both cost
  // minutes to tens of minutes and neither should begin because someone navigated.
  // Quality control is gated because it is genuinely optional - every later step
  // reads its artefact map as optional and says on screen when it is absent - so the
  // honest offer is "run it or skip it". Step 8 is gated because it is the longest
  // thing in the pipeline by an order of magnitude; it is not optional in the same
  // way, but committing half an hour of someone's CPU on a navigation would be worse
  // than asking.
  //
  // Step 7 used to be a third gate, and is not any more. It was never gated for cost -
  // it is seconds - but because its first question decided the grid's geometry *and*
  // which trained model step 8 would use, so auto-starting laid a grid nobody had
  // chosen. That question is now answered by the pipeline rather than the viewer
  // (`COMMITTED_BRANCH` in `useTiling`), so there is no unchosen grid to protect
  // against and the step starts on arrival like every other cheap one.
  //
  // Step 8 stays gated, and the asymmetry is deliberate: it is half an hour of CPU
  // against step 7's seconds, and committing that on a navigation would be worse than
  // asking. Quality control stays gated because it is genuinely optional - every later
  // step reads its artefact map as optional and says on screen when it is absent - so
  // the honest offer is "run it or skip it".
  // Step 11 joins the two gates, and for step 8's exact reason: it is the second
  // most expensive thing in the pipeline and the screen before it is where the person
  // decides how much of it to spend. Auto-starting it would spend that budget on the
  // default before they had finished reading the default.
  const isGated = isQC || isTissueType || isRoiRefinement

  // Step 7 is the one step that can be *working* while the pipeline is not running:
  // picking a field of view commits the choice and rebuilds the grid through the
  // panel's own controls rather than through `run`. Nothing may launch it, or walk
  // off it, in that window - the numbers on screen belong to the previous choice
  // until the new ones land.
  const stepBusy = isTiling && (tiling.loading || tiling.refining)

  // Every implemented step needs a real slide. Step 2 additionally needs the
  // models to be installed; the unbuilt steps need nothing, because they do
  // nothing. None of them invents a result.
  const canRun =
    (stage?.implemented ?? false) &&
    hasSlide &&
    !stepBusy &&
    (!isQC || canRunQC(shownQc)) &&
    (!isCalibration || hasTissueMask) &&
    (!isDensity || hasWhitePoint) &&
    (!isDeconvolution || hasDensity) &&
    // Just step 6 now. The two extra conditions here - a chosen branch and a chosen
    // field of view - were asking whether the viewer had answered step 7's questions.
    // The pipeline answers them, so they are true on arrival and checking them said
    // nothing.
    (!isTiling || hasChannels) &&
    (!isTissueType || (hasTiles && canRunTissueType(tissueType))) &&
    (!isRoi || hasClassMap) &&
    (!isRoiSelection || hasRoi) &&
    (!isRoiRefinement || selectedRoiCount > 0) &&
    (!isIhcAlignment || (hasRefinedRegions && hasIhcSlide)) &&
    (!isNuclei || (hasConfirmedAlignment && hasIhcSlide)) &&
    (!isCellTyping || hasNuclei) &&
    (!isCompartments || hasTypes) &&
    (!isPerCell || hasCompartments) &&
    (!isBinning || hasMeasurements) &&
    (!isAggregate || hasBins) &&
    (!isValidation || hasScore)
  const finished = run.finished && isLast

  /* --- what this step's work actually is ----------------------------------- */

  // One table rather than the same nested ternary in three places. The button, the
  // auto-start effect and the restart control all have to launch the *same* work,
  // and three copies of this chain would be three chances for them to drift.
  const task = isQC
    ? runQcBoth
    : isTissue
      ? runTissueBoth
      : isCalibration
        ? runCalibrationBoth
        : isDensity
          ? runDensityBoth
          : isDeconvolution
            ? runDeconvolutionBoth
            : isTiling
              ? tiling.start
              : isTissueType
                ? tissueType.start
                : isRoi
                  ? roi.start
                  : isRoiSelection
                    ? roiSelection.start
                    : isRoiRefinement
                      ? roiRefinement.start
                      : isIhcAlignment
                    ? ihcAlignment.start
                    : isNuclei
                      ? nuclei.start
                      : isCellTyping
                        ? cellTyping.start
                        : isCompartments
                          ? compartments.start
                          : isPerCell
                            ? perCell.start
                            : isBinning
                              ? binning.start
                              : isAggregate
                                ? scores.start
                                : isValidation
                                  ? validation.start
                                  : readBothSlides

  // Only the job-shaped steps can be stopped or re-run: they are the ones whose work
  // outlives a request, so they are the ones the server can be asked about. Step 11's
  // cancel is the odd one - it stops *after the current region* rather than inside it,
  // because a half-segmented region is not a result and the finished ones have to
  // survive being stopped.
  const restart = isQC
    ? shownQc.restart
    : isTissueType
      ? tissueType.restart
      : isRoiRefinement
        ? roiRefinement.restart
        : null
  const cancel = isQC
    ? shownQc.cancel
    : isTissueType
      ? tissueType.cancel
      : isRoiRefinement
        ? roiRefinement.cancel
        : null
  const cancelling = isQC
    ? shownQc.cancelling
    : isTissueType
      ? tissueType.cancelling
      : false

  const launch = useCallback(
    (work: () => Promise<void>) => {
      run.markAttempted(current)
      void run.run(work)
    },
    [current, run],
  )

  /* --- the fork in the rail -------------------------------------------------- */

  // Step 7's first question is a fork rather than a setting - it decides which
  // trained model step 8 runs - so the rail draws it as one, and drawing it there is
  // what makes it reachable again. The screen that asks it is replaced by its own
  // answer the moment the grid is built, which used to leave the choice behind a
  // ghost button at the bottom of a long result page. The diamond keeps it on the
  // pipeline, where the reader is already looking.
  //
  // It goes through the same `backToBranch` the in-page control does rather than a
  // second path of its own: that one clears the branch, the field of view and the
  // report together, and the effect above then invalidates every step after this
  // one. Two ways of unmaking step 7 would be two chances to unmake it differently.
  // `changeApproach` lived here: it sent the viewer back to step 7's approach
  // question and cleared the choice. With one committed configuration there is no
  // question to go back to. `tiling.backToBranch` still exists on the hook for
  // whoever re-opens the choice.

  /* --- filing a finished run ------------------------------------------------ */

  // Reaching the end of the walkthrough is what makes a run "previous", so that
  // is when it is filed: its directories are renamed out of the working tree and
  // into `data/history`, where the list screen can find it by case and antibody.
  //
  // **A rename, so there is never a second copy.** A marker's artefacts are about
  // half a gigabyte and a case's five are two and a half, so filing by copying
  // would double the largest thing on the disk to gain nothing.
  //
  // Failures are swallowed on purpose. Filing is housekeeping: a viewer who has
  // just reached the score should not be shown an error about where the bytes are
  // sitting, and the next start-up records the run wherever it is.
  const filed = useRef<string | null>(null)
  useEffect(() => {
    if (!run.finished || !slide.caseId || !slide.marker) return
    const key = `${slide.caseId}:${slide.marker}`
    if (filed.current === key) return
    filed.current = key
    void archiveHistoryMarker(slide.caseId, slide.marker).catch(() => {
      /* the start-up sweep records it wherever it ended up */
    })
  }, [run.finished, slide.caseId, slide.marker])

  /* --- starting a step by arriving on it ------------------------------------ */

  // The pipeline used to need three clicks to get one step done: run this one,
  // continue to the next, run that one. Arriving on a step now starts it, so
  // "continue" is the only button in the common path.
  //
  // Three guards, and each one exists because of a way this can go wrong:
  // `attempted` makes it happen at most once per step per slide, so a step that
  // fails is not retried for ever by the effect that started it; `isGated` holds
  // back the two expensive steps, which ask first; and `canRun` means a step whose
  // input is missing waits rather than failing on arrival.
  useEffect(() => {
    if (!stage?.implemented || isGated) return
    if (isComplete || isRunning || !canRun) return
    if (run.attempted(current) || run.error) return

    launch(task)
  }, [canRun, current, isComplete, isGated, isRunning, launch, run, stage, task])

  /* --- a decision that is its own run --------------------------------------- */

  // Step 7 asks two questions and the second one *is* the step: choosing a field of
  // view commits both answers and builds the grid, so the report is on screen before
  // the button at the bottom has been touched. Pressing it as well used to lay the
  // identical grid a second time to arrive at the number already above it. So the
  // choice marks the step done, and the button below it becomes "continue" - which is
  // the only thing left to do.
  //
  // Kept in step in both directions rather than latched on arrival of the report: the
  // in-page "choose a different approach" control throws the grid away, and a step
  // reading "Completed" over a screen that is back to asking its first question would
  // be a completion with nothing behind it. `setComplete` is a no-op when the flag
  // already agrees, so this cannot loop.
  const { setComplete } = run
  useEffect(() => {
    if (!isTiling || isRunning || stepBusy) return
    setComplete(current, tiling.report !== null)
  }, [current, isRunning, isTiling, setComplete, stepBusy, tiling.report])

  // A gated step whose result is already on disk has been run - just not in this
  // session - and the pipeline has to know that, or the viewer is stuck.
  //
  // **The trap, which is easy to walk into and hard to see from the code.** A gated
  // step is not launched by arriving on it, so it is never marked complete. But its
  // panel shows a cached report the moment one exists, and the "run it or skip it"
  // offer - the only thing carrying the Skip button - is part of the *setup* view
  // that report has just replaced. So on a slide processed at any point in the past,
  // the step offers "Back" and "Run it" and nothing else: no continue, no skip, and
  // the only way forward is to recompute an answer already on screen. Seven minutes
  // for quality control; half an hour for step 8.
  //
  // Both gated steps, not just the one it was first noticed on. They differ only in
  // how many slides have to be done: quality control runs on both, step 8 on the H&E.
  useEffect(() => {
    if (isRunning) return
    if (isQC) {
      const bothDone =
        shownQcReportsExist.he && (!hasIhcForPrefix || shownQcReportsExist.ihc)
      if (bothDone) setComplete(current, true)
      return
    }
    if (isTissueType && tissueType.report !== null) setComplete(current, true)
    // Step 11 is gated too, so it inherits the same trap: a slide whose regions were
    // refined in an earlier session shows its comparison immediately and would
    // otherwise offer no way forward but to spend the minutes again. `completed > 0`
    // rather than `report !== null`, because a report exists as soon as the first
    // region fails and a pass that produced no boundary has not run this step.
    if (isRoiRefinement && (roiRefinement.report?.completed ?? 0) > 0) {
      setComplete(current, true)
    }
  }, [
    current,
    hasIhcForPrefix,
    isQC,
    isRoiRefinement,
    isRunning,
    isTissueType,
    roiRefinement.report,
    setComplete,
    shownQcReportsExist,
    tissueType.report,
  ])

  /* --- space forward, backspace back --------------------------------------- */

  // The keys do whatever the highlighted button does, decided here once so the two
  // cannot drift: this mirrors the controls below, and the ordering of the branches is
  // the same as theirs. Null means the key does nothing - which is the honest state
  // while a step is running, while step 7 is rebuilding, on the finish screen, and on
  // any step whose input has not arrived yet.
  const advance = useCallback(() => {
    if (!stage || isRunning || stepBusy || finished) return
    if (stage.implemented && !isComplete) {
      // Includes the two gated steps, deliberately: on those the highlighted button
      // is "run it", so that is what space does. Where a choice has to be made first
      // - step 7's approach, a missing model, a step whose input is not ready -
      // `canRun` is false and the key does nothing rather than guessing an answer.
      if (canRun) launch(task)
      return
    }
    if (isComplete) {
      // On the last step the button restarts the walkthrough. A key that can throw
      // away the whole run is not a key anyone asked for, so space stops here.
      if (!isLast) run.next()
      return
    }
    // Nothing runs on this step, so moving past it is what marks it done.
    run.acknowledge()
  }, [
    canRun,
    finished,
    isComplete,
    isLast,
    isRunning,
    launch,
    run,
    stage,
    stepBusy,
    task,
  ])

  const back = useCallback(() => {
    if (isRunning || stepBusy || current === 0) return
    run.previous()
  }, [current, isRunning, run, stepBusy])

  // Whether space has anything to do, by the same branches `advance` takes. It is
  // computed rather than left to the callback returning early because a key that is
  // swallowed and does nothing is worse than an unbound one: space still scrolls the
  // step column, and on a screen that is waiting for a choice or for a slide, scrolling
  // is the reading the reader meant.
  const canAdvance =
    stage !== undefined &&
    !isRunning &&
    !stepBusy &&
    !finished &&
    (stage.implemented && !isComplete ? canRun : isComplete ? !isLast : true)

  const canGoBack = current > 0 && !isRunning && !stepBusy

  useStepKeys({
    onAdvance: canAdvance ? advance : null,
    onBack: canGoBack ? back : null,
  })

  /* --- nothing to draw yet ------------------------------------------------- */

  // Below every hook, deliberately. These two returns are purely presentational, and
  // a hook after an early return is a hook that runs on some renders and not others -
  // which React reads as a different component. That is why the flags above tolerate
  // a missing `stage` while everything below this point can rely on it.
  if (loading && stages.length === 0) {
    return (
      <div className="demo demo--loading">
        <div className="shimmer demo__skeleton" />
      </div>
    )
  }

  if (!stage) return null

  /* --- the three pipeline states, drawn from the run ----------------------- */

  // The rail has four states and the run has six. A skipped step is drawn as
  // complete - it is behind the pipeline and does not block anything - but it is
  // *labelled* as skipped, because a slide whose artefacts were segmented and one
  // whose artefacts were never looked at are different facts and every later step's
  // own report already distinguishes them.
  const trackStatus = (index: number): TrackStatus => {
    const state = run.statusOf(index)
    if (state === 'complete' || state === 'skipped') return 'complete'
    if (state === 'error') return 'error'
    if (state === 'running') return 'active'
    return index === current ? 'active' : 'pending'
  }

  const trackLabel = (index: number, state: TrackStatus): string => {
    // Callers only ever pass an index into `stages`, but the compiler cannot
    // know that; treating a missing stage as unbuilt is the safe reading.
    const built = stages[index]?.implemented ?? false
    if (run.skipped.has(index)) return 'Skipped'
    if (state === 'complete') return built ? 'Completed' : 'Stepped through'
    if (state === 'error') return 'Failed'
    if (state === 'pending') return 'Pending'
    if (isRunning) return 'Running…'
    return built ? 'Ready to run' : 'In progress'
  }

  const canSelect = (index: number) =>
    !isRunning && (index === 0 || run.completed.has(index - 1))

  // The label is the server's own - the same string the picker shows - so the rail
  // cannot name an option differently from the screen that offered it. Null until
  // the question has been answered, which is most of the walkthrough: the node is
  // drawn either way, because a fork nobody can see until after they have taken it
  // is the half of a flowchart that does not help.
  // The rail used to carry an "Approach" fork before step 7, because step 7 asked
  // which model to run and the answer changed everything after it. The pipeline has
  // committed to one configuration, so there is no decision to mark - and a fork node
  // whose value can never change is a control that teaches the reader something false
  // about how the pipeline works.
  const decisions: TrackDecision[] = []

  const currentState = trackStatus(current)

  return (
    <div className="demo">
      <aside className="demo__rail">
        <PipelineTrack
          items={stages}
          activeIndex={current}
          statusOf={trackStatus}
          labelOf={trackLabel}
          busy={isRunning}
          onSelect={run.goTo}
          canSelect={canSelect}
          decisions={decisions}
          label={`${stages.length}-step pipeline`}
        />
      </aside>

      <div className="demo__scroll" id="stage-panel" ref={stageRef}>
        <section className="demo__panel">
          {finished ? (
            <div className="finish anim-rise">
              <CompleteMark />
              <h2 className="stage__title">All steps complete</h2>
              <p className="stage__tagline">
                All {stages.length} steps have run on your case, in order. The first six ran
                once on each slide, because each one measures something about that piece of
                glass. Steps 7 to 11 found and outlined the tumour on the H&amp;E slide. Step
                12 copied those regions onto the marker slide, and steps 13 to 18 measured
                inside them: every cell found, sorted, given a measuring area, read for brown
                and turned into a score. Step 19 compared that score with pathologists.
              </p>
              <div className="controls__row">
                <Button size="lg" variant="secondary" onClick={run.reset}>
                  Start again from step 01
                </Button>
                <Button variant="ghost" onClick={() => run.goTo(0)}>
                  Review step 01
                </Button>
              </div>
            </div>
          ) : (
            <div className="stage" key={stage.id}>
              <div className="stage__eyebrow anim-rise-sm">
                <span className="eyebrow">
                  step {padIndex(stage.index)} / {padIndex(stages.length)}
                </span>
                <Badge tone={STATE_TONE[currentState]}>
                  {trackLabel(current, currentState)}
                </Badge>
                <Badge tone={APPROACH_TONE[stage.approach]}>{stage.approach}</Badge>
                {stage.trainsModel && <Badge tone="violet">uses a trained model</Badge>}
                {!stage.implemented && <Badge tone="warn">not built yet</Badge>}
                {/* One insertion point for all seventeen steps. Putting this in each
                    panel instead would be seventeen places for it to go stale, and
                    the panel does not know which slide it is on - the catalogue does. */}
                <SlideBadge
                  slide={slideChoice.selected}
                  caseId={slide.caseId}
                  isCaseScoped={slideChoice.isCaseScoped}
                  both={
                    stage.readsSlidesTogether ? slideChoice.options : undefined
                  }
                />
              </div>

              <h2 className="stage__title anim-rise" style={{ animationDelay: '60ms' }}>
                {stage.title}
              </h2>
              <p className="stage__tagline anim-rise" style={{ animationDelay: '120ms' }}>
                {stage.tagline}
              </p>

              {isHeOnlyStep && (heSlotIsNotHe || heSlotFilenameSaysIhc) ? (
                <div className="anim-rise stage__slides" style={{ animationDelay: '150ms' }}>
                  <p className="slide-warning">
                    <strong>This step needs the H&amp;E slide, and this does not look like
                    one.</strong>{' '}
                    {heSlotIsNotHe
                      ? 'Its stains were measured as blue and brown marker, not blue and pink.'
                      : 'Its filename looks like a marker slide.'}{' '}
                    The model used here was trained on H&amp;E slides. On a marker slide it
                    reports 0% tumour while still looking highly confident, so the result
                    would look normal and mean nothing. Load a case with its H&amp;E slide
                    to run this properly.
                  </p>
                </div>
              ) : null}

              {slideChoice.offersChoice ? (
                <div className="anim-rise stage__slides" style={{ animationDelay: '150ms' }}>
                  {/* The switch changes what is DISPLAYED and starts nothing. The
                      step runs on both slides as one unit of work, so there is no
                      state in which flipping this could queue a second job - which is
                      why it stays live even while the step is running. */}
                  <SlidePicker
                    options={slideChoice.options}
                    selected={slideChoice.selected}
                    onChoose={setPrefixRole}
                    summary={prefixSummary}
                    state={prefixState}
                  />
                </div>
              ) : null}

              {/* --- the working area ---------------------------------------- */}
              {replaying && hasSlide && (
                <div className="anim-rise stage__replay" style={{ animationDelay: '160ms' }}>
                  <strong>Showing a saved run.</strong> Every step already has its result
                  saved, so moving forward reads it instead of running it again. Changing a
                  setting will still recalculate the steps after it.
                </div>
              )}

              <div className="anim-rise stage__work" style={{ animationDelay: '180ms' }}>
                {!stage.implemented ? (
                  <div className="placeholder placeholder--unbuilt">
                    <span className="placeholder__badge mono">step {padIndex(stage.index)}</span>
                    <p className="placeholder__text">
                      This step is not built yet. What it would do is described below, but it
                      produces no result, so none is shown.
                    </p>
                  </div>
                ) : !hasSlide ? (
                  // Step 0 begins with a biomarker and a case folder, so the picker
                  // lives here rather than somewhere the reader has to go and find it.
                  upload.capability ? (
                    <CaseLoaderPanel />
                  ) : upload.probing ? (
                    <div className="shimmer demo__upload-skeleton" />
                  ) : (
                    // A failed probe used to leave the skeleton shimmering as if
                    // it were still loading. It is a dead end, so it says so.
                    <div className="placeholder placeholder--offline">
                      <span className="placeholder__badge mono">api unreachable</span>
                      <p className="placeholder__text">
                        The server did not respond, so there is nowhere to send a slide. Start
                        the backend on port 8000 and try again.
                      </p>
                      <Button variant="secondary" onClick={upload.retry}>
                        Try again
                      </Button>
                    </div>
                  )
                ) : isQC ? (
                  // QC owns all four of its own states - probing, setup needed,
                  // running, done - so it is handed the step's running flag
                  // rather than being switched on `isComplete` out here.
                  <QCPanel
                    qc={shownQc}
                    readout={showingIhcPrefix ? slide.ihcReadout : slide.readout}
                    running={isRunning}
                    // Gated, so the panel offers the choice rather than the page
                    // starting the run behind it.
                    onRun={() => launch(shownQc.start)}
                    onSkip={run.skip}
                    started={run.attempted(current)}
                  />
                ) : isTissue ? (
                  <TissuePanel tissue={shownTissue} running={isRunning} />
                ) : isCalibration ? (
                  <CalibrationPanel
                    calibration={shownCalibration}
                    hasTissueMask={hasTissueMask}
                    running={isRunning}
                  />
                ) : isDensity ? (
                  <DensityPanel
                    density={shownDensity}
                    hasWhitePoint={hasWhitePoint}
                    running={isRunning}
                  />
                ) : isDeconvolution ? (
                  <DeconvolutionPanel
                    deconvolution={shownDeconvolution}
                    hasDensity={hasDensity}
                    running={isRunning}
                    slideRole={slideChoice.selected?.role ?? 'unknown'}
                  />
                ) : isTiling ? (
                  <TilingPanel
                    tiling={tiling}
                    known={tileCounts.current}
                    hasChannels={hasChannels}
                    running={isRunning}
                  />
                ) : isTissueType ? (
                  // Owns all six of its own states - probing, offline, needs its
                  // model, blocked on step 7, running, done - so like QC it is handed
                  // the step's running flag rather than being switched on
                  // `isComplete` out here.
                  <TissueTypePanel
                    tissueType={tissueType}
                    hasTiles={hasTiles}
                    model={tiling.model}
                    fieldOfViewUm={tiling.fieldOfView}
                    running={isRunning}
                    onRun={() => launch(tissueType.start)}
                    onSkip={run.skip}
                    started={run.attempted(current)}
                  />
                ) : isRoi ? (
                  <RoiPanel roi={roi} hasClassMap={hasClassMap} running={isRunning} />
                ) : isRoiSelection ? (
                  <RoiSelectionPanel
                    selection={roiSelection}
                    hasRoi={hasRoi}
                    running={isRunning}
                  />
                ) : isRoiRefinement ? (
                  // Owns its own states - nothing ticked, running, partway, failed,
                  // done - and shows the finished regions *while* the pass runs, so
                  // like steps 2, 8 and 12 it takes the running flag rather than being
                  // switched on `isComplete`.
                  <RoiRefinementPanel
                    refinement={roiRefinement}
                    selectedCount={selectedRoiCount}
                    selectedWindows={selectedRoiWindows}
                    running={isRunning}
                  />
                ) : isIhcAlignment ? (
                  // Owns all of its own states - not installed, no ROI yet, running,
                  // refused, done-but-unconfirmed - so like steps 2 and 8 it is handed
                  // the running flag rather than being switched on `isComplete`.
                  <IhcAlignmentPanel
                    alignment={ihcAlignment}
                    hasRoi={hasRefinedRegions}
                    marker={slide.marker}
                    running={isRunning}
                    onRun={() => launch(ihcAlignment.start)}
                  />
                ) : isNuclei ? (
                  // Owns its own states like steps 2, 8 and 10 - model missing,
                  // alignment unconfirmed, running, done - so it takes the running
                  // flag rather than being switched on `isComplete`.
                  <NucleiPanel
                    nuclei={nuclei}
                    heUploadId={slide.uploadId}
                    ihcUploadId={slide.ihcUploadId}
                    marker={slide.marker}
                    ihcMpp={slide.ihcReadout?.mpp ?? null}
                    alignment={ihcAlignment.report}
                    hasConfirmedAlignment={hasConfirmedAlignment}
                    running={isRunning}
                    onRun={() => launch(nuclei.start)}
                  />
                ) : isCellTyping ? (
                  <CellTypingPanel
                    typing={cellTyping}
                    hasNuclei={hasNuclei}
                    ihcUploadId={slide.ihcUploadId}
                    ihcMpp={slide.ihcReadout?.mpp ?? null}
                    alignment={ihcAlignment.report}
                    geometry={nuclei.geometry}
                  />
                ) : isCompartments ? (
                  <CompartmentsPanel
                    compartments={compartments}
                    ihcUploadId={slide.ihcUploadId}
                    ihcMpp={slide.ihcReadout?.mpp ?? null}
                    alignment={ihcAlignment.report}
                    hasTypes={hasTypes}
                  />
                ) : isPerCell ? (
                  <PerCellPanel
                    perCell={perCell}
                    heUploadId={slide.uploadId}
                    ihcUploadId={slide.ihcUploadId}
                    ihcMpp={slide.ihcReadout?.mpp ?? null}
                    alignment={ihcAlignment.report}
                    geometry={nuclei.geometry}
                    hasCompartments={hasCompartments}
                    onRun={() => launch(perCell.start)}
                  />
                ) : isBinning ? (
                  <BinningPanel
                    binning={binning}
                    hasMeasurements={hasMeasurements}
                    ihcUploadId={slide.ihcUploadId}
                    ihcMpp={slide.ihcReadout?.mpp ?? null}
                    alignment={ihcAlignment.report}
                    geometry={nuclei.geometry}
                  />
                ) : isAggregate ? (
                  <ScorePanel scores={scores} hasBins={hasBins} />
                ) : isValidation ? (
                  <ValidationPanel validation={validation} hasScore={hasScore} />
                ) : isComplete && slide.readout ? (
                  <>
                    {slide.marker && (
                      <p className="stage__marker-note">
                        {slide.marker} is the biomarker on the IHC slide below. The H&E slide is
                        what steps 2–9 run on to find the tumour — it is not scored.
                      </p>
                    )}
                    <SlideReadoutView
                      readout={slide.readout}
                      busy={slide.adjusting}
                      onAdjust={(next) => void slide.adjust(next)}
                    />
                    {slide.ihcReadout && (
                      <>
                        <h3 className="stage__marker-note">IHC slide ({slide.marker})</h3>
                        <SlideReadoutView readout={slide.ihcReadout} />
                      </>
                    )}
                    {/* Reading the pyramid is this step's finish line, but it is
                        still the only screen the viewer has said "this is the
                        slide I meant" from — so the option to leave it for a
                        different one belongs here too, not just in the moment
                        before the pyramid was read. */}
                    <div className="readout__release">
                      <ReleaseSlide
                        uploadId={slide.uploadId}
                        filename={slide.status?.filename ?? null}
                        onRelease={slide.clear}
                      />
                    </div>
                  </>
                ) : (
                  <div className={isRunning ? 'placeholder placeholder--running' : 'placeholder'}>
                    {isRunning && <span className="placeholder__spinner" />}
                    <p className="placeholder__text">
                      {isRunning
                        ? `Opening ${slide.status?.filename ?? 'the slide'} and reading its pyramid…`
                        : `${slide.status?.filename ?? 'A slide'} is loaded and verified. Open it to read its pyramid.`}
                    </p>
                    {!isRunning && (
                      // Choosing a different slide is the one moment the viewer has
                      // said they are finished with this one, so it is the only honest
                      // place to offer to free the 0.8-1.6 GB it is holding. The
                      // pipeline reset deliberately does not: that means "walk this
                      // slide again", where deleting the results would cost half an
                      // hour of step 8 to save nineteen megabytes.
                      <ReleaseSlide
                        uploadId={slide.uploadId}
                        filename={slide.status?.filename ?? null}
                        onRelease={slide.clear}
                      />
                    )}
                  </div>
                )}
              </div>

              {/* --- controls ------------------------------------------------- */}
              <div className="controls">
                <div className="controls__row">
                  {current > 0 && (
                    <Button
                      variant="ghost"
                      onClick={run.previous}
                      disabled={isRunning || stepBusy}
                    >
                      ← Back
                    </Button>
                  )}

                  {isRunning ? (
                    // Running. The only useful controls are the ones that stop it,
                    // and for the two long steps that is a real server-side stop
                    // rather than an abandoned wait.
                    <>
                      <Button size="lg" loading disabled>
                        {isQC
                          ? 'Checking slide quality…'
                          : isTissue
                            ? 'Finding the tissue…'
                            : isCalibration
                              ? 'Measuring the blank glass…'
                              : isDensity
                                ? 'Measuring stain amount…'
                                : isDeconvolution
                                  ? 'Separating the stains…'
                                  : isTiling
                                    ? 'Laying out the tiles…'
                                    : isTissueType
                                      ? 'Labelling every patch…'
                                      : isRoi
                                        ? 'Outlining the tumour areas…'
                                        : isRoiSelection
                                          ? 'Cutting out each region…'
                                          : isRoiRefinement
                                            ? 'Tracing each area…'
                                            : 'Working…'}
                      </Button>
                      {cancel && (
                        <Button
                          variant="secondary"
                          loading={cancelling}
                          onClick={() => void cancel()}
                        >
                          {cancelling ? 'Stopping…' : 'Cancel'}
                        </Button>
                      )}
                    </>
                  ) : stage.implemented && !isComplete ? (
                    <>
                    <Button
                      size="lg"
                      attention={canRun}
                      disabled={!canRun}
                      onClick={() => launch(task)}
                    >
                      {run.error ? 'Try again' : stage.actionLabel}
                    </Button>
                    </>
                  ) : isComplete ? (
                    <>
                      {isLast ? (
                        <Button size="lg" variant="secondary" onClick={run.reset}>
                          Back to step 01
                        </Button>
                      ) : (
                        <Button
                          size="lg"
                          attention
                          disabled={stepBusy}
                          onClick={run.next}
                        >
                          Continue to step {padIndex(stage.index + 1)} →
                        </Button>
                      )}
                      {restart && !run.skipped.has(current) && (
                        // Only the two job-shaped steps offer this, because they are
                        // the only ones whose result the *server* is holding: for
                        // every other step, going back and moving a control already
                        // re-runs it. Re-running is the point rather than a reset -
                        // the cached answer is discarded and the work is done again.
                        <Button variant="ghost" onClick={() => launch(restart)}>
                          Run this step again
                        </Button>
                      )}
                    </>
                  ) : (
                    // Nothing runs on this step, so the only move is past it —
                    // which is what marks it done and lights the next one.
                    <Button size="lg" attention onClick={run.acknowledge}>
                      {isLast
                        ? 'Finish'
                        : `Continue to step ${padIndex(stage.index + 1)} →`}
                    </Button>
                  )}
                </div>

                {/* The keys do the same thing the buttons above do, and nobody
                    tries a key they have not been told about. Only offered when it
                    would actually move - a hint for a key that does nothing is worse
                    than no hint. */}
                {(canAdvance || canGoBack) && (
                  <p className="controls__keys">
                    {canAdvance && (
                      <>
                        <kbd className="kbd">space</kbd>{' '}
                        {stage.implemented && !isComplete
                          ? 'starts this step'
                          : 'goes to the next step'}
                      </>
                    )}
                    {canAdvance && canGoBack && <span className="controls__keys-sep">·</span>}
                    {canGoBack && (
                      <>
                        <kbd className="kbd">backspace</kbd> goes back a step
                      </>
                    )}
                  </p>
                )}

                <p className="controls__hint">
                  {run.error
                    ? run.error
                    : !stage.implemented
                      ? 'Nothing runs here yet — stepping past it moves the pipeline on.'
                      : !hasSlide
                        ? upload.capability || upload.probing
                          ? 'Upload a slide above to run this step.'
                          : 'The backend is not answering, so no slide can be uploaded yet.'
                        : isQC
                          ? qcHint(shownQc, shownQc.run, shownQc.report)
                          : isTissue
                            ? tissueHint(shownTissue)
                            : isCalibration
                              ? calibrationHint(shownCalibration, hasTissueMask)
                              : isDensity
                                ? densityHint(shownDensity, hasWhitePoint)
                                : isDeconvolution
                                  ? deconvolutionHint(shownDeconvolution, hasDensity)
                                  : isTiling
                                    ? tilingHint(tiling, hasChannels)
                                    : isTissueType
                                      ? tissueTypeHint(tissueType, hasTiles)
                                      : isRoi
                                        ? roiHint(roi, hasClassMap)
                                        : isRoiSelection
                                          ? roiSelectionHint(roiSelection, hasRoi)
                                          : isRoiRefinement
                                            ? roiRefinementHint(
                                                roiRefinement,
                                                selectedRoiCount,
                                              )
                                            : // Steps 12-19 have no hint file of their
                                              // own, and the generic line used to claim
                                              // they read the uploaded slide. Six of
                                              // them do not.
                                              slideHint(slideChoice, isComplete)}
                </p>
              </div>

              {/* An implemented step's own output says all of this, and says it
                  about the actual file; the generic prose would only repeat it. */}
              {!stage.implemented && (
                <>
                  <hr className="rule demo__divider" />
                  <StageFlow stage={stage} status={status} />
                  <div className="demo__results">
                    <StageDetails stage={stage} />
                  </div>
                </>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
