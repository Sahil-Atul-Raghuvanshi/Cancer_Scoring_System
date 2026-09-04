import { useCallback, useEffect, useRef } from 'react'

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
import type { TrackStatus } from '@/features/pipeline/components/PipelineTrack'
import { SlideReadoutView } from '@/features/pipeline/components/SlideReadoutView'
import { StageDetails } from '@/features/pipeline/components/StageDetails'
import { StageFlow } from '@/features/pipeline/components/StageFlow'
import '@/features/pipeline/components/pipeline.css'
import { usePipelineRun } from '@/features/pipeline/hooks/usePipelineRun'
import { QCPanel } from '@/features/qc/QCPanel'
import { canRunQC, qcHint } from '@/features/qc/qcStatus'
import { useQualityControl } from '@/features/qc/useQualityControl'
import { TilingPanel } from '@/features/tiling/TilingPanel'
import { tilingHint } from '@/features/tiling/tilingStatus'
import { useTiling } from '@/features/tiling/useTiling'
import { TissuePanel } from '@/features/tissue/TissuePanel'
import { tissueHint } from '@/features/tissue/tissueStatus'
import { useTissueMask } from '@/features/tissue/useTissueMask'
import { TissueTypePanel } from '@/features/tissueType/TissueTypePanel'
import { canRunTissueType, tissueTypeHint } from '@/features/tissueType/tissueTypeStatus'
import { useTissueType } from '@/features/tissueType/useTissueType'
import { ReleaseSlide } from '@/features/upload/ReleaseSlide'
import { UploadPanel } from '@/features/upload/UploadPanel'
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

export function DemoPage({ stages, loading, upload }: DemoPageProps) {
  const slide = useSlideSession()
  const run = usePipelineRun(stages)
  const qc = useQualityControl(slide.uploadId)
  const tissue = useTissueMask(slide.uploadId)

  // Step 4's input is step 3's mask, so it is handed step 3's *committed* cut
  // rather than reading one of its own. A manual cut is passed through; an
  // automatic one is passed as null, so the server re-derives it by whichever
  // rule the histogram selects instead of receiving it as a hand-set number.
  const tissueCut =
    tissue.report?.threshold.source === 'manual' ? tissue.report.threshold.value : null
  const calibration = useWhiteCalibration(slide.uploadId, tissueCut)

  // Step 5 is handed the same cut, and for the same reason one step further
  // along: a different mask gives step 4 different glass, and step 4's I0 is the
  // denominator of every density step 5 computes. It is deliberately not handed
  // step 4's *percentile* - that is step 4's own comparison control, and step 5
  // asks for the server's default so its numbers describe the calibration the
  // pipeline would really use.
  const density = useOpticalDensity(slide.uploadId, tissueCut)

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

  // Step 7 is handed step 3's cut and nothing else. Unlike step 6 it does not
  // follow step 5's tile: it is not about one field of view, it is about all of
  // them, and its input is the mask rather than the density.
  const tiling = useTiling(slide.uploadId, tissueCut)

  // Step 8 is handed the same cut, and for the same reason as step 7: its input is
  // step 7's index, and a different mask is a different set of tiles and therefore a
  // different class map.
  //
  // It is also handed step 7's *overlap*, which completes the join: step 7 already
  // lays this model's own square, read from the checkpoint's manifest, so with the
  // overlap shared the two steps run one grid. That is what makes step 7's "squares
  // to process" the literal number of forward passes step 8 makes, rather than a
  // proxy for it that was out by a factor of twenty.
  const tissueType = useTissueType(slide.uploadId, tissueCut, tiling.overlap)

  // The tile count at each overlap the viewer has actually visited, so step 7's
  // picker can put a price on each choice. Accumulated here rather than inside
  // the hook because it must survive the hook clearing its report on a rebuild -
  // and deliberately not pre-filled with an estimate: an estimated tile count
  // would be indistinguishable on screen from a measured one.
  const tileCounts = useRef(new Map<number, number>())
  if (tiling.report) {
    tileCounts.current.set(tiling.report.params.overlap, tiling.report.funnel.clean)
  }

  const { stage, status, current } = run
  const stageRef = useRef<HTMLDivElement>(null)

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
  const hasTissueMask = tissue.report !== null

  // Step 5's precondition is the strongest in the pipeline so far, and it is a
  // definition rather than a dependency: optical density *is* -log10(I / I0), so
  // without step 4's white point there is no quantity to compute - not a rougher
  // one, none.
  const isDensity = stage?.id === 'optical-density'
  const hasWhitePoint = calibration.report !== null

  // Step 6's precondition is the same kind as step 5's - a validity condition
  // rather than a dependency. Separating stains is a *linear* inverse, and two
  // stains only add up linearly in optical density; on the raw colour photo they
  // multiply, and the same matrix would return three confident numbers that are
  // not amounts of anything.
  const isDeconvolution = stage?.id === 'colour-deconvolution'
  const hasDensity = density.report !== null

  // Step 7's precondition is a promise rather than a computation. The grid is
  // pure geometry and could be laid out from the slide's dimensions alone - but
  // the index it hands on is addresses whose *pixels* are the haematoxylin
  // channel, so requiring step 6 is what keeps the fork honest.
  const isTiling = stage?.id === 'tiling'
  const hasChannels = deconvolution.report !== null

  // Step 8 is the one step that runs a trained model, so it has a precondition no
  // other implemented step has: the checkpoint must be on disk and torch must be
  // importable. Unlike step 2's models that is not a download - this one is trained
  // by this project - so the panel's setup state says so rather than pointing at a
  // release page that does not exist.
  const isTissueType = stage?.id === 'tissue-type-segmentation'
  const hasTiles = tiling.report !== null

  // Two steps are *gated*: arriving on one does not start it, because both cost
  // minutes to tens of minutes and neither should begin because someone navigated.
  // Quality control is gated because it is genuinely optional - every later step
  // reads its artefact map as optional and says on screen when it is absent - so the
  // honest offer is "run it or skip it". Step 8 is gated because it is the longest
  // thing in the pipeline by an order of magnitude; it is not optional in the same
  // way, but committing half an hour of someone's CPU on a navigation would be worse
  // than asking.
  const isGated = isQC || isTissueType

  // Every implemented step needs a real slide. Step 2 additionally needs the
  // models to be installed; the unbuilt steps need nothing, because they do
  // nothing. None of them invents a result.
  const canRun =
    (stage?.implemented ?? false) &&
    hasSlide &&
    (!isQC || canRunQC(qc)) &&
    (!isCalibration || hasTissueMask) &&
    (!isDensity || hasWhitePoint) &&
    (!isDeconvolution || hasDensity) &&
    (!isTiling || hasChannels) &&
    (!isTissueType || (hasTiles && canRunTissueType(tissueType)))
  const finished = run.finished && isLast

  /* --- what this step's work actually is ----------------------------------- */

  // One table rather than the same nested ternary in three places. The button, the
  // auto-start effect and the restart control all have to launch the *same* work,
  // and three copies of this chain would be three chances for them to drift.
  const task = isQC
    ? qc.start
    : isTissue
      ? tissue.start
      : isCalibration
        ? calibration.start
        : isDensity
          ? density.start
          : isDeconvolution
            ? deconvolution.start
            : isTiling
              ? tiling.start
              : isTissueType
                ? tissueType.start
                : slide.readSlide

  // Only the two job-shaped steps can be stopped or re-run: they are the ones whose
  // work outlives a request, so they are the ones the server can be asked about.
  const restart = isQC ? qc.restart : isTissueType ? tissueType.restart : null
  const cancel = isQC ? qc.cancel : isTissueType ? tissueType.cancel : null
  const cancelling = isQC ? qc.cancelling : isTissueType ? tissueType.cancelling : false

  const launch = useCallback(
    (work: () => Promise<void>) => {
      run.markAttempted(current)
      void run.run(work)
    },
    [current, run],
  )

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
          label={`${stages.length}-step pipeline`}
        />
      </aside>

      <div className="demo__scroll" id="stage-panel" ref={stageRef}>
        <section className="demo__panel">
          {finished ? (
            <div className="finish anim-rise">
              <CompleteMark />
              <h2 className="stage__title">Pipeline complete</h2>
              <p className="stage__tagline">
                All {stages.length} steps have been walked through, in the order they have to run.
                Steps 01 to 08 did real work on the slide you uploaded — reading its
                pyramid, running quality control, thresholding tissue against glass, measuring
                what zero stain means on it, turning one tile&rsquo;s colour into how much
                stain sits in each pixel, pulling the blue stain apart from the brown one,
                cutting the tissue into the squares a model can take, and showing every one of
                those to the trained model that says which tissue is tumour that has broken
                out of the ducts. Step 06 is where the pipeline forks: the blue picture goes to
                the model that finds the tumour, the brown picture to the measurement. Step 09
                onwards — starting with turning those patch-by-patch answers into one smooth
                region — is documented and still unbuilt.
              </p>
              <div className="controls__row">
                <Button size="lg" variant="secondary" onClick={run.reset}>
                  Start again at step 01
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
                {stage.trainsModel && <Badge tone="violet">the one you train</Badge>}
                {!stage.implemented && <Badge tone="warn">not built yet</Badge>}
              </div>

              <h2 className="stage__title anim-rise" style={{ animationDelay: '60ms' }}>
                {stage.title}
              </h2>
              <p className="stage__tagline anim-rise" style={{ animationDelay: '120ms' }}>
                {stage.tagline}
              </p>

              {/* --- the working area ---------------------------------------- */}
              <div className="anim-rise stage__work" style={{ animationDelay: '180ms' }}>
                {!stage.implemented ? (
                  <div className="placeholder placeholder--unbuilt">
                    <span className="placeholder__badge mono">step {padIndex(stage.index)}</span>
                    <p className="placeholder__text">
                      This step is not implemented. What it does, why it belongs here and how it
                      would be built are below — but it produces no output, so none is shown.
                    </p>
                  </div>
                ) : !hasSlide ? (
                  // Step 1 begins with getting a slide in, so the drop zone lives
                  // here rather than somewhere the reader has to go and find it.
                  upload.capability ? (
                    <UploadPanel capability={upload.capability} />
                  ) : upload.probing ? (
                    <div className="shimmer demo__upload-skeleton" />
                  ) : (
                    // A failed probe used to leave the skeleton shimmering as if
                    // it were still loading. It is a dead end, so it says so.
                    <div className="placeholder placeholder--offline">
                      <span className="placeholder__badge mono">api unreachable</span>
                      <p className="placeholder__text">
                        The backend did not answer, so there is nowhere to send a slide. Start
                        the API on port 8000 and try again — nothing here is faked while it is
                        down.
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
                    qc={qc}
                    readout={slide.readout}
                    running={isRunning}
                    // Gated, so the panel offers the choice rather than the page
                    // starting the run behind it.
                    onRun={() => launch(qc.start)}
                    onSkip={run.skip}
                    started={run.attempted(current)}
                  />
                ) : isTissue ? (
                  <TissuePanel tissue={tissue} running={isRunning} />
                ) : isCalibration ? (
                  <CalibrationPanel
                    calibration={calibration}
                    hasTissueMask={hasTissueMask}
                    running={isRunning}
                  />
                ) : isDensity ? (
                  <DensityPanel
                    density={density}
                    hasWhitePoint={hasWhitePoint}
                    running={isRunning}
                  />
                ) : isDeconvolution ? (
                  <DeconvolutionPanel
                    deconvolution={deconvolution}
                    hasDensity={hasDensity}
                    running={isRunning}
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
                    running={isRunning}
                    onRun={() => launch(tissueType.start)}
                    onSkip={run.skip}
                    started={run.attempted(current)}
                  />
                ) : isComplete && slide.readout ? (
                  <SlideReadoutView
                    readout={slide.readout}
                    busy={slide.adjusting}
                    onAdjust={(next) => void slide.adjust(next)}
                  />
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
                    <Button variant="ghost" onClick={run.previous} disabled={isRunning}>
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
                          ? 'Running quality control…'
                          : isTissue
                            ? 'Thresholding…'
                            : isCalibration
                              ? 'Sampling the glass…'
                              : isDensity
                                ? 'Taking the logarithm…'
                                : isDeconvolution
                                  ? 'Separating the stains…'
                                  : isTiling
                                    ? 'Laying out the grid…'
                                    : isTissueType
                                      ? 'Classifying every patch…'
                                      : 'Reading…'}
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
                    <Button
                      size="lg"
                      attention={canRun}
                      disabled={!canRun}
                      onClick={() => launch(task)}
                    >
                      {run.error ? 'Try again' : stage.actionLabel}
                    </Button>
                  ) : isComplete ? (
                    <>
                      {isLast ? (
                        <Button size="lg" variant="secondary" onClick={run.reset}>
                          Back to step 01
                        </Button>
                      ) : (
                        <Button size="lg" attention onClick={run.next}>
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
                          Run it again
                        </Button>
                      )}
                    </>
                  ) : (
                    // Nothing runs on this step, so the only move is past it —
                    // which is what marks it done and lights the next one.
                    <Button size="lg" attention onClick={run.acknowledge}>
                      {isLast
                        ? 'Finish the walkthrough'
                        : `Continue to step ${padIndex(stage.index + 1)} →`}
                    </Button>
                  )}
                </div>

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
                          ? qcHint(qc, qc.run, qc.report)
                          : isTissue
                            ? tissueHint(tissue)
                            : isCalibration
                              ? calibrationHint(calibration, hasTissueMask)
                              : isDensity
                                ? densityHint(density, hasWhitePoint)
                                : isDeconvolution
                                  ? deconvolutionHint(deconvolution, hasDensity)
                                  : isTiling
                                    ? tilingHint(tiling, hasChannels)
                                    : isTissueType
                                      ? tissueTypeHint(tissueType, hasTiles)
                                      : isComplete
                                  ? 'Every figure above was read from your slide.'
                                  : 'Reads the file you uploaded — nothing is precomputed.'}
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
