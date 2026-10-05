/**
 * Step 2's whole work area, choosing between its four states.
 *
 *   probing   the capability endpoint imports torch, which takes a moment
 *   setup     something is missing — show the checklist, not an error
 *   running   a real run, with real per-patch progress
 *   report    the result
 *
 * The model picker sits here rather than in the report, because it changes what
 * a run *is*: 5x, 7x and 10x are three different checkpoints with different
 * accuracy and very different runtimes, and on a CPU-only machine that choice
 * is the difference between three minutes and ten.
 *
 * **This step asks before it runs, where the others just go.** Arriving on any other
 * implemented step starts it; this one offers a choice, for two reasons. It costs
 * minutes on a CPU, and — unlike every other step here — it is genuinely optional:
 * steps 3, 4, 5 and 7 all read its artefact map as optional and say on screen when it
 * is absent, so skipping produces a pipeline that works and is honest about what it
 * did not check. What skipping must never do is look like a run, so the choice is
 * spelled out and the rail marks the step "Skipped" rather than "Completed".
 */

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import type { SlideReadout } from '@/types/slide'

import { QCProgress } from './QCProgress'
import { QCSetup } from './QCSetup'
import { QCView } from './QCView'
import type { QualityControl } from './useQualityControl'

import './qc.css'

interface QCPanelProps {
  qc: QualityControl
  readout: SlideReadout | null
  /** True while the pipeline considers this step to be running. */
  running: boolean
  /** Start the run. The page owns this so the step is marked as attempted. */
  onRun: () => void
  /** Walk past it, recording that it did not run. */
  onSkip: () => void
  /** Whether a start has already been asked for on this slide. */
  started: boolean
}

/** The three published GrandQC artefact models, coarsest first. */
const MODEL_CHOICES: Array<{ mpp: number; label: string; hint: string }> = [
  { mpp: 2.0, label: '5x', hint: 'fastest, least detail' },
  { mpp: 1.5, label: '7x', hint: 'recommended' },
  { mpp: 1.0, label: '10x', hint: 'most detail, slowest' },
]

export function QCPanel({
  qc,
  readout,
  running,
  onRun,
  onSkip,
  started,
}: QCPanelProps) {
  const { capability, probing, report, run, busy, error } = qc

  if (probing && !capability) {
    return <div className="shimmer demo__upload-skeleton" />
  }

  // The probe finished and returned nothing, so the backend is unreachable or
  // erroring. Say that, rather than falling through to a "ready" badge over a
  // panel with no facts in it.
  //
  // **Skip is offered here too, and leaving it out was a dead end.** This branch
  // used to carry only "Try again": on a machine where the probe keeps failing -
  // or a stored run being replayed, where quality control was never run in the
  // first place - the walkthrough stopped at step 2 with no way forward. Neither
  // run nor skip nor continue. The step below it is the only one in the pipeline
  // that is genuinely optional, and saying so in the branch next door while
  // withholding it here was the difference between a hiccup and a wall.
  if (!capability) {
    return (
      <div className="qc-setup">
        <div className="qc-setup__head">
          <Badge tone="warn">unavailable</Badge>
          <p className="qc-setup__reason">
            {error ?? 'The quality check did not respond.'} The first attempt takes a few
            seconds to warm up, so trying again usually works.
          </p>
        </div>
        <div className="qc-gate__choice">
          <Button variant="secondary" onClick={qc.retry} loading={probing}>
            Try again
          </Button>
          <Button variant="ghost" onClick={onSkip}>
            Skip this step
          </Button>
          <p className="qc-progress__note">
            This step is optional. Later steps work without it and will say on screen that
            the check was skipped.
          </p>
        </div>
      </div>
    )
  }

  if (capability.mode === 'unavailable') {
    return (
      <>
        <QCSetup capability={capability} />
        <div className="qc-gate__choice">
          <Button variant="secondary" onClick={onSkip}>
            Skip this step
          </Button>
          <p className="qc-progress__note">
            This step is optional. Later steps work without it and will say on screen that
            the check was skipped.
          </p>
        </div>
      </>
    )
  }

  if ((busy || running) && !report) {
    return <QCProgress run={run} params={run?.params ?? null} />
  }

  if (report) {
    return (
      <>
        {capability.mode === 'degraded' && (
          <p className="qc-gate__warning">{capability.reason}</p>
        )}
        <QCView report={report} readout={readout} />
      </>
    )
  }

  // Ready, not yet run. Offer the choice that actually matters before starting.
  return (
    <div className="qc-setup">
      <div className="qc-setup__head">
        <Badge tone="accent">ready</Badge>
        <p className="qc-setup__reason">{capability.reason}</p>
      </div>

      <div className="qc-fix">
        <span className="eyebrow">how closely to look</span>
        <div className="qc-map__layers">
          {MODEL_CHOICES.filter((choice) =>
            capability.availableModelMpps.includes(choice.mpp),
          ).map((choice) => (
            <button
              key={choice.mpp}
              type="button"
              className={
                qc.modelMpp === choice.mpp ? 'qc-map__tab qc-map__tab--on' : 'qc-map__tab'
              }
              onClick={() => qc.setModelMpp(choice.mpp)}
            >
              {choice.label} · {choice.mpp} µm/px
              <span className="qc-downloads__into"> {choice.hint}</span>
            </button>
          ))}
        </div>
        <p className="qc-progress__note">
          All three do the same job at different levels of detail. Looking closer finds
          smaller problems but takes longer. 7x is a good default.
          {capability.device === 'cpu' &&
            ' This computer has no graphics card, so expect a few minutes either way.'}
        </p>
      </div>

      {error && <p className="qc-inspect__error">{error}</p>}

      {/* The choice, spelled out. Not a single primary button with a quiet "skip"
          beside it: skipping is a legitimate answer here and reads as one. */}
      <div className="qc-gate__choice">
        <Button size="lg" attention onClick={onRun}>
          {started ? 'Try the check again' : 'Run the check'}
        </Button>
        <Button variant="secondary" onClick={onSkip}>
          Skip this step
        </Button>
        <p className="qc-progress__note">
          Skipping is fine and is recorded as a skip, not as a pass. If you skip, blurred and
          folded areas stay in, and later steps will still give an answer for them &mdash; it
          just will not be a reliable one.
        </p>
      </div>

      <p className="qc-citation">
        {capability.citation}
        <br />
        <em>{capability.licenceNote}</em>
      </p>
    </div>
  )
}
