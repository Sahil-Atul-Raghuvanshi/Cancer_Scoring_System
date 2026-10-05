/**
 * Step 8's work area, choosing between its six states.
 *
 *   probing    asking the server whether it can run at all
 *   offline    the probe failed — the backend is not answering
 *   setup      no checkpoint published, or torch missing, and exactly what to do
 *   blocked    step 7 has not run, so there is no tissue to classify
 *   untrained  step 7 chose a field of view no published head was fitted at
 *   running    the pass, with real progress and an estimate
 *   report     the class map
 *
 * The setup state is longer than the other steps' equivalents on purpose. Every
 * other model in this pipeline is downloaded; this one is **trained by this
 * project**, so "it is not there" has a different fix — a training run in a sibling
 * folder — and saying "install the weights" would send someone looking for a
 * download that does not exist.
 */

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'

import { TissueTypeProgress } from './TissueTypeProgress'
import { TissueTypeView } from './TissueTypeView'
import type { TissueTypeState } from './useTissueType'

import './tissueType.css'

interface TissueTypePanelProps {
  tissueType: TissueTypeState
  /** True once step 7 has built a tile index for this slide. */
  hasTiles: boolean
  /**
   * The checkpoint step 7's field of view resolved to, or null when none is published
   * at that scale. Null blocks this step: the alternative is running a head fitted at
   * a different field of view over step 7's grid, which is the one mistake this whole
   * join exists to prevent.
   */
  model: string | null
  /** The field of view step 7 chose, for saying which head is missing. */
  fieldOfViewUm: number | null
  /** True while the pipeline considers this step to be running. */
  running: boolean
  /** Start the pass. The page owns this so the step is marked as attempted. */
  onRun: () => void
  /** Walk past it, recording that it did not run. */
  onSkip: () => void
  /** Whether a start has already been asked for on this slide. */
  started: boolean
}

export function TissueTypePanel({
  tissueType,
  hasTiles,
  model,
  fieldOfViewUm,
  running,
  onRun,
  onSkip,
  started,
}: TissueTypePanelProps) {
  const { capability, probing, probeError, report, run, error } = tissueType

  if (report) {
    return (
      <TissueTypeView
        report={report}
        visible={tissueType.visible}
        allClasses={tissueType.allClasses}
        onToggleClass={tissueType.toggleClass}
        onShowAll={tissueType.showAllClasses}
      />
    )
  }

  if (tissueType.running || running) {
    return (
      <TissueTypeProgress
        run={run}
        params={run?.params ?? null}
        painted={tissueType.painted}
        paintedMasks={tissueType.paintedMasks}
        paintedCount={tissueType.paintedCount}
      />
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">this step could not run</span>
        <p className="placeholder__text">{error}</p>
        <div className="tt-gate__choice">
          <Button variant="secondary" onClick={onRun}>
            Try again
          </Button>
          <Button variant="ghost" onClick={onSkip}>
            Skip this step
          </Button>
        </div>
      </div>
    )
  }

  if (probing) {
    return <div className="shimmer tt-skeleton" />
  }

  if (probeError || !capability) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">api unreachable</span>
        <p className="placeholder__text">
          The backend did not answer, so there is no way to tell whether the model is
          installed. {probeError ?? ''}
        </p>
        <Button variant="secondary" onClick={tissueType.retryProbe}>
          Try again
        </Button>
      </div>
    )
  }

  if (!capability.ready) {
    return (
      <div className="tt-setup">
        <span className="placeholder__badge mono">this step needs its model</span>
        <p className="tt-setup__lead">{capability.reason}</p>
        <p className="tt-setup__note">
          The models trained by this project are written into{' '}
          <code>{capability.modelsRoot ?? 'models/tissue_type/'}</code> and need no
          download. The published network from another group is a 1.9 GB download,
          fetched by <code>setup.py</code>, and is licensed for research only.
        </p>

        {capability.models.length > 0 && (
          <ul className="tt-setup__models">
            {capability.models.map((entry) => (
              <li key={entry.name}>
                <span className="mono">{entry.name}</span>{' '}
                <Badge tone={entry.licenceTrack === 'permissive' ? 'success' : 'warn'}>
                  {entry.licenceTrack}
                </Badge>
                {entry.problem && (
                  <span className="tt-setup__problem"> {entry.problem}</span>
                )}
              </li>
            ))}
          </ul>
        )}

        <p className="tt-setup__note">
          To check the installation:{' '}
          <code>python scripts/check_tissue_model.py</code> in <code>backend/</code>.
        </p>
      </div>
    )
  }

  if (!hasTiles) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">cut into tiles first</span>
        <p className="placeholder__text">
          The model is ready, but it needs the list of squares worth looking at. Without
          it, every patch of empty glass would be checked too &mdash; about four times
          the work for no answer. Run the previous step, then come back.
        </p>
      </div>
    )
  }

  if (model === null) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">no model at this scale</span>
        <p className="placeholder__text">
          The squares are{' '}
          {fieldOfViewUm === null
            ? 'at the chosen size'
            : `${fieldOfViewUm.toFixed(0)} µm across`}
          , and no model has been trained for squares that size yet. The squares are
          real, but there is nothing to show them to.
        </p>
        <p className="placeholder__text">
          We never fall back to a model trained at a different size. A model shown
          squares it was never taught reads them wrongly and still reports high
          confidence, which is the one failure this step is built to prevent.
        </p>
        <div className="tt-gate__choice">
          <Button variant="ghost" onClick={onSkip}>
            Skip this step
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="tt-ready">
      <p className="tt-ready__lead">
        This shows every square the previous step kept to a trained model and asks what
        kind of tissue it is.
      </p>
      <p className="tt-ready__warn">
        <strong>It takes tens of minutes.</strong> Each square goes through the model
        separately and there are tens of thousands of them, so without a graphics card
        this is by far the slowest step. You can stop it at any point, and the result is
        saved when it finishes, so you only wait once.
      </p>
      {capability.licenceTrack !== 'permissive' && (
        <p className="tt-ready__licence">
          <Badge tone="warn">research only</Badge> {capability.licenceNote}
        </p>
      )}

      {/* Asked rather than started. Arriving on any other implemented step runs it;
          this one is the longest thing in the pipeline by an order of magnitude, and
          committing half an hour of someone's processor because they navigated would
          be worse than asking. Skipping is offered too — but note it is not the same
          kind of skip as step 2's: nothing downstream can run without a class map, so
          this is "not now" rather than "not needed". */}
      <div className="tt-gate__choice">
        <Button size="lg" attention onClick={onRun}>
          {started ? 'Try again' : 'Label every patch'}
        </Button>
        <Button variant="secondary" onClick={onSkip}>
          Skip for now
        </Button>
      </div>
      <p className="tt-ready__warn">
        Skipping leaves the pipeline with no tissue labels, so the steps after this one
        have nothing to work on. It is here so you can look through the rest of the
        pipeline without waiting &mdash; not because the step is optional.
      </p>
    </div>
  )
}
