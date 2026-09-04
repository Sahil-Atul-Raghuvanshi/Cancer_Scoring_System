/**
 * Step 8's work area, choosing between its six states.
 *
 *   probing    asking the server whether it can run at all
 *   offline    the probe failed — the backend is not answering
 *   setup      no checkpoint published, or torch missing, and exactly what to do
 *   blocked    step 7 has not run, so there is no tissue to classify
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
        onToggleClass={tissueType.toggleClass}
        onShowAll={tissueType.showAllClasses}
      />
    )
  }

  if (tissueType.running || running) {
    return <TissueTypeProgress run={run} params={run?.params ?? null} />
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">step 08 could not run</span>
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
        <span className="placeholder__badge mono">step 08 needs its model</span>
        <p className="tt-setup__lead">{capability.reason}</p>
        <p className="tt-setup__note">
          Every other model in this pipeline is downloaded from its authors. This one
          is trained by this project, so there is nothing to download &mdash; it is
          built in <code>bcss_hchannel_resnet18</code>, whose publish step writes into{' '}
          <code>{capability.modelsRoot ?? 'models/tissue_type/'}</code> so that the
          trained file and the served file are the same file.
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
          To check the installation and the training-to-serving agreement in one go:{' '}
          <code>python scripts/check_tissue_model.py</code> in <code>backend/</code>.
        </p>
      </div>
    )
  }

  if (!hasTiles) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">step 07 first</span>
        <p className="placeholder__text">
          The model is ready, but it needs to be told which parts of the scan are
          worth looking at &mdash; and that is the list of squares the previous step
          produces. Without it every patch of empty glass would be shown to the model
          too, which on this slide is about four times the work for no answer. Cut the
          tissue into squares first, then come back.
        </p>
      </div>
    )
  }

  return (
    <div className="tt-ready">
      <p className="tt-ready__lead">
        This runs the one trained model in the pipeline over every patch of tissue the
        previous step kept, and asks it what kind of tissue each one is.
      </p>
      <p className="tt-ready__note">
        The patches here are exactly the squares the previous step kept &mdash; same
        size, same places, same count. That step cuts the slide to this model&rsquo;s
        window rather than to a size of its own, because the window is fixed by how
        the model was taught and cannot be changed.
      </p>
      <p className="tt-ready__warn">
        <strong>It takes tens of minutes.</strong> Each patch is a separate pass
        through a neural network and there are tens of thousands of them, so on a
        machine with no graphics card this is the slowest thing in the pipeline by a
        wide margin. How long depends on the overlap carried over from the previous
        screen &mdash; roughly ten minutes with none, and about half an hour at a
        half. You can stop it at any point, and the result is saved when it finishes,
        so you only wait once.
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
          {started ? 'Try again' : 'Classify every patch'}
        </Button>
        <Button variant="secondary" onClick={onSkip}>
          Skip for now
        </Button>
      </div>
      <p className="tt-ready__warn">
        Skipping is recorded as a skip rather than a pass, and it leaves the pipeline
        with no class map — so the steps after this one have nothing to work on. It is
        here so you can walk the rest of the pipeline without waiting, not because the
        step is optional the way quality control is.
      </p>
    </div>
  )
}
