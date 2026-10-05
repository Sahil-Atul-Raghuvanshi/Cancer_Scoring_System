/**
 * Step 4's work area, choosing between its four states.
 *
 *   blocked   step 3 has not run, so there is no mask and therefore no glass
 *   running   the first run, which reopens the slide
 *   error     it could not run, and why — never a blank panel
 *   report    the result
 *
 * The blocked state is the one step 3's panel does not have, and it is not
 * bookkeeping: step 4's whole method is "measure the part of the slide that is not
 * tissue", so without a tissue mask there is no such part. Saying that plainly is
 * better than a disabled button with no explanation, and better still than
 * silently running step 3 from under the viewer — the mask is a decision they were
 * shown and allowed to change.
 */

import { Button } from '@/components/ui/Button'

import { CalibrationView } from './CalibrationView'
import type { WhiteCalibrationState } from './useWhiteCalibration'

import './calibration.css'

interface CalibrationPanelProps {
  calibration: WhiteCalibrationState
  /** True once step 3 has produced a mask for this slide. */
  hasTissueMask: boolean
  /** True while the pipeline considers this step to be running. */
  running: boolean
}

export function CalibrationPanel({
  calibration,
  hasTissueMask,
  running,
}: CalibrationPanelProps) {
  const { report, loading, refining, error, draft, committed, defaultPercentile, manual } =
    calibration

  if (report && draft !== null && committed !== null) {
    return (
      <CalibrationView
        report={report}
        draft={draft}
        committed={committed}
        defaultPercentile={defaultPercentile ?? committed}
        manual={manual}
        refining={refining}
        onDraft={calibration.setDraft}
        onResetToDefault={calibration.resetToDefault}
      />
    )
  }

  if (loading || running) {
    return (
      <div className="placeholder placeholder--running">
        <span className="placeholder__spinner" />
        <p className="placeholder__text">
          Finding the blank glass on your slide and measuring its colour. The first run
          reopens the scan, so it is the slowest one.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">this step could not run</span>
        <p className="placeholder__text">{error}</p>
        <Button variant="secondary" onClick={() => void calibration.start()}>
          Try again
        </Button>
      </div>
    )
  }

  if (!hasTissueMask) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">find the tissue first</span>
        <p className="placeholder__text">
          This step measures the part of the slide that is <em>not</em> tissue, so it needs
          the tissue map first. Go back to &ldquo;Find the tissue&rdquo;, then come here.
          Changing the cut there also changes the result here.
        </p>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        This step looks at everything that is not tissue, removes the parts that are not
        really blank glass &mdash; the slide edge, pen marks, the scanner background &mdash;
        and measures the colour of what is left.
      </p>
    </div>
  )
}
