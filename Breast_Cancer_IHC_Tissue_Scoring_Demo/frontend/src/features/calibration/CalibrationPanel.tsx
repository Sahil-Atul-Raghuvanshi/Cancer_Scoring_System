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
          Reading the slide on step 3&rsquo;s grid, inverting its tissue mask, discarding
          everything on the glass that is not glass, and sampling what is left. The first
          run reopens the scan, so it is the slow one.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">step 04 could not run</span>
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
        <span className="placeholder__badge mono">step 03 first</span>
        <p className="placeholder__text">
          This step measures the part of the slide that is <em>not</em> tissue, so it
          needs step 3&rsquo;s mask before it has anywhere to look. Go back and threshold
          the slide, then come here — and note that step 3&rsquo;s cut is this
          step&rsquo;s only real control: move it and the glass, and I₀ with it, move too.
        </p>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        Nothing is precomputed. Running this step inverts step 3&rsquo;s tissue mask,
        drops the outer frame, step 2&rsquo;s artefacts, the scanner&rsquo;s background
        fill and the halo around the section, then takes a high percentile of each RGB
        channel over the glass that is left — and shows you what every one of those
        exclusions cost.
      </p>
    </div>
  )
}
