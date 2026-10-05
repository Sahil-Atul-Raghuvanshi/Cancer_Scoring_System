/**
 * Step 6's work area, choosing between its four states.
 *
 *   blocked   step 5 has not run, so there is no optical density to un-mix
 *   running   the run itself
 *   error     it could not run, and why — never a blank panel
 *   report    the result
 *
 * The blocked state is worth stating rather than showing as a disabled button.
 * Separating stains is a *linear* operation, and stains only add up linearly in
 * optical density; run on the raw colour photo the same arithmetic returns three
 * confident numbers that are not amounts of anything. So step 5 is not a source of
 * better input here, it is the condition under which this step means anything.
 */

import { Button } from '@/components/ui/Button'

import { DeconvolutionView } from './DeconvolutionView'
import type { DeconvolutionState } from './useDeconvolution'

import './deconvolution.css'

interface DeconvolutionPanelProps {
  deconvolution: DeconvolutionState
  /** True once step 5 has produced an optical density tile for this slide. */
  hasDensity: boolean
  /** True while the pipeline considers this step to be running. */
  running: boolean
  /**
   * Which slide these pictures came from.
   *
   * It decides the wording, not the rendering, and the wording is the whole point:
   * panel 3 is the DAB channel, and on an H&E section there is no DAB, so calling it
   * "the picture the score is measured from" there is false. The score is measured
   * on the immunostained slide. Same arithmetic, two entirely different claims.
   */
  slideRole: 'he' | 'ihc' | 'unknown'
}

export function DeconvolutionPanel({
  deconvolution,
  hasDensity,
  running,
  slideRole,
}: DeconvolutionPanelProps) {
  const { report, loading, error } = deconvolution

  if (report) {
    return (
      <DeconvolutionView
        report={report}
        basis={deconvolution.basis}
        onBasisChange={deconvolution.setBasis}
        slideRole={slideRole}
      />
    )
  }

  if (loading || running) {
    return (
      <div className="placeholder placeholder--running">
        <span className="placeholder__spinner" />
        <p className="placeholder__text">
          Splitting the blue stain from the brown one, twice over &mdash; once with
          the standard published colours, and once with colours worked out from this
          tile, so the two can be compared.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">this step could not run</span>
        <p className="placeholder__text">{error}</p>
        <Button variant="secondary" onClick={() => void deconvolution.start()}>
          Try again
        </Button>
      </div>
    )
  }

  if (!hasDensity) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">measure stain amount first</span>
        <p className="placeholder__text">
          Stains can only be separated once colour has been turned into stain amount.
          On the raw colour photo the two dyes multiply together rather than add up, and
          a product cannot be split. Run the previous step, then come back.
        </p>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        This step splits the tile into a blue picture and a brown picture. It also shows
        what happens to the result if the stain colours are worked out from the slide
        instead of taken from the standard reference.
      </p>
    </div>
  )
}
