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
}

export function DeconvolutionPanel({
  deconvolution,
  hasDensity,
  running,
}: DeconvolutionPanelProps) {
  const { report, loading, error } = deconvolution

  if (report) {
    return (
      <DeconvolutionView
        report={report}
        basis={deconvolution.basis}
        onBasisChange={deconvolution.setBasis}
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
        <span className="placeholder__badge mono">step 06 could not run</span>
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
        <span className="placeholder__badge mono">step 05 first</span>
        <p className="placeholder__text">
          Stains can only be separated once the image has been converted to
          &ldquo;how much stain&rdquo; &mdash; that is what step 5 does. On the raw
          colour photo the two dyes <em>multiply</em> together rather than add up, and
          no amount of arithmetic can pull apart a product. Run step 5, then come
          back.
        </p>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        Nothing is precomputed. Running this step takes step 5&rsquo;s tile and splits
        it into a blue picture and a brown picture &mdash; then shows you what happens
        to the score if the stain colours are guessed from the slide instead of taken
        from the standard reference.
      </p>
    </div>
  )
}
