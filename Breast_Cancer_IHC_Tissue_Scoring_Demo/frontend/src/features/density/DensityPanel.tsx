/**
 * Step 5's work area, choosing between its four states.
 *
 *   blocked   step 4 has not run, so there is no I₀ and therefore no density
 *   running   the first run, which scores every block of the slide and reads a tile
 *   error     it could not run, and why — never a blank panel
 *   report    the result
 *
 * The blocked state is stronger here than on step 4's screen, and worth stating as
 * such. Step 4 without step 3 has nowhere to *look*; step 5 without step 4 has
 * nothing to *compute* — optical density is defined as −log₁₀(I / I₀), so until
 * something says what I₀ is there is no quantity, not merely a worse estimate of
 * one. That is a definition rather than a dependency, and the panel says so instead
 * of showing a disabled button.
 */

import { Button } from '@/components/ui/Button'

import { DensityView } from './DensityView'
import type { OpticalDensityState } from './useOpticalDensity'

import './density.css'

interface DensityPanelProps {
  density: OpticalDensityState
  /** True once step 4 has produced a white point for this slide. */
  hasWhitePoint: boolean
  /** True while the pipeline considers this step to be running. */
  running: boolean
}

export function DensityPanel({ density, hasWhitePoint, running }: DensityPanelProps) {
  const { report, loading, refining, error } = density

  if (report) {
    return (
      <DensityView report={report} refining={refining} onPick={density.pick} />
    )
  }

  if (loading || running) {
    return (
      <div className="placeholder placeholder--running">
        <span className="placeholder__spinner" />
        <p className="placeholder__text">
          Scoring every block of the slide for one that holds two stains, reading it
          off the pyramid at working magnification, and dividing by step 4&rsquo;s
          white point. The first run has to wait for steps 3 and 4 to finish, so it
          is the slow one.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">step 05 could not run</span>
        <p className="placeholder__text">{error}</p>
        <Button variant="secondary" onClick={() => void density.start()}>
          Try again
        </Button>
      </div>
    )
  }

  if (!hasWhitePoint) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">step 04 first</span>
        <p className="placeholder__text">
          Optical density is <em>defined</em> against a reference:{' '}
          <span className="mono">OD = −log₁₀(I / I₀)</span>. Until step 4 has
          measured I₀ from this slide&rsquo;s own glass there is no density to
          compute &mdash; not a rougher one, none. Go back and calibrate, then come
          here.
        </p>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        Nothing is precomputed. Running this step scores every block of the slide
        for one that holds <em>two</em> stains rather than one, reads that tile at
        working magnification, divides it by step 4&rsquo;s white point, and takes a
        logarithm &mdash; then shows you the two arms that appear, how far they land
        from Ruifrok &amp; Johnston&rsquo;s published vectors, and the places where
        the transform stops being a measurement.
      </p>
    </div>
  )
}
