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
          Looking for a tile that holds both stains, then converting its colours into
          stain amounts. The first run waits for the earlier steps to finish, so it is
          the slowest one.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">this step could not run</span>
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
        <span className="placeholder__badge mono">measure the blank glass first</span>
        <p className="placeholder__text">
          Stain amount is measured by comparing each pixel with blank glass. Until the
          blank glass on this slide has been measured there is nothing to compare
          against. Go back a step, then come here.
        </p>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        This step finds a tile that holds both stains and converts its colours into
        stain amounts. It then shows the two stain colours it found and how close they
        are to the expected ones.
      </p>
    </div>
  )
}
