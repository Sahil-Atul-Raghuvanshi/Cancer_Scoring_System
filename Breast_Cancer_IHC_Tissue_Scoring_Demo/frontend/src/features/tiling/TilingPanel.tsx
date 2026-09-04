/**
 * Step 7's work area, choosing between its four states.
 *
 *   blocked   step 6 has not run, so there is nothing to say a square contains
 *   running   the first run, which builds the grid and scores every square
 *   error     it could not run, and why — never a blank panel
 *   report    the result
 *
 * The blocked state is the odd one on this screen and worth being honest about.
 * The *grid* is pure geometry and could be laid out with nothing but the slide's
 * dimensions — but the index this step hands on is a list of addresses whose
 * pixels are the blue-stain picture, so building it before anything has defined
 * what a square's pixels are would produce an index of nothing. Requiring step 6
 * is what keeps the fork honest rather than what makes the arithmetic possible.
 */

import { Button } from '@/components/ui/Button'

import { TilingView } from './TilingView'
import type { TilingState } from './useTiling'

import './tiling.css'

interface TilingPanelProps {
  tiling: TilingState
  /** Tile counts already measured at each overlap, for the picker's price tags. */
  known: Map<number, number>
  /** True once step 6 has separated the stains for this slide. */
  hasChannels: boolean
  /** True while the pipeline considers this step to be running. */
  running: boolean
}

export function TilingPanel({
  tiling,
  known,
  hasChannels,
  running,
}: TilingPanelProps) {
  const { report, loading, refining, error } = tiling

  if (report) {
    return (
      <TilingView
        report={report}
        refining={refining}
        known={known}
        onOverlap={tiling.setOverlap}
      />
    )
  }

  if (loading || running) {
    return (
      <div className="placeholder placeholder--running">
        <span className="placeholder__spinner" />
        <p className="placeholder__text">
          Laying a grid of squares over the whole scan and checking each one
          against step 3&rsquo;s tissue map and step 2&rsquo;s list of damaged
          areas, to work out which ones are worth processing.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">step 07 could not run</span>
        <p className="placeholder__text">{error}</p>
        <Button variant="secondary" onClick={() => void tiling.start()}>
          Try again
        </Button>
      </div>
    )
  }

  if (!hasChannels) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">step 06 first</span>
        <p className="placeholder__text">
          This step writes down where each square is, but a square is only useful
          once something has decided <em>what its picture is</em> &mdash; and that
          is the blue-stain picture step 6 produces. Separate the stains first,
          then come back.
        </p>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        Nothing is precomputed. Running this step lays a grid over the scan, checks
        every square against what steps 2 and 3 found, and shows you how many are
        left &mdash; which is the amount of work every step after this one has to
        do.
      </p>
    </div>
  )
}
