/**
 * Step 7's work area, choosing between its states.
 *
 *   blocked   step 6 has not run, so there is nothing to say a square contains
 *   running   the run itself, which builds the grid and prices every square
 *   error     it could not run, and why — never a blank panel
 *   report    the result
 *
 * **This screen used to ask two questions and now answers them.** Which branch, and at
 * what field of view, were real choices while eight heads were in play; the pipeline
 * has since committed to the H&E branch at 224 µm — see `COMMITTED_BRANCH` in
 * `useTiling` for the measurements behind both halves of that. With nothing left to
 * ask, the step auto-starts like every other cheap step: the gate existed to stop a
 * grid being laid that nobody had chosen, and there is no longer an unchosen grid.
 *
 * `BranchPicker` and `FieldOfViewPicker` are still in this folder and are no longer
 * rendered. They were kept rather than deleted because re-offering the choice is a
 * decision someone may make again, and putting a file back is cheaper than writing it.
 *
 * The blocked state is the odd one on this screen and worth being honest about. The
 * *grid* is pure geometry and could be laid out with nothing but the slide's dimensions
 * — but the index this step hands on is a list of addresses whose pixels are whatever
 * the chosen option shows the model, so building it before anything has defined what a
 * square's pixels are would produce an index of nothing. Requiring step 6 is what keeps
 * the fork honest rather than what makes the arithmetic possible.
 */

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

  if (!hasChannels) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">separate the stains first</span>
        <p className="placeholder__text">
          This step records where each square is. A square is only useful once we know
          what picture it shows the model, and that is decided when the stains are
          separated. Run the previous step, then come back.
        </p>
      </div>
    )
  }

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
          Laying a grid of squares over the scan and checking each one against the
          tissue map and the damaged areas, to see which are worth processing.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">this step could not run</span>
        <p className="placeholder__text">{error}</p>
      </div>
    )
  }

  // Still reading which heads exist. Nothing is being asked here any more - the
  // configuration is committed - but the run needs the manifest scan to have landed.
  return (
    <div className="placeholder placeholder--running">
      <span className="placeholder__spinner" />
      <p className="placeholder__text">
        Loading the trained model before laying its grid over the slide.
      </p>
    </div>
  )
}
