/**
 * Step 11's work area, choosing between its five states.
 *
 *   blocked   nothing is ticked on step 10, so there is nothing to segment
 *   report    the region list and the comparison — shown *while* the pass runs, not
 *             only after it, because that is the whole promise of the screen
 *   running   the pass has started but no region has landed yet
 *   error     the pass could not start, and why
 *   idle      the offer, with what it will cost
 *
 * The order matters: `report` is tested before `running`, unlike every other gated
 * step. Steps 2 and 8 have nothing to show until they finish; this one has a finished
 * region within a minute or two and a screen that hid it behind a spinner until the
 * last region landed would be hiding the thing it exists to show.
 */

import { Button } from '@/components/ui/Button'

import { RoiRefinementView } from './RoiRefinementView'
import type { RoiRefinementState } from './useRoiRefinement'

import './roiRefinement.css'

interface RoiRefinementPanelProps {
  refinement: RoiRefinementState
  /** How many regions are ticked on step 10. Zero is the blocked state. */
  selectedCount: number
  /** Forward-pass windows the selection costs, for the offer. */
  selectedWindows: number
  /** True while the pipeline considers this step to be running. */
  running: boolean
}

export function RoiRefinementPanel({
  refinement,
  selectedCount,
  selectedWindows,
  running,
}: RoiRefinementPanelProps) {
  const { report, error } = refinement
  const busy = refinement.running || running

  if (report && report.regions.length > 0) {
    return (
      <RoiRefinementView
        report={report}
        run={refinement.run}
        running={busy}
        painted={refinement.painted}
        paintedMasks={refinement.paintedMasks}
        paintedCount={refinement.paintedCount}
        onRetry={(roiId) => void refinement.retry(roiId)}
        onRestart={() => void refinement.restart()}
        onCancel={() => void refinement.cancel()}
      />
    )
  }

  if (busy) {
    return (
      <div className="placeholder placeholder--running">
        <span className="placeholder__spinner" />
        <p className="placeholder__text">
          {refinement.message ??
            'Loading the segmentation model and cutting out the first area. Each area appears here as soon as it is done — you do not have to wait for all of them.'}
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">this step could not run</span>
        <p className="placeholder__text">{error}</p>
        <Button variant="secondary" onClick={() => void refinement.start()}>
          Try again
        </Button>
      </div>
    )
  }

  if (selectedCount === 0) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">pick the regions first</span>
        <p className="placeholder__text">
          Nothing is ticked on the previous step, so there is nothing to trace. Go back
          and tick at least one area.
        </p>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        This will look at {selectedCount} area{selectedCount === 1 ? '' : 's'} pixel by
        pixel — {selectedWindows.toLocaleString()} patches through the model — and
        replace each rough square with the real edge of the tumour inside it. It takes
        minutes, and each area shows up as soon as it is finished. Everything after this
        step measures inside those outlines.
      </p>
    </div>
  )
}
