/**
 * Step 10's work area, choosing between its four states.
 *
 *   blocked   step 9 has not built a region, so there are no candidates to offer
 *   running   the first build, which crops a card from the slide per candidate
 *   error     it could not list them, and why — never a blank panel
 *   report    the grid, and the controls that decide what goes to step 11
 *
 * No capability check, like step 9 and unlike step 8: this step runs no model, so the
 * only thing it can be missing is its input.
 */

import { Button } from '@/components/ui/Button'

import { RoiSelectionView } from './RoiSelectionView'
import type { RoiSelectionState } from './useRoiSelection'

import './roiSelection.css'

interface RoiSelectionPanelProps {
  selection: RoiSelectionState
  /** True once step 9 has built a region on this slide. */
  hasRoi: boolean
  /** True while the pipeline considers this step to be running. */
  running: boolean
}

export function RoiSelectionPanel({ selection, hasRoi, running }: RoiSelectionPanelProps) {
  const { report, loading, saving, error } = selection

  if (report) {
    return (
      <RoiSelectionView
        report={report}
        selected={selection.selected}
        saving={saving}
        onToggle={(roiId) => void selection.toggle(roiId)}
        onSelectAll={() => void selection.selectAll()}
        onDeselectAll={() => void selection.deselectAll()}
        onResetToDefault={() => void selection.resetToDefault()}
      />
    )
  }

  if (loading || running) {
    return (
      <div className="placeholder placeholder--running">
        <span className="placeholder__spinner" />
        <p className="placeholder__text">
          Collecting every area marked as tumour and cutting a picture of each one out of
          the slide, so you can see what you are choosing between.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">this step could not run</span>
        <p className="placeholder__text">{error}</p>
        <Button variant="secondary" onClick={() => void selection.start()}>
          Try again
        </Button>
      </div>
    )
  }

  if (!hasRoi) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">outline the tumour first</span>
        <p className="placeholder__text">
          This step lists the tumour areas so you can pick which ones deserve a closer
          look. Without them there is nothing to list. Go back a step, then come here.
        </p>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        This step ranks every area marked as tumour, cuts a picture of each out of the
        slide, and ticks enough of the largest to cover almost all the tumour. You can
        change what is ticked.
      </p>
    </div>
  )
}
