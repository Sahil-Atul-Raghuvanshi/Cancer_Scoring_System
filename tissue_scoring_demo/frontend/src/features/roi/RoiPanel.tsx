/**
 * Step 9's work area, choosing between its four states.
 *
 *   blocked   step 8 has not run, so there is no class map to turn into a region
 *   running   the first build, which draws the borders, the crops and the export
 *   error     it could not build, and why — never a blank panel
 *   report    the region, its borders, and the top-3 crops
 *
 * There is no capability check here, unlike step 8: this step runs no model of its
 * own, so the only thing it can be missing is step 8's class map, and that is exactly
 * the blocked state.
 */

import { Button } from '@/components/ui/Button'

import { RoiView } from './RoiView'
import type { RoiState } from './useRoi'

import './roi.css'

interface RoiPanelProps {
  roi: RoiState
  /** True once step 8 has produced a class map for this slide. */
  hasClassMap: boolean
  /** True while the pipeline considers this step to be running. */
  running: boolean
}

export function RoiPanel({ roi, hasClassMap, running }: RoiPanelProps) {
  const { report, loading, error } = roi

  if (report) {
    return <RoiView report={report} onRebuild={roi.restart} rebuilding={loading} />
  }

  if (loading || running) {
    return (
      <div className="placeholder placeholder--running">
        <span className="placeholder__spinner" />
        <p className="placeholder__text">
          Joining the labelled patches into one scoring region, and drawing a border
          around each separate tumour region. The calculation is fast; most of the wait
          is drawing the pictures.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">this step could not run</span>
        <p className="placeholder__text">{error}</p>
        <Button variant="secondary" onClick={() => void roi.start()}>
          Try again
        </Button>
      </div>
    )
  }

  if (!hasClassMap) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">label the tissue first</span>
        <p className="placeholder__text">
          This step joins the labelled patches into regions. Without those labels there
          is nothing to join. Go back and label the tissue, then come here.
        </p>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        This step joins the labelled patches into one smooth scoring region, and also
        draws a border around each separate tumour region. It then ranks them by size,
        crops the three largest of each kind from the slide, and offers them all as a
        QuPath file.
      </p>
    </div>
  )
}
