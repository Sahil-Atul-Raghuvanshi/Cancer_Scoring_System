/** Step 10's one question for the page around it: what to say under the button.
 *
 * Split out of `RoiSelectionPanel` so that file exports only a component — react-refresh
 * needs that to hot-reload it, the same reason `roiStatus` and `tissueStatus` are their
 * own files.
 */

import type { RoiSelectionState } from './useRoiSelection'

export function roiSelectionHint(selection: RoiSelectionState, hasRoi: boolean): string {
  if (selection.error) return selection.error

  const { report } = selection
  if (report) {
    if (report.candidates.length === 0) {
      return 'No tumour area on this slide is large enough to outline, so there is nothing to choose.'
    }
    const share =
      report.invasiveMm2 > 0 ? (report.selectedMm2 / report.invasiveMm2) * 100 : 0
    return (
      `${selection.selected.length} of ${report.candidates.length} areas chosen — ` +
      `${report.selectedMm2.toFixed(1)} mm², about ${share.toFixed(0)}% of the tumour on this slide. ` +
      `Only the ticked areas go to the next step.`
    )
  }

  if (selection.loading) {
    return 'Cutting a picture of each tumour area out of the slide so you can see what you are choosing between.'
  }

  if (!hasRoi) {
    return 'Outline the tumour areas first. This step lists those areas, and there is nothing to list without them.'
  }

  return 'No model here. This step ranks the areas already found, shows each one, and records which you want outlined properly.'
}
