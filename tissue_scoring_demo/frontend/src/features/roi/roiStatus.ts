/** Step 9's one question for the page around it: what to say under the button.
 *
 * Split out of `RoiPanel` so that file exports only a component — react-refresh
 * needs that to hot-reload it, the same reason `tissueStatus` and `densityStatus`
 * are their own files.
 */

import type { RoiState } from './useRoi'

export function roiHint(roi: RoiState, hasClassMap: boolean): string {
  if (roi.error) return roi.error

  const { report } = roi
  if (report) {
    const dcis = report.classAreaMm2.non_invasive_epithelium ?? 0
    const invasive = report.classAreaMm2.invasive_epithelium ?? 0
    return (
      `${report.areaMm2.toFixed(1)} mm² is the region every later step measures inside. ` +
      `Separately, ${invasive.toFixed(1)} mm² of invasive tumour and ${dcis.toFixed(1)} mm² ` +
      `of tumour inside a duct is outlined above, largest first.`
    )
  }

  if (roi.loading) {
    return 'Joining the labelled patches into clean regions and drawing a border around each one.'
  }

  if (!hasClassMap) {
    return 'Label the tissue first. This step joins those labels into regions, and there is nothing to join without them.'
  }

  return 'Joins the labelled patches into regions. No model of its own — just smoothing, closing gaps and dropping specks.'
}
