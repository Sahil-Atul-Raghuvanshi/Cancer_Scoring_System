/** Step 3's one question for the page around it: what to say under the button.
 *
 * Split out of `TissuePanel` so that file exports only a component — react-refresh
 * needs that to hot-reload it, the same reason `qcStatus` and
 * `slideSessionContext` are their own files.
 */

import type { TissueMaskState } from './useTissueMask'

export function tissueHint(tissue: TissueMaskState): string {
  if (tissue.error) return tissue.error

  const { report } = tissue
  if (report) {
    const share = `${(report.tissueShare * 100).toFixed(1)}%`
    return report.params.qcGated
      ? `${share} of your slide is tissue, cut at ${report.threshold.value} by the ${report.threshold.source} rule, over the tissue step 2 kept.`
      : `${share} of your slide is tissue. Step 2 has not run, so nothing was subtracted first — see the notes.`
  }

  if (tissue.loading) {
    return 'Downsampling your slide and thresholding its saturation channel — nothing is precomputed.'
  }

  return 'Classical and deterministic: HSV saturation, a threshold picked from the histogram’s own shape, then morphology. No model, no training.'
}
