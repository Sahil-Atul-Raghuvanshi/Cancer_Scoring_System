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
      ? `${share} of your slide is tissue. Problem areas found in the quality check were removed first.`
      : `${share} of your slide is tissue. The quality check was skipped, so nothing was removed first.`
  }

  if (tissue.loading) {
    return 'Working out where the tissue is on your slide.'
  }

  return 'Uses how much colour each pixel has to tell tissue from empty glass. No model, no training.'
}
