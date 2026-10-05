/** Step 2's two questions for the page around it: can it run, and what to say.
 *
 * Split out of `QCPanel` so that file exports only a component — react-refresh
 * needs that to hot-reload it, the same reason `slideSessionContext` is its own
 * file.
 */

import type { QCReport, QCRun } from '@/types/qc'

import type { QualityControl } from './useQualityControl'

/** Whether the step's run button should be offered at all. */
export function canRunQC(qc: QualityControl): boolean {
  return Boolean(qc.capability && qc.capability.mode !== 'unavailable')
}

/** One line for the hint under the run button. */
export function qcHint(qc: QualityControl, run: QCRun | null, report: QCReport | null): string {
  if (qc.error) return qc.error

  if (report) {
    return 'Every number above was measured from your slide. Check the map, then continue.'
  }
  if (run?.state === 'running') {
    return 'Checking the whole slide for problem areas. This takes a few minutes.'
  }

  switch (qc.capability?.mode) {
    case 'unavailable':
      return 'This step needs extra files. See the checklist above.'
    case 'degraded':
      return 'Problem areas will be found, but tissue is detected with a simpler rule.'
    default:
      return 'Finds the tissue, then marks blurry, folded and pen-marked areas.'
  }
}
