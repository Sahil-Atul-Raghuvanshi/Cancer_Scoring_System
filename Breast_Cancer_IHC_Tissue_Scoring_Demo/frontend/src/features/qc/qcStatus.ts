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
    return `Every figure above was measured from your slide by GrandQC's ${report.params.modelLabel} model.`
  }
  if (run?.state === 'running') {
    return 'Running two segmentation models over the whole slide — this is not precomputed.'
  }

  switch (qc.capability?.mode) {
    case 'unavailable':
      return 'Install the QC dependencies and download the checkpoints to run this step.'
    case 'degraded':
      return 'Runs with GrandQC artefact segmentation; tissue falls back to a classical threshold.'
    default:
      return 'Runs GrandQC tissue detection, then artefact segmentation, then the classical metrics.'
  }
}
