/** Step 11's one question for the page around it: what to say under the button.
 *
 * Its own file for `roiStatus`'s reason — react-refresh needs the panel module to
 * export only a component.
 */

import type { RoiRefinementState } from './useRoiRefinement'

export function roiRefinementHint(
  refinement: RoiRefinementState,
  selectedCount: number,
): string {
  if (refinement.error) return refinement.error

  const { report, run } = refinement

  if (refinement.running) {
    if (run?.currentRoiId) {
      return `Tracing ${run.currentRoiId} — ${run.done} of ${run.total} areas done. Each one appears as soon as it finishes.`
    }
    return refinement.message ?? 'Loading the model and starting on the first area.'
  }

  if (report && report.regions.length > 0) {
    const kept = (report.keptShare * 100).toFixed(0)
    if (report.failed > 0) {
      return (
        `${report.completed} of ${report.selected} areas traced; ${report.failed} failed and can be ` +
        `retried on their own. Everything after this step measures inside the outlines that worked.`
      )
    }
    return (
      `${report.refinedMm2.toFixed(1)} mm² of actual tumour, traced inside ${report.tileMm2.toFixed(1)} mm² of ` +
      `rough squares — ${kept}% kept. This outline, not the squares, is what every later step measures inside.`
    )
  }

  if (selectedCount === 0) {
    return 'Go back and tick at least one area. This step only runs on the areas you choose.'
  }

  return `Runs a detailed model on the ${selectedCount} chosen area${selectedCount === 1 ? '' : 's'} rather than the whole slide, replacing each rough square with the real edge of the tumour.`
}
