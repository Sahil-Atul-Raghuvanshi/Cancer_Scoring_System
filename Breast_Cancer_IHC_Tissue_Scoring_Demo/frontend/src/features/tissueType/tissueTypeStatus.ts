/** Step 8's one line for the page around it: what to say under the button.
 *
 * Split out of `TissueTypePanel` so that file exports only components —
 * react-refresh needs that to hot-reload it, the same reason every other step's
 * status helper is its own file.
 *
 * Leads with the tumour content once there is a report, because that share is what
 * the rest of the pipeline is gated on and it is the one number worth taking away.
 * Before a run it leads with the cost, because this step is slow enough that
 * starting it without knowing that is a bad surprise.
 */

import { formatCount } from '@/lib/format'

import type { TissueTypeState } from './useTissueType'

/** Whether pressing the step's button can do anything useful right now. */
export function canRunTissueType(tissueType: TissueTypeState): boolean {
  return tissueType.capability?.ready === true
}

export function tissueTypeHint(
  tissueType: TissueTypeState,
  hasTiles: boolean,
): string {
  if (tissueType.error) return tissueType.error

  const { report, capability, run } = tissueType

  if (report) {
    const scored = report.classes.find((entry) => entry.scored)
    const inSitu = report.classes.find((entry) => entry.id === 1)
    const licence =
      report.params.licenceTrack === 'permissive'
        ? ''
        : ' These numbers are for evaluation only — the model behind them was trained partly on images licensed for research use.'

    return (
      `${(report.tumourContent * 100).toFixed(1)}% of the tissue is tumour that has broken out ` +
      `(${report.scoredMm2.toFixed(1)} mm², ${formatCount(scored?.windows ?? 0)} patches), and that is the only part the score is measured on. ` +
      `${(inSitu ? inSitu.share * 100 : 0).toFixed(1)}% is tumour still inside a duct and is deliberately left out.${licence}`
    )
  }

  if (tissueType.running) {
    return run && run.total > 0
      ? `${formatCount(run.done)} of ${formatCount(run.total)} patches through the model. Every one is a separate pass, which is why this is the slowest step.`
      : 'Working out which patches to look at.'
  }

  if (tissueType.probing) return 'Checking whether the trained model is installed.'

  if (tissueType.probeError || !capability) {
    return 'The backend is not answering, so there is no way to tell whether the model is installed.'
  }

  if (!capability.ready) return capability.reason

  if (!hasTiles) {
    return 'Run step 7 first: this step needs the list of squares worth looking at, or it would show every patch of empty glass to the model as well.'
  }

  const licence =
    capability.licenceTrack === 'permissive'
      ? ''
      : ' The model that can tell the two kinds of tumour apart is licensed for research only, so its numbers must not be sold.'

  return `This is the one trained model in the pipeline, and the slowest step by a long way — tens of minutes on a machine without a graphics card, because every patch is its own pass through a network. The result is saved, so you pay it once.${licence}`
}
