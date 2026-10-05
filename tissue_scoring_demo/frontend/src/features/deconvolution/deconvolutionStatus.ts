/** Step 6's one line for the page around it: what to say under the button.
 *
 * Split out of `DeconvolutionPanel` so that file exports only a component —
 * react-refresh needs that to hot-reload it, the same reason `densityStatus`,
 * `calibrationStatus`, `tissueStatus` and `qcStatus` are their own files.
 *
 * Plain language, like the rest of this step's screen, and it leads with the
 * comparison because that is the thing worth taking away.
 */

import type { DeconvolutionState } from './useDeconvolution'

export function deconvolutionHint(
  deconvolution: DeconvolutionState,
  hasDensity: boolean,
): string {
  if (deconvolution.error) return deconvolution.error

  const { report } = deconvolution
  if (report) {
    const fixed = report.fixed.preview.positiveShare * 100
    const leftover = report.fixed.residualShare * 100

    if (!report.comparison) {
      return `The stains are separated: ${fixed.toFixed(0)}% of the stained tissue counts as positive, with ${leftover.toFixed(0)}% of the colour left unexplained. This tile did not have enough stain to work out slide-specific colours to compare against.`
    }

    const shift = Math.abs(report.comparison.shareShift) * 100
    return `The stains are separated. With the standard colours ${fixed.toFixed(0)}% of the stained tissue counts as positive; with colours guessed from this slide it would be ${(report.comparison.estimatedPositiveShare * 100).toFixed(0)}% — a ${shift.toFixed(1)} point difference from a choice that has nothing to do with the patient.`
  }

  if (deconvolution.loading) {
    return 'Splitting the blue stain from the brown one, twice over, so the two ways of defining the colours can be compared.'
  }

  if (!hasDensity) {
    return 'Measure stain amount first. Stains can only be split apart once colour has been turned into stain amount.'
  }

  return 'Splits the picture into two: the blue picture goes to the model, the brown picture goes to the measurement.'
}
