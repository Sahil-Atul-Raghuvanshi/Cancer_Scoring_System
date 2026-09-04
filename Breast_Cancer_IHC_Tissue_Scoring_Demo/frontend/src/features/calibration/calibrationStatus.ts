/** Step 4's one question for the page around it: what to say under the button.
 *
 * Split out of `CalibrationPanel` so that file exports only a component —
 * react-refresh needs that to hot-reload it, the same reason `tissueStatus` and
 * `qcStatus` are their own files.
 */

import type { WhiteCalibrationState } from './useWhiteCalibration'

export function calibrationHint(
  calibration: WhiteCalibrationState,
  hasTissueMask: boolean,
): string {
  if (calibration.error) return calibration.error

  const { report } = calibration
  if (report) {
    const { white, noise, choice } = report
    const i0 = `(${white.rgb.r.toFixed(0)}, ${white.rgb.g.toFixed(0)}, ${white.rgb.b.toFixed(0)})`
    const field = choice.mode === 'surface' ? 'varying across the slide' : 'flat'

    return `I₀ is ${i0} on this slide, ${field}, and empty glass measures ${noise.worst.toFixed(3)} OD under it — the floor every threshold downstream has to clear.`
  }

  if (calibration.loading) {
    return 'Inverting step 3’s mask and sampling the glass it left behind — nothing is precomputed.'
  }

  if (!hasTissueMask) {
    return 'Run step 3 first: this step measures the part of the slide that is not tissue, so it needs the mask before it has anywhere to look.'
  }

  return 'Classical and deterministic: five exclusions to decide what is really glass, then a high percentile of each channel over what survives. No model, no training.'
}
