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
    const field = choice.mode === 'surface' ? 'and it varies across the slide' : 'and it is even across the slide'

    return `Blank glass on this slide is ${i0} ${field}. Anything reading below ${noise.worst.toFixed(3)} is noise rather than stain.`
  }

  if (calibration.loading) {
    return 'Measuring the blank glass on your slide.'
  }

  if (!hasTissueMask) {
    return 'Find the tissue first. This step measures the part of the slide that is not tissue.'
  }

  return 'Works out what blank glass looks like on this slide, so stain can be measured against it.'
}
