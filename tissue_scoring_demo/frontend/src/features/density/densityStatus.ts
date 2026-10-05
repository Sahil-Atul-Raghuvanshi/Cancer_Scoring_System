/** Step 5's one question for the page around it: what to say under the button.
 *
 * Split out of `DensityPanel` so that file exports only a component —
 * react-refresh needs that to hot-reload it, the same reason `calibrationStatus`,
 * `tissueStatus` and `qcStatus` are their own files.
 */

import type { OpticalDensityState } from './useOpticalDensity'

export function densityHint(
  density: OpticalDensityState,
  hasWhitePoint: boolean,
): string {
  if (density.error) return density.error

  const { report } = density
  if (report) {
    const { cloud, stats } = report

    if (!cloud) {
      return `This tile converted cleanly — typical stain ${stats.meanMedian.toFixed(3)} — but too few pixels carry enough stain to identify the dyes. Try another tile.`
    }

    const arms = cloud.arms
      .map((arm) => (arm.nearest === 'dab' ? 'the brown marker' : arm.nearest))
      .join(' and ')

    return `Two stain colours found, ${cloud.separation.toFixed(0)}° apart, closest to ${arms}. The software was not told what either stain looks like.`
  }

  if (density.loading) {
    return 'Finding a tile that holds both stains and converting its colours into stain amounts.'
  }

  if (!hasWhitePoint) {
    return 'Measure the blank glass first. Stain amount is measured against it, so there is nothing to compare with until then.'
  }

  return 'Converts colour into stain amount, so that two stains add up instead of multiplying. That is what makes the next step possible.'
}
