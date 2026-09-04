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
    const { cloud, additivity, stats } = report

    if (!cloud) {
      return `The tile transformed cleanly — median ${stats.meanMedian.toFixed(3)} OD — but too few of its pixels carry enough stain for a point cloud, so there are no arms to show. Try another tile.`
    }

    const arms = cloud.arms
      .map((arm) => (arm.nearest === 'dab' ? 'DAB' : arm.nearest))
      .join(' and ')
    const proof = additivity
      ? ` Density space held its direction to ${additivity.odDrift.toFixed(2)}° across the concentration range where intensity space turned ${additivity.intensityDrift.toFixed(2)}°.`
      : ''

    return `Two arms, ${cloud.separation.toFixed(0)}° apart, landing nearest ${arms} — nothing here was told what either stain looks like.${proof}`
  }

  if (density.loading) {
    return 'Scoring every block of the slide for one that holds two stains, reading it off the pyramid, and dividing by step 4’s white point.'
  }

  if (!hasWhitePoint) {
    return 'Run step 4 first: optical density is defined as −log₁₀(I / I₀), so until something says what I₀ is there is no density to compute.'
  }

  return 'Classical and deterministic: one logarithm per channel. The reason it matters is that it turns a product of stains into a sum, which is the only form linear algebra can un-mix.'
}
