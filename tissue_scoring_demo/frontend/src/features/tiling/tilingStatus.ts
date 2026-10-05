/** Step 7's one line for the page around it: what to say under the button.
 *
 * Split out of `TilingPanel` so that file exports only a component —
 * react-refresh needs that to hot-reload it, the same reason every other step's
 * status helper is its own file.
 *
 * Leads with the funnel, because the drop is the thing worth taking away.
 */

import { formatCount } from '@/lib/format'

import type { TilingState } from './useTiling'

export function tilingHint(tiling: TilingState, hasChannels: boolean): string {
  if (tiling.error) return tiling.error

  const { report } = tiling
  if (report) {
    const { funnel, params } = report
    const gated = params.qcGated
      ? ` ${formatCount(funnel.onTissue - funnel.clean)} more were dropped for being folded, blurred or marked.`
      : ' The quality check was skipped, so none were dropped for damage.'

    return `${formatCount(funnel.every)} squares in the grid, ${formatCount(funnel.clean)} worth processing — ${funnel.reduction.toFixed(0)} times less work for every step after this one.${gated}`
  }

  if (tiling.loading) {
    return 'Laying a grid over the scan and checking every square against the tissue map and the damaged areas.'
  }

  if (!hasChannels) {
    return 'Separate the stains first. That step decides what picture each square shows the model.'
  }

  return 'Cuts the tissue into squares. Most of the slide is empty glass, so most squares can be skipped.'
}
