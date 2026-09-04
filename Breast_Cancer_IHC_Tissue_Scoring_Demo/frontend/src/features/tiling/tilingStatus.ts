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
      : ' Step 2 has not run, so none were dropped for damage — run quality control and come back.'

    return `${formatCount(funnel.every)} squares in the grid, ${formatCount(funnel.clean)} worth processing — ${funnel.reduction.toFixed(0)} times less work for every step after this one.${gated}`
  }

  if (tiling.loading) {
    return 'Laying a grid over the scan and checking every square against the tissue map and the list of damaged areas.'
  }

  if (!hasChannels) {
    return 'Run step 6 first: a square is only useful once something has decided what its picture is, and that is the blue-stain picture step 6 produces.'
  }

  return 'Plumbing, not science — but it runs here for a reason: because steps 2 and 3 already narrowed the slide down, most squares never have to be processed at all.'
}
