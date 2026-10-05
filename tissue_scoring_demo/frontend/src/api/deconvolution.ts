/**
 * Client for step 6 — colour deconvolution.
 *
 * One request, no polling. The step's own work is two 3×3 inverses over one tile,
 * and its input is step 5's already-memoised density array, so there is nothing
 * to wait on beyond the first run of step 5 itself.
 *
 * Every option here belongs to an earlier step and is passed straight through:
 * `threshold` is step 3's, `percentile` step 4's, `x` and `y` step 5's. Step 6
 * owns none of them, and it has no option of its own for the one thing it decides
 * — which stain vectors to project onto — because making that a request parameter
 * would make the DAB scale the caller's choice.
 *
 * `basis` is the one exception and it is only on the panels. The report always
 * carries both bases, so the two scores reach the screen together.
 */

import { API_PREFIX, apiGet } from './client'

import type {
  DeconvolutionBasisName,
  DeconvolutionPanelName,
  DeconvolutionReport,
} from '@/types/deconvolution'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

export interface DeconvolutionOptions {
  /** Step 3's saturation cut. Omit to use whichever rule its histogram selects. */
  threshold?: number | null
  /** Which percentile of the glass step 4 takes as I₀. Omit for its default. */
  percentile?: number | null
  /** Level-0 origin of step 5's tile, so both steps stand on the same pixels. */
  x?: number | null
  y?: number | null
}

function query(options: DeconvolutionOptions, basis?: DeconvolutionBasisName): string {
  const params = new URLSearchParams()
  if (options.threshold !== null && options.threshold !== undefined) {
    params.set('threshold', String(Math.round(options.threshold)))
  }
  if (options.percentile !== null && options.percentile !== undefined) {
    params.set('percentile', String(options.percentile))
  }
  // Both or neither: a position with one coordinate missing is not a position,
  // and the server would silently read the other as zero.
  if (
    options.x !== null &&
    options.x !== undefined &&
    options.y !== null &&
    options.y !== undefined
  ) {
    params.set('x', String(Math.max(0, Math.round(options.x))))
    params.set('y', String(Math.max(0, Math.round(options.y))))
  }
  if (basis) params.set('basis', basis)

  const encoded = params.toString()
  return encoded ? `?${encoded}` : ''
}

/**
 * Run step 6 and read the result — both bases in one response.
 *
 * A generous ceiling: on a cold cache this call has to wait for steps 3, 4 and 5
 * behind it, and the first of those may reopen a multi-gigabyte scan.
 */
export function fetchDeconvolution(
  uploadId: string,
  options: DeconvolutionOptions = {},
  signal?: AbortSignal,
): Promise<DeconvolutionReport> {
  return apiGet<DeconvolutionReport>(`/deconvolution/${uploadId}${query(options)}`, {
    signal,
    timeoutMs: 120_000,
  })
}

/**
 * URL of one panel of one basis.
 *
 * Each distinct combination is a distinct URL, so the browser's cache does the
 * work when the viewer flips the basis toggle back and forth — which is exactly
 * the interaction this step is built around, so it is worth it being instant.
 */
export function deconvolutionPanelUrl(
  uploadId: string,
  name: DeconvolutionPanelName,
  basis: DeconvolutionBasisName,
  options: DeconvolutionOptions = {},
): string {
  return url(`/deconvolution/${uploadId}/panels/${name}.png${query(options, basis)}`)
}
