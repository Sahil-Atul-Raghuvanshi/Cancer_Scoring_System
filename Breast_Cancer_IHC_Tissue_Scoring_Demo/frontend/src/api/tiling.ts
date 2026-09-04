/**
 * Client for step 7 — tiling.
 *
 * One request, no polling. The index is one pass over the grid against a mask
 * step 3 has already cached.
 *
 * `threshold` belongs to step 3 and is passed straight through: it decides what
 * is tissue, and what is tissue decides which tiles survive.
 *
 * `overlap` is step 7's own control and the only one. The tile size and the
 * working resolution are step 1's — the working magnification is one decision for
 * the whole pipeline — and the two gates are thresholds whose values are arguments
 * rather than preferences. Overlap is a genuine trade: seam quality against
 * compute, and moving it should visibly move the tile count.
 */

import { API_PREFIX, apiGet } from './client'

import type { TilingPanelName, TilingReport } from '@/types/tiling'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

export interface TilingOptions {
  /** Step 3's saturation cut. Omit to use whichever rule its histogram selects. */
  threshold?: number | null
  /** Fraction of its extent a tile shares with its neighbour. Omit for the default. */
  overlap?: number | null
}

function query(options: TilingOptions): string {
  const params = new URLSearchParams()
  if (options.threshold !== null && options.threshold !== undefined) {
    params.set('threshold', String(Math.round(options.threshold)))
  }
  if (options.overlap !== null && options.overlap !== undefined) {
    params.set('overlap', String(options.overlap))
  }

  const encoded = params.toString()
  return encoded ? `?${encoded}` : ''
}

/**
 * Build the tile index and read the result.
 *
 * A generous ceiling: on a cold cache this waits for step 3's mask behind it,
 * which may reopen a multi-gigabyte scan.
 */
export function fetchTiling(
  uploadId: string,
  options: TilingOptions = {},
  signal?: AbortSignal,
): Promise<TilingReport> {
  return apiGet<TilingReport>(`/tiling/${uploadId}${query(options)}`, {
    signal,
    timeoutMs: 120_000,
  })
}

/**
 * URL of one panel.
 *
 * Both carry the options, because both depend on the mask the index was built
 * against and on the overlap that set the grid. Each distinct choice is a
 * distinct URL, so the browser's cache works for free when the viewer moves the
 * overlap back to a value they have already seen.
 */
export function tilingPanelUrl(
  uploadId: string,
  name: TilingPanelName,
  options: TilingOptions = {},
): string {
  return url(`/tiling/${uploadId}/panels/${name}.png${query(options)}`)
}
