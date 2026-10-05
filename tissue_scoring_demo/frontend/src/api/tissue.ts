/**
 * Client for step 3 - the tissue mask.
 *
 * One request, no polling. A cold run reopens the slide and takes a few seconds;
 * a re-threshold reuses the cached saturation channel and takes about two, which
 * is the morphology itself rather than overhead.
 *
 * `threshold` is what makes the slider work. Passing one re-runs the real
 * thresholding and the real morphology on the server - the same code path Otsu
 * drives - rather than approximating it in the browser, so what the panels show
 * is a mask that actually exists.
 */

import { API_PREFIX, apiGet } from './client'

import type { TissuePanelName, TissueReport } from '@/types/tissue'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

/** `threshold` and `targetMpp`, omitted when they are not being overridden. */
export interface TissueOptions {
  /** Saturation level to cut at, 0-255. Omit to let Otsu choose. */
  threshold?: number | null
  targetMpp?: number | null
}

function query(options: TissueOptions): string {
  const params = new URLSearchParams()
  if (options.threshold !== null && options.threshold !== undefined) {
    params.set('threshold', String(Math.round(options.threshold)))
  }
  if (options.targetMpp) params.set('targetMpp', String(options.targetMpp))

  const encoded = params.toString()
  return encoded ? `?${encoded}` : ''
}

/**
 * Run step 3 and read the result.
 *
 * A generous ceiling: the first call for a slide reopens a multi-gigabyte scan
 * to read its thumbnail, which is far longer than a metadata call.
 */
export function fetchTissueReport(
  uploadId: string,
  options: TissueOptions = {},
  signal?: AbortSignal,
): Promise<TissueReport> {
  return apiGet<TissueReport>(`/tissue/${uploadId}${query(options)}`, {
    signal,
    timeoutMs: 120_000,
  })
}

/**
 * URL of one panel.
 *
 * `thumbnail` and `saturation` ignore the threshold, so their URLs are stable
 * and the browser caches them across a drag. `mask` and `overlay` carry it, so
 * each distinct cut is a distinct URL and the cache works for free when the
 * viewer drags back to a value they have already seen.
 */
export function tissuePanelUrl(
  uploadId: string,
  name: TissuePanelName,
  options: TissueOptions = {},
): string {
  const usesThreshold = name === 'mask' || name === 'overlay'
  const applicable: TissueOptions = usesThreshold
    ? options
    : { targetMpp: options.targetMpp }

  return url(`/tissue/${uploadId}/panels/${name}.png${query(applicable)}`)
}
