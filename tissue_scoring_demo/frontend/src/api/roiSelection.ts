/**
 * Client for step 10 — reviewing and selecting the candidate regions.
 *
 * One request, no polling: there is no model here. A cold build is one padded slide
 * read per candidate — seconds — and a cached read is immediate. See
 * `roi_selection_service.py` for why this is not job-shaped the way steps 8 and 11 are.
 */

import { API_PREFIX, apiGet, apiPost, apiPut } from './client'

import type { RoiSelectionReport } from '@/types/roiSelection'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

/**
 * Build the candidate list and its cards, or return the cached one.
 *
 * The timeout is generous because the cost is a slide read per candidate and a busy
 * section can offer forty of them.
 */
export function buildRoiSelection(
  uploadId: string,
  options: { rebuild?: boolean } = {},
): Promise<RoiSelectionReport> {
  const query = options.rebuild ? '?rebuild=true' : ''
  return apiPost<RoiSelectionReport>(`/roi-selection/${uploadId}/build${query}`, undefined, {
    timeoutMs: 120_000,
  })
}

/** The cached list. Rejects until it has been built. */
export function fetchRoiSelection(
  uploadId: string,
  signal?: AbortSignal,
): Promise<RoiSelectionReport> {
  return apiGet<RoiSelectionReport>(`/roi-selection/${uploadId}`, { signal, timeoutMs: 30_000 })
}

/**
 * Replace the ticked set outright and get the whole report back.
 *
 * Replacement rather than a merge is what makes "deselect all" expressible — see
 * `apiPut`. The full report comes back because ticking changes the numbers under the
 * list (selected area, share of the tumour, windows it costs), and re-fetching to learn
 * them would render a stale cost beside a fresh selection.
 */
export function setRoiSelection(
  uploadId: string,
  selected: string[],
): Promise<RoiSelectionReport> {
  return apiPut<RoiSelectionReport>(
    `/roi-selection/${uploadId}/selection`,
    { selected },
    { timeoutMs: 30_000 },
  )
}

/** URL of one candidate's card: the crop with step 8's square boundary drawn on it. */
export function roiCandidateCardUrl(uploadId: string, roiId: string): string {
  return url(`/roi-selection/${uploadId}/cards/${roiId}.png`)
}
