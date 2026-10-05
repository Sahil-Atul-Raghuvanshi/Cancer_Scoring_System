/** Step 12: sort step 11's nuclei into the ones that count and the ones that do not. */

import { apiGet } from './client'

import type { CellTypingReport, RegionTypesPayload, TypingRules } from '@/types/cellTyping'

/**
 * The mix at a given set of thresholds.
 *
 * A GET with the thresholds as query parameters, not a POST, because this
 * computes nothing that was not already implied by step 11's output - so a
 * viewer who drags a slider and reloads the page gets the same answer back.
 */
export function fetchCellTyping(
  heUploadId: string,
  ihcUploadId: string,
  rules?: Partial<TypingRules>,
  signal?: AbortSignal,
): Promise<CellTypingReport> {
  const query = new URLSearchParams({ ihcUploadId })
  for (const [key, value] of Object.entries(rules ?? {})) {
    if (value !== undefined) query.set(key, String(value))
  }
  return apiGet<CellTypingReport>(`/cell-typing/${heUploadId}?${query}`, {
    signal,
    timeoutMs: 20_000,
  })
}

/** A class per nucleus id, for colouring one region's outlines. */
export function fetchRegionTypes(
  heUploadId: string,
  ihcUploadId: string,
  rank: number,
  signal?: AbortSignal,
): Promise<RegionTypesPayload> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<RegionTypesPayload>(
    `/cell-typing/${heUploadId}/regions/${rank}?${query}`,
    { signal },
  )
}
