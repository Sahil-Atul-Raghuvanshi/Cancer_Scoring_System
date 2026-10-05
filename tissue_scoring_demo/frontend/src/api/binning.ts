/** Step 15: sort every measured cell into a level, on this antibody's own cuts. */

import { apiGet } from './client'

import type { BinnedCell, BinningReport } from '@/types/binning'

/**
 * The bins, the histogram and the cut lines.
 *
 * The cut points are deliberately not parameters. Step 13's width is one,
 * because neither default is established fact and sweeping it is the point; a
 * cut point is different, because it is the thing being calibrated - and a
 * `?odCuts=` would be a way to reach a reported number by choosing the
 * threshold that produces it.
 */
export function fetchBinning(
  heUploadId: string,
  ihcUploadId: string,
  signal?: AbortSignal,
): Promise<BinningReport> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<BinningReport>(`/binning/${heUploadId}?${query}`, {
    signal,
    timeoutMs: 60_000,
  })
}

/** One bin and one verdict per cell, for colouring them on the slide. */
export function fetchBinnedCells(
  heUploadId: string,
  ihcUploadId: string,
  signal?: AbortSignal,
): Promise<{ cells: BinnedCell[]; total: number }> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<{ cells: BinnedCell[]; total: number }>(
    `/binning/${heUploadId}/cells?${query}`,
    { signal, timeoutMs: 60_000 },
  )
}
