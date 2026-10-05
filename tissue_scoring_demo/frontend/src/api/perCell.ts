/** Step 14: measure the DAB in every tumour cell's own compartment. */

import { API_PREFIX, apiGet } from './client'

import type { CellRow, PerCellReport } from '@/types/perCell'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

/**
 * The measurement.
 *
 * Note what is *not* a parameter: which compartment, which second number and
 * which cut points. All three resolve from the antibody letter on the server. A
 * `?compartment=` would be a way to measure a cytoplasmic marker in a ring from
 * a URL, and the cleanest way to prevent that is for it not to exist.
 *
 * The timeout is generous because this reads every sampled field off the slide
 * again - tens of seconds on a large case - and a request that gave up early
 * would look on screen exactly like a step that failed.
 */
export function fetchPerCell(
  heUploadId: string,
  ihcUploadId: string,
  signal?: AbortSignal,
): Promise<PerCellReport> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<PerCellReport>(`/per-cell/${heUploadId}?${query}`, {
    signal,
    timeoutMs: 600_000,
  })
}

/** The field a given cell was measured in, for the crop behind a clicked dot. */
export function perCellFieldUrl(
  heUploadId: string,
  ihcUploadId: string,
  rank: number,
  index: number,
): string {
  const query = new URLSearchParams({ ihcUploadId })
  return `${BASE_URL}${API_PREFIX}/per-cell/${heUploadId}/fields/${rank}/${index}.png?${query}`
}

/**
 * Every measured cell, not the scatter's sample.
 *
 * Kept out of the report because the report is fetched every time the screen
 * opens and this runs to tens of thousands of rows. The slide overlay needs all
 * of them: a cell the sample skipped is still on the slide, and still has to
 * answer a click.
 */
export function fetchPerCellRows(
  heUploadId: string,
  ihcUploadId: string,
  signal?: AbortSignal,
): Promise<{ cells: CellRow[]; total: number }> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<{ cells: CellRow[]; total: number }>(
    `/per-cell/${heUploadId}/cells?${query}`,
    { signal, timeoutMs: 60_000 },
  )
}
