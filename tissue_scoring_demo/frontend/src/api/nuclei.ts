/** Step 11: find the nuclei inside the regions step 10 carried onto the IHC slide. */

import { API_PREFIX, apiGet, apiPost } from './client'

import type {
  NucleiCapability,
  NucleiReport,
  NucleiRun,
  RegionNucleiPayload,
} from '@/types/nuclei'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export function fetchNucleiCapability(signal?: AbortSignal): Promise<NucleiCapability> {
  return apiGet<NucleiCapability>('/nuclei/capability', { signal })
}

export function startNuclei(
  heUploadId: string,
  ihcUploadId: string,
  options: { restart?: boolean } = {},
): Promise<NucleiRun> {
  const query = new URLSearchParams({ ihcUploadId })
  if (options.restart) query.set('restart', 'true')
  return apiPost<NucleiRun>(`/nuclei/${heUploadId}/run?${query}`, undefined, {
    timeoutMs: 30_000,
  })
}

export function fetchNucleiRun(
  heUploadId: string,
  ihcUploadId: string,
  signal?: AbortSignal,
): Promise<NucleiRun> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<NucleiRun>(`/nuclei/${heUploadId}/run?${query}`, { signal })
}

export function fetchNucleiReport(
  heUploadId: string,
  ihcUploadId: string,
  signal?: AbortSignal,
): Promise<NucleiReport> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<NucleiReport>(`/nuclei/${heUploadId}?${query}`, { signal })
}

/**
 * One region's outlines, fetched on demand.
 *
 * Deliberately not part of the report: a region's geometry is megabytes of
 * vertices and the report is re-fetched every time the screen opens, so the
 * summary stays small and this is paid once, per region, when somebody actually
 * looks at one. The longer timeout is for the same reason.
 */
export function fetchRegionNuclei(
  heUploadId: string,
  ihcUploadId: string,
  rank: number,
  signal?: AbortSignal,
): Promise<RegionNucleiPayload> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<RegionNucleiPayload>(`/nuclei/${heUploadId}/regions/${rank}?${query}`, {
    signal,
    timeoutMs: 30_000,
  })
}

/** One sampled field: the tissue, what the model saw, or the outlines it drew. */
export function nucleiFieldUrl(
  heUploadId: string,
  ihcUploadId: string,
  rank: number,
  index: number,
  view: 'raw' | 'input' | 'overlay',
  cacheKey?: string | null,
): string {
  const query = new URLSearchParams({ ihcUploadId, view })
  if (cacheKey) query.set('v', cacheKey)
  return `${BASE_URL}${API_PREFIX}/nuclei/${heUploadId}/regions/${rank}/fields/${index}.png?${query}`
}

/** A comparison panel: the watershed baseline, or detection on the raw stain. */
export function nucleiComparisonUrl(
  heUploadId: string,
  ihcUploadId: string,
  rank: number,
  view: 'watershed' | 'rgb',
  cacheKey?: string | null,
): string {
  const query = new URLSearchParams({ ihcUploadId, view })
  if (cacheKey) query.set('v', cacheKey)
  return `${BASE_URL}${API_PREFIX}/nuclei/${heUploadId}/compare/${rank}.png?${query}`
}
