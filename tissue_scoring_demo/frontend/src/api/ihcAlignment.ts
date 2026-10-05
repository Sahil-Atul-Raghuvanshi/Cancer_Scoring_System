/** Step 10: register the case's two slides and carry the ROI across. */

import { API_PREFIX, apiGet, apiPost } from './client'

import type {
  AlignmentCapability,
  AlignmentReport,
  AlignmentRun,
} from '@/types/ihcAlignment'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export function fetchAlignmentCapability(signal?: AbortSignal): Promise<AlignmentCapability> {
  return apiGet<AlignmentCapability>('/ihc-alignment/capability', { signal })
}

export function startAlignment(
  heUploadId: string,
  ihcUploadId: string,
  options: { restart?: boolean } = {},
): Promise<AlignmentRun> {
  const query = new URLSearchParams({ ihcUploadId })
  if (options.restart) query.set('restart', 'true')
  return apiPost<AlignmentRun>(`/ihc-alignment/${heUploadId}/run?${query}`, undefined, {
    timeoutMs: 30_000,
  })
}

export function fetchAlignmentRun(
  heUploadId: string,
  ihcUploadId: string,
  signal?: AbortSignal,
): Promise<AlignmentRun> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<AlignmentRun>(`/ihc-alignment/${heUploadId}/run?${query}`, { signal })
}

export function fetchAlignmentReport(
  heUploadId: string,
  ihcUploadId: string,
  signal?: AbortSignal,
): Promise<AlignmentReport> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<AlignmentReport>(`/ihc-alignment/${heUploadId}?${query}`, { signal })
}

export function confirmAlignment(
  heUploadId: string,
  ihcUploadId: string,
  confirmed: boolean,
): Promise<AlignmentReport> {
  const query = new URLSearchParams({ ihcUploadId, confirmed: String(confirmed) })
  return apiPost<AlignmentReport>(`/ihc-alignment/${heUploadId}/confirm?${query}`)
}

/** One slide's whole-slide panel with the carried regions outlined on it. */
export function alignmentPanelUrl(
  heUploadId: string,
  ihcUploadId: string,
  name: 'he_borders' | 'ihc_borders',
  cacheKey?: string | null,
): string {
  const query = new URLSearchParams({ ihcUploadId })
  if (cacheKey) query.set('v', cacheKey)
  return `${BASE_URL}${API_PREFIX}/ihc-alignment/${heUploadId}/panels/${name}.png?${query}`
}

/** One carried region, cropped from the IHC slide (or the H&E, to compare). */
export function alignmentCropUrl(
  heUploadId: string,
  ihcUploadId: string,
  rank: number,
  source: 'ihc' | 'he',
  cacheKey?: string | null,
): string {
  const query = new URLSearchParams({ ihcUploadId, source })
  if (cacheKey) query.set('v', cacheKey)
  return `${BASE_URL}${API_PREFIX}/ihc-alignment/${heUploadId}/crops/${rank}.png?${query}`
}
