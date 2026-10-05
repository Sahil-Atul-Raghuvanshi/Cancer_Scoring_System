/**
 * Client for step 9 — the ROI mask.
 *
 * One request, no polling: the region and its borders take about a tenth of a second
 * to compute, and a cold build's real cost is the thumbnail and panels it draws
 * eagerly — roughly twenty seconds; a cached read is milliseconds. See
 * `roi_service.py`'s module docstring for why this is not job-shaped the way step 8 is.
 */

import { API_PREFIX, apiGet, apiPost } from './client'

import type { RoiClassName, RoiPanelName, RoiReport, RoiTopClass } from '@/types/roi'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

/** Build the region (and its borders, crops and export), or return the cached one. */
export function buildRoi(uploadId: string, options: { rebuild?: boolean } = {}): Promise<RoiReport> {
  const query = options.rebuild ? '?rebuild=true' : ''
  return apiPost<RoiReport>(`/roi/${uploadId}/build${query}`, undefined, { timeoutMs: 60_000 })
}

/** The cached report. Rejects with a 404-shaped error until it has been built. */
export function fetchRoiReport(uploadId: string, signal?: AbortSignal): Promise<RoiReport> {
  return apiGet<RoiReport>(`/roi/${uploadId}`, { signal, timeoutMs: 30_000 })
}

/**
 * URL of one of the seven panels — `seed`, `smoothed`, `binary`, `region`, `outline`,
 * `borders` and `borders_on_slide`. The last two are the same trace pictured twice:
 * the geometry alone, and the same geometry on the scan with each region tinted.
 */
export function roiPanelUrl(uploadId: string, name: RoiPanelName): string {
  return url(`/roi/${uploadId}/panels/${name}.png`)
}

/** URL of the `rank`-th largest (1 = largest, up to 3) DCIS or invasive crop. */
export function roiTopUrl(uploadId: string, className: RoiTopClass, rank: number): string {
  return url(`/roi/${uploadId}/top/${className}/${rank}.png`)
}

/**
 * URL of one region a viewer selected off the borders panel, enlarged from the slide
 * with its borders drawn on it. `index` is 0-based — the region's own `index`, not a
 * 1-based rank — matching whatever `classRegions[className]` the report gave it.
 */
export function roiRegionCropUrl(uploadId: string, className: RoiClassName, index: number): string {
  return url(`/roi/${uploadId}/region/${className}/${index}.png`)
}

/** URL of the QuPath-importable GeoJSON export of every border-class region. */
export function roiQupathUrl(uploadId: string): string {
  return url(`/roi/${uploadId}/qupath.geojson`)
}
