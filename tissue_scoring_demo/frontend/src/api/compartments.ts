/** Step 13: build the region of each cell where the marker is supposed to be. */

import { API_PREFIX, apiGet } from './client'

import type { CompartmentsReport, RegionCompartmentRings } from '@/types/compartments'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

/**
 * The compartments at a given width.
 *
 * Note what is *not* a parameter: which compartment the marker gets. That is
 * read from the antibody letter on the server and cannot be overridden, because
 * giving a cytoplasmic marker a membrane ring is the regression this step exists
 * to prevent - and the cleanest way to prevent it is to make it unreachable.
 */
export function fetchCompartments(
  heUploadId: string,
  ihcUploadId: string,
  options: { widthUm?: number; voronoi?: boolean } = {},
  signal?: AbortSignal,
): Promise<CompartmentsReport> {
  const query = new URLSearchParams({ ihcUploadId })
  if (options.widthUm !== undefined) query.set('widthUm', String(options.widthUm))
  if (options.voronoi === false) query.set('voronoi', 'false')
  // Generous, and measured rather than guessed. Even served from cache this is
  // ~18 s on a case with nineteen invasive regions - the cache check stats every
  // field's stored geometry and the width sweep rebuilds one field six times,
  // with a dilation per nucleus inside each. An uncached build is minutes. At
  // 30 s the request was being abandoned while the server was still working,
  // which showed up on step 13 as an aborted signal and no compartments.
  return apiGet<CompartmentsReport>(`/compartments/${heUploadId}?${query}`, {
    signal,
    timeoutMs: 300_000,
  })
}

/** One field drawn as nucleus / cell body / measured compartment. */
export function compartmentFieldUrl(
  heUploadId: string,
  ihcUploadId: string,
  rank: number,
  index: number,
  cacheKey?: string | null,
): string {
  const query = new URLSearchParams({ ihcUploadId })
  if (cacheKey) query.set('v', cacheKey)
  return `${BASE_URL}${API_PREFIX}/compartments/${heUploadId}/regions/${rank}/fields/${index}.png?${query}`
}

/**
 * One region's compartments as outlines, for drawing on the slide.
 *
 * Deliberately not part of the report: the report is fetched on every drag of
 * the width slider and this is a few megabytes of vertices. Written by the
 * report request, so asking for this before that has happened is a 404 rather
 * than a build.
 */
export function fetchCompartmentGeometry(
  heUploadId: string,
  ihcUploadId: string,
  rank: number,
  signal?: AbortSignal,
): Promise<RegionCompartmentRings> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<RegionCompartmentRings>(
    `/compartments/${heUploadId}/regions/${rank}/geometry?${query}`,
    { signal, timeoutMs: 30_000 },
  )
}
