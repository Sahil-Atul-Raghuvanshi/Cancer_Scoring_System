/**
 * Client for step 11 — BEETLE on the chosen regions, one at a time.
 *
 * Job-shaped like step 8's: start, poll, read. The difference from step 8 is that the
 * *report* is worth reading while the run is still going — every region writes its
 * result as it finishes, so a caller polls `fetchRefinementRun` for the position and
 * re-reads `fetchRefinementReport` to draw whatever has landed.
 */

import { API_PREFIX, apiGet, apiPost } from './client'

import type {
  RefinementRegionPanel,
  RefinementReport,
  RefinementRun,
  RefinementSlidePanel,
} from '@/types/roiRefinement'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

/**
 * Queue a pass over whatever is selected and not already refined.
 *
 * `roiIds` narrows it to those regions, which is how one failed region is retried
 * without disturbing the rest; the server intersects them with step 10's selection
 * either way, so this cannot reach a region nobody ticked. `rebuild` recomputes regions
 * that already finished — off by default, which is what makes a restart cheap.
 *
 * Comes back `ready` with no worker started when everything asked for is already on
 * disk. That is the resume path.
 */
export function startRefinement(
  uploadId: string,
  options: { roiIds?: string[]; rebuild?: boolean } = {},
): Promise<RefinementRun> {
  return apiPost<RefinementRun>(
    `/roi-refinement/${uploadId}/run`,
    { roiIds: options.roiIds ?? null, rebuild: options.rebuild ?? false },
    { timeoutMs: 60_000 },
  )
}

/** Ask a running pass to stop after the region it is on, keeping the finished ones. */
export function cancelRefinement(uploadId: string): Promise<RefinementRun> {
  return apiPost<RefinementRun>(`/roi-refinement/${uploadId}/cancel`, undefined, {
    timeoutMs: 15_000,
  })
}

/**
 * How far the pass has got: which region, how many regions, how many windows — and,
 * with `paintedSince`, what the region in progress has just painted.
 *
 * `paintedSince` is a position in the **current region's** paint log, not the pass's,
 * and the server hands back where it got to in `paintedCursor`. Pass that back next
 * time rather than counting locally. The cursor resets when the pass moves to the next
 * region, so it is `paint.roiId` — not a cursor that went backwards — that tells a
 * client its canvas now belongs to something else.
 */
export function fetchRefinementRun(
  uploadId: string,
  signal?: AbortSignal,
  paintedSince?: number,
): Promise<RefinementRun> {
  const feed = paintedSince === undefined ? '' : `?paintedSince=${paintedSince}`
  return apiGet<RefinementRun>(`/roi-refinement/${uploadId}/run${feed}`, {
    signal,
    timeoutMs: 20_000,
  })
}

/** Every selected region with its state and, where it finished, its boundary. */
export function fetchRefinementReport(
  uploadId: string,
  signal?: AbortSignal,
): Promise<RefinementReport> {
  return apiGet<RefinementReport>(`/roi-refinement/${uploadId}`, { signal, timeoutMs: 30_000 })
}

/**
 * URL of one region's picture.
 *
 * `tile` and `beetle_overlay` are the comparison — the same crop with different
 * geometry on it. `beetle_mask` is the one that cannot flatter the result: it shows
 * what the network said about every pixel of the box, so a refinement that had simply
 * returned the rectangle would be obvious at a glance.
 */
export function refinementRegionUrl(
  uploadId: string,
  roiId: string,
  name: RefinementRegionPanel,
): string {
  return url(`/roi-refinement/${uploadId}/regions/${roiId}/${name}.png`)
}

/** URL of a slide-level panel: the chosen squares, or the boundaries inside them. */
export function refinementPanelUrl(uploadId: string, name: RefinementSlidePanel): string {
  return url(`/roi-refinement/${uploadId}/panels/${name}.png`)
}
