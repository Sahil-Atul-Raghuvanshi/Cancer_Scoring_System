/** Previous cases: what has been scored, and getting back into it. */

import { API_PREFIX, apiDelete, apiGet, apiPost } from './client'

import type { HistoryCase } from '@/types/history'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export function fetchHistory(signal?: AbortSignal): Promise<{ cases: HistoryCase[] }> {
  return apiGet<{ cases: HistoryCase[] }>('/history', { signal, timeoutMs: 30_000 })
}

/**
 * Bring a filed run back into the working tree.
 *
 * A directory rename, so it returns in well under a second however large the run
 * is - but it is still a POST rather than a GET, because it moves bytes.
 */
export function openHistoryMarker(caseId: string, marker: string): Promise<unknown> {
  return apiPost(`/history/${encodeURIComponent(caseId)}/${marker}/open`, undefined, {
    timeoutMs: 60_000,
  })
}

/** File a finished run, so it stops taking up room in the working tree. */
export function archiveHistoryMarker(caseId: string, marker: string): Promise<unknown> {
  return apiPost(`/history/${encodeURIComponent(caseId)}/${marker}/archive`, undefined, {
    timeoutMs: 60_000,
  })
}

/** Remove one antibody's run, from whichever tree it is in. Not undoable. */
export function deleteHistoryMarker(caseId: string, marker: string): Promise<unknown> {
  return apiDelete(`/history/${encodeURIComponent(caseId)}/${marker}`, {
    timeoutMs: 60_000,
  })
}

/** The case's stored H&E overview. */
export function historyThumbnailUrl(caseId: string): string {
  return `${BASE_URL}${API_PREFIX}/history/${encodeURIComponent(caseId)}/thumbnail.png`
}
