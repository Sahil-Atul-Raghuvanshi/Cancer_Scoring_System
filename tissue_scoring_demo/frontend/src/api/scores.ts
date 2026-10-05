/** Step 16: the deliverable - percent positive and intensity, per marker. */

import { apiGet } from './client'

import type { CaseScoreReport, ScoreReport } from '@/types/scores'

/** One marker's pair, and the arithmetic that produced it. */
export function fetchScore(
  heUploadId: string,
  ihcUploadId: string,
  signal?: AbortSignal,
): Promise<ScoreReport> {
  const query = new URLSearchParams({ ihcUploadId })
  return apiGet<ScoreReport>(`/scores/${heUploadId}?${query}`, {
    signal,
    timeoutMs: 60_000,
  })
}

/**
 * The whole case: five markers, ten numbers.
 *
 * Returns a partial grid rather than failing when only some markers have been
 * scored - which is the honest picture of a run in progress, and more useful
 * than an error.
 */
export function fetchCaseScores(
  caseId: string,
  signal?: AbortSignal,
): Promise<CaseScoreReport> {
  return apiGet<CaseScoreReport>(`/scores/case/${encodeURIComponent(caseId)}`, {
    signal,
    timeoutMs: 120_000,
  })
}
