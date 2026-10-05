/** Step 17: agreement against pathologist readings, or an honest absence of it. */

import { apiGet } from './client'

import type { ValidationReport } from '@/types/validation'

/**
 * Resolves with `available: false` and a reason when there is no reader sheet,
 * rather than rejecting. "Nothing to validate against" is this step's result
 * today, not a failed request.
 */
export function fetchValidation(
  caseId: string | null,
  signal?: AbortSignal,
): Promise<ValidationReport> {
  const query = caseId ? `?caseId=${encodeURIComponent(caseId)}` : ''
  return apiGet<ValidationReport>(`/validation${query}`, { signal, timeoutMs: 120_000 })
}
