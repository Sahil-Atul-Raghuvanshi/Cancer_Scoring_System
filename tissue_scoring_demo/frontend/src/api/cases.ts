/** Resolve a case folder to its slides, and load one marker's H&E + IHC pair. */

import { apiPost } from './client'

import type { CaseResolution, CaseSession } from '@/types/case'

export function resolveCase(casePath: string, signal?: AbortSignal): Promise<CaseResolution> {
  return apiPost<CaseResolution>('/cases/resolve', { casePath }, { signal })
}

export function loadCase(
  casePath: string,
  marker: string,
  signal?: AbortSignal,
): Promise<CaseSession> {
  // Registering a multi-gigabyte local slide proves it opens before returning,
  // which is slower than a typical POST.
  return apiPost<CaseSession>('/cases/load', { casePath, marker }, { signal, timeoutMs: 60_000 })
}
