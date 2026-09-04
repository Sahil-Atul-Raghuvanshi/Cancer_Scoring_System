/**
 * Client for step 2 - quality control.
 *
 * A QC run is started, polled, then read. It cannot be one request: GrandQC's
 * artefact pass over a whole slide is minutes of CPU work, and there is no
 * useful timeout to give a request that long.
 *
 * `runQualityControl` wraps the whole cycle, so a caller that just wants "run
 * step 2 and tell me when it is done" gets a single promise, with progress
 * arriving through a callback on the way.
 */

import { API_PREFIX, ApiError, CancelledError, apiGet, describeFailure } from './client'

import type {
  QCCapability,
  QCGrid,
  QCMetricKey,
  QCRegionExplain,
  QCReport,
  QCRun,
} from '@/types/qc'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/** Surface the API's `detail` rather than a bare status code. */
async function post<T>(path: string): Promise<T> {
  const response = await fetch(url(path), {
    method: 'POST',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    throw new ApiError(await describeFailure(response, path), response.status)
  }

  return (await response.json()) as T
}

/**
 * Whether step 2 can run here, and precisely what is missing when it cannot.
 *
 * Generous timeout on purpose. The first call to this endpoint is what imports
 * PyTorch on the server, which takes about five seconds cold - comfortably past
 * the client default, so with the default it aborts and the UI reports the
 * backend as unreachable when it is merely busy starting up.
 */
export function fetchQCCapability(signal?: AbortSignal): Promise<QCCapability> {
  return apiGet<QCCapability>('/qc/capability', { signal, timeoutMs: 30_000 })
}

/** Start a run, or get back the cached one when the settings already match. */
export function startQCRun(
  uploadId: string,
  modelMpp?: number | null,
  restart = false,
): Promise<QCRun> {
  const params = new URLSearchParams()
  if (modelMpp) params.set('modelMpp', String(modelMpp))
  if (restart) params.set('restart', 'true')
  const query = params.toString()
  return post<QCRun>(`/qc/${uploadId}/run${query ? `?${query}` : ''}`)
}

/**
 * Ask a running pass to stop.
 *
 * A real server-side stop, not just an abandoned wait: the flag it sets is read by
 * the progress callback after every patch, so the run ends rather than carrying on
 * unwatched and burning the cores the next step needs.
 */
export function cancelQCRun(uploadId: string): Promise<QCRun> {
  return post<QCRun>(`/qc/${uploadId}/cancel`)
}

/**
 * Poll a run.
 *
 * Longer than the default because while a run is going the server has every
 * core busy doing inference, so even a trivial handler can be slow to be
 * scheduled. A poll that times out would abort the whole wait.
 */
export function fetchQCRun(uploadId: string, signal?: AbortSignal): Promise<QCRun> {
  return apiGet<QCRun>(`/qc/${uploadId}/run`, { signal, timeoutMs: 20_000 })
}

export function fetchQCReport(uploadId: string, signal?: AbortSignal): Promise<QCReport> {
  return apiGet<QCReport>(`/qc/${uploadId}`, { signal, timeoutMs: 30_000 })
}

export function fetchQCGrid(uploadId: string, signal?: AbortSignal): Promise<QCGrid> {
  return apiGet<QCGrid>(`/qc/${uploadId}/grid`, { signal, timeoutMs: 30_000 })
}

/** Re-measure one region finely, and compare it with this slide's clean tissue. */
export function fetchQCRegion(
  uploadId: string,
  params: { x: number; y: number; size: number; targetMpp?: number | null },
  signal?: AbortSignal,
): Promise<QCRegionExplain> {
  const query = new URLSearchParams({
    x: String(Math.round(params.x)),
    y: String(Math.round(params.y)),
    size: String(Math.round(params.size)),
  })
  if (params.targetMpp) query.set('targetMpp', String(params.targetMpp))

  return apiGet<QCRegionExplain>(`/qc/${uploadId}/explain?${query.toString()}`, {
    signal,
    timeoutMs: 60_000,
  })
}

/* --- image URLs ----------------------------------------------------------- */

/** The slide with its artefacts tinted. */
export const qcOverlayUrl = (uploadId: string) => url(`/qc/${uploadId}/overlay.png`)

/** Pass 1's tissue map, kept separate so the two models stay visibly two. */
export const qcTissueUrl = (uploadId: string) => url(`/qc/${uploadId}/tissue.png`)

/** The class map with no slide under it. */
export const qcClassesUrl = (uploadId: string) => url(`/qc/${uploadId}/classes.png`)

/** Indexed-colour mask, one byte per pixel, for download. */
export const qcMaskUrl = (uploadId: string) => url(`/qc/${uploadId}/mask.png`)

/** One classical metric over the patch grid. */
export const qcHeatmapUrl = (uploadId: string, metric: QCMetricKey) =>
  url(`/qc/${uploadId}/heatmap/${metric}.png`)

/** The pixels of an inspected region. */
export function qcRegionImageUrl(
  uploadId: string,
  params: { x: number; y: number; size: number; out?: number },
): string {
  const query = new URLSearchParams({
    x: String(Math.round(params.x)),
    y: String(Math.round(params.y)),
    size: String(Math.round(params.size)),
    out: String(params.out ?? 512),
  })
  return url(`/qc/${uploadId}/explain.png?${query.toString()}`)
}

/* --- the whole cycle ------------------------------------------------------ */

export interface RunOptions {
  modelMpp?: number | null
  /** Discard the cached run and start again. */
  restart?: boolean
  onProgress?: (run: QCRun) => void
  signal?: AbortSignal
}

/** How often to ask for progress. */
const POLL_INTERVAL_MS = 1500

/**
 * A hard ceiling on how long we will wait, so a wedged run does not leave the
 * UI spinning for ever. Generous: a 30 mm slide at 7x on a laptop CPU is a
 * genuine ten minutes of work, and cutting that off would be the wrong call.
 */
const MAX_WAIT_MS = 45 * 60 * 1000

/**
 * Run step 2 end to end: start, poll to completion, return the report.
 *
 * A run already cached on the server comes back `ready` from the first call, so
 * revisiting the step is instant rather than a re-run.
 */
export async function runQualityControl(
  uploadId: string,
  options: RunOptions = {},
): Promise<QCReport> {
  const { modelMpp, restart, onProgress, signal } = options

  let run = await startQCRun(uploadId, modelMpp, restart)
  onProgress?.(run)

  const deadline = Date.now() + MAX_WAIT_MS

  while (run.state === 'queued' || run.state === 'running') {
    if (signal?.aborted) throw new CancelledError('quality control was stopped')
    if (Date.now() > deadline) {
      throw new Error('quality control did not finish within 45 minutes')
    }

    await sleep(POLL_INTERVAL_MS)
    run = await fetchQCRun(uploadId, signal)
    onProgress?.(run)
  }

  // Cancelled is checked before failed, and is not one: the server reports it as
  // its own state precisely so the caller can tell "someone stopped this" from
  // "something broke".
  if (run.state === 'cancelled') {
    throw new CancelledError(run.message ?? 'quality control was stopped')
  }
  if (run.state === 'failed') {
    throw new Error(run.error ?? 'quality control failed')
  }
  if (run.state === 'idle') {
    throw new Error('quality control did not start')
  }

  return fetchQCReport(uploadId, signal)
}
