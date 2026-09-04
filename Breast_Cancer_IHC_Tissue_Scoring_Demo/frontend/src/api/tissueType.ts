/**
 * Client for step 8 — tissue-type segmentation.
 *
 * A pass is started, polled, then read. It cannot be one request: a whole slide is
 * tens of thousands of ResNet18 forward passes, which on a CPU is tens of minutes,
 * and there is no useful timeout to give a request that long.
 *
 * `runTissueType` wraps the whole cycle, so a caller that just wants "run step 8
 * and tell me when it is done" gets one promise with progress on the way. A pass
 * already cached on the server comes back `ready` from the first call, so
 * revisiting the step is instant rather than a re-run — which matters more here
 * than anywhere else in the pipeline, because re-running is half an hour.
 */

import { API_PREFIX, ApiError, CancelledError, apiGet, describeFailure } from './client'

import type {
  TissueTypeCapability,
  TissueTypePanelName,
  TissueTypeReport,
  TissueTypeRun,
} from '@/types/tissueType'

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
 * Whether step 8 can run here, which checkpoints are published, and under what
 * licence each would leave the result.
 *
 * Generous timeout for step 2's reason: the first call to this endpoint is what
 * imports PyTorch on the server, which takes about five seconds cold — past the
 * client default, so with the default it aborts and the UI reports the backend as
 * unreachable when it is merely starting up.
 */
export function fetchTissueTypeCapability(
  signal?: AbortSignal,
): Promise<TissueTypeCapability> {
  return apiGet<TissueTypeCapability>('/tissue-type/capability', {
    signal,
    timeoutMs: 30_000,
  })
}

export interface TissueTypeOptions {
  /** Fraction of its field of view a window shares with its neighbour. */
  overlap?: number | null
  /** Which published checkpoint scores the slide. A licence choice as much as an accuracy one. */
  model?: string | null
  /**
   * Discard the cached pass and run again.
   *
   * Needed because starting otherwise returns the cache whenever the parameters
   * match - which is what makes revisiting this screen instant rather than another
   * half hour, and is exactly wrong when the viewer is asking for a fresh pass.
   */
  restart?: boolean
}

function query(options: TissueTypeOptions): string {
  const params = new URLSearchParams()
  if (options.overlap !== null && options.overlap !== undefined) {
    params.set('overlap', String(options.overlap))
  }
  if (options.model) params.set('model', options.model)
  if (options.restart) params.set('restart', 'true')

  const encoded = params.toString()
  return encoded ? `?${encoded}` : ''
}

/** Start a pass, or get back the cached one when the parameters already match. */
export function startTissueTypeRun(
  uploadId: string,
  options: TissueTypeOptions = {},
): Promise<TissueTypeRun> {
  return post<TissueTypeRun>(`/tissue-type/${uploadId}/run${query(options)}`)
}

/**
 * Ask a running pass to stop.
 *
 * A real server-side stop, not an abandoned wait. The flag it sets is read after
 * every block of 64 patches - about two seconds - so the pass actually ends instead
 * of carrying on unwatched for another half hour with every core busy.
 */
export function cancelTissueTypeRun(uploadId: string): Promise<TissueTypeRun> {
  return post<TissueTypeRun>(`/tissue-type/${uploadId}/cancel`)
}

/**
 * Poll a pass.
 *
 * Longer than the default because while a pass is going the server has every core
 * busy doing inference, so even a trivial handler can be slow to be scheduled. A
 * poll that timed out would abort the whole wait.
 */
export function fetchTissueTypeRun(
  uploadId: string,
  signal?: AbortSignal,
): Promise<TissueTypeRun> {
  return apiGet<TissueTypeRun>(`/tissue-type/${uploadId}/run`, {
    signal,
    timeoutMs: 20_000,
  })
}

export function fetchTissueTypeReport(
  uploadId: string,
  signal?: AbortSignal,
): Promise<TissueTypeReport> {
  return apiGet<TissueTypeReport>(`/tissue-type/${uploadId}`, {
    signal,
    timeoutMs: 30_000,
  })
}

/* --- image URLs ----------------------------------------------------------- */

/**
 * URL of one panel, optionally filtered to a set of classes.
 *
 * The filter is the guide's per-class opacity toggle, and it is applied on the
 * server so that an unselected class is *absent* from the picture rather than
 * recoloured — the viewer switching fat off should watch the tissue leave the map
 * and the denominator with it. Each distinct selection is a distinct URL, so the
 * browser's cache works for free when they toggle back.
 */
export function tissueTypePanelUrl(
  uploadId: string,
  name: TissueTypePanelName,
  classes?: readonly number[] | null,
): string {
  const filter =
    classes && (name === 'map' || name === 'flat')
      ? `?classes=${[...classes].sort().join(',')}`
      : ''
  return url(`/tissue-type/${uploadId}/panels/${name}.png${filter}`)
}

/* --- the whole cycle ------------------------------------------------------ */

export interface RunOptions extends TissueTypeOptions {
  onProgress?: (run: TissueTypeRun) => void
  signal?: AbortSignal
}

/** How often to ask for progress. */
const POLL_INTERVAL_MS = 2000

/**
 * A hard ceiling, so a wedged pass does not leave the UI spinning for ever.
 *
 * Generous, and measured rather than guessed: one 198 mm² IHC slide is about 31
 * minutes at the default overlap on four CPU threads. Three times that leaves room
 * for a larger section or a busier machine without cutting off a pass that is
 * genuinely still working — and the viewer can stop it themselves at any point,
 * which is the control that actually matters at this duration.
 */
const MAX_WAIT_MS = 90 * 60 * 1000

/** Run step 8 end to end: start, poll to completion, return the report. */
export async function runTissueType(
  uploadId: string,
  options: RunOptions = {},
): Promise<TissueTypeReport> {
  const { onProgress, signal, ...rest } = options

  let run = await startTissueTypeRun(uploadId, rest)
  onProgress?.(run)

  const deadline = Date.now() + MAX_WAIT_MS

  while (run.state === 'queued' || run.state === 'running') {
    if (signal?.aborted) throw new CancelledError('the pass was stopped')
    if (Date.now() > deadline) {
      throw new Error('the pass did not finish within three hours')
    }

    await sleep(POLL_INTERVAL_MS)
    run = await fetchTissueTypeRun(uploadId, signal)
    onProgress?.(run)
  }

  // Cancelled is checked before failed, and is not one: the server reports it as
  // its own state so the caller can tell "someone stopped this" from "something
  // broke".
  if (run.state === 'cancelled') {
    throw new CancelledError(run.message ?? 'the pass was stopped')
  }
  if (run.state === 'failed') {
    throw new Error(run.error ?? 'the tissue-type pass failed')
  }
  if (run.state === 'idle') {
    throw new Error('the tissue-type pass did not start')
  }

  return fetchTissueTypeReport(uploadId, signal)
}
