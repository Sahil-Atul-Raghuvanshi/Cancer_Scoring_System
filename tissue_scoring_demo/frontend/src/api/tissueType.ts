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
  /**
   * Step 7's committed option, sent as an assertion. The server refuses a
   * disagreement rather than running a different grid - see the endpoint's docstring.
   */
  branch?: string
  /** Step 7's committed field of view, in microns. An assertion for the same reason. */
  fov?: number
}

function query(options: TissueTypeOptions): string {
  const params = new URLSearchParams()
  if (options.overlap !== null && options.overlap !== undefined) {
    params.set('overlap', String(options.overlap))
  }
  if (options.model) params.set('model', options.model)
  if (options.branch) params.set('branch', options.branch)
  if (options.fov !== undefined && options.fov !== null) {
    params.set('fov', String(options.fov))
  }
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
 *
 * `paintedSince` asks for the patches classified since that point in the run, which is
 * what lets the screen paint the pass onto the slide as it happens. Omit it and the
 * reply carries the numbers alone: the feed is one pass over a whole slide's grid, so
 * a caller that is not drawing it should not be carrying it. The reply says how far it
 * got in `paintedCursor` — pass that back next time, rather than counting locally,
 * because a reply is capped and a client rejoining a running pass catches up over
 * several polls.
 */
export function fetchTissueTypeRun(
  uploadId: string,
  signal?: AbortSignal,
  paintedSince?: number,
): Promise<TissueTypeRun> {
  const feed = paintedSince === undefined ? '' : `?paintedSince=${paintedSince}`
  return apiGet<TissueTypeRun>(`/tissue-type/${uploadId}/run${feed}`, {
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
 * Generous, and measured rather than guessed. Two very different durations have to fit
 * under one number:
 *
 *   - a trained option is one forward pass per window: about 31 minutes for a 198 mm²
 *     IHC slide at the default overlap on four CPU threads, and 82 minutes at the
 *     finest field of view.
 *   - **BEETLE is per pixel, and that is hours.** It reads every window as nine or
 *     twenty-five overlapping 512 px patches at 0.5 µm/px, which is about 1.1 s each,
 *     so a 28 mm section measured at 10 s per window over 1,800 windows is close to
 *     five hours. That is not a tuning failure — it is a gigapixel of U-Net — and the
 *     step 7 screen prices it before anyone commits.
 *
 * So eight hours, which clears the slowest real case with room for a busier machine.
 * The control that actually matters at this duration is not the timeout: it is that the
 * viewer can stop the pass themselves at any point and that the server checks for it
 * after every window.
 */
const MAX_WAIT_MS = 8 * 60 * 60 * 1000

/** Run step 8 end to end: start, poll to completion, return the report. */
export async function runTissueType(
  uploadId: string,
  options: RunOptions = {},
): Promise<TissueTypeReport> {
  const { onProgress, signal, ...rest } = options

  let run = await startTissueTypeRun(uploadId, rest)
  onProgress?.(run)

  const deadline = Date.now() + MAX_WAIT_MS

  // Where the paint feed has been read to. The server's own count rather than a local
  // one, because a reply is capped: a client that joins a pass already half done gets
  // the backlog over several polls, and each reply says where it stopped.
  let painted = 0

  while (run.state === 'queued' || run.state === 'running') {
    if (signal?.aborted) throw new CancelledError('the pass was stopped')
    if (Date.now() > deadline) {
      throw new Error('the pass did not finish within eight hours')
    }

    await sleep(POLL_INTERVAL_MS)
    run = await fetchTissueTypeRun(uploadId, signal, painted)
    painted = run.paintedCursor
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
