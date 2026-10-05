/**
 * Thin fetch wrapper for the FastAPI backend.
 *
 * `VITE_API_BASE_URL` is empty in development, which makes every request
 * relative and lets the Vite dev-server proxy forward `/api` to uvicorn.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export const API_PREFIX = '/api/v1'

/**
 * The most specific message the server offered, or a last-resort generic one.
 *
 * Two shapes have to be read, because this API produces both. FastAPI's own errors -
 * validation, and anything raised as an `HTTPException` - carry `detail`. The app's
 * unhandled-exception handler instead returns an `ErrorDetail`: `{code, message,
 * detail}` with `detail` usually null. A client that reads only `detail` therefore
 * turns **every 500 into "Request to /x failed"** and throws away the one line that
 * says what actually broke. That is exactly how a step-8 failure reached the screen
 * with its cause discarded, so both shapes are read here, once, for every caller.
 */
export async function describeFailure(response: Response, path: string): Promise<string> {
  try {
    const parsed = (await response.json()) as {
      detail?: unknown
      message?: unknown
      code?: unknown
    }

    // `detail` can be a string, or FastAPI's list of validation errors.
    if (typeof parsed.detail === 'string' && parsed.detail) return parsed.detail
    if (Array.isArray(parsed.detail) && parsed.detail.length > 0) {
      return parsed.detail
        .map((entry) =>
          typeof entry === 'object' && entry !== null && 'msg' in entry
            ? String((entry as { msg: unknown }).msg)
            : String(entry),
        )
        .join('; ')
    }

    if (typeof parsed.message === 'string' && parsed.message) {
      // The generic 500 body. Carry the code too - it is the only thing that
      // distinguishes one internal error from another in a bug report.
      return typeof parsed.code === 'string' && parsed.code
        ? `${parsed.message} (${parsed.code}, HTTP ${response.status}). The server log has the traceback.`
        : parsed.message
    }
  } catch {
    /* non-JSON body; fall through to the generic message */
  }

  return `Request to ${path} failed (HTTP ${response.status})`
}

/** An HTTP-level failure carrying the status code for callers to branch on. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

/**
 * A long-running step was stopped on purpose.
 *
 * Its own type rather than a plain `Error`, because everything that consumes a
 * step's promise has to tell a cancellation from a failure: one is a decision the
 * viewer made and the other is a problem to report. Collapsing them would make the
 * screen apologise for what the viewer just asked for.
 */
export class CancelledError extends Error {
  constructor(message = 'the step was stopped') {
    super(message)
    this.name = 'CancelledError'
  }
}

interface RequestOptions {
  signal?: AbortSignal
  timeoutMs?: number
}

/**
 * Run a fetch under a deadline, and say so when the deadline is what stopped it.
 *
 * `controller.abort()` makes `fetch` reject with a DOMException whose message is
 * "signal is aborted without reason", and that string went straight onto the
 * screen. A viewer whose step 13 had hit its limit was told the signal was
 * aborted without reason - which names neither the step, nor the limit, nor the
 * fact that waiting longer would have worked. This distinguishes the two ways a
 * request ends early: our own deadline, and the caller cancelling.
 */
async function withDeadline<T>(
  path: string,
  timeoutMs: number,
  signal: AbortSignal | undefined,
  send: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const controller = new AbortController()
  let timedOut = false
  const timeout = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  signal?.addEventListener('abort', () => controller.abort(), { once: true })

  try {
    return await send(controller.signal)
  } catch (cause) {
    if (timedOut) {
      throw new ApiError(
        `this step took longer than ${Math.round(timeoutMs / 1000)}s and the request ` +
          `was given up on (${path}). The work itself may still be running on the ` +
          'server - going back and forward again will pick it up.',
        408,
      )
    }
    throw cause
  } finally {
    window.clearTimeout(timeout)
  }
}

/**
 * POST a JSON body to the versioned API and read the JSON back.
 *
 * Mirrors `apiGet` deliberately, down to the abort plumbing: a POST that hung would
 * leave a step looking like it was still running. `tissueType.ts` has an older local
 * helper of its own that predates this one and takes no body; new callers use this.
 */
export async function apiPost<T>(
  path: string,
  body?: unknown,
  options: RequestOptions = {},
): Promise<T> {
  const { signal, timeoutMs = 6000 } = options
  return withDeadline(path, timeoutMs, signal, async (aborter) => {
    const response = await fetch(`${BASE_URL}${API_PREFIX}${path}`, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: aborter,
    })

    if (!response.ok) {
      throw new ApiError(await describeFailure(response, path), response.status)
    }

    return (await response.json()) as T
  })
}

/**
 * PUT a JSON body to the versioned API and read the JSON back.
 *
 * Shaped like `apiPost`, and separate from it for the same reason `apiDelete` is: the
 * method is the record of intent. The one thing this app PUTs is step 10's ROI
 * selection, which *replaces* the ticked set outright — that is what makes "deselect
 * all" expressible at all, and sending it as a POST would leave the server unable to
 * tell a replacement from an addition.
 */
export async function apiPut<T>(
  path: string,
  body?: unknown,
  options: RequestOptions = {},
): Promise<T> {
  const { signal, timeoutMs = 6000 } = options
  return withDeadline(path, timeoutMs, signal, async (aborter) => {
    const response = await fetch(`${BASE_URL}${API_PREFIX}${path}`, {
      method: 'PUT',
      headers: {
        Accept: 'application/json',
        ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: aborter,
    })

    if (!response.ok) {
      throw new ApiError(await describeFailure(response, path), response.status)
    }

    return (await response.json()) as T
  })
}

/**
 * DELETE a resource and read the JSON back.
 *
 * Shaped like `apiPost` down to the abort plumbing, and separate from it for one
 * reason: the method is the record of intent. The only things this app deletes
 * are stored pipeline runs, and a delete arriving as a POST would make an
 * irreversible act indistinguishable in a log from an ordinary one.
 */
export async function apiDelete<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { signal, timeoutMs = 6000 } = options
  return withDeadline(path, timeoutMs, signal, async (aborter) => {
    const response = await fetch(`${BASE_URL}${API_PREFIX}${path}`, {
      method: 'DELETE',
      headers: { Accept: 'application/json' },
      signal: aborter,
    })

    if (!response.ok) {
      throw new ApiError(await describeFailure(response, path), response.status)
    }

    return (await response.json()) as T
  })
}

/** GET a JSON resource from the versioned API. */
export async function apiGet<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { signal, timeoutMs = 6000 } = options
  return withDeadline(path, timeoutMs, signal, async (aborter) => {
    const response = await fetch(`${BASE_URL}${API_PREFIX}${path}`, {
      headers: { Accept: 'application/json' },
      signal: aborter,
    })

    if (!response.ok) {
      throw new ApiError(await describeFailure(response, path), response.status)
    }

    return (await response.json()) as T
  })
}
