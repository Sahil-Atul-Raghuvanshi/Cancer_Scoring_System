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

/** GET a JSON resource from the versioned API. */
export async function apiGet<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { signal, timeoutMs = 6000 } = options
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs)

  // Abort our request too if the caller's own signal fires.
  signal?.addEventListener('abort', () => controller.abort(), { once: true })

  try {
    const response = await fetch(`${BASE_URL}${API_PREFIX}${path}`, {
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    })

    if (!response.ok) {
      throw new ApiError(await describeFailure(response, path), response.status)
    }

    return (await response.json()) as T
  } finally {
    window.clearTimeout(timeout)
  }
}
