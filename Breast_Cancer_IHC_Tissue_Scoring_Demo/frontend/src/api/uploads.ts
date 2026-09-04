/**
 * Client half of the resumable chunked-upload protocol.
 *
 *   init -> PUT each chunk (retried with backoff) -> complete -> poll until the
 *   server-side reassembly reports ready or failed
 *
 * A whole-slide image is several gigabytes, so it never goes up as one request,
 * and a transfer can easily outlive a backend restart or a dropped connection.
 * The server remembers which parts it already holds, so recovery re-queries
 * that set and sends only what is missing rather than starting over.
 */

import { API_PREFIX, ApiError, apiGet } from './client'

import type { SlideReadout, UploadStatus } from '@/types/slide'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

/** 8 MiB, matching the backend default. */
export const DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024

/**
 * Only checksum files up to this size.
 *
 * SubtleCrypto has no streaming digest, so hashing means holding the whole file
 * in memory at once. Past this point that cost is not worth paying: the server
 * independently verifies the reassembled byte count and proves the file opens,
 * so a missing checksum is a smaller loss than an out-of-memory tab.
 */
const MAX_HASHABLE_BYTES = 64 * 1024 * 1024

const CHUNK_ATTEMPTS = 4

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/**
 * Turn a failure into something a person can act on.
 *
 * `fetch` rejects with a bare "Failed to fetch" for every network-level
 * problem, which tells the user nothing about what to do next.
 */
function describe(cause: unknown, what: string): Error {
  if (cause instanceof ApiError) return cause
  if (cause instanceof Error && /failed to fetch|networkerror|load failed/i.test(cause.message)) {
    return new Error(`Could not reach the API while ${what}. Is the backend running?`)
  }
  return cause instanceof Error ? cause : new Error(String(cause))
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(url(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body ?? {}),
  })

  if (!response.ok) {
    // The API puts the client-correctable reason in `detail`; surface it as-is
    // so the user sees "unsupported slide type '.pdf'" and not "400".
    let detail = `Request to ${path} failed`
    try {
      const parsed = (await response.json()) as { detail?: string }
      if (parsed.detail) detail = parsed.detail
    } catch {
      /* non-JSON error body; keep the generic message */
    }
    throw new ApiError(detail, response.status)
  }

  return (await response.json()) as T
}

/** Upload one part. The body is raw bytes, not multipart. */
async function putChunk(uploadId: string, index: number, blob: Blob): Promise<void> {
  const response = await fetch(url(`/uploads/${uploadId}/chunks/${index}`), {
    method: 'PUT',
    body: blob,
  })
  if (!response.ok) {
    throw new ApiError(`chunk ${index} rejected`, response.status)
  }
}

/**
 * Send one chunk, retrying a transient failure with backoff.
 *
 * A dev-server reload takes about a second, so an immediate single retry is not
 * enough to survive one — which is exactly the case this is here for. A 4xx is
 * not transient and fails straight away.
 */
async function putChunkWithRetry(uploadId: string, index: number, blob: Blob): Promise<void> {
  let last: unknown

  for (let attempt = 0; attempt < CHUNK_ATTEMPTS; attempt += 1) {
    try {
      await putChunk(uploadId, index, blob)
      return
    } catch (cause) {
      // The server rejecting the chunk is our fault, not the network's;
      // retrying an identical request will not change the answer.
      if (cause instanceof ApiError && cause.status >= 400 && cause.status < 500) throw cause
      last = cause
      await sleep(400 * 2 ** attempt) // 0.4s, 0.8s, 1.6s
    }
  }

  throw describe(last, `uploading part ${index + 1}`)
}

/** Poll status until the background reassembly finishes, or time out. */
async function pollUntilDone(uploadId: string, tries = 300): Promise<UploadStatus> {
  let networkFailures = 0

  for (let attempt = 0; attempt < tries; attempt += 1) {
    try {
      const status = await apiGet<UploadStatus>(`/uploads/${uploadId}`)
      networkFailures = 0
      if (status.state === 'ready' || status.state === 'failed') return status
    } catch (cause) {
      // Reassembling a multi-gigabyte file can outlast a brief blip; only give
      // up once polling has failed repeatedly.
      networkFailures += 1
      if (networkFailures > 10) throw describe(cause, 'waiting for the server to finish')
    }
    await sleep(1000)
  }

  throw new Error('server-side reassembly timed out')
}

export type UploadPhase = 'hashing' | 'transferring' | 'assembling'

interface UploadOptions {
  chunkSize?: number
  /** Called as the transfer advances, with a 0-1 fraction. */
  onProgress?: (fraction: number, phase: UploadPhase) => void
  signal?: AbortSignal
}

/**
 * Hash the file so the server can verify its reassembly.
 *
 * Skipped for large files and when SubtleCrypto is unavailable; the upload
 * still succeeds, it just relies on the server's size and openability checks.
 */
async function sha256(file: File): Promise<string | null> {
  if (!crypto?.subtle || file.size > MAX_HASHABLE_BYTES) return null

  try {
    const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer())
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, '0'))
      .join('')
  } catch {
    return null
  }
}

/**
 * Upload a slide and return its upload id.
 *
 * Throws with the server's own explanation if the file is rejected, a chunk
 * fails repeatedly, or the reassembly cannot open the result.
 */
export async function uploadSlide(file: File, options: UploadOptions = {}): Promise<string> {
  const { chunkSize = DEFAULT_CHUNK_SIZE, onProgress, signal } = options

  onProgress?.(0, 'hashing')
  const checksum = await sha256(file)

  const numChunks = Math.max(1, Math.ceil(file.size / chunkSize))

  let init: UploadStatus
  try {
    init = await postJson<UploadStatus>('/uploads', {
      filename: file.name,
      totalSize: file.size,
      numChunks,
      chunkSize,
      sha256: checksum,
    })
  } catch (cause) {
    throw describe(cause, 'starting the upload')
  }

  const uploadId = init.uploadId

  // The server is the authority on what it already holds. Asking first makes a
  // retried upload resume instead of re-sending parts that arrived.
  let alreadyHave = new Set<number>()
  try {
    alreadyHave = new Set((await apiGet<UploadStatus>(`/uploads/${uploadId}`)).received)
  } catch {
    /* first attempt; assume nothing has arrived yet */
  }

  onProgress?.(0, 'transferring')
  let sent = 0

  for (let index = 0; index < numChunks; index += 1) {
    if (signal?.aborted) {
      await abortUpload(uploadId).catch(() => undefined)
      throw new Error('upload cancelled')
    }

    if (!alreadyHave.has(index)) {
      await putChunkWithRetry(uploadId, index, file.slice(index * chunkSize, (index + 1) * chunkSize))
    }

    sent += 1
    onProgress?.(sent / numChunks, 'transferring')
  }

  onProgress?.(1, 'assembling')
  try {
    await postJson<UploadStatus>(`/uploads/${uploadId}/complete`)
  } catch (cause) {
    throw describe(cause, 'finishing the upload')
  }

  const done = await pollUntilDone(uploadId)
  if (done.state !== 'ready') {
    throw new Error(done.error || 'server-side reassembly failed')
  }

  return uploadId
}

/** Discard a staged upload. */
export async function abortUpload(uploadId: string): Promise<void> {
  await fetch(url(`/uploads/${uploadId}`), { method: 'DELETE' })
}

/** Current state of an upload. */
export function fetchUploadStatus(uploadId: string, signal?: AbortSignal): Promise<UploadStatus> {
  return apiGet<UploadStatus>(`/uploads/${uploadId}`, { signal })
}

/** The two adjustable inputs to step 1. */
export interface ReadoutOptions {
  /** Resolution the pipeline should work at, in microns per pixel. */
  targetMpp?: number | null
  /** The slide's own scale, for files that record none or record it wrongly. */
  mppOverride?: number | null
}

/**
 * Step 1: open the uploaded slide and read its pyramid.
 *
 * Opening a multi-gigabyte scan and walking its directory takes longer than a
 * metadata call, so this gets a much longer ceiling than the default.
 */
export function fetchSlideReadout(
  uploadId: string,
  options: ReadoutOptions = {},
  signal?: AbortSignal,
): Promise<SlideReadout> {
  const query = new URLSearchParams()
  if (options.targetMpp) query.set('targetMpp', String(options.targetMpp))
  if (options.mppOverride) query.set('mppOverride', String(options.mppOverride))

  const suffix = query.toString() ? `?${query.toString()}` : ''
  return apiGet<SlideReadout>(`/slides/${uploadId}/readout${suffix}`, {
    signal,
    timeoutMs: 60_000,
  })
}

/** URL of the whole-slide overview PNG, rendered from the pyramid. */
export function slideThumbnailUrl(uploadId: string, maxSize = 1024): string {
  return url(`/slides/${uploadId}/thumbnail?maxSize=${maxSize}`)
}

/** URL of one region of the slide as PNG. `x`/`y` are level-0 coordinates. */
export function slideRegionUrl(
  uploadId: string,
  params: { x: number; y: number; level: number; width: number; height: number },
): string {
  const query = new URLSearchParams({
    x: String(params.x),
    y: String(params.y),
    level: String(params.level),
    width: String(params.width),
    height: String(params.height),
  })
  return url(`/slides/${uploadId}/region?${query.toString()}`)
}
