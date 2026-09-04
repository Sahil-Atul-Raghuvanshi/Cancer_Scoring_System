/**
 * Client for storage housekeeping.
 *
 * **The slides are the storage.** Measured on this project's machine, `data/` is 1,542 MB
 * and one uploaded slide is 1,524 MB of it — every per-step cache together is 19 MB. So a
 * "clear the caches" button saves about one per cent and throws away the expensive part:
 * step 8's class map is half an hour of CPU, while the tissue mask under it is seconds.
 *
 * That is why `cleanup` takes a scope rather than being a single verb, and why the server
 * treats every destructive scope as a dry run unless `confirm` is passed.
 */

import { API_PREFIX, ApiError, apiGet, describeFailure } from './client'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

/** What a sweep may destroy, in increasing order. */
export type CleanupScope =
  /** Data nothing can read any more. Swept at start-up; always safe. */
  | 'orphans'
  /** The caches for a slide that is still here. Costs a re-run, not a re-upload. */
  | 'derived'
  /** One upload entirely — image, record and caches. This is the gigabytes. */
  | 'slide'
  | 'all'

export interface StorageItem {
  uploadId: string
  kind: string
  bytes: number
  /** Roughly what re-making this would take, for a cache. */
  cost: string | null
  orphaned: boolean
}

export interface UploadStorage {
  uploadId: string
  slideBytes: number
  derivedBytes: number
  derivedByStep: Record<string, number>
  orphaned: boolean
}

export interface StorageUsage {
  totalBytes: number
  slideBytes: number
  derivedBytes: number
  stagingBytes: number
  orphanBytes: number
  /** Slides over total. The number that decides whether clearing caches is worth it. */
  slideShare: number
  uploads: UploadStorage[]
  items: StorageItem[]
}

export interface CleanupResult {
  scope: CleanupScope
  /** True means nothing was deleted and `freedBytes` is what would have been. */
  dryRun: boolean
  freedBytes: number
  removed: string[]
  kept: string[]
}

export function fetchStorage(signal?: AbortSignal): Promise<StorageUsage> {
  return apiGet<StorageUsage>('/maintenance/storage', { signal, timeoutMs: 30_000 })
}

/**
 * Sweep a scope, or measure what sweeping would free.
 *
 * `confirm` defaults to false, which is a dry run. Keeping that default here as well as
 * on the server is deliberate: a caller that forgets the flag measures instead of
 * deleting, on both sides of the wire.
 */
export async function cleanup(
  scope: CleanupScope,
  options: { uploadId?: string | null; confirm?: boolean } = {},
): Promise<CleanupResult> {
  const params = new URLSearchParams({ scope })
  if (options.uploadId) params.set('uploadId', options.uploadId)
  if (options.confirm) params.set('confirm', 'true')

  const path = `/maintenance/cleanup?${params.toString()}`
  const response = await fetch(`${BASE_URL}${API_PREFIX}${path}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    throw new ApiError(await describeFailure(response, path), response.status)
  }
  return (await response.json()) as CleanupResult
}

/** `1610612736` to `1.5 GB`. Bytes are unreadable and this number is the whole argument. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['kB', 'MB', 'GB', 'TB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`
}

/* --- telling the server this page is here, and that it has gone ------------ */

export interface CleanupPolicy {
  /** Whether leaving the page schedules this slide's caches to be freed. */
  cleanupOnDisconnect: boolean
  /** How long the slide stays claimable — the window a refresh has to get back in. */
  graceSeconds: number
  /** Never `slide`: deleting a gigabyte upload on a guess is a different mistake. */
  scope: string
}

/**
 * Whether the browser should bother telling the server it is going away.
 *
 * Deliberately separate from `fetchStorage`: the page asks this on every load, and
 * `/storage` walks a multi-gigabyte tree.
 */
export function fetchCleanupPolicy(signal?: AbortSignal): Promise<CleanupPolicy> {
  return apiGet<CleanupPolicy>('/maintenance/policy', { signal, timeoutMs: 10_000 })
}

/**
 * Cancel any release scheduled for this slide.
 *
 * The call that makes a refresh survivable — a reloading page gets here long before the
 * grace period elapses, so the release its own `pagehide` just scheduled never runs.
 */
export async function claimSlide(uploadId: string): Promise<void> {
  const path = `/maintenance/claim?uploadId=${encodeURIComponent(uploadId)}`
  await fetch(`${BASE_URL}${API_PREFIX}${path}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  })
}
