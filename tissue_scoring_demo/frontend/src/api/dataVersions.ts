/**
 * Client for the versioned data tree — `storage/v1_data/`, `storage/v2_data/`, ….
 *
 * The backend serves exactly one version for its whole life, so switching restarts it.
 * `selectDataVersion` asks for the switch; `waitForDataVersion` polls until the restarted
 * backend reports the new version (or gives up, when it was not started with --reload).
 */

import { apiGet, apiPost } from './client'

export interface DataVersionInfo {
  version: string
  path: string
  /** The storage layout its version.json records. */
  layout: number
  /** False when this code can no longer read it; its data is kept, just not opened. */
  openable: boolean
  notOpenableReason: string | null
  /** False when this version scores with an earlier version's models. */
  hasOwnModels: boolean
  modelsPath: string
  /** What this version is, from its VERSION.md. Empty when it has none. */
  summary: string
}

export interface DataVersions {
  /** The version this backend process reads and writes. */
  active: string
  /** The newest version this code can open. */
  latest: string | null
  /** What was asked for when the server started, if anything. */
  requested: string | null
  /** Why `requested` could not be opened and `active` runs instead. */
  fallbackReason: string | null
  /** CSS_DATA_VERSION is set, so the picker cannot change the version. */
  pinnedByEnv: boolean
  sharedDataPath: string
  versions: DataVersionInfo[]
}

export function fetchDataVersions(): Promise<DataVersions> {
  return apiGet<DataVersions>('/data-versions')
}

export function selectDataVersion(
  version: string,
): Promise<{ selected: string; restarting: boolean }> {
  return apiPost('/data-versions/select', { version })
}

export function createDataVersion(): Promise<DataVersionInfo> {
  return apiPost<DataVersionInfo>('/data-versions')
}

/** Resolves true once the backend serves `version`, false after `timeoutMs`. */
export async function waitForDataVersion(version: string, timeoutMs = 45000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  // The old process answers for a moment before the reloader kills it.
  await new Promise((resolve) => setTimeout(resolve, 1500))
  while (Date.now() < deadline) {
    try {
      const current = await fetchDataVersions()
      if (current.active === version) return true
    } catch {
      /* restarting — not answering yet */
    }
    await new Promise((resolve) => setTimeout(resolve, 1000))
  }
  return false
}
