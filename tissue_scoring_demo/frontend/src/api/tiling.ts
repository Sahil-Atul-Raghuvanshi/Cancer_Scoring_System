/**
 * Client for step 7 — tiling.
 *
 * One request, no polling. The index is one pass over the grid against a mask
 * step 3 has already cached.
 *
 * `threshold` belongs to step 3 and is passed straight through: it decides what
 * is tissue, and what is tissue decides which tiles survive.
 *
 * `fov` and `overlap` are step 7's own two controls, and they differ in kind. The
 * overlap moves only the price. The **field of view** moves which checkpoint step 8
 * runs, because a field of view is a property of the weights — so it is the one
 * control on this screen whose effect is a different model rather than a different
 * bill. The two gates are thresholds whose values are arguments rather than
 * preferences, and the tile size is the chosen model's own.
 */

import { API_PREFIX, apiGet, apiPost } from './client'

import type {
  TilingBranchName,
  TilingBranches,
  TilingPanelName,
  TilingReport,
  TilingSelection,
} from '@/types/tiling'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

export interface TilingOptions {
  /** Step 3's saturation cut. Omit to use whichever rule its histogram selects. */
  threshold?: number | null
  /** Fraction of its extent a tile shares with its neighbour. Omit for the default. */
  overlap?: number | null
  /**
   * Microns of slide across one square. One of the offered values — anything else is
   * refused rather than snapped, because this picks the checkpoint. Omit for the default.
   */
  fov?: number | null
  /**
   * Which of step 7's options to lay the grid for. Part of every panel URL as well as
   * the report's, because the two branches show the model different things — a cached
   * URL that ignored it would hand the H&E screen a picture of a deconvolved density
   * plane its model never sees.
   */
  branch?: TilingBranchName | null
}

function query(options: TilingOptions): string {
  const params = new URLSearchParams()
  if (options.threshold !== null && options.threshold !== undefined) {
    params.set('threshold', String(Math.round(options.threshold)))
  }
  if (options.overlap !== null && options.overlap !== undefined) {
    params.set('overlap', String(options.overlap))
  }
  if (options.fov !== null && options.fov !== undefined) {
    params.set('fov', String(options.fov))
  }
  if (options.branch !== null && options.branch !== undefined) {
    params.set('branch', options.branch)
  }

  const encoded = params.toString()
  return encoded ? `?${encoded}` : ''
}

/**
 * Build the tile index and read the result.
 *
 * A generous ceiling: on a cold cache this waits for step 3's mask behind it,
 * which may reopen a multi-gigabyte scan.
 */
export function fetchTiling(
  uploadId: string,
  options: TilingOptions = {},
  signal?: AbortSignal,
): Promise<TilingReport> {
  return apiGet<TilingReport>(`/tiling/${uploadId}${query(options)}`, {
    signal,
    timeoutMs: 120_000,
  })
}

/**
 * URL of one panel.
 *
 * Both carry the options, because both depend on the mask the index was built
 * against and on the field of view and overlap that set the grid. Each distinct
 * choice is a distinct URL, so the browser's cache works for free when the viewer
 * moves back to a setting they have already seen.
 */
export function tilingPanelUrl(
  uploadId: string,
  name: TilingPanelName,
  options: TilingOptions = {},
): string {
  return url(`/tiling/${uploadId}/panels/${name}.png${query(options)}`)
}

/**
 * What this slide can be shown as, and why an option is unavailable.
 *
 * Fetched before any grid exists, which is why it is a separate call: the branch screen
 * has to say which options are open — and explain the closed one — without reading a
 * single slide pixel. Published manifests and step 5's cached verdict, nothing else.
 */
export function fetchTilingBranches(
  uploadId: string,
  signal?: AbortSignal,
): Promise<TilingBranches> {
  return apiGet<TilingBranches>(`/tiling/${uploadId}/branches`, { signal })
}

export interface TilingSelectionBody {
  branch: TilingBranchName
  fov: number
  overlap?: number | null
  threshold?: number | null
}

/**
 * Commit the choice step 8 will run on.
 *
 * A POST, and deliberately not a side effect of reading a report: this record is what
 * step 8, step 9 and the runner rebuild their grid from, so writing it from a GET would
 * make "the viewer glanced at 672 µm" indistinguishable from "the viewer chose it" —
 * and every panel request carries the same query string, so each one would rewrite it.
 */
export function commitTilingSelection(
  uploadId: string,
  body: TilingSelectionBody,
  signal?: AbortSignal,
): Promise<TilingSelection> {
  return apiPost<TilingSelection>(`/tiling/${uploadId}/selection`, body, {
    signal,
    timeoutMs: 120_000,
  })
}
