/**
 * Client for step 5 - optical density.
 *
 * One request, no polling. Given step 4's white point a run is one tile read off
 * the pyramid and a handful of numpy passes; nothing is cached on disk because
 * one 512 px tile is a thousandth of the pixels steps 3 and 4 read.
 *
 * `threshold` and `percentile` belong to steps 3 and 4 and are passed straight
 * through to them. Step 5 owns neither: the viewer moves those controls on those
 * steps' screens, and the density this step reports is whatever the resulting
 * white point produces. Two places deciding what I₀ is would be one too many, and
 * I₀ is the denominator of every number here.
 *
 * `x` and `y` are step 5's own control, and the only one. Level-0 pixel
 * coordinates — the frame of reference that does not move when a resolution
 * changes — and they snap to the nearest block the chooser actually scored.
 */

import { API_PREFIX, apiGet } from './client'

import type { DensityPanelName, DensityReport } from '@/types/density'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

export interface DensityOptions {
  /** Step 3's saturation cut. Omit to use whichever rule its histogram selects. */
  threshold?: number | null
  /** Which percentile of the glass step 4 takes as I₀. Omit for its default. */
  percentile?: number | null
  /** Level-0 origin of the tile. Omit to take the block that scored highest. */
  x?: number | null
  y?: number | null
}

function query(options: DensityOptions): string {
  const params = new URLSearchParams()
  if (options.threshold !== null && options.threshold !== undefined) {
    params.set('threshold', String(Math.round(options.threshold)))
  }
  if (options.percentile !== null && options.percentile !== undefined) {
    params.set('percentile', String(options.percentile))
  }
  // Both or neither: a position with one coordinate missing is not a position,
  // and the server would silently read it as zero.
  if (
    options.x !== null &&
    options.x !== undefined &&
    options.y !== null &&
    options.y !== undefined
  ) {
    params.set('x', String(Math.max(0, Math.round(options.x))))
    params.set('y', String(Math.max(0, Math.round(options.y))))
  }

  const encoded = params.toString()
  return encoded ? `?${encoded}` : ''
}

/**
 * Run step 5 and read the result.
 *
 * A generous ceiling: the first call for a slide has to wait for steps 3 and 4 to
 * produce a mask and a white point, and the first of those may reopen a
 * multi-gigabyte scan.
 */
export function fetchDensity(
  uploadId: string,
  options: DensityOptions = {},
  signal?: AbortSignal,
): Promise<DensityReport> {
  return apiGet<DensityReport>(`/density/${uploadId}${query(options)}`, {
    signal,
    timeoutMs: 120_000,
  })
}

/**
 * URL of one panel.
 *
 * All five carry the options, because all five depend on which tile was chosen
 * and on the white point it was divided by. Each distinct choice is a distinct
 * URL, so the browser's cache works for free when the viewer goes back to a tile
 * they have already seen.
 */
export function densityPanelUrl(
  uploadId: string,
  name: DensityPanelName,
  options: DensityOptions = {},
): string {
  return url(`/density/${uploadId}/panels/${name}.png${query(options)}`)
}

/**
 * URL of one candidate block's thumbnail, for the chooser's contact sheet.
 *
 * Carries `threshold` and `percentile` but deliberately not `x` and `y`. Those
 * two decide which blocks are candidates at all — the mask says what is tissue,
 * I₀ sets the stain gate — so a thumbnail fetched under one and captioned with
 * figures from another would be a picture of a different slide. `x` and `y` are
 * the *choice*, and a choice does not change what the alternatives look like: the
 * twelve pictures are the same whichever one the viewer is standing on, so
 * leaving them off is what keeps the sheet in the browser's cache across a pick
 * instead of re-fetching all twelve every time one is clicked.
 */
export function densityCandidateUrl(
  uploadId: string,
  candidate: { col: number; row: number },
  options: DensityOptions = {},
): string {
  // Rebuilt rather than destructured-and-spread, so that a new screening option
  // added to `DensityOptions` has to be named here to reach the sheet - the
  // failure mode of the spread is the opposite, silently forwarding whatever
  // appears and putting the tile position back into the URL.
  const screening: DensityOptions = {
    threshold: options.threshold,
    percentile: options.percentile,
  }
  return url(
    `/density/${uploadId}/candidates/${candidate.col}/${candidate.row}.png${query(screening)}`,
  )
}
