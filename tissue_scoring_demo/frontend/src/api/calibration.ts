/**
 * Client for step 4 - white calibration.
 *
 * One request, no polling. Given step 3's mask a run is a thumbnail read and a
 * handful of numpy passes; the server caches the thumbnail, so a re-run after
 * step 3's threshold moves is the arithmetic only.
 *
 * `threshold` here is *step 3's* cut, passed straight through. Step 4 has no
 * threshold of its own: the viewer moves one slider, on step 3's screen, and the
 * glass this step samples is whatever mask that produced. So this client takes it
 * as an option rather than pretending it belongs to step 4.
 */

import { API_PREFIX, apiGet } from './client'

import type {
  CalibrationComparison,
  CalibrationPanelName,
  CalibrationReport,
} from '@/types/calibration'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

function url(path: string): string {
  return `${BASE_URL}${API_PREFIX}${path}`
}

export interface CalibrationOptions {
  /** Step 3's saturation cut. Omit to use whichever rule its histogram selects. */
  threshold?: number | null
  /** Which percentile of the glass is I₀. Omit for the server's 95th. */
  percentile?: number | null
}

function query(options: CalibrationOptions): string {
  const params = new URLSearchParams()
  if (options.threshold !== null && options.threshold !== undefined) {
    params.set('threshold', String(Math.round(options.threshold)))
  }
  if (options.percentile !== null && options.percentile !== undefined) {
    params.set('percentile', String(options.percentile))
  }

  const encoded = params.toString()
  return encoded ? `?${encoded}` : ''
}

/**
 * Run step 4 and read the result.
 *
 * A generous ceiling: the first call for a slide reopens a multi-gigabyte scan to
 * read its thumbnail, and step 3's mask has to exist before this can start.
 */
export function fetchCalibration(
  uploadId: string,
  options: CalibrationOptions = {},
  signal?: AbortSignal,
): Promise<CalibrationReport> {
  return apiGet<CalibrationReport>(`/calibration/${uploadId}${query(options)}`, {
    signal,
    timeoutMs: 120_000,
  })
}

/**
 * URL of one panel.
 *
 * All four carry the options, unlike step 3's, because all four depend on which
 * pixels are glass and that depends on step 3's cut. Each distinct cut is a
 * distinct URL, so the browser's cache works for free when the viewer goes back
 * to a value they have already seen.
 */
export function calibrationPanelUrl(
  uploadId: string,
  name: CalibrationPanelName,
  options: CalibrationOptions = {},
): string {
  return url(`/calibration/${uploadId}/panels/${name}.png${query(options)}`)
}

/** URL of the I₀ swatch - the artefact to put next to another slide's. */
export function calibrationSwatchUrl(
  uploadId: string,
  options: CalibrationOptions = {},
): string {
  return url(`/calibration/${uploadId}/swatch.png${query(options)}`)
}

/**
 * Two or more slides' white points, and the OD cost of confusing them.
 *
 * The step's argument rather than its output: one slide's I₀ in isolation is a
 * number, and two slides' side by side is the reason it has to be measured per
 * slide.
 */
export function fetchCalibrationComparison(
  uploadIds: string[],
  signal?: AbortSignal,
): Promise<CalibrationComparison> {
  const ids = encodeURIComponent(uploadIds.join(','))
  return apiGet<CalibrationComparison>(`/calibration/compare?uploadIds=${ids}`, {
    signal,
    timeoutMs: 180_000,
  })
}
