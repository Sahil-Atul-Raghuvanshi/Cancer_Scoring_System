/**
 * The parts every slide viewer on this site draws the same way.
 *
 * Step 1 pans the pyramid to make "gigapixel" concrete; steps 11 to 15 pan the
 * same pyramid to show what was found on it. A viewer that behaves differently
 * between those screens teaches the reader that the two are different slides,
 * which is exactly wrong - they are one slide, looked at twice. So the controls,
 * the scale bar, the readout and the fullscreen behaviour live here once and
 * both viewers mount them.
 *
 * Fullscreen in particular has to be shared rather than re-implemented: it is a
 * real Fullscreen API call on the frame, not a CSS class that grows the panel,
 * because the readout and the scale bar have to come with it. A full-screen
 * slide with its figures left behind on the page underneath loses the only thing
 * the screen is trying to show.
 *
 * The control icons live in `icons.tsx` rather than here. This file holds no JSX
 * so that it can be a plain module: a module that mixes components with hooks and
 * helpers breaks fast refresh, and these helpers are imported by both viewers.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { MutableRefObject } from 'react'

/** A round number of microns that lands near the requested pixel width. */
export function niceScaleBar(
  micronsPerPixel: number,
  targetPx: number,
): { microns: number; px: number } {
  const raw = micronsPerPixel * targetPx
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const steps = [1, 2, 5, 10]
  const chosen = steps.map((s) => s * magnitude).find((v) => v >= raw) ?? 10 * magnitude
  return { microns: chosen, px: chosen / micronsPerPixel }
}

export function formatMicrons(microns: number): string {
  if (microns >= 1000) return `${(microns / 1000).toFixed(microns % 1000 === 0 ? 0 : 1)} mm`
  return `${microns >= 1 ? microns.toFixed(0) : microns.toFixed(2)} µm`
}

export interface FullscreenControl {
  /** Attach to the element that should fill the screen. */
  ref: MutableRefObject<HTMLDivElement | null>
  /** Whether the browser will give this frame the screen at all. */
  available: boolean
  active: boolean
  toggle: () => void
}

/**
 * Fullscreen for one frame, tracked from the browser rather than from a flag.
 *
 * Escape, the F11 key and the browser's own chrome can all leave fullscreen
 * without going through the button, so the icon has to follow
 * `document.fullscreenElement` or it ends up showing "exit" on a windowed viewer.
 *
 * Availability is checked rather than assumed: an iframe without
 * `allow="fullscreen"` reports `fullscreenEnabled === false`, and a control that
 * cannot work is worse than no control.
 */
export function useFullscreen(): FullscreenControl {
  const ref = useRef<HTMLDivElement | null>(null)
  const [available, setAvailable] = useState(false)
  const [active, setActive] = useState(false)

  useEffect(() => {
    setAvailable(
      typeof document !== 'undefined' &&
        document.fullscreenEnabled === true &&
        typeof ref.current?.requestFullscreen === 'function',
    )
  }, [])

  useEffect(() => {
    const sync = () => setActive(document.fullscreenElement === ref.current)
    document.addEventListener('fullscreenchange', sync)
    sync()
    return () => document.removeEventListener('fullscreenchange', sync)
  }, [])

  const toggle = useCallback(() => {
    const frame = ref.current
    if (!frame) return

    const request =
      document.fullscreenElement === frame
        ? document.exitFullscreen()
        : frame.requestFullscreen({ navigationUI: 'hide' })

    // Rejects when the gesture is not trusted or the permission is refused. The
    // listener above keeps the icon honest either way, so there is nothing to undo.
    void Promise.resolve(request).catch(() => setAvailable(false))
  }, [])

  return { ref, available, active, toggle }
}
