/**
 * Pan-and-zoom viewer over the slide's Deep Zoom tiles.
 *
 * This is the screen that makes "gigapixel" concrete. The image is never
 * downloaded whole - OpenSeadragon requests only the 256 px tiles currently on
 * screen, and the server renders each one on the fly from the nearest pyramid
 * level. Panning a 63,488 px slide costs a few kilobytes.
 *
 * The corner readout is the point of step 1: as you zoom, the effective
 * microns-per-pixel changes, and so does which pyramid level is doing the work.
 * Seeing those two move together is what makes the mpp-not-level-index rule
 * stick.
 *
 * Fullscreen is a real Fullscreen API call on the frame, not a CSS class that
 * grows the panel, because the readout and the scale bar have to come with it -
 * a full-screen slide with the mpp figure left behind on the page underneath
 * would lose the only thing this step is trying to show. The button hides itself
 * where the browser does not allow it rather than failing on click.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type OpenSeadragonNS from 'openseadragon'

import { API_PREFIX } from '@/api/client'
import type { PyramidLevel } from '@/types/slide'

import './pipeline.css'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

interface ViewerState {
  /** Screen pixels per image pixel. 1 means 1:1. */
  imageZoom: number
  /** Microns each screen pixel covers at the current zoom. */
  effectiveMpp: number | null
  /** Which pyramid level can serve the current zoom. */
  servingLevel: number
}

interface SlideViewerProps {
  uploadId: string
  /** Level-0 microns per pixel, when known. */
  mpp: number | null
  levels: PyramidLevel[]
  /** The level step 1 selected, highlighted when the viewer reaches it. */
  workingLevel: number
}

/** A round number of microns that lands near the requested pixel width. */
function niceScaleBar(micronsPerPixel: number, targetPx: number): { microns: number; px: number } {
  const raw = micronsPerPixel * targetPx
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const steps = [1, 2, 5, 10]
  const chosen = steps.map((s) => s * magnitude).find((v) => v >= raw) ?? 10 * magnitude
  return { microns: chosen, px: chosen / micronsPerPixel }
}

function formatMicrons(microns: number): string {
  if (microns >= 1000) return `${(microns / 1000).toFixed(microns % 1000 === 0 ? 0 : 1)} mm`
  return `${microns >= 1 ? microns.toFixed(0) : microns.toFixed(2)} µm`
}

/**
 * The three control icons, drawn rather than set in type.
 *
 * "Fit the whole slide" and "go fullscreen" are different actions that both mean
 * "bigger view" to a reader skimming glyphs, and a single ⤢ next to a + and a −
 * gets read as fullscreen every time. So the fit control shows a frame with the
 * slide sitting inside it, and the fullscreen pair shows corner brackets - opening
 * outwards to enter, inwards to leave.
 */
const ICON = {
  fit: (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden focusable="false">
      <rect
        x="1.6"
        y="2.6"
        width="12.8"
        height="10.8"
        rx="1.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        opacity="0.55"
      />
      <rect x="5.2" y="6.2" width="5.6" height="3.6" rx="1" fill="currentColor" />
    </svg>
  ),
  enter: (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden focusable="false">
      <path
        d="M2.6 6.2V2.6h3.6M13.4 6.2V2.6H9.8M2.6 9.8v3.6h3.6M13.4 9.8v3.6H9.8"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  ),
  exit: (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden focusable="false">
      <path
        d="M6.2 2.6v3.6H2.6M9.8 2.6v3.6h3.6M6.2 13.4V9.8H2.6M9.8 13.4V9.8h3.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  ),
} as const

export function SlideViewer({ uploadId, mpp, levels, workingLevel }: SlideViewerProps) {
  const frameRef = useRef<HTMLDivElement | null>(null)
  const hostRef = useRef<HTMLDivElement | null>(null)
  const viewerRef = useRef<OpenSeadragonNS.Viewer | null>(null)

  const [state, setState] = useState<ViewerState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)
  /**
   * Whether the browser will give this frame the screen at all.
   *
   * Checked rather than assumed: an iframe without `allow="fullscreen"` reports
   * `fullscreenEnabled === false`, and a control that cannot work is worse than no
   * control. Read once on mount, since it does not change while the page is open.
   */
  const [canFullscreen, setCanFullscreen] = useState(false)

  /** The finest pyramid level whose downsample is still at-or-below what is needed. */
  const levelFor = useCallback(
    (downsample: number) => {
      let best = 0
      for (const level of levels) {
        if (level.downsample <= downsample + 1e-6) best = level.level
      }
      return best
    },
    [levels],
  )

  useEffect(() => {
    const host = hostRef.current
    if (!host) return

    let viewer: OpenSeadragonNS.Viewer | null = null
    let cancelled = false
    let observer: ResizeObserver | null = null

    /**
     * OpenSeadragon measures its container once, at construction, and derives
     * every zoom from that. Mount it against a zero-width element - which is
     * what happens if styles or fonts have not settled - and it clamps the
     * width to 1 px, so the whole slide "fits" into a single pixel and nothing
     * renders. Wait for a real size first.
     */
    const start = async () => {
      // Loaded on demand: it is ~250 kB, is only needed once a slide has been
      // read, and touches `document` the moment it is imported.
      const { default: OpenSeadragon } = await import('openseadragon')
      if (cancelled || host.clientWidth < 2 || host.clientHeight < 2) return

      const instance = OpenSeadragon({
        element: host,
        // Bundled with the app rather than pulled from a CDN, so the viewer works
        // offline and cannot break when a third party changes a URL.
        prefixUrl: '',
        tileSources: `${BASE_URL}${API_PREFIX}/slides/${uploadId}.dzi`,
        crossOriginPolicy: false,
        showNavigator: true,
        navigatorPosition: 'TOP_RIGHT',
        navigatorHeight: 92,
        navigatorWidth: 92,
        navigatorAutoFade: false,
        navigatorBorderColor: '#33415a',
        navigatorDisplayRegionColor: '#38bdf8',
        // The default UI is a row of sprite images; this app supplies its own
        // controls, so the built-in buttons are turned off.
        showNavigationControl: false,
        showZoomControl: false,
        showHomeControl: false,
        showFullPageControl: false,
        // A slide has no meaningful "outside", so keep it from drifting away.
        constrainDuringPan: true,
        visibilityRatio: 0.8,
        minZoomImageRatio: 0.6,
        // Past 1:1 there is no more detail to fetch - anything further is just
        // magnified pixels, so allow a little for inspection and no more.
        maxZoomPixelRatio: 4,
        zoomPerScroll: 1.35,
        animationTime: 0.5,
        springStiffness: 8,
        gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
        immediateRender: false,
        blendTime: 0.25,
        timeout: 60_000,
        // Keep tracking the container: the panel around it can reflow.
        autoResize: true,
      })

      viewer = instance
      viewerRef.current = instance

      const report = () => {
        const zoom = instance.viewport.getZoom(true)
        const imageZoom = instance.viewport.viewportToImageZoom(zoom)
        // A degenerate container yields a nonsense zoom; do not publish it.
        if (!Number.isFinite(imageZoom) || imageZoom <= 0) return

        setState({
          imageZoom,
          effectiveMpp: mpp ? mpp / imageZoom : null,
          servingLevel: levelFor(1 / imageZoom),
        })
      }

      instance.addHandler('open', () => {
        setReady(true)
        setError(null)
        instance.viewport.goHome(true)
        report()
      })
      instance.addHandler('open-failed', () =>
        setError('The viewer could not open this slide.'),
      )
      instance.addHandler('animation', report)
      instance.addHandler('zoom', report)
      instance.addHandler('resize', report)
    }

    if (host.clientWidth >= 2 && host.clientHeight >= 2) {
      void start()
    } else {
      // Not laid out yet - build it the moment it has real dimensions.
      observer = new ResizeObserver((entries) => {
        const box = entries[0]?.contentRect
        if (!box || box.width < 2 || box.height < 2) return
        observer?.disconnect()
        observer = null
        void start()
      })
      observer.observe(host)
    }

    return () => {
      cancelled = true
      observer?.disconnect()
      viewer?.destroy()
      viewerRef.current = null
    }
  }, [levelFor, mpp, uploadId])

  useEffect(() => {
    setCanFullscreen(
      typeof document !== 'undefined' &&
        document.fullscreenEnabled === true &&
        typeof frameRef.current?.requestFullscreen === 'function',
    )
  }, [])

  /**
   * Track the browser's own view of fullscreen rather than a flag of our own.
   *
   * Escape, the F11 key and the browser's chrome can all leave fullscreen without
   * going through the button, so the icon has to follow `document.fullscreenElement`
   * or it ends up showing "exit" on a windowed viewer.
   */
  useEffect(() => {
    const sync = () => setFullscreen(document.fullscreenElement === frameRef.current)
    document.addEventListener('fullscreenchange', sync)
    sync()
    return () => document.removeEventListener('fullscreenchange', sync)
  }, [])

  const toggleFullscreen = useCallback(() => {
    const frame = frameRef.current
    if (!frame) return

    const request =
      document.fullscreenElement === frame
        ? document.exitFullscreen()
        : frame.requestFullscreen({ navigationUI: 'hide' })

    // Rejects when the gesture is not trusted or the permission is refused. The
    // listener above keeps the icon honest either way, so there is nothing to undo.
    void Promise.resolve(request).catch(() => setCanFullscreen(false))
  }, [])

  const zoomBy = (factor: number) => {
    const viewer = viewerRef.current
    if (!viewer) return
    viewer.viewport.zoomBy(factor)
    viewer.viewport.applyConstraints()
  }

  const scaleBar =
    state?.effectiveMpp && state.effectiveMpp > 0
      ? niceScaleBar(state.effectiveMpp, 110)
      : null

  return (
    <div
      ref={frameRef}
      className={fullscreen ? 'viewer viewer--fullscreen' : 'viewer'}
    >
      <div ref={hostRef} className="viewer__canvas" />

      {!ready && !error && (
        <div className="viewer__overlay">
          <span className="viewer__spinner" aria-hidden />
          <span className="viewer__overlay-text">Opening the slide…</span>
        </div>
      )}

      {error && (
        <div className="viewer__overlay">
          <span className="viewer__overlay-text viewer__overlay-text--error">{error}</span>
        </div>
      )}

      {/* --- live readout: the whole point of step 1 --------------------- */}
      {state && (
        <div className="viewer__readout mono" aria-live="off">
          <span className="viewer__stat">
            <span className="viewer__stat-key">zoom</span>
            {state.imageZoom >= 1
              ? `${state.imageZoom.toFixed(1)}:1`
              : `1:${(1 / state.imageZoom).toFixed(1)}`}
          </span>
          <span className="viewer__stat">
            <span className="viewer__stat-key">µm/px</span>
            {state.effectiveMpp ? state.effectiveMpp.toFixed(3) : '—'}
          </span>
          <span
            className={
              state.servingLevel === workingLevel
                ? 'viewer__stat viewer__stat--match'
                : 'viewer__stat'
            }
          >
            <span className="viewer__stat-key">level</span>
            {state.servingLevel}
          </span>
        </div>
      )}

      {/* --- scale bar ---------------------------------------------------- */}
      {scaleBar && (
        <div className="viewer__scale" aria-hidden>
          <div className="viewer__scale-bar" style={{ width: `${scaleBar.px}px` }} />
          <span className="viewer__scale-label mono">{formatMicrons(scaleBar.microns)}</span>
        </div>
      )}

      {/* --- controls ----------------------------------------------------- */}
      <div className="viewer__controls">
        <button type="button" onClick={() => zoomBy(1.6)} aria-label="Zoom in" title="Zoom in">
          +
        </button>
        <button
          type="button"
          onClick={() => zoomBy(1 / 1.6)}
          aria-label="Zoom out"
          title="Zoom out"
        >
          −
        </button>
        <button
          type="button"
          onClick={() => viewerRef.current?.viewport.goHome()}
          aria-label="Fit the whole slide in the frame"
          title="Fit whole slide"
        >
          {ICON.fit}
        </button>
        {canFullscreen && (
          <button
            type="button"
            className="viewer__control--wide"
            onClick={toggleFullscreen}
            aria-label={fullscreen ? 'Leave fullscreen' : 'View fullscreen'}
            aria-pressed={fullscreen}
            title={fullscreen ? 'Leave fullscreen (Esc)' : 'View fullscreen'}
          >
            {fullscreen ? ICON.exit : ICON.enter}
          </button>
        )}
      </div>

      <p className="viewer__hint mono">
        scroll to zoom · drag to pan · double-click to zoom in
        {fullscreen && ' · esc to leave fullscreen'}
      </p>
    </div>
  )
}
