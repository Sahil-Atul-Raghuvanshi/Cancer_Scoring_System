/**
 * The whole slide, pannable, with what a step found drawn on top of it.
 *
 * **Why the whole slide and not a cropped square.** Steps 11 to 15 each produce
 * something per cell, and every one of them used to be shown as a 512 px PNG of
 * one sampled field. That picture is honest about the cells and dishonest about
 * everything else: it hides where on the slide they came from, how much of the
 * tumour was looked at, and how small a cell is against the tissue it sits in. A
 * reader who has just watched step 1 pan a 63,000 px slide, and is then handed a
 * thumbnail, has no way to connect the two. So this is step 1's viewer - same
 * tile source, same controls, same real-Fullscreen-API behaviour - with a canvas
 * on top.
 *
 * **Everything is in the slide's level-0 pixels.** Step 10 warps the regions into
 * IHC level-0 pixels, step 11 stores its outlines there, and OpenSeadragon's
 * "image coordinates" are the same space. So no transform lives in this file:
 * the numbers that arrive are the numbers that get drawn, and there is no second
 * coordinate convention to keep in step.
 *
 * **A canvas, not DOM overlays.** A region holds tens of thousands of nuclei and
 * OpenSeadragon's own overlay API attaches an element each. One canvas redrawn
 * per frame, culled to the viewport, keeps panning at full speed; forty thousand
 * absolutely-positioned divs do not.
 *
 * **The shapes are painted solid, and the slider says how solid.** A translucent
 * cell over a stained slide is a colour mixed with whatever was underneath it,
 * so the same class reads as two colours in two parts of one slide and a reader
 * comparing them is comparing the tissue rather than the answer. Solid is
 * therefore the default. But a solid cell also hides the stain it was measured
 * from, and "does that outline actually sit on that nucleus" is a fair question
 * to ask of a segmentation - so the fill has a slider and the outlines never
 * fade with it. At 0 % it is the old outline-only overlay, one drag away.
 *
 * **Shapes disappear when they would be lies.** Below about five screen pixels a
 * nucleus outline is a coloured dot, and a field of coloured dots reads as a
 * heat map of something. So under that threshold the cells are not drawn at all
 * and the viewer says to zoom in - the sampled squares and the region borders
 * stay, because at that scale those are what there is to see.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import type OpenSeadragonNS from 'openseadragon'

import { API_PREFIX } from '@/api/client'
import {
  formatMicrons,
  niceScaleBar,
  useFullscreen,
} from '@/components/viewer/chrome'
import { VIEWER_ICON } from '@/components/viewer/icons'

import '@/features/pipeline/components/pipeline.css'
import './overlay.css'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

/**
 * How solid the shapes are painted, 0 to 1, remembered between steps.
 *
 * `localStorage` can throw outright - a private window, blocked site data - and
 * a viewer that will not mount because it could not read a preference is a worse
 * failure than one that opens solid. So both ends are wrapped and the default is
 * the one that needs no storage to be right.
 */
const FILL_KEY = 'demo.overlayFillOpacity'

function readStoredFill(): number {
  try {
    const raw = window.localStorage.getItem(FILL_KEY)
    if (raw === null) return 1
    const value = Number(raw)
    return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 1
  } catch {
    return 1
  }
}

function storeFill(value: number): void {
  try {
    window.localStorage.setItem(FILL_KEY, String(value))
  } catch {
    // A preference that could not be saved is not worth a broken screen.
  }
}

/** `[[x, y], …]` in the slide's level-0 pixels. */
export type Ring = number[][]

export interface OverlayPolygon {
  /** Stable across renders and unique per cell. Step 11 ids repeat between
   *  fields, so callers build this as `field:id`, never the bare id. */
  id: string
  /** Outer ring first, then holes. */
  rings: Ring[]
  stroke: string
  fill?: string | null
  /** Drawn in ascending order, so a compartment can sit under its nucleus. */
  layer?: number
  /** A dashed stroke, in screen pixels. For a shape that is shown for
   *  comparison rather than because it was used. */
  dash?: number[]
  /** Only this polygon of a multi-part shape answers a click. */
  hitTarget?: boolean
}

export interface OverlayField {
  rank: number
  index: number
  /** Top-left corner and side of the sampled square, level-0 pixels. */
  x: number
  y: number
  span: number
  polygons: OverlayPolygon[]
}

export interface OverlayRegion {
  rank: number
  rings: Ring[]
}

export interface LegendEntry {
  colour: string
  label: string
  /** Shown under the label - what this colour is actually measuring. */
  hint?: string
  count?: number
}

interface SlideOverlayViewerProps {
  /** The slide to pan. For steps 11-15 this is the IHC slide, not the H&E. */
  uploadId: string
  /** Level-0 microns per pixel, when known. Drives the scale bar and the gate. */
  mpp: number | null
  regions?: OverlayRegion[]
  fields?: OverlayField[]
  legend?: LegendEntry[]
  selectedId?: string | null
  onSelect?: (id: string | null) => void
  /** Zoom here when it changes. Level-0 pixels. */
  focus?: { x: number; y: number; span: number } | null
  /** Screen pixels a 10 µm nucleus must span before outlines are drawn. */
  minNucleusPx?: number
  /** Live count of what is on screen, published when the motion settles. */
  onVisibleChange?: (count: number) => void
  /** What the viewer is showing, in a line, under the frame. */
  caption?: ReactNode
  /** Said instead of "zoom in" when there is nothing to draw at any zoom. */
  emptyHint?: string
  className?: string
}

/**
 * A ring as `[x0, y0, x1, y1, …]`.
 *
 * Flattened once, when the shapes are indexed, rather than walked as
 * `number[][]` on every frame. Two reasons and both matter here: a pair of
 * doubles per vertex instead of a boxed array per vertex is what keeps a
 * forty-thousand-outline redraw inside a frame, and a typed array indexes to
 * `number` rather than to `number | undefined`, so the inner loop has no
 * per-vertex bounds check to write or to pay for.
 */
type FlatRing = Float64Array

interface Indexed {
  polygon: OverlayPolygon
  rings: FlatRing[]
  x0: number
  y0: number
  x1: number
  y1: number
  cx: number
  cy: number
}

interface IndexedField {
  field: OverlayField
  shapes: Indexed[]
}

interface ViewerState {
  /** Screen pixels per image pixel. */
  scale: number
  effectiveMpp: number | null
}

function flatten(rings: Ring[]): FlatRing[] {
  return rings.map((ring) => {
    const flat = new Float64Array(ring.length * 2)
    for (let i = 0; i < ring.length; i += 1) {
      const point = ring[i]
      flat[i * 2] = point?.[0] ?? 0
      flat[i * 2 + 1] = point?.[1] ?? 0
    }
    return flat
  })
}

function ringBounds(rings: FlatRing[]): { x0: number; y0: number; x1: number; y1: number } {
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const ring of rings) {
    for (let i = 0; i < ring.length; i += 2) {
      const x = ring[i]!
      const y = ring[i + 1]!
      if (x < x0) x0 = x
      if (x > x1) x1 = x
      if (y < y0) y0 = y
      if (y > y1) y1 = y
    }
  }
  return { x0, y0, x1, y1 }
}

/** Trace one ring into the current path. */
function path(
  context: CanvasRenderingContext2D,
  rings: FlatRing[],
  toX: (v: number) => number,
  toY: (v: number) => number,
): void {
  for (const ring of rings) {
    if (ring.length < 6) continue
    context.moveTo(toX(ring[0]!), toY(ring[1]!))
    for (let i = 2; i < ring.length; i += 2) context.lineTo(toX(ring[i]!), toY(ring[i + 1]!))
    context.closePath()
  }
}

/** Ray casting against the outer ring. Holes are decoration at this size. */
function insideRing(ring: FlatRing | undefined, x: number, y: number): boolean {
  if (!ring || ring.length < 6) return false
  let inside = false
  const count = ring.length / 2
  for (let i = 0, j = count - 1; i < count; j = i++) {
    const xi = ring[i * 2]!
    const yi = ring[i * 2 + 1]!
    const xj = ring[j * 2]!
    const yj = ring[j * 2 + 1]!
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

export function SlideOverlayViewer({
  uploadId,
  mpp,
  regions = [],
  fields = [],
  legend = [],
  selectedId = null,
  onSelect,
  focus = null,
  minNucleusPx = 5,
  onVisibleChange,
  caption,
  emptyHint,
  className,
}: SlideOverlayViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const viewerRef = useRef<OpenSeadragonNS.Viewer | null>(null)
  const fullscreen = useFullscreen()

  const [state, setState] = useState<ViewerState | null>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [visible, setVisible] = useState(0)
  const [gated, setGated] = useState(false)
  // Kept across steps and across reloads: a reader who turned the fill down to
  // check step 11's outlines against the stain is asking the same question of
  // step 12, and having it snap back to solid on every step would be a fight.
  const [fillAlpha, setFillAlpha] = useState(readStoredFill)

  // --- the spatial index -----------------------------------------------------
  //
  // Culling happens field-first because a field is a square of known size and
  // there are a few dozen of them, against tens of thousands of outlines. One
  // rectangle test throws away every cell in a square that is off screen.
  const index = useMemo<IndexedField[]>(
    () =>
      fields.map((field) => ({
        field,
        shapes: field.polygons.map((polygon) => {
          const rings = flatten(polygon.rings)
          const box = ringBounds(rings)
          return {
            polygon,
            rings,
            ...box,
            cx: (box.x0 + box.x1) / 2,
            cy: (box.y0 + box.y1) / 2,
          }
        }),
      })),
    [fields],
  )

  const regionRings = useMemo(
    () => regions.map((region) => ({ rank: region.rank, rings: flatten(region.rings) })),
    [regions],
  )

  // Held in refs as well so the draw loop, which is created once, always reads
  // the current data instead of the data that existed when it was created.
  const indexRef = useRef(index)
  const regionsRef = useRef(regionRings)
  const selectedRef = useRef(selectedId)
  const gateRef = useRef({ mpp, minNucleusPx })
  const fillRef = useRef(fillAlpha)
  indexRef.current = index
  regionsRef.current = regionRings
  selectedRef.current = selectedId
  gateRef.current = { mpp, minNucleusPx }
  fillRef.current = fillAlpha

  const visibleRef = useRef(0)
  const publish = useCallback(
    (count: number) => {
      if (visibleRef.current === count) return
      visibleRef.current = count
      setVisible(count)
      onVisibleChange?.(count)
    },
    [onVisibleChange],
  )

  // --- drawing ---------------------------------------------------------------

  const draw = useCallback(() => {
    const viewer = viewerRef.current
    const canvas = canvasRef.current
    if (!viewer || !canvas || !viewer.world.getItemCount()) return

    const host = viewer.element
    const width = host.clientWidth
    const height = host.clientHeight
    if (width < 2 || height < 2) return

    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
    }

    const context = canvas.getContext('2d')
    if (!context) return
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    context.clearRect(0, 0, width, height)

    const bounds = viewer.viewport.getBounds(true)
    const box = viewer.viewport.viewportToImageRectangle(bounds)
    // Screen pixels per image pixel. The one number every conversion below uses.
    const scale = width / box.width
    const toX = (x: number) => (x - box.x) * scale
    const toY = (y: number) => (y - box.y) * scale

    // --- regions: always, they are the frame of reference --------------------
    context.lineJoin = 'round'
    for (const region of regionsRef.current) {
      context.beginPath()
      path(context, region.rings, toX, toY)
      // Violet, and nothing else on these screens is violet. Step 12 paints
      // tumour cells red and step 13 paints nuclei red, so a red border round
      // the region would read as one enormous cell.
      context.strokeStyle = 'rgba(167, 139, 250, 0.95)'
      context.lineWidth = 2
      context.stroke()
      context.fillStyle = 'rgba(167, 139, 250, 0.07)'
      context.fill()
    }

    // --- the gate ------------------------------------------------------------
    const { mpp: micronsPerPixel, minNucleusPx: floor } = gateRef.current
    const nucleusPx = micronsPerPixel ? (10 / micronsPerPixel) * scale : scale * 45
    const showShapes = nucleusPx >= floor
    setGated(!showShapes)

    let count = 0
    const alpha = fillRef.current
    const selected = selectedRef.current

    for (const entry of indexRef.current) {
      const { field } = entry
      const fx0 = toX(field.x)
      const fy0 = toY(field.y)
      const side = field.span * scale
      // Off screen: one test discards every cell in it.
      if (fx0 + side < -32 || fy0 + side < -32 || fx0 > width + 32 || fy0 > height + 32) continue

      // The sampled square itself, so "only part of the region was looked at"
      // is something the reader sees rather than something they are told.
      if (side > 6) {
        context.save()
        context.setLineDash([5, 4])
        context.strokeStyle = 'rgba(56, 189, 248, 0.5)'
        context.lineWidth = 1
        context.strokeRect(fx0, fy0, side, side)
        context.restore()
      }

      if (!showShapes) continue

      let highlight: Indexed | null = null
      const ordered =
        entry.shapes.length > 1
          ? [...entry.shapes].sort((a, b) => (a.polygon.layer ?? 0) - (b.polygon.layer ?? 0))
          : entry.shapes

      for (const shape of ordered) {
        if (
          toX(shape.x1) < 0 ||
          toY(shape.y1) < 0 ||
          toX(shape.x0) > width ||
          toY(shape.y0) > height
        ) {
          continue
        }

        if (shape.polygon.id === selected) {
          highlight = shape
        }

        context.beginPath()
        path(context, shape.rings, toX, toY)

        // Only the fill takes the slider. An outline that faded with it would
        // leave the reader at 20 % with neither a solid shape nor a usable
        // boundary, which is the one setting that has to keep working.
        if (shape.polygon.fill && alpha > 0) {
          context.save()
          context.globalAlpha = alpha
          context.fillStyle = shape.polygon.fill
          context.fill('evenodd')
          context.restore()
        }
        context.strokeStyle = shape.polygon.stroke
        context.lineWidth = nucleusPx >= 18 ? 1.6 : 1
        if (shape.polygon.dash) {
          context.save()
          context.setLineDash(shape.polygon.dash)
          context.stroke()
          context.restore()
        } else {
          context.stroke()
        }

        if (shape.polygon.hitTarget !== false) count += 1
      }

      // --- the selected cell, drawn last so nothing lands on top of it -------
      if (highlight) {
        context.save()
        context.beginPath()
        path(context, highlight.rings, toX, toY)
        context.strokeStyle = '#ffffff'
        context.lineWidth = 3
        context.stroke()
        context.restore()

        // A ring around it too: at a zoom where a nucleus is twelve pixels, a
        // white outline on a white-ish nucleus is not findable on the screen.
        const radius = Math.max(14, ((highlight.x1 - highlight.x0) * scale) / 2 + 10)
        context.save()
        context.beginPath()
        context.arc(toX(highlight.cx), toY(highlight.cy), radius, 0, Math.PI * 2)
        context.strokeStyle = 'rgba(255, 255, 255, 0.85)'
        context.lineWidth = 1.5
        context.setLineDash([4, 3])
        context.stroke()
        context.restore()
      }
    }

    publish(showShapes ? count : 0)
    setState({ scale, effectiveMpp: micronsPerPixel ? micronsPerPixel / scale : null })
  }, [publish])

  /**
   * A stable handle on the current `draw`.
   *
   * The viewer is built once, in an effect, and its OpenSeadragon handlers have
   * to call whatever `draw` is current - but putting `draw` in that effect's
   * dependencies would destroy and rebuild the viewer every time it changed.
   * `draw` changes whenever `onVisibleChange` does, and a parent passing an
   * inline arrow for that would rebuild the viewer on every render: every tile
   * refetched, the reader's position lost, sixty times a second.
   *
   * So the handlers call through this, and the effect below depends only on
   * which slide is being shown.
   */
  const drawRef = useRef(draw)
  drawRef.current = draw

  // --- the viewer ------------------------------------------------------------

  useEffect(() => {
    const host = hostRef.current
    if (!host) return

    let viewer: OpenSeadragonNS.Viewer | null = null
    let cancelled = false
    let observer: ResizeObserver | null = null

    /**
     * OpenSeadragon measures its container once, at construction. Mount it
     * against a zero-width element - which is what happens if styles or fonts
     * have not settled - and it clamps the width to 1 px, so the whole slide
     * "fits" into a single pixel and nothing renders. Wait for a real size.
     */
    const start = async () => {
      const { default: OpenSeadragon } = await import('openseadragon')
      if (cancelled || host.clientWidth < 2 || host.clientHeight < 2) return

      const instance = OpenSeadragon({
        element: host,
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
        showNavigationControl: false,
        showZoomControl: false,
        showHomeControl: false,
        showFullPageControl: false,
        constrainDuringPan: true,
        visibilityRatio: 0.8,
        minZoomImageRatio: 0.6,
        maxZoomPixelRatio: 4,
        zoomPerScroll: 1.35,
        animationTime: 0.5,
        springStiffness: 8,
        // A click selects a cell here, so it must not also zoom.
        gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: true },
        immediateRender: false,
        blendTime: 0.25,
        timeout: 60_000,
        autoResize: true,
      })

      viewer = instance
      viewerRef.current = instance

      const repaint = () => drawRef.current()

      instance.addHandler('open', () => {
        setReady(true)
        setError(null)
        instance.viewport.goHome(true)
        repaint()
      })
      instance.addHandler('open-failed', () =>
        setError('The viewer could not open this slide.'),
      )
      instance.addHandler('animation', repaint)
      instance.addHandler('animation-finish', repaint)
      instance.addHandler('resize', repaint)
      instance.addHandler('update-viewport', repaint)
    }

    if (host.clientWidth >= 2 && host.clientHeight >= 2) {
      void start()
    } else {
      observer = new ResizeObserver((entries) => {
        const rect = entries[0]?.contentRect
        if (!rect || rect.width < 2 || rect.height < 2) return
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
      setReady(false)
    }
    // Only the slide. See `drawRef` above: depending on `draw` here would rebuild
    // the whole viewer whenever the overlay data or a callback prop changed.
  }, [uploadId])

  // New data for the same slide: redraw without rebuilding the viewer, which
  // would refetch every tile and throw away where the reader had panned to.
  useEffect(() => {
    draw()
  }, [draw, index, regions, selectedId, minNucleusPx, fillAlpha])

  // --- selection -------------------------------------------------------------

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || !onSelect) return

    const handler = (event: OpenSeadragonNS.CanvasClickEvent) => {
      if (!event.quick) return
      const point = viewer.viewport.pointFromPixel(event.position)
      const image = viewer.viewport.viewportToImageCoordinates(point)

      // Nearest centroid among the shapes whose box the click landed in. Nearest
      // rather than first, because compartments nest: a click inside a cell band
      // is also inside its cell, and "the one you aimed at" is the closer centre.
      let best: Indexed | null = null
      let bestDistance = Infinity
      for (const entry of indexRef.current) {
        const { field } = entry
        if (
          image.x < field.x ||
          image.y < field.y ||
          image.x > field.x + field.span ||
          image.y > field.y + field.span
        ) {
          continue
        }
        for (const shape of entry.shapes) {
          if (shape.polygon.hitTarget === false) continue
          if (
            image.x < shape.x0 ||
            image.x > shape.x1 ||
            image.y < shape.y0 ||
            image.y > shape.y1
          ) {
            continue
          }
          if (!insideRing(shape.rings[0], image.x, image.y)) continue
          const distance = (shape.cx - image.x) ** 2 + (shape.cy - image.y) ** 2
          if (distance < bestDistance) {
            bestDistance = distance
            best = shape
          }
        }
      }

      onSelect(best ? best.polygon.id : null)
    }

    viewer.addHandler('canvas-click', handler)
    return () => viewer.removeHandler('canvas-click', handler)
  }, [onSelect, ready])

  // --- focus -----------------------------------------------------------------

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || !ready || !focus) return
    // A little air around the box, and not much. At 0.6 the sampled square filled
    // under a quarter of the frame, which put a 10 um nucleus at eleven screen
    // pixels - under the threshold below which outlines are not drawn at all. So
    // flying to a region landed on the one view where its cells are invisible.
    const pad = focus.span * 0.12
    viewer.viewport.fitBounds(
      viewer.viewport.imageToViewportRectangle(
        focus.x - pad,
        focus.y - pad,
        focus.span + pad * 2,
        focus.span + pad * 2,
      ),
      false,
    )
  }, [focus, ready])

  const zoomBy = (factor: number) => {
    const viewer = viewerRef.current
    if (!viewer) return
    viewer.viewport.zoomBy(factor)
    viewer.viewport.applyConstraints()
  }

  const scaleBar =
    state?.effectiveMpp && state.effectiveMpp > 0 ? niceScaleBar(state.effectiveMpp, 110) : null

  const hasShapes = fields.some((field) => field.polygons.length > 0)

  return (
    <div className={className ? `slide-overlay ${className}` : 'slide-overlay'}>
      <div
        ref={fullscreen.ref}
        className={fullscreen.active ? 'viewer viewer--fullscreen' : 'viewer'}
      >
        <div ref={hostRef} className="viewer__canvas" />
        <canvas ref={canvasRef} className="slide-overlay__canvas" aria-hidden />

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

        {state && (
          <div className="viewer__readout mono" aria-live="off">
            <span className="viewer__stat">
              <span className="viewer__stat-key">zoom</span>
              {state.scale >= 1 ? `${state.scale.toFixed(1)}:1` : `1:${(1 / state.scale).toFixed(1)}`}
            </span>
            <span className="viewer__stat">
              <span className="viewer__stat-key">µm/px</span>
              {state.effectiveMpp ? state.effectiveMpp.toFixed(3) : '—'}
            </span>
            {state.effectiveMpp && (
              <span className="viewer__stat">
                <span className="viewer__stat-key">≈</span>
                {`${Math.round(10 / state.effectiveMpp)}×`}
              </span>
            )}
          </div>
        )}

        {scaleBar && (
          <div className="viewer__scale" aria-hidden>
            <div className="viewer__scale-bar" style={{ width: `${scaleBar.px}px` }} />
            <span className="viewer__scale-label mono">{formatMicrons(scaleBar.microns)}</span>
          </div>
        )}

        {ready && hasShapes && gated && (
          <div className="slide-overlay__gate">
            <strong>Zoom in to see the cells</strong>
            <span>
              The dashed blue squares are the parts of the tumour that were looked at. Cell
              outlines appear once one cell is big enough on screen to be worth drawing.
            </span>
          </div>
        )}

        {ready && !hasShapes && emptyHint && (
          <div className="slide-overlay__gate">
            <span>{emptyHint}</span>
          </div>
        )}

        {ready && !gated && visible > 0 && (
          <div className="slide-overlay__count">
            <strong>{visible.toLocaleString()}</strong> in view
          </div>
        )}

        {legend.length > 0 && (
          <div className="slide-overlay__legend">
            {legend.map((entry) => (
              <span key={entry.label} className="slide-overlay__legend-item" title={entry.hint}>
                <span
                  className="slide-overlay__swatch"
                  style={{ background: entry.colour }}
                  aria-hidden
                />
                {entry.label}
                {entry.count != null && (
                  <span className="slide-overlay__legend-count mono">
                    {entry.count.toLocaleString()}
                  </span>
                )}
              </span>
            ))}
          </div>
        )}

        <div className="viewer__controls">
          {hasShapes && (
            <label
              className="slide-overlay__fill"
              title="How solid the shapes are painted. Drag it down to see the stain underneath; the outlines stay put."
            >
              <span className="slide-overlay__fill-key">fill</span>
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={Math.round(fillAlpha * 100)}
                onChange={(event) => {
                  const next = Number(event.target.value) / 100
                  setFillAlpha(next)
                  storeFill(next)
                }}
                aria-label="How solid the shapes are painted"
              />
              <span className="slide-overlay__fill-value mono">
                {Math.round(fillAlpha * 100)}%
              </span>
            </label>
          )}
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
            {VIEWER_ICON.fit}
          </button>
          {fullscreen.available && (
            <button
              type="button"
              className="viewer__control--wide"
              onClick={fullscreen.toggle}
              aria-label={fullscreen.active ? 'Leave fullscreen' : 'View fullscreen'}
              aria-pressed={fullscreen.active}
              title={fullscreen.active ? 'Leave fullscreen (Esc)' : 'View fullscreen'}
            >
              {fullscreen.active ? VIEWER_ICON.exit : VIEWER_ICON.enter}
            </button>
          )}
        </div>

        <p className="viewer__hint mono">
          scroll to zoom · drag to pan{onSelect ? ' · click a cell to inspect it' : ''}
          {fullscreen.active && ' · esc to leave fullscreen'}
        </p>
      </div>

      {caption && <p className="slide-overlay__caption">{caption}</p>}
    </div>
  )
}
