/**
 * A segmentation pass, painting itself onto a picture.
 *
 * Two steps watch a model answer window by window — step 8 over the whole section, step
 * 11 over one chosen region at a time — and they draw the identical thing: each window
 * appears when the model answers for it and not before, in the class colour it was
 * given, or as the little pixel mask BEETLE produced inside it. This is that drawing,
 * with the two things that differ between the steps taken as props: **which part of the
 * slide the canvas shows**, and **what picture is underneath it**.
 *
 * **Nothing here is an animation of progress.** A square is a window that has been
 * through the network. The sweep stalls when the machine is busy and a band of glass
 * crosses faster than a band of tumour, because those are properties of the run —
 * hiding them behind a smooth animation would make this a decoration rather than a
 * readout.
 *
 * **Two kinds of patch, chosen per patch.** A window that arrives with a mask is drawn
 * as that mask; one that arrives without is a `fillRect` in its dominant class's
 * colour. Decided by whether a mask came, never by a mode flag, so a feed briefly
 * missing one degrades to a flat colour rather than to nothing. On the tile branches of
 * step 8 no masks are ever sent and every patch takes the flat path; on BEETLE — all of
 * step 11 — every patch is a mask, which is why step 11's paint shows duct outlines
 * forming rather than a grid filling in.
 *
 * **Two canvases, and that is the whole trick.** The lower one accumulates and is never
 * cleared, so painting costs one draw per new window however long the pass runs. The
 * upper one holds the leading edge — where the model is reading right now — and is the
 * only thing redrawn per update.
 *
 * **The geometry is the window grid's, in level-0 slide pixels**, which is what lets one
 * component serve both steps. A window's core is at `x_of(col)`, clamped at the last row
 * and column exactly as `WindowGrid` clamps it, and `view` then says which rectangle of
 * the slide the canvas is showing. Step 8 passes the whole slide; step 11 passes one
 * region's padded box, because `crop.restrict` narrows step 8's grid rather than
 * building a new one and a painted window's `(row, col)` is therefore still an index
 * into the slide-wide grid. Renumbering per region would mean reimplementing that clamp
 * in two places.
 *
 * A patch is drawn at its `stride`-wide core rather than its full `span`, because the
 * cores partition the tissue and the spans overlap — painting spans would overprint
 * every neighbour at 50% overlap and turn the map into mud.
 */

import { useEffect, useRef, type ReactNode } from 'react'

/** Backing-store width of the canvases. CSS sizes them, so this is resolution. */
const CANVAS_WIDTH = 1200

/**
 * How strongly a patch covers the picture underneath. The finished panels' own figure
 * on both steps, so the live paint and the result read as one picture rather than as
 * two intensities of the same colours.
 */
const PAINT_ALPHA = 0.55

/** Where the window grid lives, in level-0 slide pixels. */
export interface PaintFrame {
  /** Level-0 slide size — what the grid's clamp is against, never the canvas's size. */
  slideWidth: number
  slideHeight: number
  /** Level-0 pixels one window covers. */
  span: number
  /** Level-0 pixels between windows, and the side of the cell a window owns. */
  stride: number
}

/** Which rectangle of the slide the canvas is showing, in level-0 pixels. */
export interface PaintView {
  x: number
  y: number
  width: number
  height: number
}

/** Where a cell lands on the canvas, in canvas pixels. */
function cellRect(
  frame: PaintFrame,
  view: PaintView,
  row: number,
  col: number,
  width: number,
  height: number,
) {
  // The clamp is `WindowGrid.x_of`'s: the last row and column of the grid do not sit at
  // `index * stride`, because a window there would ask for pixels past the slide and be
  // read at the edge instead. Without it the final band of every slide would paint
  // slightly outside the section.
  const inset = (frame.span - frame.stride) / 2
  const x0 =
    Math.min(col * frame.stride, Math.max(0, frame.slideWidth - frame.span)) + inset
  const y0 =
    Math.min(row * frame.stride, Math.max(0, frame.slideHeight - frame.span)) + inset

  const sx = width / view.width
  const sy = height / view.height

  // Rounded on both edges rather than rounding an origin and a width, so neighbouring
  // cells share an edge exactly and the paint has no seams through it.
  const left = Math.round((x0 - view.x) * sx)
  const top = Math.round((y0 - view.y) * sy)
  return {
    left,
    top,
    width: Math.max(1, Math.round((x0 + frame.stride - view.x) * sx) - left),
    height: Math.max(1, Math.round((y0 + frame.stride - view.y) * sy) - top),
  }
}

/**
 * `#rrggbb` to the `[r, g, b]` a canvas ImageData wants.
 *
 * The palette arrives as hex because that is what CSS and `fillStyle` take, and the
 * flat-colour path uses it directly; only the mask path needs the bytes.
 */
function rgbOf(hex: string): [number, number, number] {
  const value = Number.parseInt(hex.replace('#', ''), 16)
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255]
}

/**
 * One window's pixel mask, drawn into its cell.
 *
 * The mask is `side` squared **class ids**, one byte each, base64'd — not an image. So
 * it is decoded into an `ImageData` of `side` squared pixels here and then scaled into
 * the cell by `drawImage`, which is the one call in the browser that will resample it
 * for free. Going via a small offscreen canvas rather than writing the cell's pixels
 * directly is what makes that scaling the browser's problem — and the scale differs by
 * step: a cell is a dozen pixels a side on step 8's whole-slide grid and the mask is 32,
 * so that is a *down*scale, while on step 11's single region a cell is far larger than
 * the mask and it is an upscale. Neither needs a resampling rule chosen here.
 *
 * `imageSmoothingEnabled` is off. Interpolating between two class ids' colours would
 * invent a colour that means nothing — a blend of "in-situ blue" and "stroma yellow" is
 * not a class — and it would soften exactly the duct edges this exists to show.
 *
 * Pixels whose class has no colour are left transparent rather than filled: that is how
 * a switched-off class and an unclaimed pixel both read as "the picture underneath,
 * showing through", the same convention the finished panels use.
 */
function drawMask(
  board: CanvasRenderingContext2D,
  scratch: HTMLCanvasElement,
  bytes: Uint8Array,
  side: number,
  palette: [number, number, number][],
  cell: { left: number; top: number; width: number; height: number },
) {
  if (scratch.width !== side || scratch.height !== side) {
    scratch.width = side
    scratch.height = side
  }
  const context = scratch.getContext('2d')
  if (!context) return

  const image = context.createImageData(side, side)
  const data = image.data
  for (let index = 0; index < side * side; index += 1) {
    // `bytes[index]` is `number | undefined` under noUncheckedIndexedAccess even though
    // the loop bound keeps it in range. Class 0 is the fallback, and the
    // `colour === undefined` test below still handles a class with no colour.
    const colour = palette[bytes[index] ?? 0]
    const at = index * 4
    if (colour === undefined) {
      data[at + 3] = 0
      continue
    }
    data[at] = colour[0]
    data[at + 1] = colour[1]
    data[at + 2] = colour[2]
    data[at + 3] = 255
  }
  context.putImageData(image, 0, 0)

  board.imageSmoothingEnabled = false
  board.drawImage(scratch, 0, 0, side, side, cell.left, cell.top, cell.width, cell.height)
}

/** Size a canvas to the view's aspect. Returns true when that cleared it. */
function fit(canvas: HTMLCanvasElement, view: PaintView): boolean {
  const height = Math.max(1, Math.round((CANVAS_WIDTH * view.height) / view.width))
  if (canvas.width === CANVAS_WIDTH && canvas.height === height) return false

  // Assigning either dimension resets the bitmap, which is exactly what is wanted the
  // first time and is why the caller is told about it.
  canvas.width = CANVAS_WIDTH
  canvas.height = height
  return true
}

export interface PaintCanvasProps {
  /** The window grid, in level-0 slide pixels. */
  frame: PaintFrame
  /** The rectangle of the slide this canvas shows, in the same pixels. */
  view: PaintView
  /** Hex colour per class id. Indexes into it are the classes in `painted`. */
  colours: string[]
  /** Whether patches carry pixel masks. False means every patch is a flat colour. */
  perPixel: boolean
  /** Side of each mask in `paintedMasks`, in pixels. Zero when `perPixel` is false. */
  maskPx: number
  /** The accumulated feed: flat `row, col, class` triples. Stable identity. */
  painted: number[]
  /** One base64 mask per patch in `painted`, same order. Stable identity. */
  paintedMasks: string[]
  /** How many patches of `painted` are filled in. What changes as the pass runs. */
  paintedCount: number
  /**
   * Identity of what is being painted. A different value means the canvas is showing
   * something else — a restarted run on step 8, the next region on step 11 — and it
   * starts again rather than drawing the new feed on top of the old paint.
   */
  runKey: string
  /** The picture the paint lands on. A thumbnail, or one region's crop. */
  children: ReactNode
  className?: string
}

export function PaintCanvas({
  frame,
  view,
  colours,
  perPixel,
  maskPx,
  painted,
  paintedMasks,
  paintedCount,
  runKey,
  children,
  className,
}: PaintCanvasProps) {
  const paintLayer = useRef<HTMLCanvasElement | null>(null)
  const edgeLayer = useRef<HTMLCanvasElement | null>(null)
  // One offscreen canvas for the whole pass, reused per patch. Creating one per patch
  // would be thousands of allocations over a pass that already has enough to do.
  const scratch = useRef<HTMLCanvasElement | null>(null)

  /** How far into the feed this canvas has drawn. The paint layer is never redrawn. */
  const drawn = useRef(0)
  const key = useRef<string | null>(null)

  useEffect(() => {
    const surface = paintLayer.current
    const edge = edgeLayer.current
    if (!surface || !edge) return

    const board = surface.getContext('2d')
    const marker = edge.getContext('2d')
    if (!board || !marker) return

    // Both, and not `a || b`: short-circuiting would leave the edge layer at its default
    // 300x150 the first time through, and its outline would then be drawn in a
    // coordinate system that has nothing to do with the paint under it.
    const surfaceResized = fit(surface, view)
    const edgeResized = fit(edge, view)
    const resized = surfaceResized || edgeResized

    // A different thing being painted, a resized canvas, or a feed that went backwards
    // — each means what is on screen belongs to something else.
    if (resized || key.current !== runKey || paintedCount < drawn.current) {
      board.clearRect(0, 0, surface.width, surface.height)
      marker.clearRect(0, 0, edge.width, edge.height)
      key.current = runKey
      drawn.current = 0
    }

    const from = drawn.current
    if (paintedCount <= from) return

    board.globalAlpha = PAINT_ALPHA
    let edgeLeft = Infinity
    let edgeTop = Infinity
    let edgeRight = -Infinity
    let edgeBottom = -Infinity

    // Decoded once per render rather than per patch, because every patch of a per-pixel
    // pass needs the same handful of entries as bytes.
    const bytePalette = colours.map(rgbOf)
    if (!scratch.current) scratch.current = document.createElement('canvas')

    for (let index = from; index < paintedCount; index += 1) {
      // Indexing a number[] yields `number | undefined` under noUncheckedIndexedAccess.
      // `paintedCount` already bounds the loop to whole triples, so these are always
      // present; 0 is a fallback the compiler needs rather than a case that occurs.
      const row = painted[index * 3] ?? 0
      const col = painted[index * 3 + 1] ?? 0
      const label = painted[index * 3 + 2] ?? 0
      const cell = cellRect(frame, view, row, col, surface.width, surface.height)

      // A window whose core falls outside the view still ran — step 11 segments every
      // window whose core reaches the padded box, and a box clipped to the slide can
      // leave one just off the edge. Skipping it here keeps it out of the leading-edge
      // rectangle, which would otherwise be stretched by a patch nobody can see.
      if (
        cell.left + cell.width <= 0 ||
        cell.top + cell.height <= 0 ||
        cell.left >= surface.width ||
        cell.top >= surface.height
      ) {
        continue
      }

      // The mask if this patch brought one, the flat colour otherwise. Decided per patch
      // and not per pass, so a feed briefly missing a mask draws the window's dominant
      // class instead of skipping it.
      const encoded = perPixel && maskPx > 0 ? paintedMasks[index] : undefined
      if (encoded !== undefined) {
        const binary = atob(encoded)
        const bytes = new Uint8Array(binary.length)
        for (let at = 0; at < binary.length; at += 1) bytes[at] = binary.charCodeAt(at)
        if (bytes.length >= maskPx * maskPx) {
          drawMask(board, scratch.current, bytes, maskPx, bytePalette, cell)
        }
      } else {
        const colour = colours[label]
        if (colour === undefined) continue
        board.fillStyle = colour
        board.fillRect(cell.left, cell.top, cell.width, cell.height)
      }

      edgeLeft = Math.min(edgeLeft, cell.left)
      edgeTop = Math.min(edgeTop, cell.top)
      edgeRight = Math.max(edgeRight, cell.left + cell.width)
      edgeBottom = Math.max(edgeBottom, cell.top + cell.height)
    }
    board.globalAlpha = 1
    drawn.current = paintedCount

    // The leading edge: the patches that arrived in this update, outlined. It is where
    // the model is reading now, which is the part of "it is working" that a bar cannot
    // show. Cleared first, because unlike the paint it is not cumulative.
    marker.clearRect(0, 0, edge.width, edge.height)
    if (edgeRight > edgeLeft) {
      marker.strokeStyle = 'rgba(255, 255, 255, 0.92)'
      marker.lineWidth = 2
      marker.strokeRect(
        edgeLeft - 1,
        edgeTop - 1,
        edgeRight - edgeLeft + 2,
        edgeBottom - edgeTop + 2,
      )
    }
  }, [colours, frame, maskPx, painted, paintedCount, paintedMasks, perPixel, runKey, view])

  return (
    <div className={className ? `paint ${className}` : 'paint'}>
      {children}
      <canvas className="paint__layer" ref={paintLayer} aria-hidden="true" />
      <canvas className="paint__layer paint__layer--edge" ref={edgeLayer} aria-hidden="true" />
    </div>
  )
}
