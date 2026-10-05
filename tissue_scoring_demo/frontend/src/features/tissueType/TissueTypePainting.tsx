/**
 * Step 8's pass, painting itself onto the slide.
 *
 * Step 8 is tens of minutes, and a bar with a percentage beside it says how much is left
 * but not *what is happening*. This does: a thumbnail of the slide with every patch
 * coloured in as its class comes back from the model, in the order the model was shown
 * them — left to right along one band, right to left along the next, top to bottom. The
 * same colours the finished map uses, so the picture that builds here is the picture the
 * next screen shows.
 *
 * **The drawing itself is `PaintCanvas`**, shared with step 11, because step 11 watches
 * the same model answer window by window over one region at a time and the only things
 * that differ are which part of the slide is on screen and what picture is underneath.
 * What is left here is step 8's own three: the whole slide as the view, step 4's
 * thumbnail as the backdrop, and the legend and caption that explain the pass.
 *
 * **The picture underneath is step 4's thumbnail, not a fresh read of the slide.** It is
 * the same array step 8 draws its finished `map` panel on, so the picture that builds
 * here and the picture on the next screen are the same picture rather than two
 * renderings of one slide at different sizes. It is also already on disk and already in
 * the browser's cache from an earlier step, which matters at exactly this moment: the
 * server has every core busy doing inference, and a new whole-slide pyramid read would
 * be competing with the pass this screen is watching. A slide overview is the fallback
 * if it is not there.
 */

import { useMemo, useState } from 'react'

import { calibrationPanelUrl } from '@/api/calibration'
import { slideThumbnailUrl } from '@/api/uploads'
import { PaintCanvas, type PaintFrame, type PaintView } from '@/components/paint/PaintCanvas'
import { formatCount } from '@/lib/format'
import type { TissueTypePaint } from '@/types/tissueType'

import '@/components/paint/paint.css'
import './tissueType.css'

interface TissueTypePaintingProps {
  uploadId: string
  /** The run's grid geometry and palette, or null before the grid is laid. */
  paint: TissueTypePaint | null
  /** The accumulated feed: flat `row, col, class` triples. Stable identity. */
  painted: number[]
  /**
   * One base64 pixel mask per patch in `painted`, same order. Stable identity, and
   * empty on the trained options — a patch with no mask is drawn as a flat colour.
   */
  paintedMasks: string[]
  /** How many patches of `painted` are filled in. What changes as the pass runs. */
  paintedCount: number
  /**
   * When this run started, as the server reports it. Its only job here is to be
   * different for a different run: a restart paints from nothing, and the canvas has to
   * be told that rather than adding a second run's patches on top of a first's.
   */
  startedAt: string | null
}

export function TissueTypePainting({
  uploadId,
  paint,
  painted,
  paintedMasks,
  paintedCount,
  startedAt,
}: TissueTypePaintingProps) {
  // Step 4's thumbnail, which is the array the finished map is drawn on. Falls back to a
  // slide overview if that step's cache is not there — the paint is still correct either
  // way, since its geometry is the slide's and not the picture's.
  const [slide, setSlide] = useState(() => calibrationPanelUrl(uploadId, 'thumbnail'))

  // Memoised because they are object props on a component whose effect depends on them:
  // rebuilt every render, the effect would run on every poll rather than on every new
  // patch, and clear the accumulated paint whenever the canvas happened to resize.
  const frame = useMemo<PaintFrame | null>(
    () =>
      paint
        ? {
            slideWidth: paint.slideWidth,
            slideHeight: paint.slideHeight,
            span: paint.span,
            stride: paint.stride,
          }
        : null,
    [paint],
  )

  /** Step 8 paints the whole section, so the view is the slide. */
  const view = useMemo<PaintView | null>(
    () =>
      paint
        ? { x: 0, y: 0, width: paint.slideWidth, height: paint.slideHeight }
        : null,
    [paint],
  )

  return (
    <figure className="tt-paint">
      <div className="tt-paint__frame">
        {paint && frame && view ? (
          <PaintCanvas
            frame={frame}
            view={view}
            colours={paint.colours}
            perPixel={paint.perPixel}
            maskPx={paint.maskPx}
            painted={painted}
            paintedMasks={paintedMasks}
            paintedCount={paintedCount}
            runKey={`${startedAt ?? 'unstarted'}|${paint.cols}x${paint.rows}|${paint.stride}|${paint.perPixel ? paint.maskPx : 'flat'}`}
          >
            <img
              className="tt-paint__slide"
              src={slide}
              alt="The slide being classified"
              onError={() => setSlide(slideThumbnailUrl(uploadId, 1600))}
            />
          </PaintCanvas>
        ) : (
          <div className="paint">
            <img
              className="tt-paint__slide"
              src={slide}
              alt="The slide being classified"
              onError={() => setSlide(slideThumbnailUrl(uploadId, 1600))}
            />
            <div className="tt-paint__waiting">
              <span className="mono">laying the grid over the tissue…</span>
            </div>
          </div>
        )}
      </div>

      {paint && (
        <div className="tt-paint__legend">
          {paint.labels.map((label, id) => (
            <span key={label} className="tt-paint__key">
              <span
                className="tt-paint__swatch"
                style={{ background: paint.colours[id] }}
              />
              {label}
            </span>
          ))}
        </div>
      )}

      <figcaption className="tt-paint__caption">
        {paintedCount > 0
          ? `${formatCount(paintedCount)} patches done. `
          : 'Nothing appears until the first patch comes back from the model. '}
        {paint?.perPixel
          ? 'The model answers for every pixel, so what appears is the shape of what it found rather than a coloured square.'
          : 'Each square is one patch, coloured by the answer the model gave.'}{' '}
        The white outline shows where the model is reading now. It slows over dense
        tissue and speeds up over glass.
      </figcaption>
    </figure>
  )
}
