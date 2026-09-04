/**
 * Panel 3 - the saturation histogram, with Otsu's line drawn on it.
 *
 * This is the panel that has to carry the argument, so several things are drawn
 * over one another on purpose:
 *
 *   the bars       how many pixels sit at each saturation level.
 *   Otsu's curve   the between-class variance at every possible cut. Its peak
 *                  *is* Otsu's answer, so the reader can see the choice being
 *                  made rather than being told the outcome.
 *   the chord      the straight line the triangle rule measures against, from
 *                  the top of the peak to the end of the tail. Drawn only when
 *                  that rule is the one in use, because otherwise it is a line
 *                  explaining a decision nobody made.
 *   three lines    Otsu's cut, the triangle's cut, and the viewer's.
 *
 * Both rules are always drawn, and the one in use is the solid one. On a slide
 * whose histogram is a spike and a tail the two land far apart, and seeing that
 * gap is the point: it is the difference between a threshold chosen by a rule
 * that fits the data and one chosen by a rule that does not.
 *
 * The bars are drawn on a log scale. On a real slide the glass hump is two or
 * three orders of magnitude taller than the tissue hump, so on a linear scale
 * the tissue hump is a flat line one pixel high and the whole point of the
 * picture is lost. The axis says so rather than leaving the reader to assume
 * linear.
 *
 * Dragging is instant because nothing here needs the server: the counts are
 * already in the browser and moving a line over them is free. What the server
 * recomputes - the mask and every area beside it - is deliberately behind a
 * short pause, so the numbers on screen always belong to a cut that was really
 * run. See `useTissueMask`.
 */

import { useCallback, useMemo, useRef } from 'react'

import { TrackSlider } from '@/components/ui/TrackSlider'
import { formatCount } from '@/lib/format'
import type { TissueHistogram, TissueThreshold } from '@/types/tissue'

const WIDTH = 256
const HEIGHT = 110

/**
 * A saturation level onto its share of the plot's width, and back.
 *
 * The plot draws every mark - the bars, Otsu's curve, the chord and all three
 * cut lines - at `x = level` in a viewBox `WIDTH` units wide, so a level's
 * position across the plot is simply `level / WIDTH`. The slider under it is
 * positioned by the same number, which is the only way the thumb and the cut line
 * can be guaranteed to agree: there is one mapping, not two that happen to match.
 *
 * `WIDTH` and not 255 is deliberate. There are 256 bins and the bar for level 255
 * occupies the last unit of the viewBox, so level 255 sits at 255/256 of the way
 * across - a whisker short of the right edge, which is where its bar starts.
 */
const axis = {
  fraction: (level: number) => level / WIDTH,
  level: (fraction: number) =>
    Math.max(0, Math.min(255, Math.round(fraction * WIDTH))),
}

interface SaturationHistogramProps {
  histogram: TissueHistogram
  threshold: TissueThreshold
  /** Where the slider is now, which may be ahead of what the server has run. */
  draft: number
  /** The cut the report on screen was actually computed at. */
  committed: number
  onDraft: (value: number) => void
}

export function SaturationHistogram({
  histogram,
  threshold,
  draft,
  committed,
  onDraft,
}: SaturationHistogramProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  const { bars, criterion, chord, peak } = useMemo(() => {
    const scaled = histogram.bins.map((count) => Math.log10(1 + count))
    const tallest = Math.max(...scaled, 1)

    // The triangle rule's construction, in the same log space the bars are
    // drawn in — drawing it against linear counts would put it somewhere the
    // reader cannot relate to the picture in front of them.
    const barTop = (level: number) => {
      const clamped = Math.max(0, Math.min(scaled.length - 1, level))
      return HEIGHT - ((scaled[clamped] ?? 0) / tallest) * HEIGHT
    }

    const chordLine = {
      x1: threshold.triangleFrom,
      y1: barTop(threshold.triangleFrom),
      x2: threshold.triangleTo,
      y2: barTop(threshold.triangleTo),
    }

    // One path rather than 256 rects: it is a fifth of the DOM and it redraws
    // in one paint while the slider is moving.
    const path = scaled
      .map((value, level) => {
        const height = (value / tallest) * HEIGHT
        return `M${level} ${HEIGHT}V${HEIGHT - height}`
      })
      .join('')

    const curve = histogram.criterion
      .map((value, level) => `${level},${(HEIGHT - value * HEIGHT).toFixed(2)}`)
      .join(' ')

    return { bars: path, criterion: curve, chord: chordLine, peak: tallest }
  }, [histogram, threshold.triangleFrom, threshold.triangleTo])

  // Click or drag anywhere on the plot, not just on the line - a 1px target is
  // not a target.
  const fromPointer = useCallback(
    (clientX: number) => {
      const box = svgRef.current?.getBoundingClientRect()
      if (!box || box.width === 0) return
      onDraft(axis.level((clientX - box.left) / box.width))
    },
    [onDraft],
  )

  const drag = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      event.currentTarget.setPointerCapture(event.pointerId)
      fromPointer(event.clientX)
    },
    [fromPointer],
  )

  const move = useCallback(
    (event: React.PointerEvent<SVGSVGElement>) => {
      if (event.buttons === 0) return
      fromPointer(event.clientX)
    },
    [fromPointer],
  )

  const pending = draft !== committed
  const usingTriangle = threshold.source === 'triangle'

  return (
    <div className="tissue-hist">
      <div className="tissue-hist__head">
        <span className="eyebrow">
          saturation histogram, and where each rule cuts it
        </span>
        <span className="tissue-hist__legend mono">
          <i className="tissue-hist__key tissue-hist__key--bars" /> pixels
          <i className="tissue-hist__key tissue-hist__key--curve" /> Otsu&rsquo;s criterion
          {usingTriangle && (
            <>
              <i className="tissue-hist__key tissue-hist__key--chord" /> triangle chord
            </>
          )}
        </span>
      </div>

      <svg
        ref={svgRef}
        className="tissue-hist__plot"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="presentation"
        onPointerDown={drag}
        onPointerMove={move}
      >
        <path className="tissue-hist__bars" d={bars} />
        <polyline className="tissue-hist__curve" points={criterion} />

        {/* The triangle rule's construction, only when that rule is in use. */}
        {usingTriangle && (
          <line
            className="tissue-hist__chord"
            x1={chord.x1}
            y1={chord.y1}
            x2={chord.x2}
            y2={chord.y2}
          />
        )}

        {/* Both automatic answers, always drawn, so neither a manual cut nor the
            rule that lost is ever off the picture. The one in use is solid. */}
        <line
          className={
            usingTriangle
              ? 'tissue-hist__rule tissue-hist__rule--otsu'
              : 'tissue-hist__rule tissue-hist__rule--otsu tissue-hist__rule--chosen'
          }
          x1={threshold.otsu}
          x2={threshold.otsu}
          y1={0}
          y2={HEIGHT}
        />
        <line
          className={
            usingTriangle
              ? 'tissue-hist__rule tissue-hist__rule--triangle tissue-hist__rule--chosen'
              : 'tissue-hist__rule tissue-hist__rule--triangle'
          }
          x1={threshold.triangle}
          x2={threshold.triangle}
          y1={0}
          y2={HEIGHT}
        />

        <line
          className={
            pending ? 'tissue-hist__cut tissue-hist__cut--pending' : 'tissue-hist__cut'
          }
          x1={draft}
          x2={draft}
          y1={0}
          y2={HEIGHT}
        />
      </svg>

      <TrackSlider
        label="Threshold"
        fraction={axis.fraction(draft)}
        value={String(draft)}
        valueText={`saturation ${draft} of 255`}
        ariaLabel="Saturation threshold"
        onFraction={(fraction) => onDraft(axis.level(fraction))}
        pending={pending}
      />

      <div className="tissue-hist__axis mono">
        <span>0 — colourless</span>
        <span>
          log counts, peak {peak.toFixed(1)} ({formatCount(histogram.countedPixels)} px counted)
        </span>
        <span>255 — vivid</span>
      </div>
    </div>
  )
}
