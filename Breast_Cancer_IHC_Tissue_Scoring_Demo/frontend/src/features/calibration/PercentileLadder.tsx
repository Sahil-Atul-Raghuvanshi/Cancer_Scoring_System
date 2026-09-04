/**
 * The percentile ladder — the argument that the 95th is not doing the work.
 *
 * A percentile is a choice, and the honest way to show a choice is not load-
 * bearing is to show that its neighbours agree with it. On clean glass I₀ is flat
 * from the 90th to the 99th, and a reader who sees that plateau needs no further
 * defence of the number. Where it climbs steeply, something bright and non-glass
 * is in the sample and the step should not be trusted — which the reader can only
 * judge if they are shown the whole range.
 *
 * Three things are drawn over one another:
 *
 *   the curve     I₀ at each percentile, per channel, so a colour cast that
 *                 changes with percentile is visible as three diverging lines.
 *   the plateau   the 90th-to-99th span, shaded, because that is the interval the
 *                 reported `plateau` statistic is measured across.
 *   the cut       where the percentile control sits now.
 *
 * The vertical axis spans only the values actually present, not 0–255. The whole
 * point is a spread of a few levels, and on an absolute scale a few levels out of
 * 240 is a flat line — which would hide precisely the thing the panel is for. The
 * axis labels say what range is shown, so the stretch is stated rather than
 * implied.
 */

import { useCallback, useMemo, useRef } from 'react'

import { TrackSlider } from '@/components/ui/TrackSlider'
import type { Channels, WhitePoint } from '@/types/calibration'

const WIDTH = 256
const HEIGHT = 96

/** The plateau interval the reported statistic is measured over. */
const PLATEAU = [90, 99] as const

interface PercentileLadderProps {
  white: WhitePoint
  /** Where the control is now, which may be ahead of what the server has run. */
  draft: number
  /** The percentile the report on screen was actually computed at. */
  committed: number
  onDraft: (value: number) => void
}

const CHANNELS: Array<{ key: keyof Channels; label: string }> = [
  { key: 'r', label: 'R' },
  { key: 'g', label: 'G' },
  { key: 'b', label: 'B' },
]

interface Sample {
  level: number
  rgb: Channels
}

/**
 * A percentile onto its horizontal position, and back.
 *
 * The ladder is sampled unevenly — 50, 75, 90, 95, 99, 99.9, 100 — so the axis is
 * the *rank* of the sample, not its value. Plotting by value would crush the top
 * four samples into the last two pixels, which is exactly where the story is.
 *
 * One pair of functions rather than the same interpolation written out at each
 * call site: the plot, the plateau shading, the cut line, the pointer handler and
 * the slider's thumb all need this mapping, and five copies would be five chances
 * to disagree about where p95 is.
 *
 * The slider is the reason this matters most. Because the axis is spaced by rank,
 * p95 sits about halfway across the plot rather than nine tenths of the way - so a
 * slider running linearly from 50 to 100 would put its thumb nowhere near the line
 * it moves. It is driven by `x`/`level` instead, and points where the line points.
 */
function axis(samples: Sample[]) {
  const last = Math.max(1, samples.length - 1)

  return {
    /** Percentile → x, in viewBox units. */
    x(level: number): number {
      const index = samples.findIndex((sample) => sample.level >= level)
      if (index < 0) return WIDTH
      if (index === 0) return 0

      const before = samples[index - 1]
      const after = samples[index]
      if (!before || !after) return 0

      const span = after.level - before.level
      const t = span > 1e-9 ? (level - before.level) / span : 0
      return ((index - 1 + t) / last) * WIDTH
    },

    /** A fraction across the plot → percentile. */
    level(fraction: number): number {
      const position = Math.max(0, Math.min(1, fraction)) * last
      const index = Math.min(samples.length - 1, Math.floor(position))

      const here = samples[index]
      const next = samples[Math.min(samples.length - 1, index + 1)]
      if (!here) return 50
      if (!next) return here.level

      return here.level + (position - index) * (next.level - here.level)
    },
  }
}

export function PercentileLadder({
  white,
  draft,
  committed,
  onDraft,
}: PercentileLadderProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  const { lines, low, high, plateauBox, samples, scale } = useMemo(() => {
    const entries: Sample[] = Object.entries(white.ladder)
      .map(([level, rgb]) => ({ level: Number(level), rgb }))
      .filter((entry) => Number.isFinite(entry.level))
      .sort((a, b) => a.level - b.level)

    const values = entries.flatMap(({ rgb }) => [rgb.r, rgb.g, rgb.b])
    const lowest = values.length ? Math.min(...values) : 0
    const highest = values.length ? Math.max(...values) : 1
    // A floor on the span so a perfectly flat ladder does not divide by zero and
    // become a line jittering on rounding error.
    const span = Math.max(highest - lowest, 1)

    const mapping = axis(entries)
    const y = (value: number) => HEIGHT - ((value - lowest) / span) * (HEIGHT - 8) - 4

    return {
      samples: entries,
      scale: mapping,
      low: lowest,
      high: highest,
      lines: CHANNELS.map(({ key, label }) => ({
        key,
        label,
        points: entries
          .map((entry, index) => {
            const x = (index / Math.max(1, entries.length - 1)) * WIDTH
            return `${x.toFixed(2)},${y(entry.rgb[key]).toFixed(2)}`
          })
          .join(' '),
      })),
      plateauBox: { x1: mapping.x(PLATEAU[0]), x2: mapping.x(PLATEAU[1]) },
    }
  }, [white.ladder])

  // Click or drag anywhere on the plot, not just on the line - a 1px target is
  // not a target.
  const fromPointer = useCallback(
    (clientX: number) => {
      const box = svgRef.current?.getBoundingClientRect()
      if (!box || box.width === 0) return
      onDraft(scale.level((clientX - box.left) / box.width))
    },
    [onDraft, scale],
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
  const cutX = scale.x(draft)
  const first = samples[0]?.level ?? 50
  const finalLevel = samples[samples.length - 1]?.level ?? 100

  return (
    <div className="ladder">
      <div className="ladder__head">
        <span className="eyebrow">I₀ at every percentile of the glass</span>
        <span className="ladder__legend mono">
          {CHANNELS.map(({ key, label }) => (
            <span key={key}>
              <i className={`ladder__key ladder__key--${key}`} />
              {label}
            </span>
          ))}
        </span>
      </div>

      <svg
        ref={svgRef}
        className="ladder__plot"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        role="presentation"
        onPointerDown={drag}
        onPointerMove={move}
      >
        {/* The interval the reported plateau statistic is measured across. */}
        <rect
          className="ladder__plateau"
          x={plateauBox.x1}
          width={Math.max(1, plateauBox.x2 - plateauBox.x1)}
          y={0}
          height={HEIGHT}
        />

        {lines.map((line) => (
          <polyline
            key={line.key}
            className={`ladder__line ladder__line--${line.key}`}
            points={line.points}
          />
        ))}

        <line
          className={pending ? 'ladder__cut ladder__cut--pending' : 'ladder__cut'}
          x1={cutX}
          x2={cutX}
          y1={0}
          y2={HEIGHT}
        />
      </svg>

      <TrackSlider
        label="Percentile"
        fraction={cutX / WIDTH}
        value={draft.toFixed(1)}
        valueText={`${draft.toFixed(1)}th percentile of the glass`}
        ariaLabel="Percentile of the glass taken as I₀"
        onFraction={(fraction) => onDraft(scale.level(fraction))}
        pending={pending}
      />

      <div className="ladder__axis mono">
        <span>p{first}</span>
        <span>
          {low.toFixed(0)}–{high.toFixed(0)} of 255 shown · shaded is the p
          {PLATEAU[0]}–p{PLATEAU[1]} plateau
        </span>
        <span>p{finalLevel}</span>
      </div>

      <p className="ladder__verdict">
        {white.plateau < 0.03 ? (
          <>
            I₀ moves <strong>{(white.plateau * 100).toFixed(2)}%</strong> across the
            shaded span, so the choice of percentile is not doing the work here. There
            is a real plateau of clean glass, and any cut through it gives much the same
            answer — which is the argument for not hand-picking the number.
          </>
        ) : (
          <>
            I₀ moves <strong>{(white.plateau * 100).toFixed(1)}%</strong> across the
            shaded span, more than a plateau of clean glass should. Something bright and
            non-glass is in the sample, or the glass genuinely varies across the slide.
            Check the field panel and step 3&rsquo;s threshold before trusting this white
            point to three figures.
          </>
        )}
      </p>
    </div>
  )
}
