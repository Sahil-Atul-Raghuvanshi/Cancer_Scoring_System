/**
 * Every measured cell as a dot, with the decision boundary drawn on it.
 *
 * The x-axis is the same on every marker: how much brown is in this cell's
 * compartment. **The y-axis is not**, and that is the thing this picture exists
 * to teach. On CD44, ABCC4 and ABCC11 it is how complete the ring is; on the two
 * cadherins it is what share of the cell body is brown. Switch the marker and
 * the axis label changes underneath the same dots - which is the clearest way to
 * show an audience that "positive" is defined per marker, not universally.
 *
 * The boundary is drawn as two lines meeting at a corner rather than one
 * diagonal, because positivity is two independent conditions and not a trade-off
 * between them: a very dark cell with a broken ring does not become positive by
 * being darker.
 */

import { useMemo, useState } from 'react'

import type { CellPoint, PerCellReport } from '@/types/perCell'

interface CellScatterProps {
  report: PerCellReport
  /** The cut on the x-axis: the optical density a cell has to clear. */
  positivityOd: number
  /** The cut on the y-axis, from this antibody's own cut-point set. */
  secondMin: number
  selected: CellPoint | null
  onSelect: (point: CellPoint | null) => void
}

const WIDTH = 560
const HEIGHT = 340
const PAD = { left: 56, right: 16, top: 16, bottom: 44 }

const AXIS_COPY: Record<PerCellReport['secondMeasure'], string> = {
  ring_completeness: 'How much of the ring is stained',
  stained_fraction: 'How much of the cell body is stained',
}

export function CellScatter({
  report,
  positivityOd,
  secondMin,
  selected,
  onSelect,
}: CellScatterProps) {
  const [hover, setHover] = useState<CellPoint | null>(null)

  // A fixed 0-1 y-axis, and an x-axis that follows the data but never shrinks
  // below the cut - a scatter whose boundary sat outside the frame would be a
  // picture of the decision with the decision cropped out.
  const maxOd = useMemo(
    () => Math.max(positivityOd * 1.6, ...report.points.map((p) => p.intensityOd), 0.5),
    [positivityOd, report.points],
  )

  const plotW = WIDTH - PAD.left - PAD.right
  const plotH = HEIGHT - PAD.top - PAD.bottom
  const sx = (od: number) => PAD.left + (Math.min(od, maxOd) / maxOd) * plotW
  const sy = (value: number) => PAD.top + (1 - Math.min(value, 1)) * plotH

  const cutX = sx(positivityOd)
  const cutY = sy(secondMin)

  const active = hover ?? selected

  return (
    <figure className="pc-scatter">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Every measured cell: brown strength against ${AXIS_COPY[
          report.secondMeasure
        ].toLowerCase()}`}
      >
        {/* The quadrant a cell has to be in to count as positive. */}
        <rect
          x={cutX}
          y={PAD.top}
          width={Math.max(0, WIDTH - PAD.right - cutX)}
          height={Math.max(0, cutY - PAD.top)}
          className="pc-scatter__positive"
        />

        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH} className="pc-scatter__axis" />
        <line
          x1={PAD.left}
          y1={PAD.top + plotH}
          x2={WIDTH - PAD.right}
          y2={PAD.top + plotH}
          className="pc-scatter__axis"
        />

        {report.points.map((point) => {
          const positive = point.intensityOd >= positivityOd && point.second >= secondMin
          const isActive = active?.cellId === point.cellId && active?.fieldIndex === point.fieldIndex
          return (
            <circle
              key={`${point.regionRank}-${point.fieldIndex}-${point.cellId}`}
              cx={sx(point.intensityOd)}
              cy={sy(point.second)}
              r={isActive ? 4.5 : 2}
              className={
                isActive
                  ? 'pc-scatter__dot pc-scatter__dot--active'
                  : positive
                    ? 'pc-scatter__dot pc-scatter__dot--positive'
                    : 'pc-scatter__dot'
              }
              onMouseEnter={() => setHover(point)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect(point)}
            />
          )
        })}

        {/* The boundary: two conditions, drawn as two lines. */}
        <line x1={cutX} y1={PAD.top} x2={cutX} y2={PAD.top + plotH} className="pc-scatter__cut" />
        <line x1={cutX} y1={cutY} x2={WIDTH - PAD.right} y2={cutY} className="pc-scatter__cut" />

        <text x={PAD.left + plotW / 2} y={HEIGHT - 8} className="pc-scatter__label" textAnchor="middle">
          How strong the brown is in this cell
        </text>
        <text
          x={-(PAD.top + plotH / 2)}
          y={14}
          className="pc-scatter__label"
          textAnchor="middle"
          transform="rotate(-90)"
        >
          {AXIS_COPY[report.secondMeasure]}
        </text>

        <text x={cutX + 6} y={PAD.top + 14} className="pc-scatter__note">
          counted as positive
        </text>
      </svg>

      <figcaption>
        {active ? (
          <>
            One cell, region {active.regionRank}, field {active.fieldIndex + 1}. Brown
            strength {active.intensityOd.toFixed(3)};{' '}
            {report.secondMeasure === 'ring_completeness'
              ? `${Math.round(active.second * 36)} of 36 segments of its ring are stained`
              : `${(active.second * 100).toFixed(0)}% of its cell body is stained`}
            .
          </>
        ) : (
          <>
            {report.sampled.toLocaleString()} of {report.cells.toLocaleString()} cells
            drawn. Hover a dot for that cell&rsquo;s numbers; click to see the piece of
            slide it came from.
          </>
        )}
      </figcaption>
    </figure>
  )
}
