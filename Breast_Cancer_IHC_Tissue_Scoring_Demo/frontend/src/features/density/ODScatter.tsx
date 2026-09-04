/**
 * The point cloud, the two arms, and the published vectors on the same axes.
 *
 * This is the picture the pipeline guide asks for by name, and the one that makes
 * step 6 obvious rather than magical. Two rays from the origin, the origin being
 * zero stain, and a wedge of mixtures between them — that geometry *is* colour
 * deconvolution, and step 6 does nothing but invert it.
 *
 * The cloud itself is a server-rendered PNG and everything drawn on it is SVG
 * over the top, which is not an implementation detail. A quarter of a million
 * points has to be a two-dimensional histogram or the wedge's shape disappears
 * under overplotting; but arms and reference vectors have to be *labelled*, and
 * labels belong in the DOM where they can be read, hovered and translated. Both
 * come from the same `Cloud`, so the two layers cannot disagree about where a
 * line is.
 *
 * Solid lines are measured from this slide. Dashed lines are published constants.
 * That distinction is the whole content of the panel and it is carried by the
 * stroke rather than by a legend, so a reader cannot mislay it.
 *
 * Under the plot, the angular histogram: the same cloud counted rather than
 * drawn. It answers a question the scatter cannot, which is how the tile's pixels
 * *distribute* between the arms — one broad hump means almost everything is a
 * mixture, a mode at each end means the tile holds regions of each stain nearly
 * alone.
 */

import { useMemo } from 'react'

import { densityPanelUrl } from '@/api/density'
import type { DensityOptions } from '@/api/density'
import { formatCount } from '@/lib/format'
import type { DensityCloud, DensityReport } from '@/types/density'

/** The plot's own coordinate box, in viewBox units. Square, because the axes must be. */
const BOX = 100

interface ODScatterProps {
  report: DensityReport
  cloud: DensityCloud
  options: DensityOptions
  /** True while a re-run is in flight, so the stale picture can dim. */
  refining: boolean
}

/** Short labels for the rays, so a legend is not needed to read the plot. */
const REFERENCE_LABEL: Record<string, string> = {
  haematoxylin: 'H',
  dab: 'DAB',
  eosin: 'E',
}

/** How a stain is written in prose. 'dab' is an acronym, not a word. */
const STAIN_LABEL: Record<string, string> = {
  haematoxylin: 'haematoxylin',
  dab: 'DAB',
  eosin: 'eosin',
}

export function ODScatter({ report, cloud, options, refining }: ODScatterProps) {
  const project = useMemo(() => {
    const xSpan = Math.max(cloud.xHigh - cloud.xLow, 1e-9)
    const ySpan = Math.max(cloud.yHigh - cloud.yLow, 1e-9)

    // Plot coordinates onto the viewBox. y is flipped because SVG counts
    // downwards and the plot counts upwards, and the flip has to happen here
    // rather than by transforming the group so the text stays the right way up.
    return (x: number, y: number): [number, number] => [
      ((x - cloud.xLow) / xSpan) * BOX,
      (1 - (y - cloud.yLow) / ySpan) * BOX,
    ]
  }, [cloud.xHigh, cloud.xLow, cloud.yHigh, cloud.yLow])

  const [ox, oy] = project(0, 0)

  const peak = Math.max(...cloud.angles, 1)
  const bars = cloud.angles.length

  return (
    <div className={refining ? 'od-scatter od-scatter--stale' : 'od-scatter'}>
      <div className="od-scatter__frame">
        {/* The cloud. Not lazy: it is the panel's subject, and the arms drawn over
            it would be meaningless floating on an empty box. */}
        <img
          className="od-scatter__cloud"
          src={densityPanelUrl(report.uploadId, 'scatter', options)}
          alt={`${formatCount(cloud.plotted)} pixels of this tile plotted in optical density space, forming a wedge between two arms ${cloud.separation.toFixed(0)} degrees apart`}
        />

        <svg
          className="od-scatter__overlay"
          viewBox={`0 0 ${BOX} ${BOX}`}
          preserveAspectRatio="none"
          role="presentation"
        >
          {/* Published vectors, dashed. Drawn under the arms so a coincidence
              reads as the arm landing on the reference rather than hiding it. */}
          {cloud.references.map((reference) => {
            const [x, y] = project(reference.plotX, reference.plotY)
            // A published vector can point outside the wedge the data occupies -
            // eosin usually does, which is the point of drawing it - so its ray
            // leaves the plot. The line may be clipped; the *label* may not, or the
            // reader is left with an unnamed stroke running off the edge.
            const lx = Math.min(BOX - 4, Math.max(4, x))
            const ly = Math.min(BOX - 3, Math.max(4, y))
            return (
              <g className="od-scatter__reference" key={reference.name}>
                <line x1={ox} y1={oy} x2={x} y2={y} />
                <text x={lx} y={ly} dx={x > ox ? -1.5 : 1.5} dy={y > oy ? -1 : 2.5}>
                  {REFERENCE_LABEL[reference.name] ?? reference.name}
                </text>
              </g>
            )
          })}

          {/* Arms, solid. Measured from this slide. */}
          {cloud.arms.map((arm) => {
            const [x, y] = project(arm.plotX, arm.plotY)
            return (
              <g className="od-scatter__arm" key={arm.angle}>
                <line x1={ox} y1={oy} x2={x} y2={y} />
              </g>
            )
          })}

          {/* Zero stain. Every ray starts here, and a wedge without a marked
              origin is a shape in a box rather than a pair of directions. */}
          <circle className="od-scatter__origin" cx={ox} cy={oy} r={1.4} />
        </svg>
      </div>

      <div className="od-scatter__legend">
        <span className="od-scatter__key od-scatter__key--arm">
          measured on this slide
        </span>
        <span className="od-scatter__key od-scatter__key--reference">
          Ruifrok &amp; Johnston, published
        </span>
        <span className="od-scatter__key od-scatter__key--origin">zero stain</span>
      </div>

      {/* --- the same cloud, counted ------------------------------------- */}
      <div className="od-angles">
        <span className="eyebrow">how the tile’s pixels sit between the two arms</span>

        <svg
          className="od-angles__plot"
          viewBox={`0 0 ${bars} 32`}
          preserveAspectRatio="none"
          role="presentation"
        >
          {cloud.angles.map((count, index) => {
            const height = (count / peak) * 30
            return (
              <rect
                key={index}
                x={index + 0.1}
                width={0.8}
                y={31 - height}
                height={Math.max(0.4, height)}
              />
            )
          })}
        </svg>

        <div className="od-angles__axis mono">
          <span>
            {STAIN_LABEL[cloud.arms[0]?.nearest ?? ''] ?? 'one'} edge
          </span>
          <span className="od-angles__mid">mixtures of both</span>
          <span>
            {STAIN_LABEL[cloud.arms[1]?.nearest ?? ''] ?? 'other'} edge
          </span>
        </div>

        <p className="od-angles__note">
          One broad hump means almost every pixel here is a{' '}
          <em>mixture</em> of the two stains. A peak at each end would mean the
          tile holds regions of each stain nearly alone. Either shape is fine for
          step 6 — what step 6 needs is the two <em>edges</em>, and those are what
          the arms above are.
        </p>
      </div>
    </div>
  )
}
