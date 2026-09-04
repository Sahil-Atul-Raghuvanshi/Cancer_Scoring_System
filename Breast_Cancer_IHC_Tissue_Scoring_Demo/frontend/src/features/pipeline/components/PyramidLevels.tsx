/**
 * The pyramid inside the file, drawn as a pyramid.
 *
 * A table of numbers tells you a slide has eight levels; seeing each level's
 * footprint shrink tells you what a pyramid *is*. Every figure on a row was
 * read from the file.
 *
 * About the previews: each row shows the same whole-slide overview, sized to
 * represent that level's footprint. It is not that level's own pixels - level 0
 * is 126,976 px square and could never be shown whole, which is the entire
 * reason pyramids exist. The box size is the honest part; the picture inside it
 * is there so the shrinking has something to shrink.
 */

import type { ReactNode } from 'react'

import { slideThumbnailUrl } from '@/api/uploads'
import type { PyramidLevel, SlideReadout } from '@/types/slide'

import './pipeline.css'

/** Longest side of the level-0 preview, in px. */
const BASE_PREVIEW = 190
const MIN_PREVIEW = 10

/**
 * Preview size for a level.
 *
 * Scaled by the square root of the downsample rather than the downsample
 * itself: at a 256x range a linear scale would put level 7 below one pixel.
 * The compression is noted in the legend so the ratios are not read as exact.
 */
function previewSize(level: PyramidLevel, aspect: number): { width: number; height: number } {
  const longest = Math.max(MIN_PREVIEW, BASE_PREVIEW / Math.sqrt(level.downsample))
  return aspect >= 1
    ? { width: longest, height: longest / aspect }
    : { width: longest * aspect, height: longest }
}

/* --- Icons ---------------------------------------------------------------- */

function DimensionsIcon() {
  return (
    <svg viewBox="0 0 16 16" className="pyramid__icon" aria-hidden>
      <path
        d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function GridIcon() {
  return (
    <svg viewBox="0 0 16 16" className="pyramid__icon" aria-hidden>
      <rect
        x="2.5"
        y="2.5"
        width="11"
        height="11"
        rx="1"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
      />
      <path d="M6.2 2.5v11M9.8 2.5v11M2.5 6.2h11M2.5 9.8h11" stroke="currentColor" strokeWidth="1.1" />
    </svg>
  )
}

function RulerIcon() {
  return (
    <svg viewBox="0 0 16 16" className="pyramid__icon" aria-hidden>
      <path
        d="M10.6 1.9 14.1 5.4 5.4 14.1 1.9 10.6z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      <path
        d="M9.1 4.4l1.2 1.2M7 6.5l1.2 1.2M4.9 8.6l1.2 1.2"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
    </svg>
  )
}

/* --- Row stat ------------------------------------------------------------- */

function Stat({
  icon,
  label,
  value,
  /** Skips the uppercase transform, for labels that carry a unit symbol. */
  unit = false,
}: {
  icon: ReactNode
  label: string
  value: string
  unit?: boolean
}) {
  return (
    <div className="pyramid__stat">
      <dt className={unit ? 'pyramid__stat-label pyramid__stat-label--unit' : 'pyramid__stat-label'}>
        {icon}
        <span>{label}</span>
      </dt>
      <dd className="pyramid__stat-value">{value}</dd>
    </div>
  )
}

/* --- The pyramid ---------------------------------------------------------- */

export function PyramidLevels({ readout }: { readout: SlideReadout }) {
  const aspect = readout.widthPx / readout.heightPx
  // One image, reused at every size: the rows differ in scale, not content.
  const preview = slideThumbnailUrl(readout.uploadId, 320)

  return (
    <section className="pyramid" aria-label="Pyramid levels">
      <header className="pyramid__head">
        <h3 className="pyramid__title">Pyramid (multi-resolution levels)</h3>
        <p className="pyramid__sub mono">
          Tiles are {readout.tileSize} px <span aria-hidden>·</span> Working level:{' '}
          {readout.workingLevel}
        </p>
      </header>

      <ol className="pyramid__levels">
        {readout.levels.map((level, index) => {
          const size = previewSize(level, aspect)
          return (
            <li
              key={level.level}
              className={
                level.isWorkingLevel ? 'pyramid__level pyramid__level--working' : 'pyramid__level'
              }
              style={{ animationDelay: `${index * 55}ms` }}
            >
              <div className="pyramid__marker">
                <span className="pyramid__number">{level.level}</span>
                {level.isWorkingLevel && <span className="pyramid__working">working</span>}
              </div>

              <div className="pyramid__preview-cell">
                <img
                  src={preview}
                  alt=""
                  className="pyramid__preview"
                  style={{ width: `${size.width}px`, height: `${size.height}px` }}
                  loading={index < 3 ? 'eager' : 'lazy'}
                />
              </div>

              <dl className="pyramid__stats">
                <Stat
                  icon={<DimensionsIcon />}
                  label="Dimensions"
                  value={`${level.width.toLocaleString()} × ${level.height.toLocaleString()}`}
                />
                <Stat icon={<GridIcon />} label="Downsample" value={`${level.downsample}×`} />
                <Stat
                  icon={<RulerIcon />}
                  label="µm/px"
                  unit
                  value={level.mpp !== null ? String(level.mpp) : '—'}
                />
                <Stat
                  icon={<GridIcon />}
                  label="Tiles"
                  value={level.tiles.toLocaleString()}
                />
              </dl>
            </li>
          )
        })}
      </ol>

      <footer className="pyramid__legend">
        <div className="pyramid__legend-item">
          <DimensionsIcon />
          <div>
            <span className="pyramid__legend-term">Dimensions</span>
            <span className="pyramid__legend-gloss">Width × height (px)</span>
          </div>
        </div>
        <div className="pyramid__legend-item">
          <GridIcon />
          <div>
            <span className="pyramid__legend-term">Downsample</span>
            <span className="pyramid__legend-gloss">Relative to level 0</span>
          </div>
        </div>
        <div className="pyramid__legend-item">
          <RulerIcon />
          <div>
            <span className="pyramid__legend-term">µm/px</span>
            <span className="pyramid__legend-gloss">Microns per pixel</span>
          </div>
        </div>
        <div className="pyramid__legend-item">
          <GridIcon />
          <div>
            <span className="pyramid__legend-term">Tiles</span>
            <span className="pyramid__legend-gloss">
              At {readout.tileSize} px
            </span>
          </div>
        </div>
      </footer>
    </section>
  )
}
