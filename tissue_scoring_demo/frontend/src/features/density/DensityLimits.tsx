/**
 * Which pixels can be trusted — in plain words, with the colour key beside the map.
 *
 * A step that silently repairs its own inputs is a step nobody downstream can
 * reason about. So every pixel that is not a real reading is counted, tinted, and
 * named, and the panel beside the counts says *where* they are - which is the
 * difference between a caveat and a check. A reader can see whether the amber sits
 * on glass inside the tile, which is benign, or across the tissue, which is not.
 *
 * **Written for someone meeting optical density for the first time.** Each row
 * leads with what the pixel *is* and what happens to it, in one sentence with no
 * arithmetic in it. The reason it has to work that way follows underneath, marked
 * as secondary, because a reader who wants the quantisation argument should be able
 * to reach it and a reader who does not should be able to skip four paragraphs of
 * it without missing the point of the screen.
 *
 * Three things carry the numbers rather than stating them:
 *
 *   the budget bar   the tile split into used / too faint / too dark. Those three
 *                    sum to the whole tile exactly, by construction in the server:
 *                    faint is `mean OD < beta`, unstable is `>= beta but pointing
 *                    unreliably`, admitted is the rest. The floor and negative
 *                    flags overlap those categories rather than adding to them, so
 *                    they are not segments - five slices would claim a partition
 *                    that does not exist.
 *   the colour key   the same four tints the server paints, at the same values, so
 *                    the legend and the picture cannot drift apart.
 *   the signal scale the tile's median and 99th percentile against step 4's noise
 *                    floor, on one axis, so "well above the floor" is a distance.
 *
 * The round trip leads because it is the reassuring one and the most easily
 * misunderstood. Optical density is a *change of units* with an exact inverse,
 * I = I₀ × 10^(−OD), not a filter or an enhancement. Applying that inverse and
 * reporting the residual in intensity levels is the only honest way to say
 * "nothing was thrown away", and it is a measurement rather than a promise.
 */

import { densityPanelUrl } from '@/api/density'
import type { DensityOptions } from '@/api/density'
import type { DensityReport } from '@/types/density'

interface DensityLimitsProps {
  report: DensityReport
  options: DensityOptions
}

/**
 * The four mark colours, at the values `overlay.py` paints them.
 *
 * Written out here rather than taken from the theme tokens because the legend's
 * only job is to match the picture: `--text-muted` is two levels off
 * `TRANSPARENT_TINT`, which is invisible in isolation and wrong in a key sitting
 * beside the image it describes. If the server's tints change, these change.
 */
const MARK = {
  floor: 'rgb(248, 113, 113)',
  negative: 'rgb(251, 191, 36)',
  unstable: 'rgb(167, 139, 250)',
  faint: 'rgb(110, 124, 148)',
} as const

/**
 * A position on the signal scale, as a percentage of its own top end.
 *
 * Clamped and guarded because the scale is drawn against the tile's own 99th
 * percentile, and a tile of blank glass has a 99th percentile at or below the noise
 * floor - which would put the floor band past the end of the bar, or divide by
 * zero. A floor that fills the whole scale is the honest picture of that tile.
 */
function scalePosition(value: number, top: number): string {
  if (!(top > 0)) return '100%'
  return `${Math.max(0, Math.min(100, (value / top) * 100)).toFixed(2)}%`
}

/** A share as a percentage, without pretending to precision it does not have. */
function share(value: number): string {
  if (value <= 0) return '0%'
  if (value < 0.001) return '<0.1%'
  return `${(value * 100).toFixed(value < 0.05 ? 2 : 1)}%`
}

export function DensityLimits({ report, options }: DensityLimitsProps) {
  const { limits, cloud, stats } = report

  const floor = report.white.noiseFloor
  const headroom = stats.meanP99 / Math.max(floor, 1e-6)

  // The bar, and the key beside the map. One list, so a category cannot appear in
  // one and not the other.
  const used = cloud ? cloud.admittedShare : null
  const faint = cloud ? cloud.faintShare : limits.transparentShare
  const unstable = cloud ? cloud.unstableShare : null

  const reasons = [
    {
      key: 'faint',
      colour: MARK.faint,
      name: 'grey',
      value: faint,
      plain: 'Barely any stain in it.',
      does: 'Not used to identify the stains.',
      why: (
        <>
          A pixel with almost no stain still has a colour, but that colour is mostly
          rounding error. Including these would fill the chart with noise that looks
          just like a mixture of the two stains. The cut-off is{' '}
          <span className="mono">{limits.beta}</span>.
        </>
      ),
    },
    unstable !== null
      ? {
          key: 'unstable',
          colour: MARK.unstable,
          name: 'violet',
          value: unstable,
          plain: 'So dark that almost no light got through.',
          does: 'Also not used to identify the stains.',
          why: (
            <>
              When a pixel is nearly black, one step of camera noise changes its
              reading enormously. These pixels are also the most strongly stained, so
              they sit right where the stain colours are read from. They are dropped
              once the colour could be off by more than{' '}
              {cloud?.toleranceDeg.toFixed(1)}°.
            </>
          ),
        }
      : null,
    {
      key: 'negative',
      colour: MARK.negative,
      name: 'amber',
      value: limits.negativeShare,
      plain: 'Slightly brighter than blank glass.',
      does: 'Reported as it is. Never rounded up to zero.',
      why: (
        <>
          This gives a negative amount of stain, which is impossible. It means the
          blank-glass value is a shade too low here, not that anything is wrong with
          the tissue. The worst case here is{' '}
          <span className="mono">{limits.negativeWorst.toFixed(3)}</span>. Rounding it
          to zero would make an impossible reading look like a real faint one.
        </>
      ),
    },
    {
      key: 'floor',
      colour: MARK.floor,
      name: 'red',
      value: limits.floorShare,
      plain: 'No light recorded at all.',
      does: 'Treated as “at least this much stain”, not an exact value.',
      why: (
        <>
          A reading of zero light cannot be converted into a number, so a tiny minimum
          is used instead. The result is a lower bound: the real value is higher, and
          there is no way to know by how much.
        </>
      ),
    },
  ].filter((row): row is NonNullable<typeof row> => row !== null)

  return (
    <div className="od-limits">
      <div className="od-limits__head">
        <span className="eyebrow">which pixels can be trusted?</span>
        {used !== null && (
          <span className="od-limits__headline">
            <strong className="mono">{share(used)}</strong> of this tile is reliable
            enough to identify the stains
          </span>
        )}
      </div>

      {/* used + faint + unstable = the tile, exactly - see the module docstring. */}
      {used !== null && unstable !== null && (
        <div className="od-budget">
          <div className="od-budget__bar">
            <span
              className="od-budget__seg od-budget__seg--used"
              style={{ width: `${(used * 100).toFixed(2)}%` }}
              title={`${share(used)} used`}
            />
            <span
              className="od-budget__seg"
              style={{ width: `${(faint * 100).toFixed(2)}%`, background: MARK.faint }}
              title={`${share(faint)} too faint`}
            />
            <span
              className="od-budget__seg"
              style={{
                width: `${(unstable * 100).toFixed(2)}%`,
                background: MARK.unstable,
                minWidth: '3px',
              }}
              title={`${share(unstable)} too dark`}
            />
          </div>
          <div className="od-budget__keys">
            <span className="od-budget__key">
              <i className="od-budget__key-used" aria-hidden />
              <strong className="mono">{share(used)}</strong> used
            </span>
            <span className="od-budget__key">
              <i style={{ background: MARK.faint }} aria-hidden />
              <strong className="mono">{share(faint)}</strong> too faint
            </span>
            <span className="od-budget__key">
              <i style={{ background: MARK.unstable }} aria-hidden />
              <strong className="mono">{share(unstable)}</strong> too dark
            </span>
          </div>
        </div>
      )}

      <div className="od-limits__body">
        <figure className="od-limits__panel">
          <img
            src={densityPanelUrl(report.uploadId, 'limits', options)}
            alt="The tile, with unreliable pixels coloured by reason"
          />
          <figcaption>
            Unreliable pixels are coloured by reason. Grey over the pale areas is
            normal and is usually the biggest group. Violet along the dark membranes is
            also normal; violet everywhere would mean the scan is too dark.
          </figcaption>
        </figure>

        <dl className="od-limits__rows">
          {reasons.map((row) => (
            <div className="od-reason" key={row.key}>
              <dt>
                <span
                  className="od-reason__swatch"
                  style={{ background: row.colour }}
                  aria-hidden
                />
                <span className="od-reason__share mono">{share(row.value)}</span>
                <span className="od-reason__name mono">{row.name}</span>
                <span className="od-reason__plain">{row.plain}</span>
              </dt>
              <dd>
                <span className="od-reason__does">{row.does}</span>
                <span className="od-reason__why">{row.why}</span>
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <p className="od-limits__overlap">
        Grey and violet are the two excluded slices of the bar above. Red and amber are
        warnings that sit <em>inside</em> those groups, so they are not separate
        slices.
      </p>

      {/* The floor, the median and the top end on one axis: "well above the noise"
          as a distance rather than as three numbers to hold in mind. */}
      <div className="od-signal">
        <span className="eyebrow">is there enough stain to measure?</span>
        <div className="od-signal__scale">
          <span
            className="od-signal__floor"
            style={{ width: scalePosition(floor, stats.meanP99) }}
          />
          <span
            className="od-signal__tick od-signal__tick--median"
            style={{ left: scalePosition(stats.meanMedian, stats.meanP99) }}
          />
        </div>
        <div className="od-signal__marks mono">
          <span>
            {floor.toFixed(3)}
            <em>blank glass</em>
          </span>
          <span>
            {stats.meanMedian.toFixed(3)}
            <em>typical pixel</em>
          </span>
          <span>
            {stats.meanP99.toFixed(3)}
            <em>strongest pixels</em>
          </span>
        </div>
        <p>
          Blank glass on this slide reads <strong>{floor.toFixed(3)}</strong>. Anything
          below that cannot be told apart from no stain at all. A typical pixel here is
          well clear of it, and the strongest reach{' '}
          <strong>{headroom.toFixed(0)}×</strong> that level, so there is real stain to
          measure.
        </p>
      </div>

      <div className="od-limits__roundtrip">
        <div className="od-limits__figure">
          <span className="od-limits__value mono">
            {(limits.roundtripExactShare * 100).toFixed(1)}%
          </span>
          <span className="od-limits__caption">converts back exactly</span>
          <span className="od-limits__sub mono">
            worst error {limits.roundtripMax.toFixed(4)} of one brightness level
          </span>
        </div>
        <p>
          Nothing here changes the picture. This step only changes the units, and the
          conversion can be undone exactly &mdash; that share of the tile comes back to
          within half a brightness level. Nothing was sharpened, smoothed or cut off,
          and this number is the proof rather than a promise.
        </p>
      </div>
    </div>
  )
}
