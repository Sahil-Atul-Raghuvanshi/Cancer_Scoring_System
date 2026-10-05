/**
 * The arms against the published vectors — the step's headline claim, as numbers.
 *
 * Nothing in step 5 is told what haematoxylin or DAB look like. It projects the
 * tile's densities onto their own plane and takes the angular extremes. So "the
 * two arms *are* the two stains" is a claim with a truth value, and this card is
 * where it gets one: each arm's angle from the nearest published vector, in three
 * dimensions rather than in the projection the scatter draws.
 *
 * **Every number here is also drawn.** A reader who does not already know what
 * "23.3 degrees between two unit vectors" feels like cannot judge the claim from
 * the figure alone, so each one carries a picture of itself:
 *
 *   the swatch pair   the colour each direction *is*, beside the colour the
 *                     published vector is. Two browns that match settle the
 *                     question before the degrees are read - see `stainColour`.
 *   the match gauge   the angle on a fixed 0-45 degree scale with the matching
 *                     band shaded, so near and far are positions rather than
 *                     values to be compared in the reader's head.
 *   the split bar     the two shares as one bar in the two arms' own colours,
 *                     rather than as two percentages in two different cards.
 *   the wedge         both wedges from one apex, with the excess between them
 *                     shaded. "Wider than the published pair" is a shape.
 *
 * Three things stay on screen that a summary would drop.
 *
 * **The wedge's width against the references' own.** "61 degrees" means nothing
 * alone. Beside "37 degrees between Ruifrok's haematoxylin and DAB" it means the
 * wedge is wider than the published pair, which is a question rather than a
 * result - a third absorber, extremes reaching into unreliable pixels, or DAB
 * itself, whose colour shifts with concentration because it is a scattering
 * precipitate.
 *
 * **The out-of-plane term on every reference.** A unit vector projected onto a
 * plane gets shorter, and how much shorter is how much of it points out of the
 * page. Without that number a reference drawn near an arm looks like a match when
 * it may be some distance behind it.
 *
 * **That these estimates are not what gets measured with.** Estimating stain
 * vectors per image is Macenko's method and it is exactly what has just happened
 * here - and step 6 will not use the result, because a per-image estimate gives
 * every slide its own scale. The arms are the evidence that the *fixed* vectors
 * fit this slide, and saying so is the difference between a demo and a
 * misdirection.
 */

import { Badge } from '@/components/ui/Badge'
import type { Channels } from '@/types/calibration'
import type { DensityCloud } from '@/types/density'

interface ArmVerdictProps {
  cloud: DensityCloud
}

const STAIN_LABEL: Record<string, string> = {
  haematoxylin: 'haematoxylin',
  dab: 'DAB',
  eosin: 'eosin',
}

/** The two stains an H-DAB assay actually carries. Eosin is the control. */
const ASSAY_STAINS = new Set(['haematoxylin', 'dab'])

/**
 * Ruifrok's vectors, so a measured arm can be shown beside the colour it is being
 * compared against.
 *
 * Duplicated from the server rather than read off `cloud.references`, because
 * those carry the *nearer arm's* angle and not a per-stain identity a swatch can
 * be keyed on. Constants either way, and published ones: if these drift from
 * `REFERENCE_VECTORS` in density.py the swatch pairs stop meaning anything, which
 * is why they are named the same thing in both places.
 */
const REFERENCE_VECTORS: Record<string, Channels> = {
  haematoxylin: { r: 0.65, g: 0.704, b: 0.286 },
  dab: { r: 0.268, g: 0.57, b: 0.776 },
  eosin: { r: 0.072, g: 0.99, b: 0.105 },
}

/**
 * The optical density a swatch is drawn at.
 *
 * A direction has no colour on its own - only a direction *and* an amount do. 0.9
 * is chosen to be recognisable rather than to be typical: it is well above step
 * 4's noise floor and below the point where every stain crushes towards black, so
 * haematoxylin reads as blue-violet and DAB as brown, which is what a pathologist
 * would call them. The caption says the number, because a swatch at a different
 * density would be a different colour and the reader should know which one this is.
 */
const SWATCH_OD = 0.9

/**
 * The colour a pixel of one stain alone would be, from that stain's direction.
 *
 * Beer-Lambert, run backwards. `OD = -log10(I / I0)` per channel, so a pixel whose
 * density is `d` along unit direction `v` transmits `10^(-d * v)` of the incident
 * light in each channel - and against a white I0 that is the pixel's own colour.
 * This is the same arithmetic step 5 does forwards, which is the point: the swatch
 * is not an illustration of the vector, it *is* the vector.
 */
function stainColour(vector: Channels, od: number = SWATCH_OD): string {
  const channel = (value: number) =>
    Math.round(255 * Math.pow(10, -od * Math.max(0, value)))
  return `rgb(${channel(vector.r)}, ${channel(vector.g)}, ${channel(vector.b)})`
}

/**
 * Top of the match gauge's scale, in degrees.
 *
 * Ruifrok's haematoxylin and DAB sit 43 degrees apart in three dimensions, so 45
 * is "as far apart as two different stains" - a fixed ceiling every gauge on the
 * card shares, which is what makes their markers comparable by eye.
 */
const GAUGE_MAX = 45

/**
 * Below this many degrees an arm is treated as landing on its stain.
 *
 * Read as a band on the gauge rather than as a pass/fail test - the badge's
 * verdict turns on which reference is *nearest*, not on this number. It is here so
 * that "close" has a visible extent instead of being a feeling about a figure.
 */
const GAUGE_MATCH = 20

/** One angle on the shared 0-45° scale, with the matching band behind it. */
function MatchGauge({ degrees, tone }: { degrees: number; tone: 'on' | 'off' }) {
  const position = `${Math.min(100, (degrees / GAUGE_MAX) * 100).toFixed(2)}%`

  return (
    <div className="od-gauge" role="img" aria-label={`${degrees.toFixed(1)} degrees of ${GAUGE_MAX}`}>
      <span className="od-gauge__band" style={{ width: `${(GAUGE_MATCH / GAUGE_MAX) * 100}%` }} />
      <span className={`od-gauge__mark od-gauge__mark--${tone}`} style={{ left: position }} />
    </div>
  )
}

/* --- the wedge drawing ----------------------------------------------------- */

/** The apex, in viewBox units. Both wedges open from it, because it is zero stain. */
const WEDGE_APEX = { x: 26, y: 120 } as const

/** Longest ray the box holds, and the furthest a ray may travel from the axis. */
const WEDGE_RADIUS = 196
const WEDGE_REACH = 96

/**
 * Both wedges as coordinates, from the two angles actually measured.
 *
 * Computed rather than drawn once, because the whole claim of the figure is the
 * comparison between two numbers that change with the tile - a wedge sketched at
 * one slide's separation would be a picture of a different tile on every other
 * one. The radius shrinks as the wedge widens so the rays stay inside the box: the
 * *angle* is the content, and it is preserved exactly either way.
 */
function wedgeGeometry(separation: number, reference: number) {
  const widest = (Math.max(separation, reference) / 2) * (Math.PI / 180)
  const radius = Math.min(WEDGE_RADIUS, WEDGE_REACH / Math.max(Math.sin(widest), 1e-3))

  const ray = (degrees: number, side: 1 | -1) => {
    const half = (degrees / 2) * (Math.PI / 180)
    return {
      x: WEDGE_APEX.x + radius * Math.cos(half),
      y: WEDGE_APEX.y - side * radius * Math.sin(half),
    }
  }

  return {
    measured: { upper: ray(separation, 1), lower: ray(separation, -1) },
    published: { upper: ray(reference, 1), lower: ray(reference, -1) },
    /** Only meaningful when the measured wedge is the wider of the two. */
    excess: separation > reference,
  }
}

const point = ({ x, y }: { x: number; y: number }) => `${x.toFixed(1)},${y.toFixed(1)}`

export function ArmVerdict({ cloud }: ArmVerdictProps) {
  const onTarget = cloud.arms.every((arm) => ASSAY_STAINS.has(arm.nearest))
  const wide = cloud.separation > 1.4 * cloud.referenceSeparation
  const wedge = wedgeGeometry(cloud.separation, cloud.referenceSeparation)

  // Keep the two ray labels off each other when the wedges are close in width.
  const measuredLabelY = wedge.measured.upper.y
  const publishedLabelY = Math.max(wedge.published.upper.y, measuredLabelY + 18)

  return (
    <div className="od-arms">
      <div className="od-arms__head">
        <span className="eyebrow">do the colours found match the expected stains?</span>
        <Badge tone={cloud.twoArmed && onTarget ? 'success' : 'warn'}>
          {!cloud.twoArmed
            ? 'only one colour found'
            : onTarget
              ? 'both colours match the expected stains'
              : 'one colour does not match'}
        </Badge>
      </div>

      <p className="od-arms__lead">
        We did not tell the software what the stains look like. It found the two main
        colour directions in this tile on its own. If they land close to the known stain
        colours, the tile is what we expect it to be.
      </p>

      <div className="od-arms__list">
        {cloud.arms.map((arm) => {
          const on = ASSAY_STAINS.has(arm.nearest)
          const reference = REFERENCE_VECTORS[arm.nearest]

          return (
            <div
              className={on ? 'od-arm od-arm--on' : 'od-arm'}
              key={arm.angle}
            >
              {/* The claim, as two colours. Everything below is the same claim in
                  numbers, and a reader who trusts their eyes is already done. */}
              <div className="od-arm__swatches">
                <span
                  className="od-arm__chip"
                  style={{ background: stainColour(arm.vector) }}
                  aria-hidden
                />
                <span
                  className="od-arm__chip od-arm__chip--reference"
                  style={{
                    background: reference ? stainColour(reference) : 'transparent',
                  }}
                  aria-hidden
                />
                <span className="od-arm__swatch-key">
                  found <span className="od-arm__swatch-sep">/</span> expected
                </span>
              </div>

              <p className="od-arm__claim">
                <span className="od-arm__degrees mono">
                  {arm.degreesFromNearest.toFixed(1)}°
                </span>
                <span className="od-arm__from">
                  from {STAIN_LABEL[arm.nearest] ?? arm.nearest}
                </span>
              </p>

              <MatchGauge degrees={arm.degreesFromNearest} tone={on ? 'on' : 'off'} />
              <div className="od-arm__scale mono">
                <span>0° = exact match</span>
                <span>{GAUGE_MAX}° = a different stain</span>
              </div>

              <span className="od-arm__vector mono">
                ({arm.vector.r.toFixed(3)}, {arm.vector.g.toFixed(3)},{' '}
                {arm.vector.b.toFixed(3)})
              </span>
            </div>
          )
        })}
      </div>

      {/* The two shares as one bar rather than as a sentence in each card: they
          sum to the tile, and two percentages in two places do not say so. */}
      <div className="od-split">
        <span className="eyebrow">how the tile splits between the two stains</span>
        <div className="od-split__bar">
          {cloud.arms.map((arm) => (
            <span
              key={arm.angle}
              className="od-split__seg"
              style={{
                width: `${(arm.share * 100).toFixed(2)}%`,
                background: stainColour(arm.vector, 0.55),
              }}
              title={`${(arm.share * 100).toFixed(0)}% nearer the ${STAIN_LABEL[arm.nearest] ?? arm.nearest} edge`}
            />
          ))}
        </div>
        <div className="od-split__keys">
          {cloud.arms.map((arm) => (
            <span className="od-split__key" key={arm.angle}>
              <i style={{ background: stainColour(arm.vector, 0.55) }} aria-hidden />
              <strong className="mono">{(arm.share * 100).toFixed(0)}%</strong>{' '}
              mostly {STAIN_LABEL[arm.nearest] ?? arm.nearest}
            </span>
          ))}
        </div>
      </div>

      <details className="od-detail">
        <summary>The technical detail</summary>
        <div className="od-detail__body">

      <div className="od-wedge">
        <figure className="od-wedge__figure">
          <svg
            viewBox="0 0 360 240"
            className="od-wedge__svg"
            role="img"
            aria-label={`The measured wedge spans ${cloud.separation.toFixed(0)} degrees against the published pair's ${cloud.referenceSeparation.toFixed(0)}`}
          >
            {/* Both wedges from one apex, because the apex is zero stain for both
                and the comparison is only meaningful there. */}
            <polygon
              className="od-wedge__fill"
              points={`${point(WEDGE_APEX)} ${point(wedge.measured.upper)} ${point(wedge.measured.lower)}`}
            />

            {/* The difference between the two, shaded. This is what "wider" is. */}
            {wedge.excess && (
              <>
                <polygon
                  className="od-wedge__excess"
                  points={`${point(WEDGE_APEX)} ${point(wedge.measured.upper)} ${point(wedge.published.upper)}`}
                />
                <polygon
                  className="od-wedge__excess"
                  points={`${point(WEDGE_APEX)} ${point(wedge.measured.lower)} ${point(wedge.published.lower)}`}
                />
              </>
            )}

            <line
              className="od-wedge__arm"
              x1={WEDGE_APEX.x}
              y1={WEDGE_APEX.y}
              x2={wedge.measured.upper.x}
              y2={wedge.measured.upper.y}
            />
            <line
              className="od-wedge__arm"
              x1={WEDGE_APEX.x}
              y1={WEDGE_APEX.y}
              x2={wedge.measured.lower.x}
              y2={wedge.measured.lower.y}
            />

            <line
              className="od-wedge__ref"
              x1={WEDGE_APEX.x}
              y1={WEDGE_APEX.y}
              x2={wedge.published.upper.x}
              y2={wedge.published.upper.y}
            />
            <line
              className="od-wedge__ref"
              x1={WEDGE_APEX.x}
              y1={WEDGE_APEX.y}
              x2={wedge.published.lower.x}
              y2={wedge.published.lower.y}
            />

            <circle
              className="od-wedge__apex"
              cx={WEDGE_APEX.x}
              cy={WEDGE_APEX.y}
              r="4"
            />

            <text
              className="od-wedge__label"
              x={wedge.measured.upper.x + 8}
              y={measuredLabelY}
            >
              {cloud.separation.toFixed(0)}° measured here
            </text>
            <text
              className="od-wedge__label od-wedge__label--dim"
              x={wedge.published.upper.x + 8}
              y={publishedLabelY}
            >
              {cloud.referenceSeparation.toFixed(0)}° expected
            </text>
            <text className="od-wedge__label od-wedge__label--dim" x="18" y="146">
              zero stain
            </text>
          </svg>
          <figcaption>
            Solid: the two colours found in this tile. Dashed: where the standard pair
            would sit. Shaded: the difference.
          </figcaption>
        </figure>

        <div className="od-wedge__read">
          {wide ? (
            <>
              <p>
                The two colours are <strong>further apart</strong> than the standard
                pair. That is a question, not a better result. Three things cause it:
              </p>
              <ol className="od-causes">
                <li>
                  <span className="od-causes__what">a third substance</span>
                  something in the tile that is neither stain
                </li>
                <li>
                  <span className="od-causes__what">very dark pixels</span>
                  where a pixel is nearly black, its colour is mostly rounding error
                  <span className="od-causes__handled mono">
                    {(cloud.unstableShare * 100).toFixed(1)}% already excluded
                  </span>
                </li>
                <li>
                  <span className="od-causes__what">the brown marker itself</span>
                  its colour genuinely shifts as it gets stronger
                </li>
              </ol>
            </>
          ) : (
            <p>
              The two colours are <strong>closer together</strong> than the standard
              pair, so they overlap more than usual on this tile. They can still be
              separated, just with less margin.
            </p>
          )}
        </div>
      </div>

      <div className="od-refs">
        <span className="eyebrow">the standard stain colours, and how well they fit</span>

        <div className="od-refs__grid">
          <span className="od-refs__col mono">stain</span>
          <span className="od-refs__col mono">distance from nearest match</span>
          <span className="od-refs__col mono">how much sits off the chart</span>

          {cloud.references.map((reference) => {
            const control = !ASSAY_STAINS.has(reference.name)
            return (
              <div className="od-refs__row" key={reference.name}>
                <span className="od-refs__name">
                  <i
                    className="od-refs__chip"
                    style={{ background: stainColour(reference.vector) }}
                    aria-hidden
                  />
                  {STAIN_LABEL[reference.name] ?? reference.name}
                  {control && <em className="od-refs__tag">control</em>}
                </span>

                <span className="od-refs__angle">
                  <span className="mono">{reference.degreesFromArm.toFixed(1)}°</span>
                  <MatchGauge
                    degrees={reference.degreesFromArm}
                    tone={control ? 'off' : 'on'}
                  />
                </span>

                <span className="od-refs__plane">
                  <span className="mono">{(reference.outOfPlane * 100).toFixed(0)}%</span>
                  <span className="od-refs__meter" aria-hidden>
                    <i style={{ width: `${(reference.outOfPlane * 100).toFixed(0)}%` }} />
                  </span>
                  <span className="od-refs__plane-read">
                    {reference.outOfPlane < 0.1
                      ? 'shown accurately'
                      : reference.outOfPlane < 0.3
                        ? 'slightly off the chart'
                        : 'mostly off the chart'}
                  </span>
                </span>
              </div>
            )
          })}
        </div>

        <p className="od-refs__note">
          The chart is flat but the colours live in three dimensions, so the last
          column says how much of each stain is hidden behind the page. A stain that
          looks close on screen but sits far off the chart is not really a match.
          Eosin is the control: on a slide with a brown marker it should <em>not</em>{' '}
          match anything.
        </p>
      </div>

      <p className="od-arms__rule">
        These colours are for checking, not for measuring. Working out the stain
        colours from each image separately would give every slide its own scale, so
        the same number would mean different amounts of stain on different slides.
        The next step measures with <strong>fixed</strong> standard colours instead,
        and what you see here is the evidence that those fixed colours fit this slide.
      </p>

      <p className="od-arms__foot mono">
        each swatch shows one stain on its own — a colour direction means nothing
        without an amount
      </p>

        </div>
      </details>
    </div>
  )
}
