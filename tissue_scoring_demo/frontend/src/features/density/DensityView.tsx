/**
 * Step 5's screen, in the order the argument runs.
 *
 *   1. the headline — the equation, and the two numbers that make it a measurement
 *   2. the triptych — RGB tile → density heatmap → the point cloud with its arms
 *   3. the arm verdict — are the arms the stains, in degrees, against the published
 *      vectors
 *   4. the staining verdict — which dyes those arms mean the section carries, named
 *      in words on the screen that measured it rather than only on step 7's, which
 *      is where it used to first appear
 *   5. the tile chooser — which field of view, why, and every alternative clickable
 *   6. the limits — where this stops being a measurement, and the inverse that
 *      proves the rest of it is
 *   7. the citation
 *
 * 3 and 4 are deliberately adjacent and in that order. The arms are the evidence and
 * the staining verdict is the conclusion drawn from them, so putting the conclusion
 * anywhere else would separate a claim from the only numbers that support it.
 *
 * The headline pairs I₀ with the noise floor deliberately, exactly as step 4's
 * does: a density is *defined* against I₀, so the number that produced every figure
 * on the screen has to be visible on it, and the floor is what says which of those
 * figures could be nothing at all.
 */

import { Badge } from '@/components/ui/Badge'
import type { DensityOptions } from '@/api/density'
import { formatCount } from '@/lib/format'
import type { DensityReport } from '@/types/density'

import { ArmVerdict } from './ArmVerdict'
import { DensityLimits } from './DensityLimits'
import { DensityTriptych } from './DensityTriptych'
import { StainingVerdict } from './StainingVerdict'
import { TileChooser } from './TileChooser'
import type { TilePick } from './useOpticalDensity'

import './density.css'

interface DensityViewProps {
  report: DensityReport
  refining: boolean
  onPick: (tile: TilePick | null) => void
}

export function DensityView({ report, refining, onPick }: DensityViewProps) {
  const { params, white, tile, stats, cloud, staining } = report

  const options: DensityOptions = {
    threshold:
      params.tissueThresholdSource === 'manual' ? params.tissueThreshold : null,
    // Both or neither, and only when the viewer moved off the score's own pick -
    // otherwise the URL would pin a tile the server chose, and a re-run after a
    // threshold change would keep a tile that no longer scores best.
    x: tile.requested ? tile.x : null,
    y: tile.requested ? tile.y : null,
  }

  return (
    // `aria-busy` and one class for the whole screen, because a pick invalidates
    // the whole screen. Every figure here - the p99, the arm separation, the
    // limits, the round-trip - is a statement about one field of view, so while a
    // new one is being read they are all describing a tile the viewer has already
    // moved off. The alternative is worse than dimming: numbers that quietly
    // belong to the previous tile, beside a map already outlining the next.
    <div className={refining ? 'od od--refining' : 'od'} aria-busy={refining}>
      {/* --- 1. the headline ---------------------------------------------- */}
      <div className="od-headline">
        {/* The equation and the figures share one row, and the prose sits under
            both. A two-column grid with the prose spanning would leave the shorter
            column padded out with dead space, because a grid row is as tall as its
            tallest cell. */}
        <div className="od-headline__top">
        <div className="od-headline__equation">
          <span className="od-headline__formula mono">stain = −log₁₀(pixel / blank glass)</span>
          <span className="od-headline__i0 mono">
            blank glass = ({white.rgb.r.toFixed(0)}, {white.rgb.g.toFixed(0)},{' '}
            {white.rgb.b.toFixed(0)})
            {white.mode === 'surface' && ' — varies across the slide'}
          </span>
          <span
            className="od-headline__chip"
            style={{ background: white.hex }}
            role="img"
            aria-label="the blank-glass colour this tile was compared against"
          />
        </div>

        <div className="od-headline__figures">
          <div className="od-headline__figure">
            <span className="od-headline__value mono">{stats.meanP99.toFixed(2)}</span>
            <span className="od-headline__caption">strongest stain in this tile</span>
            <span className="od-headline__sub mono">
              typical {stats.meanMedian.toFixed(3)} · anything under{' '}
              {white.noiseFloor.toFixed(3)} is noise
            </span>
          </div>

          <div className="od-headline__figure od-headline__figure--muted">
            <span className="od-headline__value mono">
              {cloud ? `${cloud.separation.toFixed(0)}°` : '—'}
            </span>
            <span className="od-headline__caption">between the two stain colours</span>
            <span className="od-headline__sub mono">
              {cloud
                ? `${formatCount(cloud.plotted)} pixels used · ${cloud.referenceSeparation.toFixed(0)}° expected`
                : 'not enough stain in this tile to tell'}
            </span>
          </div>
        </div>
        </div>

        <p className="od-headline__why">
          Colour on its own does not tell you how much stain there is: twice the stain
          is not twice the colour. This step converts each pixel into a stain amount by
          comparing it with blank glass. Once that is done, two stains simply add up,
          which is what lets the next step pull them apart.
        </p>
      </div>

      {/* --- 2. the triptych ---------------------------------------------- */}
      <DensityTriptych report={report} options={options} refining={refining} />

      {/* --- 3. the arms -------------------------------------------------- */}
      {cloud && (
        <div className="od-stale">
          <ArmVerdict cloud={cloud} />
        </div>
      )}

      {/* --- 4. which dyes, in words -------------------------------------- */}
      {/* Both or neither. `staining` is null exactly when no cloud was built, and a
          verdict with no arms behind it would be a claim with its evidence missing -
          which is the one thing this card is for. */}
      {cloud && staining && (
        <div className="od-stale">
          <StainingVerdict staining={staining} cloud={cloud} />
        </div>
      )}

      {/* --- 5. which tile, and why --------------------------------------- */}
      <TileChooser
        report={report}
        options={options}
        refining={refining}
        onPick={onPick}
      />

      {/* --- 6. where it stops being a measurement ------------------------ */}
      <div className="od-stale">
        <DensityLimits report={report} options={options} />
      </div>

      {/* --- 7. the citation ---------------------------------------------- */}
      <p className="od-notes__citation">
        <Badge tone="neutral">
          Ruifrok &amp; Johnston 2001 · Macenko 2009 · Beer 1852
        </Badge>{' '}
        {report.citation}
      </p>
    </div>
  )
}
