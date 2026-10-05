/**
 * The pipeline guide's triptych, in its order: RGB tile → density heatmap → the
 * point cloud.
 *
 * The sequence carries the argument by itself — colour, then how much stain, then
 * the geometry that makes un-mixing possible — which is why it is a strip and not
 * three cards a reader might take in any order.
 *
 * The scatter is a component rather than an image because it needs labelled
 * overlays; the first two are images because nothing should be drawn on them. The
 * tile in particular is served untouched, with no contrast curve applied for
 * legibility, so a reader comparing the heatmap against it is comparing a density
 * against its actual input.
 */

import { densityPanelUrl } from '@/api/density'
import type { DensityOptions } from '@/api/density'
import type { DensityReport } from '@/types/density'

import { ODScatter } from './ODScatter'

interface DensityTriptychProps {
  report: DensityReport
  options: DensityOptions
  refining: boolean
}

export function DensityTriptych({ report, options, refining }: DensityTriptychProps) {
  const { cloud, stats, params } = report

  return (
    <div className={refining ? 'od-strip od-strip--stale' : 'od-strip'}>
      <figure className="od-strip__item">
        <div className="od-strip__frame">
          <img
            src={densityPanelUrl(report.uploadId, 'tile', options)}
            alt="The tile exactly as it was scanned"
          />
        </div>
        <figcaption>
          <span className="od-strip__step mono">1</span>
          <strong>Colour</strong>
          <span className="od-strip__caption">
            A {params.tileUm.toFixed(0)} µm square of the slide, exactly as scanned. The
            other two panels come from these same pixels.
          </span>
        </figcaption>
      </figure>

      <figure className="od-strip__item">
        <div className="od-strip__frame">
          <img
            src={densityPanelUrl(report.uploadId, 'density', options)}
            alt="The same tile, showing how much stain is in each pixel"
          />
        </div>
        <figcaption>
          <span className="od-strip__step mono">2</span>
          <strong>How much stain</strong>
            <span className="od-strip__caption">
            Brighter means more stain, up to {stats.meanP99.toFixed(2)}. A dark nucleus
            and a dark shadow look the same in panel 1, but not here &mdash; a shadow
            holds no stain.
          </span>
        </figcaption>
      </figure>

      <figure className="od-strip__item od-strip__item--wide">
        <div className="od-strip__frame od-strip__frame--plot">
          {cloud ? (
            <ODScatter
              report={report}
              cloud={cloud}
              options={options}
              refining={refining}
            />
          ) : (
            <div className="od-strip__empty">
              <p>
                Too few pixels in this tile carry enough stain to plot, so there is
                nothing to draw. That is a real answer, not a failure &mdash; a pale
                tile simply has no stain colours in it. Pick another tile below.
              </p>
            </div>
          )}
        </div>
        <figcaption>
          <span className="od-strip__step mono">3</span>
          <strong>The two stains, as a chart</strong>
          <span className="od-strip__caption">
            {cloud
              ? `${cloud.plotted.toLocaleString('en-GB')} pixels plotted. The two straight edges are the two stains on their own; everything between them is a mixture. This shape is what lets the next step separate them.`
              : 'This chart is where the two stains would show up. Not on this tile.'}
          </span>
        </figcaption>
      </figure>
    </div>
  )
}
