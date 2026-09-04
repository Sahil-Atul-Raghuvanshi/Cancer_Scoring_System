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
            alt="The tile as it was read off the slide, unaltered"
          />
        </div>
        <figcaption>
          <span className="od-strip__step mono">1</span>
          <strong>Colour</strong>
          <span className="od-strip__caption">
            {params.tileUm.toFixed(0)} µm of slide at {params.tileMpp.toFixed(2)} µm/px,
            unaltered. Every other panel is derived from exactly these pixels.
          </span>
        </figcaption>
      </figure>

      <figure className="od-strip__item">
        <div className="od-strip__frame">
          <img
            src={densityPanelUrl(report.uploadId, 'density', options)}
            alt="The same tile as an optical density heatmap"
          />
        </div>
        <figcaption>
          <span className="od-strip__step mono">2</span>
          <strong>How much stain</strong>
          <span className="od-strip__caption">
            −log₁₀(I / I₀), stretched 0 to {stats.meanP99.toFixed(2)} OD. A dark
            nucleus and a dark shadow look alike in panel 1 and do not here — a
            shadow contains no absorber.
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
                Too few of this tile’s pixels carry enough stain for a point cloud,
                so there are no arms to draw. That is a real state rather than a
                failure — a tile of pale stroma has no stain vectors in it — and
                inventing two arms through noise is the one thing this picture must
                not do. Pick another tile below.
              </p>
            </div>
          )}
        </div>
        <figcaption>
          <span className="od-strip__step mono">3</span>
          <strong>Why un-mixing is possible</strong>
          <span className="od-strip__caption">
            {cloud
              ? `${cloud.plotted.toLocaleString('en-GB')} pixels in density space. Two rays from zero stain, a wedge of mixtures between them — this geometry is what step 6 inverts, and it is either in the data or it is not.`
              : 'The cloud is where step 6’s method would be visible. Not on this tile.'}
          </span>
        </figcaption>
      </figure>
    </div>
  )
}
