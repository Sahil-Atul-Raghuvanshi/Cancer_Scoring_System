/**
 * The step's picture: the mixed tile, then one picture per stain, then what is
 * left over.
 *
 * Read left to right it makes the whole argument without a single technical word.
 * The first panel is what the scanner saw — blue and brown tangled together in
 * every pixel. The next two are the same tile with each colour on its own. The
 * last one should look nearly empty, and "nearly empty" is the good outcome: it
 * means the two stains account for what was there.
 *
 * Each stain is drawn in its own dye's colour, so nobody has to consult a legend
 * to know which picture is which. Brighter always means more of that stain — a
 * single progression per panel, never a rainbow, because a rainbow invents bands
 * the data does not have.
 */

import { deconvolutionPanelUrl } from '@/api/deconvolution'
import type { DeconvolutionOptions } from '@/api/deconvolution'
import type { DeconvolutionBasisName, DeconvolutionReport } from '@/types/deconvolution'

interface StainStripProps {
  report: DeconvolutionReport
  basis: DeconvolutionBasisName
  options: DeconvolutionOptions
}

export function StainStrip({ report, basis, options }: StainStripProps) {
  const { tile } = report
  const active = basis === 'estimated' ? report.estimated : report.fixed

  const dab = active?.channels.find((channel) => channel.name === 'dab')
  const leftover = active?.residualShare ?? report.fixed.residualShare

  return (
    <div className="cd-strip">
      <figure className="cd-strip__item">
        <div className="cd-strip__frame">
          <img
            src={deconvolutionPanelUrl(report.uploadId, 'tile', basis, options)}
            alt="The tile as the scanner recorded it, with both stains mixed together"
          />
        </div>
        <figcaption>
          <span className="cd-strip__step mono">1</span>
          <strong>Both stains, mixed</strong>
          <span className="cd-strip__caption">
            {tile.tileUm.toFixed(0)} µm of the slide, exactly as scanned. Blue and
            brown sit on top of each other in every pixel here.
          </span>
        </figcaption>
      </figure>

      <figure className="cd-strip__item">
        <div className="cd-strip__frame">
          <img
            src={deconvolutionPanelUrl(report.uploadId, 'haematoxylin', basis, options)}
            alt="How much blue stain each pixel carries"
          />
        </div>
        <figcaption>
          <span className="cd-strip__step mono">2</span>
          <strong className="cd-strip__label cd-strip__label--h">Blue stain only</strong>
          <span className="cd-strip__caption">
            The counterstain, which marks where the cell nuclei are. This picture is
            what the tissue-detection model gets instead of the colour photo.
          </span>
        </figcaption>
      </figure>

      <figure className="cd-strip__item">
        <div className="cd-strip__frame">
          <img
            src={deconvolutionPanelUrl(report.uploadId, 'dab', basis, options)}
            alt="How much brown marker each pixel carries"
          />
        </div>
        <figcaption>
          <span className="cd-strip__step mono">3</span>
          <strong className="cd-strip__label cd-strip__label--dab">
            Brown marker only
          </strong>
          <span className="cd-strip__caption">
            The marker being tested for. This is the picture the score is measured
            from{dab ? `, and it peaks at ${dab.p99.toFixed(2)} here` : ''}.
          </span>
        </figcaption>
      </figure>

      <figure className="cd-strip__item">
        <div className="cd-strip__frame">
          <img
            src={deconvolutionPanelUrl(report.uploadId, 'residual', basis, options)}
            alt="What neither stain explains"
          />
        </div>
        <figcaption>
          <span className="cd-strip__step mono">4</span>
          <strong>Left over</strong>
          <span className="cd-strip__caption">
            Anything the two stains could not account for &mdash;{' '}
            {(leftover * 100).toFixed(0)}% of the colour here. Nearly empty is the
            result you want.
          </span>
        </figcaption>
      </figure>
    </div>
  )
}
