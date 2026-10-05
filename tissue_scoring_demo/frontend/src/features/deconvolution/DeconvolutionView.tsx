/**
 * Step 6's screen, written for someone who has never read a pathology paper.
 *
 *   1. what just happened, in two sentences
 *   2. the pictures — mixed, blue, brown (immunostained slides only), left over
 *   3. where the fork goes, because this is the step the pipeline splits at
 *   4. the colour choice, and the two scores it produces
 *   5. three checks, in plain words
 *   6. the technical detail, folded away for anyone who wants it
 *
 * The deliberate departure from step 5's screen is the language. Step 5 argues a
 * point about Beer–Lambert to a reader willing to follow it; this step has a
 * simpler and more important thing to convey — the stains are now separate, and
 * the choice of what the colours mean changes the answer — so every number on
 * screen is captioned in ordinary words and the vector arithmetic lives behind a
 * disclosure. Nothing is hidden that changes the conclusion; what is folded away
 * is the justification, not the result.
 */

import { Badge } from '@/components/ui/Badge'
import type { DeconvolutionOptions } from '@/api/deconvolution'
import type { DeconvolutionBasisName, DeconvolutionReport } from '@/types/deconvolution'

import { BasisToggle } from './BasisToggle'
import { StainStrip } from './StainStrip'

import './deconvolution.css'

interface DeconvolutionViewProps {
  report: DeconvolutionReport
  basis: DeconvolutionBasisName
  onBasisChange: (basis: DeconvolutionBasisName) => void
  /**
   * Which slide these pictures came from.
   *
   * It decides both the wording and which panels are drawn, and that is the whole
   * point: the brown panel is the DAB channel, and on an H&E section there is no DAB.
   * Calling it "the picture the score is measured from" there is false, and drawing it
   * at all is worse — the panel is stretched to its own range, so a dye that was never
   * applied comes out looking like a confident brown picture. The score is measured on
   * the immunostained slide. Same arithmetic, two entirely different claims.
   */
  slideRole: 'he' | 'ihc' | 'unknown'
}

export function DeconvolutionView({
  report,
  basis,
  onBasisChange,
  slideRole,
}: DeconvolutionViewProps) {
  const { fixed, tile, params } = report

  const options: DeconvolutionOptions = {
    threshold: null,
    x: tile.x,
    y: tile.y,
  }

  const isHe = slideRole === 'he'
  const dab = fixed.channels.find((channel) => channel.name === 'dab')
  const haem = fixed.channels.find((channel) => channel.name === 'haematoxylin')
  const negatives = Math.max(
    ...fixed.channels
      .filter((channel) => channel.name !== 'residual')
      .map((channel) => channel.negativeShare),
  )

  return (
    <div className="cd">
      {/* --- 1. what just happened ---------------------------------------- */}
      <div className="cd-headline">
        {isHe ? (
          <>
            <p className="cd-headline__lead">
              This is the <strong>H&amp;E</strong> slide. It has two dyes: a{' '}
              <strong>blue</strong> one on the cell nuclei and a pink one on everything
              around them. They are mixed together in every pixel.
            </p>
            <p className="cd-headline__lead">
              This step pulls the blue out on its own. There is no brown marker on this
              slide &mdash; that is measured on the marker slides of the same case.
              Nothing here is guessed or trained: it is arithmetic that can be undone
              exactly.
            </p>
          </>
        ) : (
          <>
            <p className="cd-headline__lead">
              This slide has two dyes: a <strong>blue</strong> one on every cell nucleus,
              and a <strong>brown</strong> one that only appears where the marker is
              present. In the scan they are mixed in every pixel, so a dark blue nucleus
              with a little brown looks much like a moderately brown one.
            </p>
            <p className="cd-headline__lead">
              This step splits them into two separate pictures, so the brown can be
              measured on its own. Nothing here is guessed or trained: it is arithmetic
              that can be undone exactly.
            </p>
          </>
        )}

        <div className="cd-headline__figures">
          {isHe ? (
            <>
              <div className="cd-headline__figure">
                <span className="cd-headline__value mono">
                  {(tile.stainedShare * 100).toFixed(0)}%
                </span>
                <span className="cd-headline__caption">of this tile carries stain</span>
                <span className="cd-headline__sub">
                  the rest is background and is not measured
                </span>
              </div>

              <div className="cd-headline__figure">
                <span className="cd-headline__value mono">
                  {haem ? haem.p99.toFixed(2) : '—'}
                </span>
                <span className="cd-headline__caption">strongest blue reading</span>
                <span className="cd-headline__sub">
                  a typical stained pixel reads {haem ? haem.median.toFixed(2) : '—'}
                </span>
              </div>
            </>
          ) : (
            <>
              <div className="cd-headline__figure">
                <span className="cd-headline__value mono">
                  {(fixed.preview.positiveShare * 100).toFixed(0)}%
                </span>
                <span className="cd-headline__caption">
                  of the stained tissue is positive
                </span>
                <span className="cd-headline__sub">
                  any pixel whose brown reading passes {params.positiveCut}
                </span>
              </div>

              <div className="cd-headline__figure">
                <span className="cd-headline__value mono">
                  {dab ? dab.p99.toFixed(2) : '—'}
                </span>
                <span className="cd-headline__caption">strongest brown reading</span>
                <span className="cd-headline__sub">
                  a typical stained pixel reads {dab ? dab.median.toFixed(2) : '—'}
                </span>
              </div>
            </>
          )}

          <div className="cd-headline__figure">
            <span className="cd-headline__value mono">
              {(fixed.residualShare * 100).toFixed(0)}%
            </span>
            <span className="cd-headline__caption">left over</span>
            <span className="cd-headline__sub">
              the two dyes explain the rest of the colour
            </span>
          </div>
        </div>
      </div>

      {/* --- 2. the pictures ---------------------------------------------- */}
      <StainStrip report={report} basis={basis} options={options} slideRole={slideRole} />

      {/* --- 3. the fork -------------------------------------------------- */}
      <section className="cd-fork">
        <h3 className="cd-card__title">The two pictures go to different places</h3>
        <div className="cd-fork__body">
          <div className="cd-fork__arm">
            <span className="cd-fork__tag cd-fork__tag--h">Blue picture</span>
            <p>
              Goes to the model that finds the tumour, instead of the colour photo. With
              the brown removed, the same model can be used on every slide in the panel.
            </p>
          </div>
          <div className="cd-fork__arm">
            <span className="cd-fork__tag cd-fork__tag--dab">Brown picture</span>
            {isHe ? (
              <p>
                Comes from the marker slides of this case, not from this one. Every later
                step that counts or scores reads that picture, never the colour photo, so
                a mixture is never mistaken for a strong result.
              </p>
            ) : (
              <p>
                Goes to the measurement. Every later step that counts or scores reads this
                picture, never the colour photo, so a mixture is never mistaken for a
                strong result.
              </p>
            )}
          </div>
        </div>
        <p className="cd-card__note">
          Both come from the same calculation on the same pixels, so the two pictures
          can never disagree about what is on the slide.
        </p>
      </section>

      {/* --- 4. the choice that changes the answer ------------------------ */}
      <BasisToggle report={report} basis={basis} onChange={onBasisChange} />

      {/* --- 5. the checks, in plain words -------------------------------- */}
      <section className="cd-checks">
        <h3 className="cd-card__title">Three checks on this result</h3>
        <ul className="cd-checks__list">
          <li>
            <strong>Nothing was lost.</strong> Adding the pictures back together returns
            the original tile to within {fixed.exactnessMax.toExponential(0)}, which is
            rounding error. This step separates the stains; it does not sharpen, clean or
            filter the image.
          </li>
          <li>
            <strong>The dyes explain the slide.</strong> Only{' '}
            {(fixed.residualShare * 100).toFixed(0)}% of the colour is left over.
            {fixed.residualShare > 0.15
              ? ' That is more than a clean slide should leave, so something here is neither dye. Check the last picture above.'
              : ' A nearly empty last picture is what you want to see.'}
          </li>
          <li>
            <strong>Impossible readings are shown, not hidden.</strong>{' '}
            {negatives > 0.005
              ? `${(negatives * 100).toFixed(0)}% of stained pixels came out with a negative amount of one dye, which cannot happen. Those pixels are an unusual colour the two standard colours cannot describe. They are counted here rather than quietly rounded up to zero.`
              : 'Almost no pixel came out with a negative amount of a dye, so the standard colours fit this slide well.'}
          </li>
        </ul>
      </section>

      {/* --- 6. the detail, for anyone who wants it ----------------------- */}
      <details className="cd-detail">
        <summary>The technical detail</summary>
        <div className="cd-detail__body">
          <p className="cd-detail__intro">
            The same result in technical language. Nothing here changes the answer.
          </p>
          <ul className="cd-detail__notes">
            {report.notes.map((note) => (
              <li key={note.slice(0, 48)}>{note}</li>
            ))}
          </ul>

          <table className="cd-detail__table">
            <caption>
              The stain colours actually used &mdash; one per row.
            </caption>
            <thead>
              <tr>
                <th scope="col">Stain</th>
                <th scope="col">R</th>
                <th scope="col">G</th>
                <th scope="col">B</th>
                <th scope="col">Typical</th>
                <th scope="col">Strongest</th>
                <th scope="col">Negative</th>
              </tr>
            </thead>
            <tbody>
              {fixed.channels.map((channel) => (
                <tr key={channel.name}>
                  <th scope="row">{channel.name === 'dab' ? 'DAB' : channel.name}</th>
                  <td className="mono">{channel.vector.r.toFixed(3)}</td>
                  <td className="mono">{channel.vector.g.toFixed(3)}</td>
                  <td className="mono">{channel.vector.b.toFixed(3)}</td>
                  <td className="mono">{channel.median.toFixed(3)}</td>
                  <td className="mono">{channel.p99.toFixed(3)}</td>
                  <td className="mono">{(channel.negativeShare * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="cd-detail__foot">
            Tile at ({tile.x}, {tile.y}), {tile.size} px at {tile.mpp.toFixed(2)} µm/px,
            covering {tile.tileUm.toFixed(0)} µm. The figures cover the{' '}
            {(tile.stainedShare * 100).toFixed(0)}% of it that carries stain. How much
            the two channels overlapped: {report.mixedCorrelation.toFixed(2)} before
            separating, {fixed.channelCorrelation.toFixed(2)} after.
          </p>
        </div>
      </details>

      <p className="cd-citation">
        <Badge tone="neutral">Ruifrok &amp; Johnston 2001</Badge> {report.citation}
      </p>
    </div>
  )
}
