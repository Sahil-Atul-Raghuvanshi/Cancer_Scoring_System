/**
 * Step 6's screen, written for someone who has never read a pathology paper.
 *
 *   1. what just happened, in two sentences
 *   2. the pictures — mixed, blue, brown, left over
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
}

export function DeconvolutionView({
  report,
  basis,
  onBasisChange,
}: DeconvolutionViewProps) {
  const { fixed, tile, params } = report

  const options: DeconvolutionOptions = {
    threshold: null,
    x: tile.x,
    y: tile.y,
  }

  const dab = fixed.channels.find((channel) => channel.name === 'dab')
  const negatives = Math.max(
    ...fixed.channels
      .filter((channel) => channel.name !== 'residual')
      .map((channel) => channel.negativeShare),
  )

  return (
    <div className="cd">
      {/* --- 1. what just happened ---------------------------------------- */}
      <div className="cd-headline">
        <p className="cd-headline__lead">
          Two dyes were used on this slide: a <strong>blue</strong> one that stains
          every cell nucleus, and a <strong>brown</strong> one that only appears where
          the marker being tested for is present. In the scan they are mixed together
          in every single pixel &mdash; a dark blue nucleus with a little brown on it
          looks much the same as a moderately brown one.
        </p>
        <p className="cd-headline__lead">
          This step pulls them apart into two separate pictures, so the brown can be
          measured on its own. <strong>Nothing is guessed and nothing is trained</strong>
          &nbsp;&mdash; it is one piece of arithmetic that can be undone exactly.
        </p>

        <div className="cd-headline__figures">
          <div className="cd-headline__figure">
            <span className="cd-headline__value mono">
              {(fixed.preview.positiveShare * 100).toFixed(0)}%
            </span>
            <span className="cd-headline__caption">of the stained tissue is positive</span>
            <span className="cd-headline__sub">
              counting any pixel whose brown reading passes {params.positiveCut}
            </span>
          </div>

          <div className="cd-headline__figure">
            <span className="cd-headline__value mono">
              {dab ? dab.p99.toFixed(2) : '—'}
            </span>
            <span className="cd-headline__caption">strongest brown reading</span>
            <span className="cd-headline__sub">
              typical stained pixel reads {dab ? dab.median.toFixed(2) : '—'}
            </span>
          </div>

          <div className="cd-headline__figure">
            <span className="cd-headline__value mono">
              {(fixed.residualShare * 100).toFixed(0)}%
            </span>
            <span className="cd-headline__caption">left unexplained</span>
            <span className="cd-headline__sub">
              the two dyes account for the rest of the colour
            </span>
          </div>
        </div>
      </div>

      {/* --- 2. the pictures ---------------------------------------------- */}
      <StainStrip report={report} basis={basis} options={options} />

      {/* --- 3. the fork -------------------------------------------------- */}
      <section className="cd-fork">
        <h3 className="cd-card__title">The two pictures go to different places</h3>
        <div className="cd-fork__body">
          <div className="cd-fork__arm">
            <span className="cd-fork__tag cd-fork__tag--h">Blue picture</span>
            <p>
              Goes to the model that finds the tumour, in place of the colour photo.
              Because the brown has been removed, the same model works on this slide
              and on every other stain in the panel &mdash; one model instead of six.
            </p>
          </div>
          <div className="cd-fork__arm">
            <span className="cd-fork__tag cd-fork__tag--dab">Brown picture</span>
            <p>
              Goes to the measurement. Every later step that counts, thresholds or
              scores reads this picture and never the colour photo, so a mixture is
              never mistaken for a strong result.
            </p>
          </div>
        </div>
        <p className="cd-card__note">
          Both come out of the one calculation above, on the same pixels. That matters:
          if the model&rsquo;s picture and the measurement&rsquo;s picture came from
          separate passes, they could quietly disagree about what is on the slide.
        </p>
      </section>

      {/* --- 4. the choice that changes the answer ------------------------ */}
      <BasisToggle report={report} basis={basis} onChange={onBasisChange} />

      {/* --- 5. the checks, in plain words -------------------------------- */}
      <section className="cd-checks">
        <h3 className="cd-card__title">Three checks on this result</h3>
        <ul className="cd-checks__list">
          <li>
            <strong>Nothing was lost.</strong> Adding the three pictures back together
            returns the original tile to within {fixed.exactnessMax.toExponential(0)}{' '}
            &mdash; which is rounding error and nothing else. This step separates the
            stains, it does not clean up, sharpen or filter the image.
          </li>
          <li>
            <strong>The two dyes explain the slide.</strong> Only{' '}
            {(fixed.residualShare * 100).toFixed(0)}% of the colour is left over.
            {fixed.residualShare > 0.15
              ? ' That is higher than a clean slide should leave, so something here is neither dye — worth checking the fourth picture above.'
              : ' A near-empty fourth picture is what that looks like.'}
          </li>
          <li>
            <strong>Impossible readings are reported, not hidden.</strong>{' '}
            {negatives > 0.005
              ? `${(negatives * 100).toFixed(0)}% of stained pixels came out with a negative amount of one dye, which cannot happen physically. Those pixels are a slightly odd colour that the two standard colours cannot describe. They are counted here rather than quietly rounded up to zero.`
              : 'Almost no pixel came out with a negative amount of a dye, so the standard colours describe this slide well.'}
          </li>
        </ul>
      </section>

      {/* --- 6. the detail, for anyone who wants it ----------------------- */}
      <details className="cd-detail">
        <summary>The technical detail</summary>
        <div className="cd-detail__body">
          <p className="cd-detail__intro">
            Everything above in the language of the papers it comes from. Nothing here
            changes the result &mdash; it is the justification, not the answer.
          </p>
          <ul className="cd-detail__notes">
            {report.notes.map((note) => (
              <li key={note.slice(0, 48)}>{note}</li>
            ))}
          </ul>

          <table className="cd-detail__table">
            <caption>
              The stain matrix actually used &mdash; one unit optical-density vector
              per column.
            </caption>
            <thead>
              <tr>
                <th scope="col">Channel</th>
                <th scope="col">R</th>
                <th scope="col">G</th>
                <th scope="col">B</th>
                <th scope="col">Median</th>
                <th scope="col">99th pct</th>
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
            covering {tile.tileUm.toFixed(0)} µm. Statistics are over the{' '}
            {(tile.stainedShare * 100).toFixed(0)}% of it carrying at least{' '}
            {params.beta} mean optical density. Channel correlation before un-mixing{' '}
            {report.mixedCorrelation.toFixed(2)}, after {fixed.channelCorrelation.toFixed(2)}.
          </p>
        </div>
      </details>

      <p className="cd-citation">
        <Badge tone="neutral">Ruifrok &amp; Johnston 2001</Badge> {report.citation}
      </p>
    </div>
  )
}
