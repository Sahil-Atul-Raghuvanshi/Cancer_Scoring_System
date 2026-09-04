/**
 * The one screen in the demo that shows a choice changing the answer.
 *
 * Two ways to decide what "blue" and "brown" mean. Use the standard published
 * colours, or work them out from each slide's own colours. Both are legitimate
 * image processing; only one of them can be measured against, and this is where a
 * viewer sees why rather than being told.
 *
 * The design has one job: make the two scores impossible to read as one number.
 * They sit side by side, the same size, with the gap between them written out —
 * because "68.6%" beside "83.1%" on the same tile, under the same rule, is the
 * entire argument for using fixed colours, and any layout that buries either half
 * loses it.
 *
 * Flipping the toggle changes the pictures above too. That is deliberate and it
 * is free: both bases came down in one response, and each panel URL is its own
 * cache entry, so switching back and forth is instant after the first look.
 */

import type { DeconvolutionBasisName, DeconvolutionReport } from '@/types/deconvolution'

interface BasisToggleProps {
  report: DeconvolutionReport
  basis: DeconvolutionBasisName
  onChange: (basis: DeconvolutionBasisName) => void
}

export function BasisToggle({ report, basis, onChange }: BasisToggleProps) {
  const { comparison, estimated, fixed } = report

  // No estimate on this tile. The step still worked — the published colours do not
  // depend on the tile at all, which is the property being argued for — so this
  // says so rather than hiding the section or showing a dead control.
  if (!comparison || !estimated) {
    return (
      <section className="cd-card">
        <h3 className="cd-card__title">Where do the colours come from?</h3>
        <p className="cd-card__lead">
          This step used the standard published colours for blue and brown &mdash; the
          same numbers on every slide, which is what makes two slides comparable.
        </p>
        <p className="cd-card__note">
          On this particular tile there was not enough stain to work out a second,
          slide-specific set of colours to compare against. That does not affect the
          result: the published colours do not depend on the tile, which is exactly
          the point of using them. Pick a tile with more staining on step 5 to see
          the comparison.
        </p>
      </section>
    )
  }

  const shift = Math.abs(comparison.shareShift) * 100
  const higher = comparison.shareShift > 0

  return (
    <section className="cd-card">
      <h3 className="cd-card__title">Where do the colours come from?</h3>
      <p className="cd-card__lead">
        To split the stains apart, the software has to be told what &ldquo;blue&rdquo;
        and &ldquo;brown&rdquo; look like. There are two ways to do that, and{' '}
        <strong>they give different answers on the same tile</strong>. Switch between
        them:
      </p>

      <div className="cd-toggle" role="group" aria-label="Which stain colours to use">
        <button
          type="button"
          className={basis === 'fixed' ? 'cd-toggle__pick is-active' : 'cd-toggle__pick'}
          aria-pressed={basis === 'fixed'}
          onClick={() => onChange('fixed')}
        >
          <span className="cd-toggle__name">Standard colours</span>
          <span className="cd-toggle__score mono">
            {(fixed.preview.positiveShare * 100).toFixed(1)}%
          </span>
          <span className="cd-toggle__sub">
            The same published numbers on every slide
          </span>
        </button>

        <button
          type="button"
          className={
            basis === 'estimated' ? 'cd-toggle__pick is-active' : 'cd-toggle__pick'
          }
          aria-pressed={basis === 'estimated'}
          onClick={() => onChange('estimated')}
        >
          <span className="cd-toggle__name">Colours from this slide</span>
          <span className="cd-toggle__score mono">
            {(estimated.preview.positiveShare * 100).toFixed(1)}%
          </span>
          <span className="cd-toggle__sub">
            Worked out from this tile, so different on every slide
          </span>
        </button>
      </div>

      <p className="cd-card__verdict">
        Same tile. Same maths. Same threshold for &ldquo;this pixel counts as
        positive&rdquo;. Only the colour definitions changed &mdash; and the result
        moved by <strong>{shift.toFixed(1)} percentage points</strong>
        {higher ? ' upwards' : ' downwards'}.
      </p>

      <p className="cd-card__note">
        That is why this pipeline always uses the <strong>standard colours</strong>.
        Working the colours out per slide fits each slide a little better, but it
        also gives each slide its own scale &mdash; so a strongly stained slide and a
        weakly stained one can come out looking the same, and two patients&rsquo;
        results can no longer be compared. The slide-specific colours here sit{' '}
        {Math.max(comparison.haematoxylinDegrees, comparison.dabDegrees).toFixed(0)}°
        away from the standard ones and read the brown{' '}
        {comparison.dabScale < 1 ? 'weaker' : 'stronger'} by a factor of{' '}
        {comparison.dabScale.toFixed(2)}.
      </p>
    </section>
  )
}
