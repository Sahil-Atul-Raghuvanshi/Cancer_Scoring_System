/**
 * Step 3's screen, in the order the argument runs.
 *
 *   1. the headline - how much of the slide survives, and what that buys
 *   2. the panels - slide, saturation channel, mask on the slide
 *   3. the histogram - Otsu's criterion, Otsu's cut, and the viewer's
 *   4. the cleanup ladder - each morphological move and what it changed
 *   5. the citation
 *
 * Rule 1 - fat is tissue and stays in this mask - is not on screen any more, but
 * it still binds the code: nothing here removes a region for being pale, and the
 * enclosed-void fill exists to put fat lobules and gland lumina back. Fat leaves
 * at step 9, as a class the model names.
 *
 * The threshold comparison is the point of the step, so it is never hidden: the
 * Otsu value and the manual value sit next to each other, with the variance
 * ratio between them turning "Otsu lands where a human would" into a number.
 */

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { formatCount } from '@/lib/format'
import type { TissueReport } from '@/types/tissue'

import { CleanupLadder } from './CleanupLadder'
import { SaturationHistogram } from './SaturationHistogram'
import { TissuePanels } from './TissuePanels'

import './tissue.css'

interface TissueViewProps {
  report: TissueReport
  draft: number
  committed: number
  manual: boolean
  refining: boolean
  onDraft: (value: number) => void
  onResetToAutomatic: () => void
}

export function TissueView({
  report,
  draft,
  committed,
  manual,
  refining,
  onDraft,
  onResetToAutomatic,
}: TissueViewProps) {
  const { threshold, components } = report

  // The slider is ahead of the server. Everything numeric on this screen still
  // belongs to `committed`, so nothing may be phrased as if it described `draft`
  // - quoting the previous cut's variance ratio next to a new slider position is
  // the one dishonesty this whole screen exists to avoid.
  const pending = draft !== committed
  // Stale from the moment the slider moves, not from when the request goes out:
  // the debounce is part of the wait and these figures are the previous cut's for
  // all of it. The rule cards read off the committed report, so the highlight
  // moves onto the new rule exactly when this clears.
  const busy = pending || refining

  // Which card the numbers on screen actually came from - read off the report,
  // not off `manual`, which flips the instant the viewer touches the slider.
  const liveCut = threshold.source
  const agrees = threshold.varianceRatio >= 0.995

  return (
    <div className="tissue">
      {/* --- 1. the headline ---------------------------------------------- */}
      <div className="tissue-headline">
        <div className="tissue-headline__figure">
          <span className="tissue-headline__value">
            {(report.tissueShare * 100).toFixed(1)}%
          </span>
          <span className="tissue-headline__caption">of the slide is tissue</span>
          <span className="tissue-headline__sub mono">
            {report.tissueAreaMm2.toFixed(1)} mm² of {report.slideAreaMm2.toFixed(0)} mm²
          </span>
        </div>

        <div className="tissue-headline__arrow" aria-hidden>
          →
        </div>

        <div className="tissue-headline__figure tissue-headline__figure--muted">
          <span className="tissue-headline__value">
            {(report.glassShare * 100).toFixed(1)}%
          </span>
          <span className="tissue-headline__caption">glass, discarded here</span>
          <span className="tissue-headline__sub mono">
            area steps 4 onward never visit
          </span>
        </div>

        <p className="tissue-headline__why">
          This is the whole reason step 3 runs before anything expensive. Every learned
          step downstream costs per unit area, and this one is a few seconds of arithmetic
          that removes most of the area — without needing to know anything about what the
          tissue <em>is</em>.
        </p>
      </div>

      {/* --- 2. the panels ------------------------------------------------ */}
      {/* Dimmed from the moment the slider moves, not just once the request is
          in flight: the debounce is part of the wait, and these pixels are the
          previous cut's for all of it. */}
      <TissuePanels report={report} refining={refining || pending} />

      {/* --- 3. the threshold -------------------------------------------- */}
      <div className="tissue-threshold">
        <SaturationHistogram
          histogram={report.histogram}
          threshold={threshold}
          draft={draft}
          committed={committed}
          onDraft={onDraft}
        />

        <div
          className={busy ? 'tissue-threshold__side is-busy' : 'tissue-threshold__side'}
          aria-busy={busy}
        >
          {busy && (
            <span className="is-busy__marker">
              <Spinner size="sm" label={`thresholding at ${draft}`} />
            </span>
          )}

          <div className="tissue-cuts is-busy__figures">
            <div
              className={
                liveCut === 'otsu' ? 'tissue-cut tissue-cut--live' : 'tissue-cut'
              }
            >
              <span className="tissue-cut__label">Otsu</span>
              <span className="tissue-cut__value mono">{threshold.otsu}</span>
              <span className="tissue-cut__hint">
                the best split of two humps — sound only if the histogram has two
              </span>
            </div>

            <div
              className={
                liveCut === 'triangle' ? 'tissue-cut tissue-cut--live' : 'tissue-cut'
              }
            >
              <span className="tissue-cut__label">Triangle</span>
              <span className="tissue-cut__value mono">{threshold.triangle}</span>
              <span className="tissue-cut__hint">
                where the tail departs furthest from the chord — for one peak and a tail
              </span>
            </div>

            <div
              className={
                liveCut === 'manual' ? 'tissue-cut tissue-cut--live' : 'tissue-cut'
              }
            >
              <span className="tissue-cut__label">Yours</span>
              <span className="tissue-cut__value mono">{draft}</span>
              <span className="tissue-cut__hint">
                {pending
                  ? 'not run yet — the figures below are still the previous cut'
                  : liveCut === 'manual'
                    ? 'this mask is the one your number produces'
                    : 'drag the histogram to move it'}
              </span>
            </div>
          </div>

          {/* The shape statistic the rule choice turned on. Stated as a number,
              because "the histogram is not bimodal" is otherwise an assertion. */}
          <p className="tissue-shape">
            <span className="mono">{(threshold.modalShare * 100).toFixed(1)}%</span> of the
            counted pixels sit on level{' '}
            <span className="mono">{threshold.modalLevel}</span> alone
            {threshold.bimodal ? (
              <>
                {' '}
                — under the <span className="mono">
                  {(threshold.spikeShare * 100).toFixed(0)}%
                </span>{' '}
                mark, so this histogram really is two humps and Otsu is the right rule for it.
              </>
            ) : (
              <>
                {' '}
                — over the <span className="mono">
                  {(threshold.spikeShare * 100).toFixed(0)}%
                </span>{' '}
                mark, so this is a spike and a tail, not two humps. Otsu&rsquo;s criterion is a
                ratio of variances and a spike has none, so it runs out into the tail. The
                triangle rule was used instead.
              </>
            )}
          </p>

          <p className="tissue-threshold__verdict">
            {pending ? (
              <>
                Thresholding at <strong>{draft}</strong>… Every figure on this screen still
                belongs to the cut at <strong>{committed}</strong>, and will until the server
                has actually built the mask — this step will not show you a number it has not
                computed.
              </>
            ) : liveCut === 'otsu' ? (
              <>
                Otsu&rsquo;s cut sits at <strong>{threshold.otsu}</strong>, in the valley
                between the two humps. Drag it and watch both the mask and the area move —
                and watch how far you have to go before the picture stops looking right.
              </>
            ) : liveCut === 'triangle' ? (
              <>
                The cut sits at <strong>{threshold.triangle}</strong>, on the shoulder where
                the background spike stops and the tissue&rsquo;s tail begins. Otsu would have
                said <strong>{threshold.otsu}</strong> — drag the cut up there and watch the
                mask lose most of the section. That is the whole reason the rule was chosen
                from the histogram&rsquo;s shape rather than assumed.
              </>
            ) : threshold.bimodal && agrees ? (
              <>
                Your cut scores <strong>{(threshold.varianceRatio * 100).toFixed(1)}%</strong>{' '}
                of Otsu&rsquo;s criterion, which is to say you and the algorithm have landed
                in the same place. That agreement is the argument for not hand-picking the
                number in the first place.
              </>
            ) : threshold.bimodal ? (
              <>
                Your cut scores{' '}
                <strong>{(threshold.varianceRatio * 100).toFixed(1)}%</strong> of Otsu&rsquo;s
                criterion — a worse split of this histogram than the automatic one. The mask
                above is the mask your number produces.
              </>
            ) : (
              <>
                Your cut is <strong>{threshold.value}</strong>, against the triangle
                rule&rsquo;s <strong>{threshold.triangle}</strong> and Otsu&rsquo;s{' '}
                <strong>{threshold.otsu}</strong>. No criterion score is quoted for it: on a
                spike-and-tail histogram Otsu&rsquo;s criterion is the wrong yardstick, and
                scoring against it would dress up &ldquo;disagrees with Otsu&rdquo; as
                &ldquo;is worse&rdquo;.
              </>
            )}
          </p>

          {/* The ratio is only a score when the shape Otsu assumes is present,
              so on a spike-and-tail histogram it is not offered as one. */}
          {threshold.meanBelow !== null && threshold.meanAbove !== null && (
            <dl className="tissue-facts mono is-busy__figures">
              <div>
                <dt>mean below</dt>
                <dd>{threshold.meanBelow.toFixed(1)}</dd>
              </div>
              <div>
                <dt>mean above</dt>
                <dd>{threshold.meanAbove.toFixed(1)}</dd>
              </div>
              <div>
                <dt>separation</dt>
                <dd>{threshold.separation?.toFixed(1)}</dd>
              </div>
            </dl>
          )}

          {/* Offered for as long as the cut is off the automatic answer, the
              wait included. The value comes off `bimodal` and not off `source`,
              so it names the rule the slide's own histogram shape chose rather
              than whatever is on screen now. */}
          {manual && (
            <Button variant="secondary" onClick={onResetToAutomatic} loading={refining}>
              Hand it back to the {threshold.bimodal ? 'Otsu' : 'triangle'} rule (
              {threshold.bimodal ? threshold.otsu : threshold.triangle})
            </Button>
          )}
        </div>
      </div>

      {/* --- 4. the cleanup ---------------------------------------------- */}
      <CleanupLadder stages={report.stages} />

      <div className="tissue-components">
        <span className="eyebrow">connected components</span>
        <dl className="tissue-facts tissue-facts--wide mono">
          <div>
            <dt>found</dt>
            <dd>{formatCount(components.found)}</dd>
          </div>
          <div>
            <dt>kept</dt>
            <dd>{formatCount(components.kept)}</dd>
          </div>
          <div>
            <dt>dropped</dt>
            <dd>
              {formatCount(components.dropped)}{' '}
              <span className="tissue-facts__aside">
                ({components.droppedAreaMm2.toFixed(3)} mm²)
              </span>
            </dd>
          </div>
          <div>
            <dt>largest</dt>
            <dd>
              {components.largestAreaMm2.toFixed(2)} mm²{' '}
              <span className="tissue-facts__aside">
                ({(components.largestShare * 100).toFixed(0)}% of tissue)
              </span>
            </dd>
          </div>
          <div>
            <dt>cutoff</dt>
            <dd>
              {components.minAreaMm2} mm²{' '}
              <span className="tissue-facts__aside">
                = {formatCount(components.minAreaPx)} px here
              </span>
            </dd>
          </div>
        </dl>
      </div>

      {/* --- 5. the citation ---------------------------------------------- */}
      <p className="tissue-notes__citation">
        <Badge tone="neutral">Otsu 1979 · Zack 1977</Badge> {report.citation}
      </p>
    </div>
  )
}
