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
          <span className="tissue-headline__caption">glass, dropped here</span>
          <span className="tissue-headline__sub mono">
            later steps never look at this area
          </span>
        </div>

        <p className="tissue-headline__why">
          Later steps are slow and cost time per area. This one takes a few seconds and
          removes most of the slide, so everything after it runs on a much smaller picture.
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
                splits the chart into two groups &mdash; works when it has two clear humps
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
                for charts with one tall spike and a long tail
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
                  ? 'not applied yet — the numbers below are from the previous value'
                  : liveCut === 'manual'
                    ? 'the picture above uses your value'
                    : 'drag the chart to change it'}
              </span>
            </div>
          </div>

          {/* The shape statistic the rule choice turned on. Stated as a number,
              because "the histogram is not bimodal" is otherwise an assertion. */}
          <p className="tissue-shape">
            <span className="mono">{(threshold.modalShare * 100).toFixed(1)}%</span> of the
            pixels sit at one single value (
            <span className="mono">{threshold.modalLevel}</span>)
            {threshold.bimodal ? (
              <>
                {' '}
                &mdash; below the <span className="mono">
                  {(threshold.spikeShare * 100).toFixed(0)}%
                </span>{' '}
                mark, so the chart has two real humps and the Otsu rule fits it.
              </>
            ) : (
              <>
                {' '}
                &mdash; above the <span className="mono">
                  {(threshold.spikeShare * 100).toFixed(0)}%
                </span>{' '}
                mark, so this is one spike and a long tail, not two humps. The Otsu rule does
                not work on that shape, so the triangle rule was used instead.
              </>
            )}
          </p>

          <p className="tissue-threshold__verdict">
            {pending ? (
              <>
                Working at <strong>{draft}</strong>… Every number on this screen still belongs
                to <strong>{committed}</strong> until the new picture is ready. We never show
                a number we have not actually measured.
              </>
            ) : liveCut === 'otsu' ? (
              <>
                The automatic cut is <strong>{threshold.otsu}</strong>, in the dip between the
                two humps. Drag it and watch the picture and the area change.
              </>
            ) : liveCut === 'triangle' ? (
              <>
                The cut is <strong>{threshold.triangle}</strong>, where the background spike
                ends and the tissue begins. The Otsu rule would have said{' '}
                <strong>{threshold.otsu}</strong> &mdash; drag the cut up there and most of
                the tissue disappears. That is why the rule is picked from the chart shape.
              </>
            ) : threshold.bimodal && agrees ? (
              <>
                Your cut scores <strong>{(threshold.varianceRatio * 100).toFixed(1)}%</strong>{' '}
                against the automatic one, so you and the software agree. Good sign.
              </>
            ) : threshold.bimodal ? (
              <>
                Your cut scores{' '}
                <strong>{(threshold.varianceRatio * 100).toFixed(1)}%</strong> against the
                automatic one, so it splits the slide less cleanly. The picture above uses
                your value.
              </>
            ) : (
              <>
                Your cut is <strong>{threshold.value}</strong>. The triangle rule says{' '}
                <strong>{threshold.triangle}</strong> and the Otsu rule says{' '}
                <strong>{threshold.otsu}</strong>. No score is shown, because on a chart of
                this shape the Otsu score is not a fair comparison.
              </>
            )}
          </p>

          {/* The ratio is only a score when the shape Otsu assumes is present,
              so on a spike-and-tail histogram it is not offered as one. */}
          {threshold.meanBelow !== null && threshold.meanAbove !== null && (
            <dl className="tissue-facts mono is-busy__figures">
              <div>
                <dt>average below the cut</dt>
                <dd>{threshold.meanBelow.toFixed(1)}</dd>
              </div>
              <div>
                <dt>average above the cut</dt>
                <dd>{threshold.meanAbove.toFixed(1)}</dd>
              </div>
              <div>
                <dt>gap between them</dt>
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
              Back to the automatic value (
              {threshold.bimodal ? threshold.otsu : threshold.triangle})
            </Button>
          )}
        </div>
      </div>

      {/* --- 4. the cleanup ---------------------------------------------- */}
      <CleanupLadder stages={report.stages} />

      <div className="tissue-components">
        <span className="eyebrow">separate pieces of tissue</span>
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
            <dt>dropped as too small</dt>
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
            <dt>smallest piece kept</dt>
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
