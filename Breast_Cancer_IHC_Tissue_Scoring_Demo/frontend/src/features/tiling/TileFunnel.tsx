/**
 * The funnel — the reason this step runs where it does, as three numbers.
 *
 * Drawn as three bars on one shared scale rather than three separate figures,
 * because the *drop* is the content and a drop is a shape before it is an
 * arithmetic fact. A reader sees the second bar as a sliver of the first and has
 * understood the argument before reading a single label.
 *
 * The bars are widths on the whole-grid count, so they are directly comparable —
 * the commonest mistake here would be scaling each bar to its own maximum, which
 * would draw three full-width bars and show nothing at all.
 */

import { formatCount } from '@/lib/format'
import type { TilingReport } from '@/types/tiling'

interface TileFunnelProps {
  report: TilingReport
}

export function TileFunnel({ report }: TileFunnelProps) {
  const { funnel, params } = report
  const total = Math.max(funnel.every, 1)

  const rows = [
    {
      key: 'every',
      label: 'Every square of the grid',
      value: funnel.every,
      note: 'Including the empty glass, which is most of the scan',
      tone: 'off',
    },
    {
      key: 'tissue',
      label: 'Squares with tissue on them',
      value: funnel.onTissue,
      note: `At least ${(params.minTissueShare * 100).toFixed(0)}% tissue, from step 3's map`,
      tone: 'mid',
    },
    {
      key: 'clean',
      label: 'Squares the model will actually see',
      value: funnel.clean,
      note: params.qcGated
        ? 'Also clear enough of the flaws step 2 found'
        : 'Step 2 has not run, so nothing was removed for flaws',
      tone: 'kept',
    },
  ] as const

  return (
    <section className="tl-funnel">
      <h3 className="tl-card__title">Cutting the slide down to what matters</h3>
      <p className="tl-card__lead">
        The model looks at one small square at a time. Laid over the whole scan
        that would be a very large number of squares &mdash; and most of them are
        empty glass. Because the earlier steps already worked out where the tissue
        is and which parts are damaged, most of those squares never have to be
        looked at:
      </p>

      <ul className="tl-funnel__rows">
        {rows.map((row) => (
          <li key={row.key} className="tl-funnel__row">
            <div className="tl-funnel__head">
              <span className="tl-funnel__label">{row.label}</span>
              <span className="tl-funnel__value mono">{formatCount(row.value)}</span>
            </div>
            <div className="tl-funnel__track">
              <span
                className={`tl-funnel__bar tl-funnel__bar--${row.tone}`}
                style={{ width: `${Math.max((row.value / total) * 100, 0.4)}%` }}
              />
            </div>
            <span className="tl-funnel__note">{row.note}</span>
          </li>
        ))}
      </ul>

      <p className="tl-card__verdict">
        <strong>{funnel.reduction.toFixed(0)}× less work.</strong> Each remaining
        square has to be run through the tissue-detection model one by one, and
        that is the slowest part of the whole pipeline &mdash; so this is where the
        time is saved. Doing the tiling before steps 2 and 3 would have meant
        paying for all {formatCount(funnel.every)}.
      </p>
    </section>
  )
}
