/**
 * Step 16's screen: the deliverable, with no hidden arithmetic behind it.
 *
 * The pair comes first and large, because it is the whole output — two numbers,
 * and everything else on the page exists to make them legible or to say what is
 * wrong with them.
 *
 * Then the cascade, written out with the actual numbers substituted in: counts,
 * the formula, the rounding as its own line, the pair. A reader should be able
 * to follow it end to end without taking any step on trust, and the rounding is
 * shown rather than applied quietly because it changes the answer.
 *
 * Then the caveats, before the reference scores rather than after them. A
 * provisional cut point or a denominator missing 70 % of its cells is not a
 * footnote to these numbers; it is a condition on reading them at all.
 *
 * The H-score, the Allred score and the ASCO/CAP call come last, under a
 * heading that says they are not the deliverable. OncoStem's own sheet has ten
 * columns and no combined score anywhere.
 */

import { useState } from 'react'

import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import type { MarkerScore } from '@/types/scores'

import type { ScoresStateValue } from './useScores'

import './scores.css'

interface ScorePanelProps {
  scores: ScoresStateValue
  /** Step 15 must have binned the cells there is anything to aggregate. */
  hasBins: boolean
}

function Heterogeneity({ score }: { score: MarkerScore }) {
  const tiles = score.heterogeneity
  if (tiles.length === 0) return null

  const shade = (percent: number) => {
    const t = Math.max(0, Math.min(1, percent / 100))
    return `rgba(201, 138, 58, ${0.12 + t * 0.85})`
  }

  return (
    <figure className="sc-hetero">
      <div className="sc-hetero__grid">
        {tiles.map((tile) => (
          <div
            key={`${tile.regionRank}-${tile.fieldIndex}`}
            className="sc-hetero__tile"
            style={{ background: shade(tile.percent) }}
            title={`Region ${tile.regionRank}, field ${tile.fieldIndex + 1}: ${tile.percent}% of ${tile.cells} cells positive`}
          >
            <span>{Math.round(tile.percent)}</span>
          </div>
        ))}
      </div>
      <figcaption>
        Each square is one place on the slide that was checked, shaded by how many of its
        cells were positive. The reported {score.percent}% is an average of these. They
        are clearly not all the same, which is what a single percentage hides.
      </figcaption>
    </figure>
  )
}

export function ScorePanel({ scores, hasBins }: ScorePanelProps) {
  const [showDetail, setShowDetail] = useState(false)
  const { report, loading, error } = scores

  if (!hasBins) {
    return (
      <div className="sc-panel sc-panel--waiting">
        <p>The cells have not been graded yet. Go back a step first.</p>
      </div>
    )
  }
  if (loading) {
    return (
      <div className="sc-panel sc-panel--waiting">
        <Spinner />
        <p>Working out the score.</p>
      </div>
    )
  }
  if (error) {
    return (
      <div className="sc-panel sc-panel--error">
        <p>{error}</p>
      </div>
    )
  }
  if (!report) return null

  const score = report.score

  return (
    <div className="sc-panel">
      <header className="sc-panel__head">
        <div>
          <h3>The two numbers this system reports</h3>
          <p>
            Each marker gets two numbers, so five markers give ten numbers for a case.
            There is deliberately no single combined score: merging the two would throw
            away information.
          </p>
        </div>
        <Badge tone={score.cutsProvisional ? 'warn' : 'accent'}>{score.markerName}</Badge>
      </header>

      <div className="sc-pair">
        <div className="sc-pair__card">
          <span className="sc-pair__eyebrow">Percent positive</span>
          <strong>{score.percent}%</strong>
          <span className="sc-pair__sub">
            of {score.cells.toLocaleString()} tumour cells showed the marker in the right
            part of the cell
          </span>
        </div>
        <div className="sc-pair__card">
          <span className="sc-pair__eyebrow">Staining strength</span>
          <strong>{score.intensity}</strong>
          <span className="sc-pair__sub">
            on the 0–2 scale{score.intensityLabel ? ` — ${score.intensityLabel}` : ''}
          </span>
        </div>
      </div>

      <section className="sc-block">
        <h4>How those two numbers were reached</h4>
        <ol className="sc-cascade">
          {score.cascade.map((step) => (
            <li key={step.label}>
              <span className="sc-cascade__label">{step.label}</span>
              <span className="sc-cascade__expr">{step.expression}</span>
              <span className="sc-cascade__value">{step.value}</span>
            </li>
          ))}
        </ol>
          <p className="sc-note">
          Both roundings matter. Pathologists report percentages in multiples of 5, and
          119 of 120 real readings land exactly on one of the six allowed strength values.
          An unrounded 61.7% and 1.34 would not look more precise &mdash; it would look
          like a different measurement from the one that was asked for.
        </p>
      </section>

      {score.caveats.length > 0 ? (
        <section className="sc-block sc-caveats">
          <h4>Read this before using these numbers</h4>
          <ul>
            {score.caveats.map((caveat) => (
              <li key={caveat}>{caveat}</li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="sc-block">
        <h4>How the result varies across the slide</h4>
        <Heterogeneity score={score} />
      </section>

      <section className="sc-block sc-reference">
        <h4>Reference scores &mdash; not the reported result</h4>
        <p className="sc-note">
          These are the standard scores used in this field, shown so the two numbers above
          can be compared against something familiar. None of them is an output of this
          system, and the ASCO/CAP call is not a HER2 result &mdash; none of these five
          markers is HER2.
        </p>
        <dl className="sc-reference__grid">
          <div>
            <dt>H-score</dt>
            <dd>{score.hScore} <small>of 300</small></dd>
          </div>
          <div>
            <dt>Allred</dt>
            <dd>
              {score.allredTotal} <small>of 8 ({score.allredProportion} + {score.allredIntensity})</small>
            </dd>
          </div>
          <div>
            <dt>ASCO/CAP-style call</dt>
            <dd>{score.her2Call ?? 'n/a'} <small>{score.her2Call ? '' : 'cytoplasmic marker'}</small></dd>
          </div>
        </dl>
      </section>

      <details
        className="sc-detail"
        open={showDetail}
        onToggle={(event) => setShowDetail(event.currentTarget.open)}
      >
        <summary>The technical detail</summary>
        <dl>
          <dt>Cells by grade</dt>
          <dd>
            0: {score.binCounts[0]?.toLocaleString()} · 1+:{' '}
            {score.binCounts[1]?.toLocaleString()} · 2+:{' '}
            {score.binCounts[2]?.toLocaleString()} · 3+:{' '}
            {score.binCounts[3]?.toLocaleString()}
          </dd>

          <dt>Combining the regions</dt>
          <dd>
            weighted by area {score.percentAreaWeighted}% (this is the reported one) · all
            cells pooled together {score.percentPooled}% · plain average of the regions{' '}
            {score.percentPlainMean}%.
          </dd>

          <dt>Partly stained cells</dt>
          <dd>
            rule in use: {score.partialRule}.{' '}
            {Object.entries(score.percentByPartialRule)
              .map(([rule, value]) => `${rule} → ${value}%`)
              .join(' · ')}
          </dd>

          <dt>Cut points</dt>
          <dd>
            {score.odCuts.map((cut) => cut.toFixed(2)).join(' / ')}, with{' '}
            {score.secondMeasure.replace(/_/g, ' ')} ≥ {score.secondMin}
            {score.cutsProvisional ? ' — provisional' : ''}.
          </dd>

          <dt>Regions</dt>
          <dd>
            {score.regions
              .map(
                (region) =>
                  `region ${region.rank}: ${region.percentRaw}% of ${region.cells.toLocaleString()} cells over ${region.areaMm2.toFixed(2)} mm²`,
              )
              .join('; ')}
          </dd>
        </dl>
        <ul>
          {report.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      </details>
    </div>
  )
}
