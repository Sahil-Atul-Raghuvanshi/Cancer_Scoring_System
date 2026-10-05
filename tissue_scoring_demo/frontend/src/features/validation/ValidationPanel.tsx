/**
 * Step 17's screen: whether the numbers agree with pathologists.
 *
 * The screen has one job and it is mostly an honest one. There is no reader
 * sheet in this repository, so what it usually shows is that nothing has been
 * validated — stated plainly, on its own, rather than softened or buried under
 * the statistics it would have computed.
 *
 * That is deliberate. Without this step the pipeline is a rendering engine, not
 * a measurement tool, and a screen that quietly omitted the absence would let a
 * viewer leave the walkthrough believing the numbers had been checked.
 *
 * When a sheet is present, the standard is inter-pathologist agreement, not a
 * ground truth: four trained readers produce four different numbers on the same
 * slide, and the spread between them is the resolution of the measurement.
 */

import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'

import type { CaseScoreReport } from '@/types/scores'

import type { ValidationStateValue } from './useValidation'

import './validation.css'

interface ValidationPanelProps {
  validation: ValidationStateValue
  /** Step 16 must have produced a score there is anything to compare. */
  hasScore: boolean
}

/**
 * What this case scored, per marker — the numbers agreement is measured over.
 *
 * Step 16 reports one marker, because one marker's slide is loaded. This step is the
 * only one keyed on the case, and a verdict shown without the numbers it is a verdict
 * on asks the reader to take it on trust. Markers that have not been run yet are
 * listed with their state rather than omitted: a partial grid is the honest picture of
 * a case in progress, and an absent row is indistinguishable from a marker that
 * scored nothing.
 */
function CaseGrid({ report }: { report: CaseScoreReport }) {
  if (report.rows.length === 0) return null

  return (
    <section className="vl-case">
      <div className="vl-case__head">
        <h4>This case, marker by marker</h4>
        <span className="mono vl-case__id">{report.caseId}</span>
        {!report.complete ? <Badge tone="warn">partial</Badge> : null}
      </div>
      <table className="vl-case__table">
        <thead>
          <tr>
            <th scope="col">Marker</th>
            <th scope="col">Positive</th>
            <th scope="col">Intensity</th>
            <th scope="col">Cells</th>
          </tr>
        </thead>
        <tbody>
          {report.rows.map((row) => (
            <tr key={row.marker} className={row.percent === null ? 'is-unscored' : undefined}>
              <th scope="row">
                <span className="mono">{row.marker}</span> {row.markerName}
              </th>
              <td className="mono">
                {row.percent === null ? '—' : `${row.percent.toFixed(1)}%`}
              </td>
              <td>{row.percent === null ? row.state : row.intensityLabel}</td>
              <td className="mono">{row.cells ? row.cells.toLocaleString() : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {report.notes.length > 0 ? (
        <ul className="vl-case__notes">
          {report.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      ) : null}
    </section>
  )
}

export function ValidationPanel({ validation, hasScore }: ValidationPanelProps) {
  const { report, caseScores, loading, error } = validation

  if (!hasScore) {
    return (
      <div className="vl-panel vl-panel--waiting">
        <p>There is no score yet. Go back and calculate the score first.</p>
      </div>
    )
  }
  if (loading) {
    return (
      <div className="vl-panel vl-panel--waiting">
        <Spinner />
        <p>Looking for pathologist readings to compare against.</p>
      </div>
    )
  }
  if (error) {
    return (
      <div className="vl-panel vl-panel--error">
        <p>{error}</p>
      </div>
    )
  }
  if (!report) return null

  return (
    <div className="vl-panel">
      <header className="vl-panel__head">
        <div>
          <h3>Do these numbers agree with pathologists?</h3>
          <p>
            This is the only step that can tell you whether all the earlier ones were
            right.
          </p>
        </div>
        <Badge tone={report.available ? 'accent' : 'warn'}>
          {report.available ? 'compared' : 'not validated'}
        </Badge>
      </header>

      {caseScores ? <CaseGrid report={caseScores} /> : null}

      {!report.available ? (
        <section className="vl-absent">
          <h4>Nothing has been compared yet</h4>
          <p>{report.reason}</p>
          <p>
            Until pathologist readings are loaded, these numbers only show that the
            calculation runs. Nobody has checked them against a human reader.
          </p>
        </section>
      ) : (
        <>
          <p className="vl-note">
            The target is within ±10 percentage points of what the pathologists agreed,
            because that is how much four trained pathologists differ from each other on
            the same slides. Inside that range is as close as humans get to each other.
          </p>
          <table className="vl-table">
            <thead>
              <tr>
                <th>Marker</th>
                <th>Cases</th>
                <th>Average difference</th>
                <th>Within ±10</th>
                <th>Worst difference</th>
                <th>Spread between pathologists</th>
                <th>Strength agreement</th>
              </tr>
            </thead>
            <tbody>
              {report.markers.map((entry) => (
                <tr key={entry.marker}>
                  <td>{entry.marker}</td>
                  <td>{entry.cases}</td>
                  <td>
                    {entry.percentBias >= 0 ? '+' : ''}
                    {entry.percentBias}
                  </td>
                  <td>
                    {entry.percentWithinTolerance} of {entry.cases}
                  </td>
                  <td>{entry.percentMaxError}</td>
                  <td>{entry.readerSpread ?? '—'}</td>
                  <td>
                    {entry.intensityKappa === null
                      ? 'undefined'
                      : entry.intensityKappa.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <ul className="vl-notes">
        {report.notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>

      {report.cutsProvisional ? (
        <p className="vl-flag">
          The cut points behind these scores have not been tuned against pathologist
          readings yet. So any agreement figure above is measuring untuned settings, which
          is worth knowing whichever way it comes out.
        </p>
      ) : null}
    </div>
  )
}
