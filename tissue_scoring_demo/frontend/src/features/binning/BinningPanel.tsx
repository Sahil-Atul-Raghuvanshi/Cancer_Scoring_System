/**
 * Step 15's screen: turning a measurement into a level.
 *
 * The screen has to keep two things apart that look alike, because conflating
 * them is the common bug this step is organised against. Each **cell** gets 0 /
 * 1+ / 2+ / 3+ — internal machinery for counting positives. The **slide** gets
 * an intensity on OncoStem's 0–2 scale, and that is the thing reported, decided
 * once at step 16. They are on different scales and read by different people, so
 * they are in separate blocks with a sentence between them saying so.
 *
 * The second half of the screen is the argument for absolute cut points, made by
 * showing rather than asserting: what per-slide percentiles and what one shared
 * table would have reported on these same cells, with the difference in points.
 */

import { useMemo, useState } from 'react'

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { SlideOverlayViewer } from '@/components/viewer/SlideOverlayViewer'
import { FlyTo } from '@/features/cells/FlyTo'
import type { FlyTarget } from '@/features/cells/overlay'
import { CELL_COLOUR, cellKey, flyTargets, nucleiOverlay, regionOutlines } from '@/features/cells/overlay'
import type { BinningReport } from '@/types/binning'
import type { AlignmentReport } from '@/types/ihcAlignment'
import type { RegionNucleiPayload } from '@/types/nuclei'

import type { BinningStateValue } from './useBinning'

import './binning.css'

interface BinningPanelProps {
  binning: BinningStateValue
  /** Step 14 must have measured the cells there is anything to bin. */
  hasMeasurements: boolean
  ihcUploadId: string | null
  /** The IHC slide's level-0 microns per pixel, for the scale bar. */
  ihcMpp: number | null
  /** Step 10's report, for the region borders drawn on the slide. */
  alignment: AlignmentReport | null
  /** Step 11's outlines - the shapes this step paints its levels onto. */
  geometry: Record<number, RegionNucleiPayload>
}

const BIN_COPY: Record<string, string> = {
  '0': 'no brown worth counting',
  '1+': 'faint brown',
  '2+': 'clearly brown',
  '3+': 'strongly brown',
}

function CutHistogram({ report }: { report: BinningReport }) {
  const bars = report.histogram
  if (bars.length === 0) return null
  const tallest = Math.max(...bars.map((bar) => bar.count), 1)
  // The `bars.length === 0` return above guarantees a last bar; indexing cannot
  // express that, so read it once and let the fallback stand for the impossible case.
  const span = bars.at(-1)?.upper || 1
  const at = (od: number) => `${Math.min(100, (od / span) * 100)}%`

  return (
    <figure className="bn-hist">
      <div className="bn-hist__bars">
        {bars.map((bar) => {
          const level = report.params.odCuts.filter((cut) => bar.lower >= cut).length
          return (
            <div
              key={bar.lower}
              className={`bn-hist__bar bn-hist__bar--l${level}`}
              style={{ height: `${(bar.count / tallest) * 100}%` }}
              title={`${bar.count} cells between ${bar.lower.toFixed(2)} and ${bar.upper.toFixed(2)}`}
            />
          )
        })}

        {report.valleyOd !== null ? (
          <div className="bn-hist__valley" style={{ left: at(report.valleyOd) }}>
            <span>dip between the two groups</span>
          </div>
        ) : null}

        {report.cutLines.map((line) => (
          <div key={line.od} className="bn-hist__cut" style={{ left: at(line.od) }}>
            <span>{line.separates}</span>
          </div>
        ))}
      </div>
      <figcaption>
        How strong the brown is across all {report.cells.toLocaleString()} cells, with the
        three cut lines on it.{' '}
        {report.valleyOd === null ? (
          <>
            There is only one hump here, not two, so stained and unstained cells are not
            clearly separated on this slide. Worth knowing before reading the score.
          </>
        ) : (
          <>
            A well-placed first cut sits in the dip between the pale hump and the brown
            one.
          </>
        )}
      </figcaption>
    </figure>
  )
}

export function BinningPanel({
  binning,
  hasMeasurements,
  ihcUploadId,
  ihcMpp,
  alignment,
  geometry,
}: BinningPanelProps) {
  const [showDetail, setShowDetail] = useState(false)
  const { report, cells, loading, error, selected, select } = binning

  const regions = useMemo(() => regionOutlines(alignment), [alignment])

  // Coloured by the level the server put each cell in - never by a level worked
  // out here. See `useBinning`: the rule that decides positivity is an open
  // question with three implemented answers, and one copy of it is the most this
  // pipeline can afford.
  const fields = useMemo(
    () =>
      nucleiOverlay(geometry, (nucleus, fieldIndex) => {
        const cell = cells.get(cellKey(fieldIndex, nucleus.id))
        if (!cell) return null
        return CELL_COLOUR.bins[cell.bin] ?? null
      }),
    [cells, geometry],
  )

  const targets = useMemo(() => flyTargets(fields), [fields])
  const [flownTo, setFlownTo] = useState<number | null>(null)
  const [flyBox, setFlyBox] = useState<{ x: number; y: number; span: number } | null>(null)
  const fly = (target: FlyTarget | null) => {
    setFlownTo(target?.rank ?? null)
    setFlyBox(target ? target.box : null)
  }

  const chosen = selected ? cells.get(selected) : undefined

  if (!hasMeasurements) {
    return (
      <div className="bn-panel bn-panel--waiting">
        <p>The cells have not been measured yet. Go back a step first.</p>
      </div>
    )
  }
  if (loading) {
    return (
      <div className="bn-panel bn-panel--waiting">
        <Spinner />
        <p>Applying this marker&rsquo;s cut points.</p>
      </div>
    )
  }
  if (error) {
    return (
      <div className="bn-panel bn-panel--error">
        <p>{error}</p>
      </div>
    )
  }
  if (!report) return null

  const worstComparison = [...report.comparisons].sort(
    (a, b) => Math.abs(b.deltaPoints) - Math.abs(a.deltaPoints),
  )[0]

  return (
    <div className="bn-panel">
      <header className="bn-panel__head">
        <div>
          <h3>Giving each cell a grade</h3>
          <p>
            Each cell&rsquo;s brown strength becomes a grade. This turns what the software
            measured into the words a pathologist uses.
          </p>
        </div>
        <Badge tone={report.params.cutsProvisional ? 'warn' : 'accent'}>
          {report.params.cutsProvisional ? 'provisional cut points' : report.markerName}
        </Badge>
      </header>

      <section className="bn-block">
        <h4>Per cell &mdash; the working, not the final answer</h4>
        <div className="bn-bins">
          {report.bins.map((bin) => (
            <div key={bin.bin} className={`bn-bins__card bn-bins__card--l${bin.bin}`}>
              <span className="bn-bins__level">{bin.label}</span>
              <strong>{bin.count.toLocaleString()}</strong>
              <span className="bn-bins__share">
                {(bin.share * 100).toFixed(1)}% of cells
              </span>
              <span className="bn-bins__copy">{BIN_COPY[bin.label]}</span>
            </div>
          ))}
        </div>
        <p className="bn-note">
          These four grades are used to count the positive cells. They are <em>not</em> the
          number that gets reported.
        </p>

      <FlyTo targets={targets} active={flownTo} onFly={fly} noun="binned cells" />

        {ihcUploadId && (
          <SlideOverlayViewer
            uploadId={ihcUploadId}
            mpp={ihcMpp}
            regions={regions}
            fields={fields}
            focus={flyBox}
            selectedId={selected}
            onSelect={select}
            minNucleusPx={7}
            legend={report.bins.map((bin) => ({
              colour: CELL_COLOUR.bins[bin.bin]?.fill ?? '#64748b',
              label: bin.label + ' — ' + (BIN_COPY[bin.label] ?? ''),
              count: bin.count,
            }))}
            emptyHint="The binned cells have not been loaded, so there is nothing to draw."
            caption={
              <>
                The same cells as the previous step, now coloured by grade. Grey is
                negative, and the amber gets brighter as the brown gets stronger. Zoom in
                and click a cell to see its numbers, its grade, and whether it counts
                towards the percentage.
              </>
            }
          />
        )}

        {chosen ? (
          <div className="cell-inspect bn-inspect">
            <div className="cell-inspect__head">
              <h4 className="cell-inspect__title">
                This cell is <strong>{chosen.label}</strong>
              </h4>
              <span className="cell-inspect__where mono">
                region {chosen.regionRank} · square {chosen.fieldIndex} · cell {chosen.cellId}
              </span>
            </div>

            <div className="cell-inspect__rows">
              <div className="cell-inspect__row">
                <span className="cell-inspect__label">
                  How strong its brown is
                  <span className="cell-inspect__hint">
                    the cuts for {report.markerName} are{' '}
                    {report.params.odCuts.map((cut) => cut.toFixed(2)).join(' · ')}
                  </span>
                </span>
                <span className="cell-inspect__value">{chosen.intensityOd.toFixed(3)}</span>
              </div>

              <div className="cell-inspect__row">
                <span className="cell-inspect__label">
                  Its second number
                  <span className="cell-inspect__hint">
                    {report.params.secondMeasure === 'ring_completeness'
                      ? 'how complete the ring is; the cut is ' +
                        Math.round(report.params.secondMin * 36) +
                        ' of 36'
                      : 'how much of the body is stained; the cut is ' +
                        (report.params.secondMin * 100).toFixed(0) +
                        '%'}
                  </span>
                </span>
                <span className="cell-inspect__value">
                  {report.params.secondMeasure === 'ring_completeness'
                    ? Math.round(chosen.second * 36) + ' of 36'
                    : (chosen.second * 100).toFixed(0) + '%'}
                </span>
              </div>

              <div className="cell-inspect__row">
                <span className="cell-inspect__label">
                  What it contributes to the percentage
                  <span className="cell-inspect__hint">
                    using the &ldquo;{report.params.partialRule}&rdquo; rule for partly
                    stained cells &mdash; the setting with the biggest effect here
                  </span>
                </span>
                <span className="cell-inspect__value">{chosen.weight}</span>
              </div>
            </div>

            <span
              className={
                chosen.positive
                  ? 'cell-inspect__verdict cell-inspect__verdict--positive'
                  : 'cell-inspect__verdict cell-inspect__verdict--negative'
              }
            >
              {chosen.positive
                ? 'Counts towards the percentage'
                : 'Does not count towards the percentage'}
            </span>

            <Button variant="ghost" onClick={() => select(null)}>
              Close
            </Button>
          </div>
        ) : null}

        <div className="bn-totals">
          <div>
            <span className="bn-totals__label">Cells measured</span>
            <span className="bn-totals__value mono">{report.cells.toLocaleString()}</span>
          </div>
          <div>
            <span className="bn-totals__label">Dark enough</span>
            <span className="bn-totals__value mono">
              {report.overOdCut.toLocaleString()}
            </span>
          </div>
          <div>
            <span className="bn-totals__label">Positive (passed both tests)</span>
            <span className="bn-totals__value mono">
              {report.positiveCells.toLocaleString()}
            </span>
          </div>
          <div>
            <span className="bn-totals__label">Percentage positive</span>
            <span className="bn-totals__value mono">
              {(report.positiveShare * 100).toFixed(1)}%
            </span>
          </div>
        </div>
        <p className="bn-note">
          The gap between those two figures is the cells that are brown enough but whose
          staining is too patchy to count. That is exactly what the second test is for.
        </p>
      </section>

      <section className="bn-block">
        <h4>Per slide &mdash; the grade that is actually reported</h4>
        <p className="bn-note">
          The reported strength for the whole slide is one number between 0 and 2, and it
          can only be one of these six values. It is decided at the next step, across the
          positive cells. 119 of 120 real pathologist readings land exactly on one of
          these.
        </p>
        <table className="bn-bands">
          <thead>
            <tr>
              <th>If the average brown is</th>
              <th>Reported as</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {report.bandTable.map((row) => (
              <tr key={row.band}>
                <td>
                  {row.odBelow === null
                    ? 'anything darker'
                    : `below ${row.odBelow.toFixed(2)}`}
                </td>
                <td>
                  <strong>{row.band}</strong>
                </td>
                <td>{row.label}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <CutHistogram report={report} />

      <section className="bn-block">
        <h4>Why these cut points and not others</h4>
        <p className="bn-note">
          The cuts are fixed amounts of brown, on a scale that means the same thing on
          every slide &mdash; not &ldquo;the brownest cells on <em>this</em> slide&rdquo;.
          Here is what the alternatives would have reported on exactly the same cells.
        </p>
        <ul className="bn-compare">
          {report.comparisons.map((entry) => (
            <li key={entry.scheme}>
              <div className="bn-compare__head">
                <strong>{entry.label}</strong>
                <span
                  className={
                    Math.abs(entry.deltaPoints) >= 5
                      ? 'bn-compare__delta bn-compare__delta--big'
                      : 'bn-compare__delta'
                  }
                >
                  {entry.deltaPoints >= 0 ? '+' : ''}
                  {entry.deltaPoints} points
                </span>
              </div>
              <p>{entry.note}</p>
            </li>
          ))}
        </ul>
        {worstComparison && Math.abs(worstComparison.deltaPoints) >= 5 ? (
          <p className="bn-flag">
            The largest of those differences is {Math.abs(worstComparison.deltaPoints)}{' '}
            percentage points &mdash; more than the spread between four pathologists
            reading the same slide. That is how much this choice matters.
          </p>
        ) : null}
      </section>

      <details
        className="bn-detail"
        open={showDetail}
        onToggle={(event) => setShowDetail(event.currentTarget.open)}
      >
        <summary>The technical detail</summary>
        <dl>
          <dt>How the cuts were chosen</dt>
          <dd>
            {report.params.scheme} cut points, on a scale that means the same thing on
            every slide.
          </dd>
          <dt>The cuts</dt>
          <dd>
            {report.params.odCuts.map((cut) => cut.toFixed(2)).join(' / ')}, separating 0
            from 1+, 1+ from 2+ and 2+ from 3+.
          </dd>
          <dt>Second test</dt>
          <dd>
            {report.params.secondMeasure.replace(/_/g, ' ')} at or above{' '}
            {report.params.secondMin}.
          </dd>
          <dt>Cells over the density cut</dt>
          <dd>
            {report.overOdCut.toLocaleString()} — of which{' '}
            {report.positiveCells.toLocaleString()} also clear the second condition, which
            is what &ldquo;positive&rdquo; means for this marker.
          </dd>
          <dt>Partial-staining rule</dt>
          <dd>{report.params.partialRule} (open question Q1).</dd>
          <dt>Source</dt>
          <dd>
            {report.params.cutsSource} (version {report.params.cutsVersion})
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
