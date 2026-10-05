/**
 * Step 14's screen: the first moment anything is actually measured.
 *
 * Three things have to land, in this order.
 *
 * **That this is the first measurement.** Everything up to here was deciding
 * what to measure. So the headline is the two numbers every cell now has, in
 * plain words, before any of the machinery behind them.
 *
 * **The fork.** Membrane markers get a ring completeness; cytoplasmic markers
 * get a stained fraction, and for them completeness is not a harder measurement
 * but a meaningless one. The scatter's y-axis says which, and the copy says why
 * the other one would be wrong for this marker.
 *
 * **That every dot is a real cell.** Clicking one shows the piece of slide it
 * came from. A scatter nobody can get behind is a picture of a claim; one you
 * can click through is evidence.
 */

import { useMemo, useState } from 'react'

import { perCellFieldUrl } from '@/api/perCell'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { SlideOverlayViewer } from '@/components/viewer/SlideOverlayViewer'
import { FlyTo } from '@/features/cells/FlyTo'
import type { FlyTarget } from '@/features/cells/overlay'
import { cellKey, flyTargets, nucleiOverlay, odColour, regionOutlines } from '@/features/cells/overlay'
import type { AlignmentReport } from '@/types/ihcAlignment'
import type { RegionNucleiPayload } from '@/types/nuclei'
import type { PerCellReport } from '@/types/perCell'

import { CellScatter } from './CellScatter'
import type { PerCellStateValue } from './usePerCell'

import './perCell.css'

interface PerCellPanelProps {
  perCell: PerCellStateValue
  heUploadId: string | null
  ihcUploadId: string | null
  /** The IHC slide's level-0 microns per pixel, for the scale bar. */
  ihcMpp: number | null
  /** Step 10's report, for the region borders drawn on the slide. */
  alignment: AlignmentReport | null
  /** Step 11's outlines - the shapes this step paints its measurement onto. */
  geometry: Record<number, RegionNucleiPayload>
  /** Step 13 must have built the compartments there is anything to measure in. */
  hasCompartments: boolean
  onRun: () => void
}

/** How a cell's two numbers read in words, for the card and for the legend. */
function secondLabel(report: PerCellReport, value: number): string {
  return report.secondMeasure === 'ring_completeness'
    ? Math.round(value * 36) + ' of 36 segments'
    : (value * 100).toFixed(0) + '% of the body'
}

const SECOND_COPY: Record<
  PerCellReport['secondMeasure'],
  { title: string; body: string; why: string }
> = {
  ring_completeness: {
    title: 'How complete the ring is',
    body:
      'The ring around each cell is split into 36 segments, like a clock face, and each ' +
      'segment is marked as stained or not. A cell with 34 of 36 stained has a real, ' +
      'unbroken rim of colour all the way round.',
    why:
      'A cell with only 6 of 36 stained just has a few brown specks. Averaged over the ' +
      'whole ring those specks look like faint, even staining — which is why this is ' +
      'measured separately and has to pass its own test.',
  },
  stained_fraction: {
    title: 'How much of the cell body is stained',
    body:
      'This marker sits throughout the cell body rather than on its rim, so we measure ' +
      'what share of the body is brown.',
    why:
      'Ring completeness is deliberately not calculated for this marker. A cell body ' +
      'has no rim, so the number would be meaningless — and a near-zero value would ' +
      'mark cells negative for a shape they never had.',
  },
}

function Histogram({ report }: { report: PerCellReport }) {
  const bars = report.histogram
  if (bars.length === 0) return null
  const tallest = Math.max(...bars.map((bar) => bar.count), 1)
  const cut = report.params.positivityOd
  // The `bars.length === 0` return above guarantees a last bar; indexing cannot
  // express that, so read it once and let the fallback stand for the impossible case.
  const span = bars.at(-1)?.upper || 1

  return (
    <figure className="pc-hist">
      <div className="pc-hist__bars">
        {bars.map((bar) => (
          <div
            key={bar.lower}
            className={
              bar.lower >= cut ? 'pc-hist__bar pc-hist__bar--over' : 'pc-hist__bar'
            }
            style={{ height: `${(bar.count / tallest) * 100}%` }}
            title={`${bar.count} cells between ${bar.lower.toFixed(2)} and ${bar.upper.toFixed(2)}`}
          />
        ))}
        <div className="pc-hist__cut" style={{ left: `${(cut / span) * 100}%` }}>
          <span>counted as brown from here</span>
        </div>
      </div>
      <figcaption>
        How strong the brown is across all {report.cells.toLocaleString()} cells. Two
        humps &mdash; one pale, one brown &mdash; with the line in the dip between them is
        what a well-placed cut looks like.
      </figcaption>
    </figure>
  )
}

export function PerCellPanel({
  perCell,
  heUploadId,
  ihcUploadId,
  ihcMpp,
  alignment,
  geometry,
  hasCompartments,
  onRun,
}: PerCellPanelProps) {
  const [showDetail, setShowDetail] = useState(false)
  const { report, rows, loading, error, selected, selectedKey, select, selectKey } = perCell

  const regions = useMemo(() => regionOutlines(alignment), [alignment])

  /**
   * Where the colour ramp saturates: the 95th percentile of what was measured.
   *
   * Read from the data rather than fixed, because optical density has no natural
   * maximum - a fixed ceiling would render a weakly stained slide uniformly
   * blank and a strong one uniformly bright, which is the per-slide-percentile
   * failure this pipeline argues against, committed in the colours instead of in
   * the cuts. The percentile rather than the maximum so one saturated speck
   * cannot flatten the whole ramp.
   *
   * It is a *display* ceiling and nothing is computed from it. The cuts that
   * decide anything are absolute and arrive from the server.
   */
  const ceiling = useMemo(() => {
    if (rows.size === 0) return report?.params.positivityOd ?? 1
    const ordered = [...rows.values()].map((row) => row.intensityOd).sort((a, b) => a - b)
    return ordered[Math.floor(ordered.length * 0.95)] ?? 1
  }, [report, rows])

  // A cell with no measurement is left undrawn rather than given a default
  // colour: step 14 measures tumour cells only, and painting the rest would put
  // cells on screen that are not in the number underneath.
  const fields = useMemo(
    () =>
      nucleiOverlay(geometry, (nucleus, fieldIndex) => {
        const row = rows.get(cellKey(fieldIndex, nucleus.id))
        return row ? odColour(row.intensityOd, ceiling) : null
      }),
    [ceiling, geometry, rows],
  )

  const targets = useMemo(() => flyTargets(fields), [fields])
  const [flownTo, setFlownTo] = useState<number | null>(null)
  const [flyBox, setFlyBox] = useState<{ x: number; y: number; span: number } | null>(null)
  const fly = (target: FlyTarget | null) => {
    setFlownTo(target?.rank ?? null)
    setFlyBox(target ? target.box : null)
  }

  if (!hasCompartments) {
    return (
      <div className="pc-panel pc-panel--waiting">
        <p>
          The measuring areas have not been marked yet. Go back a step first &mdash; there
          is nothing to measure until we know which part of each cell to look at.
        </p>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="pc-panel pc-panel--waiting">
        <Spinner />
        <p>
          Reading the brown off the slide, cell by cell. This re-opens the slide at the
          same places as before, so it takes up to a minute.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="pc-panel pc-panel--error">
        <p>{error}</p>
        <Button onClick={onRun}>Try again</Button>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="pc-panel pc-panel--waiting">
        <p>Ready to measure every tumour cell.</p>
        <Button onClick={onRun}>Measure every cell</Button>
      </div>
    )
  }

  const copy = SECOND_COPY[report.secondMeasure]

  return (
    <div className="pc-panel">
      <header className="pc-panel__head">
        <div>
          <h3>Every cell now has two numbers</h3>
          <p>
            This is the first step that measures anything. Everything before it was
            deciding <em>what</em> to measure. {report.cells.toLocaleString()} tumour cells
            were read, one at a time.
          </p>
        </div>
        <Badge tone="accent">{report.markerName}</Badge>
      </header>

      <div className="pc-numbers">
        <div className="pc-numbers__card">
          <span className="pc-numbers__eyebrow">Number 1 — the same for every marker</span>
          <strong>How strong the brown is</strong>
          <p>
            The average darkness of the brown inside that cell&rsquo;s own measuring area.
            We use the average rather than the darkest spot, so one stray dark pixel
            cannot decide a cell.
          </p>
          <span className="pc-numbers__value">
            {report.meanOd.toFixed(3)} <small>average across all cells</small>
          </span>
        </div>

        <div className="pc-numbers__card">
          <span className="pc-numbers__eyebrow">
            Number 2 — this one depends on the marker
          </span>
          <strong>{copy.title}</strong>
          <p>{copy.body}</p>
          <span className="pc-numbers__value">
            {report.secondMeasure === 'ring_completeness'
              ? `${Math.round(report.meanSecond * 36)} of 36`
              : `${(report.meanSecond * 100).toFixed(0)}%`}{' '}
            <small>average across all cells</small>
          </span>
        </div>
      </div>

      <p className="pc-why">{copy.why}</p>

      <FlyTo targets={targets} active={flownTo} onFly={fly} noun="measured cells" />

      {ihcUploadId && (
        <SlideOverlayViewer
          uploadId={ihcUploadId}
          mpp={ihcMpp}
          regions={regions}
          fields={fields}
          focus={flyBox}
          selectedId={selectedKey}
          onSelect={selectKey}
          minNucleusPx={7}
          legend={[
            { colour: 'rgb(100, 116, 139)', label: 'No brown' },
            { colour: 'rgb(177, 170, 105)', label: 'Some' },
            { colour: 'rgb(253, 224, 71)', label: 'Strongly stained' },
            {
              colour: 'rgba(167, 139, 250, 0.95)',
              label: 'Invasive tumour region',
            },
          ]}
          emptyHint="The measured cells have not been loaded, so there is nothing to draw."
          caption={
            <>
              Every measured tumour cell, coloured by how dark the brown is inside its own
              measuring area. Zoom in and click a cell to see both of its numbers and the
              piece of slide it came from. Clicking a dot on the chart below selects the
              same cell here.
            </>
          }
        />
      )}


      <CellScatter
        report={report}
        positivityOd={report.params.positivityOd}
        secondMin={report.params.secondMin}
        selected={selected}
        onSelect={select}
      />

      {selected ? (
        <div className="pc-crop">
          {heUploadId && ihcUploadId && (
            <img
              src={perCellFieldUrl(
                heUploadId,
                ihcUploadId,
                selected.regionRank,
                selected.fieldIndex,
              )}
              alt={'The piece of slide cell ' + selected.cellId + ' was measured in'}
            />
          )}
          <div className="cell-inspect">
            <div className="cell-inspect__head">
              <h4 className="cell-inspect__title">This one cell</h4>
              <span className="cell-inspect__where mono">
                region {selected.regionRank} · square {selected.fieldIndex} · cell{' '}
                {selected.cellId}
              </span>
            </div>

            <div className="cell-inspect__rows">
              <div className="cell-inspect__row">
                <span className="cell-inspect__label">
                  How strong the brown is
                  <span className="cell-inspect__hint">
                    averaged across this cell&rsquo;s {report.params.compartment}; the cut
                    for {report.markerName} is {report.params.positivityOd}
                  </span>
                </span>
                <span className="cell-inspect__value">
                  {selected.intensityOd.toFixed(3)}
                </span>
              </div>

              <div className="cell-inspect__row">
                <span className="cell-inspect__label">
                  {copy.title}
                  <span className="cell-inspect__hint">
                    the cut is {secondLabel(report, report.params.secondMin)}
                  </span>
                </span>
                <span className="cell-inspect__value">
                  {secondLabel(report, selected.second)}
                </span>
              </div>

              <div className="cell-inspect__meter" aria-hidden>
                <span style={{ width: (Math.min(1, selected.second) * 100).toFixed(0) + '%' }} />
              </div>

              <div className="cell-inspect__row">
                <span className="cell-inspect__label">
                  Darkest spot in the measuring area
                  <span className="cell-inspect__hint">
                    recorded so the average can be checked &mdash; never used in the score
                  </span>
                </span>
                <span className="cell-inspect__value">{selected.maxOd.toFixed(3)}</span>
              </div>

              <div className="cell-inspect__row">
                <span className="cell-inspect__label">
                  Size of the measuring area
                  <span className="cell-inspect__hint">{selected.pixels} pixels</span>
                </span>
                <span className="cell-inspect__value">
                  {selected.areaUm2.toFixed(1)} µm²
                </span>
              </div>
            </div>

            <span
              className={
                selected.intensityOd >= report.params.positivityOd &&
                selected.second >= report.params.secondMin
                  ? 'cell-inspect__verdict cell-inspect__verdict--positive'
                  : 'cell-inspect__verdict cell-inspect__verdict--negative'
              }
            >
              {selected.intensityOd >= report.params.positivityOd &&
              selected.second >= report.params.secondMin
                ? 'Both tests passed — this cell counts as positive'
                : selected.intensityOd >= report.params.positivityOd
                  ? 'Dark enough, but the staining is too patchy to count'
                  : 'Not dark enough to count as positive'}
            </span>

            <Button variant="ghost" onClick={() => select(null)}>
              Close
            </Button>
          </div>
        </div>
      ) : null}

      <Histogram report={report} />

      {report.crowdedCells ? (
        <p className="pc-flag">
          {report.crowdedCells.toLocaleString()} cells were surrounded by neighbours on
          most sides, so their ring could not be seen the whole way round however strongly
          they were stained. They are counted and flagged, not quietly dropped.
        </p>
      ) : null}

      <details
        className="pc-detail"
        open={showDetail}
        onToggle={(event) => setShowDetail(event.currentTarget.open)}
      >
        <summary>The technical detail</summary>
        <dl>
          <dt>What is measured</dt>
          <dd>
            The {report.params.intensityStatistic} brown reading across the measuring area.
            The same for every marker, so two markers stay comparable. The darkest spot
            per cell is recorded but never used in a score.
          </dd>

          <dt>Measuring area</dt>
          <dd>
            {report.params.compartment}, {report.params.expansionUm} µm
            {report.params.ringUm !== null ? `, thinned to a ${report.params.ringUm} µm ring` : ''}.
          </dd>

          <dt>Cut for counting as positive</dt>
          <dd>
            {report.params.positivityOd}, and{' '}
            {report.secondMeasure === 'ring_completeness'
              ? `${Math.round(report.params.secondMin * 36)} of 36 ring segments`
              : `${(report.params.secondMin * 100).toFixed(0)}% of the cell body`}{' '}
            &mdash; both from {report.markerName}&rsquo;s own cut-point table (version{' '}
            {report.params.cutsVersion}
            {report.params.cutsProvisional ? ', provisional' : ''}).
          </dd>

          {report.params.ringBins !== null ? (
            <>
              <dt>Ring segments</dt>
              <dd>{report.params.ringBins} segments of 10° each.</dd>
            </>
          ) : null}

          <dt>Regions</dt>
          <dd>
            {report.regions
              .map(
                (region) =>
                  `region ${region.rank}: ${region.cells.toLocaleString()} cells over ${region.areaMm2.toFixed(2)} mm², mean ${region.meanOd.toFixed(3)} OD`,
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
