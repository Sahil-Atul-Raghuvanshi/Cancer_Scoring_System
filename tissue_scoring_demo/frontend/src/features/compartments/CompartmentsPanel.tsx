/**
 * Step 13's screen: the part of each cell the stain is supposed to be in.
 *
 * Two things have to land, and the layout puts them in this order.
 *
 * **The fork.** This is the first step that reads the antibody letter, and a
 * viewer should see *which* shape this marker gets and that it was not a choice
 * anybody made on this screen. So the compartment is stated as a fact with the
 * marker's name attached, and the other option is shown beside it as the thing
 * that would have been wrong.
 *
 * **The width, and that nobody has established it.** The slider moves it and the
 * measured area follows; the sweep underneath shows the whole curve. A viewer
 * who drags from 2 µm to 8 µm and watches the area go up six-fold has learned
 * why this parameter has to be published with the score.
 */

import { useMemo, useState } from 'react'

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { SlideOverlayViewer } from '@/components/viewer/SlideOverlayViewer'
import { FlyTo } from '@/features/cells/FlyTo'
import type { FlyTarget } from '@/features/cells/overlay'
import {
  CELL_COLOUR,
  compartmentOverlay,
  compartmentsByKey,
  flyTargets,
  regionOutlines,
  ringCentre,
} from '@/features/cells/overlay'
import type { CompartmentLayers } from '@/features/cells/overlay'
import type { CompartmentsReport } from '@/types/compartments'
import type { AlignmentReport } from '@/types/ihcAlignment'

import type { CompartmentsStateValue } from './useCompartments'

import './compartments.css'

interface CompartmentsPanelProps {
  compartments: CompartmentsStateValue
  ihcUploadId: string | null
  /** The IHC slide's level-0 microns per pixel, for the scale bar. */
  ihcMpp: number | null
  /** Step 10's report, for the region borders drawn on the slide. */
  alignment: AlignmentReport | null
  /** Step 12 must have sorted the cells: compartments are for tumour cells only. */
  hasTypes: boolean
}

const KIND_COPY: Record<
  CompartmentsReport['compartment'],
  { title: string; body: string; other: string }
> = {
  membrane: {
    title: 'a ring around the outside of the cell',
    body:
      'This marker sits in the cell membrane — the outer skin of the cell — so the brown ' +
      'is measured in a thin ring just outside the nucleus.',
    other:
      'Measuring the wider cell body instead would include parts of the cell where this ' +
      'marker never appears, and the result would come out too weak.',
  },
  cytoplasm: {
    title: 'a band of the cell body',
    body:
      'This marker sits throughout the body of the cell rather than on its outer skin, ' +
      'so a wider band is measured instead of a thin ring.',
    other:
      'Measuring a thin ring instead would sample the very edge of the cell, where it ' +
      'touches its neighbour. The number would then follow how tightly packed the tissue ' +
      'is rather than how much marker the cell actually made.',
  },
}

function Sweep({ report }: { report: CompartmentsReport }) {
  const points = report.widthSensitivity
  if (points.length === 0) return null

  const max = Math.max(...points.map((p) => p.meanMeasuredUm2))
  const chosen = report.params.expansionUm

  return (
    <div className="cp-sweep">
      {points.map((point) => {
        const active = Math.abs(point.widthUm - chosen) < 0.01
        return (
          <div key={point.widthUm} className={active ? 'cp-sweep__row is-active' : 'cp-sweep__row'}>
            <span className="cp-sweep__width mono">{point.widthUm.toFixed(0)} µm</span>
            <div className="cp-sweep__track">
              <div
                className="cp-sweep__fill"
                style={{ width: `${max > 0 ? (point.meanMeasuredUm2 / max) * 100 : 0}%` }}
              />
            </div>
            <span className="cp-sweep__value mono">
              {point.meanMeasuredUm2.toFixed(0)} µm²
            </span>
            <span className="cp-sweep__contested mono">
              {(point.contestedShare * 100).toFixed(0)}% overlapping neighbours
            </span>
          </div>
        )
      })}
    </div>
  )
}

export function CompartmentsPanel({
  compartments,
  ihcUploadId,
  ihcMpp,
  alignment,
  hasTypes,
}: CompartmentsPanelProps) {
  const { report, geometry, widthUm, loading, error, setWidth, resetWidth } = compartments
  // All three parts of the cell on by default. The one that is measured is
  // drawn at full strength and the other faint, so "which of these is the
  // number" is answered by the picture rather than by the key.
  const [layers, setLayers] = useState<CompartmentLayers>({
    nucleus: true,
    cytoplasm: true,
    membrane: true,
    alternate: false,
  })
  const [selected, setSelected] = useState<string | null>(null)

  const fields = useMemo(() => compartmentOverlay(geometry, layers), [geometry, layers])
  const regions = useMemo(() => regionOutlines(alignment), [alignment])
  const byKey = useMemo(() => compartmentsByKey(geometry), [geometry])
  const targets = useMemo(() => flyTargets(fields), [fields])
  const [flownTo, setFlownTo] = useState<number | null>(null)
  const [flyBox, setFlyBox] = useState<{ x: number; y: number; span: number } | null>(null)
  const fly = (target: FlyTarget | null) => {
    setFlownTo(target?.rank ?? null)
    setFlyBox(target ? target.box : null)
  }

  const chosen = selected ? byKey.get(selected) : undefined
  // Flying to a cell rather than opening a second viewer beside the first: the
  // ask is for one cell blown up huge, and the viewer that can already do that
  // is the one on screen. Two viewers would be two sets of tiles for one slide.
  const focus = useMemo(() => {
    // A clicked cell wins over a region: the click is the more specific request,
    // and it is the one that just happened.
    if (!chosen) return flyBox
    const centre = ringCentre(chosen.cell.nucleus)
    if (!centre) return null
    const span = ihcMpp && ihcMpp > 0 ? 60 / ihcMpp : 280
    return { x: centre.x - span / 2, y: centre.y - span / 2, span }
  }, [chosen, flyBox, ihcMpp])

  if (!hasTypes) {
    return (
      <div className="cp-empty">
        <h3>Where should the marker be?</h3>
        <p>
          The cells have not been sorted yet. This step only marks measuring areas on
          tumour cells, so go back and sort the cells first.
        </p>
      </div>
    )

  }

  if (!report) {
    return (
      <div className="cp-empty">
        {loading ? <Spinner /> : null}
        <p>{error ?? 'Building the compartments…'}</p>
      </div>
    )
  }

  const copy = KIND_COPY[report.compartment]
  const width = widthUm ?? report.params.expansionUm
  const alternateKind = report.compartment === 'membrane' ? 'cytoplasm' : 'membrane'
  const alternateColour = CELL_COLOUR.compartments[alternateKind]
  const widths = Object.values(geometry)[0]?.widthsUm ?? null

  /** Where a region sits, in words, for the legend and the card. */
  const extent = (kind: 'nucleus' | 'cytoplasm' | 'membrane'): string | undefined => {
    if (!widths) return undefined
    if (kind === 'nucleus') return 'the segmentation itself'
    const inner = Math.max(0, widths.body - widths.shell)
    if (kind === 'membrane') {
      return widths.shell > 0
        ? `${inner.toFixed(1)}–${widths.body.toFixed(1)} µm out from the nucleus`
        : 'this antibody has no membrane compartment'
    }
    return `0–${inner.toFixed(1)} µm out from the nucleus`
  }

  return (
    <div className="cp-panel">
      <header className="cp-panel__head">
        <div>
          <h3>
            {report.markerName} is measured in {copy.title}
          </h3>
          <p>{copy.body}</p>
        </div>
        <Badge tone="neutral">{report.compartment}</Badge>
      </header>

      <div className="cp-fork">
        <p>
          <strong>This is the first step that depends on which marker was used.</strong>{' '}
          {copy.other}
        </p>
        <p className="cp-fork__note">
          This is decided by the marker on the slide and cannot be changed here. Making it
          a setting would make it possible to get wrong.
        </p>
      </div>

      <div className="cp-layers" role="group" aria-label="What to draw">
        <span className="cp-layers__label">Draw</span>
        {(
          [
            ['nucleus', 'Nucleus', CELL_COLOUR.compartments.nucleus.fill],
            ['cytoplasm', 'Cytoplasm', CELL_COLOUR.compartments.cytoplasm.fill],
            ['membrane', 'Membrane', CELL_COLOUR.compartments.membrane.fill],
            [
              'alternate',
              alternateColour.label + ' \u2014 the other fork',
              alternateColour.fill,
            ],
          ] as [keyof CompartmentLayers, string, string][]
        ).map(([key, label, colour]) => (
          <button
            key={key}
            type="button"
            aria-pressed={layers[key]}
            className={layers[key] ? 'cp-layer is-on' : 'cp-layer'}
            onClick={() => setLayers((current) => ({ ...current, [key]: !current[key] }))}
          >
            <span className="cp-layer__swatch" style={{ background: colour }} aria-hidden />
            {label}
          </button>
        ))}
      </div>

      <FlyTo targets={targets} active={flownTo} onFly={fly} noun="compartments" />

      {ihcUploadId && (
        <SlideOverlayViewer
          uploadId={ihcUploadId}
          mpp={ihcMpp}
          regions={regions}
          fields={fields}
          focus={focus}
          selectedId={selected}
          onSelect={setSelected}
          // A compartment is a band a few microns wide, so it needs more zoom
          // than a nucleus outline does before it says anything at all. Eleven
          // screen pixels per 10 um puts a 4 um ring at four pixels, which is
          // thin but honest; below that it is a smear.
          minNucleusPx={11}
          legend={[
            {
              colour: CELL_COLOUR.compartments.nucleus.fill,
              label: 'Nucleus \u2014 never measured',
              hint: extent('nucleus'),
            },
            {
              colour: CELL_COLOUR.compartments.cytoplasm.fill,
              label:
                'Cytoplasm \u2014 ' +
                (report.compartment === 'cytoplasm'
                  ? 'measured for ' + report.markerName
                  : 'not measured for ' + report.markerName),
              hint: extent('cytoplasm'),
            },
            {
              colour: CELL_COLOUR.compartments.membrane.fill,
              label:
                'Membrane \u2014 ' +
                (report.compartment === 'membrane'
                  ? 'measured for ' + report.markerName
                  : 'not measured for ' + report.markerName),
              hint: extent('membrane'),
            },
            {
              colour: alternateColour.fill,
              label: 'What a ' + alternateColour.label.toLowerCase() + ' marker would use',
              hint: widths ? widths.alternate.toFixed(1) + ' \u00b5m, dashed' : undefined,
            },
          ]}
          emptyHint="The compartment outlines have not been built for this pair yet."
          caption={
            <>
              Zoom in until one cell fills a good part of the frame, then click it. Each
              cell is drawn in three parts: the nucleus, the cell body and the outer
              membrane ring. The part this marker is measured in is drawn in the bright
              colour; the others are drawn dark. Turn on the dashed layer to see what the other kind of marker
              would have measured on the same cell.
            </>
          }
        />
      )}

      <div className="cp-body">
        <div className="cp-inspect">
          {chosen ? (
            <div className="cell-inspect">
              <div className="cell-inspect__head">
                <h4 className="cell-inspect__title">One cell, up close</h4>
                <span className="cell-inspect__where mono">
                  region {chosen.rank} · square {chosen.fieldIndex} · cell {chosen.cell.id}
                </span>
              </div>
              <div className="cell-inspect__rows">
                <div className="cell-inspect__row">
                  <span className="cell-inspect__label">
                    Nucleus
                    <span className="cell-inspect__hint">
                      found earlier; nothing is measured here
                    </span>
                  </span>
                  <span className="cell-inspect__value">
                    {chosen.cell.nucleus.length > 0 ? 'drawn' : '—'}
                  </span>
                </div>
                <div className="cell-inspect__row">
                  <span className="cell-inspect__label">
                    Cytoplasm
                    <span className="cell-inspect__hint">{extent('cytoplasm')}</span>
                  </span>
                  <span className="cell-inspect__value">
                    {chosen.cell.cytoplasm.length === 0
                      ? '—'
                      : report.compartment === 'cytoplasm'
                        ? 'measured'
                        : 'not measured'}
                  </span>
                </div>
                <div className="cell-inspect__row">
                  <span className="cell-inspect__label">
                    Membrane
                    <span className="cell-inspect__hint">{extent('membrane')}</span>
                  </span>
                  <span className="cell-inspect__value">
                    {chosen.cell.membrane.length === 0
                      ? '—'
                      : report.compartment === 'membrane'
                        ? 'measured'
                        : 'not measured'}
                  </span>
                </div>
                <div className="cell-inspect__row">
                  <span className="cell-inspect__label">
                    The other option
                    <span className="cell-inspect__hint">
                      what a {alternateColour.label.toLowerCase()} marker would have used
                      here &mdash; drawn, never measured
                    </span>
                  </span>
                  <span className="cell-inspect__value">
                    {chosen.cell.alternate.length > 0 ? 'drawn' : '—'}
                  </span>
                </div>
              </div>
              <Button variant="ghost" onClick={() => setSelected(null)}>
                Back to the whole slide
              </Button>
            </div>
          ) : (
            <div className="cell-inspect cell-inspect--empty">
              Click any cell on the slide to zoom to it and see which parts of it this
              marker does and does not measure.
            </div>
          )}
        </div>

        <div className="cp-controls">
          <div className="cp-stats">
            <div>
              <span className="cp-stats__label">Cells</span>
              <span className="cp-stats__value mono">{report.cells.toLocaleString()}</span>
            </div>
            <div>
              <span className="cp-stats__label">Measured region per cell</span>
              <span className="cp-stats__value mono">
                {report.meanMeasuredUm2.toFixed(0)} µm²
              </span>
            </div>
            <div>
              <span className="cp-stats__label">Whole cell</span>
              <span className="cp-stats__value mono">{report.meanCellUm2.toFixed(0)} µm²</span>
            </div>
          </div>

          <label className="cp-width">
            <span className="cp-width__label">
              How wide is the measured region?
              <span className="mono"> {width.toFixed(1)} µm</span>
            </span>
            <input
              type="range"
              min={2}
              max={10}
              step={0.5}
              value={width}
              onChange={(event) => setWidth(Number(event.target.value))}
            />
            <span className="cp-width__hint">
              The default for {report.markerName} is{' '}
              <span className="mono">{report.params.expansionUm.toFixed(1)} µm</span>. Move
              the slider and the band on the slide above is redrawn at the new width.
            </span>
          </label>
          <Button variant="ghost" onClick={resetWidth}>
            Back to the default
          </Button>
          {loading && <span className="cp-busy mono">recalculating…</span>}
        </div>
      </div>

      <section className="cp-sensitivity">
        <h4>This width is not a settled number, so here is what it changes</h4>
        <p>
          A wider band means a bigger measuring area, and a bigger chance of running into
          the cell next door. Both are shown, so you can see how much this setting affects
          the result.
        </p>
        <Sweep report={report} />
      </section>

      <div className="cp-regions">
        <h4>By region</h4>
        <table>
          <thead>
            <tr>
              <th>Region</th>
              <th>Tumour cells</th>
              <th>Measuring area</th>
              <th>Whole cell</th>
              <th>Nucleus</th>
              <th>Overlapping neighbours</th>
            </tr>
          </thead>
          <tbody>
            {report.regions.map((region) => (
              <tr key={region.rank}>
                <td>{region.rank}</td>
                <td className="mono">{region.cells.toLocaleString()}</td>
                <td className="mono">{region.meanMeasuredUm2.toFixed(0)} µm²</td>
                <td className="mono">{region.meanCellUm2.toFixed(0)} µm²</td>
                <td className="mono">{region.meanNucleusUm2.toFixed(0)} µm²</td>
                <td className="mono">{(region.contestedShare * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {report.notes.map((note) => (
        <p key={note} className="cp-note">
          {note}
        </p>
      ))}

      <details className="cp-detail">
        <summary>The technical detail</summary>
        <ul>
          <li>
            Which part of the cell is measured comes from the marker: a membrane ring at{' '}
            <span className="mono">{report.params.expansionUm} µm</span> for CD44, ABCC4 and
            ABCC11, and a 6 µm cell-body band for the two cadherins. The second measurement
            for this marker will be <span className="mono">{report.secondMeasure}</span>.
          </li>
          <li>
            Each cell only grows out as far as the halfway point between it and its
            neighbours, so two cells can never claim the same pixel. &ldquo;Overlapping
            neighbours&rdquo; above is how many pixels would have been claimed twice
            without that rule &mdash; measured, not estimated.
          </li>
          <li>
            Non-tumour cells are removed <em>before</em> the areas are grown, not after.
            Removing them afterwards would leave a dent in the neighbouring tumour
            cell&rsquo;s area.
          </li>
          <li>
            Every distance is in microns and converted using the slide&rsquo;s own scale, so
            it means the same thing on every scanner.
          </li>
          {report.typedCells != null && (
            <li className="mono">
              {report.cells.toLocaleString()} tumour cells given compartments.
            </li>
          )}
        </ul>
      </details>
    </div>
  )
}
