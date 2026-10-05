/**
 * Step 12's screen: which cells count, and how much that depends on the rules.
 *
 * The screen is built around an admission. These classes come from thresholds
 * somebody reasoned about, not from a model fitted on labelled cells, so the
 * thresholds are **on the page as sliders** and the mix moves as they move. A
 * viewer who drags one and watches the tumour share swing has learned the most
 * important thing about this step in about two seconds, and no paragraph would
 * have taught it as well.
 *
 * The sensitivity table underneath says the same thing in numbers, for the
 * thresholds nobody drags.
 */

import { useMemo, useState } from 'react'

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { SlideOverlayViewer } from '@/components/viewer/SlideOverlayViewer'
import { FlyTo } from '@/features/cells/FlyTo'
import type { FlyTarget } from '@/features/cells/overlay'
import { CELL_COLOUR, cellKey, flyTargets, nucleiOverlay, regionOutlines } from '@/features/cells/overlay'
import type { AlignmentReport } from '@/types/ihcAlignment'
import type { RegionNucleiPayload } from '@/types/nuclei'
import type { SensitivitySweep, TypeCount, TypingRules } from '@/types/cellTyping'

import type { CellTypingStateValue } from './useCellTyping'

import './cellTyping.css'

interface CellTypingPanelProps {
  typing: CellTypingStateValue
  /** Step 11 must have found nuclei before there is anything to sort. */
  hasNuclei: boolean
  ihcUploadId: string | null
  /** The IHC slide's level-0 microns per pixel, for the scale bar. */
  ihcMpp: number | null
  /** Step 10's report, for the region borders drawn on the slide. */
  alignment: AlignmentReport | null
  /** Step 11's stored outlines - the shapes this step colours in. */
  geometry: Record<number, RegionNucleiPayload>
}

const SLIDERS: {
  key: keyof TypingRules
  label: string
  hint: string
  min: number
  max: number
  step: number
  unit: string
}[] = [
  {
    key: 'lymphocyteMaxAreaUm2',
    label: 'Immune cells are smaller than',
    hint: 'immune cell nuclei are about 6–7 µm across; tumour ones start near 8',
    min: 20,
    max: 60,
    step: 1,
    unit: 'µm²',
  },
  {
    key: 'lymphocyteMinCircularity',
    label: '…and rounder than',
    hint: '1.00 is a perfect circle; immune cell nuclei are almost round',
    min: 0.5,
    max: 0.95,
    step: 0.01,
    unit: '',
  },
  {
    key: 'lymphocyteMinDarkness',
    label: '…and darker than average by',
    hint: 'compared with the rest of this field, because stains differ between slides',
    min: 0.9,
    max: 1.3,
    step: 0.01,
    unit: '×',
  },
  {
    key: 'spindleMinEccentricity',
    label: 'Support cells are longer than',
    hint: '0 is a circle, 1 is a line; support cell nuclei are cigar-shaped',
    min: 0.7,
    max: 0.98,
    step: 0.01,
    unit: '',
  },
]

const PARAMETER_LABEL: Record<string, string> = {
  lymphocyte_max_area_um2: 'Immune cell size limit',
  lymphocyte_min_circularity: 'Immune cell roundness',
  lymphocyte_min_darkness: 'Immune cell darkness',
  spindle_min_eccentricity: 'Support cell elongation',
}

function rgb(colour: [number, number, number]): string {
  return `rgb(${colour[0]}, ${colour[1]}, ${colour[2]})`
}

function MixBar({ counts }: { counts: TypeCount[] }) {
  const total = counts.reduce((sum, entry) => sum + entry.count, 0)
  if (total === 0) return null

  return (
    <div className="ct-mix">
      <div className="ct-mix__bar">
        {counts.map((entry) => (
          <div
            key={entry.type}
            className="ct-mix__segment"
            style={{ width: `${entry.share * 100}%`, background: rgb(entry.colour) }}
            title={`${entry.label}: ${entry.count.toLocaleString()} (${(entry.share * 100).toFixed(1)}%)`}
          />
        ))}
      </div>
      <ul className="ct-mix__legend">
        {counts.map((entry) => (
          <li key={entry.type}>
            <span className="ct-mix__swatch" style={{ background: rgb(entry.colour) }} />
            <span className="ct-mix__label">{entry.label}</span>
            <span className="ct-mix__value mono">
              {entry.count.toLocaleString()} · {(entry.share * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Sensitivity({ sweeps }: { sweeps: SensitivitySweep[] }) {
  const ordered = [...sweeps].sort((a, b) => b.swing - a.swing)

  return (
    <table className="ct-sensitivity">
      <thead>
        <tr>
          <th>Threshold</th>
          <th>Set to</th>
          <th>How much it changes the tumour share</th>
        </tr>
      </thead>
      <tbody>
        {ordered.map((sweep) => (
          <tr key={sweep.parameter} className={sweep.swing >= 0.1 ? 'is-heavy' : undefined}>
            <td>{PARAMETER_LABEL[sweep.parameter] ?? sweep.parameter}</td>
            <td className="mono">{sweep.baseline}</td>
            <td>
              <div className="ct-swing">
                <div
                  className="ct-swing__fill"
                  style={{ width: `${Math.min(100, sweep.swing * 100)}%` }}
                />
                <span className="mono">{(sweep.swing * 100).toFixed(1)} points</span>
              </div>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function CellTypingPanel({
  typing,
  hasNuclei,
  ihcUploadId,
  ihcMpp,
  alignment,
  geometry,
}: CellTypingPanelProps) {
  const { report, types, rules, loading, error, setRule, resetRules } = typing

  // Colour by class, and draw nothing for a cell this step has no answer for.
  // An unclassified cell painted in a default colour would be indistinguishable
  // from a classified one, and the whole point of the screen is which is which.
  const fields = useMemo(
    () =>
      nucleiOverlay(geometry, (nucleus, fieldIndex, rank) => {
        const cls = types[rank]?.[cellKey(fieldIndex, nucleus.id)]
        if (cls === undefined) return null
        return CELL_COLOUR.classes[cls] ?? null
      }),
    [geometry, types],
  )

  const targets = useMemo(() => flyTargets(fields), [fields])
  const [flownTo, setFlownTo] = useState<number | null>(null)
  const [flyBox, setFlyBox] = useState<{ x: number; y: number; span: number } | null>(null)
  const fly = (target: FlyTarget | null) => {
    setFlownTo(target?.rank ?? null)
    setFlyBox(target ? target.box : null)
  }

  const regions = useMemo(() => regionOutlines(alignment), [alignment])

  if (!hasNuclei) {
    return (
      <div className="ct-empty">
        <h3>Which cells should count?</h3>
        <p>
          No cells have been found yet. Go back and find the cells first &mdash; there is
          nothing to sort until then.
        </p>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="ct-empty">
        {loading ? <Spinner /> : null}
        <p>{error ?? 'Sorting the cells…'}</p>
      </div>
    )
  }

  const current = { ...report.rules, ...rules }
  const worst = report.sensitivity.reduce((max, s) => Math.max(max, s.swing), 0)

  return (
    <div className="ct-panel">
      <header className="ct-panel__head">
        <div>
          <h3>{(report.tumourShare * 100).toFixed(1)}% of the cells are tumour cells</h3>
          <p>
            Not every cell inside a tumour region is a tumour cell. Immune cells and
            supporting cells are mixed in. Leaving them in the count waters down the final
            percentage, and how many there are differs from slide to slide.
          </p>
        </div>
        <Badge tone={report.trustworthy ? 'neutral' : 'warn'}>
          {report.counted.toLocaleString()} cells
        </Badge>
      </header>

      {!report.trustworthy && (
        <div className="ct-warn">
          <Badge tone="warn">Read this before the numbers</Badge>
          <p>{report.trustReason}</p>
          <p>
            The mix below is still shown so you can see the evidence, but it is a mix of
            fragments rather than whole cells. Do not rely on it until cell detection on
            this slide improves.
          </p>
        </div>
      )}

      <MixBar counts={report.counts} />

      <FlyTo targets={targets} active={flownTo} onFly={fly} noun="cells" />

      {ihcUploadId && (
        <SlideOverlayViewer
          uploadId={ihcUploadId}
          mpp={ihcMpp}
          regions={regions}
          fields={fields}
          focus={flyBox}
          legend={report.counts.map((entry) => ({
            colour: rgb(entry.colour),
            label: entry.label,
            count: entry.count,
          }))}
          emptyHint="The cell outlines have not been loaded, so there is nothing to colour."
          caption={
            <>
              The same cells as the previous step, now coloured by what each one is. Red
              cells go into the measurement. Yellow and green are inside the tumour region
              but are not tumour cells, so counting them would water down the percentage.
              Move a slider below and these colours change with it.
            </>
          }
        />
      )}

      <section className="ct-rules">
        <div className="ct-rules__head">
          <h4>The rules &mdash; move one and watch the mix change</h4>
          <Button variant="ghost" onClick={resetRules}>
            Reset
          </Button>
        </div>
        <p className="ct-rules__note">
          These are simple rules about what a cell looks like, not a trained model &mdash;
          nobody has labelled cells on these slides. That is why the rules are shown here
          rather than hidden in the code.
        </p>

        <div className="ct-rules__grid">
          {SLIDERS.map((slider) => {
            const value = current[slider.key]
            return (
              <label key={slider.key} className="ct-rule">
                <span className="ct-rule__label">{slider.label}</span>
                <input
                  type="range"
                  min={slider.min}
                  max={slider.max}
                  step={slider.step}
                  value={value}
                  onChange={(event) => setRule(slider.key, Number(event.target.value))}
                />
                <span className="ct-rule__value mono">
                  {slider.step < 1 ? value.toFixed(2) : value.toFixed(0)}
                  {slider.unit}
                </span>
                <span className="ct-rule__hint">{slider.hint}</span>
              </label>
            )
          })}
        </div>
        {loading && <span className="ct-rules__busy mono">recalculating…</span>}
      </section>

      <section className="ct-sensitivity-block">
        <h4>How much do the rules decide the answer?</h4>
        <p>
          Each row moves one slider across a sensible range and reports how far the tumour
          share moves with it. A small number means the tissue is deciding the answer; a
          large one means the slider is.
          {worst >= 0.1 && (
            <>
              {' '}
              <strong>
                Here the worst is {(worst * 100).toFixed(0)} points, which is large &mdash;
                that setting is deciding more than the cells are.
              </strong>
            </>
          )}
        </p>
        <Sensitivity sweeps={report.sensitivity} />
      </section>

      <div className="ct-regions">
        <h4>By region</h4>
        <table>
          <thead>
            <tr>
              <th>Region</th>
              <th>Cells</th>
              <th>Tumour cells / mm²</th>
              <th>Typical tumour cell size</th>
              <th>Typical immune cell size</th>
            </tr>
          </thead>
          <tbody>
            {report.regions.map((region) => (
              <tr key={region.rank}>
                <td>{region.rank}</td>
                <td className="mono">{region.counted.toLocaleString()}</td>
                <td className="mono">{region.tumourPerMm2.toLocaleString()}</td>
                <td className="mono">{region.medianTumourAreaUm2.toFixed(0)} µm²</td>
                <td className="mono">{region.medianLymphocyteAreaUm2.toFixed(0)} µm²</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {report.notes.map((note) => (
        <p key={note} className="ct-note">
          {note}
        </p>
      ))}

      <details className="ct-detail">
        <summary>The technical detail</summary>
        <ul>
          <li>
            Cells are tested in order: immune cell first (small <em>and</em> round{' '}
            <em>and</em> dark), then support cell, then tumour as the default. Tumour is
            the default on purpose: inside a region already judged to be invasive tumour, a
            cell counts as tumour unless it clearly looks like something else.
          </li>
          <li>
            Darkness is measured against the rest of the same field, not on a fixed scale,
            so the same rule works whatever the slide&rsquo;s staining strength.
          </li>
          <li>
            Only counted cells are sorted. Cells on a square&rsquo;s edge were seen cut
            off, so their size and shape are wrong &mdash; and those are exactly what this
            step decides on.
          </li>
          <li className="mono">
            Typical cell size {report.medianAreaUm2.toFixed(1)} µm². Below 25 µm² these
            rules are not reliable; a good H&amp;E slide measures about 43.
          </li>
        </ul>
      </details>
    </div>
  )
}
