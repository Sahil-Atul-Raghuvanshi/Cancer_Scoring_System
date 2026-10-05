/**
 * Step 11's screen: the stained slide, pannable, with every nucleus drawn on it.
 *
 * The page is written for somebody who is not a pathologist, per the house rule
 * for step screens: plain language on the surface, and the numbers a specialist
 * wants folded into a `<details>` at the end. The three things a non-expert has
 * to leave with are that nuclei were found one by one rather than as a blob,
 * that only a sample of each region was looked at, and that the brown stain was
 * deliberately removed before anything looked for a cell.
 *
 * **Why the whole slide rather than a cropped square.** This used to show one
 * 512 px field at a time. That picture is honest about the cells and silent
 * about everything else - where on the slide they are, how much of the tumour
 * was sampled, how small a cell is against the tissue around it. Panning the
 * real slide answers all three at once: the violet borders are the tumour step
 * 10 carried across, the dashed blue squares are what was actually looked at,
 * and the outlines only appear when you are close enough for one to mean
 * something.
 *
 * The two comparison panels are arguments, not decoration, and they stay on the
 * page rather than in the disclosure because they are the part a viewer
 * remembers: a naive watershed against the model on the same field, and the same
 * model run on the raw stain against the counterstain.
 */

import { useMemo, useState } from 'react'

import { nucleiComparisonUrl } from '@/api/nuclei'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { SlideOverlayViewer } from '@/components/viewer/SlideOverlayViewer'
import { FlyTo } from '@/features/cells/FlyTo'
import type { FlyTarget } from '@/features/cells/overlay'
import { CELL_COLOUR, flyTargets, nucleiOverlay, regionOutlines } from '@/features/cells/overlay'
import type { AlignmentReport } from '@/types/ihcAlignment'
import type { NucleiReport } from '@/types/nuclei'

import type { NucleiStateValue } from './useNuclei'

import './nuclei.css'

interface NucleiPanelProps {
  nuclei: NucleiStateValue
  heUploadId: string | null
  ihcUploadId: string | null
  marker: string | null
  /** The IHC slide's level-0 microns per pixel, for the scale bar. */
  ihcMpp: number | null
  /** Step 10's report, for the region borders drawn on the slide. */
  alignment: AlignmentReport | null
  /** Step 10 must have carried regions across and a person must have confirmed them. */
  hasConfirmedAlignment: boolean
  running: boolean
  onRun: () => void
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="nuc-stat">
      <div className="nuc-stat__label">{label}</div>
      <div className="nuc-stat__value mono">{value}</div>
      {hint && <div className="nuc-stat__hint">{hint}</div>}
    </div>
  )
}

function DensityCheck({ report }: { report: NucleiReport }) {
  const shortfall = report.densityShortfall
  const he = report.heDensityPerMm2

  if (he == null || shortfall == null) return null
  const bad = shortfall >= 0.3

  return (
    <div className={bad ? 'nuc-check nuc-check--warn' : 'nuc-check'}>
      <div className="nuc-check__head">
        <Badge tone={bad ? 'warn' : 'success'}>
          {bad ? 'Fewer cells than expected' : 'Cell density looks right'}
        </Badge>
      </div>
      <p>
        This stained section gave{' '}
        <strong className="mono">{report.densityPerMm2.toLocaleString()}</strong> nuclei per
        mm², against <strong className="mono">{he.toLocaleString()}</strong> on the same
        patient&rsquo;s H&amp;E slide inside the same regions.
      </p>
      {bad ? (
        <p>
          The two slides are neighbouring slices of the same block, so they hold roughly the
          same cells. A gap this large ({(shortfall * 100).toFixed(0)}%) means cells are
          being <em>missed</em> here, not that they are absent &mdash; heavy brown staining
          hides the blue. Every missed cell makes the final percentage come out too high.
        </p>
      ) : (
        <p>
          The two agree, which is what neighbouring slices should do. This check is the
          cheapest way to spot cell detection that has quietly stopped working.
        </p>
      )}
    </div>
  )
}

function Comparisons({
  report,
  heUploadId,
  ihcUploadId,
  rank,
}: {
  report: NucleiReport
  heUploadId: string
  ihcUploadId: string
  rank: number
}) {
  const entry = report.comparison.find((item) => item.regionRank === rank)
  if (!entry) return null

  return (
    <div className="nuc-compare">
      <section>
        <h4>Why each cell has to be outlined separately</h4>
        <p>
          On the left, the model. On the right, the old method &mdash; find the dark pixels,
          then split the blobs where they pinch. Where two cells touch, the old method merges
          them into one or cuts one in half. Every such mistake changes the count.
        </p>
        <img
          src={nucleiComparisonUrl(heUploadId, ihcUploadId, rank, 'watershed', report.generatedAt)}
          alt="The model's outlines beside a classical watershed on the same field"
        />
        <p className="nuc-compare__figures mono">
          {entry.instansegHaematoxylin} found by the model · {entry.watershedHaematoxylin} by the
          classical method
        </p>
      </section>

      <section>
        <h4>Why the brown is removed first</h4>
        <p>
          The brown is what we are measuring. If it is left in, strongly stained cells become
          easier to find &mdash; so the measurement would be choosing which cells to count,
          and the percentage would creep up on its own.
        </p>
        <img
          src={nucleiComparisonUrl(heUploadId, ihcUploadId, rank, 'rgb', report.generatedAt)}
          alt="The same field segmented without removing the brown stain"
        />
        <p className="nuc-compare__figures mono">
          {entry.instansegHaematoxylin} with the brown removed ·{' '}
          {entry.instansegRgb} with it left in
        </p>
      </section>
    </div>
  )
}

export function NucleiPanel({
  nuclei,
  heUploadId,
  ihcUploadId,
  marker,
  ihcMpp,
  alignment,
  hasConfirmedAlignment,
  running,
  onRun,
}: NucleiPanelProps) {
  // The outlines themselves are fetched by the page, once, for whichever of
  // steps 11 to 15 is on screen - see `showsCells` in `DemoPage`.
  const { report, capability, progress, message, error, geometry } = nuclei
  const [rank, setRank] = useState<number | null>(null)
  const [focus, setFocus] = useState<{ x: number; y: number; span: number } | null>(null)

  const region =
    (rank === null ? null : report?.regions.find((item) => item.rank === rank)) ??
    report?.regions[0] ??
    null

  // Every nucleus in one colour: nothing has been decided about any of them yet,
  // and a palette here would imply that something had. The ones in the border
  // band are the same blue, darkened, rather than a second colour, because they
  // are not a different kind of cell - they are the same cell, excluded from the
  // count along with the strip of area they sit in.
  const fields = useMemo(
    () =>
      nucleiOverlay(geometry, (nucleus) =>
        nucleus.counted ? CELL_COLOUR.nucleus : CELL_COLOUR.uncounted,
      ),
    [geometry],
  )

  const regions = useMemo(() => regionOutlines(alignment), [alignment])
  const targets = useMemo(() => flyTargets(fields), [fields])

  if (capability && !capability.available) {
    return (
      <div className="nuc-empty">
        <Badge tone="warn">Not installed</Badge>
        <p>{capability.reason}</p>
        <p className="nuc-empty__path mono">{capability.modelsDir}</p>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="nuc-empty">
        <h3>Find every cell, one at a time</h3>
        <p>
          The tumour regions are now on the marker slide. This step finds the individual
          cells inside them. The hard part is telling touching cells apart: if two are
          recorded as one, a cell disappears from the count.
        </p>
        <p>
          Cells are found on the <strong>blue stain</strong>, with the brown removed first.
          Only a sample of squares from each region is checked, spread evenly across it,
          which takes a minute or two instead of half an hour.
        </p>

        {!hasConfirmedAlignment && (
          <p className="nuc-empty__blocked">
            The slide matching has not been confirmed yet. Somebody has to check the two
            slides side by side and agree the regions landed on the same tissue before
            anything is measured inside them.
          </p>
        )}

        {error && <p className="nuc-empty__error">{error}</p>}

        {running ? (
          <div className="nuc-progress">
            <Spinner />
            <div className="nuc-progress__bar">
              <div style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
            <span className="mono">{message ?? 'working'}</span>
          </div>
        ) : (
          <Button onClick={onRun} disabled={!hasConfirmedAlignment || !heUploadId || !ihcUploadId}>
            Find the cells
          </Button>
        )}
      </div>
    )
  }

  if (!heUploadId || !ihcUploadId || !region) return null

  return (
    <div className="nuc-panel">
      <header className="nuc-panel__head">
        <div>
          <h3>
            {report.counted.toLocaleString()} cells found
            {marker && <span className="nuc-panel__marker"> · {marker}</span>}
          </h3>
          <p>
            Across{' '}
            <span className="mono">{report.regions.length}</span>{' '}
            {report.regions.length === 1 ? 'region' : 'regions'} of invasive tumour, from{' '}
            <span className="mono">{report.sampledMm2.toFixed(2)} mm²</span> of tissue. Every
            cell was found on the blue stain, with the brown removed first.
          </p>
        </div>
        <Badge tone="neutral">{report.modelName ?? 'model'}</Badge>
      </header>

      <DensityCheck report={report} />

      <FlyTo
        targets={targets}
        active={rank}
        onFly={(target: FlyTarget | null) => {
          setRank(target?.rank ?? null)
          setFocus(target ? target.box : null)
        }}
        noun="nuclei"
      />

      <SlideOverlayViewer
        uploadId={ihcUploadId}
        mpp={ihcMpp}
        regions={regions}
        fields={fields}
        focus={focus}
        legend={[
          { colour: CELL_COLOUR.nucleus.fill, label: 'A cell', count: report.counted },
          {
            colour: CELL_COLOUR.uncounted.fill,
            label: 'On the square’s edge — not counted',
          },
          { colour: 'rgba(56, 189, 248, 0.5)', label: 'A square that was checked' },
          { colour: 'rgba(167, 139, 250, 0.95)', label: 'Invasive tumour region' },
        ]}
        emptyHint="No outlines have been stored for this pair yet."
        caption={
          <>
            This is the live slide &mdash; scroll to zoom, drag to move, and use the corner
            control for full screen. Cells were only looked for inside the dashed squares.
            They are spread evenly rather than picked, so the cell density they report is
            representative of the whole region.
          </>
        }
      />

      <div className="nuc-region__stats">
        <Stat
          label="Cells counted"
          value={region.counted.toLocaleString()}
          hint={`region ${region.rank} · ${region.detected.toLocaleString()} found, edge ones excluded`}
        />
        <Stat
          label="Cells per mm²"
          value={region.densityPerMm2.toLocaleString()}
          hint="how tightly packed the tissue is"
        />
        <Stat
          label="Share checked"
          value={`${(region.sampledShare * 100).toFixed(1)}%`}
          hint={`${region.fields.length} of ${region.fieldsAvailable} squares`}
        />
        <Stat
          label="Typical cell size"
          value={`${region.medianAreaUm2.toFixed(0)} µm²`}
          hint="the middle value"
        />
      </div>

      <Comparisons
        report={report}
        heUploadId={heUploadId}
        ihcUploadId={ihcUploadId}
        rank={region.rank}
      />

      {report.notes.map((note) => (
        <p key={note} className="nuc-note">
          {note}
        </p>
      ))}

      <details className="nuc-detail">
        <summary>The technical detail</summary>

        <h4>Model</h4>
        <ul>
          <li>
            <strong>{report.modelName}</strong> {report.modelVersion} ({report.modelLicence}),
            run as TorchScript on the CPU at {report.modelMpp} µm/px — the input scale its own
            metadata declares. The <em>display</em> above will zoom past that; the model does
            not, because 0.25 µm/px is outside what it was trained on.
          </li>
          <li>
            The checkpoint is verified against its sha256 and then required to reproduce the
            upstream&rsquo;s own test tensor exactly before it will serve anything. Without
            the percentile stretch its metadata specifies, this model returns zero nuclei
            silently — which would read as &ldquo;this tissue has no cells&rdquo;.
          </li>
        </ul>

        <h4>Stain separation</h4>
        <ul>
          <li>
            Un-mixed with the <strong>{report.stain.basis}</strong> basis, haematoxylin
            redrawn at a contrast of {report.stain.gain}× against white.
          </li>
          {report.stain.macenkoHaematoxylinDriftDeg != null && (
            <li>
              A per-slide Macenko estimate was computed as a check and{' '}
              <em>not</em> used: its haematoxylin arm sits{' '}
              <span className="mono">
                {report.stain.macenkoHaematoxylinDriftDeg.toFixed(1)}°
              </span>{' '}
              from the published direction while its DAB arm sits{' '}
              <span className="mono">
                {report.stain.macenkoDabDriftDeg?.toFixed(1)}°
              </span>
              . Under heavy DAB there is no second colour in the cloud to find, so the
              estimate returns the darkness axis instead of the counterstain.
            </li>
          )}
        </ul>

        <h4>How the squares were chosen</h4>
        <ul>
          <li>
            Fields are taken at a uniform stride across each region, never ranked by stain
            content — ranking would pick the densest fields and report their density as the
            region&rsquo;s, which would destroy the cross-slide check above.
          </li>
          <li>
            The budget is split between regions <strong>in proportion to their area</strong>,
            so a region holding half the tumour gets about half the squares. An equal split
            would make these cells a sample of the region <em>list</em> rather than of the
            tumour — and on this project&rsquo;s own case that put 42% of the cells inside a
            region holding 84% of the invasive tissue.
          </li>
          <li>
            A nucleus whose centre falls in the edge band is drawn but not counted, and that
            band&rsquo;s area is removed from the denominator with it.
          </li>
          {report.regions.map((item) => (
            <li key={item.rank}>
              Region {item.rank}: {item.fields.length} of {item.fieldsAvailable} fields,{' '}
              <span className="mono">{item.sampledMm2.toFixed(3)}</span> of{' '}
              <span className="mono">{item.areaMm2.toFixed(2)}</span> mm², field-to-field
              density spread {(item.densityCv * 100).toFixed(0)}%.
            </li>
          ))}
        </ul>

        {Object.keys(report.densityByMarker).length > 1 && (
          <>
            <h4>Across this case&rsquo;s markers</h4>
            <ul>
              {Object.entries(report.densityByMarker).map(([letter, density]) => (
                <li key={letter}>
                  <span className="mono">{letter}</span>: {density.toLocaleString()} nuclei/mm²
                </li>
              ))}
            </ul>
          </>
        )}

        {report.seconds != null && (
          <p className="mono">Took {Math.round(report.seconds)} s.</p>
        )}
      </details>
    </div>
  )
}
