import { useState } from 'react'

import { alignmentCropUrl, alignmentPanelUrl } from '@/api/ihcAlignment'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import type { AlignmentDiagnostics } from '@/types/ihcAlignment'

import type { IhcAlignmentStateValue } from './useIhcAlignment'

import './ihcAlignment.css'

interface IhcAlignmentPanelProps {
  alignment: IhcAlignmentStateValue
  /** Step 9 has to have produced regions before there is anything to carry across. */
  hasRoi: boolean
  marker: string | null
  running: boolean
  onRun: () => void
}

function Measure({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint?: string
}) {
  return (
    <div className="align-measure">
      <div className="align-measure__label">{label}</div>
      <div className="align-measure__value mono">{value}</div>
      {hint && <div className="align-measure__hint">{hint}</div>}
    </div>
  )
}

function Measures({ diagnostics }: { diagnostics: AlignmentDiagnostics }) {
  const nmi = diagnostics.alignmentNmi
  const base = diagnostics.alignmentNmiUnregistered

  return (
    <div className="align-measures">
      <Measure
        label="There-and-back error"
        value={
          diagnostics.roundTripMedianUm != null
            ? `${diagnostics.roundTripMedianUm.toFixed(1)} µm`
            : '—'
        }
        hint="move the regions across and back — how far they drift"
      />
      <Measure
        label="How well the shapes agree"
        value={nmi != null ? nmi.toFixed(3) : '—'}
        hint={base != null ? `${base.toFixed(3)} before matching` : undefined}
      />
      <Measure
        label="Tissue area"
        value={
          diagnostics.heTissueMm2 != null && diagnostics.ihcTissueMm2 != null
            ? `${diagnostics.heTissueMm2.toFixed(0)} / ${diagnostics.ihcTissueMm2.toFixed(0)} mm²`
            : '—'
        }
        hint="H&E slide against marker slide"
      />
      <Measure
        label="Took"
        value={diagnostics.seconds != null ? `${Math.round(diagnostics.seconds)} s` : '—'}
        hint={diagnostics.reusedRegistration ? 'reused an earlier result' : undefined}
      />
    </div>
  )
}

/**
 * Step 10: the two slides side by side, and the decision that follows.
 *
 * The regions are drawn as **borders only** on both panels. That is the whole
 * design of this screen: a viewer is deciding whether the same tissue is
 * inside the outline on each side, and a filled region hides the pixels they
 * need in order to tell.
 */
export function IhcAlignmentPanel({
  alignment,
  hasRoi,
  marker,
  running,
  onRun,
}: IhcAlignmentPanelProps) {
  const { capability, report, state, message, error } = alignment
  const [confirming, setConfirming] = useState(false)
  //: Which slide the region crops are taken from. Defaults to the IHC slide - that
  //: is the result being judged - with the H&E a click away as the thing to judge it
  //: against. Two serial sections are never identical, so the question is whether the
  //: same architecture is in frame, and that cannot be answered from one picture.
  const [cropSource, setCropSource] = useState<'ihc' | 'he'>('ihc')

  if (capability && !capability.available) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">slide matching not installed</span>
        <p className="placeholder__text">{capability.reason}</p>
      </div>
    )
  }

  if (!hasRoi) {
    return (
      <div className="placeholder">
        <p className="placeholder__text">
          The tumour regions have to be found on the H&amp;E slide before they can be
          copied onto the marker slide. Go back and outline them first.
        </p>
      </div>
    )
  }

  if (running || alignment.running) {
    return (
      <div className="placeholder placeholder--running" aria-live="polite">
        <span className="placeholder__spinner" />
        <p className="placeholder__text">
          {message ?? 'Matching the two slides…'} This takes a few minutes. The slides are
          first lined up roughly, then stretched to fit each other more exactly.
        </p>
      </div>
    )
  }

  // Checked before `error`, deliberately. A refusal rejects the step's promise
  // so the stepper does not read "Completed", which also surfaces it as an
  // error here - but a refusal is a measured result with reasons attached, and
  // showing a bare error box instead would throw all of that away.
  if (report?.state === 'refused') {
    return (
      <div className="align-refused">
        <Badge tone="danger">refused</Badge>
        <p className="align-refused__lead">
          The two slides were matched and measured, but the result is not reliable enough
          to use. No regions were copied across.
        </p>
        <ul className="align-refused__reasons">
          {report.refusalReasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
        <Measures diagnostics={report.diagnostics} />
        <Button variant="secondary" onClick={onRun}>
          Try matching again
        </Button>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">matching failed</span>
        <p className="placeholder__text">{error}</p>
        <Button variant="secondary" onClick={onRun}>
          Try again
        </Button>
      </div>
    )
  }

  if (!report) {
    return (
      <div className="align-start">
        <p className="align-start__text">
          The tumour regions were found on the H&E slide. They now have to be copied onto
          the {marker ? `${marker} ` : ''}marker slide. The two are different slices of
          the same block, so the same spot is not in the same place on both.
        </p>
        <Button onClick={onRun} attention>
          Match the slides
        </Button>
      </div>
    )
  }

  const cacheKey = report.generatedAt
  const he = report.heUploadId
  const ihc = report.ihcUploadId

  return (
    <div className="align">
      <div className="align__header">
        <Badge tone={report.confirmed ? 'success' : 'warn'}>
          {report.confirmed ? 'confirmed' : 'needs your check'}
        </Badge>
        <span className="align__count">
          {report.regions.length} tumour region{report.regions.length === 1 ? '' : 's'}{' '}
          copied across
        </span>
        {report.areaCoverage > 0 ? (
          <span
            className={
              report.areaCoverage >= 0.8 ? 'align__coverage' : 'align__coverage is-thin'
            }
          >
            {Math.round(report.areaCoverage * 100)}% of the tumour
            <small>
              {report.carriedMm2.toFixed(1)} of {report.invasiveMm2.toFixed(1)} mm²
            </small>
          </span>
        ) : null}
      </div>

      <p className="align__lead">
        The same regions, drawn on both slides as outlines only. Check that each numbered
        region contains the same structures on both sides.
      </p>

      {report.areaCoverage > 0 && report.areaCoverage < 0.8 ? (
        <p className="align__thin">
          These regions cover {Math.round(report.areaCoverage * 100)}% of the tumour found
          on the H&amp;E slide. The rest will not be included in the score, whereas a
          pathologist would look at all of it.
        </p>
      ) : null}

      <div className="align__pair">
        <figure className="align__figure">
          <img src={alignmentPanelUrl(he, ihc, 'he_borders', cacheKey)} alt="H&E slide with the tumour regions outlined" />
          <figcaption>H&E slide &mdash; where the regions were found</figcaption>
        </figure>
        <figure className="align__figure">
          <img src={alignmentPanelUrl(he, ihc, 'ihc_borders', cacheKey)} alt="Marker slide with the copied regions outlined" />
          <figcaption>{marker ? `${marker} slide` : 'Marker slide'} &mdash; where they landed</figcaption>
        </figure>
      </div>

      <Measures diagnostics={report.diagnostics} />

      {report.notes.length > 0 && (
        <ul className="align__notes">
          {report.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}

      {/* The decision this screen asks for is "did these regions land on the same
          tissue", and that is a comparison - so the crops have to be showable from
          both slides, not just the one they landed on. The server has always served
          either; only the IHC side was ever requested. */}
      <div className="align__crops-head">
        <h3 className="align__subhead">
          {report.regions.length === 1
            ? 'The region'
            : `The ${report.regions.length} regions`}
          , on the {cropSource === 'ihc' ? `${marker ?? 'marker'}` : 'H&E'} slide
        </h3>
        <div className="align__crop-toggle" role="group" aria-label="Which slide to crop from">
          <button
            type="button"
            className={cropSource === 'ihc' ? 'is-active' : undefined}
            aria-pressed={cropSource === 'ihc'}
            onClick={() => setCropSource('ihc')}
          >
            Where they landed
          </button>
          <button
            type="button"
            className={cropSource === 'he' ? 'is-active' : undefined}
            aria-pressed={cropSource === 'he'}
            onClick={() => setCropSource('he')}
          >
            Where they came from
          </button>
        </div>
      </div>
      <div className="align__crops">
        {report.regions.map((region) => (
          <figure key={region.rank} className="align__crop">
            <img
              src={alignmentCropUrl(he, ihc, region.rank, cropSource, cacheKey)}
              alt={
                cropSource === 'ihc'
                  ? `Tumour region ${region.rank} on the marker slide`
                  : `Tumour region ${region.rank} on the H&E slide it came from`
              }
            />
            <figcaption>
              <span className="mono">{region.rank}</span> · {region.areaMm2.toFixed(2)} mm²
            </figcaption>
          </figure>
        ))}
      </div>

      <div className="align__decision">
        <p className="align__decision-text">
          Nothing is measured inside these regions until you confirm they landed on the
          right tissue.
        </p>
        <div className="align__decision-actions">
          <Button
            onClick={() => {
              setConfirming(true)
              void alignment.confirm(!report.confirmed).finally(() => setConfirming(false))
            }}
            loading={confirming}
            variant={report.confirmed ? 'secondary' : 'primary'}
            attention={!report.confirmed}
          >
            {report.confirmed ? 'Undo confirmation' : 'Yes, this looks correct'}
          </Button>
          <Button variant="ghost" onClick={onRun}>
            Try matching again
          </Button>
        </div>
      </div>

      {state === 'ready' && !report.confirmed && (
        <p className="align__blocked">
          The next steps stay locked until you confirm this.
        </p>
      )}
    </div>
  )
}
