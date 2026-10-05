/**
 * Step 2's screen, in the order the story wants telling.
 *
 *   1. the two models, named, so "QC" is not one opaque box
 *   2. the overlay, with the QC on/off switch over it
 *   3. what each artefact class costs in tissue area
 *   4. the classical metrics that explain those calls
 *   5. one region, inspected closely
 *   6. the citation
 *
 * The overlay honours the switch: QC off shows the plain slide, QC on shows the
 * artefacts tinted. That is the whole argument of the step in one gesture — the
 * area that lights up is the area a score would otherwise have been computed
 * over.
 */

import { useState } from 'react'

import { qcClassesUrl, qcMaskUrl, qcOverlayUrl, qcTissueUrl } from '@/api/qc'
import { slideThumbnailUrl } from '@/api/uploads'
import { Badge } from '@/components/ui/Badge'
import { formatCount } from '@/lib/format'
import type { QCReport } from '@/types/qc'
import type { SlideReadout } from '@/types/slide'

import { ArtefactBars } from './ArtefactBars'
import { FeatureExplain } from './FeatureExplain'
import { QCGate } from './QCGate'
import { RegionInspector } from './RegionInspector'

import './qc.css'

type MapLayer = 'overlay' | 'tissue' | 'classes'

interface QCViewProps {
  report: QCReport
  readout: SlideReadout | null
}

export function QCView({ report, readout }: QCViewProps) {
  const [qcOn, setQcOn] = useState(true)
  const [layer, setLayer] = useState<MapLayer>('overlay')

  const { uploadId } = report
  const tissueModel = report.models.find((model) => model.role === 'tissue')
  const artefactModel = report.models.find((model) => model.role === 'artefact')

  const mapSrc =
    !qcOn && layer === 'overlay'
      ? slideThumbnailUrl(uploadId, 1600)
      : layer === 'tissue'
        ? qcTissueUrl(uploadId)
        : layer === 'classes'
          ? qcClassesUrl(uploadId)
          : qcOverlayUrl(uploadId)

  return (
    <div className="qc">
      {/* --- 1. the two models -------------------------------------------- */}
      <div className="qc-models">
        <span className="eyebrow">what ran</span>
        <ol className="qc-models__list">
          <li className="qc-models__item">
            <span className="qc-models__step mono">1</span>
            <div>
              <strong>Find the tissue</strong>
              <span className="qc-models__meta mono">
                {tissueModel?.name} · {tissueModel?.mpp} µm/px · {tissueModel?.classes} classes
              </span>
              <span className="qc-models__why">
                Finds the tissue first. This is quick, and it saves the slower check from
                looking at empty glass.
              </span>
            </div>
          </li>
          <li className="qc-models__item">
            <span className="qc-models__step mono">2</span>
            <div>
              <strong>Find the problem areas</strong>
              <span className="qc-models__meta mono">
                {artefactModel?.name} · {artefactModel?.magnification} ·{' '}
                {artefactModel?.mpp} µm/px · {artefactModel?.classes} channels
              </span>
              <span className="qc-models__why">
                Marks each problem area: folds, dark spots, pen marks, slide edges and blurry
                patches.
              </span>
            </div>
          </li>
        </ol>
        <div className="qc-models__stats mono">
          {formatCount(report.grid.patchesInferred)} patches checked ·{' '}
          {formatCount(report.grid.patchesSkipped)} skipped as empty glass ·{' '}
          {formatCount(report.grid.measuredCells)} small areas measured ·{' '}
          {report.run.durationSeconds ?? '—'} s
        </div>
      </div>

      {/* --- 2. the map --------------------------------------------------- */}
      <div className="qc-map">
        <div className="qc-map__bar">
          <div className="qc-map__layers">
            {(
              [
                ['overlay', 'Problems on the slide'],
                ['tissue', 'Where the tissue is'],
                ['classes', 'Problems only'],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                className={layer === key ? 'qc-map__tab qc-map__tab--on' : 'qc-map__tab'}
                onClick={() => setLayer(key)}
              >
                {label}
              </button>
            ))}
          </div>
          <a className="qc-map__download mono" href={qcMaskUrl(uploadId)} download>
            download the map
          </a>
        </div>

        <figure className="qc-map__figure">
          <img
            src={mapSrc}
            alt={
              layer === 'tissue'
                ? 'Where the tissue was found'
                : qcOn
                  ? 'Problem areas found on the slide'
                  : 'The slide with no quality check applied'
            }
          />
          <figcaption className="qc-map__caption mono">
            {layer === 'overlay' && !qcOn
              ? 'Check off — the slide exactly as scanned, problems included'
              : layer === 'overlay'
                ? 'Check on — coloured areas are left out of every later step'
                : layer === 'tissue'
                  ? 'Blue = tissue'
                  : 'Each colour is a different kind of problem'}
          </figcaption>
        </figure>

        <QCGate gate={report.gate} tissue={report.tissue} on={qcOn} onToggle={setQcOn} />
      </div>

      {/* --- 3. the cost -------------------------------------------------- */}
      <ArtefactBars classes={report.classes} />

      {/* --- 4. the explanation ------------------------------------------- */}
      <FeatureExplain uploadId={uploadId} metrics={report.metrics} classes={report.classes} />

      {/* --- 5. one region, closely --------------------------------------- */}
      {readout && (
        <RegionInspector
          uploadId={uploadId}
          slideWidth={readout.widthPx}
          slideHeight={readout.heightPx}
          classes={report.classes}
          targetMpp={readout.targetMpp}
        />
      )}

      {/* --- 6. the citation ---------------------------------------------- */}
      <p className="qc-notes__citation">
        <Badge tone="neutral">GrandQC</Badge> {report.citation}
      </p>
    </div>
  )
}
