/**
 * The explanation half of step 2: classical metrics, per artefact class.
 *
 * This is the panel that turns "the model painted this red" into something a
 * viewer can check. For each metric it shows how the cells GrandQC called
 * `fold`, `pen`, `focus` and so on measure against the cells it called clean —
 * on this same slide, at the same resolution.
 *
 * Two rules are enforced in the markup rather than left to a caption:
 *
 *   1. Everything is a ratio to this slide's own clean tissue. There is no
 *      absolute threshold shown anywhere, because these metrics are not
 *      comparable between slides or between magnifications.
 *   2. A class measured from a handful of cells says nothing, so the sample
 *      size is shown on hover and thin classes are visibly greyed.
 *
 * The heatmap below the table is unlabelled by design now that its caption is
 * gone: the selected metric is the highlighted row in the table above it, and the
 * image's `alt` still names it. Two facts the caption used to carry and the code
 * still relies on - the map is one value per measured cell with glass left as
 * background rather than drawn as a zero, and everything is measured at the
 * artefact model's own resolution, not the pipeline's - live in the region
 * inspector, which re-measures one region at the working resolution.
 */

import { useState } from 'react'

import { qcHeatmapUrl } from '@/api/qc'
import { Badge } from '@/components/ui/Badge'
import type { QCClassShare, QCMetricKey, QCMetricSummary } from '@/types/qc'

import './qc.css'

/**
 * Below this many measured cells, a class mean is noise and is greyed out.
 *
 * Cells are sub-blocks of a model patch — sixteen per patch — so this is a
 * couple of patches' worth of evidence, not a couple of pixels'.
 */
const THIN_SAMPLE = 20

interface FeatureExplainProps {
  uploadId: string
  metrics: QCMetricSummary[]
  classes: QCClassShare[]
}

function ratioLabel(ratio: number): string {
  return ratio < 1 ? `${(ratio * 100).toFixed(0)}%` : `${ratio.toFixed(2)}×`
}

export function FeatureExplain({ uploadId, metrics, classes }: FeatureExplainProps) {
  const [heatmap, setHeatmap] = useState<QCMetricKey>('tenengrad')

  // Only classes GrandQC actually found on this slide are worth a column.
  const present = classes.filter(
    (item) => item.isArtefact && item.shareOfTissue > 0,
  )
  const selected = metrics.find((metric) => metric.key === heatmap)

  if (present.length === 0) {
    return (
      <div className="qc-explain qc-explain--empty">
        <span className="eyebrow">why a region fails</span>
        <p className="qc-explain__none">
          GrandQC found no artefacts on this slide, so there is nothing to explain. The metrics
          below were still measured on every tissue cell — they are what a comparison would be
          drawn against if there were.
        </p>
        <MetricList metrics={metrics} />
      </div>
    )
  }

  return (
    <div className="qc-explain">
      <div className="qc-explain__head">
        <span className="eyebrow">why those regions failed</span>
        <Badge tone="neutral">classical · explains, does not decide</Badge>
      </div>

      <p className="qc-explain__lede">
        Each artefact class, measured against the clean tissue on this same slide. A ratio near
        100% means the metric cannot tell the two apart — which is exactly why the decision is
        not made this way.
      </p>

      <div className="qc-explain__table-wrap">
        <table className="qc-explain__table">
          <thead>
            <tr>
              <th scope="col">Metric</th>
              {present.map((item) => (
                <th key={item.key} scope="col">
                  <span
                    className="qc-bar__swatch"
                    style={{ background: item.colour }}
                    aria-hidden
                  />
                  {item.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.map((metric) => (
              <tr
                key={metric.key}
                className={metric.key === heatmap ? 'qc-explain__row--active' : undefined}
              >
                <th scope="row">
                  <button
                    type="button"
                    className="qc-explain__metric"
                    onClick={() => setHeatmap(metric.key)}
                    title={metric.description}
                  >
                    {metric.label}
                    <span className="qc-explain__arrow" aria-hidden>
                      {metric.lowIsBad ? '↓ suspicious' : '↕'}
                    </span>
                  </button>
                </th>
                {present.map((item) => {
                  const ratio = metric.ratioToClean[item.key]
                  const sample = metric.sampleSizes[item.key] ?? 0
                  const thin = sample < THIN_SAMPLE
                  return (
                    <td
                      key={item.key}
                      className={thin ? 'qc-explain__cell--thin' : undefined}
                      title={thin ? `only ${sample} cells measured` : `${sample} cells measured`}
                    >
                      {ratio === undefined ? (
                        <span className="qc-explain__na">—</span>
                      ) : (
                        <span className="mono">{ratioLabel(ratio)}</span>
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* --- the map for the selected metric ------------------------------- */}
      <figure className="qc-heatmap">
        <img
          src={qcHeatmapUrl(uploadId, heatmap)}
          alt={`${selected?.label ?? heatmap} across the slide`}
          loading="lazy"
        />
      </figure>
    </div>
  )
}

/** The clean-tissue baselines on their own, when there is nothing to compare. */
function MetricList({ metrics }: { metrics: QCMetricSummary[] }) {
  return (
    <ul className="qc-metric-list">
      {metrics.map((metric) => (
        <li key={metric.key}>
          <span className="qc-metric-list__label">{metric.label}</span>
          <span className="qc-metric-list__value mono">
            {metric.cleanMean === null ? '—' : metric.cleanMean.toPrecision(3)}
          </span>
          <span className="qc-metric-list__unit">{metric.unit}</span>
        </li>
      ))}
    </ul>
  )
}
