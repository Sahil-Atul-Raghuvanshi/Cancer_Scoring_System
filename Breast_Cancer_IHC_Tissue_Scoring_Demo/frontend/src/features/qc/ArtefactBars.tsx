/**
 * How much of the tissue each artefact class accounts for.
 *
 * Bars are scaled to the largest artefact on this slide, not to 100%: on a good
 * slide every artefact is under 2%, and bars scaled to 100% would all be
 * invisible slivers that say nothing. The number next to each bar is the true
 * percentage, so the scaling exaggerates nothing - it only makes the comparison
 * between classes readable.
 */

import { Badge } from '@/components/ui/Badge'
import type { QCClassShare } from '@/types/qc'

import './qc.css'

interface ArtefactBarsProps {
  classes: QCClassShare[]
  /** Highlighted class, when the region inspector has one selected. */
  active?: string | null
  onSelect?: (key: string) => void
}

function percent(value: number): string {
  if (value === 0) return '0%'
  if (value < 0.0001) return '<0.01%'
  return `${(value * 100).toFixed(2)}%`
}

export function ArtefactBars({ classes, active, onSelect }: ArtefactBarsProps) {
  const artefacts = classes.filter((item) => item.isArtefact)
  const clean = classes.find((item) => item.key === 'tissue')
  const peak = Math.max(...artefacts.map((item) => item.shareOfTissue), 0.0001)

  const total = artefacts.reduce((sum, item) => sum + item.shareOfTissue, 0)

  return (
    <div className="qc-bars">
      <div className="qc-bars__head">
        <span className="eyebrow">artefacts, as a share of tissue</span>
        {total === 0 && <Badge tone="success">none detected</Badge>}
      </div>

      {clean && (
        <div className="qc-bars__clean">
          <span className="qc-bars__clean-value mono">{percent(clean.shareOfTissue)}</span>
          <span className="qc-bars__clean-label">
            of the tissue is clean and passes into step 3
          </span>
        </div>
      )}

      <ul className="qc-bars__list">
        {artefacts.map((item) => (
          <li
            key={item.key}
            className={
              active === item.key ? 'qc-bar qc-bar--active' : 'qc-bar'
            }
          >
            <button
              type="button"
              className="qc-bar__button"
              onClick={() => onSelect?.(item.key)}
              disabled={!onSelect}
              title={item.blurb}
            >
              <span className="qc-bar__label">
                <span
                  className="qc-bar__swatch"
                  style={{ background: item.colour }}
                  aria-hidden
                />
                {item.label}
              </span>

              <span className="qc-bar__track">
                <span
                  className="qc-bar__fill"
                  style={{
                    width: `${Math.max(item.shareOfTissue > 0 ? 2 : 0, (item.shareOfTissue / peak) * 100)}%`,
                    background: item.colour,
                  }}
                />
              </span>

              <span className="qc-bar__value mono">{percent(item.shareOfTissue)}</span>
              <span className="qc-bar__area mono">
                {item.areaMm2 !== null ? `${item.areaMm2.toFixed(2)} mm²` : '—'}
              </span>
            </button>
          </li>
        ))}
      </ul>

      <p className="qc-bars__foot">
        Bars are scaled to the largest artefact on this slide so the classes can be compared;
        the percentages are absolute. Glass is excluded from the denominator — it was never
        analysable.
      </p>
    </div>
  )
}
