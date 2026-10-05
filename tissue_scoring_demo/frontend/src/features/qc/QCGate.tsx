/**
 * The QC on/off comparison.
 *
 * The pipeline guide asks for two scores side by side, and this deliberately
 * does not show that: steps 3 to 16 are not built, so there is no score to
 * move, and inventing one would make the walkthrough describe software that
 * does not exist. What it shows instead is the quantity that genuinely changes
 * today - the tissue area downstream steps would receive - which is the
 * denominator that score will eventually be divided by.
 *
 * The toggle is not cosmetic: flipping it swaps which mask the viewer is
 * looking at, so the highlighted area and the number move together.
 */

import { Badge } from '@/components/ui/Badge'
import type { QCGate as QCGateData, QCTissue } from '@/types/qc'

import './qc.css'

interface QCGateProps {
  gate: QCGateData
  tissue: QCTissue
  on: boolean
  onToggle: (on: boolean) => void
}

function area(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(2)} mm²`
}

export function QCGate({ gate, tissue, on, onToggle }: QCGateProps) {
  const removed = gate.removedShare
  const material = removed >= 0.01

  return (
    <div className="qc-gate">
      <div className="qc-gate__head">
        <span className="eyebrow">how much tissue is left to use</span>
        <button
          type="button"
          className={on ? 'qc-switch qc-switch--on' : 'qc-switch'}
          onClick={() => onToggle(!on)}
          aria-pressed={on}
        >
          <span className="qc-switch__label">QC</span>
          <span className="qc-switch__track" aria-hidden>
            <span className="qc-switch__knob" />
          </span>
          <span className="qc-switch__state mono">{on ? 'ON' : 'OFF'}</span>
        </button>
      </div>

      <div className="qc-gate__pair">
        <div className={on ? 'qc-gate__side' : 'qc-gate__side qc-gate__side--live'}>
          <span className="qc-gate__caption">QC off</span>
          <span className="qc-gate__value mono">{area(gate.qcOffAreaMm2)}</span>
          <span className="qc-gate__hint">all tissue, problems included</span>
        </div>

        <div className="qc-gate__arrow" aria-hidden>
          →
        </div>

        <div className={on ? 'qc-gate__side qc-gate__side--live' : 'qc-gate__side'}>
          <span className="qc-gate__caption">QC on</span>
          <span className="qc-gate__value mono">{area(gate.qcOnAreaMm2)}</span>
          <span className="qc-gate__hint">problem areas removed</span>
        </div>
      </div>

      <div className="qc-gate__delta">
        <Badge tone={material ? 'warn' : 'neutral'}>
          {`−${(removed * 100).toFixed(2)}% of tissue`}
        </Badge>
        <span className="qc-gate__delta-text">
          {material ? (
            <>
              {area(gate.removedAreaMm2)} of tissue has a known problem and is now excluded.
              Without this check, cells and stain would have been counted from pixels the
              scanner got wrong.
            </>
          ) : (
            <>
              Only {area(gate.removedAreaMm2)} was removed, so this slide is clean. That is a
              useful result in itself.
            </>
          )}
        </span>
      </div>

      <p className="qc-gate__note">{gate.note}</p>

      {tissue.tissueSource === 'otsu' && (
        <p className="qc-gate__warning">
          The tissue model is missing, so tissue was found with a simple colour rule instead.
          That rule also counts pen marks as tissue, so treat the area above as rough. Add{' '}
          <code className="mono">Tissue_Detection_MPP10.pth</code> for accurate figures.
        </p>
      )}
    </div>
  )
}
