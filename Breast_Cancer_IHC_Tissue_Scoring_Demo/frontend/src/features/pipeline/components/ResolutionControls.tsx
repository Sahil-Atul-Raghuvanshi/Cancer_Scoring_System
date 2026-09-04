/**
 * The two adjustable inputs to step 1.
 *
 * They are different things and the UI has to keep them apart:
 *
 *   Target      what the pipeline wants to work at. Changing it changes which
 *               pyramid level gets read, and how much software downsampling
 *               closes the gap.
 *   Slide scale what this file claims a pixel measures. Normally the scanner
 *               records it; when it does not, or records it wrongly, it can be
 *               supplied here - but that is an assertion, not a measurement,
 *               so the readout is labelled accordingly.
 */

import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/Badge'
import { cn } from '@/lib/cn'
import type { SlideReadout } from '@/types/slide'

import './pipeline.css'

/** The magnifications the pipeline's steps are actually specified at. */
const TARGET_PRESETS = [
  { mpp: 0.25, label: '0.25', hint: '40× · nuclei, membranes' },
  { mpp: 0.5, label: '0.5', hint: '20× · tissue-type models' },
  { mpp: 1.0, label: '1.0', hint: '10× · region context' },
  { mpp: 2.0, label: '2.0', hint: '5× · tissue mask, QC' },
]

interface ResolutionControlsProps {
  readout: SlideReadout
  busy: boolean
  onChange: (next: { targetMpp?: number | null; mppOverride?: number | null }) => void
}

export function ResolutionControls({ readout, busy, onChange }: ResolutionControlsProps) {
  const { targetMpp, mppSource, scannerMpp, mpp } = readout
  const [draft, setDraft] = useState(mpp ? String(mpp) : '')

  // Keep the field in step when the readout changes underneath it.
  useEffect(() => {
    setDraft(mpp ? String(mpp) : '')
  }, [mpp])

  const submitOverride = () => {
    const value = Number.parseFloat(draft)
    if (Number.isFinite(value) && value > 0 && value !== mpp) {
      onChange({ mppOverride: value })
    }
  }

  const isCustomTarget = !TARGET_PRESETS.some((preset) => preset.mpp === targetMpp)

  return (
    <div className={cn('resolution', busy && 'resolution--busy')}>
      {/* --- target ------------------------------------------------------- */}
      <div className="resolution__group">
        <div className="resolution__label">
          <span className="eyebrow">work at</span>
          <span className="resolution__help">
            which level gets read follows from this
          </span>
        </div>

        <div className="resolution__presets" role="group" aria-label="Target resolution">
          {TARGET_PRESETS.map((preset) => (
            <button
              key={preset.mpp}
              type="button"
              className="resolution__preset"
              aria-pressed={targetMpp === preset.mpp}
              title={preset.hint}
              disabled={busy}
              onClick={() => onChange({ targetMpp: preset.mpp })}
            >
              {preset.label}
            </button>
          ))}
          {isCustomTarget && (
            <span className="resolution__preset resolution__preset--custom" aria-current>
              {targetMpp}
            </span>
          )}
          <span className="resolution__unit mono">µm/px</span>
        </div>
      </div>

      {/* --- the slide's own scale ---------------------------------------- */}
      <div className="resolution__group">
        <div className="resolution__label">
          <span className="eyebrow">slide scale</span>
          {mppSource === 'scanner' && <Badge tone="success">from scanner</Badge>}
          {mppSource === 'override' && <Badge tone="warn">set by you</Badge>}
          {mppSource === 'unknown' && <Badge tone="danger">not recorded</Badge>}
        </div>

        <div className="resolution__field">
          <input
            type="number"
            min="0.01"
            max="64"
            step="0.0001"
            inputMode="decimal"
            className="resolution__input mono"
            aria-label="Microns per pixel at level 0"
            placeholder="e.g. 0.25"
            value={draft}
            disabled={busy}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={submitOverride}
            onKeyDown={(event) => {
              if (event.key === 'Enter') submitOverride()
            }}
          />
          <span className="resolution__unit mono">µm/px</span>

          {mppSource === 'override' && (
            <button
              type="button"
              className="resolution__reset"
              disabled={busy}
              onClick={() => onChange({ mppOverride: null })}
            >
              {scannerMpp ? `reset to ${scannerMpp}` : 'clear'}
            </button>
          )}
        </div>

        <p className="resolution__note">
          {mppSource === 'scanner'
            ? 'Measured and recorded by the scanner. Change it only to correct a file you know is wrong.'
            : mppSource === 'override'
              ? 'This is your assertion, not a measurement — every figure below now derives from it.'
              : 'This file records no physical scale, so nothing downstream is comparable until one is supplied.'}
        </p>
      </div>
    </div>
  )
}
