/**
 * The two adjustable inputs to step 1.
 *
 * They are different things and the UI has to keep them apart:
 *
 *   Target      the resolution to answer the question AT. Changing it changes
 *               which pyramid level this readout reports, and how much software
 *               downsampling would close the gap.
 *   Slide scale what this file claims a pixel measures. Normally the scanner
 *               records it; when it does not, or records it wrongly, it can be
 *               supplied here - but that is an assertion, not a measurement,
 *               so the readout is labelled accordingly.
 *
 * **The target does not set a resolution for the rest of the pipeline, and saying
 * so would be the more flattering lie.** Every later step declares its own physical
 * resolution, because each is answering a different question: the tissue mask and
 * the white point work at 2 um/px, quality control at 1.5, the tissue-type model at
 * whatever its checkpoint was fitted at. That is the pipeline's own rule - convert
 * using microns, never a level index - and a single global target would quietly
 * override all of it. So this control answers "at 20x, which level would be read on
 * THIS scanner, and is it an exact match?", which is a real property of the file and
 * the reason the presets are labelled with the steps that use each scale.
 */

import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/Badge'
import { cn } from '@/lib/cn'
import type { SlideReadout } from '@/types/slide'

import './pipeline.css'

/** The magnifications the pipeline's steps are actually specified at. */
const TARGET_PRESETS = [
  { mpp: 0.25, label: '0.25', hint: '40× — close up, for cells' },
  { mpp: 0.5, label: '0.5', hint: '20× — for tissue types' },
  { mpp: 1.0, label: '1.0', hint: '10× — for wider context' },
  { mpp: 2.0, label: '2.0', hint: '5× — zoomed out, for whole-slide checks' },
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
          <span className="eyebrow">zoom level to work at</span>
          <span className="resolution__help">
            this decides which level of the file is read
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
          <span className="eyebrow">pixel size on this slide</span>
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
            aria-label="Microns per pixel at full size"
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
            ? 'Recorded by the scanner. Only change this if you know the file is wrong.'
            : mppSource === 'override'
              ? 'You set this by hand. Every number below is now based on it.'
              : 'This file does not record a pixel size. Enter one, or results cannot be compared with other slides.'}
        </p>
      </div>
    </div>
  )
}
