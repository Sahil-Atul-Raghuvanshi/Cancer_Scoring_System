/**
 * Step 1's result: the real slide, and the pyramid actually found inside it.
 *
 * Every figure here was read from the uploaded file. The one that matters is
 * mpp - the working level is derived from it, because the same level index is
 * a different resolution on a different scanner.
 */

import { cn } from '@/lib/cn'
import type { SlideReadout } from '@/types/slide'

import { PyramidLevels } from './PyramidLevels'
import { SlideViewer } from './SlideViewer'
import { ResolutionControls } from './ResolutionControls'

import './pipeline.css'

function Figure({
  label,
  value,
  hint,
  emphasis = false,
  delay = 0,
}: {
  label: string
  value: string
  hint?: string
  emphasis?: boolean
  delay?: number
}) {
  return (
    <div
      className={emphasis ? 'metric metric--emphasis' : 'metric'}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="metric__label">{label}</div>
      <div className="metric__value">
        <span style={{ animationDelay: `${delay + 60}ms` }}>{value}</span>
      </div>
      {hint && <div className="metric__hint">{hint}</div>}
    </div>
  )
}

interface SlideReadoutViewProps {
  readout: SlideReadout
  busy?: boolean
  onAdjust?: (next: { targetMpp?: number | null; mppOverride?: number | null }) => void
}

export function SlideReadoutView({ readout, busy = false, onAdjust }: SlideReadoutViewProps) {
  const {
    mpp,
    magnification,
    workingLevel,
    workingMpp,
    workingDownsample,
    exactLevelMatch,
    targetMpp,
  } = readout

  return (
    <div className={cn('readout', busy && 'readout--busy')}>
      {/* --- the slide itself --------------------------------------------- */}
      <figure className="readout__slide">
        <SlideViewer
          uploadId={readout.uploadId}
          mpp={readout.mpp}
          levels={readout.levels}
          workingLevel={readout.workingLevel}
        />
        <figcaption className="readout__caption mono">
          drag to move, scroll to zoom &mdash; tiles load as you go
        </figcaption>
      </figure>

      {/* --- the two knobs ------------------------------------------------ */}
      {onAdjust && <ResolutionControls readout={readout} busy={busy} onChange={onAdjust} />}

      {/* --- headline figures --------------------------------------------- */}
      <div className="metrics">
        <Figure
          label="Dimensions"
          value={`${readout.widthPx.toLocaleString()} × ${readout.heightPx.toLocaleString()} px`}
          hint={`${readout.megapixels.toLocaleString()} megapixels`}
          delay={0}
        />
        <Figure
          label="Resolution"
          value={mpp ? `${mpp} µm/px` : 'not recorded'}
          hint={
            readout.mppSource === 'override'
              ? `${magnification} · set by you, not measured`
              : mpp
                ? magnification
                : 'no physical scale in this file'
          }
          emphasis
          delay={70}
        />
        <Figure
          label="Pyramid levels"
          value={String(readout.levelCount)}
          hint={readout.vendor ? `${readout.vendor} scan` : undefined}
          delay={140}
        />
        <Figure
          label="File size"
          value={`${readout.fileSizeMb} MB`}
          hint={readout.filename}
          delay={210}
        />
      </div>

      {/* --- the pyramid ------------------------------------------------- */}
      <PyramidLevels readout={readout} />

      {/* --- the conversion this step exists to make ----------------------- */}
      <div className="conversion">
        <span className="eyebrow">which zoom level is used</span>
        {mpp ? (
          <p className="conversion__body">
            We want to work at <strong className="mono">{targetMpp} µm/px</strong>. The
            closest level in this file is{' '}
            <strong className="mono">level {workingLevel}</strong> at{' '}
            <strong className="mono">{workingMpp} µm/px</strong>
            {exactLevelMatch ? (
              <> &mdash; an exact match, so it is used as it is.</>
            ) : (
              <>
                , so it is shrunk <strong className="mono">{workingDownsample}×</strong> to
                match. We never blow a coarser level up, because that would invent detail
                the scanner never captured.
              </>
            )}
          </p>
        ) : (
          <p className="conversion__body">
            This file does not say how big a pixel is in real life, so we fall back to the
            largest level. Every later step is set in microns, so a slide without this scale
            cannot be scored properly.
          </p>
        )}
      </div>
    </div>
  )
}
