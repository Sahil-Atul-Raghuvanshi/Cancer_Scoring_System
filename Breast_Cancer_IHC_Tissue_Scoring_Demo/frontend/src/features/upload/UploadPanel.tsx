import { useCallback, useRef, useState } from 'react'
import type { DragEvent } from 'react'

import { Button } from '@/components/ui/Button'
import { cn } from '@/lib/cn'
import type { UploadCapability } from '@/types/pipeline'

import { useSlideSession } from './slideSessionContext'

import './upload.css'

function SlideIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect x="4" y="2.5" width="16" height="19" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <path d="M8 7h8M8 11h8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <circle cx="10" cy="16" r="1.6" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="14.5" cy="17" r="1.2" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  )
}

const PHASE_COPY = {
  hashing: 'Checksumming the file…',
  transferring: 'Transferring chunks…',
  assembling: 'Reassembling and validating on the server…',
} as const

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`
  return `${(bytes / 1024).toFixed(0)} KB`
}

/**
 * The slide drop zone, shown at step 1 until a slide is loaded.
 *
 * The transfer is chunked: a whole-slide image is several gigabytes, so it goes
 * up in parts that can each be retried, and the server reassembles it and
 * proves it opens before reporting the upload ready. Once that happens the
 * walkthrough takes over and this panel is replaced.
 */
export function UploadPanel({ capability }: { capability: UploadCapability }) {
  const { stage, progress, phase, filename, error, upload } = useSlideSession()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const busy = stage === 'uploading'
  const accept = capability.acceptedFormats.join(',')

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0]
      if (file && !busy) void upload(file)
    },
    [busy, upload],
  )

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault()
      setDragging(false)
      handleFiles(event.dataTransfer.files)
    },
    [handleFiles],
  )

  // --- transfer in flight ---------------------------------------------------
  if (busy) {
    return (
      <div className="dropzone dropzone--busy" aria-live="polite">
        <span className="dropzone__icon">
          <SlideIcon />
        </span>
        <span className="dropzone__title">{filename}</span>
        <p className="dropzone__body">{PHASE_COPY[phase ?? 'transferring']}</p>

        <div className="progress" role="progressbar" aria-valuenow={Math.round(progress * 100)}>
          <div
            className={cn(
              'progress__fill',
              phase !== 'transferring' && 'progress__fill--indeterminate',
            )}
            style={phase === 'transferring' ? { transform: `scaleX(${progress})` } : undefined}
          />
        </div>
        {phase === 'transferring' && (
          <span className="dropzone__percent mono">{Math.round(progress * 100)} %</span>
        )}
      </div>
    )
  }

  // --- idle / error ---------------------------------------------------------
  return (
    <div
      className={cn(
        'dropzone',
        dragging && 'dropzone--dragging',
        stage === 'error' && 'dropzone--error',
      )}
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="visually-hidden"
        onChange={(event) => handleFiles(event.target.files)}
      />

      <span className="dropzone__icon">
        <SlideIcon />
      </span>

      <span className="dropzone__title">
        {dragging ? 'Drop to upload' : 'Upload a whole-slide image'}
      </span>

      {stage === 'error' ? (
        <p className="dropzone__body dropzone__body--error">{error}</p>
      ) : (
        <p className="dropzone__body">
          Drag a slide here, or choose one. It uploads in {formatBytes(capability.chunkSizeBytes)}{' '}
          chunks and is verified server-side before this step reads it.
        </p>
      )}

      <Button onClick={() => inputRef.current?.click()} attention={stage !== 'error'}>
        {stage === 'error' ? 'Try another slide' : 'Choose a slide'}
      </Button>

      <span className="dropzone__formats">
        {capability.acceptedFormats.map((format) => (
          <span key={format} className="dropzone__format">
            {format}
          </span>
        ))}
        <span className="dropzone__format">up to {capability.maxFileSizeMb} MB</span>
      </span>
    </div>
  )
}
