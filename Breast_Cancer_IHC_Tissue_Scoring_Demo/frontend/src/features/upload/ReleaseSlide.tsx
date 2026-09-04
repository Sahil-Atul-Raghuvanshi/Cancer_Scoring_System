/**
 * "Use a different slide", with the disk cost of the one being left behind.
 *
 * **Why this hangs off *this* button and not off the pipeline reset.** Resetting the
 * walkthrough means "walk the same slide again" - deleting its results there would throw
 * away half an hour of step 8 to save nineteen megabytes, which is the worst trade in the
 * app. Choosing a *different* slide is the one moment the viewer has actually said they
 * are finished with this one, so it is the only honest place to offer the deletion.
 *
 * **And it offers the deletion that matters.** A slide is 0.8-1.6 GB; everything derived
 * from it is about 19 MB. A "clear cached results" button would free one per cent and
 * cost the expensive part, so the default offer here is the slide itself, with the cache
 * as the smaller, separate option for anyone who wants to keep the upload.
 *
 * Nothing is deleted without the number being on screen first: the panel asks the server
 * what this slide occupies before it offers to remove it.
 */

import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/Button'
import { cleanup, fetchStorage, formatBytes } from '@/api/maintenance'
import type { UploadStorage } from '@/api/maintenance'

import './upload.css'

interface ReleaseSlideProps {
  uploadId: string | null
  filename: string | null
  /** Forget the slide in this browser. Always runs, whatever the viewer chooses. */
  onRelease: () => void
}

export function ReleaseSlide({ uploadId, filename, onRelease }: ReleaseSlideProps) {
  const [asking, setAsking] = useState(false)
  const [usage, setUsage] = useState<UploadStorage | null>(null)
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)

  // Measured only once the viewer has asked, so the walkthrough never pays for a
  // directory walk it might not need.
  useEffect(() => {
    if (!asking || !uploadId) return

    const controller = new AbortController()
    fetchStorage(controller.signal)
      .then((report) => {
        if (controller.signal.aborted) return
        setUsage(report.uploads.find((entry) => entry.uploadId === uploadId) ?? null)
      })
      .catch(() => {
        // The offer still works without the numbers; it just cannot show them.
        if (!controller.signal.aborted) setUsage(null)
      })

    return () => controller.abort()
  }, [asking, uploadId])

  if (!asking) {
    return (
      <Button variant="ghost" onClick={() => setAsking(true)}>
        Use a different slide
      </Button>
    )
  }

  const release = async (scope: 'slide' | 'derived' | null) => {
    setBusy(true)
    setProblem(null)
    try {
      if (scope && uploadId) {
        await cleanup(scope, { uploadId, confirm: true })
      }
      onRelease()
    } catch (cause) {
      setProblem(cause instanceof Error ? cause.message : 'could not free the space')
      setBusy(false)
    }
  }

  const slide = usage?.slideBytes ?? 0
  const derived = usage?.derivedBytes ?? 0

  return (
    <div className="release">
      <p className="release__lead">
        Leaving <strong>{filename ?? 'this slide'}</strong> behind. It is still on the
        server
        {usage ? (
          <>
            {' '}
            using <strong>{formatBytes(slide + derived)}</strong> &mdash;{' '}
            {formatBytes(slide)} for the scan itself and {formatBytes(derived)} for
            everything worked out from it.
          </>
        ) : (
          '.'
        )}
      </p>

      <div className="release__choices">
        <Button variant="secondary" loading={busy} onClick={() => void release('slide')}>
          Delete it and free {usage ? formatBytes(slide + derived) : 'the space'}
        </Button>
        <Button variant="ghost" disabled={busy} onClick={() => void release(null)}>
          Keep it on the server
        </Button>
      </div>

      <p className="release__note">
        Deleting means uploading the file again to come back to it. Keeping it costs disk
        but nothing else &mdash; and the results already worked out stay valid, so
        returning to this slide is instant rather than another half hour.
        {derived > 0 && (
          <>
            {' '}
            <button
              type="button"
              className="release__link"
              disabled={busy}
              onClick={() => void release('derived')}
            >
              Keep the scan but drop the {formatBytes(derived)} of results
            </button>{' '}
            if you would rather re-run the steps than re-upload.
          </>
        )}
      </p>

      {problem && <p className="release__problem">{problem}</p>}
    </div>
  )
}
