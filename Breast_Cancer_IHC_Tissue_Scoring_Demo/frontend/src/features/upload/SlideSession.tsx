/**
 * The slide currently loaded into the app.
 *
 * One slide at a time, shared between the home page (which uploads it) and the
 * walkthrough (which reads it in step 1). The upload id is kept in
 * `sessionStorage` so a page reload does not lose a multi-gigabyte transfer
 * that already finished.
 *
 * Note the split: uploading only proves the file arrived intact and can be
 * opened. Actually reading the pyramid is step 1's job, and it happens when the
 * user runs that step - not here.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'

import { fetchSlideReadout, fetchUploadStatus, uploadSlide } from '@/api/uploads'
import type { UploadPhase } from '@/api/uploads'
import type { SlideReadout, UploadStatus } from '@/types/slide'

import { SlideSessionContext } from './slideSessionContext'
import { useSlideLifetime } from './useSlideLifetime'
import type { SlideSessionValue, SlideStage } from './slideSessionContext'

const STORAGE_KEY = 'ihc.uploadId'

function readStored(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEY)
  } catch {
    return null // private mode, or storage disabled
  }
}

function writeStored(uploadId: string | null): void {
  try {
    if (uploadId) sessionStorage.setItem(STORAGE_KEY, uploadId)
    else sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    /* storage unavailable; the session simply will not survive a reload */
  }
}

export function SlideSessionProvider({ children }: { children: ReactNode }) {
  const [stage, setStage] = useState<SlideStage>('empty')
  const [uploadId, setUploadId] = useState<string | null>(null)
  const [status, setStatus] = useState<UploadStatus | null>(null)
  const [readout, setReadout] = useState<SlideReadout | null>(null)
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState<UploadPhase | null>(null)
  const [filename, setFilename] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [targetMpp, setTargetMpp] = useState<number | null>(null)
  const [mppOverride, setMppOverride] = useState<number | null>(null)
  const [adjusting, setAdjusting] = useState(false)

  // Recover a slide uploaded before a reload. If the server no longer holds it,
  // fall back to empty rather than showing a slide that is not there.
  useEffect(() => {
    const stored = readStored()
    if (!stored) return

    const controller = new AbortController()

    fetchUploadStatus(stored, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return
        if (result.state !== 'ready') throw new Error('not ready')
        setUploadId(stored)
        setStatus(result)
        setFilename(result.filename)
        setStage('loaded')
      })
      .catch(() => {
        if (controller.signal.aborted) return
        writeStored(null)
      })

    return () => controller.abort()
  }, [])

  const upload = useCallback(async (file: File) => {
    setStage('uploading')
    setProgress(0)
    setPhase('hashing')
    setFilename(file.name)
    setError(null)
    setReadout(null)
    setStatus(null)

    try {
      const id = await uploadSlide(file, {
        onProgress: (fraction, current) => {
          setProgress(fraction)
          setPhase(current)
        },
      })

      setUploadId(id)
      writeStored(id)
      setStatus(await fetchUploadStatus(id))
      setStage('loaded')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'upload failed')
      setStage('error')
    } finally {
      setPhase(null)
    }
  }, [])

  const readSlide = useCallback(async () => {
    if (!uploadId) throw new Error('no slide loaded')
    setReadout(await fetchSlideReadout(uploadId, { targetMpp, mppOverride }))
  }, [mppOverride, targetMpp, uploadId])

  /**
   * Re-read the slide at a different resolution.
   *
   * The pyramid is fixed, but which level the pipeline works at is not - it
   * falls out of the target, so changing the target has to go back to the
   * server rather than being recomputed in the browser from a stale readout.
   */
  const adjust = useCallback(
    async (next: { targetMpp?: number | null; mppOverride?: number | null }) => {
      if (!uploadId) return

      const nextTarget = next.targetMpp !== undefined ? next.targetMpp : targetMpp
      const nextOverride = next.mppOverride !== undefined ? next.mppOverride : mppOverride

      setTargetMpp(nextTarget)
      setMppOverride(nextOverride)
      setAdjusting(true)
      try {
        setReadout(
          await fetchSlideReadout(uploadId, {
            targetMpp: nextTarget,
            mppOverride: nextOverride,
          }),
        )
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : 'could not re-read the slide')
      } finally {
        setAdjusting(false)
      }
    },
    [mppOverride, targetMpp, uploadId],
  )

  // Claim this slide while the page is open, and tell the server when it goes away.
  // Does nothing unless the server has the feature on, which it does not by default -
  // see the hook for why the guess is only safe because it is recoverable.
  useSlideLifetime(uploadId)

  const clear = useCallback(() => {
    writeStored(null)
    setStage('empty')
    setUploadId(null)
    setStatus(null)
    setReadout(null)
    setProgress(0)
    setPhase(null)
    setFilename(null)
    setError(null)
    setTargetMpp(null)
    setMppOverride(null)
  }, [])

  const value: SlideSessionValue = useMemo(
    () => ({
      stage,
      uploadId,
      status,
      readout,
      progress,
      phase,
      filename,
      error,
      upload,
      readSlide,
      targetMpp,
      mppOverride,
      adjust,
      adjusting,
      clear,
    }),
    [
      stage,
      uploadId,
      status,
      readout,
      progress,
      phase,
      filename,
      error,
      upload,
      readSlide,
      targetMpp,
      mppOverride,
      adjust,
      adjusting,
      clear,
    ],
  )

  return <SlideSessionContext.Provider value={value}>{children}</SlideSessionContext.Provider>
}

