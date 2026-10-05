/**
 * Owns step 2's state: what the server can do, what a run is doing, what it found.
 *
 * The step is unusual in this app because it is the first one whose work does
 * not fit in a request. So this hook holds three things at once - a capability
 * report, a live run, and a finished report - and the view picks which of them
 * to show.
 *
 * A report already cached on the server is fetched on mount, so stepping back
 * onto step 2 shows the previous result immediately instead of re-running
 * minutes of inference.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { ApiError, CancelledError } from '@/api/client'
import {
  cancelQCRun,
  fetchQCCapability,
  fetchQCReport,
  runQualityControl,
} from '@/api/qc'
import type { QCCapability, QCReport, QCRun } from '@/types/qc'

export interface QualityControl {
  capability: QCCapability | null
  /** True until the capability probe has answered - it imports torch, so it is slow. */
  probing: boolean
  report: QCReport | null
  run: QCRun | null
  busy: boolean
  error: string | null
  /** Which artefact model to use, by training resolution. Null means the default. */
  modelMpp: number | null
  setModelMpp: (mpp: number | null) => void
  /** Run step 2. Rejects on failure so the pipeline can show the step as errored. */
  start: () => Promise<void>
  /**
   * Throw the cached run away and run again.
   *
   * The one thing `start` cannot do: it returns the server's cache when the settings
   * match, which is what makes revisiting the step instant and is exactly wrong when
   * the viewer is asking for a fresh run.
   */
  restart: () => Promise<void>
  /**
   * Ask the running pass to stop.
   *
   * A real server-side stop, not an abandoned wait - the run ends rather than
   * carrying on unwatched with every core busy. `start` then rejects with a
   * `CancelledError`, which the pipeline treats as "back to ready" rather than as a
   * failure, because it is a decision and not a fault.
   */
  cancel: () => Promise<void>
  /** True while a stop has been asked for but the run has not yet acknowledged it. */
  cancelling: boolean
  /** Re-probe the server. The capability call is the one that can time out. */
  retry: () => void
  reset: () => void
}

export function useQualityControl(uploadId: string | null): QualityControl {
  const [capability, setCapability] = useState<QCCapability | null>(null)
  const [probing, setProbing] = useState(true)
  const [report, setReport] = useState<QCReport | null>(null)
  const [run, setRun] = useState<QCRun | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modelMpp, setModelMpp] = useState<number | null>(null)
  const [cancelling, setCancelling] = useState(false)

  // Guards a late response from a run the user has already navigated away from.
  const active = useRef(true)
  useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])

  // Capability is a property of the server, not of the slide, so it is fetched
  // once rather than per upload. `attempt` lets the user ask again after a
  // failure without reloading the page.
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setProbing(true)

    fetchQCCapability(controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return
        setCapability(result)
        setModelMpp((current) => current ?? result.defaultModelMpp)
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setError(
          cause instanceof Error
            ? `could not reach the QC service: ${cause.message}`
            : 'could not reach the QC service',
        )
      })
      .finally(() => {
        if (!controller.signal.aborted) setProbing(false)
      })

    return () => controller.abort()
  }, [attempt])

  // Pick up a report the server already holds for this slide.
  useEffect(() => {
    setReport(null)
    setRun(null)
    setError(null)
    if (!uploadId) return

    const controller = new AbortController()

    fetchQCReport(uploadId, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setReport(result)
      })
      .catch((cause: unknown) => {
        // A 409 here is the normal case: this slide has simply not been run yet.
        if (cause instanceof ApiError && cause.status === 409) return
      })

    return () => controller.abort()
  }, [uploadId])

  const launch = useCallback(
    async (restart: boolean) => {
      if (!uploadId) throw new Error('no slide loaded')

      setBusy(true)
      setCancelling(false)
      setError(null)
      setRun(null)
      if (restart) setReport(null)

      try {
        const result = await runQualityControl(uploadId, {
          modelMpp,
          restart,
          onProgress: (next) => {
            if (active.current) setRun(next)
          },
        })
        if (active.current) setReport(result)
      } catch (cause) {
        // A cancellation is not an error and must not be recorded as one, or the
        // screen apologises for what the viewer just asked for. It is re-thrown
        // unchanged so the pipeline can tell the two apart.
        if (cause instanceof CancelledError) {
          if (active.current) setError(null)
          throw cause
        }
        const message = cause instanceof Error ? cause.message : 'quality control failed'
        if (active.current) setError(message)
        throw new Error(message)
      } finally {
        if (active.current) {
          setBusy(false)
          setCancelling(false)
        }
      }
    },
    [modelMpp, uploadId],
  )

  const start = useCallback(() => launch(false), [launch])
  const restart = useCallback(() => launch(true), [launch])

  const cancel = useCallback(async () => {
    if (!uploadId) return
    setCancelling(true)
    try {
      const stopped = await cancelQCRun(uploadId)
      if (active.current) setRun(stopped)
    } catch {
      // Cancelling something that has already finished is not a failure worth
      // reporting - the outcome the viewer wanted is the state, and they will see it.
      if (active.current) setCancelling(false)
    }
  }, [uploadId])

  const retry = useCallback(() => {
    setError(null)
    setAttempt((count) => count + 1)
  }, [])

  const reset = useCallback(() => {
    setReport(null)
    setRun(null)
    setError(null)
  }, [])

  return useMemo(
    () => ({
      capability,
      probing,
      report,
      run,
      busy,
      error,
      modelMpp,
      setModelMpp,
      start,
      restart,
      cancel,
      cancelling,
      retry,
      reset,
    }),
    [
      busy,
      cancel,
      cancelling,
      capability,
      error,
      modelMpp,
      probing,
      report,
      reset,
      restart,
      retry,
      run,
      start,
    ],
  )
}
