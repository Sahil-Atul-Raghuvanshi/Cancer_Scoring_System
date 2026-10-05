/**
 * Owns step 10's state: the registration, its verdict, and the sign-off.
 *
 * Job-shaped like step 8's rather than request-shaped like step 9's, because
 * registering two whole-slide images takes minutes in another process. Start
 * it, poll it, then read the report.
 *
 * The one piece of state that is not the server's: `confirmed`. A registration
 * that passes every threshold is still not approved until a person says the
 * regions landed on the right tissue, so this hook exposes that as an action
 * rather than inferring it from the report's state.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  confirmAlignment,
  fetchAlignmentCapability,
  fetchAlignmentReport,
  fetchAlignmentRun,
  startAlignment,
} from '@/api/ihcAlignment'
import type {
  AlignmentCapability,
  AlignmentReport,
  AlignmentState,
} from '@/types/ihcAlignment'

const POLL_MS = 2000

export interface IhcAlignmentStateValue {
  capability: AlignmentCapability | null
  report: AlignmentReport | null
  state: AlignmentState | null
  message: string | null
  running: boolean
  error: string | null

  start: () => Promise<void>
  restart: () => Promise<void>
  confirm: (confirmed: boolean) => Promise<void>
  reset: () => void
}

export function useIhcAlignment(
  heUploadId: string | null,
  ihcUploadId: string | null,
  /** Step 9's stamp: a rebuilt ROI is different regions, so this alignment is stale. */
  roiVersion: string | null = null,
): IhcAlignmentStateValue {
  const [capability, setCapability] = useState<AlignmentCapability | null>(null)
  const [report, setReport] = useState<AlignmentReport | null>(null)
  const [state, setState] = useState<AlignmentState | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    fetchAlignmentCapability(controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setCapability(result)
      })
      .catch(() => {
        /* the panel reports this as "not installed" via a null capability */
      })
    return () => controller.abort()
  }, [])

  const clear = useCallback(() => {
    setReport(null)
    setState(null)
    setMessage(null)
    setError(null)
  }, [])

  useEffect(() => {
    clear()
  }, [clear, heUploadId, ihcUploadId, roiVersion])

  // Pick up an alignment this pair already has, so returning to the step does
  // not offer to re-run five minutes of work that is already on disk.
  useEffect(() => {
    if (!heUploadId || !ihcUploadId) return
    const controller = new AbortController()
    fetchAlignmentReport(heUploadId, ihcUploadId, controller.signal)
      .then((existing) => {
        if (controller.signal.aborted) return
        setReport(existing)
        setState(existing.state)
      })
      .catch(() => {
        /* nothing aligned yet; the panel offers to run it */
      })
    return () => controller.abort()
  }, [heUploadId, ihcUploadId, roiVersion])

  const poll = useCallback(async () => {
    if (!heUploadId || !ihcUploadId) return

    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, POLL_MS))
      if (!live.current) return

      const run = await fetchAlignmentRun(heUploadId, ihcUploadId)
      if (!live.current) return

      setState(run.state)
      setMessage(run.message)

      if (run.state === 'queued' || run.state === 'running') continue

      if (run.state === 'failed') {
        setError(run.error ?? 'the registration failed')
        return
      }

      // Both "ready" and "refused" have a report; a refusal is a result, and
      // its numbers are the whole explanation.
      const finished = await fetchAlignmentReport(heUploadId, ihcUploadId)
      setReport(finished)

      // A refusal must not leave the stepper reading "Completed". The report is
      // already stored above, so the panel still renders the refusal and its
      // numbers - this only stops the *pipeline* from counting the step as done.
      if (finished.state === 'refused') {
        throw new Error(finished.refusalReasons[0] ?? 'the registration was refused')
      }
      return
    }
  }, [heUploadId, ihcUploadId])

  const launch = useCallback(
    async (restart: boolean) => {
      if (!heUploadId || !ihcUploadId) {
        throw new Error('step 10 needs both the H&E and the IHC slide of one case')
      }

      setError(null)
      setState('queued')
      setMessage(restart ? 'registering again' : 'starting')

      try {
        const run = await startAlignment(heUploadId, ihcUploadId, { restart })
        if (!live.current) return

        setState(run.state)
        if (run.state === 'ready' || run.state === 'refused') {
          const finished = await fetchAlignmentReport(heUploadId, ihcUploadId)
          setReport(finished)
          if (finished.state === 'refused') {
            throw new Error(finished.refusalReasons[0] ?? 'the registration was refused')
          }
          return
        }
        await poll()
      } catch (cause) {
        const detail =
          cause instanceof Error ? cause.message : 'could not align the two slides'
        // A refusal already set `report`, and the panel renders that rather
        // than an error box - so the state stays as the server reported it.
        if (live.current) {
          setError(detail)
          setState((current) => (current === 'refused' ? current : 'failed'))
        }
        throw new Error(detail)
      }
    },
    [heUploadId, ihcUploadId, poll],
  )

  const start = useCallback(() => launch(false), [launch])
  const restart = useCallback(() => launch(true), [launch])

  const confirm = useCallback(
    async (confirmed: boolean) => {
      if (!heUploadId || !ihcUploadId) return
      const updated = await confirmAlignment(heUploadId, ihcUploadId, confirmed)
      if (live.current) setReport(updated)
    },
    [heUploadId, ihcUploadId],
  )

  return useMemo(
    () => ({
      capability,
      report,
      state,
      message,
      running: state === 'queued' || state === 'running',
      error,
      start,
      restart,
      confirm,
      reset: clear,
    }),
    [capability, clear, confirm, error, message, report, restart, start, state],
  )
}
