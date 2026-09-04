/**
 * Owns step 8's state: the capability probe, the long pass, and the class filter.
 *
 * Job-shaped like step 2's hook rather than request-shaped like step 7's, and for a
 * stronger reason: a whole-slide pass is tens of minutes, so it has to be started,
 * polled and cached. What the server caches, this hook picks up for free — a pass
 * already on disk comes back `ready` from the first call, so stepping back onto the
 * screen is instant instead of another half hour.
 *
 * Its *input* is step 7's index, so the hook takes step 3's committed cut as an
 * argument exactly as steps 4 to 7's hooks do. When the viewer goes back and moves
 * that slider the class map on screen belongs to a different set of tiles, so it is
 * cleared rather than left up beside numbers that no longer describe it.
 *
 * The class filter is deliberately *not* server state: it is a way of looking at a
 * finished pass, so toggling it changes an image URL and nothing else. That is what
 * makes "switch fat off and watch the denominator change" instant.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { CancelledError } from '@/api/client'
import {
  cancelTissueTypeRun,
  fetchTissueTypeCapability,
  fetchTissueTypeReport,
  fetchTissueTypeRun,
  runTissueType,
} from '@/api/tissueType'
import type {
  TissueTypeCapability,
  TissueTypeReport,
  TissueTypeRun,
} from '@/types/tissueType'

/** Every class, which is what the map shows until the viewer switches one off. */
export const ALL_CLASSES = [0, 1, 2] as const

export interface TissueTypeState {
  /** What the server can do, or null while probing. */
  capability: TissueTypeCapability | null
  probing: boolean
  /** Non-null when the capability probe itself failed — the backend is unreachable. */
  probeError: string | null
  retryProbe: () => void

  report: TissueTypeReport | null
  /** Live state of the pass, including its progress while it runs. */
  run: TissueTypeRun | null
  running: boolean
  error: string | null

  /** Which classes the map draws. A view of a finished pass, not a re-run. */
  visible: number[]
  toggleClass: (id: number) => void
  showAllClasses: () => void

  /** Run step 8. Rejects on failure so the pipeline can mark the step errored. */
  start: () => Promise<void>
  /**
   * Throw the cached pass away and run again.
   *
   * Matters more here than anywhere else in the pipeline: `start` returns the cache
   * when the parameters match, which is what makes revisiting this screen instant
   * rather than another half hour - and is exactly wrong when the viewer is asking
   * for a fresh pass.
   */
  restart: () => Promise<void>
  /**
   * Ask the running pass to stop.
   *
   * A real server-side stop rather than an abandoned wait, which matters at this
   * duration: an unwatched pass would keep every core busy for another half hour and
   * starve whatever the viewer moved on to. `start` then rejects with a
   * `CancelledError`, which the pipeline treats as "back to ready" rather than a
   * failure.
   */
  cancel: () => Promise<void>
  /** True while a stop has been asked for but the pass has not yet acknowledged it. */
  cancelling: boolean
  reset: () => void
}

export function useTissueType(
  uploadId: string | null,
  /** Step 3's committed cut, or null to let its own rule choose. */
  tissueThreshold: number | null,
  /** Step 7's committed overlap, or null before it has run - see `launch`. */
  tileOverlap: number | null,
): TissueTypeState {
  const [capability, setCapability] = useState<TissueTypeCapability | null>(null)
  const [probing, setProbing] = useState(true)
  const [probeError, setProbeError] = useState<string | null>(null)
  const [probeAttempt, setProbeAttempt] = useState(0)

  const [report, setReport] = useState<TissueTypeReport | null>(null)
  const [run, setRun] = useState<TissueTypeRun | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [visible, setVisible] = useState<number[]>([...ALL_CLASSES])
  const [cancelling, setCancelling] = useState(false)

  // Guards a response arriving after the viewer has left the step.
  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  /* --- can it run at all -------------------------------------------------- */

  useEffect(() => {
    const controller = new AbortController()
    setProbing(true)
    setProbeError(null)

    fetchTissueTypeCapability(controller.signal)
      .then((result) => {
        if (!live.current) return
        setCapability(result)
      })
      .catch((cause: unknown) => {
        if (!live.current || controller.signal.aborted) return
        setCapability(null)
        setProbeError(
          cause instanceof Error
            ? cause.message
            : 'could not ask the server whether step 8 can run',
        )
      })
      .finally(() => {
        if (live.current) setProbing(false)
      })

    return () => controller.abort()
  }, [probeAttempt])

  const retryProbe = useCallback(() => setProbeAttempt((value) => value + 1), [])

  /* --- what is already on the server -------------------------------------- */

  const clear = useCallback(() => {
    setReport(null)
    setRun(null)
    setError(null)
    setRunning(false)
    setCancelling(false)
    setVisible([...ALL_CLASSES])
  }, [])

  useEffect(() => {
    clear()
  }, [clear, uploadId])

  // Step 3's cut changed, so the mask did, so step 7's tiles did, so every window
  // here did. Clearing rather than quietly refetching: the counts on screen belong
  // to the previous mask, and leaving them up beside a new one is the mismatch
  // every earlier step's screen is built to avoid.
  useEffect(() => {
    clear()
  }, [clear, tissueThreshold])

  // And the same for step 7's overlap, which this step now runs at: a different
  // overlap is a different set of windows and therefore a different class map, so
  // the one on screen belongs to the previous choice.
  useEffect(() => {
    clear()
  }, [clear, tileOverlap])

  // Pick up a pass this browser did not start - one cached from an earlier session,
  // or still running from a reload. Half an hour of someone else's compute is worth
  // finding rather than repeating.
  useEffect(() => {
    if (!uploadId) return

    const controller = new AbortController()
    fetchTissueTypeRun(uploadId, controller.signal)
      .then((state) => {
        if (!live.current || controller.signal.aborted) return
        if (state.state !== 'ready') return

        setRun(state)
        return fetchTissueTypeReport(uploadId, controller.signal).then((cached) => {
          if (!live.current || controller.signal.aborted) return
          setReport(cached)
        })
      })
      .catch(() => {
        // Nothing cached, or the slide is not ready. Both are ordinary states and
        // the screen says so on its own; a probe failure is not an error here.
      })

    return () => controller.abort()
  }, [uploadId])

  /* --- running ------------------------------------------------------------ */

  const launch = useCallback(
    async (restart: boolean) => {
      if (!uploadId) throw new Error('no slide loaded')

      setRunning(true)
      setCancelling(false)
      setError(null)
      if (restart) setReport(null)

      try {
        // Step 7's overlap, and no model named. The overlap is the one parameter
        // the two grids share, so a viewer who picked "no overlap" one screen back
        // gets it here too rather than watching four times the windows go past. Null
        // before step 7 has run, in which case the server falls back to step 7's own
        // default - the value that screen would have used anyway.
        const result = await runTissueType(uploadId, {
          overlap: tileOverlap,
          restart,
          onProgress: (state) => {
            if (live.current) setRun(state)
          },
        })
        if (!live.current) return

        setReport(result)
        setRun(result.run)
      } catch (cause) {
        // A cancellation is not an error and must not be recorded as one. It is
        // re-thrown unchanged so the pipeline can tell the two apart.
        if (cause instanceof CancelledError) {
          if (live.current) setError(null)
          throw cause
        }
        const message =
          cause instanceof Error ? cause.message : 'could not classify the tissue'
        if (live.current) setError(message)
        throw new Error(message)
      } finally {
        if (live.current) {
          setRunning(false)
          setCancelling(false)
        }
      }
    },
    [uploadId, tileOverlap],
  )

  const start = useCallback(() => launch(false), [launch])
  const restart = useCallback(() => launch(true), [launch])

  const cancel = useCallback(async () => {
    if (!uploadId) return
    setCancelling(true)
    try {
      const stopped = await cancelTissueTypeRun(uploadId)
      if (live.current) setRun(stopped)
    } catch {
      // Cancelling something that has already finished is not a failure worth
      // reporting - the outcome the viewer wanted is the state, and they will see it.
      if (live.current) setCancelling(false)
    }
  }, [uploadId])

  /* --- looking at the result ---------------------------------------------- */

  const toggleClass = useCallback((id: number) => {
    setVisible((current) =>
      current.includes(id)
        ? current.filter((entry) => entry !== id)
        : [...current, id].sort(),
    )
  }, [])

  const showAllClasses = useCallback(() => setVisible([...ALL_CLASSES]), [])

  return useMemo(
    () => ({
      capability,
      probing,
      probeError,
      retryProbe,
      report,
      run,
      running,
      error,
      visible,
      toggleClass,
      showAllClasses,
      start,
      restart,
      cancel,
      cancelling,
      reset: clear,
    }),
    [
      cancel,
      cancelling,
      capability,
      clear,
      error,
      probeError,
      probing,
      report,
      restart,
      retryProbe,
      run,
      running,
      showAllClasses,
      start,
      toggleClass,
      visible,
    ],
  )
}
