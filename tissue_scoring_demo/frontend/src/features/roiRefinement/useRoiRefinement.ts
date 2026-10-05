/**
 * Owns step 11's state: the pass, and every region's own result as it lands.
 *
 * Job-shaped like step 8's hook — start, poll, read — with one difference that shapes
 * the whole thing: **the report is worth re-reading on every poll**, not only at the
 * end. Each region writes its result the moment it finishes, so polling fetches the
 * position *and* the report, and a finished region is on screen while the next one is
 * still going. That is what "completed regions appear immediately" means in practice.
 *
 * Retry is the same call with a narrower scope. `retry(roiId)` starts a pass over one
 * region; the server intersects it with step 10's selection, so this cannot reach a
 * region nobody ticked, and the regions that already succeeded are untouched.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  cancelRefinement,
  fetchRefinementReport,
  fetchRefinementRun,
  startRefinement,
} from '@/api/roiRefinement'
import type {
  RefinementReport,
  RefinementRun,
  RefinementState,
} from '@/types/roiRefinement'

/**
 * Two seconds, matching step 10's alignment poll. A region is minutes, so a faster poll
 * would only add requests; a slower one would leave a finished region unshown for
 * longer than it took to finish.
 */
const POLL_MS = 2000

export interface RoiRefinementState {
  report: RefinementReport | null
  run: RefinementRun | null
  state: RefinementState | null
  message: string | null
  running: boolean
  error: string | null

  /**
   * The region in progress, painting itself window by window.
   *
   * Held as refs rather than as state for the same reason step 8 holds its feed that
   * way: a pass is thousands of windows, and copying a growing array on every poll to
   * satisfy React's identity rule would be thousands of copies of an array the canvas
   * only ever appends to and never re-reads. So these keep the same array object
   * throughout a region, `paintedCount` is what changes, and the canvas draws the new
   * tail.
   *
   * **They are cleared whenever the pass moves to the next region**, because each
   * region is its own picture. `run.paint.roiId` says which region they belong to.
   */
  painted: number[]
  paintedMasks: string[]
  /** Windows in `painted`. The number that moves; the arrays' identity does not. */
  paintedCount: number

  start: () => Promise<void>
  /** Refine every selected region again, including the ones already finished. */
  restart: () => Promise<void>
  /** Run one region on its own, leaving the others alone. */
  retry: (roiId: string) => Promise<void>
  cancel: () => Promise<void>
  reset: () => void
}

export function useRoiRefinement(
  uploadId: string | null,
  /**
   * Step 10's stamp — its `generatedAt` plus who chose. Not read for its value, only
   * for when it changes: a different selection is a different mask, so the pass on
   * screen is no longer describing what is ticked.
   */
  selectionVersion: string | null = null,
): RoiRefinementState {
  const [report, setReport] = useState<RefinementReport | null>(null)
  const [run, setRun] = useState<RefinementRun | null>(null)
  const [state, setState] = useState<RefinementState | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // The paint feed for the region in progress. See the state contract for why these
  // are refs, and why they are per region rather than per pass.
  const painted = useRef<number[]>([])
  const paintedMasks = useRef<string[]>([])
  const [paintedCount, setPaintedCount] = useState(0)
  /** How far into the current region's log we have read. Reset with the arrays. */
  const cursor = useRef(0)
  /** Which region the arrays hold. `null` when nothing is being painted. */
  const paintedRoi = useRef<string | null>(null)

  const clearPaint = useCallback(() => {
    // Emptied in place, so the canvas keeps the array it was given and only sees the
    // count go backwards - which is exactly the signal it treats as "start again".
    painted.current.length = 0
    paintedMasks.current.length = 0
    cursor.current = 0
    paintedRoi.current = null
    setPaintedCount(0)
  }, [])

  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  const clear = useCallback(() => {
    setReport(null)
    setRun(null)
    setState(null)
    setMessage(null)
    setError(null)
    clearPaint()
  }, [clearPaint])

  useEffect(() => {
    clear()
  }, [clear, selectionVersion, uploadId])

  // Pick up a pass this slide already has, so arriving on the step shows the regions
  // that are already refined instead of offering to spend the minutes again.
  useEffect(() => {
    if (!uploadId) return
    const controller = new AbortController()
    fetchRefinementReport(uploadId, controller.signal)
      .then((existing) => {
        if (controller.signal.aborted) return
        setReport(existing)
        setState(existing.state)
      })
      .catch(() => {
        /* nothing refined yet; the panel offers to run it */
      })
    return () => controller.abort()
  }, [selectionVersion, uploadId])

  const poll = useCallback(async () => {
    if (!uploadId) return

    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, POLL_MS))
      if (!live.current) return

      let position = await fetchRefinementRun(uploadId, undefined, cursor.current)
      if (!live.current) return

      // The pass moves from region to region, and each one is its own canvas. The
      // cursor we just sent belongs to the region we *had*, so when the answer says the
      // paint is now a different region that cursor is meaningless against its log -
      // and worse than meaningless: the server would have sliced the new region's
      // windows from that offset and quietly skipped everything before it. So the feed
      // is thrown away and asked for again from the start.
      //
      // One extra request per region, a dozen or so over a pass. `paint.roiId` is what
      // is compared and not the cursor, because a cursor going backwards is not a
      // reliable signal: a long region followed by a short one can produce a stale
      // cursor that is still in range.
      const roiId = position.paint?.roiId ?? null
      if (roiId !== paintedRoi.current) {
        clearPaint()
        paintedRoi.current = roiId
        if (roiId !== null) {
          position = await fetchRefinementRun(uploadId, undefined, 0)
          if (!live.current) return
          // A second handover inside one poll would mean regions finishing faster than
          // the poll interval, which BEETLE cannot do. If it happens anyway, the next
          // poll sees the mismatch and starts again.
          paintedRoi.current = position.paint?.roiId ?? null
        }
      }

      if (position.paintedCells.length > 0) {
        // The masks go first, so that by the time `paintedCount` moves - which is what
        // the canvas watches - every window it is about to draw already has its mask
        // beside it. The other order lets a render land between the two, find the mask
        // missing and paint that window as a flat colour it would never revisit.
        for (const mask of position.paintedMasks) paintedMasks.current.push(mask)
        for (const value of position.paintedCells) painted.current.push(value)
        setPaintedCount(painted.current.length / 3)
      }
      cursor.current = position.paintedCursor

      setRun(position)
      setState(position.state)
      setMessage(position.message)

      // The report, every time round, not only at the end. A region that finished
      // during this interval is already on disk, and the screen's whole promise is
      // that it appears now rather than when the last region does.
      try {
        const latest = await fetchRefinementReport(uploadId)
        if (live.current) setReport(latest)
      } catch {
        /* no report until the first region lands; the position alone is enough */
      }

      if (position.state === 'queued' || position.state === 'running') continue

      // The pass is over: the last region's overlay is on disk and the canvas should
      // give way to it rather than holding the final frame of the paint.
      clearPaint()

      if (position.state === 'failed') {
        setError(position.error ?? 'the refinement failed')
        return
      }
      return
    }
  }, [clearPaint, uploadId])

  const launch = useCallback(
    async (options: { roiIds?: string[]; rebuild?: boolean }) => {
      if (!uploadId) throw new Error('no slide loaded')

      setError(null)
      setState('queued')
      setMessage(options.roiIds ? `re-running ${options.roiIds.join(', ')}` : 'starting')
      // A pass paints from nothing. Cleared on every start rather than only on a fresh
      // slide, so a retry does not add its windows to the canvas of the run before it.
      clearPaint()

      try {
        const started = await startRefinement(uploadId, options)
        if (!live.current) return

        setRun(started)
        setState(started.state)

        // Everything asked for was already on disk: the server finalised the report
        // without starting a worker, and there is nothing to poll.
        if (started.state !== 'queued' && started.state !== 'running') {
          const finished = await fetchRefinementReport(uploadId)
          if (live.current) {
            setReport(finished)
            setState(finished.state)
          }
          return
        }

        await poll()
      } catch (cause) {
        const detail =
          cause instanceof Error ? cause.message : 'could not refine the regions'
        if (live.current) {
          setError(detail)
          setState('failed')
        }
        throw new Error(detail)
      }
    },
    [clearPaint, poll, uploadId],
  )

  const start = useCallback(() => launch({}), [launch])
  const restart = useCallback(() => launch({ rebuild: true }), [launch])
  const retry = useCallback(
    (roiId: string) => launch({ roiIds: [roiId], rebuild: true }),
    [launch],
  )

  const cancel = useCallback(async () => {
    if (!uploadId) return
    const stopped = await cancelRefinement(uploadId)
    if (live.current) {
      setRun(stopped)
      setMessage(stopped.message)
    }
  }, [uploadId])

  return useMemo(
    () => ({
      report,
      run,
      state,
      message,
      running: state === 'queued' || state === 'running',
      error,
      painted: painted.current,
      paintedMasks: paintedMasks.current,
      paintedCount,
      start,
      restart,
      retry,
      cancel,
      reset: clear,
    }),
    [
      cancel,
      clear,
      error,
      message,
      paintedCount,
      report,
      restart,
      retry,
      run,
      start,
      state,
    ],
  )
}
