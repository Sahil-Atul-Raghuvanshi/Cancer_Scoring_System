/**
 * Owns step 4's state, and one piece of choreography that matters.
 *
 * Step 4 has no knob of its own. Its input is step 3's mask, so the thing that
 * moves I₀ is step 3's threshold - and this hook takes that as an argument rather
 * than owning a copy. When the viewer goes back to step 3 and drags the cut, this
 * step's result is stale and has to be re-derived, because a different mask is a
 * different set of glass pixels.
 *
 * The percentile *is* step 4's own control, and it is here for a different reason
 * from step 3's slider: not to be tuned, but to be swept. Dragging it and
 * watching I₀ not move is the argument that the 95th is a stable choice rather
 * than a fitted one. So the same draft/committed split step 3 uses applies, and
 * for the same reason - the ladder in the browser can redraw instantly while the
 * numbers on screen only ever describe a run the server really performed.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchCalibration } from '@/api/calibration'
import type { CalibrationReport } from '@/types/calibration'

/**
 * How long the percentile control must be still before the server is asked.
 *
 * Shorter than step 3's 350 ms: a re-run at a new percentile is a few numpy
 * passes over a cached thumbnail rather than a second and a half of morphology,
 * so waiting longer would only make it feel slower than it is.
 */
const COMMIT_DELAY_MS = 220

export interface WhiteCalibrationState {
  report: CalibrationReport | null
  /** True while the first run for this slide is in flight. */
  loading: boolean
  /** True while a re-run is in flight and a previous report is on screen. */
  refining: boolean
  error: string | null

  /** Where the percentile control is now. Null until the first report. */
  draft: number | null
  /** The percentile the report on screen was computed at. */
  committed: number | null
  /**
   * The percentile the server chooses on its own, held separately from the report.
   *
   * It has to be held separately, because a manual run comes back reporting the
   * percentile it was *asked* for - so reading the default off the current report
   * would make it whatever the viewer last dragged to, and "hand it back" would
   * hand it back to itself. Captured only from runs that had no percentile asked
   * of them, which is the one place the server's own answer appears.
   */
  defaultPercentile: number | null
  /** True once the viewer has moved off the server's default. */
  manual: boolean

  setDraft: (value: number) => void
  /** Hand the percentile back to the server's own default. */
  resetToDefault: () => void
  /** Run step 4. Rejects on failure so the pipeline can mark the step errored. */
  start: () => Promise<void>
  reset: () => void
}

export function useWhiteCalibration(
  uploadId: string | null,
  /**
   * Step 3's committed cut, or null to let its own rule choose.
   *
   * Passed in rather than read from step 4's own report, so that returning to
   * step 3 and moving the slider invalidates this step - which it must, because
   * a different mask is a different set of glass pixels.
   */
  tissueThreshold: number | null,
): WhiteCalibrationState {
  const [report, setReport] = useState<CalibrationReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [refining, setRefining] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [draft, setDraftValue] = useState<number | null>(null)
  const [committed, setCommitted] = useState<number | null>(null)
  const [defaultPercentile, setDefaultPercentile] = useState<number | null>(null)
  const [manual, setManual] = useState(false)

  // Guards a response arriving after the viewer has left the step.
  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  const clear = useCallback(() => {
    setReport(null)
    setError(null)
    setDraftValue(null)
    setCommitted(null)
    setDefaultPercentile(null)
    setManual(false)
    setRefining(false)
  }, [])

  useEffect(() => {
    clear()
  }, [clear, uploadId])

  // Step 3's cut changed, so this step's glass did too. Clearing rather than
  // silently refetching is deliberate: the numbers on screen belong to the
  // previous mask, and leaving them up beside a new one is exactly the kind of
  // quiet mismatch step 3's own screen is built to avoid.
  useEffect(() => {
    clear()
  }, [clear, tissueThreshold])

  const start = useCallback(async () => {
    if (!uploadId) throw new Error('no slide loaded')

    setLoading(true)
    setError(null)

    try {
      const result = await fetchCalibration(uploadId, { threshold: tissueThreshold })
      if (!live.current) return

      setReport(result)
      setDraftValue(result.white.percentile)
      setCommitted(result.white.percentile)
      // Nothing was asked of this run, so what it reports *is* the default.
      setDefaultPercentile(result.white.percentile)
      setManual(false)
    } catch (cause) {
      const message =
        cause instanceof Error ? cause.message : 'could not estimate the white point'
      if (live.current) setError(message)
      throw new Error(message)
    } finally {
      if (live.current) setLoading(false)
    }
  }, [tissueThreshold, uploadId])

  const setDraft = useCallback((value: number) => {
    // Below the 50th a "high percentile of the glass" stops being one, and above
    // the 100th there is nothing. Clamped here rather than at the server so the
    // control cannot show a value the run would refuse.
    setDraftValue(Math.max(50, Math.min(100, Math.round(value * 10) / 10)))
    setManual(true)
  }, [])

  const resetToDefault = useCallback(() => {
    if (defaultPercentile === null) return
    setDraftValue(defaultPercentile)
    setManual(false)
  }, [defaultPercentile])

  useEffect(() => {
    if (!uploadId || draft === null || report === null) return
    if (draft === committed) return

    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setRefining(true)

      fetchCalibration(
        uploadId,
        { threshold: tissueThreshold, percentile: manual ? draft : null },
        controller.signal,
      )
        .then((result) => {
          if (controller.signal.aborted || !live.current) return
          setReport(result)
          setCommitted(result.white.percentile)
          // Snap onto whatever the run actually used, so handing control back to
          // the default converges instead of refiring for ever - the same trap
          // step 3's threshold effect documents. Only a run with no percentile
          // asked of it may restate the default; a manual run echoes the request.
          if (!manual) {
            setDraftValue(result.white.percentile)
            setDefaultPercentile(result.white.percentile)
          }
          setError(null)
        })
        .catch((cause: unknown) => {
          if (controller.signal.aborted || !live.current) return
          setError(
            cause instanceof Error
              ? `could not re-sample the glass: ${cause.message}`
              : 'could not re-sample the glass',
          )
        })
        .finally(() => {
          if (!controller.signal.aborted && live.current) setRefining(false)
        })
    }, COMMIT_DELAY_MS)

    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [committed, draft, manual, report, tissueThreshold, uploadId])

  return useMemo(
    () => ({
      report,
      loading,
      refining,
      error,
      draft,
      committed,
      defaultPercentile,
      manual,
      setDraft,
      resetToDefault,
      start,
      reset: clear,
    }),
    [
      clear,
      committed,
      defaultPercentile,
      draft,
      error,
      loading,
      manual,
      refining,
      report,
      resetToDefault,
      setDraft,
      start,
    ],
  )
}
