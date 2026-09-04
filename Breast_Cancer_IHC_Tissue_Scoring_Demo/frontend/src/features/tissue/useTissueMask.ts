/**
 * Owns step 3's state, and one piece of choreography that matters.
 *
 * The threshold slider has two values, not one:
 *
 *   `draft`     where the viewer's finger is. Changes on every pixel of drag,
 *               and the histogram redraws from it instantly, because the
 *               histogram is already in the browser and moving a line over it
 *               costs nothing.
 *   `committed` what the server has actually run. Follows `draft` after a pause.
 *
 * Keeping them apart is what makes the slider feel live without lying. The line
 * on the histogram tracks the finger; the mask, the areas and the component
 * counts only ever show a cut the server genuinely computed. The alternative -
 * one value, refetched per drag event - would either stutter or, worse, leave
 * numbers on screen that belong to a threshold the viewer has already moved past.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchTissueReport } from '@/api/tissue'
import type { TissueReport, TissueThreshold } from '@/types/tissue'

/**
 * Where the automatic rule puts the cut.
 *
 * Not always Otsu: on a spike-and-tail histogram the server uses Zack's
 * triangle rule instead, and both answers are in every report. Reading this off
 * `bimodal` rather than off `source` means it still gives the automatic answer
 * while a manual run is on screen.
 */
function automaticCut(threshold: TissueThreshold): number {
  return threshold.bimodal ? threshold.otsu : threshold.triangle
}

/**
 * How long the slider must be still before the server is asked.
 *
 * A re-threshold is about two seconds of genuine morphology, so firing per drag
 * event would queue work the viewer has already superseded. 350 ms is past the
 * gap between drag events and under the point where a deliberate move feels
 * ignored.
 */
const COMMIT_DELAY_MS = 350

export interface TissueMaskState {
  report: TissueReport | null
  /** True while the first run for this slide is in flight. */
  loading: boolean
  /** True while a re-threshold is in flight and a previous report is on screen. */
  refining: boolean
  error: string | null

  /** Where the slider is now. Null until the first report names a cut. */
  draft: number | null
  /** The cut the report on screen was computed at. */
  committed: number | null
  /** True once the viewer has moved off the automatic rule's answer. */
  manual: boolean

  setDraft: (value: number) => void
  /** Hand the cut back to whichever rule the histogram's shape selects. */
  resetToAutomatic: () => void
  /** Run step 3. Rejects on failure so the pipeline can mark the step errored. */
  start: () => Promise<void>
  reset: () => void
}

export function useTissueMask(uploadId: string | null): TissueMaskState {
  const [report, setReport] = useState<TissueReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [refining, setRefining] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [draft, setDraftValue] = useState<number | null>(null)
  const [committed, setCommitted] = useState<number | null>(null)
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
    setManual(false)
    setRefining(false)
  }, [])

  useEffect(() => {
    clear()
  }, [clear, uploadId])

  /** The first run: the histogram's shape picks the rule, and it seeds the slider. */
  const start = useCallback(async () => {
    if (!uploadId) throw new Error('no slide loaded')

    setLoading(true)
    setError(null)

    try {
      const result = await fetchTissueReport(uploadId)
      if (!live.current) return

      setReport(result)
      setDraftValue(result.threshold.value)
      setCommitted(result.threshold.value)
      setManual(false)
    } catch (cause) {
      const message =
        cause instanceof Error ? cause.message : 'could not build the tissue mask'
      if (live.current) setError(message)
      throw new Error(message)
    } finally {
      if (live.current) setLoading(false)
    }
  }, [uploadId])

  const setDraft = useCallback((value: number) => {
    setDraftValue(Math.max(0, Math.min(255, Math.round(value))))
    setManual(true)
  }, [])

  const resetToAutomatic = useCallback(() => {
    if (!report) return
    setDraftValue(automaticCut(report.threshold))
    setManual(false)
  }, [report])

  // Commit the draft once the slider has been still. `manual` is in the guard
  // rather than the effect body so returning to the automatic value re-runs
  // *without* a threshold parameter - which is what makes the report come back
  // labelled by its rule again instead of as a manual run that happens to agree.
  useEffect(() => {
    if (!uploadId || draft === null || report === null) return
    if (draft === committed && manual === (report.threshold.source === 'manual')) return

    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setRefining(true)

      fetchTissueReport(uploadId, { threshold: manual ? draft : null }, controller.signal)
        .then((result) => {
          if (controller.signal.aborted || !live.current) return
          setReport(result)
          setCommitted(result.threshold.value)
          // Snap the slider onto whatever an automatic run actually chose.
          // Without this the loop never converges: handing control back to the
          // rule leaves `draft` on the value the viewer released at while
          // `committed` becomes the rule's, and the effect refires for ever.
          if (!manual) setDraftValue(result.threshold.value)
          setError(null)
        })
        .catch((cause: unknown) => {
          if (controller.signal.aborted || !live.current) return
          setError(
            cause instanceof Error
              ? `could not re-threshold: ${cause.message}`
              : 'could not re-threshold',
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
  }, [committed, draft, manual, report, uploadId])

  return useMemo(
    () => ({
      report,
      loading,
      refining,
      error,
      draft,
      committed,
      manual,
      setDraft,
      resetToAutomatic,
      start,
      reset: clear,
    }),
    [
      clear,
      committed,
      draft,
      error,
      loading,
      manual,
      refining,
      report,
      resetToAutomatic,
      setDraft,
      start,
    ],
  )
}
