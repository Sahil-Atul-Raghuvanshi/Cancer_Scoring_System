/**
 * Owns step 7's state, and its one control.
 *
 * Overlap is a *committed* control rather than a drafted one, which is the
 * difference from step 4's percentile slider. A percentile can be swept and
 * watched because every value in between is meaningful; overlap in practice is a
 * choice between a few settings — none, a quarter, a half — and the guide's band
 * is 25–50%. So it is three buttons and one request each, with the previous
 * report staying up, dimmed, until the new one lands.
 *
 * Its *input* is step 3's mask, so this hook takes step 3's committed cut as an
 * argument rather than owning a copy, exactly as steps 4, 5 and 6's hooks do.
 * When the viewer goes back and moves that slider, this step's result is stale:
 * a different mask is a different set of tiles.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchTiling } from '@/api/tiling'
import type { TilingReport } from '@/types/tiling'

/** The overlap settings the screen offers, in the order it offers them. */
export const OVERLAP_CHOICES = [0, 0.25, 0.5] as const

export interface TilingState {
  report: TilingReport | null
  /** True while the first run for this slide is in flight. */
  loading: boolean
  /** True while a re-run at a different overlap is in flight and a report is up. */
  refining: boolean
  error: string | null

  /** The overlap the report on screen was built at, or null before the first run. */
  overlap: number | null

  /** Rebuild the index at a different overlap. */
  setOverlap: (overlap: number) => void
  /** Run step 7. Rejects on failure so the pipeline can mark the step errored. */
  start: () => Promise<void>
  reset: () => void
}

export function useTiling(
  uploadId: string | null,
  /** Step 3's committed cut, or null to let its own rule choose. */
  tissueThreshold: number | null,
): TilingState {
  const [report, setReport] = useState<TilingReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [refining, setRefining] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [overlap, setOverlapValue] = useState<number | null>(null)

  // Guards a response arriving after the viewer has left the step.
  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  // Only the newest request may write to state. Without this, clicking through
  // the three overlaps quickly can leave whichever finishes last on screen, which
  // may not be the one the viewer asked for.
  const inFlight = useRef(0)

  const clear = useCallback(() => {
    setReport(null)
    setError(null)
    setOverlapValue(null)
    setRefining(false)
  }, [])

  useEffect(() => {
    clear()
  }, [clear, uploadId])

  // Step 3's cut changed, so the mask did, so every tile here did. Clearing
  // rather than quietly refetching: the counts on screen belong to the previous
  // mask, and leaving them up beside a new one is the mismatch every earlier
  // step's screen is built to avoid.
  useEffect(() => {
    clear()
  }, [clear, tissueThreshold])

  const start = useCallback(async () => {
    if (!uploadId) throw new Error('no slide loaded')

    setLoading(true)
    setError(null)

    try {
      // No overlap named on the first run, so the report describes the setting
      // the pipeline would really use rather than one this screen picked.
      const result = await fetchTiling(uploadId, { threshold: tissueThreshold })
      if (!live.current) return

      setReport(result)
      setOverlapValue(result.params.overlap)
    } catch (cause) {
      const message =
        cause instanceof Error ? cause.message : 'could not build a tile index'
      if (live.current) setError(message)
      throw new Error(message)
    } finally {
      if (live.current) setLoading(false)
    }
  }, [tissueThreshold, uploadId])

  const setOverlap = useCallback(
    (next: number) => {
      if (!uploadId) return

      const ticket = ++inFlight.current
      setOverlapValue(next)
      setRefining(true)

      fetchTiling(uploadId, { threshold: tissueThreshold, overlap: next })
        .then((result) => {
          if (!live.current || ticket !== inFlight.current) return
          setReport(result)
          // Snap onto what the server actually used, so the screen describes the
          // run that happened rather than the one that was asked for.
          setOverlapValue(result.params.overlap)
          setError(null)
        })
        .catch((cause: unknown) => {
          if (!live.current || ticket !== inFlight.current) return
          setError(
            cause instanceof Error
              ? `could not rebuild the index: ${cause.message}`
              : 'could not rebuild the index',
          )
        })
        .finally(() => {
          if (live.current && ticket === inFlight.current) setRefining(false)
        })
    },
    [tissueThreshold, uploadId],
  )

  return useMemo(
    () => ({
      report,
      loading,
      refining,
      error,
      overlap,
      setOverlap,
      start,
      reset: clear,
    }),
    [clear, error, loading, overlap, refining, report, setOverlap, start],
  )
}
