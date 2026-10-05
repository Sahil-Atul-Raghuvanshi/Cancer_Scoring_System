/**
 * Owns step 9's state: the region, its borders, and the top-3 crops and export that
 * come with it.
 *
 * Request-shaped like step 3's hook rather than job-shaped like step 8's: `buildRoi`
 * is one call that returns instantly on a slide already built, because the server
 * itself decides whether its cache still answers the question being asked. There is
 * nothing here to poll and nothing to cancel.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { buildRoi } from '@/api/roi'
import type { RoiReport } from '@/types/roi'

export interface RoiState {
  report: RoiReport | null
  /** True while the first build for this slide is in flight. */
  loading: boolean
  error: string | null

  /** Build the region, or fetch the cached one. Rejects on failure — see the caller. */
  start: () => Promise<void>
  /** Throw the cached region away and build again, borders and crops included. */
  restart: () => Promise<void>
  reset: () => void
}

export function useRoi(
  uploadId: string | null,
  /**
   * Step 8's own stamp on the pass this region would be built from — its
   * `generatedAt`, or null before a class map exists. Not read for its value, only
   * for when it changes: a fresh run makes step 9's server-side cache stale too (the
   * server discards it itself, per `roi_service.discard`), and this hook has no other
   * way to notice that its own report is now describing labels that no longer exist.
   */
  classMapVersion: string | null = null,
): RoiState {
  const [report, setReport] = useState<RoiReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

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
    setLoading(false)
  }, [])

  useEffect(() => {
    clear()
  }, [clear, uploadId, classMapVersion])

  const launch = useCallback(
    async (rebuild: boolean) => {
      if (!uploadId) throw new Error('no slide loaded')

      setLoading(true)
      setError(null)

      try {
        const result = await buildRoi(uploadId, { rebuild })
        if (!live.current) return
        setReport(result)
      } catch (cause) {
        const message =
          cause instanceof Error ? cause.message : 'could not build the ROI'
        if (live.current) setError(message)
        throw new Error(message)
      } finally {
        if (live.current) setLoading(false)
      }
    },
    [uploadId],
  )

  const start = useCallback(() => launch(false), [launch])
  const restart = useCallback(() => launch(true), [launch])

  return useMemo(
    () => ({
      report,
      loading,
      error,
      start,
      restart,
      reset: clear,
    }),
    [clear, error, loading, report, restart, start],
  )
}
