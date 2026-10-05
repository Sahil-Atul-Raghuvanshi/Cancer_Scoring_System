/**
 * Owns step 16's state.
 *
 * Request-shaped like step 15's and for a stronger reason: this step contains no
 * image processing at all. Every number on it is a count or a mean over rows
 * step 14 stored, so it is arithmetic over a file that is already on disk.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchScore } from '@/api/scores'
import type { ScoreReport } from '@/types/scores'

export interface ScoresStateValue {
  report: ScoreReport | null
  loading: boolean
  error: string | null
  start: () => Promise<void>
}

export function useScores(
  heUploadId: string | null,
  ihcUploadId: string | null,
  /** Step 15's stamp: different cut points are a different score. */
  binningVersion: string | null = null,
): ScoresStateValue {
  const [report, setReport] = useState<ScoreReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)

  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  useEffect(() => {
    setReport(null)
    setError(null)
  }, [heUploadId, ihcUploadId, binningVersion])

  useEffect(() => {
    if (!heUploadId || !ihcUploadId || !binningVersion) return

    const controller = new AbortController()
    setLoading(true)
    fetchScore(heUploadId, ihcUploadId, controller.signal)
      .then((result) => {
        if (controller.signal.aborted || !live.current) return
        setReport(result)
        setError(null)
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted || !live.current) return
        setError(cause instanceof Error ? cause.message : 'could not compute the score')
      })
      .finally(() => {
        if (live.current) setLoading(false)
      })

    return () => controller.abort()
  }, [binningVersion, heUploadId, ihcUploadId, nonce])

  const start = useCallback(async () => {
    setNonce((value) => value + 1)
  }, [])

  return useMemo(
    () => ({ report, loading, error, start }),
    [error, loading, report, start],
  )
}
