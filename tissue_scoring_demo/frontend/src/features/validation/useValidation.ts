/**
 * Owns step 17's state. Two requests; no job, no slide, no model.
 *
 * The second request is the case's own score grid, and it belongs here rather than
 * on step 16. Step 16 reports ONE marker - the one whose slide is loaded - while this
 * step is the only one in the pipeline keyed on the case, and agreement is measured
 * over five markers of one block against four readers of the same block. Showing the
 * agreement statistics without the five numbers they are computed over left the
 * reader with a verdict and no way to see what it was a verdict on; the endpoint
 * existed and served exactly that grid, and nothing called it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchCaseScores } from '@/api/scores'
import { fetchValidation } from '@/api/validation'
import type { CaseScoreReport } from '@/types/scores'
import type { ValidationReport } from '@/types/validation'

export interface ValidationStateValue {
  report: ValidationReport | null
  /**
   * What this case scored, per marker. Null when no case is loaded - a lone slide
   * pair is not a case - or when the grid could not be read.
   *
   * Deliberately NOT allowed to fail the step: the agreement report is step 17's
   * answer, and this is the evidence beside it. A case whose grid 404s should still
   * show whether the numbers agreed with anyone.
   */
  caseScores: CaseScoreReport | null
  loading: boolean
  error: string | null
  start: () => Promise<void>
}

export function useValidation(
  caseId: string | null,
  /** Step 16's stamp: a new score is a new thing to compare. */
  scoreVersion: string | null = null,
): ValidationStateValue {
  const [report, setReport] = useState<ValidationReport | null>(null)
  const [caseScores, setCaseScores] = useState<CaseScoreReport | null>(null)
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
    if (!scoreVersion) return

    const controller = new AbortController()
    setLoading(true)
    fetchValidation(caseId, controller.signal)
      .then((result) => {
        if (controller.signal.aborted || !live.current) return
        setReport(result)
        setError(null)
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted || !live.current) return
        setError(cause instanceof Error ? cause.message : 'could not check agreement')
      })
      .finally(() => {
        if (live.current) setLoading(false)
      })

    return () => controller.abort()
  }, [caseId, nonce, scoreVersion])

  // The grid, alongside. Its own effect because it has its own failure mode and must
  // not take the agreement report down with it - see `caseScores` above.
  useEffect(() => {
    if (!scoreVersion || !caseId) {
      setCaseScores(null)
      return
    }

    const controller = new AbortController()
    fetchCaseScores(caseId, controller.signal)
      .then((result) => {
        if (controller.signal.aborted || !live.current) return
        setCaseScores(result)
      })
      .catch(() => {
        if (controller.signal.aborted || !live.current) return
        setCaseScores(null)
      })

    return () => controller.abort()
  }, [caseId, nonce, scoreVersion])

  const start = useCallback(async () => {
    setNonce((value) => value + 1)
  }, [])

  return useMemo(
    () => ({ report, caseScores, loading, error, start }),
    [caseScores, error, loading, report, start],
  )
}
