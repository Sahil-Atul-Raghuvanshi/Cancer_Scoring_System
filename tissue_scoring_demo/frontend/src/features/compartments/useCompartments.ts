/**
 * Owns step 13's state. Request-shaped like step 12's, for the same reason:
 * the backend reads label maps step 11 already stored and runs two distance
 * transforms per field, so a width change is a new request rather than a job.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchCompartmentGeometry, fetchCompartments } from '@/api/compartments'
import type { CompartmentsReport, RegionCompartmentRings } from '@/types/compartments'

const DEBOUNCE_MS = 300

export interface CompartmentsStateValue {
  report: CompartmentsReport | null
  /**
   * The outlines behind the report, keyed by region rank.
   *
   * Re-fetched whenever the report's stamp moves, so the shapes on the slide
   * are the shapes at the width currently on the slider. The report is what
   * says the server has finished writing them, which is why the fetch keys on
   * its stamp rather than on the width.
   */
  geometry: Record<number, RegionCompartmentRings>
  /** The width on screen, which leads the report while a drag settles. */
  widthUm: number | null
  loading: boolean
  error: string | null

  setWidth: (value: number) => void
  resetWidth: () => void
  start: () => Promise<void>
}

export function useCompartments(
  heUploadId: string | null,
  ihcUploadId: string | null,
  /** Step 11's stamp: re-segmented nuclei are different cells to grow. */
  nucleiVersion: string | null = null,
): CompartmentsStateValue {
  const [report, setReport] = useState<CompartmentsReport | null>(null)
  const [geometry, setGeometry] = useState<Record<number, RegionCompartmentRings>>({})
  const [widthUm, setWidthUm] = useState<number | null>(null)
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
    setGeometry({})
    setWidthUm(null)
    setError(null)
  }, [heUploadId, ihcUploadId, nucleiVersion])

  const stamp = report?.generatedAt ?? null
  const ranks = report?.regions.map((region) => region.rank).join(',') ?? ''
  useEffect(() => {
    if (!heUploadId || !ihcUploadId || !stamp || !ranks) return

    const controller = new AbortController()
    Promise.all(
      ranks
        .split(',')
        .map(Number)
        .map((rank) =>
          fetchCompartmentGeometry(heUploadId, ihcUploadId, rank, controller.signal).catch(
            () => null,
          ),
        ),
    ).then((payloads) => {
      if (controller.signal.aborted || !live.current) return
      const next: Record<number, RegionCompartmentRings> = {}
      for (const payload of payloads) if (payload) next[payload.rank] = payload
      setGeometry(next)
    })

    return () => controller.abort()
  }, [heUploadId, ihcUploadId, ranks, stamp])

  useEffect(() => {
    if (!heUploadId || !ihcUploadId) return

    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setLoading(true)
      fetchCompartments(
        heUploadId,
        ihcUploadId,
        widthUm === null ? {} : { widthUm },
        controller.signal,
      )
        .then((result) => {
          if (controller.signal.aborted || !live.current) return
          setReport(result)
          setError(null)
        })
        .catch((cause: unknown) => {
          if (controller.signal.aborted || !live.current) return
          setError(
            cause instanceof Error ? cause.message : 'could not build the compartments',
          )
        })
        .finally(() => {
          if (live.current) setLoading(false)
        })
    }, DEBOUNCE_MS)

    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [heUploadId, ihcUploadId, nonce, nucleiVersion, widthUm])

  const setWidth = useCallback((value: number) => setWidthUm(value), [])
  const resetWidth = useCallback(() => setWidthUm(null), [])
  const start = useCallback(async () => {
    setNonce((value) => value + 1)
  }, [])

  return useMemo(
    () => ({ report, geometry, widthUm, loading, error, setWidth, resetWidth, start }),
    [error, geometry, loading, report, resetWidth, setWidth, start, widthUm],
  )
}
