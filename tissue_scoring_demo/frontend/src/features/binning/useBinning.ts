/**
 * Owns step 15's state.
 *
 * Request-shaped, and fast: the backend applies five numbers to rows step 14
 * already stored - no slide is opened and nothing is segmented. That is what
 * makes arriving on the step the request, with no button in between.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchBinnedCells, fetchBinning } from '@/api/binning'
import { cellKey } from '@/features/cells/overlay'
import type { BinnedCell, BinningReport } from '@/types/binning'

export interface BinningStateValue {
  report: BinningReport | null
  loading: boolean
  error: string | null
  /**
   * Every cell's level and verdict, keyed `field:id`.
   *
   * From the server, not worked out here. The bin is one comparison and the
   * cuts are on the report, but positivity is two conditions plus a
   * partial-staining rule that is an open question with three implemented
   * answers - and a copy of that rule in the browser would be a second place
   * for it to be decided. The colours on the slide and the percentage in the
   * table have to come from one application of it.
   */
  cells: Map<string, BinnedCell>
  selected: string | null
  select: (key: string | null) => void
  start: () => Promise<void>
}

export function useBinning(
  heUploadId: string | null,
  ihcUploadId: string | null,
  /** Step 14's stamp: re-measured cells are different numbers to bin. */
  measurementVersion: string | null = null,
): BinningStateValue {
  const [report, setReport] = useState<BinningReport | null>(null)
  const [cells, setCells] = useState<Map<string, BinnedCell>>(() => new Map())
  const [selected, setSelected] = useState<string | null>(null)
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
    setCells(new Map())
    setSelected(null)
    setError(null)
  }, [heUploadId, ihcUploadId, measurementVersion])

  const stamp = report?.generatedAt ?? null
  useEffect(() => {
    if (!heUploadId || !ihcUploadId || !stamp) return

    const controller = new AbortController()
    fetchBinnedCells(heUploadId, ihcUploadId, controller.signal)
      .then((payload) => {
        if (controller.signal.aborted || !live.current) return
        const next = new Map<string, BinnedCell>()
        for (const cell of payload.cells) {
          next.set(cellKey(cell.fieldIndex, cell.cellId), cell)
        }
        setCells(next)
      })
      .catch(() => {
        // The histogram and the totals still work; only the slide is lost, and
        // the viewer says so rather than drawing an empty one.
      })

    return () => controller.abort()
  }, [heUploadId, ihcUploadId, stamp])

  useEffect(() => {
    if (!heUploadId || !ihcUploadId || !measurementVersion) return

    const controller = new AbortController()
    setLoading(true)
    fetchBinning(heUploadId, ihcUploadId, controller.signal)
      .then((result) => {
        if (controller.signal.aborted || !live.current) return
        setReport(result)
        setError(null)
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted || !live.current) return
        setError(cause instanceof Error ? cause.message : 'could not bin the cells')
      })
      .finally(() => {
        if (live.current) setLoading(false)
      })

    return () => controller.abort()
  }, [heUploadId, ihcUploadId, measurementVersion, nonce])

  const start = useCallback(async () => {
    setNonce((value) => value + 1)
  }, [])

  const select = useCallback((key: string | null) => setSelected(key), [])

  return useMemo(
    () => ({ report, cells, loading, error, selected, select, start }),
    [cells, error, loading, report, select, selected, start],
  )
}
