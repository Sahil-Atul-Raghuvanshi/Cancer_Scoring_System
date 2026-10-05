/**
 * Owns step 14's state.
 *
 * Job-shaped in feel but request-shaped in fact: the backend reads the label
 * maps step 11 stored and the fields step 11 sampled, which is tens of seconds
 * rather than the minutes step 11 itself took. So there is no poll loop - one
 * request, held open, with a long timeout.
 *
 * It does not start on arrival. Step 14 reopens the slide, and a viewer moving
 * back and forth through the walkthrough should not set that going each time
 * they pass; `start` is called by the page when the step is actually run.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchPerCell, fetchPerCellRows } from '@/api/perCell'
import { cellKey } from '@/features/cells/overlay'
import type { CellPoint, CellRow, PerCellReport } from '@/types/perCell'

export interface PerCellStateValue {
  report: PerCellReport | null
  loading: boolean
  error: string | null
  /** The cell a viewer clicked, from the scatter or from the slide. */
  selected: CellPoint | null
  select: (point: CellPoint | null) => void
  /**
   * Every measured cell, keyed the one legal way - `field:id`.
   *
   * The report carries a sample for the scatter; the slide needs all of them,
   * because a cell the sample skipped is still on the slide and still has to
   * answer a click. Fetched once the report exists and kept.
   */
  rows: Map<string, CellRow>
  /** The same selection as `selected`, in the key the slide overlay uses. */
  selectedKey: string | null
  selectKey: (key: string | null) => void
  start: () => Promise<void>
}

export function usePerCell(
  heUploadId: string | null,
  ihcUploadId: string | null,
  /** Step 13's stamp: different compartments are different pixels to measure. */
  compartmentsVersion: string | null = null,
): PerCellStateValue {
  const [report, setReport] = useState<PerCellReport | null>(null)
  const [rows, setRows] = useState<Map<string, CellRow>>(() => new Map())
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<CellPoint | null>(null)

  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  // A new pair, or re-shaped compartments, means the rows on screen describe
  // something that no longer exists.
  useEffect(() => {
    setReport(null)
    setRows(new Map())
    setSelected(null)
    setError(null)
  }, [heUploadId, ihcUploadId, compartmentsVersion])

  const stamp = report?.generatedAt ?? null
  useEffect(() => {
    if (!heUploadId || !ihcUploadId || !stamp) return

    const controller = new AbortController()
    fetchPerCellRows(heUploadId, ihcUploadId, controller.signal)
      .then((payload) => {
        if (controller.signal.aborted || !live.current) return
        const next = new Map<string, CellRow>()
        for (const row of payload.cells) next.set(cellKey(row.fieldIndex, row.cellId), row)
        setRows(next)
      })
      .catch(() => {
        // The scatter and the numbers still work; only the slide overlay is lost,
        // and it says so rather than showing an empty slide as if nothing was found.
      })

    return () => controller.abort()
  }, [heUploadId, ihcUploadId, stamp])

  const start = useCallback(async () => {
    if (!heUploadId || !ihcUploadId) return
    setLoading(true)
    setError(null)
    try {
      const result = await fetchPerCell(heUploadId, ihcUploadId)
      if (!live.current) return
      setReport(result)
    } catch (cause: unknown) {
      if (!live.current) return
      setError(cause instanceof Error ? cause.message : 'could not measure the cells')
      throw cause
    } finally {
      if (live.current) setLoading(false)
    }
  }, [heUploadId, ihcUploadId])

  const select = useCallback((point: CellPoint | null) => setSelected(point), [])

  const selectKey = useCallback(
    (key: string | null) => setSelected(key ? (rows.get(key) ?? null) : null),
    [rows],
  )

  const selectedKey = selected ? cellKey(selected.fieldIndex, selected.cellId) : null

  return useMemo(
    () => ({ report, rows, loading, error, selected, selectedKey, select, selectKey, start }),
    [error, loading, report, rows, select, selectKey, selected, selectedKey, start],
  )
}
