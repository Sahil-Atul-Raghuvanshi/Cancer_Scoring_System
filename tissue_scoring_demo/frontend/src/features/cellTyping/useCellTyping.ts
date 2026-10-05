/**
 * Owns step 12's state, and it is the only step hook with no job in it.
 *
 * Steps 8, 10 and 11 start work, poll it and read a report. This one asks a
 * question and gets an answer, because the backend opens no slide and runs no
 * model - it applies five thresholds to numbers step 11 already stored. That is
 * what lets the thresholds be live on screen, and it is why the interesting part
 * of this hook is debouncing a slider rather than polling a worker.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchCellTyping, fetchRegionTypes } from '@/api/cellTyping'
import type { CellTypingReport, TypingRules } from '@/types/cellTyping'

/** Long enough that dragging a slider does not fire a request per pixel. */
const DEBOUNCE_MS = 250

export interface CellTypingStateValue {
  report: CellTypingReport | null
  /**
   * A class per cell, keyed `field:id`, per region.
   *
   * Fetched separately from the report and re-fetched whenever it changes, so
   * the colours on the slide are the colours of the thresholds currently on the
   * sliders. A stale map here would be the one thing this screen must not do:
   * show a mix in the bar that the picture beside it disagrees with.
   */
  types: Record<number, Record<string, number>>
  /** The thresholds currently on screen, which lead the report while a drag settles. */
  rules: Partial<TypingRules>
  loading: boolean
  error: string | null

  setRule: (key: keyof TypingRules, value: number) => void
  resetRules: () => void
  reload: () => void
  /**
   * The same shape every other step hook exposes, so `DemoPage`'s task table can
   * hold a stable reference rather than an arrow built on each render. There is
   * no job to start here - this re-asks the question.
   */
  start: () => Promise<void>
}

export function useCellTyping(
  heUploadId: string | null,
  ihcUploadId: string | null,
  /** Step 11's stamp: re-segmented nuclei are different cells to sort. */
  nucleiVersion: string | null = null,
): CellTypingStateValue {
  const [report, setReport] = useState<CellTypingReport | null>(null)
  const [types, setTypes] = useState<Record<number, Record<string, number>>>({})
  const [rules, setRules] = useState<Partial<TypingRules>>({})
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
    setTypes({})
    setRules({})
    setError(null)
  }, [heUploadId, ihcUploadId, nucleiVersion])

  // The per-cell classes behind the report just fetched. Keyed on the report's
  // own stamp rather than on the rules, because the stamp is what the server
  // says it wrote - asking on the rules would race the debounce and colour the
  // slide from a set of thresholds nobody is looking at.
  const stamp = report?.generatedAt ?? null
  const ranks = report?.regions.map((region) => region.rank).join(',') ?? ''
  useEffect(() => {
    if (!heUploadId || !ihcUploadId || !stamp || !ranks) return

    const controller = new AbortController()
    const wanted = ranks.split(',').map(Number)

    Promise.all(
      wanted.map((rank) =>
        fetchRegionTypes(heUploadId, ihcUploadId, rank, controller.signal).catch(() => null),
      ),
    ).then((payloads) => {
      if (controller.signal.aborted || !live.current) return
      const next: Record<number, Record<string, number>> = {}
      for (const payload of payloads) if (payload) next[payload.rank] = payload.types
      setTypes(next)
    })

    return () => controller.abort()
  }, [heUploadId, ihcUploadId, ranks, stamp])

  useEffect(() => {
    if (!heUploadId || !ihcUploadId) return

    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setLoading(true)
      fetchCellTyping(heUploadId, ihcUploadId, rules, controller.signal)
        .then((result) => {
          if (controller.signal.aborted || !live.current) return
          setReport(result)
          setError(null)
        })
        .catch((cause: unknown) => {
          if (controller.signal.aborted || !live.current) return
          // Step 11 not having run is the ordinary case here, not a fault: the
          // panel says so rather than showing an error box.
          setError(cause instanceof Error ? cause.message : 'could not sort the cells')
        })
        .finally(() => {
          if (live.current) setLoading(false)
        })
    }, DEBOUNCE_MS)

    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [heUploadId, ihcUploadId, nonce, nucleiVersion, rules])

  const setRule = useCallback((key: keyof TypingRules, value: number) => {
    setRules((current) => ({ ...current, [key]: value }))
  }, [])

  const resetRules = useCallback(() => setRules({}), [])
  const reload = useCallback(() => setNonce((value) => value + 1), [])
  const start = useCallback(async () => {
    setNonce((value) => value + 1)
  }, [])

  return useMemo(
    () => ({ report, types, rules, loading, error, setRule, resetRules, reload, start }),
    [error, loading, reload, report, resetRules, rules, setRule, start, types],
  )
}
