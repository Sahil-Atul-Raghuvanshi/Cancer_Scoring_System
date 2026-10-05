/**
 * Owns the previous-cases list and the three things you can do to a run.
 *
 * Every action re-fetches the whole list rather than patching the entry it
 * touched. The list is small, the actions move directories, and a local patch
 * would be this screen's own opinion of what happened on disk - which is exactly
 * the thing the backend refuses to keep, because it reads state off the files
 * rather than from a flag somebody wrote.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  archiveHistoryMarker,
  deleteHistoryMarker,
  fetchHistory,
  openHistoryMarker,
} from '@/api/history'
import type { HistoryCase } from '@/types/history'

export interface HistoryStateValue {
  cases: HistoryCase[]
  loading: boolean
  error: string | null
  /** The `caseId:marker` currently being moved, so one row can show a spinner. */
  busy: string | null
  reload: () => Promise<void>
  /** Rename a filed run back into the working tree. Resolves when it is there. */
  open: (caseId: string, marker: string) => Promise<void>
  archive: (caseId: string, marker: string) => Promise<void>
  remove: (caseId: string, marker: string) => Promise<void>
}

export function useHistory(): HistoryStateValue {
  const [cases, setCases] = useState<HistoryCase[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      const payload = await fetchHistory()
      if (!live.current) return
      setCases(payload.cases)
      setError(null)
    } catch (cause) {
      if (!live.current) return
      setError(cause instanceof Error ? cause.message : 'could not read the stored cases')
    } finally {
      if (live.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    void reload()
  }, [reload])

  const act = useCallback(
    async (
      caseId: string,
      marker: string,
      action: (caseId: string, marker: string) => Promise<unknown>,
    ) => {
      setBusy(`${caseId}:${marker}`)
      setError(null)
      try {
        await action(caseId, marker)
        await reload()
      } catch (cause) {
        if (live.current) {
          setError(cause instanceof Error ? cause.message : 'that did not work')
        }
        throw cause
      } finally {
        if (live.current) setBusy(null)
      }
    },
    [reload],
  )

  const open = useCallback(
    (caseId: string, marker: string) => act(caseId, marker, openHistoryMarker),
    [act],
  )
  const archive = useCallback(
    (caseId: string, marker: string) => act(caseId, marker, archiveHistoryMarker),
    [act],
  )
  const remove = useCallback(
    (caseId: string, marker: string) => act(caseId, marker, deleteHistoryMarker),
    [act],
  )

  return useMemo(
    () => ({ cases, loading, error, busy, reload, open, archive, remove }),
    [archive, busy, cases, error, loading, open, reload, remove],
  )
}
