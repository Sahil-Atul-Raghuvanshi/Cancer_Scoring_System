/**
 * Owns step 10's state: the candidate list, and which of them are ticked.
 *
 * Request-shaped like step 9's hook rather than job-shaped like step 8's — there is no
 * model here, so nothing to poll and nothing to cancel.
 *
 * The one thing worth stating is how ticking works. The server is the owner of the
 * selection: `setRoiSelection` replaces it outright and returns the whole report, so
 * every number under the list (selected area, share of the tumour, the windows step 11
 * will spend) arrives already consistent with the ticks. This hook holds an optimistic
 * copy of the ticked ids so a checkbox responds immediately, and reconciles to whatever
 * the server says. On failure it reverts — a tick that stayed on screen after the server
 * rejected it would let somebody run step 11 on a different set than they are looking at.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { buildRoiSelection, setRoiSelection } from '@/api/roiSelection'
import type { RoiSelectionReport } from '@/types/roiSelection'

export interface RoiSelectionState {
  report: RoiSelectionReport | null
  /** The ticked ids, optimistic — what the checkboxes render from. */
  selected: string[]
  /** True while the first build for this slide is in flight. */
  loading: boolean
  /** True while a tick is being sent. The list stays usable; only the run button waits. */
  saving: boolean
  error: string | null

  start: () => Promise<void>
  restart: () => Promise<void>
  toggle: (roiId: string) => Promise<void>
  selectAll: () => Promise<void>
  deselectAll: () => Promise<void>
  /** Back to the coverage rule's own answer, whatever has been ticked since. */
  resetToDefault: () => Promise<void>
  reset: () => void
}

export function useRoiSelection(
  uploadId: string | null,
  /**
   * Step 8's stamp on the pass these candidates come from. Not read for its value, only
   * for when it changes: the ids are area ranks within a class map, so a fresh run of
   * step 8 makes every id on screen point at different tissue.
   */
  classMapVersion: string | null = null,
): RoiSelectionState {
  const [report, setReport] = useState<RoiSelectionReport | null>(null)
  const [selected, setSelected] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  const clear = useCallback(() => {
    setReport(null)
    setSelected([])
    setError(null)
    setLoading(false)
    setSaving(false)
  }, [])

  useEffect(() => {
    clear()
  }, [clear, uploadId, classMapVersion])

  const adopt = useCallback((result: RoiSelectionReport) => {
    setReport(result)
    setSelected(result.selected)
  }, [])

  const launch = useCallback(
    async (rebuild: boolean) => {
      if (!uploadId) throw new Error('no slide loaded')

      setLoading(true)
      setError(null)

      try {
        const result = await buildRoiSelection(uploadId, { rebuild })
        if (!live.current) return
        adopt(result)
      } catch (cause) {
        const message =
          cause instanceof Error ? cause.message : 'could not list the candidate regions'
        if (live.current) setError(message)
        throw new Error(message)
      } finally {
        if (live.current) setLoading(false)
      }
    },
    [adopt, uploadId],
  )

  /**
   * Send a selection, showing it immediately and reverting if the server refuses.
   *
   * Sequenced rather than concurrent: a viewer clicking four checkboxes quickly would
   * otherwise have four replacements in flight, and the last reply to arrive would win
   * regardless of which click it came from.
   */
  const pending = useRef<Promise<unknown>>(Promise.resolve())
  const commit = useCallback(
    async (next: string[]) => {
      if (!uploadId) return
      const previous = selected

      setSelected(next)
      setSaving(true)
      setError(null)

      const run = pending.current
        .catch(() => undefined)
        .then(() => setRoiSelection(uploadId, next))

      pending.current = run
      try {
        const result = await run
        if (!live.current) return
        adopt(result)
      } catch (cause) {
        const message =
          cause instanceof Error ? cause.message : 'could not record the selection'
        if (live.current) {
          setSelected(previous)
          setError(message)
        }
      } finally {
        if (live.current && pending.current === run) setSaving(false)
      }
    },
    [adopt, selected, uploadId],
  )

  const toggle = useCallback(
    (roiId: string) => {
      const next = selected.includes(roiId)
        ? selected.filter((one) => one !== roiId)
        : // Kept in candidate order rather than click order, so the list the server
          // stores reads the same way the screen does and step 11 works largest first.
          (report?.candidates ?? [])
            .map((one) => one.roiId)
            .filter((one) => one === roiId || selected.includes(one))
      return commit(next)
    },
    [commit, report, selected],
  )

  const selectAll = useCallback(
    () => commit((report?.candidates ?? []).map((one) => one.roiId)),
    [commit, report],
  )

  const deselectAll = useCallback(() => commit([]), [commit])

  const resetToDefault = useCallback(
    () => commit(report?.defaultSelection ?? []),
    [commit, report],
  )

  return useMemo(
    () => ({
      report,
      selected,
      loading,
      saving,
      error,
      start: () => launch(false),
      restart: () => launch(true),
      toggle,
      selectAll,
      deselectAll,
      resetToDefault,
      reset: clear,
    }),
    [
      clear,
      deselectAll,
      error,
      launch,
      loading,
      report,
      resetToDefault,
      saving,
      selectAll,
      selected,
      toggle,
    ],
  )
}
