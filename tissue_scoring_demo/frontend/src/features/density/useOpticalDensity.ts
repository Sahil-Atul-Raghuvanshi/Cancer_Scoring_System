/**
 * Owns step 5's state, and one piece of choreography that matters.
 *
 * Step 5's own control is the tile, and only the tile. Its *inputs* are steps 3
 * and 4's decisions — the tissue mask and the white point — so this hook takes
 * step 3's committed cut as an argument rather than owning a copy, exactly as step
 * 4's hook does. When the viewer goes back and moves that slider, this step's
 * result is stale: a different mask is a different white point, and a different
 * white point is a different density for every pixel on screen.
 *
 * The tile is a *committed* control rather than a drafted one, which is the
 * difference from step 4's percentile. A percentile can be swept and watched; a
 * tile cannot be interpolated, so there is nothing to preview between two of them
 * and a debounce would only add lag. Picking a tile fires one request and the
 * previous report stays up, dimmed, until the new one lands.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchDensity } from '@/api/density'
import type { DensityReport } from '@/types/density'

/** A tile position in level-0 coordinates, or null for "let the score choose". */
export interface TilePick {
  x: number
  y: number
}

export interface OpticalDensityState {
  report: DensityReport | null
  /** True while the first run for this slide is in flight. */
  loading: boolean
  /** True while a re-run on a different tile is in flight and a report is on screen. */
  refining: boolean
  error: string | null

  /** The tile the report on screen was computed on, or null for the best-scoring one. */
  picked: TilePick | null
  /** True once the viewer has chosen a tile rather than taking the score's. */
  manual: boolean

  /** Transform a different tile. Null hands the choice back to the score. */
  pick: (tile: TilePick | null) => void
  /** Run step 5. Rejects on failure so the pipeline can mark the step errored. */
  start: () => Promise<void>
  reset: () => void
}

export function useOpticalDensity(
  uploadId: string | null,
  /**
   * Step 3's committed cut, or null to let its own rule choose.
   *
   * Passed in rather than read from step 5's own report, so that returning to step
   * 3 and moving the slider invalidates this step — which it must, because a
   * different mask gives step 4 different glass, and step 4's I₀ is the
   * denominator of every density here.
   */
  tissueThreshold: number | null,
): OpticalDensityState {
  const [report, setReport] = useState<DensityReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [refining, setRefining] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [picked, setPicked] = useState<TilePick | null>(null)
  const [manual, setManual] = useState(false)

  // Guards a response arriving after the viewer has left the step.
  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  // Only the newest pick may write to state. Without this, clicking three tiles
  // quickly can leave whichever request happens to finish last on screen, which
  // may not be the one the viewer asked for.
  const inFlight = useRef(0)

  const clear = useCallback(() => {
    setReport(null)
    setError(null)
    setPicked(null)
    setManual(false)
    setRefining(false)
  }, [])

  useEffect(() => {
    clear()
  }, [clear, uploadId])

  // Step 3's cut changed, so step 4's white point did, so every density here did.
  // Clearing rather than silently refetching is deliberate: the numbers on screen
  // belong to the previous white point, and leaving them up beside a new mask is
  // exactly the quiet mismatch the earlier steps' screens are built to avoid.
  useEffect(() => {
    clear()
  }, [clear, tissueThreshold])

  const start = useCallback(async () => {
    if (!uploadId) throw new Error('no slide loaded')

    setLoading(true)
    setError(null)

    try {
      const result = await fetchDensity(uploadId, { threshold: tissueThreshold })
      if (!live.current) return

      setReport(result)
      setPicked(null)
      setManual(false)
    } catch (cause) {
      const message =
        cause instanceof Error ? cause.message : 'could not transform a tile to optical density'
      if (live.current) setError(message)
      throw new Error(message)
    } finally {
      if (live.current) setLoading(false)
    }
  }, [tissueThreshold, uploadId])

  const pick = useCallback(
    (tile: TilePick | null) => {
      if (!uploadId) return

      const ticket = ++inFlight.current
      setPicked(tile)
      setManual(tile !== null)
      setRefining(true)

      fetchDensity(uploadId, {
        threshold: tissueThreshold,
        x: tile?.x ?? null,
        y: tile?.y ?? null,
      })
        .then((result) => {
          if (!live.current || ticket !== inFlight.current) return
          setReport(result)
          // Snap onto the tile the server really used - it may have moved the
          // request to the nearest block it actually scored, and the screen must
          // describe the run that happened rather than the one that was asked for.
          setPicked({ x: result.tile.x, y: result.tile.y })
          setManual(result.tile.requested)
          setError(null)
        })
        .catch((cause: unknown) => {
          if (!live.current || ticket !== inFlight.current) return
          setError(
            cause instanceof Error
              ? `could not transform that tile: ${cause.message}`
              : 'could not transform that tile',
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
      picked,
      manual,
      pick,
      start,
      reset: clear,
    }),
    [clear, error, loading, manual, pick, picked, refining, report, start],
  )
}
