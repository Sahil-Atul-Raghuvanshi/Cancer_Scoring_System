/**
 * Owns step 11's state: the pass, its progress, and one region's outlines.
 *
 * Job-shaped like steps 8 and 10 - start it, poll it, read the report - because
 * segmenting a few dozen fields is a minute or two of CPU in a worker thread.
 *
 * The one thing this hook does that steps 8 and 10 do not is fetch geometry
 * separately. A region's outlines are megabytes of vertices and the report is
 * re-fetched every time the screen opens, so the summary and the polygons are
 * two requests: the summary always, the polygons only for the region a viewer is
 * actually looking at, and cached per region once fetched.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  fetchNucleiCapability,
  fetchNucleiReport,
  fetchNucleiRun,
  fetchRegionNuclei,
  startNuclei,
} from '@/api/nuclei'
import type {
  NucleiCapability,
  NucleiReport,
  NucleiState,
  RegionNucleiPayload,
} from '@/types/nuclei'

const POLL_MS = 1500

export interface NucleiStateValue {
  capability: NucleiCapability | null
  report: NucleiReport | null
  state: NucleiState | null
  message: string | null
  progress: number
  running: boolean
  error: string | null

  /** Outlines for the regions a viewer has opened, keyed by rank. */
  geometry: Record<number, RegionNucleiPayload>
  loadingRegion: number | null
  loadRegion: (rank: number) => Promise<void>
  /** Every region's outlines, for the screens that pan the whole slide. */
  loadAllRegions: () => Promise<void>
  /** True while the first fetch of the whole set is still in flight. */
  loadingGeometry: boolean

  start: () => Promise<void>
  restart: () => Promise<void>
  reset: () => void
}

export function useNuclei(
  heUploadId: string | null,
  ihcUploadId: string | null,
  /** Step 10's stamp: a re-run alignment is different regions, so this is stale. */
  alignmentVersion: string | null = null,
): NucleiStateValue {
  const [capability, setCapability] = useState<NucleiCapability | null>(null)
  const [report, setReport] = useState<NucleiReport | null>(null)
  const [state, setState] = useState<NucleiState | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [geometry, setGeometry] = useState<Record<number, RegionNucleiPayload>>({})
  const [loadingRegion, setLoadingRegion] = useState<number | null>(null)
  const [loadingGeometry, setLoadingGeometry] = useState(false)

  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    fetchNucleiCapability(controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setCapability(result)
      })
      .catch(() => {
        /* the panel reports this as "not installed" via a null capability */
      })
    return () => controller.abort()
  }, [])

  const clear = useCallback(() => {
    setReport(null)
    setState(null)
    setMessage(null)
    setProgress(0)
    setError(null)
    setGeometry({})
  }, [])

  useEffect(() => {
    clear()
  }, [clear, heUploadId, ihcUploadId, alignmentVersion])

  // Pick up a pass this pair already has, so returning to the step does not
  // offer to redo work that is already on disk.
  useEffect(() => {
    if (!heUploadId || !ihcUploadId) return
    const controller = new AbortController()
    fetchNucleiReport(heUploadId, ihcUploadId, controller.signal)
      .then((existing) => {
        if (controller.signal.aborted) return
        setReport(existing)
        setState(existing.state)
        setProgress(1)
      })
      .catch(() => {
        /* nothing segmented yet; the panel offers to run it */
      })
    return () => controller.abort()
  }, [heUploadId, ihcUploadId, alignmentVersion])

  const poll = useCallback(async () => {
    if (!heUploadId || !ihcUploadId) return

    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, POLL_MS))
      if (!live.current) return

      const run = await fetchNucleiRun(heUploadId, ihcUploadId)
      if (!live.current) return

      setState(run.state)
      setMessage(run.message)
      setProgress(run.progress)

      if (run.state === 'queued' || run.state === 'running') continue

      if (run.state === 'failed') {
        setError(run.error ?? 'the nuclei pass failed')
        throw new Error(run.error ?? 'the nuclei pass failed')
      }

      setReport(await fetchNucleiReport(heUploadId, ihcUploadId))
      setProgress(1)
      return
    }
  }, [heUploadId, ihcUploadId])

  const launch = useCallback(
    async (restart: boolean) => {
      if (!heUploadId || !ihcUploadId) {
        throw new Error('step 11 needs both the H&E and the IHC slide of one case')
      }

      setError(null)
      setState('queued')
      setProgress(0)
      setMessage(restart ? 'segmenting again' : 'starting')
      setGeometry({})

      try {
        const run = await startNuclei(heUploadId, ihcUploadId, { restart })
        if (!live.current) return

        setState(run.state)
        if (run.state === 'ready') {
          setReport(await fetchNucleiReport(heUploadId, ihcUploadId))
          setProgress(1)
          return
        }
        await poll()
      } catch (cause) {
        const detail = cause instanceof Error ? cause.message : 'could not segment the nuclei'
        if (live.current) {
          setError(detail)
          setState('failed')
        }
        throw new Error(detail)
      }
    },
    [heUploadId, ihcUploadId, poll],
  )

  const start = useCallback(() => launch(false), [launch])
  const restart = useCallback(() => launch(true), [launch])

  const loadRegion = useCallback(
    async (rank: number) => {
      if (!heUploadId || !ihcUploadId) return
      if (geometry[rank]) return

      setLoadingRegion(rank)
      try {
        const payload = await fetchRegionNuclei(heUploadId, ihcUploadId, rank)
        if (live.current) setGeometry((current) => ({ ...current, [rank]: payload }))
      } catch {
        /* the viewer still gets the rendered overlay; only the live count is lost */
      } finally {
        if (live.current) setLoadingRegion(null)
      }
    },
    [geometry, heUploadId, ihcUploadId],
  )

  /**
   * Every region at once, for the screens that pan the whole slide.
   *
   * Steps 11 to 15 let a viewer scroll from one region of tumour to the next,
   * and a fetch triggered by arriving somewhere would mean panning into a region
   * that is briefly empty - which looks exactly like a region the model found
   * nothing in. There are three of them and they are a few megabytes, so they
   * are fetched together and kept.
   *
   * Failures are per region on purpose: two regions of outlines and one that
   * could not be read is a better screen than no outlines at all, and the count
   * under the frame counts what is actually drawn.
   */
  const loadAllRegions = useCallback(async () => {
    if (!heUploadId || !ihcUploadId || !report) return
    const wanted = report.regions.map((region) => region.rank).filter((rank) => !geometry[rank])
    if (wanted.length === 0) return

    setLoadingGeometry(true)
    try {
      const payloads = await Promise.all(
        wanted.map((rank) =>
          fetchRegionNuclei(heUploadId, ihcUploadId, rank).catch(() => null),
        ),
      )
      if (!live.current) return
      setGeometry((current) => {
        const next = { ...current }
        for (const payload of payloads) if (payload) next[payload.rank] = payload
        return next
      })
    } finally {
      if (live.current) setLoadingGeometry(false)
    }
  }, [geometry, heUploadId, ihcUploadId, report])

  return useMemo(
    () => ({
      capability,
      report,
      state,
      message,
      progress,
      running: state === 'queued' || state === 'running',
      error,
      geometry,
      loadingRegion,
      loadRegion,
      loadAllRegions,
      loadingGeometry,
      start,
      restart,
      reset: clear,
    }),
    [
      capability,
      clear,
      error,
      geometry,
      loadAllRegions,
      loadRegion,
      loadingGeometry,
      loadingRegion,
      message,
      progress,
      report,
      restart,
      start,
      state,
    ],
  )
}
