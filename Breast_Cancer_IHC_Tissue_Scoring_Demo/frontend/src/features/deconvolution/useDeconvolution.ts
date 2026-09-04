/**
 * Owns step 6's state.
 *
 * Simpler than step 5's hook, because step 6 has no control of its own. Its
 * inputs are steps 3, 4 and 5's decisions and its one decision — which stain
 * vectors to measure with — is not adjustable by design. So there is nothing to
 * debounce and nothing to re-fetch: one run, and a toggle that only swaps which
 * of two already-loaded answers is on screen.
 *
 * The one piece of choreography that matters is the tile. Step 6 must un-mix the
 * *same* field of view step 5 just showed, or the fork is a drawing rather than a
 * fact about the code — so step 5's chosen tile is passed in here and travels with
 * every request. When the viewer goes back and picks a different tile, or moves
 * step 3's threshold, this step's result is stale and is cleared rather than left
 * on screen beside inputs that no longer produced it.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { fetchDeconvolution } from '@/api/deconvolution'
import type { DeconvolutionBasisName, DeconvolutionReport } from '@/types/deconvolution'

/** The tile step 5 settled on, in level-0 coordinates. */
export interface TileOrigin {
  x: number
  y: number
}

export interface DeconvolutionState {
  report: DeconvolutionReport | null
  loading: boolean
  error: string | null

  /** Which basis the panels on screen are drawn from. */
  basis: DeconvolutionBasisName
  setBasis: (basis: DeconvolutionBasisName) => void

  /** Run step 6. Rejects on failure so the pipeline can mark the step errored. */
  start: () => Promise<void>
  reset: () => void
}

export function useDeconvolution(
  uploadId: string | null,
  /** Step 3's committed cut, or null to let its own rule choose. */
  tissueThreshold: number | null,
  /**
   * Step 5's tile, or null before step 5 has run.
   *
   * Passed in rather than left to the server to re-choose, even though the server
   * would choose the same tile by the same score. The point is that the two steps
   * are demonstrably looking at the same pixels: if the viewer picks a different
   * tile on step 5's screen, this step follows it there.
   */
  tile: TileOrigin | null,
): DeconvolutionState {
  const [report, setReport] = useState<DeconvolutionReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [basis, setBasis] = useState<DeconvolutionBasisName>('fixed')

  // Guards a response arriving after the viewer has left the step.
  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  const clear = useCallback(() => {
    setReport(null)
    setError(null)
    // Back to the basis the measurement branch actually uses. Leaving the toggle
    // on "estimated" across a re-run would mean the next slide opens showing the
    // vectors this step exists to argue against.
    setBasis('fixed')
  }, [])

  useEffect(() => {
    clear()
  }, [clear, uploadId])

  // Step 3's cut changed, so step 4's white point did, so step 5's densities did,
  // so these channels did. Clearing rather than quietly refetching: the numbers on
  // screen belong to the previous white point, and leaving them up beside a new
  // mask is the exact mismatch every earlier step's screen is built to avoid.
  useEffect(() => {
    clear()
  }, [clear, tissueThreshold])

  // And the same for the tile. A different field of view is a different set of
  // stain readings; there is nothing about the old ones worth keeping on screen.
  const tileKey = tile ? `${tile.x},${tile.y}` : null
  useEffect(() => {
    clear()
  }, [clear, tileKey])

  const start = useCallback(async () => {
    if (!uploadId) throw new Error('no slide loaded')

    setLoading(true)
    setError(null)

    try {
      const result = await fetchDeconvolution(uploadId, {
        threshold: tissueThreshold,
        x: tile?.x ?? null,
        y: tile?.y ?? null,
      })
      if (!live.current) return

      setReport(result)
      setBasis('fixed')
    } catch (cause) {
      const message =
        cause instanceof Error ? cause.message : 'could not separate the stains'
      if (live.current) setError(message)
      throw new Error(message)
    } finally {
      if (live.current) setLoading(false)
    }
  }, [tile?.x, tile?.y, tissueThreshold, uploadId])

  return useMemo(
    () => ({
      report,
      loading,
      error,
      basis,
      setBasis,
      start,
      reset: clear,
    }),
    [basis, clear, error, loading, report, start],
  )
}
