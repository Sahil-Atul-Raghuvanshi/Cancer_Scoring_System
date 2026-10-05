/**
 * Owns step 8's state: the capability probe, the long pass, and the class filter.
 *
 * Job-shaped like step 2's hook rather than request-shaped like step 7's, and for a
 * stronger reason: a whole-slide pass is tens of minutes, so it has to be started,
 * polled and cached. What the server caches, this hook picks up for free — a pass
 * already on disk comes back `ready` from the first call, so stepping back onto the
 * screen is instant instead of another half hour.
 *
 * Its *input* is step 7's index, so the hook takes step 3's committed cut as an
 * argument exactly as steps 4 to 7's hooks do. When the viewer goes back and moves
 * that slider the class map on screen belongs to a different set of tiles, so it is
 * cleared rather than left up beside numbers that no longer describe it.
 *
 * The class filter is deliberately *not* server state: it is a way of looking at a
 * finished pass, so toggling it changes an image URL and nothing else. That is what
 * makes "switch fat off and watch the denominator change" instant.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { CancelledError } from '@/api/client'
import {
  cancelTissueTypeRun,
  fetchTissueTypeCapability,
  fetchTissueTypeReport,
  fetchTissueTypeRun,
  runTissueType,
} from '@/api/tissueType'
import type {
  TissueTypeCapability,
  TissueTypeReport,
  TissueTypeRun,
} from '@/types/tissueType'

/**
 * Every class of a *trained* pass, which is what the map shows until the viewer
 * switches one off.
 *
 * **Step 8 no longer has one class count.** The two trained options emit three classes;
 * the BEETLE option emits its own five. So this is the default and the fallback, and
 * the live count comes from the server - `report.classes` once a pass has finished, or
 * `run.paint.colours` while one is going. `allClasses` on the state is that count
 * resolved, and the class filter is built from it rather than from this constant, which
 * is what stops a five-class pass being drawn with two of its classes permanently off.
 */
export const ALL_CLASSES = [0, 1, 2] as const

/** `[0, 1, ..., count - 1]`. The filter's "everything" for a pass of `count` classes. */
function everyClass(count: number): number[] {
  return Array.from({ length: Math.max(1, count) }, (_, index) => index)
}

export interface TissueTypeState {
  /** What the server can do, or null while probing. */
  capability: TissueTypeCapability | null
  probing: boolean
  /** Non-null when the capability probe itself failed — the backend is unreachable. */
  probeError: string | null
  retryProbe: () => void

  report: TissueTypeReport | null
  /** Live state of the pass, including its progress while it runs. */
  run: TissueTypeRun | null
  running: boolean
  error: string | null

  /**
   * Every patch classified so far, as flat `row, col, class` triples — the pass being
   * painted onto the slide while it runs.
   *
   * **One array, accumulated in a ref, handed out by identity.** A poll returns only
   * what the screen has not seen, so the screen has to keep the rest; keeping it in
   * React state instead would copy a growing array of a hundred thousand numbers on
   * every poll, and the canvas that draws it only ever needs the new tail. So this is
   * the same array object throughout a run, `paintedCount` is what changes, and the
   * canvas draws from where it left off to there. That also makes a dropped render
   * harmless: the canvas catches up from its own mark rather than missing a batch.
   */
  painted: number[]
  /** Patches in `painted`. The number that moves; the array's identity does not. */
  paintedCount: number
  /**
   * One base64 pixel mask per patch in `painted`, same order, on the BEETLE option.
   *
   * Accumulated in a ref beside `painted` and for the same reason: a poll returns only
   * what the screen has not seen, and the canvas only ever needs the new tail. Empty on
   * the trained options, where a patch is one colour.
   */
  paintedMasks: string[]

  /** Class ids this pass emits — three on a trained option, five on BEETLE. */
  allClasses: number[]

  /** Which classes the map draws. A view of a finished pass, not a re-run. */
  visible: number[]
  toggleClass: (id: number) => void
  showAllClasses: () => void

  /** Run step 8. Rejects on failure so the pipeline can mark the step errored. */
  start: () => Promise<void>
  /**
   * Throw the cached pass away and run again.
   *
   * Matters more here than anywhere else in the pipeline: `start` returns the cache
   * when the parameters match, which is what makes revisiting this screen instant
   * rather than another half hour - and is exactly wrong when the viewer is asking
   * for a fresh pass.
   */
  restart: () => Promise<void>
  /**
   * Ask the running pass to stop.
   *
   * A real server-side stop rather than an abandoned wait, which matters at this
   * duration: an unwatched pass would keep every core busy for another half hour and
   * starve whatever the viewer moved on to. `start` then rejects with a
   * `CancelledError`, which the pipeline treats as "back to ready" rather than a
   * failure.
   */
  cancel: () => Promise<void>
  /** True while a stop has been asked for but the pass has not yet acknowledged it. */
  cancelling: boolean
  reset: () => void
}

export function useTissueType(
  uploadId: string | null,
  /** Step 3's committed cut, or null to let its own rule choose. */
  tissueThreshold: number | null,
  /** Step 7's committed overlap, or null before it has run - see `launch`. */
  tileOverlap: number | null,
  /**
   * Which published checkpoint to run - **step 7's, not this screen's**.
   *
   * The checkpoint is decided by the field of view chosen on step 7, because a field of
   * view is a property of the weights: "448 µm" and "the head fitted at 448 µm" are one
   * decision, and taking it in two places is how the two steps come apart. Null means
   * step 7's field of view has no published head yet; the run is refused rather than
   * quietly falling back to a checkpoint fitted at a different scale.
   */
  modelName: string | null = null,
  /**
   * Step 7's committed option - `h_channel` or `he`. Sent as an **assertion**: the
   * server runs whatever step 7 committed and returns a conflict if this disagrees,
   * rather than running a grid other than the one that was priced. Null before step 7
   * has run.
   */
  branch: string | null = null,
  /** Step 7's committed field of view, in microns. An assertion for the same reason. */
  fieldOfView: number | null = null,
): TissueTypeState {
  const [capability, setCapability] = useState<TissueTypeCapability | null>(null)
  const [probing, setProbing] = useState(true)
  const [probeError, setProbeError] = useState<string | null>(null)
  const [probeAttempt, setProbeAttempt] = useState(0)

  const [report, setReport] = useState<TissueTypeReport | null>(null)
  const [run, setRun] = useState<TissueTypeRun | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [visible, setVisible] = useState<number[]>([...ALL_CLASSES])
  const [cancelling, setCancelling] = useState(false)

  // The paint feed. See `painted` on the state contract for why it is a ref.
  const painted = useRef<number[]>([])
  const paintedMasks = useRef<string[]>([])
  const [paintedCount, setPaintedCount] = useState(0)

  const clearPaint = useCallback(() => {
    // Emptied in place rather than replaced, so the canvas keeps the same arrays and
    // notices the reset through the count going back to zero.
    painted.current.length = 0
    paintedMasks.current.length = 0
    setPaintedCount(0)
  }, [])

  // Guards a response arriving after the viewer has left the step.
  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  /* --- can it run at all -------------------------------------------------- */

  useEffect(() => {
    const controller = new AbortController()
    setProbing(true)
    setProbeError(null)

    fetchTissueTypeCapability(controller.signal)
      .then((result) => {
        if (!live.current) return
        setCapability(result)
      })
      .catch((cause: unknown) => {
        if (!live.current || controller.signal.aborted) return
        setCapability(null)
        setProbeError(
          cause instanceof Error
            ? cause.message
            : 'could not ask the server whether step 8 can run',
        )
      })
      .finally(() => {
        if (live.current) setProbing(false)
      })

    return () => controller.abort()
  }, [probeAttempt])

  const retryProbe = useCallback(() => setProbeAttempt((value) => value + 1), [])

  /* --- what is already on the server -------------------------------------- */

  const clear = useCallback(() => {
    setReport(null)
    setRun(null)
    setError(null)
    setRunning(false)
    setCancelling(false)
    setVisible([...ALL_CLASSES])
    clearPaint()
  }, [clearPaint])

  // Everything this step's answer is a statement about, as one key.
  //
  // One effect rather than five, because five would be five chances to forget one -
  // and one *was* forgotten. `modelName` was missing from the list, so changing the
  // field of view on step 7 left the previous class map on screen beside numbers that
  // no longer described it, which is the exact mismatch every earlier step's screen is
  // built to avoid. A derived key cannot go stale by omission in the same way: adding
  // an input to the run means adding it here, in the one place, or the key does not
  // change and the bug is visible immediately.
  //
  // The mask is in it because a different mask is a different set of step 7 tiles and
  // therefore a different set of windows. The overlap is in it for the same reason one
  // step in. The branch and the field of view are in it because they decide which
  // trained model ran, and a class map is a statement about a model as much as about a
  // slide.
  const inputsKey = [
    uploadId,
    tissueThreshold,
    tileOverlap,
    modelName,
    branch,
    fieldOfView,
  ].join('|')

  useEffect(() => {
    clear()
  }, [clear, inputsKey])

  // Pick up a pass this browser did not start - one cached from an earlier session,
  // or still running from a reload. Half an hour of someone else's compute is worth
  // finding rather than repeating.
  //
  // **Keyed on `inputsKey`, not on `uploadId`, and that difference was a bug.** The
  // effect above clears the report whenever any input changes; this one used to look
  // for a stored pass only when the *slide* changed. Step 7's model, branch and field
  // of view are part of the key and they arrive a moment after the page loads - so
  // the sequence on any already-processed slide was: fetch the stored report, then
  // step 7 lands, then the key changes, then the report is cleared, and nothing ever
  // looks again. Step 8 then offered to classify every patch from scratch, ten to
  // thirty minutes, with the answer already sitting on disk.
  //
  // Two effects that disagree about what invalidates a result will always drift like
  // this. They share the key now.
  useEffect(() => {
    if (!uploadId) return

    const controller = new AbortController()
    fetchTissueTypeRun(uploadId, controller.signal)
      .then((state) => {
        if (!live.current || controller.signal.aborted) return
        if (state.state !== 'ready') return

        setRun(state)
        return fetchTissueTypeReport(uploadId, controller.signal).then((cached) => {
          if (!live.current || controller.signal.aborted) return
          setReport(cached)
        })
      })
      .catch(() => {
        // Nothing cached, or the slide is not ready. Both are ordinary states and
        // the screen says so on its own; a probe failure is not an error here.
      })

    return () => controller.abort()
    // `inputsKey` covers `uploadId`; both are listed so the dependency is legible.
  }, [inputsKey, uploadId])

  /* --- running ------------------------------------------------------------ */

  const launch = useCallback(
    async (restart: boolean) => {
      if (!uploadId) throw new Error('no slide loaded')

      setRunning(true)
      setCancelling(false)
      setError(null)
      if (restart) setReport(null)

      // A run paints from nothing. Cleared on every start rather than only on a
      // restart, because starting also picks up a pass that is already going, and the
      // server replays that one from the beginning of its feed.
      clearPaint()

      try {
        // Step 7's overlap, and no model named. The overlap is the one parameter
        // the two grids share, so a viewer who picked "no overlap" one screen back
        // gets it here too rather than watching four times the windows go past. Null
        // before step 7 has run, in which case the server falls back to step 7's own
        // default - the value that screen would have used anyway.
        const result = await runTissueType(uploadId, {
          overlap: tileOverlap,
          // Null means "the server's default", which is what every caller did before
          // there was a choice - so not passing one keeps the old behaviour exactly.
          model: modelName ?? undefined,
          branch: branch ?? undefined,
          fov: fieldOfView ?? undefined,
          restart,
          onProgress: (state) => {
            if (!live.current) return
            if (state.paintedCells.length > 0) {
              // Appended one at a time rather than spread: a client rejoining a pass
              // part-way gets thousands of patches in one reply, and spreading that
              // into `push` is an argument list thousands long.
              //
              // The masks go first, so that by the time `paintedCount` moves — which is
              // what makes the canvas draw — every patch it is about to read already has
              // its mask. The other order would let one render pass find a patch whose
              // mask is not there yet and paint it as a flat colour, and the canvas
              // never revisits a patch it has drawn.
              //
              // Defaulted rather than read straight, so a server that does not send the
              // field at all degrades to flat-colour cells instead of taking the whole
              // step down. Absent is not the same as empty here: empty is what every
              // ResNet branch sends and flat colour is the right drawing for it, so the
              // fallback lands on correct behaviour rather than on a blank screen.
              for (const mask of state.paintedMasks ?? []) paintedMasks.current.push(mask)
              for (const value of state.paintedCells) painted.current.push(value)
              setPaintedCount(painted.current.length / 3)
            }
            setRun(state)
          },
        })
        if (!live.current) return

        setReport(result)
        setRun(result.run)
      } catch (cause) {
        // A cancellation is not an error and must not be recorded as one. It is
        // re-thrown unchanged so the pipeline can tell the two apart.
        if (cause instanceof CancelledError) {
          if (live.current) setError(null)
          throw cause
        }
        const message =
          cause instanceof Error ? cause.message : 'could not classify the tissue'
        if (live.current) setError(message)
        throw new Error(message)
      } finally {
        if (live.current) {
          setRunning(false)
          setCancelling(false)
        }
      }
    },
    [uploadId, tileOverlap, modelName, branch, fieldOfView, clearPaint],
  )

  const start = useCallback(() => launch(false), [launch])
  const restart = useCallback(() => launch(true), [launch])

  const cancel = useCallback(async () => {
    if (!uploadId) return
    setCancelling(true)
    try {
      const stopped = await cancelTissueTypeRun(uploadId)
      if (live.current) setRun(stopped)
    } catch {
      // Cancelling something that has already finished is not a failure worth
      // reporting - the outcome the viewer wanted is the state, and they will see it.
      if (live.current) setCancelling(false)
    }
  }, [uploadId])

  /* --- looking at the result ---------------------------------------------- */

  // How many classes this pass emits, from the server rather than from a constant. The
  // finished report is the authority; while a pass runs, its palette is - it is sent
  // with the paint geometry precisely so the live legend and the finished one cannot
  // disagree. Falls back to the trained options' three before either exists.
  const classCount =
    report?.classes.length ?? run?.paint?.colours.length ?? ALL_CLASSES.length

  const allClasses = useMemo(() => everyClass(classCount), [classCount])

  // Reset the filter when the *number* of classes changes, which happens when the
  // viewer switches between a trained option and BEETLE on step 7. Without this the
  // filter kept three ids and a five-class map would be drawn with two of its classes
  // permanently switched off - and nothing on screen would say why.
  //
  // Keyed on the count and not on `visible`, so a viewer's own toggles inside one pass
  // are left alone.
  const builtFor = useRef(classCount)
  useEffect(() => {
    if (builtFor.current === classCount) return
    builtFor.current = classCount
    setVisible(everyClass(classCount))
  }, [classCount])

  const toggleClass = useCallback((id: number) => {
    setVisible((current) =>
      current.includes(id)
        ? current.filter((entry) => entry !== id)
        : [...current, id].sort((a, b) => a - b),
    )
  }, [])

  const showAllClasses = useCallback(
    () => setVisible(everyClass(classCount)),
    [classCount],
  )

  return useMemo(
    () => ({
      capability,
      probing,
      probeError,
      retryProbe,
      report,
      run,
      running,
      error,
      painted: painted.current,
      paintedCount,
      paintedMasks: paintedMasks.current,
      allClasses,
      visible,
      toggleClass,
      showAllClasses,
      start,
      restart,
      cancel,
      cancelling,
      reset: clear,
    }),
    [
      allClasses,
      cancel,
      cancelling,
      capability,
      clear,
      error,
      paintedCount,
      probeError,
      probing,
      report,
      restart,
      retryProbe,
      run,
      running,
      showAllClasses,
      start,
      toggleClass,
      visible,
    ],
  )
}
