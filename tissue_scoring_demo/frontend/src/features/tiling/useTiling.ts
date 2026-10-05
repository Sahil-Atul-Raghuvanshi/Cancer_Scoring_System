/**
 * Owns step 7's state and its two choices, which are asked in order.
 *
 * **First: what the model is shown.** Blue stain only, the full colour photograph, or
 * another research group's released network. That decision is new and it is the larger
 * of the two: the options read different things off the same square, so they are
 * different models with different training data, not one model at three settings - and
 * BEETLE is not even the same *shape* of answer, being five classes per pixel against
 * three per window.
 *
 * **Then: how much ground one square covers.** 112, 224, 448 or 672 µm. This used to be
 * the screen's only choice and it is still a real one — a duct wall does not fit in
 * 112 µm and 672 µm sees it whole at a third of the detail.
 *
 * Both are committed to the server rather than merely sent with a request, because
 * step 8, step 9 and the pipeline runner all rebuild step 7's grid without naming one,
 * and what they get has to be what the viewer chose. `start`, `setFieldOfView` and
 * `setOverlap` therefore go through the *same* commit — one write path, and so one
 * invalidation path.
 *
 * Its *input* is step 3's mask, so this hook takes step 3's committed cut as an
 * argument rather than owning a copy, exactly as steps 4, 5 and 6's hooks do. When the
 * viewer goes back and moves that slider, this step's result is stale: a different mask
 * is a different set of tiles.
 *
 * **It also takes step 5's verdict on the dyes, and that argument is load-bearing.**
 * The server offers both colour options - the full-colour head and BEETLE - only where
 * step 5 found a second dye, because neither normalises a stain away, and it
 * reads that verdict out of a per-slide cache step 5 fills when it runs - never
 * computing one itself, because doing so from step 7 would read a tile, step 3's mask
 * and step 4's white point behind a screen that promises to be free, and would evict the
 * memo steps 5 and 6 share. That is the right division, and it leaves this hook holding
 * the ordering problem: the branch payload is fetched as soon as there is a slide, which
 * is before step 5 has run in every walkthrough, and it used to be fetched exactly once.
 * So an H&E slide arrived at step 7 with its H&E option disabled and the reason reading
 * "run step 5 and come back" - to a reader who had just run step 5. Taking the verdict
 * as an argument makes it a dependency of that fetch, so those options enable themselves
 * as soon as the measurement exists rather than on a reload.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { commitTilingSelection, fetchTiling, fetchTilingBranches } from '@/api/tiling'
import type { TilingBranchName, TilingBranches, TilingReport } from '@/types/tiling'

/** The overlap settings the screen offers. Tiling runs edge to edge, so: one. */
export const OVERLAP_CHOICES = [0] as const

/**
 * The three options, in the order the screen offers them.
 *
 * All three now run. `beetle` is last because it is the odd one out rather than because
 * it is unbuilt: the first two are checkpoints of ours that answer per window in three
 * classes, and it is another group's released network answering per pixel in five. The
 * server decides availability per option and per slide — both colour branches need step
 * 5 to have found a second dye, and `beetle` needs its 1.9 GB archive on disk as well —
 * and this list only fixes the order and gives each one a label.
 */
export const BRANCH_CHOICES = ['h_channel', 'he', 'beetle'] as const

/**
 * The fields of view the screen offers, in microns, in the order it offers them.
 *
 * Mirrors `FIELDS_OF_VIEW` in `backend/app/services/tiling_service.py`, and the server
 * is the authority: it refuses anything not on its own list rather than snapping to the
 * nearest, so a value here the server does not know is a visible error rather than a
 * silent substitution. The branch payload's `fieldsOfView` carries the live answer,
 * including which of them has a checkpoint *on that branch*; this constant is only the
 * order and the default.
 */
export const FIELD_OF_VIEW_CHOICES = [112, 224, 448, 672] as const

/** Which of the screen's two questions is open. Derived, never stored. */
/**
 * The one configuration this pipeline runs, and the reason the pickers are gone.
 *
 * Step 7 used to ask two questions - which branch, and at what field of view - and
 * eight published heads answered them. Both questions are now settled, so the screen
 * states the answer instead of collecting it.
 *
 * `he` because the h_channel branch's whole justification was serving all six slides
 * of a case from one model, and measured it does not: 0% invasive on all five
 * immunostained markers, at 0.96-0.98 confidence. The region is found on the H&E and
 * registered across (step 10), so the generality is generality nothing uses - and on
 * the H&E, eosin is half the evidence, because a duct wall and a collagen band are the
 * same shade of nothing in the H channel.
 *
 * `224` because it is the widest field that still keeps a large tile count. The
 * trade is real and measured in `RESULTS_224_VS_448.md`: a wide window sees
 * architecture but inflates area - the 448 um head reported 77.5 mm2 of invasive
 * against the 112 um head's 42.0 mm2 over identical tissue - so 224 um is the
 * compromise, not a free choice.
 *
 * The pickers themselves are still on disk, unreferenced. Re-offering a choice is
 * putting them back, not writing them again.
 */
export const COMMITTED_BRANCH = 'he' as TilingBranchName
export const COMMITTED_FOV_UM = 224

export type TilingLevel = 'branch' | 'fov' | 'report'

export interface TilingState {
  /** What this slide can be shown as. Fetched on mount; null until it lands. */
  branches: TilingBranches | null
  /** True while that first, cheap request is in flight. */
  probing: boolean

  report: TilingReport | null
  /** True while the first run for this slide is in flight. */
  loading: boolean
  /** True while a re-run at a different setting is in flight and a report is up. */
  refining: boolean
  error: string | null

  /** The chosen option, or null while the branch question is still open. */
  branch: TilingBranchName | null
  /** The field of view chosen but not yet run, or null. */
  pendingFov: number | null
  /** Which question the screen should be showing. */
  level: TilingLevel

  /** The overlap the report on screen was built at, or null before the first run. */
  overlap: number | null
  /** The field of view the report on screen was built at, or null before the first run. */
  fieldOfView: number | null
  /**
   * The checkpoint step 8 will run, or null when none is published at this choice. Null
   * is not an error — step 7's own numbers are still real, because the grid was laid on
   * the geometry that head *will* use — but it does mean step 8 cannot start, and the
   * screen says so rather than letting it fail later.
   */
  model: string | null

  /** Choose an option. Clears the field of view and any report under it. */
  chooseBranch: (branch: TilingBranchName) => void
  /** Choose a field of view, commit both, and run. */
  chooseFieldOfView: (um: number) => void
  /** Back to the field-of-view question, keeping the branch. */
  backToFov: () => void
  /** Back to the branch question. */
  backToBranch: () => void

  /** Rebuild the index at a different overlap. */
  setOverlap: (overlap: number) => void
  /** Rebuild at a different field of view. Changes which model step 8 runs. */
  setFieldOfView: (um: number) => void
  /** Run step 7 at whatever is chosen. Rejects on failure so the pipeline can mark it. */
  start: () => Promise<void>
  reset: () => void
}

export function useTiling(
  uploadId: string | null,
  /** Step 3's committed cut, or null to let its own rule choose. */
  tissueThreshold: number | null,
  /**
   * Step 5's verdict on which dyes this section carries - `he`, `haematoxylin_dab`,
   * `single_stain`, `unknown` - or null before step 5 has run. Not used to decide
   * anything here: the server is the authority on what is offered, and this is only what
   * tells this hook that its answer may have changed. See the note at the top.
   */
  stainingVerdict: string | null = null,
): TilingState {
  const [branches, setBranches] = useState<TilingBranches | null>(null)
  const [probing, setProbing] = useState(false)
  const [report, setReport] = useState<TilingReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [refining, setRefining] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Seeded, not null: there is no question to ask, so the step arrives already
  // knowing what it runs and `start()` is immediately valid.
  const [branch, setBranchValue] = useState<TilingBranchName | null>(COMMITTED_BRANCH)
  const [pendingFov, setPendingFov] = useState<number | null>(COMMITTED_FOV_UM)
  const [overlap, setOverlapValue] = useState<number | null>(null)
  const [fieldOfView, setFieldOfViewValue] = useState<number | null>(null)

  // Guards a response arriving after the viewer has left the step.
  const live = useRef(true)
  useEffect(() => {
    live.current = true
    return () => {
      live.current = false
    }
  }, [])

  // Only the newest request may write to state, so a response that arrives after
  // a later one has already landed cannot overwrite it.
  const inFlight = useRef(0)

  // Clearing throws away the *result*, never the configuration: a new slide or a
  // moved threshold means these counts are stale, not that the pipeline has stopped
  // knowing which model it runs.
  const clear = useCallback(() => {
    setReport(null)
    setError(null)
    setBranchValue(COMMITTED_BRANCH)
    setPendingFov(COMMITTED_FOV_UM)
    setOverlapValue(null)
    setFieldOfViewValue(null)
    setRefining(false)
  }, [])

  useEffect(() => {
    clear()
  }, [clear, uploadId])

  // Step 3's cut changed, so the mask did, so every tile here did. Clearing
  // rather than quietly refetching: the counts on screen belong to the previous
  // mask, and leaving them up beside a new one is the mismatch every earlier
  // step's screen is built to avoid.
  useEffect(() => {
    clear()
  }, [clear, tissueThreshold])

  // What this slide can be shown as. Cheap — manifests plus step 5's cached verdict —
  // and fetched as soon as there is a slide, so the branch question can be asked
  // without the viewer waiting on anything.
  //
  // Re-fetched when step 5's verdict changes, which is what keeps "cheap and early"
  // from meaning "wrong". The first fetch happens before step 5 has run, so it gets
  // `unknown` and the H&E option comes back disabled; the refetch is how that option
  // turns on. Only `branches` is written here — not the committed branch, the field of
  // view or the report — so a viewer who has already chosen something is not moved off
  // it by a verdict landing behind them.
  useEffect(() => {
    if (!uploadId) {
      setBranches(null)
      return
    }

    const controller = new AbortController()
    setProbing(true)
    fetchTilingBranches(uploadId, controller.signal)
      .then((result) => {
        if (live.current) setBranches(result)
      })
      .catch((cause: unknown) => {
        if (!live.current || controller.signal.aborted) return
        setError(
          cause instanceof Error
            ? `could not read what this slide can be shown as: ${cause.message}`
            : 'could not read what this slide can be shown as',
        )
      })
      .finally(() => {
        if (live.current) setProbing(false)
      })

    return () => controller.abort()
  }, [stainingVerdict, uploadId])

  /**
   * Commit a choice and read back the grid it produces.
   *
   * Both settings are sent every time rather than only the one that moved. A request
   * naming just the new field of view would let the server fall back to its own default
   * overlap, so changing the field of view could silently change the overlap too — and
   * the square count on screen would move for two reasons at once with only one of them
   * visible.
   */
  const commit = useCallback(
    (next: { branch?: TilingBranchName; fov?: number; overlap?: number }) => {
      if (!uploadId) return

      const nextBranch = next.branch ?? branch
      const nextFov = next.fov ?? pendingFov ?? fieldOfView
      const nextOverlap = next.overlap ?? overlap ?? undefined
      if (!nextBranch || nextFov === null || nextFov === undefined) return

      const ticket = ++inFlight.current
      if (next.overlap !== undefined) setOverlapValue(next.overlap)
      if (next.fov !== undefined) setPendingFov(next.fov)
      setRefining(report !== null)
      if (report === null) setLoading(true)

      commitTilingSelection(uploadId, {
        branch: nextBranch,
        fov: nextFov,
        overlap: nextOverlap,
        threshold: tissueThreshold,
      })
        .then(() =>
          fetchTiling(uploadId, {
            threshold: tissueThreshold,
            overlap: nextOverlap,
            fov: nextFov,
            branch: nextBranch,
          }),
        )
        .then((result) => {
          if (!live.current || ticket !== inFlight.current) return
          setReport(result)
          // Snap onto what the server actually used, so the screen describes the
          // run that happened rather than the one that was asked for.
          setOverlapValue(result.params.overlap)
          setFieldOfViewValue(result.params.fieldOfViewUm)
          setBranchValue(result.params.branch)
          setError(null)
        })
        .catch((cause: unknown) => {
          if (!live.current || ticket !== inFlight.current) return
          setError(
            cause instanceof Error
              ? `could not build a tile index: ${cause.message}`
              : 'could not build a tile index',
          )
        })
        .finally(() => {
          if (live.current && ticket === inFlight.current) {
            setRefining(false)
            setLoading(false)
          }
        })
    },
    [branch, fieldOfView, overlap, pendingFov, report, tissueThreshold, uploadId],
  )

  const chooseBranch = useCallback((next: TilingBranchName) => {
    // The field of view and any report under it belong to the previous option, so
    // they go. Availability differs per branch, so a scale that was chosen here may
    // not even be offered there.
    setBranchValue(next)
    setPendingFov(null)
    setReport(null)
    setFieldOfViewValue(null)
    setError(null)
  }, [])

  const chooseFieldOfView = useCallback(
    (um: number) => commit({ fov: um }),
    [commit],
  )

  const backToFov = useCallback(() => {
    setReport(null)
    setFieldOfViewValue(null)
    setError(null)
  }, [])

  const backToBranch = useCallback(() => {
    setBranchValue(null)
    setPendingFov(null)
    setReport(null)
    setFieldOfViewValue(null)
    setError(null)
  }, [])

  const start = useCallback(async () => {
    if (!uploadId) throw new Error('no slide loaded')
    if (!branch || pendingFov === null) {
      // The screen asks before it runs, so this is only reachable if something else
      // launched the step. Saying so beats laying a grid nobody chose.
      throw new Error('choose an approach and a field of view first')
    }

    setLoading(true)
    setError(null)

    try {
      await commitTilingSelection(uploadId, {
        branch,
        fov: pendingFov,
        overlap: overlap ?? undefined,
        threshold: tissueThreshold,
      })
      const result = await fetchTiling(uploadId, {
        threshold: tissueThreshold,
        overlap: overlap ?? undefined,
        fov: pendingFov,
        branch,
      })
      if (!live.current) return

      setReport(result)
      setOverlapValue(result.params.overlap)
      setFieldOfViewValue(result.params.fieldOfViewUm)
      setBranchValue(result.params.branch)
    } catch (cause) {
      const message =
        cause instanceof Error ? cause.message : 'could not build a tile index'
      if (live.current) setError(message)
      throw new Error(message)
    } finally {
      if (live.current) setLoading(false)
    }
  }, [branch, overlap, pendingFov, tissueThreshold, uploadId])

  const setOverlap = useCallback(
    (next: number) => commit({ overlap: next }),
    [commit],
  )

  const setFieldOfView = useCallback((next: number) => commit({ fov: next }), [commit])

  const level: TilingLevel = report !== null ? 'report' : branch ? 'fov' : 'branch'

  return useMemo(
    () => ({
      branches,
      probing,
      report,
      loading,
      refining,
      error,
      branch,
      pendingFov,
      level,
      overlap,
      fieldOfView,
      model: report?.params.model ?? null,
      chooseBranch,
      chooseFieldOfView,
      backToFov,
      backToBranch,
      setOverlap,
      setFieldOfView,
      start,
      reset: clear,
    }),
    [
      backToBranch,
      backToFov,
      branch,
      branches,
      chooseBranch,
      chooseFieldOfView,
      clear,
      error,
      fieldOfView,
      level,
      loading,
      overlap,
      pendingFov,
      probing,
      refining,
      report,
      setFieldOfView,
      setOverlap,
      start,
    ],
  )
}
