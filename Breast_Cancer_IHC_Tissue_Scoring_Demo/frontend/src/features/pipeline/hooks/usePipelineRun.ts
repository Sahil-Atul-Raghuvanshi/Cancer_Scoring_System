import { useCallback, useMemo, useRef, useState } from 'react'

import { CancelledError } from '@/api/client'
import type { PipelineStage } from '@/types/pipeline'

export type StageStatus =
  | 'locked'
  | 'ready'
  | 'running'
  | 'complete'
  /** Walked past without running. Distinct from complete - see `skip`. */
  | 'skipped'
  | 'error'

interface PipelineRun {
  /** Index into `stages` of the step currently on screen. */
  current: number
  stage: PipelineStage | undefined
  status: StageStatus
  error: string | null
  completed: Set<number>
  /** Steps the viewer chose not to run. A subset of `completed`. */
  skipped: Set<number>
  /** Runs the step's real work; resolves when it finishes. */
  run: (task?: () => Promise<void>) => Promise<void>
  /** Steps past a stage that has nothing to run, in one move. */
  acknowledge: () => void
  /** Steps past a stage that *could* have run, recording that it did not. */
  skip: () => void
  /** Every stage has been through the pipeline. */
  finished: boolean
  next: () => void
  previous: () => void
  goTo: (index: number) => void
  reset: () => void
  statusOf: (index: number) => StageStatus
  /**
   * Whether this step has already been offered a start on this slide.
   *
   * The page auto-starts a step on arrival, and that has to happen **once**: a step
   * that fails would otherwise be retried by the same effect that started it, for
   * ever, hammering the server with the request that just failed.
   */
  attempted: (index: number) => boolean
  markAttempted: (index: number) => void
}

/**
 * Drives the click-through.
 *
 * A step is *ready*, then *running* while its work is actually in flight, then
 * *complete*. There is no timer here: the running state lasts exactly as long as the
 * real task does, so a step that takes 40 ms shows 40 ms of work and a step that
 * reads a two-gigabyte file shows all of it.
 *
 * **Three states rather than two, and the third is the point.** A step can be
 * *complete* (it ran and produced a result) or *skipped* (the viewer chose to walk
 * past work it could have done). Quality control is the case that matters: a slide
 * whose artefacts were segmented and a slide whose artefacts were never looked at
 * produce different tile counts, different tissue areas and different confidence in
 * everything downstream, and every later step's report already says which happened.
 * Recording a skip as a completion would be the one place in this app that blurs
 * them.
 *
 * **A cancellation is neither.** It leaves the step exactly where it was - ready to
 * run - and records no error, because stopping a run is a decision the viewer made
 * rather than a fault to report.
 */
export function usePipelineRun(stages: PipelineStage[]): PipelineRun {
  const [current, setCurrent] = useState(0)
  const [completed, setCompleted] = useState<Set<number>>(() => new Set())
  const [skipped, setSkipped] = useState<Set<number>>(() => new Set())
  const [running, setRunning] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Which steps have been offered an automatic start. A ref rather than state
  // because nothing renders from it and a re-render on every arrival would be a
  // second reason for the auto-start effect to fire.
  const attempts = useRef<Set<number>>(new Set())

  const stage = stages[current]

  const run = useCallback(
    async (task?: () => Promise<void>) => {
      if (!stages[current] || running !== null || completed.has(current)) return

      const index = current
      setRunning(index)
      setError(null)

      try {
        await task?.()
        setCompleted((previous) => new Set(previous).add(index))
      } catch (cause) {
        // Stopped on purpose: back to ready, no error, and the step stays available
        // to start again. Recording this as a failure would make the screen
        // apologise for what the viewer just asked for.
        if (cause instanceof CancelledError) return
        setError(cause instanceof Error ? cause.message : 'this step failed')
      } finally {
        setRunning(null)
      }
    },
    [completed, current, running, stages],
  )

  /**
   * The steps that are documented but not built have no work to do, so the only
   * thing that can happen on one is walking past it. Marking done and moving on has
   * to happen in a single pass: `goTo` reads `completed`, and two separate updates
   * would leave it reading the pre-update set.
   */
  const acknowledge = useCallback(() => {
    if (running !== null) return
    setError(null)
    setCompleted((previous) => new Set(previous).add(current))
    setCurrent((index) => Math.min(index + 1, stages.length - 1))
  }, [current, running, stages.length])

  /**
   * Walk past a step that *could* have run, and record that it did not.
   *
   * Only quality control offers this today, and it is offered because the pipeline
   * genuinely works without it - every later step reads its artefact map as optional
   * and says on screen when it is absent. What is not acceptable is a skip that
   * looks like a run, so this marks the step `skipped` as well as complete and the
   * rail says so.
   */
  const skip = useCallback(() => {
    if (running !== null) return
    setError(null)
    setSkipped((previous) => new Set(previous).add(current))
    setCompleted((previous) => new Set(previous).add(current))
    setCurrent((index) => Math.min(index + 1, stages.length - 1))
  }, [current, running, stages.length])

  const statusOf = useCallback(
    (index: number): StageStatus => {
      if (running === index) return 'running'
      if (skipped.has(index)) return 'skipped'
      if (completed.has(index)) return 'complete'
      if (error !== null && index === current) return 'error'
      // A step unlocks once every step before it has completed.
      const unlocked = index === 0 || completed.has(index - 1)
      return unlocked ? 'ready' : 'locked'
    },
    [completed, current, error, running, skipped],
  )

  const goTo = useCallback(
    (index: number) => {
      if (index < 0 || index >= stages.length || running !== null) return
      if (index > 0 && !completed.has(index - 1)) return
      setError(null)
      setCurrent(index)
    },
    [completed, running, stages.length],
  )

  const next = useCallback(() => goTo(current + 1), [current, goTo])
  const previous = useCallback(() => goTo(current - 1), [current, goTo])

  const reset = useCallback(() => {
    setRunning(null)
    setCompleted(new Set())
    setSkipped(new Set())
    setError(null)
    setCurrent(0)
    attempts.current = new Set()
  }, [])

  const attempted = useCallback((index: number) => attempts.current.has(index), [])
  const markAttempted = useCallback((index: number) => {
    attempts.current.add(index)
  }, [])

  const status = statusOf(current)
  const finished = stages.length > 0 && completed.size === stages.length

  return useMemo(
    () => ({
      current,
      stage,
      status,
      error,
      completed,
      skipped,
      finished,
      run,
      acknowledge,
      skip,
      next,
      previous,
      goTo,
      reset,
      statusOf,
      attempted,
      markAttempted,
    }),
    [
      acknowledge,
      attempted,
      completed,
      current,
      error,
      finished,
      goTo,
      markAttempted,
      next,
      previous,
      reset,
      run,
      skip,
      skipped,
      stage,
      status,
      statusOf,
    ],
  )
}
