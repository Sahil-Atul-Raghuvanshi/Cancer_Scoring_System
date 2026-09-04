/**
 * What step 8 shows while it is working, which is a long time.
 *
 * This is the slowest thing in the pipeline by a wide margin — tens of thousands of
 * forward passes, measured at about ten patches a second on four CPU threads — so
 * the progress here has to be real and it has to carry an estimate. A bar with no
 * number beside it is indistinguishable from a hung process at this duration.
 *
 * The estimate is computed from the run's own rate rather than from a constant,
 * because the rate depends on how much of the slide is tissue and on the pyramid the
 * scanner wrote. It appears only once there is enough of a sample to mean anything.
 */

import { Badge } from '@/components/ui/Badge'
import { formatCount } from '@/lib/format'
import type { TissueTypeParams, TissueTypeRun } from '@/types/tissueType'

import './tissueType.css'

const PHASE_LABEL: Record<string, string> = {
  grid: 'Working out which patches to look at',
  classifying: 'Showing each patch to the model',
  rendering: 'Drawing the map',
}

/** Below this the rate is too noisy to turn into a number a viewer should trust. */
const MIN_SAMPLE = 200

function remaining(run: TissueTypeRun): string | null {
  if (!run.startedAt || run.done < MIN_SAMPLE || run.total <= run.done) return null

  const elapsed = (Date.now() - Date.parse(run.startedAt)) / 1000
  if (!Number.isFinite(elapsed) || elapsed <= 0) return null

  const rate = run.done / elapsed
  if (rate <= 0) return null

  const seconds = (run.total - run.done) / rate
  if (seconds < 90) return 'under two minutes left'

  const minutes = Math.round(seconds / 60)
  return minutes >= 60
    ? `about ${(minutes / 60).toFixed(1)} hours left`
    : `about ${minutes} minutes left`
}

export function TissueTypeProgress({
  run,
  params,
}: {
  run: TissueTypeRun | null
  params: TissueTypeParams | null
}) {
  const phase = run?.phase ? PHASE_LABEL[run.phase] : 'Starting up'
  const fraction = run?.progress ?? 0
  const eta = run ? remaining(run) : null

  return (
    <div className="tt-progress">
      <div className="tt-progress__head">
        <Badge tone="accent">running</Badge>
        <span className="tt-progress__phase">{phase}</span>
        <span className="tt-progress__pct mono">{Math.round(fraction * 100)}%</span>
      </div>

      <div className="tt-progress__track">
        <span className="tt-progress__fill" style={{ width: `${fraction * 100}%` }} />
      </div>

      <div className="tt-progress__detail mono">
        {run && run.total > 0
          ? `${formatCount(run.done)} / ${formatCount(run.total)} patches`
          : (run?.message ?? 'queued')}
        {eta && ` · ${eta}`}
        {params &&
          ` · ${params.windowPx} px at ${params.mpp} µm/px · ${(params.overlap * 100).toFixed(0)}% overlap`}
      </div>

      <p className="tt-progress__note">
        Nothing here is simulated &mdash; the bar moves as each patch comes back from
        the model. This is the slowest step in the pipeline: on a machine with no
        graphics card it is genuinely tens of minutes for one slide, because every
        patch is a separate pass through a neural network and there are tens of
        thousands of them. The result is saved, so coming back to this screen later is
        instant rather than another wait.
      </p>

      <p className="tt-progress__note">
        The total is the square count from two screens back, unchanged: these are those
        squares, being shown to the model one at a time.
      </p>
    </div>
  )
}
