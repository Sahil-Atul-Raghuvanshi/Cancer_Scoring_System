/**
 * What step 2 shows while it is working.
 *
 * A whole-slide artefact pass on a CPU is minutes, not milliseconds, so the
 * progress here is real: it counts the patches the server has actually pushed
 * through the model, reported by the run itself. Naming the current pass also
 * makes the two-model design visible at the one moment the viewer is watching
 * most closely.
 */

import { Badge } from '@/components/ui/Badge'
import { formatCount } from '@/lib/format'
import type { QCParams, QCRun } from '@/types/qc'

import './qc.css'

const PHASE_LABEL: Record<string, string> = {
  tissue: 'Pass 1 — finding tissue at 10 µm/px',
  artefacts: 'Pass 2 — segmenting artefacts',
  rendering: 'Rendering the overlays',
  summarising: 'Counting pixels',
}

export function QCProgress({ run, params }: { run: QCRun | null; params: QCParams | null }) {
  const phase = run?.phase ? PHASE_LABEL[run.phase] : 'Starting up'
  const fraction = run?.progress ?? 0

  return (
    <div className="qc-progress">
      <div className="qc-progress__head">
        <Badge tone="accent">running</Badge>
        <span className="qc-progress__phase">{phase}</span>
        <span className="qc-progress__pct mono">{Math.round(fraction * 100)}%</span>
      </div>

      <div className="qc-progress__track">
        <span className="qc-progress__fill" style={{ width: `${fraction * 100}%` }} />
      </div>

      <div className="qc-progress__detail mono">
        {run && run.total > 0
          ? `${formatCount(run.done)} / ${formatCount(run.total)} patches`
          : (run?.message ?? 'queued')}
        {params && ` · ${params.modelLabel} model at ${params.modelMpp} µm/px`}
      </div>

      <p className="qc-progress__note">
        Nothing here is simulated — the bar moves as each 512 px patch comes back from the model.
        On a machine with no GPU this is genuinely several minutes; the tissue pass first excuses
        every patch that is only glass, which is what keeps it to minutes rather than hours.
      </p>
    </div>
  )
}
