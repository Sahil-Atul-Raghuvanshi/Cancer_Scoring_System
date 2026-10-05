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
  tissue: 'Step 1 of 2 — finding the tissue',
  artefacts: 'Step 2 of 2 — finding the problem areas',
  rendering: 'Drawing the maps',
  summarising: 'Adding up the areas',
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
        {params && ` · ${params.modelLabel} model`}
      </div>

      <p className="qc-progress__note">
        The bar moves as each patch is checked. On a computer without a graphics card this takes
        a few minutes. Empty glass is skipped, which keeps it to minutes rather than hours.
      </p>
    </div>
  )
}
