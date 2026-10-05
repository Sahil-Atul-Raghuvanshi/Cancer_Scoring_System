import type { PipelineStage } from '@/types/pipeline'

import type { StageStatus } from '../hooks/usePipelineRun'

import './pipeline.css'

/** `input → step → output`, the shape the whole walkthrough repeats. */
export function StageFlow({ stage, status }: { stage: PipelineStage; status: StageStatus }) {
  return (
    <div className="flow">
      <span className="flow__node">{stage.inputLabel}</span>
      <span className="flow__arrow" aria-hidden>
        →
      </span>
      <span className="flow__node">{stage.title.toLowerCase()}</span>
      <span className="flow__arrow" aria-hidden>
        →
      </span>
      <span className={status === 'complete' ? 'flow__node flow__node--out' : 'flow__node'}>
        {status === 'complete' ? stage.outputLabel : 'pending'}
      </span>
    </div>
  )
}
