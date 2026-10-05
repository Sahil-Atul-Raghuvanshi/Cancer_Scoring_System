import type { PipelineStage } from '@/types/pipeline'

import './pipeline.css'

const APPROACH_COPY: Record<PipelineStage['approach'], string> = {
  classical: 'Classical image processing — deterministic, inspectable, no training.',
  library: 'Library work — an existing reader does the heavy lifting.',
  pretrained: 'A model someone else already trained; you download the weights.',
  trained: 'Supervised deep learning — the one model you train yourself.',
  logic: 'Pure logic — the published guideline, applied to counts.',
  plumbing: 'Plumbing — an index and a filter, no cleverness required.',
}

function WarningIcon() {
  return (
    <svg className="rule-callout__icon" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M12 8v5M12 16.5v.5M10.3 3.9 2.6 17.4A2 2 0 0 0 4.3 20.4h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/** The written explanation that accompanies a completed step. */
export function StageDetails({ stage }: { stage: PipelineStage }) {
  const blocks = [
    { label: stage.implemented ? 'What it does' : 'What it would do', body: stage.what },
    { label: 'Why it belongs here', body: stage.whyHere },
    { label: 'How it is built', body: `${stage.how} ${APPROACH_COPY[stage.approach]}` },
  ]

  return (
    <>
      <div className="explain">
        {blocks.map((block, index) => (
          <div
            key={block.label}
            className="explain__block"
            style={{ animationDelay: `${index * 110}ms` }}
          >
            <div className="explain__label">{block.label}</div>
            <p className="explain__body">{block.body}</p>
          </div>
        ))}
      </div>

      {stage.rule && (
        <div className="rule-callout" style={{ animationDelay: '340ms' }}>
          <WarningIcon />
          <p className="rule-callout__text">{stage.rule}</p>
        </div>
      )}

      {stage.references.length > 0 && (
        <div className="refs">
          {stage.references.map((reference, index) => (
            <span
              key={reference}
              className="ref-chip anim-rise-sm"
              style={{ animationDelay: `${420 + index * 70}ms` }}
            >
              {reference}
            </span>
          ))}
        </div>
      )}
    </>
  )
}
