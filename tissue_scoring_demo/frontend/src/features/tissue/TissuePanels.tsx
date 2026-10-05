/**
 * Panels 1, 2 and 4 - what went in, what the decision is made on, what came out.
 *
 * Shown as a strip rather than a tabbed viewer, because the sequence *is* the
 * explanation: a reader who sees the saturation channel next to the thumbnail
 * understands why saturation is the channel being thresholded without being told.
 * Panel 3, the histogram, sits below because it needs the full width to be
 * legible and it is the only interactive one.
 *
 * The two threshold-dependent panels carry the cut in their URL, so the browser
 * caches each distinct one and dragging back to a value already seen is free.
 */

import { tissuePanelUrl } from '@/api/tissue'
import type { TissuePanelName, TissueReport } from '@/types/tissue'

interface TissuePanelsProps {
  report: TissueReport
  /** True while a re-threshold is in flight, so the stale panels can dim. */
  refining: boolean
}

const CAPTIONS: Array<{
  name: TissuePanelName
  step: string
  title: string
  caption: string
}> = [
  {
    name: 'thumbnail',
    step: '1',
    title: 'The slide',
    caption: 'Most of it is empty glass. That is what we want to remove.',
  },
  {
    name: 'saturation',
    step: '2',
    title: 'How much colour',
    caption:
      'Glass is bright but has no colour, so it turns black here. Pale tissue still has colour, so it stays visible.',
  },
  {
    name: 'overlay',
    step: '4',
    title: 'The tissue we keep',
    caption: 'Highlighted areas go to the next step. Dimmed areas are dropped.',
  },
]

export function TissuePanels({ report, refining }: TissuePanelsProps) {
  const { uploadId, threshold, params } = report

  // The committed cut, not the slider's draft: these images are the ones the
  // server rendered, and labelling them with a threshold they were not built at
  // would be the one dishonesty this whole screen exists to avoid.
  const options = {
    threshold: threshold.source === 'manual' ? threshold.value : null,
    targetMpp: params.targetMpp,
  }

  return (
    <div className={refining ? 'tissue-strip tissue-strip--stale' : 'tissue-strip'}>
      {CAPTIONS.map((panel) => (
        <figure className="tissue-strip__item" key={panel.name}>
          <div className="tissue-strip__frame">
            {/* Not lazy: the strip is the step's primary content, and the
                sequence only reads as an argument if all three arrive together. */}
            <img src={tissuePanelUrl(uploadId, panel.name, options)} alt={panel.caption} />
          </div>
          <figcaption>
            <span className="tissue-strip__step mono">{panel.step}</span>
            <strong>{panel.title}</strong>
            <span className="tissue-strip__caption">{panel.caption}</span>
          </figcaption>
        </figure>
      ))}
    </div>
  )
}
