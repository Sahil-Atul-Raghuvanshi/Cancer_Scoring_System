/**
 * Step 3's work area, choosing between its three states.
 *
 *   running   the first run, which reopens the slide
 *   error     it could not run, and why - never a blank panel
 *   report    the result
 *
 * There is no capability check here, unlike step 2. Step 3 needs no checkpoints
 * and no torch; the only thing it can be missing is a physical scale on the
 * slide, and that arrives as an error from the server with the fix in it.
 */

import { Button } from '@/components/ui/Button'

import { TissueView } from './TissueView'
import type { TissueMaskState } from './useTissueMask'

import './tissue.css'

interface TissuePanelProps {
  tissue: TissueMaskState
  /** True while the pipeline considers this step to be running. */
  running: boolean
}

export function TissuePanel({ tissue, running }: TissuePanelProps) {
  const { report, loading, refining, error, draft, committed, manual } = tissue

  if (report && draft !== null && committed !== null) {
    return (
      <TissueView
        report={report}
        draft={draft}
        committed={committed}
        manual={manual}
        refining={refining}
        onDraft={tissue.setDraft}
        onResetToAutomatic={tissue.resetToAutomatic}
      />
    )
  }

  if (loading || running) {
    return (
      <div className="placeholder placeholder--running">
        <span className="placeholder__spinner" />
        <p className="placeholder__text">
          Reading the slide at the working resolution, taking its saturation channel, and
          letting Otsu pick the cut. The first run reopens the scan, so it is the slow one.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">step 03 could not run</span>
        <p className="placeholder__text">{error}</p>
        <Button variant="secondary" onClick={() => void tissue.start()}>
          Try again
        </Button>
      </div>
    )
  }

  return (
    <div className="placeholder">
      <p className="placeholder__text">
        Nothing is precomputed. Running this step downsamples your slide, converts it to
        HSV, thresholds the saturation channel with Otsu&rsquo;s method and cleans the result
        up with morphology — then shows you every one of those moves.
      </p>
    </div>
  )
}
