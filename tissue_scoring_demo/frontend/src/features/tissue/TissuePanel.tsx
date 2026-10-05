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
          Looking at the slide and working out where the tissue is. The first run reopens the
          scan, so it is the slowest one.
        </p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="placeholder placeholder--offline">
        <span className="placeholder__badge mono">this step could not run</span>
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
        This step looks at how much colour each part of the slide has. Glass is bright but
        colourless; tissue has colour. It then tidies up the result and shows you each move
        it made.
      </p>
    </div>
  )
}
