/** The slide-session context and its hook, split out so the provider file
 * exports only a component (react-refresh requires that). */

import { createContext, useContext } from 'react'

import type { UploadPhase } from '@/api/uploads'
import type { SlideReadout, UploadStatus } from '@/types/slide'

export type SlideStage = 'empty' | 'uploading' | 'loaded' | 'error'

export interface SlideSessionValue {
  stage: SlideStage
  uploadId: string | null
  status: UploadStatus | null
  /** Populated by step 1, not by the upload. */
  readout: SlideReadout | null
  progress: number
  phase: UploadPhase | null
  filename: string | null
  error: string | null
  upload: (file: File) => Promise<void>
  /** Step 1: open the slide and read its pyramid. */
  readSlide: () => Promise<void>

  /** The resolution step 1 is asked to work at. */
  targetMpp: number | null
  /** A caller-supplied scale for files that record none. */
  mppOverride: number | null
  /** Re-read the slide with different resolution settings. */
  adjust: (next: { targetMpp?: number | null; mppOverride?: number | null }) => Promise<void>
  /** True while a re-read is in flight, so the view can dim rather than blank. */
  adjusting: boolean

  clear: () => void
}

export const SlideSessionContext = createContext<SlideSessionValue | null>(null)

export function useSlideSession(): SlideSessionValue {
  const value = useContext(SlideSessionContext)
  if (!value) {
    throw new Error('useSlideSession must be used inside a SlideSessionProvider')
  }
  return value
}
