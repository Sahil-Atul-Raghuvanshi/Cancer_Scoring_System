/** The slide-session context and its hook, split out so the provider file
 * exports only a component (react-refresh requires that). */

import { createContext, useContext } from 'react'

import type { UploadPhase } from '@/api/uploads'
import type { CaseResolution } from '@/types/case'
import type { SlideReadout, UploadStatus } from '@/types/slide'

export type SlideStage = 'empty' | 'uploading' | 'loaded' | 'error'

export interface SlideSessionValue {
  stage: SlideStage
  /** The H&E upload id. Steps 2-9 all key off this one - see `docs/design/five-marker-implementation.md`
   * §2.4: they are marker-blind and ask for "the current slide", so this stays the H&E id even
   * though a second, IHC slide is also loaded. */
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

  /** The chosen biomarker letter (e.g. 'A' for CD44), once a case is loaded. */
  marker: string | null
  /** The case folder a slide pair was loaded from. */
  casePath: string | null
  /**
   * The case's id (e.g. 'CAN_00270'), once a case is loaded. Null for a lone
   * uploaded file, which does not belong to a case.
   *
   * Its own field rather than read back out of `filename`: that one holds the
   * case id after `loadCase` and a real filename after `upload`, so deriving the
   * case id from it would work right up until somebody uploaded a file named
   * after a case.
   */
  caseId: string | null
  /** The IHC slide's upload id - the one the new alignment step (10) warps a mask onto. */
  ihcUploadId: string | null
  /** The IHC slide's own pyramid readout, shown alongside the H&E's in step 1. */
  ihcReadout: SlideReadout | null
  /** Step 0: find what a case folder holds, without loading anything yet. */
  resolveCase: (casePath: string) => Promise<CaseResolution>
  /** Step 0: register the case's H&E slide and the chosen marker's IHC slide. */
  loadCase: (casePath: string, marker: string) => Promise<void>
  /** Step 1: open the IHC slide and read its pyramid, alongside the H&E's. */
  readIhcSlide: () => Promise<void>

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
