/** Types for previous cases. Mirrors `backend/app/services/history_service.py`. */

/**
 * Where a run's files currently are.
 *
 * There is one copy and it is in one of the two trees - `data/demo` while it is
 * being worked on, `data/history` once it has been filed. Opening and filing are
 * directory renames, so this is never "both".
 */
export type RunLocation = 'demo' | 'history'

/**
 * How far a marker got.
 *
 * Read off the disk - which step reports exist - rather than from a flag written
 * when something finished, because a flag is wrong after a crash and this is
 * what decides whether the screen offers to replay or to resume.
 */
export type RunState = 'complete' | 'partial' | 'not-started'

export interface MarkerScore {
  percent: number | null
  intensity: number | null
  intensityLabel: string | null
  markerName: string | null
}

export interface HistoryMarker {
  marker: string
  name: string
  ihcUploadId: string | null
  location: RunLocation | null
  state: RunState
  /** Keyed by step number as a string, e.g. `{"11": true}`. */
  steps: Record<string, boolean>
  /** The highest step with a stored report, or 0. */
  lastStep: number
  score: MarkerScore | null
  updatedAt: string | null
}

export interface HistoryCase {
  caseId: string
  casePath: string
  heUploadId: string | null
  hasThumbnail: boolean
  markers: HistoryMarker[]
  completed: number
  started: number
  updatedAt: string
}
