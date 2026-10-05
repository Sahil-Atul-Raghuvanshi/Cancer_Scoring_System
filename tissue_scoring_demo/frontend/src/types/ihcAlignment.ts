/** Types for step 10. Mirrors `backend/app/schemas/ihc_alignment.py`. */

export type AlignmentState = 'queued' | 'running' | 'ready' | 'refused' | 'failed'

export interface AlignmentRun {
  heUploadId: string
  ihcUploadId: string
  state: AlignmentState
  message: string | null
  startedAt: string | null
  finishedAt: string | null
  duration: number | null
  error: string | null
}

export interface AlignmentDiagnostics {
  matchedKeypoints: number | null
  residualErrorUm: number | null
  rigidErrorUm: number | null
  originalErrorUm: number | null
  roundTripMedianUm: number | null
  roundTripMaxUm: number | null
  alignmentNmi: number | null
  alignmentNmiUnregistered: number | null
  probePointsOnIhcSlide: number | null
  heTissueMm2: number | null
  ihcTissueMm2: number | null
  tissueAreaRatio: number | null
  heMpp: number | null
  ihcMpp: number | null
  clampedVertices: number | null
  totalVertices: number | null
  reusedRegistration: boolean | null
  seconds: number | null
}

export interface AlignedRegion {
  index: number
  rank: number
  cells: number
  areaMm2: number
  heRings: number[][][]
  ihcRings: number[][][]
}

export interface AlignmentReport {
  heUploadId: string
  ihcUploadId: string
  marker: string | null
  state: AlignmentState
  generatedAt: string
  regions: AlignedRegion[]
  diagnostics: AlignmentDiagnostics

  /**
   * Share of the invasive carcinoma these regions cover, 0-1.
   *
   * The honest headline for this step. OncoStem scan the entire slide and
   * average every field, so whatever is not carried is tumour the score never
   * sees - and a region count says nothing about how much that is.
   */
  areaCoverage: number
  carriedMm2: number
  invasiveMm2: number
  refusalReasons: string[]
  confirmed: boolean
  confirmedAt: string | null
  notes: string[]
}

export interface AlignmentCapability {
  available: boolean
  serviceDir: string
  reason: string | null
}
