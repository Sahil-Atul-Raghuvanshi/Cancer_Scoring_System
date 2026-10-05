/** Types for step 17. Mirrors `backend/app/services/validation_service.py`. */

export interface MarkerAgreement {
  marker: string
  cases: number
  /** Ours minus theirs, in percentage points. */
  percentBias: number
  percentLimits: number[]
  percentWithinTolerance: number
  percentMaxError: number
  intensityBias: number
  intensityLimits: number[]
  /** Null where a kappa would be undefined - fewer than two bands used. */
  intensityKappa: number | null
  /** The readers' own spread, which is what our error is read against. */
  readerSpread: number | null
}

export interface ValidationReport {
  generatedAt: string
  caseId: string | null
  readerSheet: string
  cutsProvisional: boolean
  /** False means there is nothing to compare against - a result, not an error. */
  available: boolean
  reason: string
  markers: MarkerAgreement[]
  notes: string[]
}
