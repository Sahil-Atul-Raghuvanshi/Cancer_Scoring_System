/** Types for case-folder resolution. Mirrors `backend/app/schemas/case.py`. */

export interface CaseResolution {
  caseId: string
  casePath: string
  /** Marker letter (or 'HE') -> absolute slide path. */
  found: Record<string, string>
  /** Marker letters (or 'HE') with no file found. */
  missing: string[]
}

export interface CaseSession {
  caseId: string
  marker: string
  casePath: string
  heUploadId: string
  ihcUploadId: string
}
