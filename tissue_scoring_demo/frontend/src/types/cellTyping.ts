/** Types for step 12. Mirrors `backend/app/schemas/cell_typing.py`. */

export interface TypingRules {
  lymphocyteMaxAreaUm2: number
  lymphocyteMinCircularity: number
  lymphocyteMinDarkness: number
  spindleMinEccentricity: number
  spindleMaxAreaUm2: number
}

export interface TypeCount {
  type: string
  label: string
  count: number
  share: number
  /** `r,g,b`, matching the overlay, so the legend and the picture agree. */
  colour: [number, number, number]
}

export interface RegionTyping {
  rank: number
  areaMm2: number
  counted: number
  counts: TypeCount[]
  tumourPerMm2: number
  medianTumourAreaUm2: number
  medianLymphocyteAreaUm2: number
}

export interface SensitivityPoint {
  value: number
  tumourShare: number
}

export interface SensitivitySweep {
  parameter: string
  baseline: number
  points: SensitivityPoint[]
  /** Largest minus smallest tumour share: how much the threshold decides. */
  swing: number
}

export interface CellTypingReport {
  heUploadId: string
  ihcUploadId: string
  marker: string | null
  generatedAt: string
  nucleiGeneratedAt: string | null

  rules: TypingRules
  regions: RegionTyping[]

  counted: number
  counts: TypeCount[]
  tumourShare: number

  sensitivity: SensitivitySweep[]

  /** Whether the nuclei underneath are good enough for this to mean anything. */
  trustworthy: boolean
  trustReason: string | null
  medianAreaUm2: number

  notes: string[]
}

/** A class per nucleus id, fetched per region for the overlay. */
export interface RegionTypesPayload {
  rank: number
  types: Record<string, number>
}
