/** Types for step 16. Mirrors `backend/app/schemas/scores.py`. */

export interface RegionScore {
  rank: number
  areaMm2: number
  cells: number
  positiveCells: number
  percentRaw: number
  intensityRaw: number
  /** The sample behind it (P-06). Absent on reports from before. */
  fields?: number
  sampledMm2?: number
  estimatedCells?: number
  weightShare?: number
}

export interface HeterogeneityTile {
  regionRank: number
  fieldIndex: number
  x: number
  y: number
  span: number
  cells: number
  percent: number
}

export interface CascadeStep {
  label: string
  expression: string
  value: string
}

/**
 * One antibody's numbers.
 *
 * `percent` and `intensity` are the contract - a client that renders only those
 * two has rendered the whole deliverable. Everything else on this interface is
 * context, a check, or the field's standard vocabulary.
 */
export interface MarkerScore {
  marker: string
  markerName: string
  percent: number
  intensity: number
  intensityLabel: string

  percentRaw: number
  intensityRaw: number
  percentPooled: number
  intensityPooled: number

  compartment: string
  secondMeasure: string

  cells: number
  positiveCells: number
  binCounts: number[]
  binShares: number[]

  /** Reference vocabulary. Never the deliverable. */
  hScore: number
  allredProportion: number
  allredIntensity: number
  allredTotal: number
  her2Call: string | null
  her2Note: string

  percentAreaWeighted: number
  percentPlainMean: number
  averagingGapPoints: number
  /** "estimated_cells", "pooled", or "area_weighted" on reports from before P-06. */
  averagingUsed: string
  /** 95 % interval on `percentRaw` (P-06). Null on reports from before. */
  percentCiLow?: number | null
  percentCiHigh?: number | null
  singleFieldWeight?: number
  unsampledRegions?: number
  unsampledAreaMm2?: number
  totalAreaMm2?: number
  partialRule: string
  percentByPartialRule: Record<string, number>

  regions: RegionScore[]
  heterogeneity: HeterogeneityTile[]
  cascade: CascadeStep[]

  odCuts: number[]
  secondMin: number
  cutsProvisional: boolean

  /** What has to be known before these two numbers are used. */
  caveats: string[]

  /**
   * What the two numbers are: `measured`, `provisional` (a check a person should have
   * made was not, or a calibration is unfitted) or `not_a_measurement`. Caveats explain;
   * this decides. With the headings that decided it.
   */
  status?: ScoreStatus
  statusReasons?: string[]

  /**
   * Which cells `percent` and `intensity` count: every cell in the region ("all") or
   * only those step 14 called tumour. With "all", the tumour-only figures sit beside
   * them as a sensitivity (P-04).
   */
  population?: 'all' | 'tumour'
  percentTumourOnly?: number | null
  intensityTumourOnly?: number | null
  cellsTumourOnly?: number | null
}

export type ScoreStatus = 'measured' | 'provisional' | 'not_a_measurement'

export interface ScoreReport {
  heUploadId: string
  ihcUploadId: string
  generatedAt: string
  measuredAt: string | null
  score: MarkerScore
  notes: string[]
  /** What the score was made with (P-15). Null on a report from before stamping. */
  provenance?: Record<string, unknown> | null
}

export interface CaseScoreRow {
  marker: string
  markerName: string
  percent: number | null
  intensity: number | null
  intensityLabel: string
  cells: number
  state: string
  /** 95 % interval on the percentage (P-06). */
  percentCiLow?: number | null
  percentCiHigh?: number | null
  detail: string
  /** The score's own status when scored; null otherwise. */
  status?: ScoreStatus | null
}

export interface CaseScoreReport {
  caseId: string
  generatedAt: string
  rows: CaseScoreRow[]
  complete: boolean
  notes: string[]
}
