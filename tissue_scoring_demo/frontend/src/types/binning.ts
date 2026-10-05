/** Types for step 15. Mirrors `backend/app/schemas/binning.py`. */

/**
 * One of the four per-cell levels.
 *
 * Deliberately a different type from the reported band below. They run on
 * different scales - 0/1+/2+/3+ against 0-2 - and are read by different people,
 * and a single type spanning both is how the two get conflated.
 */
export interface BinCount {
  bin: number
  label: string
  count: number
  share: number
  odFrom: number
  odTo: number | null
}

/** One row of the antibody's optical-density-to-band table. */
export interface BandTableRow {
  odBelow: number | null
  band: number
  label: string
}

export interface CutLine {
  od: number
  label: string
  separates: string
}

export interface SchemeComparison {
  scheme: string
  label: string
  odCuts: number[]
  counts: number[]
  positiveShare: number
  deltaPoints: number
  note: string
}

export interface BinningParams {
  marker: string
  markerName: string
  /** Always "absolute" - the whole argument of this step. */
  scheme: string
  odCuts: number[]
  secondMeasure: string
  secondMin: number
  partialRule: string
  cutsVersion: number
  cutsProvisional: boolean
  cutsSource: string
}

export interface ODHistogramBin {
  lower: number
  upper: number
  count: number
}

export interface BinningReport {
  heUploadId: string
  ihcUploadId: string
  marker: string
  markerName: string

  generatedAt: string
  measuredAt: string | null

  params: BinningParams
  cells: number
  bins: BinCount[]

  overOdCut: number
  positiveCells: number
  positiveShare: number

  histogram: ODHistogramBin[]
  cutLines: CutLine[]
  /** Where the valley between the two humps is, or null if there is only one. */
  valleyOd: number | null

  bandTable: BandTableRow[]
  comparisons: SchemeComparison[]

  notes: string[]
}

/**
 * One cell's level and verdict, computed on the server.
 *
 * The bin is one `searchsorted` and the cuts are already on the report, so the
 * browser could work this out - and deliberately does not. Positivity is two
 * conditions plus a partial-staining rule that is an open question with three
 * implemented answers, and a second copy of that rule here would be a second
 * place for it to be decided.
 */
export interface BinnedCell {
  cellId: number
  regionRank: number
  fieldIndex: number
  intensityOd: number
  second: number
  /** 0, 1, 2 or 3. */
  bin: number
  label: string
  positive: boolean
  /** 1, 0.5 or 0 - what this cell contributes under the partial rule. */
  weight: number
}
