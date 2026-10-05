/** Types for step 14. Mirrors `backend/app/schemas/per_cell.py`. */

/**
 * Which second number this marker's cells carry.
 *
 * It travels on the report *and* on every point rather than being derived in the
 * browser from the marker letter. A y-axis label worked out on this side would
 * be a second place for the membrane/cytoplasm fork to be got wrong, and the one
 * place it must not be got wrong is the axis a viewer reads the cells off.
 */
export type SecondMeasure = 'ring_completeness' | 'stained_fraction'

export interface MeasurementParams {
  /** Always "mean". Reported so it cannot change silently between markers. */
  intensityStatistic: string
  compartment: 'membrane' | 'cytoplasm'
  expansionUm: number
  ringUm: number | null
  positivityOd: number
  /** 36 for a membrane marker; null for a cytoplasmic one, which has no ring. */
  ringBins: number | null
  /**
   * The cut on the second number. From the server, so the boundary drawn on the
   * scatter is the one actually being applied rather than one hard-coded here.
   */
  secondMin: number
  cutsVersion: number
  cutsProvisional: boolean
}

export interface CellPoint {
  cellId: number
  regionRank: number
  fieldIndex: number
  x: number
  y: number
  intensityOd: number
  maxOd: number
  second: number
  secondMeasure: SecondMeasure
  pixels: number
  areaUm2: number
  occupiedBins: number | null
}

export interface FieldMeasurement {
  regionRank: number
  index: number
  cells: number
  meanOd: number
  meanSecond: number
}

export interface RegionMeasurement {
  rank: number
  areaMm2: number
  cells: number
  meanOd: number
  medianOd: number
  meanSecond: number
  fields: FieldMeasurement[]
}

export interface ODHistogramBin {
  lower: number
  upper: number
  count: number
}

export interface PerCellReport {
  heUploadId: string
  ihcUploadId: string
  marker: string
  markerName: string
  secondMeasure: SecondMeasure

  generatedAt: string
  nucleiGeneratedAt: string | null

  params: MeasurementParams
  regions: RegionMeasurement[]

  cells: number
  meanOd: number
  medianOd: number
  meanSecond: number

  histogram: ODHistogramBin[]

  /** A sample for the scatter. `cells` is the real count. */
  points: CellPoint[]
  sampled: number

  /** Membrane only: cells too crowded for completeness to mean anything. */
  crowdedCells: number | null

  notes: string[]
}

/**
 * One measured cell, unsampled.
 *
 * The report carries a few thousand `CellPoint`s for the scatter; this is every
 * row, and it is what the slide overlay needs - a cell the sample happened to
 * skip still has to be drawn and still has to answer a click.
 *
 * Structurally a superset of `CellPoint`, which is why the two screens can share
 * one selection: a row can stand in for a point anywhere one is wanted.
 */
export type CellRow = CellPoint
