/** Types for step 13. Mirrors `backend/app/schemas/compartments.py`. */

export type CompartmentKind = 'membrane' | 'cytoplasm'

export interface CompartmentParams {
  expansionUm: number
  ringUm: number | null
  voronoiConstrained: boolean
}

export interface FieldCompartments {
  regionRank: number
  index: number
  cells: number
  meanMeasuredUm2: number
  meanCellUm2: number
  meanNucleusUm2: number
  contestedPx: number
}

export interface RegionCompartments {
  rank: number
  cells: number
  meanMeasuredUm2: number
  meanCellUm2: number
  meanNucleusUm2: number
  contestedShare: number
  fields: FieldCompartments[]
}

export interface WidthSensitivityPoint {
  widthUm: number
  meanMeasuredUm2: number
  contestedShare: number
}

export interface CompartmentsReport {
  heUploadId: string
  ihcUploadId: string
  marker: string | null
  markerName: string | null
  /** The fork, resolved from the antibody letter. Never a request parameter. */
  compartment: CompartmentKind
  secondMeasure: string | null

  generatedAt: string
  nucleiGeneratedAt: string | null

  params: CompartmentParams
  regions: RegionCompartments[]

  cells: number
  meanMeasuredUm2: number
  meanCellUm2: number

  widthSensitivity: WidthSensitivityPoint[]

  tumourOnly: boolean
  typedCells: number | null

  notes: string[]
}

/**
 * One region's compartment outlines, in the IHC slide's level-0 pixels.
 *
 * Fetched separately from the report for the same reason step 11 keeps its
 * nuclei separate: the report is re-fetched on every drag of the width slider
 * and this is megabytes of vertices.
 *
 * Both bands are stored for every cell, not only the one this antibody uses.
 * The screen draws the used one filled and the other as a dashed outline, which
 * is the only way a reader can see that the fork happened at all - and the fork
 * is what step 13 is about.
 */
export interface CompartmentCellRings {
  id: number
  /** Outer ring first, then holes. A band is an annulus, so it has both. */
  nucleus: number[][][]
  /** The cell body inside the shell. The whole body for a cytoplasmic marker. */
  cytoplasm: number[][][]
  /** The shell at the outer edge. Empty for a cytoplasmic marker, which has none. */
  membrane: number[][][]
  /** What the other kind of antibody would have measured here. May overlap. */
  alternate: number[][][]
}

export interface CompartmentFieldRings {
  index: number
  x: number
  y: number
  span: number
  cells: CompartmentCellRings[]
}

export interface RegionCompartmentRings {
  rank: number
  /** Which of the three regions this antibody actually measures. */
  measured: CompartmentKind
  widthsUm: {
    /** How far the cell is grown from its nucleus. */
    body: number
    /** Thickness of the measured shell. 0 for a cytoplasmic marker. */
    shell: number
    /** How far the other fork would have grown. */
    alternate: number
  }
  fields: CompartmentFieldRings[]
}
