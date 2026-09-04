/**
 * Types for step 7 — tiling. Mirrors `backend/app/schemas/tiling.py`.
 *
 * The funnel is the thing this step exists to report: how many tiles the grid
 * holds, how many are on tissue, and how many of those quality control left
 * alone. Counts rather than shares, because "21,609 became 5,070" is the compute
 * bill for the region model and "77% dropped" is a statistic.
 *
 * No tile pixels come down the wire. What this step produces is addresses, and
 * the pixels a tile resolves to are step 6's haematoxylin channel, computed on
 * demand — one answer to "what does the model see", kept in one place.
 */

export type TilingPanelName = 'grid' | 'sample'
export type TileRejection = 'tissue' | 'clean'

export interface TilingParams {
  targetMpp: number
  tileSize: number
  /** The physical field of view, in microns. The number that has to fit an acinus. */
  tileUm: number
  /** Fraction of its own extent a tile shares with its neighbour. */
  overlap: number
  span: number
  stride: number

  minTissueShare: number
  minCleanShare: number

  /** Resolution of step 3's mask, which is what the two shares were measured on. */
  maskMpp: number
  /** How coarse that makes each share — one mask pixel as a fraction of a tile. */
  shareQuantisation: number

  tissueThreshold: number
  tissueThresholdSource: 'otsu' | 'triangle' | 'manual'
  qcGated: boolean
  qcSource: 'grandqc' | 'otsu' | 'unknown' | null
}

export interface TilingFunnel {
  /** Tiles in the grid over the whole canvas, glass included. */
  every: number
  onTissue: number
  /** The output: tiles clearing both gates. */
  clean: number
  reduction: number
}

export interface TilingTile {
  col: number
  row: number
  x: number
  y: number
  span: number

  fx: number
  fy: number
  fw: number
  fh: number

  tissueShare: number
  /** Share of *this tile's tissue* step 2 left in play. */
  cleanShare: number
  kept: boolean
  rejectedBy: TileRejection | null
}

export interface TilingCoverage {
  /** Unique slide area the kept tiles cover, counting overlap once. */
  coveredMm2: number
  tissueMm2: number
  coverage: number
}

export interface TilingSample {
  x: number
  y: number
  span: number
  size: number
  mpp: number
  tissueShare: number
}

export interface TilingReport {
  uploadId: string
  filename: string
  generatedAt: string

  params: TilingParams
  funnel: TilingFunnel
  coverage: TilingCoverage

  cols: number
  rows: number

  /** A sampled slice of the index, spaced through it rather than its first rows. */
  tiles: TilingTile[]
  listed: number

  /** Null when no tile survived both gates. */
  sample: TilingSample | null

  notes: string[]
  citation: string
}
