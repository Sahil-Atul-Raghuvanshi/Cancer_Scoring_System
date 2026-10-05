/**
 * Types for step 6 — colour deconvolution. Mirrors
 * `backend/app/schemas/deconvolution.py`.
 *
 * The response carries the same tile separated **twice**: once on Ruifrok &
 * Johnston's published stain vectors, and once on vectors estimated from this
 * tile's own colours. Both arrive together because the comparison is the point of
 * the step — same tile, same arithmetic, same absolute cut, two different scores
 * — and a screen that fetched the second half separately could show the two
 * halves of that argument describing two different runs.
 *
 * `Channels` is imported from step 4's types, as step 5's are: a stain vector is
 * three numbers for exactly the reason a white point is.
 */

import type { Channels } from './calibration'

export type DeconvolutionBasisName = 'fixed' | 'estimated'
export type DeconvolutionPanelName = 'tile' | 'haematoxylin' | 'dab' | 'residual'
export type StainChannelName = 'haematoxylin' | 'dab' | 'residual'

export interface DeconvolutionParams {
  /**
   * The absolute DAB optical density a pixel must clear to count as positive in
   * the preview score. Absolute, not a per-slide percentile — which is the whole
   * argument, because a percentile returns the same number under either basis.
   */
  positiveCut: number
  armPercentile: number
  /** Step 5's cut for "this pixel carries stain". Every figure here is over those. */
  beta: number
  minEstimatePixels: number
  channelBins: number
}

export interface DeconvolutionChannel {
  name: StainChannelName
  /** The unit optical-density direction this channel was projected onto. */
  vector: Channels

  median: number
  p99: number
  maximum: number
  mean: number
  /** Pixels needing a negative amount of this stain. Counted, never clipped. */
  negativeShare: number

  histogram: number[]
  /** Both bases share this range, so the comparison is not redrawn under itself. */
  histogramLow: number
  histogramHigh: number
}

export interface DeconvolutionPreview {
  cut: number
  positiveShare: number
  meanDab: number
  p99Dab: number
}

export interface DeconvolutionBasis {
  kind: DeconvolutionBasisName
  /** Three columns: haematoxylin, DAB, and the direction no stain occupies. */
  matrix: Channels[]
  channels: DeconvolutionChannel[]
  preview: DeconvolutionPreview

  /** Zero for the fixed basis by construction — it *is* the published pair. */
  degreesFromPublished: number[]

  /** Worst error of un-mixing and recomposing. Float rounding on a square basis. */
  exactnessMax: number
  exactnessMean: number
  /** Share of the tile's density the third column had to absorb. */
  residualShare: number
  /** How much the two stain channels still share, after separating them. */
  channelCorrelation: number
}

export interface DeconvolutionComparison {
  haematoxylinDegrees: number
  dabDegrees: number
  fixedPositiveShare: number
  estimatedPositiveShare: number
  /** Estimated minus fixed. Signed — it can go either way. */
  shareShift: number
  /** How much the estimated basis rescales the DAB reading. 1.0 would be agreement. */
  dabScale: number
  departed: boolean
  departureDeg: number
}

export interface DeconvolutionTile {
  x: number
  y: number
  size: number
  mpp: number
  tileUm: number
  level: number
  rank: number
  requested: boolean
  candidatesScored: number
  /** Share of the tile carrying stain — what every statistic here covers. */
  stainedShare: number
  admittedShare: number
}

export interface ArmAgreement {
  stain: string
  armVector: Channels
  nearest: string
  degrees: number
}

export interface DeconvolutionReport {
  uploadId: string
  filename: string
  generatedAt: string

  params: DeconvolutionParams
  tile: DeconvolutionTile

  fixed: DeconvolutionBasis
  /** Null when the tile cannot support a per-image estimate — not a failed run. */
  estimated: DeconvolutionBasis | null
  estimatedRefusal: string | null
  comparison: DeconvolutionComparison | null

  /** How alike the three optical-density channels were before un-mixing. */
  mixedCorrelation: number
  armAgreement: ArmAgreement[]

  notes: string[]
  citation: string
}
