/**
 * Types for step 3 - the tissue mask. Mirrors `backend/app/schemas/tissue.py`.
 *
 * Three things are reported separately and must stay separate on screen:
 * the decision (one mask and its area), the threshold (Otsu's cut and, when the
 * viewer has moved the slider, the manual one beside it), and the cleanup (the
 * mask after each morphological move). A single before-and-after would hide the
 * fact that closing and opening pull in opposite directions.
 */

export type TissuePanelName = 'thumbnail' | 'saturation' | 'mask' | 'overlay'

/** Which rule set the cut. `triangle` is Zack 1977, for spike-and-tail histograms. */
export type TissueThresholdSource = 'otsu' | 'triangle' | 'manual'

export type TissueStageKey = 'threshold' | 'closing' | 'opening' | 'components' | 'fill'

export interface TissueParams {
  /** Resolution asked for, in microns per pixel. */
  targetMpp: number
  /** Resolution achieved. Coarser on a large slide - the longest edge is capped. */
  maskMpp: number
  maskWidth: number
  maskHeight: number
  /** True when the size cap, not the target, set `maskMpp`. */
  capped: boolean

  closeUm: number
  openUm: number
  closePx: number
  openPx: number
  minComponentMm2: number
  fillHoleMaxMm2: number

  /** Whether step 2's artefact map was subtracted before thresholding. */
  qcGated: boolean
  /** 'otsu' means step 2 fell back to a threshold — say so rather than hide it. */
  qcSource: 'grandqc' | 'otsu' | 'unknown' | null
}

/**
 * The cut, which rule produced it, and what the other rules said.
 *
 * Both automatic rules are always present, whichever was used and even on a
 * manual run: a threshold is the most consequential number in this step, so it
 * is never shown without its alternatives.
 */
export interface TissueThreshold {
  value: number
  source: TissueThresholdSource

  otsu: number
  triangle: number

  /** The busiest saturation level, and its share of the histogram. */
  modalLevel: number
  modalShare: number
  /** Share at or above which Otsu's two-hump assumption is treated as broken. */
  spikeShare: number
  /** `modalShare < spikeShare` — Otsu's assumption holds. */
  bimodal: boolean

  /** The chord the triangle rule measures against, for drawing the construction. */
  triangleFrom: number
  triangleTo: number

  meanBelow: number | null
  meanAbove: number | null
  separation: number | null
  /**
   * Otsu's criterion at this cut over its value at Otsu's own cut. Meaningful
   * only when `bimodal` — on a spike-and-tail histogram Otsu's criterion is the
   * wrong yardstick, so a low ratio says the cut disagrees with Otsu, not that
   * it is worse.
   */
  varianceRatio: number
}

export interface TissueHistogram {
  /** 256 counts, over the pixels step 2 left in play — not the whole image. */
  bins: number[]
  countedPixels: number
  excludedPixels: number
  /** Otsu's between-class variance per level, normalised to a 0-1 peak. */
  criterion: number[]
}

export interface TissueStage {
  key: TissueStageKey
  label: string
  what: string
  extentUm: number | null
  pixels: number
  areaMm2: number
  /** Change from the previous move. Negative removes area. */
  deltaPixels: number
  deltaAreaMm2: number
  deltaShare: number
}

export interface TissueComponents {
  found: number
  kept: number
  dropped: number
  keptPixels: number
  droppedPixels: number
  droppedAreaMm2: number
  largestPixels: number
  largestAreaMm2: number
  largestShare: number
  minAreaMm2: number
  /** The same cutoff in pixels at this resolution, so the conversion is checkable. */
  minAreaPx: number
}

export interface TissueReport {
  uploadId: string
  filename: string
  generatedAt: string

  params: TissueParams
  threshold: TissueThreshold
  histogram: TissueHistogram
  stages: TissueStage[]
  components: TissueComponents

  tissuePixels: number
  tissueAreaMm2: number
  slideAreaMm2: number
  tissueShare: number
  glassShare: number

  holesFilledPixels: number
  holesFilledAreaMm2: number

  notes: string[]
  citation: string
}
