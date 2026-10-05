/**
 * Types for step 4 - white calibration. Mirrors `backend/app/schemas/calibration.py`.
 *
 * Four things are reported separately and must stay separate on screen:
 *
 *   the glass       which pixels were measured, and what each of the five
 *                   exclusions removed. A naive inversion of step 3's mask
 *                   includes pen ink, the coverslip edge, the halo around the
 *                   section and the scanner's own background fill — every one of
 *                   which is darker than glass, and so biases I₀ down.
 *   the white point I₀ per channel, with the percentile ladder around it, so the
 *                   choice of the 95th can be seen to be stable rather than
 *                   taken on trust.
 *   the field       whether I₀ was allowed to vary across the slide, and the
 *                   diagnostics of the fitted surface either way. Both answers
 *                   are always present.
 *   the consequence what empty glass measures as in optical density once the
 *                   calibration is applied to it — the step's own error bar, in
 *                   the units step 5 produces.
 *
 * Nothing here rescales the image, and that is the whole of Rule 2. Calibration
 * records what "zero stain" means; normalisation rewrites the pixels so they look
 * like another slide's. The first preserves the measurement, the second destroys
 * it.
 */

export type CalibrationPanelName = 'thumbnail' | 'glass' | 'field' | 'corrected'

/** Flat means one I₀ for the whole slide; surface means it varies with position. */
export type CalibrationFieldMode = 'flat' | 'surface'

export type CalibrationExclusionKey =
  | 'complement'
  | 'border'
  | 'artefacts'
  | 'fill'
  | 'fill_seam'
  | 'clearance'

/**
 * One value per RGB channel.
 *
 * Three numbers and not one, because a scanner's illuminant is not neutral.
 * Collapsing them would bake its colour cast into every optical density.
 */
export interface Channels {
  r: number
  g: number
  b: number
}

export interface CalibrationParams {
  targetMpp: number
  /** Resolution achieved — step 3's grid, so the mask needs no resampling. */
  calibrationMpp: number
  width: number
  height: number
  capped: boolean

  percentile: number
  borderUm: number
  clearanceUm: number
  fillClearanceUm: number
  borderPx: number
  clearancePx: number
  fillClearancePx: number
  fillMinShare: number
  fillMaxShoulder: number
  patchUm: number
  patchPx: number
  patchMinGlass: number
  minPatches: number
  vignetteSnr: number
  maxLeverage: number
  odFloor: number

  /** The step 3 cut this glass set was derived from — move it and I₀ moves. */
  tissueThreshold: number
  tissueThresholdSource: 'otsu' | 'triangle' | 'manual'
  qcGated: boolean
  qcSource: 'grandqc' | 'otsu' | 'unknown' | null
}

/**
 * A digital background fill found in the candidate glass, and excluded from it.
 *
 * The part of the slide canvas the scanner never imaged. Not glass — nothing was
 * measured there — and darker than the real glass, so leaving it in drags I₀ down
 * and reports apparent stain over a region that does not exist.
 */
export interface BackgroundFill {
  rgb: Channels
  hex: string
  share: number
  /**
   * Colours within one level in every channel, over this colour's own count.
   * Near zero for a digital constant; 0.24 and up for real glass, whose sensor
   * noise guarantees it neighbours. That gap is what makes this a test.
   */
  shoulder: number
  pixels: number
}

export interface CalibrationExclusion {
  key: CalibrationExclusionKey
  label: string
  what: string
  extentUm: number | null
  pixels: number
  areaMm2: number
  /** Change from the previous move. Never positive — this is a narrowing. */
  deltaPixels: number
  deltaAreaMm2: number
  deltaShare: number
}

export interface WhitePoint {
  /** I₀ — the intensity that corresponds to zero stain. */
  rgb: Channels
  hex: string
  percentile: number

  /** I₀ at each of several percentiles, keyed by percentile as written. */
  ladder: Record<string, Channels>
  /**
   * Spread across the 90th to 99th percentiles, worst channel. Small means the
   * choice of percentile is not doing the work.
   */
  plateau: number

  clipped: Channels
  /** A channel pinned at 255: I₀ is an underestimate and OD is compressed. */
  saturated: boolean
  /** Spread of the three channels over their mean. Zero would be neutral. */
  cast: number
  sampledPixels: number
  sampledAreaMm2: number
}

/** Bounds are fractions of the slide, so the browser needs no resolution. */
export interface CalibrationPatch {
  col: number
  row: number
  x: number
  y: number
  width: number
  height: number

  glassPixels: number
  glassShare: number
  used: boolean
  rgb: Channels | null
  hex: string | null
}

export interface CalibrationSurface {
  coefficients: number[][]
  terms: string[]

  r2: Channels
  residualRms: Channels
  swing: Channels
  mean: Channels

  patchesUsed: number
  /**
   * 99th percentile of extrapolation leverage over the tissue — how far outside
   * its own samples the fit reaches where a density will actually be computed.
   * Measured over the tissue and not the frame: glass samples ring the section,
   * so the frame's corners are always extrapolated and never read.
   */
  leverage: number
  leverageMax: number
  /** swing / residualRms, worst channel — the statistic the choice turns on. */
  snr: number
  /** The swing as a fraction of the level — vignetting, in percent. */
  amplitude: number
  /**
   * What ignoring the surface would cost, in optical density. The only currency
   * in which "1.6% of I₀" means anything to a pipeline whose output is a density.
   */
  odError: number
}

export interface CalibrationChoice {
  mode: CalibrationFieldMode
  reason: string
  /** Whether a surface was fitted at all, used or not. */
  fitted: boolean
  patchesUsed: number
  minPatches: number
  snr: number
  snrRequired: number
  leverage: number
  leverageLimit: number
}

export interface CalibrationNoiseFloor {
  /** OD of the dimmest sampled glass — the largest apparent stain on nothing. */
  floor: Channels
  median: Channels
  percentile: number
  worst: number
}

export interface CalibrationReport {
  uploadId: string
  filename: string
  generatedAt: string

  params: CalibrationParams
  white: WhitePoint
  exclusions: CalibrationExclusion[]
  fills: BackgroundFill[]
  patches: CalibrationPatch[]
  surface: CalibrationSurface | null
  choice: CalibrationChoice
  noise: CalibrationNoiseFloor

  glassPixels: number
  glassAreaMm2: number
  glassShare: number

  notes: string[]
  citation: string
}

/** What using one slide's I₀ on another would cost, in optical density. */
export interface CalibrationDifference {
  uploadId: string
  otherUploadId: string
  /** log10(this I₀ / the other's), per channel — a constant *added* to every OD. */
  odShift: Channels
  worst: number
}

export interface CalibrationSummary {
  uploadId: string
  filename: string
  rgb: Channels
  hex: string
  mode: CalibrationFieldMode
  saturated: boolean
  noiseFloor: number
  glassShare: number
}

export interface CalibrationComparison {
  slides: CalibrationSummary[]
  differences: CalibrationDifference[]
  worstShift: number
  note: string
}
