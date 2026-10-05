/**
 * Types for step 5 - optical density. Mirrors `backend/app/schemas/density.py`.
 *
 * Five things are reported separately and must stay separate on screen:
 *
 *   the tile        which field of view was transformed, why that one, and what
 *                   the alternatives scored. A density is per pixel, so unlike
 *                   steps 3 and 4 this step has to choose a place to stand — and
 *                   the choice is not neutral, because most of a section is
 *                   counterstain and a tile of counterstain has one arm, not two.
 *   the densities   where they landed, per channel and in total. Percentiles
 *                   rather than a mean and a spread, because the distribution is
 *                   skewed by construction: a lot of faint counterstain, a little
 *                   strong DAB.
 *   the limits      the three places a density is not a reading — the intensity
 *                   floor, pixels brighter than I₀, pixels too faint to have a
 *                   direction — plus the exact inverse applied back, which is
 *                   what shows the transform threw nothing away.
 *   the cloud       the densities as a point cloud, the two arms in it, and where
 *                   Ruifrok & Johnston's published vectors fall on the same axes.
 *                   This is the picture that makes step 6 obvious rather than
 *                   magical, and it is either in the data or it is not.
 *   the additivity  Beer–Lambert, tested on the reader's own tile. One stain
 *                   keeps one direction in density space as it darkens and does
 *                   not in intensity space, and both drifts are in degrees.
 *
 * `Channels` is imported from step 4's types rather than redefined. It is the
 * same thing — one value per RGB channel — and the reason step 4 gives for it
 * being three numbers and not one is exactly the reason a density is three
 * numbers too.
 *
 * Nothing here is rewritten, and that is the whole of Rule 2. This step is the
 * trunk: step 6 deconvolves these densities once and hands the haematoxylin
 * channel to the model and the DAB channel to the measurement, so both arms
 * leave from the numbers below.
 */

import type { Channels } from './calibration'

export type DensityPanelName = 'map' | 'tile' | 'density' | 'scatter' | 'limits'

/** The stains Ruifrok & Johnston published vectors for, as the API names them. */
export type ReferenceStain = 'haematoxylin' | 'dab' | 'eosin'

export interface DensityParams {
  targetMpp: number
  /** What the tile was actually read at. Differs only when the pyramid cannot supply the target. */
  tileMpp: number
  tileSize: number
  /** The physical field of view, in microns. */
  tileUm: number
  level: number
  downsample: number
  /**
   * True when the pyramid overshot and the read was area-averaged down. The
   * averaging happens in intensity space, *before* the logarithm — a coarser
   * sensor averages transmitted light, and averaging densities instead takes the
   * mean of logarithms, which is a different and lower number.
   */
  resampled: boolean

  odFloor: number
  /** Macenko's β: below this mean density a pixel has no reliable direction. */
  beta: number
  /** Macenko's α: which tail of the angular distribution is taken as an arm. */
  armPercentile: number
  /**
   * How far 8-bit quantisation may move a pixel's direction before it is kept out
   * of the cloud. The mirror image of β — β drops pixels with too little stain,
   * this drops the very dark ones, where one intensity level moves a channel's
   * density by 1/(I ln10), which is 0.43 at I = 1.
   */
  angularToleranceDeg: number

  screeningMpp: number
  screeningBlockPx: number
  minTissueShare: number
  /** A multiple of step 4's measured noise floor, not a constant. */
  minStain: number
  minStainMultiple: number

  tissueThreshold: number
  tissueThresholdSource: 'otsu' | 'triangle' | 'manual'
  qcGated: boolean
  qcSource: 'grandqc' | 'otsu' | 'unknown' | null
}

/** Step 4's white point, as this step divided by it. */
export interface WhiteUsed {
  rgb: Channels
  hex: string
  /** 'surface' means the tile was divided by the field at its own position. */
  mode: 'flat' | 'surface'
  percentile: number
  saturated: boolean
  /** What empty glass measures as. The floor every number here is read against. */
  noiseFloor: number
}

/** One block the chooser scored. Bounds are fractions of the slide. */
export interface TileCandidate {
  col: number
  row: number
  x: number
  y: number

  fx: number
  fy: number
  fw: number
  fh: number

  tissueShare: number
  consideredShare: number
  /** Mean optical density over the block's tissue. */
  stain: number
  /**
   * sqrt(second eigenvalue / first) of the block's density cloud — how far it
   * spreads off a single ray. One stain scores near zero, two open a wedge.
   */
  mixing: number
  score: number
  chosen: boolean
}

export interface TileOut {
  x: number
  y: number
  span: number
  size: number
  mpp: number
  level: number
  downsample: number
  resampled: boolean

  tissueShare: number
  consideredShare: number
  stain: number
  mixing: number
  score: number
  rank: number
  /** True when the caller named a position and the chooser snapped to it. */
  requested: boolean
  candidatesScored: number
}

export interface DensityStats {
  median: Channels
  p99: Channels
  maximum: Channels

  meanMedian: number
  meanP99: number
  meanMaximum: number
  /** Below zero when some pixel of the tile is brighter than I₀. */
  meanMinimum: number

  histogram: number[]
  /** Read off the data, not fixed at zero — the spread is a few tenths of an OD. */
  histogramLow: number
  histogramHigh: number
}

export interface DensityLimits {
  /** A channel recorded no light. Those densities are lower bounds. */
  floorShare: number
  /** Brighter than I₀, so negative density — which cannot happen physically. */
  negativeShare: number
  negativeWorst: number
  transparentShare: number
  beta: number

  /** Error of applying I = I₀ × 10^(−OD) back, in intensity levels. */
  roundtripMax: number
  roundtripMean: number
  roundtripExactShare: number
}

/** One edge of the wedge: a direction the data says a stain lies along. */
export interface DensityArm {
  angle: number
  vector: Channels
  share: number
  plotX: number
  plotY: number
  nearest: ReferenceStain
  /** In three dimensions, not in the projection. The claim, as a number. */
  degreesFromNearest: number
}

export interface DensityReference {
  name: ReferenceStain
  vector: Channels
  plotX: number
  plotY: number
  /**
   * How much of this unit vector points out of the plotted plane. A reference
   * drawn near an arm but with a large value here is a shadow, not a match.
   */
  outOfPlane: number
  degreesFromArm: number
}

export interface DensityCloud {
  basis: Channels[]
  /** Share of the cloud's energy the plane holds. Two absorbers span one plane. */
  explained: number
  plotted: number
  admittedShare: number
  /** Turned away for carrying less than β. */
  faintShare: number
  /** Turned away for being too dark for its direction to survive quantisation. */
  unstableShare: number
  toleranceDeg: number

  xLow: number
  xHigh: number
  yLow: number
  yHigh: number

  arms: DensityArm[]
  references: DensityReference[]
  separation: number
  /** False when the two arms are the two tails of one lobe — a tile with one stain. */
  twoArmed: boolean

  angles: number[]
  angleLow: number
  angleHigh: number
  angleCentre: number
  /** Degrees between Ruifrok's own haematoxylin and DAB — something to read against. */
  referenceSeparation: number
}

export interface DriftBin {
  density: number
  pixels: number
  odDegrees: number
  intensityDegrees: number
}

export interface DensityAdditivity {
  arm: ReferenceStain
  bins: DriftBin[]
  /** Degrees the direction turns from faintest to darkest, in density space. */
  odDrift: number
  /** The same in intensity space, where I = I₀ × 10^(−cv) curves. */
  intensityDrift: number
  pixels: number
}

/**
 * Which dyes this section carries, decided from the point cloud step 5 fitted.
 *
 * On step 5 because that is the screen that owns the evidence — the arms and their
 * angles are right there. Step 7 gates its full-colour option on the same verdict and
 * quotes it rather than deriving one of its own, so there is one answer to "does this
 * slide have two dyes" and one place it is measured.
 *
 * Step 5's own notes already say this in prose, which is where a reader meets it. This
 * structured form is for step 7.
 */
export interface DensityStaining {
  /** `he` | `haematoxylin_dab` | `single_stain` | `unknown` */
  staining: string
  isHe: boolean
  reason: string
  /**
   * How far the arm nearest eosin sits from Ruifrok's published eosin vector. Null when
   * no arm is nearest eosin at all — which is what every immunostained section in this
   * project's panel measured.
   */
  eosinArmDegrees: number | null
  toleranceDeg: number | null
}

export interface DensityReport {
  uploadId: string
  filename: string
  generatedAt: string

  params: DensityParams
  white: WhiteUsed
  tile: TileOut
  candidates: TileCandidate[]

  stats: DensityStats
  limits: DensityLimits
  /** Null when too few pixels carried any stain for a cloud to mean anything. */
  cloud: DensityCloud | null
  /** Null when the dominant arm had too few pixels to bin. */
  additivity: DensityAdditivity | null

  /** Null when this tile built no point cloud — there is then nothing to read. */
  staining: DensityStaining | null

  notes: string[]
  citation: string
}
