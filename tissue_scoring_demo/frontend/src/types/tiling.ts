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

/**
 * What step 7 can show the model, and all three of them run.
 *
 * `h_channel` and `he` are checkpoints of ours: the same window, deconvolved down to the
 * blue stain or handed over as the colour photograph, three classes per window either
 * way. `beetle` is another research group's released nnU-Net, which answers per *pixel*
 * in five classes and has no checkpoint, manifest or input contract of ours behind it.
 * Availability is the server's answer and differs per slide: both colour branches need
 * step 5 to have found a second dye, since neither normalises a stain away, and `beetle`
 * needs its 1.9 GB archive on disk on top of that. An unavailable option is rendered
 * disabled with its reason rather than hidden.
 */
export type TilingBranchName = 'h_channel' | 'he' | 'beetle'

/**
 * Which dyes step 5 found, and therefore whether the H&E branch is offered.
 *
 * Every number behind the verdict comes down with it, because `reason` is rendered
 * verbatim under a disabled option and a greyed-out button with no explanation is the
 * least useful thing a screen can do.
 */
export interface TilingStaining {
  /** `he` | `haematoxylin_dab` | `single_stain` | `unknown` */
  staining: string
  isHe: boolean
  /** One sentence, naming the number that decided. Written to be read on screen. */
  reason: string
  /** `step5-cloud` when step 5 measured it, `step5-not-run` when it has not. */
  source: string

  twoArmed: boolean | null
  separation: number | null
  referenceSeparation: number | null
  eosinArmDegrees: number | null
  toleranceDeg: number | null
  tileX: number | null
  tileY: number | null
}

/** One of step 7's three options, and whether this slide can take it. */
export interface TilingBranchOut {
  id: TilingBranchName
  label: string
  blurb: string
  /**
   * Whether this pipeline has built the option at all, as opposed to whether this
   * slide can take it. True for all three now; kept because it and `enabled` fail for
   * different reasons and the screen has to be able to say which.
   */
  implemented: boolean
  /**
   * Whether this slide permits it — false for either colour branch on an immunostained
   * section, and for `beetle` additionally when its archive has not been downloaded. The
   * slide is asked first, so a section with one dye gets the staining reason rather than
   * a 1.9 GB download it could not use.
   */
  enabled: boolean
  /** Why not, in words, whenever `enabled` or `implemented` is false. */
  reason: string | null
  /** Per branch: a scale can have a head on one branch and not the other. */
  fieldsOfView: TilingFieldOfView[]
}

/** The branch screen's payload, fetched before any grid exists. */
export interface TilingBranches {
  uploadId: string
  filename: string
  generatedAt: string
  staining: TilingStaining
  branches: TilingBranchOut[]
  defaultBranch: TilingBranchName
  notes: string[]
  citation: string
}

/** The choice a viewer committed, as the server recorded it. */
export interface TilingSelection {
  branch: TilingBranchName
  fieldOfViewUm: number
  overlap: number
  tissueThreshold: number | null
  model: string | null
  tilePx: number | null
  mpp: number | null
  chosenAt: string | null
}

export interface TilingParams {
  targetMpp: number
  tileSize: number
  /** The physical field of view, in microns. The number that has to fit an acinus. */
  tileUm: number
  /** Fraction of its own extent a tile shares with its neighbour. */
  overlap: number
  /**
   * The field of view this grid was laid at, in microns — step 7's one consequential
   * choice, because it decides which checkpoint step 8 runs. Equal to `tileUm`; named
   * separately because that describes the geometry and this records the choice.
   */
  fieldOfViewUm: number
  /**
   * The checkpoint fitted at that field of view, matched on its manifest's own
   * geometry rather than on its name. **Null when none is published yet**, in which
   * case the grid above is the planned one and step 8 cannot run at this setting.
   */
  model: string | null
  /**
   * Which of step 7's options laid this grid. Recorded because it, and not the field
   * of view alone, decides which checkpoint step 8 runs: the two branches share all
   * four geometries, so `fieldOfViewUm` no longer names one head.
   */
  branch: TilingBranchName
  /**
   * What that checkpoint declares it is shown — `haematoxylin` for a deconvolved
   * density plane, `rgb_he` for the colour photograph. Read from the manifest, so it
   * is what will actually be fed. Null when no head is published at this choice.
   */
  inputChannel: string | null
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

/**
 * One field of view step 7 offers, and whether a model exists for it.
 *
 * Sent for every offered value, not only the chosen one, because the control is a
 * comparison: choosing among 112, 224, 448 and 672 µm weighs a square count against a
 * physical scale, and both halves belong on screen at once. An option whose head has
 * not been trained yet is shown as unavailable rather than hidden — hiding it would
 * make the choice look smaller than it is.
 *
 * The list is data from the backend rather than a constant here, so a newly published
 * head appears in the picker without a frontend change.
 */
export interface TilingFieldOfView {
  um: number
  tilePx: number
  mpp: number
  model: string | null
  available: boolean
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
  /** Every field of view on offer, chosen or not. */
  fieldsOfView: TilingFieldOfView[]
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
