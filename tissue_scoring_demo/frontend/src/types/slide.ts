/** Types for slide upload and the step-1 readout. Mirrors `backend/app/schemas`. */

export type UploadState = 'initiated' | 'uploading' | 'processing' | 'ready' | 'failed'

export interface UploadStatus {
  uploadId: string
  state: UploadState
  filename: string
  totalSize: number
  chunkSize: number
  numChunks: number
  /** Chunk indices the server holds — the basis for resuming a transfer. */
  received: number[]
  receivedCount: number
  finalPath: string | null
  error: string | null
}

export interface PyramidLevel {
  level: number
  width: number
  height: number
  /** Level-0 pixels per pixel at this level. */
  downsample: number
  /** Microns per pixel at this level, when the scanner recorded a scale. */
  mpp: number | null
  tiles: number
  isWorkingLevel: boolean
}

/**
 * Step 1 output.
 *
 * The field that matters is `mpp`. Everything downstream is defined in microns
 * per pixel, and `workingLevel` is derived from it rather than hard-coded,
 * because the same level index is a different resolution on a different
 * scanner.
 */
export interface SlideReadout {
  uploadId: string
  /**
   * The antibody letter read off the filename — 'HE', or one of AFRUW — or null
   * when the name does not follow the convention. A SUGGESTION, never a
   * decision: a silently mis-detected marker measures the right cells in the
   * wrong compartment, which is the kind of wrong that does not look like an
   * error. Present it as "we think" and let a person correct it.
   */
  markerLetter: string | null
  /** Display name for `markerLetter` — 'H&E', 'CD44' — or null. */
  marker: string | null
  /** Which half of a case this looks like. Matches a stage's `runsOn` entries. */
  slideRole: 'he' | 'ihc' | 'unknown'
  filename: string

  widthPx: number
  heightPx: number
  megapixels: number

  mpp: number | null
  /** Where `mpp` came from: measured by the scanner, asserted by the user, or absent. */
  mppSource: 'scanner' | 'override' | 'unknown'
  /** What the file itself recorded, kept even when overridden. */
  scannerMpp: number | null
  magnification: string
  objectivePower: number | null
  vendor: string | null

  levelCount: number
  levels: PyramidLevel[]

  tileSize: number
  tilesAtLevel0: number

  targetMpp: number
  workingLevel: number
  workingMpp: number | null
  /** Extra software downsample applied after reading the working level. */
  workingDownsample: number
  /** Whether a level sits on the target exactly, needing no resample. */
  exactLevelMatch: boolean

  fileSizeMb: number
  /** Names only. Label and macro carry case identifiers and are never served. */
  associatedImages: string[]
}
