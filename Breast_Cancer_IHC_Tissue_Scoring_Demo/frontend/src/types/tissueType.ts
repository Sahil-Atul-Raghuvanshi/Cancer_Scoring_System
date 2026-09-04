/**
 * Types for step 8 — tissue-type segmentation. Mirrors
 * `backend/app/schemas/tissue_type.py`.
 *
 * This is the one step that runs a trained model, so the report leads with
 * *which* model: a 45 MB file's opinion, its licence, and what it scored on
 * held-out data. A reader who does not know which checkpoint produced a class map
 * cannot check anything about it.
 *
 * No probability array comes down the wire. Counts, areas, shares and a confidence
 * summary do; the full per-window softmax is step 10's input and stays on the
 * server, because it is `rows × cols × 3` floats over a whole slide and a browser
 * has no use for it that a picture does not serve better.
 */

export type TissueTypePanelName = 'map' | 'flat' | 'scored' | 'confidence'

/** 'permissive' ships; 'research-only' does not. Restrictive always wins. */
export type LicenceTrack = 'permissive' | 'research-only' | 'unknown'

export type TissueTypeRunState =
  | 'idle'
  | 'queued'
  | 'running'
  | 'ready'
  /** Stopped on purpose. Its own state, because it is a decision and not a fault. */
  | 'cancelled'
  | 'failed'
export type CaveatSeverity = 'blocking' | 'warning' | 'note'

export interface TissueTypeModelInfo {
  name: string
  found: boolean
  path: string | null
  /** Whether this is the checkpoint that will actually run. */
  selected: boolean

  init: string | null
  tilePx: number | null
  mpp: number | null
  /** Whether each window is divided by its own 99th-percentile density. */
  standardise: boolean | null
  created: string | null
  sha256: string | null
  bytes: number | null

  licenceTrack: LicenceTrack
  /** Every upstream term, so the report can name which one binds. */
  licences: Record<string, string>
  trainingSource: string | null

  heldOutAccuracy: number | null
  diceInvasive: number | null
  diceNonInvasive: number | null
  /**
   * Human-labelled in-situ tiles in the held-out set. A Dice figure ships with its
   * tile count or it does not ship — and on BCSS alone this number is nine.
   */
  nonInvasiveTestTiles: number | null

  problem: string | null
}

export interface TissueTypeCapability {
  ready: boolean
  reason: string

  torchInstalled: boolean
  torchVersion: string | null
  device: string
  deviceName: string | null
  threads: number | null

  modelsRoot: string | null
  models: TissueTypeModelInfo[]
  defaultModel: string

  licenceTrack: LicenceTrack
  licenceNote: string

  citation: string
}

export interface TissueTypeParams {
  model: string
  modelSha256: string | null
  licenceTrack: LicenceTrack

  /** Window side in its own pixels. The checkpoint's, never this app's. */
  windowPx: number
  /** The window's field of view in microns — the scale architecture is read at. */
  windowUm: number
  mpp: number
  overlap: number
  standardise: boolean
  gamma: number
  invertPolarity: boolean

  span: number
  stride: number

  /** Step 7's overlap, which set the tiles this ran over. */
  tileOverlap: number
  tissueThreshold: number
  tissueThresholdSource: string
  qcGated: boolean
  qcSource: string | null

  /**
   * Share of a window that must be tissue before the model is shown it. Step 8's own
   * gate, not step 7's: step 7's is on a 512 px tile and is deliberately generous, so a
   * 224 px window whose centre sits in a kept tile can itself be nearly all glass —
   * which is where a false in-situ rim around the section came from.
   */
  minTissueShare: number
  blockWindows: number
  batchSize: number
  device: string
}

export interface TissueTypeRun {
  uploadId: string
  state: TissueTypeRunState
  phase: 'grid' | 'classifying' | 'rendering' | null
  message: string | null
  progress: number
  /** Windows classified so far, and how many the grid marked. */
  done: number
  total: number
  startedAt: string | null
  finishedAt: string | null
  durationSeconds: number | null
  error: string | null
  params: TissueTypeParams | null
}

export interface TissueTypeClass {
  id: number
  key: string
  /** Plain-language name for the screen. `meaning` is the rigorous version. */
  label: string
  blurb: string
  meaning: string
  colour: string
  /** Whether this class enters the final score. Exactly one does. */
  scored: boolean

  windows: number
  share: number
  /** Unique area, counted once per stride-sized cell — so overlap cannot inflate it. */
  areaMm2: number
  meanConfidence: number
}

export interface TissueTypeGrid {
  cols: number
  rows: number
  every: number
  classified: number

  /** Step 7's kept tiles, and its tile geometry — deliberately not this step's. */
  tilesKept: number
  tilePx: number
  tileUm: number

  minTissueShare: number
  /** Windows on kept tissue that held too little tissue of their own to be shown. */
  gatedOut: number
  blocksRead: number
  batches: number
  seconds: number
}

export interface TissueTypeCaveat {
  key: string
  severity: CaveatSeverity
  headline: string
  detail: string
}

export interface TissueTypeReport {
  uploadId: string
  filename: string
  generatedAt: string

  params: TissueTypeParams
  run: TissueTypeRun

  classes: TissueTypeClass[]
  grid: TissueTypeGrid

  /**
   * Share of the classified tissue that is invasive carcinoma. Not "how much
   * tumour": in-situ disease and fat are already out of this denominator.
   */
  tumourContent: number
  scoredMm2: number
  tissueMm2: number
  meanConfidence: number

  model: TissueTypeModelInfo
  caveats: TissueTypeCaveat[]

  notes: string[]
  citation: string
}
