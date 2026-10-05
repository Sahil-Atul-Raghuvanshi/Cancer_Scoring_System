/**
 * Types for step 2 - quality control. Mirrors `backend/app/schemas/qc.py`.
 *
 * The split in these types is the point of the step: `classes` and `gate` are
 * GrandQC's decision, `metrics` are the classical explanation of it. Nothing in
 * `metrics` feeds the decision, and the UI should never present it as if it did.
 */

export type QCMode = 'full' | 'degraded' | 'unavailable'

export type QCRunState =
  | 'idle'
  | 'queued'
  | 'running'
  | 'ready'
  /** Stopped on purpose. Its own state, because it is a decision and not a fault. */
  | 'cancelled'
  | 'failed'

/** GrandQC's class keys, as the mask codes them. */
export type QCClassKey =
  | 'tissue'
  | 'fold'
  | 'darkspot'
  | 'pen'
  | 'edge'
  | 'focus'
  | 'background'

/** Metric keys, camelCased at the API boundary. */
export type QCMetricKey =
  | 'tenengrad'
  | 'laplacianVariance'
  | 'rmsContrast'
  | 'michelsonContrast'
  | 'brightness'
  | 'saturation'
  | 'textureEnergy'

export interface QCModelInfo {
  role: 'tissue' | 'artefact'
  name: string
  found: boolean
  path: string | null
  mpp: number | null
  magnification: string | null
  classes: number | null
}

export interface QCDownload {
  label: string
  url: string
  files: string[]
  targetDir: string
}

export interface QCCapability {
  ready: boolean
  mode: QCMode
  reason: string

  torchInstalled: boolean
  torchVersion: string | null
  smpVersion: string | null
  device: string
  deviceName: string | null

  featuresAvailable: boolean
  featuresProblem: string | null

  modelsRoot: string | null
  searchedPaths: string[]
  models: QCModelInfo[]
  availableModelMpps: number[]
  defaultModelMpp: number
  downloads: QCDownload[]

  citation: string
  licenceNote: string
}

export interface QCParams {
  mode: string
  modelMpp: number
  /** The artefact model's magnification, e.g. '7x'. */
  modelLabel: string
  tissueModelMpp: number
  patchSize: number
  readFromLevel0: boolean
  collectFeatures: boolean
}

export interface QCRun {
  uploadId: string
  state: QCRunState
  phase: 'tissue' | 'artefacts' | 'rendering' | 'summarising' | null
  message: string | null
  progress: number
  done: number
  total: number
  startedAt: string | null
  finishedAt: string | null
  durationSeconds: number | null
  error: string | null
  params: QCParams | null
}

export interface QCClassShare {
  id: number
  key: QCClassKey
  label: string
  blurb: string
  colour: string
  isArtefact: boolean
  pixels: number
  /** Fraction of tissue pixels. Glass is excluded from the denominator. */
  shareOfTissue: number
  areaMm2: number | null
  /** Metric the explanation panel leads with. Presentation only. */
  explainedBy: QCMetricKey | null
}

export interface QCTissue {
  tissuePixels: number
  cleanPixels: number
  artefactPixels: number
  backgroundPixels: number
  unanalysedPixels: number
  tissueAreaMm2: number | null
  cleanAreaMm2: number | null
  rejectedAreaMm2: number | null
  rejectedShare: number
  tissueShareOfSlide: number
  /** 'otsu' means the GrandQC tissue model was absent — say so on screen. */
  tissueSource: 'grandqc' | 'otsu'
}

export interface QCGate {
  qcOffPixels: number
  qcOnPixels: number
  qcOffAreaMm2: number | null
  qcOnAreaMm2: number | null
  removedAreaMm2: number | null
  removedShare: number
  note: string
}

export interface QCMetricSummary {
  key: QCMetricKey
  label: string
  unit: string
  description: string
  lowIsBad: boolean
  measuredAtMpp: number
  cleanMean: number | null
  byClass: Partial<Record<QCClassKey, number>>
  /** Class mean over clean-tissue mean. 1.0 means indistinguishable. */
  ratioToClean: Partial<Record<QCClassKey, number>>
  sampleSizes: Partial<Record<QCClassKey, number>>
}

/**
 * Two grids, and they are not the same thing. A *patch* is one run of the
 * model; a *cell* is a sub-block of a patch and is what the metrics are
 * measured on. Classing whole patches would leave the explanation panel with
 * nothing to say about folds, pen or blur — almost nothing is wrong with a
 * whole millimetre of tissue at once.
 */
export interface QCGridMeta {
  /** Measurement cells across and down. */
  cols: number
  rows: number
  /** Model patches across and down — one forward pass each. */
  patchCols: number
  patchRows: number
  blocksPerPatch: number

  patchSize: number
  cellMpp: number
  cellExtentPx: number
  cellExtentUm: number

  measuredCells: number
  skippedCells: number
  patchesInferred: number
  patchesSkipped: number
}

export interface QCReport {
  uploadId: string
  filename: string
  generatedAt: string
  params: QCParams
  run: QCRun
  classes: QCClassShare[]
  tissue: QCTissue
  gate: QCGate
  metrics: QCMetricSummary[]
  grid: QCGridMeta
  models: QCModelInfo[]
  notes: string[]
  citation: string
}

export interface QCGridCell {
  col: number
  row: number
  dominant: QCClassKey | null
  tissueShare: number
  metrics: Partial<Record<QCMetricKey, number>> | null
}

export interface QCGrid {
  uploadId: string
  grid: QCGridMeta
  metricKeys: QCMetricKey[]
  cells: QCGridCell[]
}

export interface QCRegionMetric {
  key: QCMetricKey
  label: string
  unit: string
  description: string
  lowIsBad: boolean
  value: number
  cleanMean: number | null
  ratioToClean: number | null
}

export interface QCRegionExplain {
  uploadId: string
  x: number
  y: number
  sizePx: number
  measuredAtMpp: number
  dominant: QCClassKey | null
  classShares: Partial<Record<QCClassKey, number>>
  metrics: QCRegionMetric[]
  verdict: string
}
