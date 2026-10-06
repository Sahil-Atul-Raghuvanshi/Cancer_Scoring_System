/** Types for step 11. Mirrors `backend/app/schemas/nuclei.py`. */

export type NucleiState = 'queued' | 'running' | 'ready' | 'failed'

/** What the segmenter was shown. Only `haematoxylin` feeds a denominator. */
export type NucleiChannel = 'haematoxylin' | 'rgb'

export interface NucleiCapability {
  available: boolean
  reason: string | null
  modelsDir: string
  model: string | null
  version: string | null
  licence: string | null
  mpp: number | null
  /** The configured detector: 'cellpose' (default since P-03) or 'instanseg'. */
  engine: string | null
}

export interface NucleiRun {
  heUploadId: string
  ihcUploadId: string
  state: NucleiState
  message: string | null
  /** 0-1 across every region, so the bar means one thing throughout. */
  progress: number
  startedAt: string | null
  finishedAt: string | null
  duration: number | null
  error: string | null
}

export interface NucleusOut {
  id: number
  x: number
  y: number
  areaUm2: number
  perimeterUm: number
  circularity: number
  eccentricity: number
  haematoxylin: number
  /** Outer ring first, then holes. IHC slide level-0 pixels. */
  rings: number[][][]
  /** False when it sits in the border band and is excluded from the count. */
  counted: boolean
}

export interface FieldOut {
  index: number
  x: number
  y: number
  /** Side in level-0 pixels - what was read off the slide. */
  span: number
  /** Side in model pixels - what was segmented. */
  size: number
  mpp: number
  tissue: number
  detected: number
  counted: number
  countedMm2: number
  densityPerMm2: number
  /** Share of the counted area that is tissue; glass and scanner fill are not. */
  tissueShare: number | null
  densityPerTissueMm2: number | null
}

export interface RegionNuclei {
  rank: number
  index: number
  areaMm2: number
  fields: FieldOut[]
  fieldsAvailable: number
  sampledMm2: number
  sampledShare: number
  detected: number
  counted: number
  densityPerMm2: number
  densityCv: number
  tissueMm2: number | null
  densityPerTissueMm2: number | null
  medianAreaUm2: number
  medianCircularity: number
}

export interface StainReport {
  basis: string
  gain: number
  macenkoExplained: number | null
  macenkoHaematoxylinDriftDeg: number | null
  macenkoDabDriftDeg: number | null
  notes: string[]
}

export interface ComparisonOut {
  fieldIndex: number
  regionRank: number
  instansegHaematoxylin: number
  instansegRgb: number
  watershedHaematoxylin: number
  /** The detector that counts, and its count on this field (left-hand panel). */
  engine: string
  production: number | null
}

export interface NucleiReport {
  heUploadId: string
  ihcUploadId: string
  marker: string | null
  state: NucleiState
  generatedAt: string

  /** Which detector found these nuclei. Null on reports older than the choice (InstanSeg). */
  engine: string | null
  modelName: string | null
  modelVersion: string | null
  modelLicence: string | null
  modelMpp: number | null

  regions: RegionNuclei[]
  stain: StainReport
  comparison: ComparisonOut[]

  detected: number
  counted: number
  sampledMm2: number
  densityPerMm2: number
  /** Tissue inside the sampled squares, and nuclei per mm2 of it (P-03). */
  tissueMm2: number | null
  tissueShare: number | null
  densityPerTissueMm2: number | null
  densityByMarker: Record<string, number>

  /**
   * The same measurement on this case's H&E slide, inside the same regions.
   *
   * The reference the IHC figure is read against, measured on every run rather
   * than waiting for a second marker - otherwise the check only starts working
   * after a failure has already gone unnoticed once.
   */
  heDensityPerMm2: number | null
  heMedianAreaUm2: number | null
  heDensityPerTissueMm2: number | null
  heTissueShare: number | null
  /**
   * How far the IHC density falls short of the H&E's, as a fraction - per mm2 of
   * tissue on both sides since P-03; per mm2 of sampled area on older reports.
   */
  densityShortfall: number | null
  /** The old per-area figure, kept beside it. Null on older reports. */
  areaShortfall: number | null

  seconds: number | null
  notes: string[]
}

/** One region's stored nuclei, fetched separately from the report. */
export interface RegionNucleiPayload {
  rank: number
  fields: {
    index: number
    x: number
    y: number
    span: number
    size: number
    nuclei: NucleusOut[]
  }[]
}
