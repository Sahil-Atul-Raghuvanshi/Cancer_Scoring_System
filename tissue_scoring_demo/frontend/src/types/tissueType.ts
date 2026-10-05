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

export type TissueTypePanelName =
  | 'map'
  | 'flat'
  | 'scored'
  | 'confidence'
  /**
   * The uncertainty score before it is thresholded into the purple class. Served by
   * the two trained options only — the per-pixel option derives no such layer, and
   * asking for it there is a refusal rather than a blank picture.
   */
  | 'uncertainty'

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

  /**
   * `resnet18` or `concat_resnet18_mlp` for a checkpoint of ours, `beetle_nnunet_2d`
   * for the released network step 7's third option runs. The first two answer once per
   * window; the third answers once per pixel.
   */
  arch: string
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

  /**
   * BEETLE's own fields, and null on a checkpoint of ours.
   *
   * Everything above is read from a manifest this project wrote, so `init`,
   * `heldOutAccuracy`, `diceInvasive` and `nonInvasiveTestTiles` all mean "measured on
   * our held-out split with our labels" — and for a downloaded release none of them has
   * been. They stay null rather than being filled with figures from another dataset's
   * split under headings a reader will compare against ours.
   */
  perPixel: boolean
  /** The class names this model emits, in code order. Five for BEETLE. */
  classes: string[] | null
  folds: number[] | null
  foldsAvailable: number | null
  patchPx: number | null
  /** How to cite this model, where it is somebody else's. */
  citation: string | null

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
  /** The refusal rule the pass ran under. Null on the per-pixel option. */
  familiarity?: string | null
  /** Step 3's mask by identity. */
  tissueMaskKey?: string | null
  modelSha256: string | null
  licenceTrack: LicenceTrack

  /**
   * Which input contract the checkpoint declared, as the run actually applied it.
   *
   * Read this rather than the model's name: the two branches share all four window
   * geometries by construction, so a filename is a label somebody typed while this
   * comes from the manifest's own `input.channel`. The screen describes what the
   * model was shown, and describing the wrong one is the failure the whole
   * branch/geometry match exists to prevent.
   */
  branch: string
  inputChannel: string | null

  /** Window side in its own pixels. The checkpoint's, never this app's. */
  windowPx: number
  /** The window's field of view in microns — the scale architecture is read at. */
  windowUm: number
  mpp: number
  overlap: number
  /**
   * The trained options' input contract, lifted from a verified manifest — so null on
   * BEETLE, whose preprocessing is a division by 255 and nothing else. Null means "not
   * applicable here", not "off".
   */
  standardise: boolean | null
  gamma: number | null
  invertPolarity: boolean | null

  /**
   * The second check's settings, as this run applied them. Null on the per-pixel
   * option, which does not run one.
   */
  uncertainty: TissueTypeUncertaintyParams | null

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

  /**
   * Whether this pass answered per pixel. False on the two trained options, which give
   * one class per square; true on BEETLE. It decides how every count on the report
   * should be read — `pixels` rather than `windows`.
   */
  perPixel: boolean
  /**
   * The class names this pass emitted, in code order. Five on BEETLE, whose own label
   * set is reported rather than collapsed onto the three the trained options share;
   * null on those, where the three are fixed.
   */
  pixelClasses: string[] | null
  /** Which of BEETLE's five released folds were averaged. */
  folds: number[] | null
  /** BEETLE's sliding-window step as a fraction of its patch. 0.5 is its own default. */
  patchStep: number | null
  patchPx: number | null
  /** Microns per pixel the per-pixel mask is stored at. */
  maskMpp: number | null
  /** Forward passes the pass actually made — the figure that describes BEETLE's work. */
  patches: number | null
}

/**
 * Everything the progress screen needs to paint a running pass onto the slide.
 *
 * Step 8 is tens of minutes, so it is watched rather than waited for — and a bar with
 * a number beside it does not say *where* on the section the model has got to. With
 * this, the screen draws each patch onto a thumbnail as its class comes back: the pass
 * drawing itself, in the colours the finished map uses.
 *
 * The geometry is in **level-0 slide pixels**, which is the one frame of reference that
 * does not move when a thumbnail size is chosen — so a cell becomes a fraction of the
 * picture and the same numbers work at any display size. `stride` is what gets drawn
 * and `span` is only what the model saw: the cores partition the tissue, the spans
 * overlap, and painting spans would overprint every neighbour at 50% overlap.
 */
export interface TissueTypePaint {
  cols: number
  rows: number
  /** Level-0 pixels one window covers — the model's field of view. */
  span: number
  /** Level-0 pixels between windows, and the side of the cell that gets painted. */
  stride: number
  slideWidth: number
  slideHeight: number
  /** Hex colour per class, indexed by class id. The finished map's palette. */
  colours: string[]
  /** Plain-language name per class, indexed by class id. The report's own wording. */
  labels: string[]

  /**
   * Whether each painted window carries a pixel mask rather than one class.
   *
   * False on the trained options, where a window is a flat colour. True on BEETLE,
   * where `paintedMasks` holds the shapes it found inside each window — so the canvas
   * draws a little image into the cell instead of filling it.
   */
  perPixel: boolean
  /** Side of each mask in `paintedMasks`, in pixels. Zero when `perPixel` is false. */
  maskPx: number
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

  /** Painting geometry, once the grid exists. Null while it is still being laid. */
  paint: TissueTypePaint | null
  /**
   * Windows classified since the cursor asked for, as flat triples — row, column,
   * class, row, column, class. Flat because this is polled for the length of the pass
   * and every window travels through it exactly once.
   */
  paintedCells: number[]
  /**
   * One base64 pixel mask per window in `paintedCells`, in the same order, on the
   * BEETLE option only. Each decodes to `paint.maskPx` squared bytes of class ids.
   * Empty on the trained options, where a window has one class and `paintedCells`
   * already carries it.
   */
  paintedMasks: string[]
  /**
   * How far into the pass's paint log the reply reaches. Poll with it next. It counts
   * windows, so it indexes `paintedMasks` directly and `paintedCells` in threes.
   */
  paintedCursor: number
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

  /**
   * Windows this class won. On BEETLE a window has no single class, so this counts the
   * windows it *dominated* and `pixels` is what `share` is computed from.
   */
  windows: number
  /** Mask pixels this class claimed, on the per-pixel option only. */
  pixels: number | null
  /**
   * Share of the classified windows, or — per pixel — share of the tissue pixels.
   * Glass is out of that denominator either way.
   */
  share: number
  /** Unique area. Once per stride-sized cell, or the mask pixels' own area. */
  areaMm2: number
  meanConfidence: number
}

export interface TissueTypeUncertaintyParams {
  /**
   * Neighbourhood scale in microns — a physical distance, not a count of windows, so
   * changing the overlap changes how densely the tissue is sampled and not which of
   * it is flagged.
   */
  sigmaUm: number
  /** Where a connected in-situ region stops being a plausible duct system. */
  areaRefMm2: number
  areaMaxMm2: number
  fillLo: number
  fillHi: number
  /**
   * Area a region needs before its shape is read at all. A duct cut across is a solid
   * disc, so solidity means nothing below the size of one duct.
   */
  minShapeMm2: number
  minShapeWindows: number
  /** Score at or above which a patch is drawn as undetermined. */
  threshold: number
}

export interface TissueTypeUncertainty {
  params: TissueTypeUncertaintyParams
  /** The neighbourhood in grid cells, after the conversion from microns. */
  sigmaCells: number

  /** In-situ patches the second check would not stand behind. */
  windows: number
  /**
   * Share of the classified patches. The denominator is unchanged by this layer — a
   * flagged patch is still a classified one.
   */
  share: number
  areaMm2: number

  components: number
  /** Of those, how many the shape-and-size term flagged on its own. */
  flaggedComponents: number
  /** Largest in-situ region. The first figure to read when a whole field goes purple. */
  largestComponentMm2: number
  meanScore: number
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

  /** Shape of the per-pixel mask. Null on the trained options. */
  maskHeight: number | null
  maskWidth: number | null
  /** Mask pixels BEETLE called tissue — the denominator the shares are over. */
  tissuePixels: number | null
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

  /**
   * What the second check would not stand behind. Null on the per-pixel option and on
   * a pass that ran before the check existed. It can only ever qualify in-situ
   * patches, which are excluded from the score anyway — so `tumourContent` and
   * `scoredMm2` are identical with it and without it.
   */
  uncertainty: TissueTypeUncertainty | null

  /**
   * Patches refused before their answer was kept: scanner fill, or something unlike
   * anything the model was trained on. They have no class and are in no number above.
   * Null on the per-pixel option and on a pass that ran before the check existed.
   */
  refused?: TissueTypeRefusal | null

  model: TissueTypeModelInfo
  caveats: TissueTypeCaveat[]

  notes: string[]
  citation: string
}

export interface TissueTypeRefusal {
  flatWindows: number
  unfamiliarWindows: number
  refusedMm2: number
  refusedShare: number
  maxFlatShare: number
  distanceThreshold: number | null
  distanceCalibration: string | null
}
