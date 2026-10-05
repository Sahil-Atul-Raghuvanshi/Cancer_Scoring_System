/**
 * Types for step 11 — refining each chosen region per pixel.
 * Mirrors `backend/app/schemas/roi_refinement.py`.
 *
 * The report is a live list rather than a summary written at the end: one row per
 * selected region, each carrying its own state, its own numbers and enough geometry to
 * draw its before and after. A screen polls the run for the position and re-reads the
 * report to draw whatever has landed, so a finished region is on screen while the next
 * one is still going.
 */

/** The pass as a whole. `partial` means some regions finished and some failed. */
export type RefinementState =
  | 'queued'
  | 'running'
  | 'ready'
  | 'partial'
  | 'failed'
  | 'cancelled'

/**
 * One region's position in the sequence `pending → extracting → segmenting → tracing →
 * complete`, with `failed` reachable from any of them. The stage list on screen is
 * rendered from this alone — the server does not send a percentage per region, because
 * BEETLE reports completion per window and there is no honest sub-window figure.
 */
export type RegionState =
  | 'pending'
  | 'extracting'
  | 'segmenting'
  | 'tracing'
  | 'complete'
  | 'failed'
  | 'skipped'

/**
 * One region's own position, for the bar on its own row.
 *
 * Separate from `RefinedRegion` because it exists *before* that does. A region has a
 * window count and a state from the moment the pass is queued; it has rings and an area
 * only once it finishes, so a list that drew its bars from the report alone could not
 * draw one for a region that had not started.
 *
 * `windowsTotal` is step 9's count until the region starts and the grid's own after, so
 * it can move once, upward, at the moment a region begins. Clamp the share.
 */
export interface RegionProgress {
  roiId: string
  index: number
  state: RegionState
  windowsDone: number
  windowsTotal: number
}

/**
 * Everything needed to paint one region's segmentation as it happens.
 *
 * The same shape as step 8's `TissueTypePaint` because it feeds the same drawing code.
 * The difference is the frame: step 8 paints the whole section onto a thumbnail, this
 * paints one padded box onto that region's own crop.
 *
 * **The grid coordinates are still the slide's.** `crop.restrict` narrows step 8's grid
 * rather than building a new one, so a painted window's `(row, col)` indexes the
 * slide-wide grid; `span`, `stride` and the slide's size are what let a client reproduce
 * the grid's clamp exactly, and `crop*` is what it projects into afterwards.
 */
export interface RefinementPaint {
  /** The region this feed belongs to. When it changes, the canvas starts again. */
  roiId: string

  /** The padded box being segmented, in level-0 pixels — the extent of `input.png`. */
  cropX: number
  cropY: number
  cropWidth: number
  cropHeight: number

  span: number
  stride: number
  slideWidth: number
  slideHeight: number

  /** Hex colour and plain-language name per BEETLE class id. */
  colours: string[]
  labels: string[]

  /** Always true on this step — step 11 is BEETLE only — and carried for symmetry. */
  perPixel: boolean
  /** Side of each window's mask in `paintedMasks`, in pixels. */
  maskPx: number
}

export interface RefinementRun {
  uploadId: string
  state: RefinementState
  message: string | null
  /** Regions finished, of regions selected. */
  done: number
  total: number
  /** Windows finished, of windows selected — what stops one big region looking stuck. */
  windowsDone: number
  windowsTotal: number
  /** The region being worked on right now, so the list never has to guess. */
  currentRoiId: string | null

  /**
   * Every selected region's own position, in the order they will be run. Present from
   * the moment the pass is queued, so a region still waiting has an empty bar rather
   * than no row.
   */
  progress: RegionProgress[]

  /** Geometry for painting the region in progress, or null before the first starts. */
  paint: RefinementPaint | null
  /** Windows segmented since `paintedSince`, as flat `row, col, class` triples. */
  paintedCells: number[]
  /** One base64 pixel mask per window in `paintedCells`, same order. */
  paintedMasks: string[]
  /**
   * How far into the *current region's* paint log `paintedCells` reaches. Pass it back
   * as the next `paintedSince`. It counts windows, so it indexes `paintedMasks`
   * directly and `paintedCells` in threes — and it resets when the region does, which
   * is why `paint.roiId` rather than this is what says a new canvas is needed.
   */
  paintedCursor: number
  startedAt: string | null
  finishedAt: string | null
  duration: number | null
  error: string | null
}

export interface RefinedRegion {
  roiId: string
  index: number
  state: RegionState
  error: string | null

  /** The padded box BEETLE was shown, in H&E level-0 pixels. */
  cropX: number
  cropY: number
  cropWidth: number
  cropHeight: number
  padUm: number
  maskMpp: number
  baseMpp: number
  maskWidth: number
  maskHeight: number

  /** The coarse candidate, as step 9 traced it. The left-hand picture's geometry. */
  tileRings: [number, number][][]
  tileAreaMm2: number

  /**
   * What BEETLE drew. **One entry per connected focus**, each its own outer ring
   * followed by its holes. Never flatten these into one list before filling or
   * rasterising: ring 0 is the outer boundary and every later ring is a hole, so a
   * flattened multi-focus region becomes its first focus with the others punched out.
   * Stroking every ring of every focus is fine — a line does not care which is which.
   */
  pieces: number[][][][]
  focusCount: number
  holes: number
  areaMm2: number

  /**
   * `areaMm2 / tileAreaMm2`. Under 1 on essentially every region, and that is the step
   * working: a square drawn around tumour contains stroma, fat and glass.
   */
  keptShare: number

  windows: number
  seconds: number
  /** Share of the box's classified pixels each BEETLE class won, by class name. */
  classShare: Record<string, number>
}

export interface RefinementReport {
  uploadId: string
  state: RefinementState
  generatedAt: string

  slideWidth: number
  slideHeight: number

  model: string | null
  fieldOfViewUm: number
  maskMpp: number
  padUm: number
  folds: number

  regions: RefinedRegion[]

  /** The combined mask: the denominator everything after this step measures inside. */
  refinedMm2: number
  /** The same regions as coarse boxes, for the comparison. */
  tileMm2: number
  keptShare: number
  /** All invasive carcinoma step 8 called, including regions nobody selected. */
  invasiveMm2: number

  completed: number
  failed: number
  selected: number

  seconds: number
  windows: number

  notes: string[]
}

/** One region's pictures. `tile` and `beetleOverlay` are the before and after. */
export type RefinementRegionPanel = 'input' | 'tile' | 'beetle_mask' | 'beetle_overlay'

/** The section, twice: the chosen squares, and what BEETLE drew inside them. */
export type RefinementSlidePanel = 'coarse' | 'refined'
