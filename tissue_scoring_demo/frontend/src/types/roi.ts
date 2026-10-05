/**
 * Types for step 9 — the ROI mask. Mirrors `backend/app/schemas/roi.py`.
 *
 * Two products travel on one report, and they answer different questions.
 * `regions`/`ledger`/`params` describe the single smoothed, closed, in-situ-protected
 * region every later step measures inside — the denominator. `classRegions`/
 * `classAreaMm2` describe something read straight off step 8's own tile calls, with no
 * smoothing or morphology: every connected patch of invasive, DCIS or the "cannot be
 * determined" overlay, bordered and ranked by area — never stroma. See
 * `roi_service.py`'s module docstring for why the two live on one report rather than
 * two.
 */

export type RoiPanelName =
  | 'seed'
  | 'smoothed'
  | 'binary'
  | 'region'
  | 'outline'
  | 'borders'
  | 'borders_on_slide'

export type RoiTopClass = 'dcis' | 'invasive'

export interface RoiRegion {
  /** Rank against the other regions of the same class or focus — 0 is the largest. */
  index: number
  cells: number
  areaMm2: number
  /** Rings other than the outer one — tissue excluded from within this region. */
  holes: number
  /**
   * Closed rings of (x, y) level-0 slide pixels, outer boundary first.
   * Rectilinear — a union of whole grid cells, not a curve fitted to one.
   */
  rings: [number, number][][]
}

export interface RoiLedger {
  seedMm2: number
  seedComponents: number
  mergedMm2: number
  protectedMm2: number
  protectedCells: number
  droppedMm2: number
  droppedComponents: number
}

export interface RoiParams {
  sigma: number
  threshold: number
  closeCells: number
  protectInSitu: number
  minAreaMm2: number
  keepLargest: number | null
}

/**
 * Step 8's name for each border class, and the key `classRegions`/`classAreaMm2` are
 * indexed by. `uncertain` is a display-only overlay, not one of step 8's own three
 * classes — see `classes.py` — so it is spelled out here rather than derived.
 */
export type RoiClassName = 'non_invasive_epithelium' | 'invasive_epithelium' | 'uncertain'

export interface RoiReport {
  uploadId: string
  generatedAt: string

  /** The ROI — the denominator every later step measures in. */
  areaMm2: number
  cells: number
  /** Classified tissue from step 8, for the share below. */
  tissueMm2: number
  /** ROI as a fraction of the classified tissue. */
  roiShare: number

  /**
   * The slide's own size, in level-0 pixels — the frame every ring above is in. A
   * region's ring divided by these two numbers lands as a fraction of the borders
   * panel, wherever that panel is actually displayed.
   */
  slideWidth: number
  slideHeight: number

  regions: RoiRegion[]
  /** Total holes across all foci. */
  holes: number

  ledger: RoiLedger
  params: RoiParams

  /**
   * Every connected patch of invasive, DCIS or uncertain tissue, sorted largest first
   * within each class — so `classRegions[name].slice(0, 3)` is the top 3 the crop and
   * QuPath exports are built from. Never has a `non_epithelium` key: stroma gets no
   * borders.
   */
  classRegions: Partial<Record<RoiClassName, RoiRegion[]>>
  /** Total area per class, in mm² — the sum of `classRegions[name]`. */
  classAreaMm2: Partial<Record<RoiClassName, number>>

  /** What this region does and does not license, in the words the UI shows. */
  notes: string[]
}
