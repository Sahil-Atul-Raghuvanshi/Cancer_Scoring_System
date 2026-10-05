/**
 * Types for step 10 — reviewing and selecting the candidate regions.
 * Mirrors `backend/app/schemas/roi_selection.py`.
 *
 * The report has two halves that change for different reasons. The candidates are a
 * function of step 8's class map: rebuild step 8 and they are all different. The
 * selection is a person's answer and survives the cards being redrawn. `classMapKey`
 * is what ties the two together — a selection made against a different class map is
 * dropped by the server rather than reinterpreted, because the ids are *area ranks*
 * and on a new class map `ROI-007` is different tissue.
 */

export interface RoiCandidate {
  /**
   * Stable within one class map, `ROI-001` upward, ranked by area — `ROI-001` is the
   * largest invasive patch on the slide.
   */
  roiId: string
  /** Step 9's own 0-based rank for this region. `roiId` is this plus one, padded. */
  index: number

  /** Bounding box in H&E level-0 pixels. */
  x: number
  y: number
  width: number
  height: number

  /** What step 8 called it. `invasive` today, and only invasive is ever offered. */
  candidateClass: string
  /**
   * Mean probability the tile model gave this region's own class over the windows it
   * is made of. A description of the model's call, not of the tissue — a region can be
   * 0.97 confident and still be a rim of glass. Not a gate anywhere.
   */
  confidence: number
  /** Windows of step 8's grid this region covers. */
  cells: number
  areaMm2: number

  /** Closed rings of (x, y) level-0 pixels, outer boundary first then holes. */
  rings: [number, number][][]

  /** `(row, col)` of every window inside this region, and the same list as names. */
  tileCells: [number, number][]
  tileIds: string[]

  /** Forward-pass windows BEETLE would spend on this region on step 11. */
  windows: number
}

export interface RoiSelectionReport {
  uploadId: string
  generatedAt: string

  /** The frame every `rings` above is in. */
  slideWidth: number
  slideHeight: number

  candidates: RoiCandidate[]

  /** Currently ticked, in candidate order. */
  selected: string[]
  /**
   * What the pipeline ticks on its own: enough of the largest regions to cover most of
   * the offered area. Kept beside `selected` so the screen can offer "back to the
   * default" without having to re-derive the rule.
   */
  defaultSelection: string[]
  /**
   * False until somebody has actually chosen. An unattended run refines the default
   * and says so here rather than claiming a review that never happened.
   */
  chosenByPerson: boolean
  chosenAt: string | null

  /** Which class map these ids belong to. See this module's header. */
  classMapKey: string | null

  invasiveMm2: number
  offeredMm2: number
  selectedMm2: number
  /** Forward-pass windows the ticked regions cost — step 11's bill, before it runs. */
  selectedWindows: number

  droppedSmall: number
  droppedSmallMm2: number
  droppedCapped: number
  droppedCappedMm2: number

  notes: string[]
}
