/**
 * Types mirroring the FastAPI response models in `backend/app/schemas`.
 * The backend serialises camelCase, so these names match the wire format 1:1.
 */

export type Approach =
  | 'classical'
  | 'library'
  | 'pretrained'
  | 'trained'
  | 'logic'
  | 'plumbing'

/**
 * Which fork of the pipeline a stage belongs to. The pipeline splits at colour
 * deconvolution: one deconvolution feeds both arms, handing the haematoxylin
 * channel to the model and the DAB channel to the measurement. A stage marked
 * shared runs before that fork, or — like deconvolution itself — serves both
 * sides of it.
 */
export type Branch = 'shared' | 'model' | 'measurement'

/**
 * Which slide of a case a stage reads.
 *
 * A case is two physical sections cut from one block: an H&E, where structure is
 * legible enough to find the tumour, and one immunostained slide per antibody,
 * where the brown that gets scored lives. `case` is a third answer rather than a
 * synonym for both — step 17 compares five markers of one block against four
 * readers of the same block, so a single slide is not a thing it can run on.
 */
export type SlideRole = 'he' | 'ihc' | 'case'

export interface PipelineStage {
  id: string
  index: number
  title: string
  tagline: string
  what: string
  whyHere: string
  how: string
  approach: Approach
  branch: Branch
  trainsModel: boolean
  inputLabel: string
  outputLabel: string
  actionLabel: string
  rule?: string | null
  references: string[]
  /** Whether this step actually runs. All seventeen do today. */
  implemented: boolean
  /** Which slide or slides this stage reads. See `readsSlidesTogether`. */
  runsOn: SlideRole[]
  /**
   * True when the stage needs both slides at the same moment rather than running
   * once per slide.
   *
   * This is the difference between two kinds of two-slide stage, and collapsing them
   * puts a dead control on screen: steps 2-6 have a separate answer per slide, so the
   * screen offers a switch; step 1 reads both pyramids so the readouts land side by
   * side, and step 10 registers one onto the other. Neither of those has a "which
   * slide" to choose.
   */
  readsSlidesTogether: boolean
}

export interface PipelineSummary {
  totalSteps: number
  implementedSteps: number
  trainedSteps: number
  pretrainedSteps: number
  classicalSteps: number
}

export interface UploadCapability {
  enabled: boolean
  reason: string
  acceptedFormats: string[]
  maxFileSizeMb: number
  chunkSizeBytes: number
}

export interface ListResponse<T> {
  items: T[]
  total: number
}

/** Where the stage catalogue on screen came from. */
export type CatalogSource = 'api' | 'bundled'
