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
  /** Whether this step actually runs. Steps 1 to 5 do today. */
  implemented: boolean
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
