import { apiGet } from './client'

import type {
  ListResponse,
  PipelineStage,
  PipelineSummary,
  UploadCapability,
} from '@/types/pipeline'

/** Fetch the ordered stage catalogue. */
export async function fetchStages(signal?: AbortSignal): Promise<PipelineStage[]> {
  const body = await apiGet<ListResponse<PipelineStage>>('/pipeline/stages', { signal })
  return body.items
}

/** Fetch the implemented / trained / pre-trained / classical counts. */
export function fetchSummary(signal?: AbortSignal): Promise<PipelineSummary> {
  return apiGet<PipelineSummary>('/pipeline/summary', { signal })
}

/** Fetch whether slide upload is accepted, and the limits that apply. */
export function fetchUploadCapability(signal?: AbortSignal): Promise<UploadCapability> {
  return apiGet<UploadCapability>('/slides/upload-capability', { signal })
}
