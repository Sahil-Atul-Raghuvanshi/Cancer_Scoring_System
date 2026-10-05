/** The five-antibody panel, fetched once at boot for the biomarker picker. */

import { apiGet } from './client'

import type { PanelResponse } from '@/types/panel'

export function fetchPanel(signal?: AbortSignal): Promise<PanelResponse> {
  return apiGet<PanelResponse>('/panel', { signal })
}
