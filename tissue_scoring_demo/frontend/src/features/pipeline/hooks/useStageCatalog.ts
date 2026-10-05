import { useEffect, useState } from 'react'

import { fetchStages } from '@/api/pipeline'
import type { CatalogSource, PipelineStage } from '@/types/pipeline'

import { BUNDLED_STAGES } from '../data/stages'

interface CatalogState {
  stages: PipelineStage[]
  /** Whether the catalogue on screen came from the API or the bundle. */
  source: CatalogSource
  loading: boolean
}

/**
 * Load the stage catalogue from the backend, falling back to the bundled copy.
 *
 * The walkthrough is UI content, so it must still run with the API down — the
 * fallback is deliberate, and the source is surfaced so the header can say
 * which one is in use rather than quietly pretending.
 */
export function useStageCatalog(): CatalogState {
  const [state, setState] = useState<CatalogState>({
    stages: BUNDLED_STAGES,
    source: 'bundled',
    loading: true,
  })

  useEffect(() => {
    const controller = new AbortController()

    fetchStages(controller.signal)
      .then((stages) => {
        if (controller.signal.aborted) return
        setState({ stages, source: 'api', loading: false })
      })
      .catch(() => {
        if (controller.signal.aborted) return
        setState({ stages: BUNDLED_STAGES, source: 'bundled', loading: false })
      })

    return () => controller.abort()
  }, [])

  return state
}
