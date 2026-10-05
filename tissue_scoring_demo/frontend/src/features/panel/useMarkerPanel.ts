import { useEffect, useState } from 'react'

import { fetchPanel } from '@/api/panel'
import type { MarkerInfo } from '@/types/panel'

/** The five-antibody panel plus H&E, fetched once. */
export function useMarkerPanel() {
  const [markers, setMarkers] = useState<MarkerInfo[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    fetchPanel(controller.signal)
      .then((response) => {
        if (!controller.signal.aborted) setMarkers(response.markers)
      })
      .catch((cause) => {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : 'could not load the marker panel')
        }
      })
    return () => controller.abort()
  }, [])

  return { markers, error }
}
