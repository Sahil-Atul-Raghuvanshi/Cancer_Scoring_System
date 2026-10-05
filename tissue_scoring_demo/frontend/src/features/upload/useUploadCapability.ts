import { useCallback, useEffect, useState } from 'react'

import { fetchUploadCapability } from '@/api/pipeline'
import type { UploadCapability } from '@/types/pipeline'

export interface UploadCapabilityState {
  /** The limits the server will accept, once it has told us what they are. */
  capability: UploadCapability | null
  /** Still waiting on the first answer. */
  probing: boolean
  /** Ask again — the backend may have come up since. */
  retry: () => void
}

/**
 * Ask the backend what it will accept for an upload.
 *
 * There is no fallback here, unlike the stage catalogue: the limits belong to
 * the server that will do the storing, so inventing them would mean offering a
 * drop zone that cannot work. What the failure must not do is look like
 * loading — hence `probing` resolving to false either way, so the caller can
 * say the API is down instead of shimmering forever.
 */
export function useUploadCapability(): UploadCapabilityState {
  const [capability, setCapability] = useState<UploadCapability | null>(null)
  const [probing, setProbing] = useState(true)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setProbing(true)

    fetchUploadCapability(controller.signal)
      .then((next) => {
        if (controller.signal.aborted) return
        setCapability(next)
        setProbing(false)
      })
      .catch(() => {
        if (controller.signal.aborted) return
        setCapability(null)
        setProbing(false)
      })

    return () => controller.abort()
  }, [attempt])

  const retry = useCallback(() => setAttempt((count) => count + 1), [])

  return { capability, probing, retry }
}
