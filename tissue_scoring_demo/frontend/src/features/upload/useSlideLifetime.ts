/**
 * Tells the server this page is holding a slide, and that it has gone away.
 *
 * **Off unless the server says otherwise**, and the server ships it off. The whole
 * feature is a guess about intent: `pagehide` fires when a tab closes, when it is
 * refreshed, and when the viewer follows a link away, and nothing in the browser
 * distinguishes them. Guessing wrong costs half an hour of step 8 to save about 19 MB.
 *
 * What makes it survivable is that the guess is *recoverable*. `/release` only schedules;
 * `/claim` cancels. A page that reloads claims its slide within a second, well inside the
 * grace period, so a refresh undoes the release it just caused and only a genuine close
 * runs it. That is the design - not the beacon, which is merely how a closing page manages
 * to reach a server at all.
 *
 * **Why `sendBeacon` and not `fetch`.** A page in `pagehide` is being torn down: a normal
 * `fetch` is routinely cancelled mid-flight, and `keepalive` is not honoured everywhere.
 * `sendBeacon` hands the request to the browser to deliver after the page is gone. It
 * cannot set headers, cannot be awaited and its response is discarded - hence a POST with
 * everything in the query string and an endpoint that returns 200 whatever happens.
 *
 * **`pagehide`, not `beforeunload`.** `beforeunload` is unreliable on mobile Safari and
 * blocks the back/forward cache; `pagehide` fires in both the discard and the cache case
 * and is what the platform now recommends.
 */

import { useEffect } from 'react'

import { API_PREFIX } from '@/api/client'
import { claimSlide, fetchCleanupPolicy } from '@/api/maintenance'

const BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

/**
 * Claim `uploadId` while this page is open, and release it when the page goes away.
 *
 * Does nothing at all when no slide is loaded, or when the server has the feature off -
 * in which case no handler is even registered, so there is no cost to shipping this.
 */
export function useSlideLifetime(uploadId: string | null): void {
  useEffect(() => {
    if (!uploadId) return

    const controller = new AbortController()
    let release: (() => void) | null = null

    fetchCleanupPolicy(controller.signal)
      .then((policy) => {
        if (controller.signal.aborted || !policy.cleanupOnDisconnect) return

        // Claim first, unconditionally. If this page is a refresh of one that just
        // released the slide, this is the call that saves its results - so it goes out
        // before anything else and its failure is not fatal.
        void claimSlide(uploadId).catch(() => {})

        const url =
          `${BASE_URL}${API_PREFIX}/maintenance/release` +
          `?uploadId=${encodeURIComponent(uploadId)}`

        const onHide = () => {
          // `persisted` means the page went into the back/forward cache rather than
          // being discarded - it can come back without a reload, and it would not
          // re-claim if it did. Releasing then would be a wrong guess this design
          // cannot recover from, so it is skipped.
          navigator.sendBeacon?.(url)
        }

        const handler = (event: PageTransitionEvent) => {
          if (!event.persisted) onHide()
        }

        window.addEventListener('pagehide', handler)
        release = () => window.removeEventListener('pagehide', handler)
      })
      .catch(() => {
        // The server did not answer. Registering nothing is the right failure: an
        // unreachable policy endpoint is no reason to start deleting things.
      })

    return () => {
      controller.abort()
      release?.()
    }
  }, [uploadId])
}
