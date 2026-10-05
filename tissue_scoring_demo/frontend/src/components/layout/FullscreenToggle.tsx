import { useCallback, useEffect, useState } from 'react'

function ExpandIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M9 4H4v5M15 4h5v5M15 20h5v-5M9 20H4v-5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function CollapseIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 9h5V4M20 9h-5V4M20 15h-5v5M4 15h5v5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/**
 * Take the whole screen, and give it back.
 *
 * Worth a button rather than leaving it to F11: the pipeline rail and the step
 * panel both size themselves off the viewport, so the browser chrome costs
 * real pipeline steps on the rail and real height in the slide viewer.
 */
export function FullscreenToggle() {
  // Read once, lazily: whether the document may go fullscreen at all is fixed
  // by the embedding, not by anything that happens while the app runs.
  const [supported] = useState(() => typeof document !== 'undefined' && document.fullscreenEnabled)
  const [active, setActive] = useState(false)

  // The browser owns this state — Escape and F11 change it without asking us —
  // so it is mirrored from the event, never assumed from the click.
  useEffect(() => {
    const sync = () => setActive(document.fullscreenElement !== null)
    sync()
    document.addEventListener('fullscreenchange', sync)
    return () => document.removeEventListener('fullscreenchange', sync)
  }, [])

  const toggle = useCallback(() => {
    // A refused request leaves the page exactly as it was and fires no event,
    // so there is nothing to roll back — only a rejection to not crash on.
    const request = document.fullscreenElement
      ? document.exitFullscreen()
      : document.documentElement.requestFullscreen()
    void request.catch(() => undefined)
  }, [])

  if (!supported) return null

  return (
    <button
      type="button"
      className="header__action"
      onClick={toggle}
      aria-pressed={active}
      title={active ? 'Leave fullscreen' : 'Fullscreen'}
      aria-label={active ? 'Leave fullscreen' : 'Enter fullscreen'}
    >
      {active ? <CollapseIcon /> : <ExpandIcon />}
    </button>
  )
}
