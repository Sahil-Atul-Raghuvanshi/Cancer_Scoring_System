/** The viewer control icons, drawn rather than set in type. */

/**
 * The three control icons, drawn rather than set in type.
 *
 * "Fit the whole slide" and "go fullscreen" are different actions that both mean
 * "bigger view" to a reader skimming glyphs, and a single ⤢ next to a + and a −
 * gets read as fullscreen every time. So the fit control shows a frame with the
 * slide sitting inside it, and the fullscreen pair shows corner brackets -
 * opening outwards to enter, inwards to leave.
 */
export const VIEWER_ICON = {
  fit: (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden focusable="false">
      <rect
        x="1.6"
        y="2.6"
        width="12.8"
        height="10.8"
        rx="1.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        opacity="0.55"
      />
      <rect x="5.2" y="6.2" width="5.6" height="3.6" rx="1" fill="currentColor" />
    </svg>
  ),
  enter: (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden focusable="false">
      <path
        d="M2.6 6.2V2.6h3.6M13.4 6.2V2.6H9.8M2.6 9.8v3.6h3.6M13.4 9.8v3.6H9.8"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  ),
  exit: (
    <svg viewBox="0 0 16 16" width="15" height="15" aria-hidden focusable="false">
      <path
        d="M6.2 2.6v3.6H2.6M9.8 2.6v3.6h3.6M6.2 13.4V9.8H2.6M9.8 13.4V9.8h3.6"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  ),
} as const
