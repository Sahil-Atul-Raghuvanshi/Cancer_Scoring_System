/**
 * The "this block is being recomputed" marker.
 *
 * Every screen in this app keeps the same contract: the numbers on it describe a
 * run the server really performed, never the control's current position. That
 * contract is only legible if the gap between the two is visible, so a block whose
 * figures are stale says so — and stops saying so the instant the new report
 * lands, which is what makes the freshly highlighted value readable as an answer
 * rather than as something that was always there.
 *
 * Deliberately not a full-block skeleton. The stale numbers stay on screen and
 * stay readable: they are the previous run's real answer, and covering them would
 * throw away the comparison the viewer is in the middle of making.
 */

import './ui.css'

interface SpinnerProps {
  /** Sits beside the ring. Keep it to a few words — it is read mid-drag. */
  label?: string
  size?: 'sm' | 'md'
}

export function Spinner({ label, size = 'md' }: SpinnerProps) {
  return (
    <span className={`spinner spinner--${size}`} role="status">
      <span className="spinner__ring" aria-hidden />
      {label && <span className="spinner__label">{label}</span>}
    </span>
  )
}
