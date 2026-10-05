/**
 * A slider whose thumb sits directly under the line it moves on the plot above.
 *
 * Two things stop a plain `<input type="range">` from lining up with a plot, and
 * this component exists because both of them matter here: a slider under a chart
 * that disagrees with the chart is not a control, it is a second opinion.
 *
 *   1. **The track has to be the same box as the plot.** An inline
 *      `[label] [track] [value]` row gives the track whatever width is left over
 *      after the text, so its ends are inset from the plot's by however wide the
 *      caption happens to be. Here the label and the value sit on their own row
 *      above, and the rail is a full-width block — the same box the plot is.
 *
 *   2. **A native thumb never reaches the ends.** Its centre travels from half a
 *      thumb-width in to half a thumb-width short, so even a full-width native
 *      track is out by ~7px at either extreme — which is exactly where a reader
 *      checks the alignment, because that is where the interesting cuts are. So
 *      the visible thumb is ours, positioned at `left: fraction%` with a centring
 *      translate, and the real input is laid transparently over the rail purely
 *      to keep pointer, keyboard and screen-reader behaviour.
 *
 * The caller passes a **fraction of the plot's own horizontal axis**, not a value,
 * and gets one back. That is the whole contract, and it is what makes this work on
 * an axis that is not linear in the quantity: step 4's percentile ladder is spaced
 * by the *rank* of its samples, so p95 sits halfway across the plot rather than
 * nine tenths of the way. A slider linear in percentile could only ever point
 * somewhere else. Converting at the edge, where the plot's mapping already lives,
 * means there is one mapping on the screen instead of two.
 */

import { useCallback } from 'react'

import './ui.css'

/**
 * Resolution of the underlying integer input.
 *
 * The input works in axis steps, not in the caller's units, so this is just how
 * finely a drag can land: a thousandth of the plot's width is well under a pixel
 * at any width this renders at, so the quantisation is never visible.
 */
const STEPS = 1000

interface TrackSliderProps {
  /** Sits above the rail, left. Rendered uppercase by the stylesheet. */
  label: string
  /** Where the thumb goes: 0 at the plot's left edge, 1 at its right. */
  fraction: number
  /** The current value, already formatted. Sits above the rail, right. */
  value: string
  /** What assistive tech should read instead of the raw axis step. */
  valueText: string
  ariaLabel: string
  /** Given a fraction of the plot's axis, 0 to 1. */
  onFraction: (fraction: number) => void
  /**
   * True while the value on screen is ahead of the run behind it. Tints the
   * thumb, so the control says "not yet" in the same instant the plot does.
   */
  pending?: boolean
}

export function TrackSlider({
  label,
  fraction,
  value,
  valueText,
  ariaLabel,
  onFraction,
  pending = false,
}: TrackSliderProps) {
  const clamped = Math.max(0, Math.min(1, fraction))
  const percent = `${(clamped * 100).toFixed(3)}%`

  const change = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      onFraction(Number(event.target.value) / STEPS)
    },
    [onFraction],
  )

  return (
    <div className="track">
      <div className="track__head">
        <span className="track__label">{label}</span>
        <output className="mono track__value">{value}</output>
      </div>

      {/* The input comes first so the decorations can be its CSS siblings - that
          is how focus, which lands on an invisible control, gets drawn on the
          thumb the viewer can actually see. They take no pointer events, so the
          input is still what a click and a drag reach. */}
      <div className="track__rail">
        <input
          className="track__input"
          type="range"
          min={0}
          max={STEPS}
          step={1}
          value={Math.round(clamped * STEPS)}
          onChange={change}
          aria-label={ariaLabel}
          aria-valuetext={valueText}
        />
        <span className="track__fill" style={{ width: percent }} aria-hidden />
        <span
          className={pending ? 'track__thumb track__thumb--pending' : 'track__thumb'}
          style={{ left: percent }}
          aria-hidden
        />
      </div>
    </div>
  )
}
