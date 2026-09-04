/**
 * Step 7's one control, and the trade it represents.
 *
 * Three buttons rather than a slider. Overlap is in practice a choice between a
 * few settings — the guide's band is 25–50% — and every value between them means
 * the same thing, so a continuous control would suggest a precision that is not
 * there and would need debouncing to avoid firing a rebuild per pixel of drag.
 *
 * Each button carries the tile count it produces once that setting has been
 * visited, because the count *is* the cost and a reader choosing between them
 * should be choosing with the price visible. Before a setting has been tried its
 * slot stays empty rather than showing an estimate: an estimated tile count would
 * be indistinguishable on screen from a measured one.
 */

import { formatCount } from '@/lib/format'

import { OVERLAP_CHOICES } from './useTiling'

interface OverlapPickerProps {
  /** The overlap the report on screen was built at. */
  current: number
  /** Tile counts already measured, keyed by overlap. */
  known: Map<number, number>
  refining: boolean
  onPick: (overlap: number) => void
}

// Keyed on the literal union rather than on `number`, so adding a choice to
// `OVERLAP_CHOICES` without labelling it is a compile error rather than an
// undefined reaching the screen.
const LABELS: Record<(typeof OVERLAP_CHOICES)[number], { name: string; note: string }> = {
  0: {
    name: 'No overlap',
    note: 'Fewest squares, but the model has to guess at every edge',
  },
  0.25: {
    name: 'Quarter overlap',
    note: 'The usual choice — edges get a second look without much extra cost',
  },
  0.5: {
    name: 'Half overlap',
    note: 'Every edge well covered, but roughly four times the work',
  },
}

export function OverlapPicker({
  current,
  known,
  refining,
  onPick,
}: OverlapPickerProps) {
  return (
    <section className="tl-card">
      <h3 className="tl-card__title">How much should the squares overlap?</h3>
      <p className="tl-card__lead">
        The model cannot see past the edge of a square, so its answers are least
        reliable right at the borders. Letting neighbouring squares overlap means
        every border is also somewhere near the middle of another square, and the
        two answers can be averaged &mdash; which is what stops the finished map
        looking like a tiled floor. It costs more squares to process, though, so
        it is a real trade:
      </p>

      <div
        className="tl-picker"
        role="group"
        aria-label="How much neighbouring squares overlap"
      >
        {OVERLAP_CHOICES.map((choice) => {
          const active = Math.abs(choice - current) < 1e-6
          const count = known.get(choice)
          return (
            <button
              key={choice}
              type="button"
              className={active ? 'tl-picker__pick is-active' : 'tl-picker__pick'}
              aria-pressed={active}
              disabled={refining}
              onClick={() => onPick(choice)}
            >
              <span className="tl-picker__name">{LABELS[choice].name}</span>
              <span className="tl-picker__count mono">
                {count === undefined ? '—' : formatCount(count)}
              </span>
              <span className="tl-picker__unit">squares to process</span>
              <span className="tl-picker__note">{LABELS[choice].note}</span>
            </button>
          )
        })}
      </div>

      <p className="tl-card__note">
        Notice that the amount of slide covered barely changes between these
        &mdash; overlapping does not let the model see <em>more</em> of the slide,
        it lets it see the same tissue more than once. That is the whole trade:
        better answers at the seams, paid for in processing time.
      </p>

      <p className="tl-card__note">
        These are the same squares step 8 will show the model, one at a time, and that
        step is the slowest part of the whole run. So the count above is not an
        estimate of the work &mdash; it <em>is</em> the work: on this machine, roughly
        ten minutes with no overlap and about half an hour at a half.
      </p>
    </section>
  )
}
