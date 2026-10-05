/**
 * Step 7's overlap setting, and what it means.
 *
 * There used to be three buttons here — none, a quarter, a half. There is one
 * now: this pipeline tiles edge to edge, so the setting is stated rather than
 * chosen. It is still rendered as the picker it was, with the tile count in it,
 * because that count is the thing the reader is here for — it is the literal
 * number of forward passes step 8 will make — and because the shape leaves room
 * for a second setting to come back without the screen being rebuilt around it.
 *
 * The count appears once the run that measured it has landed. Before that its
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
    note: 'Each square is processed once, and the squares meet edge to edge',
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
      <h3 className="tl-card__title">How much do the squares overlap?</h3>
      <p className="tl-card__lead">
        They do not. The squares sit side by side, so every piece of tissue is
        processed exactly once. Overlapping would give the model a second look at the
        borders, where it is least reliable, but it would also multiply the number of
        squares and the time the next step takes.
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
        The cost of not overlapping is faint seams: neighbouring squares can disagree
        about the tissue right on their shared border. We accept that to save time.
      </p>

      <p className="tl-card__note">
        These are the same squares the next step reads, one at a time, and that step is
        the slowest part of the run. So the count above is not an estimate of the work
        &mdash; it <em>is</em> the work: roughly ten minutes on this machine.
      </p>
    </section>
  )
}
