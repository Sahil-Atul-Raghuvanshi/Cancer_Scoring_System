/**
 * The `H&E | IHC` switch on steps 2 to 6.
 *
 * Those five steps are a per-slide prefix: each measures a property of one piece of
 * glass, so each has an answer for both slides of a case and the two answers are
 * different. Step 4 is the clearest - a white point is per slide by definition, and
 * the guide asks for the two to be shown together because that comparison *is* the
 * step's argument.
 *
 * Shaped after `features/deconvolution/BasisToggle.tsx` rather than the tab strips
 * elsewhere, because this is the same kind of control: two things that are meant to
 * be compared, each carrying the number that differs, not two views of one thing.
 */

import { cn } from '@/lib/cn'

import type { SlideOption } from './slideChoice'
import './slideRole.css'

interface SlidePickerProps {
  options: SlideOption[]
  selected: SlideOption | null
  onChoose: (role: 'he' | 'ihc') => void
  /** A headline per slide - the I0, the tissue share - so the switch shows the difference. */
  summary?: (slide: SlideOption) => string | null
  /**
   * Whether each slide's step has run yet.
   *
   * A per-slide step runs on both slides as one unit of work, so while it is going the
   * viewer is watching one slide and the other is either finished or still to come.
   * Without this the only way to find out which was to click the other tab, which is
   * exactly the kind of hidden state the badge and this picker exist to remove.
   */
  state?: (slide: SlideOption) => 'done' | 'running' | 'pending'
}

export function SlidePicker({
  options,
  selected,
  onChoose,
  summary,
  state,
}: SlidePickerProps) {
  if (options.length < 2) return null

  return (
    <div className="slide-pick">
      <div className="slide-pick__row" role="group" aria-label="Which slide to show">
        {options.map((slide) => {
          const active = slide.role === selected?.role
          const headline = summary?.(slide) ?? null
          const progress = state?.(slide) ?? 'pending'
          return (
            <button
              key={slide.role}
              type="button"
              className={cn('slide-pick__option', active && 'is-active')}
              aria-pressed={active}
              onClick={() => onChoose(slide.role)}
            >
              <span className="slide-pick__name">
                {slide.role === 'ihc' ? `${slide.label} IHC` : slide.label}
                <span
                  className={cn('slide-pick__state', `slide-pick__state--${progress}`)}
                  title={
                    progress === 'done'
                      ? 'this step has finished on this slide'
                      : progress === 'running'
                        ? 'running on this slide now'
                        : 'not run on this slide yet'
                  }
                >
                  {progress === 'done' ? 'done' : progress === 'running' ? 'running' : 'waiting'}
                </span>
              </span>
              {headline ? <span className="slide-pick__value mono">{headline}</span> : null}
              <span className="slide-pick__hint">
                {slide.role === 'ihc'
                  ? 'the slide the score is measured on'
                  : 'the slide the tumour is found on'}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
