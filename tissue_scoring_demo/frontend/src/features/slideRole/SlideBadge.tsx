/**
 * Names the slide a panel's pictures came from.
 *
 * Every panel gets one. That sounds like decoration and is not: a case is two
 * physical sections, most steps read one of them, and until this existed the only
 * screen in the whole walkthrough that named a slide against its picture was step
 * 10's alignment pair. Everywhere else a reader had to know the pipeline to know
 * what they were looking at - and on step 6 what they were looking at was an H&E
 * tile under a caption about the brown the score is measured from.
 */

import { cn } from '@/lib/cn'

import type { SlideOption } from './slideChoice'
import './slideRole.css'

interface SlideBadgeProps {
  slide: SlideOption | null
  /** Step 17 reads no slide at all; it is keyed on the case. */
  caseId?: string | null
  isCaseScoped?: boolean
  /**
   * Every slide the step reads, for the two steps that read both at once.
   *
   * Step 1 opens both pyramids and step 10 registers one onto the other, so naming
   * only the first would be the same half-truth this component exists to remove -
   * it would put "H&E" on a screen showing the alignment of two slides.
   */
  both?: SlideOption[]
  className?: string
}

export function SlideBadge({
  slide,
  caseId,
  isCaseScoped,
  both,
  className,
}: SlideBadgeProps) {
  if (both && both.length > 1) {
    return (
      <span className={cn('slide-badge', 'slide-badge--both', className)}>
        <span className="slide-badge__role">
          {both
            .map((entry) => (entry.role === 'ihc' ? `${entry.label} IHC` : entry.label))
            .join(' + ')}
        </span>
      </span>
    )
  }

  if (isCaseScoped) {
    return (
      <span className={cn('slide-badge', 'slide-badge--case', className)}>
        <span className="slide-badge__role">Whole case</span>
        {caseId ? <span className="slide-badge__file mono">{caseId}</span> : null}
      </span>
    )
  }

  if (!slide) return null

  return (
    <span
      className={cn(
        'slide-badge',
        slide.role === 'ihc' ? 'slide-badge--ihc' : 'slide-badge--he',
        slide.suspect && 'slide-badge--suspect',
        className,
      )}
      title={slide.filename ?? undefined}
    >
      <span className="slide-badge__role">
        {slide.role === 'ihc' ? `${slide.label} IHC` : slide.label}
      </span>
      {slide.filename ? <span className="slide-badge__file mono">{slide.filename}</span> : null}
    </span>
  )
}
