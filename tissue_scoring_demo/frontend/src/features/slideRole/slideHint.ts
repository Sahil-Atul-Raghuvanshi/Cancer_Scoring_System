/**
 * The one-line hint under a step that has no hint file of its own.
 *
 * Steps 2 to 9 each ship a `*Status.ts` that says something specific. Steps 10 to
 * 17 do not, so they fell through to a generic pair of sentences - and one of them,
 * "Reads the file you uploaded", was simply false for six steps in a row. The file
 * you uploaded is the H&E; steps 11 to 16 read the immunostained slide, and step 17
 * reads neither. A reader who believed that line would think the score came off the
 * slide they had just watched the tumour being found on.
 *
 * Naming the slide is the fix, and it costs nothing to keep true, because the name
 * comes from the same `runsOn` declaration the badge and the toggle are drawn from.
 */

import type { SlideChoice } from './slideChoice'

export function slideHint(choice: SlideChoice, complete: boolean): string {
  if (choice.isCaseScoped) {
    return complete
      ? 'Every number above covers the whole case, not one slide.'
      : 'Looks at the whole case — every marker, against every pathologist.'
  }

  const slide = choice.selected
  if (!slide) {
    return complete
      ? 'Every number above was measured from your slide.'
      : 'Reads the slide you loaded.'
  }

  const name = slide.role === 'ihc' ? `${slide.label} marker slide` : 'H&E slide'
  return complete
    ? `Every number above was measured from the ${name}.`
    : `Reads the ${name}.`
}
