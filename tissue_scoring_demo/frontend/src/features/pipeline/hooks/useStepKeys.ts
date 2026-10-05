/**
 * Space walks the pipeline forward, backspace walks it back.
 *
 * The walkthrough is read one step at a time and the mouse is only ever used to hit
 * the one highlighted button at the bottom of a long column, so the keys are bound to
 * that same button rather than to their own idea of what "forward" means. The page
 * passes in the action it would run on a click; this hook only decides *when* a
 * keypress counts as one.
 *
 * Four things stop it, and each one is a way a naive `keydown` listener gets this
 * wrong:
 *
 * - **Typing.** A space in the filename box is a space, not a step. Anything editable
 *   keeps its keys.
 * - **A focused control.** Clicking a button focuses it, and the browser already
 *   activates a focused button on space. Advancing as well would fire the same action
 *   twice - so on a button, a link, a slider or a `<summary>`, this stands aside and
 *   lets the browser do it.
 * - **A held key.** `repeat` events would run the whole pipeline off the end of one
 *   long press.
 * - **A modifier.** Ctrl+space and friends belong to the browser and to the operating
 *   system, not to us.
 *
 * Passing `null` for either action disables that key, which is how a running step
 * stops space from starting a second one: the page hands over nothing while work is
 * in flight rather than this hook trying to guess the run's state.
 */

import { useEffect } from 'react'

interface StepKeys {
  /** What space should do, or null when there is nothing to advance to. */
  onAdvance: (() => void) | null
  /** What backspace should do, or null on the first step. */
  onBack: (() => void) | null
}

/** Fields where a keystroke is text the viewer is entering. */
const EDITABLE = 'input, textarea, select, [contenteditable="true"]'

/**
 * Things the browser itself activates on space. Left alone so a keypress produces one
 * action rather than two - see the note above.
 */
const ACTIVATABLE = `${EDITABLE}, button, a[href], summary, [role="button"], [role="switch"], [role="checkbox"], [role="tab"], [role="slider"]`

function matches(target: EventTarget | null, selector: string): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  return target.closest(selector) !== null
}

export function useStepKeys({ onAdvance, onBack }: StepKeys) {
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat) return
      if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return

      // `' '` in every browser this runs in; `'Spacebar'` is the old IE/Edge spelling
      // and costs one comparison to accept.
      if (event.key === ' ' || event.key === 'Spacebar') {
        if (!onAdvance || matches(event.target, ACTIVATABLE)) return
        // Space scrolls the step column by default, and that is exactly the reading
        // this replaces: the reader has finished the step and wants the next one.
        event.preventDefault()
        onAdvance()
        return
      }

      if (event.key === 'Backspace') {
        // Only text fields keep backspace. A focused button does not, because there
        // is nothing backspace does to a button that going back would be competing
        // with.
        if (!onBack || matches(event.target, EDITABLE)) return
        event.preventDefault()
        onBack()
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onAdvance, onBack])
}
