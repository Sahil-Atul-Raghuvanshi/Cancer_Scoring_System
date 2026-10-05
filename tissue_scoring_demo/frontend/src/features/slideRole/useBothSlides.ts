/**
 * Run one step on every slide it declares, in a fixed order.
 *
 * A step whose `runsOn` names two slides is not "a step you can point at either
 * slide" - it is a step with two answers, both of which the pipeline needs. Step 4
 * is the plain case: step 14 divides the IHC slide's DAB by the IHC slide's white
 * point, so a run that produced only the H&E's has not half-finished step 4, it has
 * left the measurement arm without its denominator.
 *
 * That is why this returns one callable rather than letting the screen start whichever
 * slide is on top. The pipeline marks a step complete when its task resolves, so
 * folding both slides into one task is what makes "step 3 is done" mean "step 3 is
 * done on both", and therefore what lets step 4 refuse to start until it is true.
 *
 * **H&E first, always, and for quality control it is load-bearing.** `qc_service`
 * holds a single global run lock and refuses a second run while any run is going, so
 * the two QC passes have to be sequential rather than raced. Awaiting in order costs
 * nothing on the cheap steps and is the only thing that works on the expensive one.
 */

import { useCallback } from 'react'

type Start = () => Promise<void>

export function useBothSlides(heStart: Start, ihcStart: Start, hasIhcSlide: boolean): Start {
  return useCallback(async () => {
    await heStart()
    // A lone uploaded file has no second slide. Not an error - the walkthrough
    // supports one slide, it just cannot reach step 10.
    if (hasIhcSlide) await ihcStart()
  }, [hasIhcSlide, heStart, ihcStart])
}
