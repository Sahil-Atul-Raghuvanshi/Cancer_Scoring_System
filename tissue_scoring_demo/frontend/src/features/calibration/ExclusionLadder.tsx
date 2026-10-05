/**
 * The six exclusions, in order, with what each one cost.
 *
 * Modelled on step 3's cleanup ladder, and different in one way that matters:
 * step 3's moves pull in opposite directions - closing adds area, opening removes
 * it - whereas every move here only ever removes. So the bar shrinks monotonically
 * and the interesting fact is not *whether* a move cost anything but *which* move
 * cost the most.
 *
 * That is worth showing rather than summarising, because on the demo's slides the
 * answer is surprising: the exclusion that removes almost everything is the
 * scanner's background fill, which is not in the pipeline guide's description of
 * this step at all. A reader who sees one bar collapse by 88% has learnt
 * something about their scanner that no summary figure would have told them.
 */

import { formatCount } from '@/lib/format'
import type { CalibrationExclusion } from '@/types/calibration'

interface ExclusionLadderProps {
  exclusions: CalibrationExclusion[]
}

export function ExclusionLadder({ exclusions }: ExclusionLadderProps) {
  const widest = Math.max(...exclusions.map((step) => step.pixels), 1)

  return (
    <div className="excl">
      <span className="eyebrow">what counts as blank glass, step by step</span>

      <ol className="excl__list">
        {exclusions.map((step, index) => {
          // A step that removed nothing is not a step that did nothing wrong -
          // often it is a step that was skipped, and its own `what` says which.
          const removed = step.deltaPixels < 0

          return (
            <li className="excl__row" key={step.key}>
              <div className="excl__head">
                <span className="excl__index mono">{index + 1}</span>
                <strong>{step.label}</strong>
                {step.extentUm !== null && (
                  <span className="excl__extent mono">{step.extentUm}&nbsp;µm</span>
                )}
              </div>

              <p className="excl__what">{step.what}</p>

              <div className="excl__track">
                <span
                  className="excl__fill"
                  style={{ width: `${(step.pixels / widest) * 100}%` }}
                />
              </div>

              <div className="excl__figures mono">
                <span className="excl__area">{step.areaMm2.toFixed(2)} mm²</span>
                <span className="excl__pixels">{formatCount(step.pixels)} px</span>
                {index === 0 ? (
                  <span className="excl__delta excl__delta--base">
                    starting point
                  </span>
                ) : (
                  <span
                    className={
                      removed ? 'excl__delta excl__delta--down' : 'excl__delta excl__delta--base'
                    }
                  >
                    {removed
                      ? `−${Math.abs(step.deltaAreaMm2).toFixed(2)} mm² (−${Math.abs(
                          step.deltaShare * 100,
                        ).toFixed(1)}%)`
                      : 'nothing to remove'}
                  </span>
                )}
              </div>
            </li>
          )
        })}
      </ol>

    </div>
  )
}
