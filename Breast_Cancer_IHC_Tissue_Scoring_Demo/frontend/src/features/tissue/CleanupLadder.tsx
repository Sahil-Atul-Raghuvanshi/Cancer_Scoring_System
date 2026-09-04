/**
 * The cleanup, one move at a time, with what each one changed.
 *
 * A single before-and-after would hide the interesting fact: closing and opening
 * pull in opposite directions, and on a fragmented specimen closing can add more
 * area than the threshold found in the first place. Showing the ladder makes each
 * morphological step a claim the reader can check rather than a word in a list.
 *
 * Every extent is shown in microns with its pixel equivalent beside it, because
 * the pixel number is the one that changes on a different scanner and the micron
 * number is the one that does not.
 */

import { formatCount } from '@/lib/format'
import type { TissueStage } from '@/types/tissue'

interface CleanupLadderProps {
  stages: TissueStage[]
}

export function CleanupLadder({ stages }: CleanupLadderProps) {
  const widest = Math.max(...stages.map((stage) => stage.pixels), 1)

  return (
    <div className="tissue-ladder">
      <span className="eyebrow">the cleanup, in order</span>

      <ol className="tissue-ladder__list">
        {stages.map((stage, index) => {
          const grew = stage.deltaPixels > 0
          const changed = stage.deltaPixels !== 0

          return (
            <li className="tissue-ladder__row" key={stage.key}>
              <div className="tissue-ladder__head">
                <span className="tissue-ladder__index mono">{index + 1}</span>
                <strong>{stage.label}</strong>
                {stage.extentUm !== null && (
                  <span className="tissue-ladder__extent mono">
                    {stage.extentUm}&nbsp;µm
                  </span>
                )}
              </div>

              <p className="tissue-ladder__what">{stage.what}</p>

              <div className="tissue-ladder__track">
                <span
                  className="tissue-ladder__fill"
                  style={{ width: `${(stage.pixels / widest) * 100}%` }}
                />
              </div>

              <div className="tissue-ladder__figures mono">
                <span className="tissue-ladder__area">{stage.areaMm2.toFixed(2)} mm²</span>
                <span className="tissue-ladder__pixels">
                  {formatCount(stage.pixels)} px
                </span>
                {index === 0 ? (
                  <span className="tissue-ladder__delta tissue-ladder__delta--base">
                    starting point
                  </span>
                ) : (
                  <span
                    className={
                      changed
                        ? grew
                          ? 'tissue-ladder__delta tissue-ladder__delta--up'
                          : 'tissue-ladder__delta tissue-ladder__delta--down'
                        : 'tissue-ladder__delta tissue-ladder__delta--base'
                    }
                  >
                    {changed
                      ? `${grew ? '+' : '−'}${Math.abs(stage.deltaAreaMm2).toFixed(2)} mm² (${
                          grew ? '+' : '−'
                        }${Math.abs(stage.deltaShare * 100).toFixed(1)}%)`
                      : 'no change'}
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
