/**
 * "Take me to where the cells are."
 *
 * Steps 11 to 15 all pan a whole slide on which the measured cells occupy a
 * fraction of a percent of the area - a few dozen sampled squares inside a few
 * regions of invasive tumour, on a section that is mostly fat, stroma and glass.
 * Zoomed out you correctly see none of them; zoomed in at a random spot you also
 * correctly see none of them. Without a way to jump, the screens are a hunt.
 *
 * So every one of these steps gets the same row of chips, built from the overlay
 * it is already drawing rather than from a second source: the regions that have
 * shapes in them, largest first, and the square inside each that holds the most.
 * A region with nothing drawn is not offered, because flying to it would land on
 * the same empty tissue the reader just left.
 */

import type { FlyTarget } from '@/features/cells/overlay'

interface FlyToProps {
  targets: FlyTarget[]
  /** The rank currently flown to, or null for the whole slide. */
  active: number | null
  onFly: (target: FlyTarget | null) => void
  /** What the counts are counting, for the tooltip. */
  noun?: string
}

export function FlyTo({ targets, active, onFly, noun = 'cells' }: FlyToProps) {
  if (targets.length === 0) return null

  return (
    <div className="nuc-jump">
      <span className="nuc-jump__label">Jump to</span>
      {targets.map((target) => (
        <button
          key={target.rank}
          type="button"
          className={target.rank === active ? 'nuc-chip is-active' : 'nuc-chip'}
          onClick={() => onFly(target)}
          title={`${target.count.toLocaleString()} ${noun} in region ${target.rank}`}
        >
          Region {target.rank}
          <span className="nuc-chip__sub mono">{target.count.toLocaleString()}</span>
        </button>
      ))}
      <button
        type="button"
        className={active === null ? 'nuc-chip is-active' : 'nuc-chip'}
        onClick={() => onFly(null)}
        title="Zoom back out to the whole slide"
      >
        Whole slide
      </button>
    </div>
  )
}
