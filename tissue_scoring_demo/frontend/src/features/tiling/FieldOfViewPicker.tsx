/**
 * NO LONGER RENDERED, as of 2026-09-15. Kept, not deleted.
 *
 * Step 7 used to ask two questions - which branch, and at what field of view - and
 * this was one of them. The pipeline has since committed to the H&E branch at 224 um
 * (`COMMITTED_BRANCH` in `useTiling`, which carries the measurements behind both
 * halves of that decision), so the screen states the answer instead of collecting it.
 *
 * This file stays because re-opening the choice is a decision someone may well make
 * again - a new head at a new scale, or evidence that changes which branch to trust -
 * and putting a file back is cheaper than writing it. The hook still exposes
 * `chooseBranch`, `chooseFieldOfView`, `backToBranch` and `backToFov` for the same
 * reason. Nothing imports this; deleting it would lose the only worked example of how
 * the choice was presented.
 */
/**
 * Step 7's real choice: how much slide fits inside one square.
 *
 * This is not the same kind of control as the overlap beside it, and the screen has
 * to say so. Overlap moves the price. This moves **which model runs** — a field of
 * view is a property of the weights, so the head fitted at 224 µm and the head fitted
 * at 448 µm are two different models, and choosing the scale *is* choosing between
 * them. That is the whole reason it is a decision rather than a setting.
 *
 * The reason it is a decision worth putting in front of a reader is a measured
 * failure, and the copy says it plainly rather than in a disclosure: a duct is
 * 300–1500 µm across, so a window smaller than that cannot show a duct wall, and a
 * model that cannot see the wall has no way to tell carcinoma still inside a duct
 * from carcinoma that has broken out. That is the difference the whole test rests on.
 *
 * An option with no published checkpoint is shown, disabled, saying so. Hiding it
 * would make the choice look smaller than it is, and a reader comparing scales needs
 * to see the one that is coming.
 */

import { formatCount } from '@/lib/format'
import type { TilingBranchName, TilingFieldOfView } from '@/types/tiling'

import { FIELD_OF_VIEW_CHOICES } from './useTiling'

interface FieldOfViewPickerProps {
  /** The field of view the report on screen was built at, or null before the first run. */
  current: number | null
  /** Which option this is picking a scale for. Availability differs between them. */
  branch: TilingBranchName
  /** What the server offers on that branch, and which of them has a checkpoint. */
  offered: TilingFieldOfView[]
  /** Square counts already measured, keyed `${branch}:${um}`. */
  known: Map<string, number>
  refining: boolean
  onPick: (um: number) => void
}

// Keyed on the literal union rather than on `number`, so adding a choice to
// `FIELD_OF_VIEW_CHOICES` without labelling it is a compile error rather than an
// undefined reaching the screen.
const LABELS: Record<
  (typeof FIELD_OF_VIEW_CHOICES)[number],
  { name: string; note: string }
> = {
  112: {
    name: '112 µm',
    note: 'About eleven nuclei across — the cells are as sharp as they get here, but no duct wall will fit',
  },
  224: {
    name: '224 µm',
    note: 'About twenty nuclei across — a small duct fits, a large one does not',
  },
  448: {
    name: '448 µm',
    note: 'Four times the ground per square, so most ducts fit whole — but each square is read at half the detail',
  },
  672: {
    name: '672 µm',
    note: 'Nine times the ground of 224 µm, so even a large duct fits whole — at a third of the detail inside it',
  },
}

/**
 * Why a scale is unavailable, said in terms of the option that is unavailable.
 *
 * The two branches of ours are unavailable *per scale*, because a scale is a checkpoint
 * and one may be trained while another is not. BEETLE is not: one release covers all
 * four extents at its own fixed 0.5 µm/px, so its four scales are available together or
 * none is, and the only thing that can make them unavailable is the archive being
 * absent. Saying "not developed yet" there was true when the branch was a placeholder
 * and is now the one wrong answer — it would send a reader looking for training that is
 * never going to happen instead of for a download.
 *
 * Note that none of these three is the *slide's* reason. Whether this section can take a
 * branch at all — both colour branches need step 5 to have found a second dye — is
 * answered one level up, on the branch itself, and a reader never reaches this picker for
 * a branch that failed there. What is left for this screen is only which scales the
 * chosen branch can be run at.
 */
const UNAVAILABLE: Record<TilingBranchName, string> = {
  h_channel: 'no model trained yet',
  he: 'H&E model not trained yet at this scale',
  beetle: 'BEETLE’s weights are not installed',
}

export function FieldOfViewPicker({
  current,
  branch,
  offered,
  known,
  refining,
  onPick,
}: FieldOfViewPickerProps) {
  const byUm = new Map(offered.map((entry) => [Math.round(entry.um), entry]))

  return (
    <section className="tl-card">
      <h3 className="tl-card__title">How much slide fits in one square?</h3>
      <p className="tl-card__lead">
        This is the one choice on this screen that changes the answer rather than the
        price. Step 8 has to tell carcinoma that is still inside a duct from carcinoma
        that has broken out of one, and the difference between those is the duct
        wall &mdash; so a square has to be big enough to contain it. A duct runs 300 to
        1500 µm across. A square smaller than that, landed in the middle of one, shows
        a sheet of tumour cells with no wall, no surrounding stroma and no rim: there is
        nothing in the picture to tell the two apart, however good the model is.
      </p>

      <div
        className="tl-picker"
        role="group"
        aria-label="How much slide fits in one square"
      >
        {FIELD_OF_VIEW_CHOICES.map((choice) => {
          const active = current !== null && Math.abs(choice - current) < 1
          const entry = byUm.get(choice)
          const available = entry?.available ?? false
          // Keyed per branch. The counts are identical across branches at the same
          // geometry — the grid is the grid — but keying on the branch keeps the
          // "measured, never estimated" invariant literally true rather than true by
          // an argument somebody has to remember.
          const count = known.get(`${branch}:${choice}`)

          const classes = ['tl-picker__pick']
          if (active) classes.push('is-active')
          if (!available) classes.push('is-unavailable')

          return (
            <button
              key={choice}
              type="button"
              className={classes.join(' ')}
              aria-pressed={active}
              disabled={refining || !available}
              onClick={() => onPick(choice)}
            >
              <span className="tl-picker__name">{LABELS[choice].name}</span>
              <span className="tl-picker__count mono">
                {count === undefined ? '—' : formatCount(count)}
              </span>
              <span className="tl-picker__unit">squares to process</span>
              <span className="tl-picker__note">{LABELS[choice].note}</span>
              <span className="tl-picker__model mono">
                {available ? entry?.model : UNAVAILABLE[branch]}
              </span>
            </button>
          )
        })}
      </div>

      <p className="tl-card__note">
        Bigger squares are not simply better. Each one is still read into the same 224
        pixels, so doubling the ground it covers halves the detail inside it &mdash; at
        448 µm a nucleus is about five pixels across instead of ten. The wider square
        wins on architecture and loses on cells, which is why this is a choice and not
        a default.
      </p>

      <p className="tl-card__note">
        These are four separately trained models, not one model at four zoom levels.
        Picking a scale here picks the checkpoint step 8 runs, and everything that step
        reports &mdash; its accuracy, its licence, what it cannot do &mdash; belongs to
        that checkpoint.
      </p>
    </section>
  )
}
