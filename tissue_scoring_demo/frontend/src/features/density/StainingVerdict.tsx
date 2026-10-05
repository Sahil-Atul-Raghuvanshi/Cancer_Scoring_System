/**
 * Which dyes this section carries — named on the screen that measured it.
 *
 * The verdict already existed. `step05_optical_density/staining.py` reads it off the
 * point cloud the card above draws, and step 7 has been gating its full-colour option
 * on it since that branch was built. What did not exist was anywhere for a *reader* to
 * see it: the report carried `staining` on the wire and this screen rendered none of
 * it, so the one place the evidence lives showed everything except the conclusion, and
 * step 7 was the first screen to say the word "H&E". That is backwards — the reader
 * meets the greyed-out option before the measurement that greyed it out.
 *
 * **So this card states the conclusion, and it is a restatement rather than a second
 * opinion.** Every number here comes out of `report.staining` and `report.cloud`;
 * nothing is recomputed in the browser. If this card and step 7's picker ever disagree,
 * that is a bug in one of them and not two defensible readings — which is exactly why
 * neither derives its own answer.
 *
 * **The plain sentence is the point of the card and the angles are the footnote.** A
 * reader who does not know what a stain vector is still has to come away knowing
 * whether this is an H&E section or an immunostained one, because that is what decides
 * which model can read it two steps later. The degrees go in the disclosure, where a
 * reader who wants to check the claim can find them.
 *
 * **Step 6 calls an eosin arm a fault and this card calls it the expected finding, and
 * both are right.** They answer different questions — step 6 asks whether this is the
 * haematoxylin-and-DAB section it was told to unmix, this asks whether the section has
 * two dyes at all — so the disclosure says which question is being answered rather
 * than leaving a reader to reconcile two screens.
 */

import { Badge } from '@/components/ui/Badge'
import type { BadgeTone } from '@/components/ui/Badge'
import type { DensityCloud, DensityStaining } from '@/types/density'

interface StainingVerdictProps {
  staining: DensityStaining
  /** The cloud the verdict was read from. Its separation and arms are the evidence. */
  cloud: DensityCloud
}

/**
 * How each verdict reads on screen: a headline, the tone it earns, and one plain
 * sentence.
 *
 * Keyed on the four values `SlideStaining` can take, with a fallback below rather than
 * a non-null assertion — the field is a `str` on the wire, so a fifth value added to
 * the enum server-side must not put `undefined` on the screen.
 */
const VERDICTS: Record<
  string,
  { headline: string; tone: BadgeTone; lead: string }
> = {
  he: {
    headline: 'haematoxylin and eosin',
    tone: 'success',
    lead:
      'This is an H&E slide: a blue dye darkens cell nuclei and a pink dye stains the tissue around them. Both are clearly present. There is no brown marker on this slide.',
  },
  haematoxylin_dab: {
    headline: 'haematoxylin and DAB',
    tone: 'accent',
    lead:
      'This is a marker slide: the blue dye darkens cell nuclei, and the brown marker shows where the protein being tested for was found. There is no pink dye here.',
  },
  single_stain: {
    headline: 'one dye in this tile',
    tone: 'warn',
    lead:
      'Only one dye showed up in this tile. That usually means the tile holds little more than nuclei. It says something about the tile, not the slide &mdash; pick a tile with more structure and the second dye normally appears.',
  },
  unknown: {
    headline: 'not settled by this tile',
    tone: 'warn',
    lead:
      'This tile could not tell us which dyes the slide carries. The numbers above are fine; they just do not split into two recognisable dyes here.',
  },
}

const FALLBACK = {
  headline: 'unrecognised verdict',
  tone: 'warn' as BadgeTone,
  lead:
    'The result came back in a form this screen does not recognise. The original message is in the detail below.',
}

/** `haematoxylin` reads as itself; `dab` has to be shouted. */
const STAIN_LABEL: Record<string, string> = {
  haematoxylin: 'haematoxylin',
  dab: 'DAB',
  eosin: 'eosin',
}

function label(nearest: string): string {
  return STAIN_LABEL[nearest] ?? nearest
}

export function StainingVerdict({ staining, cloud }: StainingVerdictProps) {
  const copy = VERDICTS[staining.staining] ?? FALLBACK

  return (
    <div className="od-stain">
      <div className="od-stain__head">
        <span className="eyebrow">which dyes is this slide stained with?</span>
        <Badge tone={copy.tone}>{copy.headline}</Badge>
      </div>

      <p className="od-stain__lead">{copy.lead}</p>

      <div className="od-stain__figures">
        <div className="od-stain__figure">
          <span className="od-stain__value mono">
            {staining.eosinArmDegrees === null
              ? '—'
              : `${staining.eosinArmDegrees.toFixed(1)}°`}
          </span>
          <span className="od-stain__caption">
            {staining.eosinArmDegrees === null
              ? 'no pink dye found'
              : 'away from the standard pink dye'}
          </span>
          <span className="od-stain__sub mono">
            {staining.toleranceDeg === null
              ? 'nothing to compare against'
              : `up to ${staining.toleranceDeg.toFixed(0)}° counts as a match`}
          </span>
        </div>

        <div className="od-stain__figure">
          <span className="od-stain__value mono">{cloud.separation.toFixed(0)}°</span>
          <span className="od-stain__caption">between the two colours found</span>
          <span className="od-stain__sub mono">
            {cloud.twoArmed ? 'far enough apart to be two dyes' : 'too close to be two dyes'}
          </span>
        </div>

        {/* A cloud always carries exactly two arms — `density.py` builds the tuple from
            the wedge's low and high edge — so this is a `join` over two and not a list
            of unknown length. `single_stain` does not mean one arm; it means the two
            arms sit too close together to be two dyes, which is the figure beside this
            one. The fallback is here only so a cloud that ever stopped carrying two
            would read as a dash rather than as blank space. */}
        <div className="od-stain__figure">
          <span className="od-stain__value">
            {cloud.arms.length > 0
              ? cloud.arms.map((arm) => label(arm.nearest)).join(' + ')
              : '—'}
          </span>
          <span className="od-stain__caption">the closest known stains</span>
          <span className="od-stain__sub mono">
            {cloud.arms.length > 0
              ? `${cloud.arms
                  .map((arm) => `${arm.degreesFromNearest.toFixed(0)}°`)
                  .join(' · ')} away`
              : 'nothing found in this tile'}
          </span>
        </div>
      </div>

      {/* The step's own sentence, verbatim. It names the number that decided, and it is
          the *same string* step 7 shows against a disabled option — quoting it here is
          what makes the two screens one answer rather than two. */}
      <p className="od-stain__reason">{staining.reason}</p>

      <details className="od-detail">
        <summary>The technical detail</summary>
        <div className="od-detail__body">
          <p className="od-detail__intro">
            The result above is already stated in plain words. This is the reasoning
            behind it.
          </p>
          <p>
            Nothing new is measured here. The step has already found the two main colour
            directions in this tile and compared each with the known stain colours, in
            three dimensions rather than off the flat chart. The test is one question on
            top of that: does either direction land nearest <em>eosin</em>, the pink dye,
            and is it within{' '}
            {staining.toleranceDeg === null
              ? 'the allowed angle'
              : `${staining.toleranceDeg.toFixed(0)}°`}
            ?
          </p>
          <p>
            Across eighteen tiles from six sections in this project, every H&amp;E tile
            landed 5.0 to 9.4° from the standard pink, and no marker tile landed nearest
            pink at all. So the deciding fact is <em>which</em> known stain a direction
            lands nearest; the angle limit is only a safety margin on top.
          </p>
          <p>
            The gap between the two directions is shown as evidence but is deliberately
            not used to decide. H&amp;E tiles here have the <em>narrower</em> gap (18.8
            to 38.1°) because pink and blue are closer together than blue and brown. Had
            we used the gap as the test, it would have got every slide wrong.
          </p>
          <p>
            This card and the next step ask different questions. This one asks whether
            the slide carries two dyes at all. The next one asks whether it carries the
            specific pair it is set up to separate. Both answers can be correct at once.
          </p>
          <p>
            The result follows whichever tile is chosen below, so moving the tile updates
            this card.
          </p>
        </div>
      </details>
    </div>
  )
}
