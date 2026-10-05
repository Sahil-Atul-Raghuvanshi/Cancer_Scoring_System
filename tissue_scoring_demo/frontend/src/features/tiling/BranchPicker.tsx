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
 * Step 7's first question: what should the model be shown?
 *
 * The larger of the screen's two choices, and the newer one. The field-of-view picker
 * beside it chooses how much ground a square covers; this chooses what is *in* the
 * square by the time the network sees it — and the options are genuinely different
 * models, not one model with a setting. The first two are ours, fitted on differently
 * prepared copies of the same tiles; the third is somebody else's, fitted on their own.
 *
 * **Blue stain only** deconvolves the image and keeps the haematoxylin channel. That is
 * the one dye present on every slide in the panel, the H&E and all five immunostains,
 * which is what lets a single model read the whole set.
 *
 * **Full colour** hands the network the photograph. On an H&E section the pink dye is
 * half the evidence — it is what makes a duct wall look different from a band of scar
 * tissue — and the blue channel alone throws it away. The cost is that the option is
 * meaningless anywhere else: on an immunostained slide there is no pink, so the option
 * is offered only where step 5 actually found two dyes, and it says so when it cannot.
 *
 * **Run by BEETLE** hands the slide to another research group's released network rather
 * than to a checkpoint of ours, and it is the only option here that answers *per pixel*
 * — five classes rather than three, with a learned "this is not tissue" among them. It
 * needs no training of ours and no field of view of ours to be published: one release
 * covers all four scales.
 *
 * **It is gated on step 5's verdict too, and for a stronger reason than the full-colour
 * option is.** BEETLE reads the colour photograph with no deconvolution and no stain
 * normalisation, so the dyes on the section *are* its input distribution; on an
 * immunostained slide it would be reading brown where it expects pink. The difference
 * from the full-colour option is that there is no recourse — those are somebody else's
 * released weights and there is no version of them fitted on anything else, where our
 * own heads could at least be retrained. So it needs both things: the two dyes, and its
 * 1.9 GB archive on disk, which is what `setup.py` fetches. The server asks them in that
 * order, because a download cannot make a one-dye section readable and telling a reader
 * to fetch 1.9 GB for a slide that could never use it is worse than saying nothing.
 *
 * Every unavailable option keeps its label and its description and carries the reason
 * it is unavailable. Hiding it would make the pipeline look smaller than it is, and the
 * reason is often the most informative thing on the screen — on an immunostained slide
 * it is the sentence that explains what step 5 measured, and on BEETLE it names a
 * download nobody could guess at from the screen.
 *
 * **Nothing here decides anything.** `implemented`, `enabled` and `reason` all come off
 * the server, which reads step 5's verdict and stats BEETLE's archive; this component
 * renders that answer and never derives one. That is why the H&E option turning itself
 * on is a question of *when the payload is fetched* rather than of logic on this screen
 * — see `useTiling`'s note on step 5's verdict.
 */

import type { TilingBranchName, TilingBranchOut, TilingStaining } from '@/types/tiling'

import { BRANCH_CHOICES } from './useTiling'

interface BranchPickerProps {
  /** The chosen option, or null while the question is open. */
  current: TilingBranchName | null
  /** What the server offers, with availability and a reason per option. */
  offered: TilingBranchOut[]
  /** Step 5's verdict, for the technical disclosure below. */
  staining: TilingStaining
  busy: boolean
  onPick: (branch: TilingBranchName) => void
}

// Keyed on the literal union rather than on `string`, so adding an option to
// `BRANCH_CHOICES` without labelling it is a compile error rather than an undefined
// reaching the screen. The same trick `FieldOfViewPicker`'s `LABELS` uses.
const LABELS: Record<TilingBranchName, { name: string; lead: string }> = {
  h_channel: {
    name: 'Blue stain only',
    lead: 'Throw the colour away and show the model the blue nuclear stain on its own.',
  },
  he: {
    name: 'Full colour (H&E)',
    lead: 'Show the model the photograph as it is, pink and blue together.',
  },
  beetle: {
    name: 'Run by BEETLE',
    lead:
      'Hand the slide to another research group’s published network, which labels every pixel rather than every square, and tells apart five kinds of tissue rather than three.',
  },
}

export function BranchPicker({
  current,
  offered,
  staining,
  busy,
  onPick,
}: BranchPickerProps) {
  const byId = new Map(offered.map((entry) => [entry.id, entry]))

  return (
    <section className="tl-card">
      <h3 className="tl-card__title">What should the model look at?</h3>
      <p className="tl-card__lead">
        The slide is a colour photograph of tissue stained with two dyes: a blue one
        that darkens cell nuclei and, on some slides, a pink one that stains everything
        around them. Before the model reads a square, we choose how much of that picture
        to hand it. Keeping only the blue works on every slide in the set. Keeping both
        gives the model more to go on, but only a slide that actually has both can be
        read that way &mdash; whether this one does was measured on step 5, not guessed
        from its name. The third option skips our models entirely and hands the slide to
        a network another research group trained and published; it reads the colour too,
        so it needs the same two dyes.
      </p>

      <div className="tl-picker tl-picker--branch" role="group" aria-label="What should the model look at">
        {BRANCH_CHOICES.map((choice) => {
          const entry = byId.get(choice)
          const usable = (entry?.implemented ?? false) && (entry?.enabled ?? false)
          const active = current === choice

          const classes = ['tl-picker__pick']
          if (active) classes.push('is-active')
          if (!usable) classes.push('is-unavailable')

          return (
            <button
              key={choice}
              type="button"
              className={classes.join(' ')}
              aria-pressed={active}
              disabled={busy || !usable}
              onClick={() => onPick(choice)}
            >
              <span className="tl-picker__name">{LABELS[choice].name}</span>
              <span className="tl-picker__note">{LABELS[choice].lead}</span>
              {entry?.blurb ? (
                <span className="tl-picker__note">{entry.blurb}</span>
              ) : null}
              <span className="tl-picker__model">
                {usable ? 'available for this slide' : (entry?.reason ?? 'unavailable')}
              </span>
            </button>
          )
        })}
      </div>

      <details className="tl-detail">
        <summary>The technical detail</summary>
        <p>
          The blue-stain option runs Ruifrok and Johnston&rsquo;s colour deconvolution on
          the fixed haematoxylin&ndash;DAB basis and keeps the first channel as optical
          density, clipped, rescaled and repeated into three identical planes so a stock
          three-channel ResNet18 can read it. Fixed vectors and never per-image
          estimates: an estimate rescales itself to whatever slide it is given, so the
          same amount of dye would mean a different number on every slide and the
          model&rsquo;s input distribution would drift underneath it.
        </p>
        <p>
          The full-colour option applies none of that. The tile goes to the network as
          8-bit sRGB scaled to [0,&nbsp;1] and normalised with the ImageNet statistics
          the backbone was fitted under &mdash; no optical density, no white point, no
          deconvolution, no per-tile rescaling. Every one of those is a statement about
          dye concentration, and this branch deliberately makes none of them: it shows
          the model the photograph. The shortness of the transform is the point, because
          there is nothing in it that can drift between training and serving.
        </p>
        <p>
          Whether this slide carries two dyes is read off step 5&rsquo;s point cloud
          rather than its filename. Step 5 fits a plane to a tile&rsquo;s optical
          densities and reports the two directions the data occupies; the full-colour
          option is offered when those two directions are genuinely separate
          {staining.separation !== null ? ` (${staining.separation.toFixed(0)}° apart here)` : ''}
          and one of them lands near Ruifrok&rsquo;s published eosin vector
          {staining.eosinArmDegrees !== null
            ? ` (${staining.eosinArmDegrees.toFixed(0)}° away, against the ${staining.toleranceDeg?.toFixed(0) ?? '—'}° allowed)`
            : ''}
          . Step 5&rsquo;s verdict for this slide: <code>{staining.staining}</code>, from{' '}
          <code>{staining.source}</code>.
        </p>
        <p>
          BEETLE shares none of that. It is a released nnU-Net, so it comes with its own
          published preprocessing rule and this pipeline applies it rather than choosing
          one: a division by 255 and nothing else &mdash; no z-scoring, no ImageNet
          statistics, no optical density. It also fixes its own resolution at 0.5 µm/px,
          because that is what it was fitted at, so a field of view here changes how much
          slide is in one window rather than how finely that window is read; one release
          therefore covers all four scales. And it answers per pixel in five classes,
          including a learned &ldquo;this is not tissue&rdquo; that neither of our models
          has any way to say.
        </p>
        <p>
          That same division by 255 is why BEETLE is offered only where step 5 found two
          dyes. Nothing in it normalises a stain, so the colour of the section as it was
          scanned <em>is</em> the distribution it was fitted on, and an immunostained
          slide is brown where that distribution is pink. The claim is made from the
          released archive rather than from the paper: its <code>dataset.json</code>{' '}
          names all three channels <code>rgb_to_0_1</code>, which is checkable here,
          where what it was stained with is not recorded anywhere in the download. And
          unlike the two options above it, these weights cannot be retrained for another
          stain &mdash; they are a release, not a checkpoint of ours.
        </p>
        <p>
          Note that step 6 treats an eosin-pointing direction as a <em>fault</em>, and
          that is not a contradiction. Step 6 asks whether this is the
          haematoxylin-and-DAB section it was told to unmix; this screen asks whether the
          section has two dyes at all. Both readings are correct answers to different
          questions.
        </p>
      </details>
    </section>
  )
}
