/**
 * Step 8's screen, written for someone who has never read a pathology paper.
 *
 *   1. what just happened, and the one number the whole score rests on
 *   2. the map, with the three classes switchable — the denominator changing
 *   3. the three classes as bars, with what each one is and why it is excluded
 *   4. how sure the model was, beside the map it drew
 *   5. what this model cannot do — the caveats, not folded away
 *   6. the technical detail, folded away
 *
 * Two decisions about the ordering are worth stating.
 *
 * **The class toggle sits on the map rather than in a settings row.** The guide asks
 * this screen to let a viewer switch fat off and *watch the denominator change*, and
 * that only lands if the number and the picture move together in front of them.
 *
 * **The caveats are not in the disclosure.** Everything else technical is folded
 * away, per the house rule for these screens — but "this model's in-situ class was
 * tested against nine pathologist-drawn tiles" and "this result may not be sold" are
 * not detail, they are the result's terms of use. A reader who skips the disclosure
 * should still see them.
 */

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { tissueTypePanelUrl } from '@/api/tissueType'
import { formatCount } from '@/lib/format'
import type { TissueTypeReport } from '@/types/tissueType'

import './tissueType.css'

interface TissueTypeViewProps {
  report: TissueTypeReport
  /** Which classes the map draws. */
  visible: number[]
  onToggleClass: (id: number) => void
  onShowAll: () => void
}

const SEVERITY_TONE = {
  blocking: 'danger',
  warning: 'warn',
  note: 'neutral',
} as const

export function TissueTypeView({
  report,
  visible,
  onToggleClass,
  onShowAll,
}: TissueTypeViewProps) {
  const { params, grid, classes, model } = report

  const inSitu = classes.find((entry) => entry.id === 1)
  const notTumour = classes.find((entry) => entry.id === 0)

  // What the map is showing, as an area. The point of the toggle: this is the
  // number that moves when a class is switched off, and switching the two excluded
  // classes off leaves exactly the tissue the score is measured on.
  const shownMm2 = classes
    .filter((entry) => visible.includes(entry.id))
    .reduce((total, entry) => total + entry.areaMm2, 0)

  const everything = visible.length === classes.length
  const minutes = grid.seconds / 60

  return (
    <div className="tt">
      {/* --- 1. what just happened ---------------------------------------- */}
      <div className="tt-headline">
        <p className="tt-headline__lead">
          Every patch of tissue was shown to a trained model, which answered one
          question about each one: <strong>what kind of tissue is this?</strong> It
          has three possible answers, and only one of them is scored.
        </p>
        <p className="tt-headline__lead">
          This is the only step in the pipeline with a learned model behind it. It
          was not trained here &mdash; it was trained offline and is being run, which
          is why it produces the same answer every time you open this slide.
        </p>

        <div className="tt-headline__figures">
          <div className="tt-headline__figure tt-headline__figure--hero">
            <span className="tt-headline__value mono">
              {(report.tumourContent * 100).toFixed(1)}%
            </span>
            <span className="tt-headline__caption">
              of the tissue is tumour that has broken out
            </span>
            <span className="tt-headline__sub">
              {report.scoredMm2.toFixed(1)} mm² &mdash; the only tissue the final
              score is measured on
            </span>
          </div>

          <div className="tt-headline__figure">
            <span className="tt-headline__value mono">
              {formatCount(grid.classified)}
            </span>
            <span className="tt-headline__caption">patches classified</span>
            <span className="tt-headline__sub">
              inside the {formatCount(grid.tilesKept)} squares the previous step kept
            </span>
          </div>

          <div className="tt-headline__figure">
            <span className="tt-headline__value mono">
              {params.windowUm.toFixed(0)} µm
            </span>
            <span className="tt-headline__caption">across each patch</span>
            <span className="tt-headline__sub">
              about {(params.windowUm / 10).toFixed(0)} cells &mdash; the smallest
              view that holds a whole duct
            </span>
          </div>

          <div className="tt-headline__figure">
            <span className="tt-headline__value mono">
              {minutes >= 1 ? `${minutes.toFixed(0)} min` : `${grid.seconds.toFixed(0)} s`}
            </span>
            <span className="tt-headline__caption">of real work</span>
            <span className="tt-headline__sub">
              {formatCount(grid.batches)} passes through the model on{' '}
              {params.device === 'cpu' ? 'the processor' : params.device}
            </span>
          </div>
        </div>
      </div>

      {/* --- 2. the map, with the toggle ---------------------------------- */}
      <figure className="tt-map">
        <div className="tt-map__frame">
          <img
            src={tissueTypePanelUrl(report.uploadId, 'map', visible)}
            alt="The slide with each patch coloured by the kind of tissue the model found"
          />
        </div>

        <div className="tt-toggle">
          <span className="tt-toggle__label">Show</span>
          {classes.map((entry) => {
            const on = visible.includes(entry.id)
            return (
              <button
                key={entry.id}
                type="button"
                className={on ? 'tt-chip tt-chip--on' : 'tt-chip'}
                onClick={() => onToggleClass(entry.id)}
                aria-pressed={on}
              >
                <span className="tt-chip__dot" style={{ background: entry.colour }} />
                {entry.label}
                <span className="tt-chip__figure mono">
                  {entry.areaMm2.toFixed(1)} mm²
                </span>
              </button>
            )
          })}
          {!everything && (
            <Button variant="ghost" onClick={onShowAll}>
              Show all
            </Button>
          )}
        </div>

        <figcaption className="tt-map__caption">
          Showing <strong className="mono">{shownMm2.toFixed(1)} mm²</strong> of{' '}
          {(report.scoredMm2 + (inSitu?.areaMm2 ?? 0) + (notTumour?.areaMm2 ?? 0)).toFixed(
            1,
          )}{' '}
          mm². Switch the two excluded kinds off and what is left is the tissue the
          score is actually measured on &mdash; watch the figure drop as you do. A
          class that is switched off is not recoloured, it is not drawn at all, so
          the scan shows through where it was.
        </figcaption>
      </figure>

      {/* --- 3. the three answers ----------------------------------------- */}
      <div className="tt-classes">
        {classes.map((entry) => (
          <div
            key={entry.id}
            className={entry.scored ? 'tt-class tt-class--scored' : 'tt-class'}
          >
            <div className="tt-class__head">
              <span className="tt-class__dot" style={{ background: entry.colour }} />
              <span className="tt-class__label">{entry.label}</span>
              {entry.scored ? (
                <Badge tone="accent">scored</Badge>
              ) : (
                <Badge tone="neutral">excluded</Badge>
              )}
            </div>

            <div className="tt-class__bar">
              <span
                className="tt-class__fill"
                style={{ width: `${entry.share * 100}%`, background: entry.colour }}
              />
            </div>

            <div className="tt-class__figures mono">
              <span className="tt-class__share">{(entry.share * 100).toFixed(1)}%</span>
              <span>{entry.areaMm2.toFixed(1)} mm²</span>
              <span>{formatCount(entry.windows)} patches</span>
              <span>{(entry.meanConfidence * 100).toFixed(0)}% sure</span>
            </div>

            <p className="tt-class__blurb">{entry.blurb}</p>
          </div>
        ))}
      </div>

      {/* --- 4. how sure it was ------------------------------------------- */}
      <div className="tt-confidence">
        <figure className="tt-confidence__figure">
          <div className="tt-map__frame">
            <img
              src={tissueTypePanelUrl(report.uploadId, 'confidence')}
              alt="How certain the model was about each patch, red for unsure and green for sure"
            />
          </div>
          <figcaption className="tt-map__caption">
            Red where the model was barely choosing, green where it was sure.
          </figcaption>
        </figure>

        <div className="tt-confidence__text">
          <h3 className="tt-confidence__title">
            It was {(report.meanConfidence * 100).toFixed(0)}% sure, on average
          </h3>
          <p>
            With three answers to choose between, a pure guess is 33%. The map above
            draws a barely-decided patch in exactly the same colour as a certain one,
            which is why this is shown separately rather than kept behind the scenes.
          </p>
          <p>
            The next step turns these patch-by-patch answers into one smooth region,
            and it leans on this certainty rather than on the colours. Where this
            picture is red, that boundary will move.
          </p>
        </div>
      </div>

      {/* --- 5. what it cannot do ----------------------------------------- */}
      {report.caveats.length > 0 && (
        <div className="tt-caveats">
          <h3 className="tt-caveats__title">What this model cannot do</h3>
          <p className="tt-caveats__intro">
            Not folded away with the rest of the technical detail, because these are
            the result&rsquo;s terms rather than its background.
          </p>
          <ul className="tt-caveats__list">
            {report.caveats.map((caveat) => (
              <li key={caveat.key} className={`tt-caveat tt-caveat--${caveat.severity}`}>
                <div className="tt-caveat__head">
                  <Badge tone={SEVERITY_TONE[caveat.severity]}>{caveat.severity}</Badge>
                  <span className="tt-caveat__headline">{caveat.headline}</span>
                </div>
                <p className="tt-caveat__detail">{caveat.detail}</p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* --- 6. the detail ------------------------------------------------ */}
      <details className="tt-detail">
        <summary>The technical detail</summary>
        <div className="tt-detail__body">
          <p className="tt-detail__intro">
            Everything above in the language of the papers it comes from. Nothing here
            changes the result.
          </p>
          <ul className="tt-detail__notes">
            {report.notes.map((note) => (
              <li key={note.slice(0, 48)}>{note}</li>
            ))}
          </ul>

          <p className="tt-detail__foot">
            <strong>{model.name}</strong>, ResNet18 from a {model.init ?? 'pretrained'}{' '}
            initialisation, frozen body and a three-class head, sha256{' '}
            {(model.sha256 ?? '').slice(0, 16)}…, verified against its manifest at
            load. Input: the haematoxylin channel of a {params.windowPx} px window at{' '}
            {params.mpp} µm/px ({params.windowUm.toFixed(0)} µm), Ruifrok&rsquo;s fixed
            H-DAB basis, clipped at 1.5 OD, γ = {params.gamma}, per-tile p99
            standardisation {params.standardise ? 'on' : 'off'}, polarity{' '}
            {params.invertPolarity ? 'inverted' : 'nuclei bright'}, ImageNet
            normalisation. Grid {grid.cols} × {grid.rows} at{' '}
            {(params.overlap * 100).toFixed(0)}% overlap &mdash; span{' '}
            {formatCount(params.span)} level-0 px, stride {formatCount(params.stride)}{' '}
            &mdash; of which {formatCount(grid.classified)} windows were classified,
            against step 7&rsquo;s {formatCount(grid.tilesKept)} kept {grid.tilePx} px
            / {grid.tileUm.toFixed(0)} µm tiles. Step 7 lays this model&rsquo;s own
            window, read from the same manifest, so those are one set of squares and
            not two: the centre test and this step&rsquo;s own{' '}
            {(grid.minTissueShare * 100).toFixed(0)}% tissue gate both pass everything
            step 7 kept
            {grid.gatedOut > 0 &&
              `, except ${formatCount(grid.gatedOut)} that this step still found too empty — which means the two grids have come apart, and an overlap named on this step's own route is the way that happens`}
            .{' '}
            {formatCount(grid.blocksRead)} pyramid reads,{' '}
            {formatCount(grid.batches)} batches of {params.batchSize}.
            {model.heldOutAccuracy !== null && (
              <>
                {' '}
                Held-out accuracy {(model.heldOutAccuracy * 100).toFixed(1)}%, per-class
                Dice{' '}
                {model.diceInvasive !== null ? model.diceInvasive.toFixed(3) : '—'}{' '}
                invasive /{' '}
                {model.diceNonInvasive !== null
                  ? model.diceNonInvasive.toFixed(3)
                  : '—'}{' '}
                non-invasive
                {model.nonInvasiveTestTiles !== null &&
                  ` (against ${formatCount(model.nonInvasiveTestTiles)} human-labelled in-situ tiles)`}
                .
              </>
            )}{' '}
            Trained on {model.trainingSource ?? 'an unrecorded source'}. Licence track{' '}
            <strong>{params.licenceTrack}</strong>:{' '}
            {Object.entries(model.licences)
              .map(([source, terms]) => `${source} — ${terms}`)
              .join('; ') || 'not recorded'}
            . Areas are counted once per stride-sized cell rather than once per
            window, so they do not move when the overlap does.
          </p>
        </div>
      </details>

      <p className="tt-citation">
        <Badge tone="neutral">BCSS 2019 · BRACS 2022 · Ruifrok 2001 · Tellez 2019</Badge>{' '}
        {report.citation}
      </p>
    </div>
  )
}
