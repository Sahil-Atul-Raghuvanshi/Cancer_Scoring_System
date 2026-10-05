/**
 * Step 8's screen, written for someone who has never read a pathology paper.
 *
 *   1. what just happened, and the one number the whole score rests on
 *   2. the map, with the classes switchable — the denominator changing
 *   3. the classes as bars, with what each one is and why it is excluded
 *   4. how sure the model was, beside the map it drew
 *   5. the second check, and what it would not stand behind
 *   6. what this model cannot do — the caveats, not folded away
 *   7. the technical detail, folded away
 *
 * Three decisions about the ordering are worth stating.
 *
 * **The class toggle sits on the map rather than in a settings row.** The guide asks
 * this screen to let a viewer switch fat off and *watch the denominator change*, and
 * that only lands if the number and the picture move together in front of them.
 *
 * **The second check gets its own section rather than a footnote on the map.** The
 * purple class is not a fourth thing the model can see — it is a rule applied to the
 * model's output afterwards, and a viewer who reads it as a fourth prediction will
 * trust it in a way it has not earned. So it is introduced where there is room to say
 * what it asks, what it costs (nothing — the tissue was excluded anyway) and what it
 * cannot see, which is the missed in-situ disease that is this class's real failure.
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
  /**
   * Every class id this pass emits. Three on a trained option, five on BEETLE — so
   * "everything is shown" is a comparison against this rather than against a constant.
   */
  allClasses: number[]
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
  allClasses,
  onToggleClass,
  onShowAll,
}: TissueTypeViewProps) {
  const { params, grid, classes, model } = report

  // **Which class means what depends on which option ran.** The trained options emit
  // three - 0 not tumour, 1 in-situ, 2 invasive - and BEETLE emits five, where 0 is
  // "not tissue" and the same two lesions sit at 2 and 3. So these are found by the
  // `scored` flag and by name rather than by a hard-coded id, which would silently
  // point at the wrong class on a five-class report.
  const scored = classes.find((entry) => entry.scored)

  // The unit this pass counted in. Everything below reads "patches" on a trained
  // option and "pixels" on BEETLE, because a share of pixels and a share of patches
  // are different claims and the screen must not present one as the other.
  const perPixel = params.perPixel

  // What the map is showing, as an area. The point of the toggle: this is the
  // number that moves when a class is switched off, and switching the two excluded
  // classes off leaves exactly the tissue the score is measured on.
  const shownMm2 = classes
    .filter((entry) => visible.includes(entry.id))
    .reduce((total, entry) => total + entry.areaMm2, 0)

  // Summed over the classes the report actually carries, not over three named ones.
  // It used to be `scoredMm2` plus two classes looked up by name, which silently
  // stopped being the whole map the moment a fourth class existed — the caption would
  // have claimed a total the picture disagreed with.
  const totalMm2 = classes.reduce((total, entry) => total + entry.areaMm2, 0)

  // The second check's own block, when the pass produced one. Null on the per-pixel
  // option and on a pass that ran before the check existed, and the whole section
  // below is conditional on it rather than rendering zeros.
  const uncertainty = report.uncertainty

  const everything = visible.length === allClasses.length
  const minutes = grid.seconds / 60

  return (
    <div className="tt">
      {/* --- 1. what just happened ---------------------------------------- */}
      <div className="tt-headline">
        <p className="tt-headline__lead">
          Every patch of tissue was shown to a trained model, which answered one
          question: <strong>what kind of tissue is this?</strong> It has{' '}
          {allClasses.length} possible answers, and only one of them &mdash;{' '}
          {scored?.label.toLowerCase() ?? 'invasive tumour'} &mdash; is scored.
          {perPixel
            ? ' It answered for every pixel, so the map below follows the edge of each duct.'
            : ' It answered once per patch, so the map below is a grid of squares.'}
        </p>
        <p className="tt-headline__lead">
          The model was trained beforehand, not here, so it gives the same answer every
          time you open this slide.
        </p>

        <div className="tt-headline__figures">
          <div className="tt-headline__figure tt-headline__figure--hero">
            <span className="tt-headline__value mono">
              {(report.tumourContent * 100).toFixed(1)}%
            </span>
            <span className="tt-headline__caption">
              of the tissue is invasive tumour
            </span>
            <span className="tt-headline__sub">
              {report.scoredMm2.toFixed(1)} mm² &mdash; the only tissue the final score
              is measured on
            </span>
          </div>

          <div className="tt-headline__figure">
            <span className="tt-headline__value mono">
              {formatCount(grid.classified)}
            </span>
            <span className="tt-headline__caption">
              {perPixel ? 'patches segmented' : 'patches classified'}
            </span>
            <span className="tt-headline__sub">
              inside the {formatCount(grid.tilesKept)} squares the previous step kept
              {perPixel && grid.tissuePixels
                ? ` · ${formatCount(grid.tissuePixels)} pixels given a class`
                : ''}
            </span>
          </div>

          <div className="tt-headline__figure">
            <span className="tt-headline__value mono">
              {params.windowUm.toFixed(0)} µm
            </span>
            <span className="tt-headline__caption">across each patch</span>
            <span className="tt-headline__sub">
              about {(params.windowUm / 10).toFixed(0)} cells wide
              {perPixel ? '' : ' — the smallest view that holds a whole duct'}
            </span>
          </div>

          <div className="tt-headline__figure">
            <span className="tt-headline__value mono">
              {minutes >= 1 ? `${minutes.toFixed(0)} min` : `${grid.seconds.toFixed(0)} s`}
            </span>
            <span className="tt-headline__caption">taken to run</span>
            <span className="tt-headline__sub">
              {formatCount(perPixel ? (params.patches ?? grid.batches) : grid.batches)}{' '}
              passes through the model on{' '}
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
          {totalMm2.toFixed(1)} mm². Switch the excluded kinds off and what is left is
          the tissue the score is measured on. A kind that is switched off is not
          drawn at all, so the slide shows through where it was.
        </figcaption>
      </figure>

      {/* --- 3. the answers ------------------------------------------------ */}
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
              <span>
                {perPixel && entry.pixels !== null
                  ? `${formatCount(entry.pixels)} pixels`
                  : `${formatCount(entry.windows)} patches`}
              </span>
              <span>{(entry.meanConfidence * 100).toFixed(0)}% confident</span>
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
            Green where the model was confident, red where it was close to guessing.
          </figcaption>
        </figure>

        <div className="tt-confidence__text">
          <h3 className="tt-confidence__title">
            {(report.meanConfidence * 100).toFixed(0)}% confident on average
          </h3>
          <p>
            With three answers to choose from, a pure guess would be 33%. The map above
            colours an uncertain patch exactly like a certain one, which is why this is
            shown separately.
          </p>
          <p>
            The next step joins these patches into smooth regions, using this confidence
            rather than the colours. Where this picture is red, the boundary is less
            reliable.
          </p>
        </div>
      </div>

      {/* --- 5. the second check ------------------------------------------ */}
      {uncertainty && (
        <div className="tt-confidence">
          <figure className="tt-confidence__figure">
            <div className="tt-map__frame">
              <img
                src={tissueTypePanelUrl(report.uploadId, 'uncertainty')}
                alt="The contained-tumour patches, shaded by how close each one came to being marked undetermined"
              />
            </div>
            <figcaption className="tt-map__caption">
              Only the contained-tumour patches are shaded. Blue passed the second
              check, purple did not. Brighter means further from the line.
            </figcaption>
          </figure>

          <div className="tt-confidence__text">
            <h3 className="tt-confidence__title">
              {uncertainty.windows > 0
                ? `${(uncertainty.share * 100).toFixed(1)}% of the tissue could not be determined`
                : 'Every contained-tumour patch passed the second check'}
            </h3>
            <p>
              The model has three answers and no way to say &ldquo;I don&rsquo;t
              know&rdquo;, so anything that fits neither of the other two ends up in
              contained tumour. A second check asks two things the model cannot: does
              the surrounding tissue look invasive, and is this region even duct-shaped?
              Ducts branch and have edges, so a solid sheet several millimetres across
              is not a duct.
            </p>
            {uncertainty.windows > 0 && (
              <p>
                {formatCount(uncertainty.windows)} patches &mdash;{' '}
                <strong className="mono">{uncertainty.areaMm2.toFixed(1)} mm²</strong>{' '}
                &mdash; are drawn purple, out of {formatCount(uncertainty.components)}{' '}
                separate regions, the largest covering{' '}
                {uncertainty.largestComponentMm2.toFixed(1)} mm². None of this was going
                to be scored anyway, so the figure at the top of the page does not
                change. Purple simply marks where a person should take a look.
              </p>
            )}
            <p>
              One limit worth knowing: this check only looks at patches the model already
              called contained tumour. Contained tumour it missed is never examined, so a
              slide with no purple on it has not been fully checked.
            </p>
          </div>
        </div>
      )}

      {/* --- 6. what it cannot do ----------------------------------------- */}
      {report.caveats.length > 0 && (
        <div className="tt-caveats">
          <h3 className="tt-caveats__title">What this model cannot do</h3>
          <p className="tt-caveats__intro">
            These are limits on the result above, so they are shown here rather than
            hidden in the technical detail.
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

      {/* --- 7. the detail ------------------------------------------------ */}
      <details className="tt-detail">
        <summary>The technical detail</summary>
        <div className="tt-detail__body">
          <p className="tt-detail__intro">
            The same result in technical language. Nothing here changes the answer.
          </p>
          <ul className="tt-detail__notes">
            {report.notes.map((note) => (
              <li key={note.slice(0, 48)}>{note}</li>
            ))}
          </ul>

          <p className="tt-detail__foot">
            {perPixel ? (
              <>
                <strong>{model.name}</strong>, nnU-Net 2D <code>PlainConvUNet</code> as
                released &mdash; a published network from another group, not a
                checkpoint of this project&rsquo;s, so it declares no manifest and there
                is nothing here measured on our held-out split. Fold
                {(model.folds?.length ?? 1) === 1 ? '' : 's'}{' '}
                {(model.folds ?? [0]).join(', ')} of {model.foldsAvailable ?? 5} averaged
                in probability space after the softmax. Input: the sRGB photograph of a{' '}
                {params.windowPx} px window at {params.mpp} µm/px (
                {params.windowUm.toFixed(0)} µm), divided by 255 and nothing else &mdash;
                no optical density, no deconvolution, no polarity, no per-tile
                standardisation, no ImageNet statistics. The window is read as{' '}
                {params.patchPx} px patches at a {((params.patchStep ?? 0.5) * 100).toFixed(0)}%
                step, blended by nnU-Net&rsquo;s Gaussian; its answer is kept at{' '}
                {params.maskMpp} µm/px by area-averaging the class probabilities, never
                by resampling labels. Grid {grid.cols} × {grid.rows} at{' '}
              </>
            ) : (
              <>
                <strong>{model.name}</strong>, ResNet18 from a{' '}
                {model.init ?? 'pretrained'} initialisation, frozen body and a
                three-class head, sha256 {(model.sha256 ?? '').slice(0, 16)}…, verified
                against its manifest at load. Input:{' '}
                {params.branch === 'he' ? (
                  <>
                    the colour photograph of a {params.windowPx} px window at{' '}
                    {params.mpp} µm/px ({params.windowUm.toFixed(0)} µm), ImageNet
                    normalisation and nothing else &mdash; no stain separation, because
                    the second dye is the evidence this head was fitted on
                  </>
                ) : (
                  <>
                    the haematoxylin channel of a {params.windowPx} px window at{' '}
                    {params.mpp} µm/px ({params.windowUm.toFixed(0)} µm),
                    Ruifrok&rsquo;s fixed H-DAB basis, clipped at 1.5 OD, γ ={' '}
                    {params.gamma}, per-tile p99 standardisation{' '}
                    {params.standardise ? 'on' : 'off'}, polarity{' '}
                    {params.invertPolarity ? 'inverted' : 'nuclei bright'}, ImageNet
                    normalisation
                  </>
                )}
                . Grid {grid.cols} × {grid.rows} at{' '}
              </>
            )}
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
            {perPixel
              ? `${formatCount(params.patches ?? 0)} forward passes in batches of ${params.batchSize}`
              : `${formatCount(grid.batches)} batches of ${params.batchSize}`}
            .
            {perPixel && model.citation && ` ${model.citation}`}
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
