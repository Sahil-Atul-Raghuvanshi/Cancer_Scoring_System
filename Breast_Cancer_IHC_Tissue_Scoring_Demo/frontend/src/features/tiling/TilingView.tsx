/**
 * Step 7's screen, written for someone who has never read a pathology paper.
 *
 *   1. what just happened, in two sentences
 *   2. the map — the grid on the slide, with a legend for the three colours
 *   3. the funnel — the numbers dropping
 *   4. one square, as the model receives it
 *   5. the overlap choice, and what it costs
 *   6. the technical detail, folded away
 *
 * The map comes before the funnel deliberately. The counts only mean something
 * once a reader has seen that the lit squares trace the tissue and the dark ones
 * are empty glass — with that picture in mind "21,609 became 5,070" is obvious,
 * and without it the numbers are trivia.
 */

import { Badge } from '@/components/ui/Badge'
import { tilingPanelUrl } from '@/api/tiling'
import type { TilingOptions } from '@/api/tiling'
import { formatCount } from '@/lib/format'
import type { TilingReport } from '@/types/tiling'

import { OverlapPicker } from './OverlapPicker'
import { TileFunnel } from './TileFunnel'

import './tiling.css'

interface TilingViewProps {
  report: TilingReport
  refining: boolean
  /** Tile counts already measured at each overlap, so the picker can price itself. */
  known: Map<number, number>
  onOverlap: (overlap: number) => void
}

export function TilingView({
  report,
  refining,
  known,
  onOverlap,
}: TilingViewProps) {
  const { params, funnel, coverage, sample, tiles } = report

  const options: TilingOptions = {
    threshold:
      params.tissueThresholdSource === 'manual' ? params.tissueThreshold : null,
    overlap: params.overlap,
  }

  const flagged = tiles.filter((tile) => tile.rejectedBy === 'clean').length

  // The slide's own width, to within one stride. Not `span × cols`, which is the
  // tempting arithmetic and is wrong by exactly the overlap: the squares are laid
  // a stride apart, not an extent apart, so at 25% overlap that product would
  // overstate the scan by a third.
  const slideWidth = (report.cols - 1) * params.stride + params.span

  return (
    // One class for the whole screen, because a rebuild invalidates all of it:
    // every count, the map and the covered area are statements about one grid, so
    // while a new one is being built they all describe a setting the viewer has
    // already moved off. The picker itself stays live — it is the control in use.
    <div className={refining ? 'tl tl--refining' : 'tl'} aria-busy={refining}>
      {/* --- 1. what just happened ---------------------------------------- */}
      <div className="tl-headline">
        <p className="tl-headline__lead">
          The scan is far too big to hand to a model in one piece &mdash; it is
          around {formatCount(slideWidth)} pixels across. So the tissue is cut into
          small squares, {params.tileSize} pixels each, and the model is shown one
          square at a time.
        </p>
        <p className="tl-headline__lead">
          Nothing is copied or saved here. This step just writes down{' '}
          <strong>where</strong> each square is. The picture for a square is worked
          out only when it is needed &mdash; and it is the blue-stain picture from
          step 6, not the colour photo.
        </p>

        <div className="tl-headline__figures">
          <div className="tl-headline__figure">
            <span className="tl-headline__value mono">
              {formatCount(funnel.clean)}
            </span>
            <span className="tl-headline__caption">squares to process</span>
            <span className="tl-headline__sub">
              down from {formatCount(funnel.every)} in the full grid
            </span>
          </div>

          <div className="tl-headline__figure">
            <span className="tl-headline__value mono">
              {params.tileUm.toFixed(0)} µm
            </span>
            <span className="tl-headline__caption">across each square</span>
            <span className="tl-headline__sub">
              about {(params.tileUm / 10).toFixed(0)} cells wide &mdash; enough to
              see how the glands are arranged
            </span>
          </div>

          <div className="tl-headline__figure">
            <span className="tl-headline__value mono">
              {coverage.tissueMm2.toFixed(0)} mm²
            </span>
            <span className="tl-headline__caption">of tissue to cover</span>
            <span className="tl-headline__sub">
              the squares reach {coverage.coveredMm2.toFixed(0)} mm², a little more
              because squares do not follow the edge exactly
            </span>
          </div>
        </div>
      </div>

      {/* --- 2. the map --------------------------------------------------- */}
      <figure className="tl-map">
        <div className="tl-map__frame">
          <img
            src={tilingPanelUrl(report.uploadId, 'grid', options)}
            alt="The tile grid drawn over the slide, with the kept squares highlighted"
          />
        </div>
        <figcaption>
          <div className="tl-legend">
            <span className="tl-legend__item">
              <span className="tl-legend__swatch tl-legend__swatch--kept" />
              Kept &mdash; has tissue, and is clean
            </span>
            <span className="tl-legend__item">
              <span className="tl-legend__swatch tl-legend__swatch--flagged" />
              Dropped &mdash; too much of it is damaged
            </span>
            <span className="tl-legend__item">
              <span className="tl-legend__swatch tl-legend__swatch--off" />
              Dropped &mdash; no tissue there
            </span>
          </div>
          <span className="tl-map__caption">
            {report.cols} × {report.rows} squares over the whole scan. The lit ones
            trace the tissue; everything dark is empty glass that never has to be
            looked at.
            {flagged > 0 &&
              ' The amber patches are places step 2 found folds, blurring or marker pen.'}
          </span>
        </figcaption>
      </figure>

      {/* --- 3. the funnel ------------------------------------------------ */}
      <TileFunnel report={report} />

      {/* --- 4. one square ------------------------------------------------ */}
      {sample && (
        <section className="tl-sample">
          <div className="tl-sample__frame">
            <img
              src={tilingPanelUrl(report.uploadId, 'sample', options)}
              alt="One kept square, as the blue-stain picture the model receives"
            />
          </div>
          <div className="tl-sample__body">
            <h3 className="tl-card__title">This is what one square looks like</h3>
            <p className="tl-card__lead">
              Taken from ({formatCount(sample.x)}, {formatCount(sample.y)}) on the
              slide, {sample.size} pixels across. It is shown as the{' '}
              <strong>blue-stain picture</strong> from step 6 rather than as a
              colour photo, because that is exactly what the model is given.
            </p>
            <p className="tl-card__note">
              Removing the brown before the model sees anything is what lets one
              model work on this slide and on the five other stains in the panel,
              instead of needing a separate model for each. It is the same
              calculation you saw on the previous step &mdash; not a second copy of
              it, which matters, because two copies would slowly drift apart and
              the different stains would stop agreeing.
            </p>
          </div>
        </section>
      )}

      {/* --- 5. the overlap choice ---------------------------------------- */}
      <div className="tl-live">
        <OverlapPicker
          current={params.overlap}
          known={known}
          refining={refining}
          onPick={onOverlap}
        />
      </div>

      {/* --- 6. the detail ------------------------------------------------ */}
      <details className="tl-detail">
        <summary>The technical detail</summary>
        <div className="tl-detail__body">
          <p className="tl-detail__intro">
            Everything above in the language of the papers it comes from. Nothing
            here changes the result.
          </p>
          <ul className="tl-detail__notes">
            {report.notes.map((note) => (
              <li key={note.slice(0, 48)}>{note}</li>
            ))}
          </ul>
          <p className="tl-detail__foot">
            Grid {report.cols} × {report.rows}, tile {params.tileSize} px at{' '}
            {params.targetMpp} µm/px ({params.tileUm.toFixed(0)} µm), span{' '}
            {formatCount(params.span)} level-0 px, stride{' '}
            {formatCount(params.stride)}, overlap{' '}
            {(params.overlap * 100).toFixed(0)}%. Gates: tissue ≥{' '}
            {(params.minTissueShare * 100).toFixed(0)}%, clean ≥{' '}
            {(params.minCleanShare * 100).toFixed(0)}% of a tile&rsquo;s tissue,
            both measured on step 3&rsquo;s mask at {params.maskMpp.toFixed(2)}{' '}
            µm/px and therefore quantised to{' '}
            {(params.shareQuantisation * 100).toFixed(2)}% of a tile. Step 3 cut at{' '}
            {params.tissueThreshold} by the {params.tissueThresholdSource} rule
            {params.qcGated
              ? `, artefacts from ${params.qcSource ?? 'step 2'}`
              : ', with no quality control applied'}
            . Listing shows {report.listed} of {formatCount(funnel.clean)} kept
            tiles.
          </p>
        </div>
      </details>

      <p className="tl-citation">
        <Badge tone="neutral">Slideflow 2024 · Tellez 2019</Badge> {report.citation}
      </p>
    </div>
  )
}
