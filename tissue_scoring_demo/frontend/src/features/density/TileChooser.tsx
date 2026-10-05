/**
 * Which tile, why that one, and every alternative — as twelve pictures.
 *
 * Step 5 is the first step in the pipeline that has to *pick a place to stand*,
 * because a density is per pixel where a mask and a white point are per slide. The
 * pick is not neutral: two thirds of a stained section is counterstain and stroma,
 * and a tile of that has one arm in its point cloud rather than two. A demo that
 * picked at random would regularly put a one-armed scatter under a caption
 * explaining that the two arms are the two stains.
 *
 * So the choice is scored, and the scoring is shown rather than summarised. The
 * viewer can click any of the twelve and watch the arms move — which is a better
 * argument for the rule than the rule's own description, and the only way to make
 * a chosen tile something other than an assertion.
 *
 * **Why a contact sheet and not a list of bars.** The score is a claim about what
 * is *in* a field of view: stain, and more than one kind of it. A bar says a block
 * scored 0.182 × 0.122, which a reader can only take on trust; a picture of the
 * block lets them see the brown and the blue in it and check the number against
 * their own eyes. The bar is kept under each picture rather than dropped, because
 * a sheet of thumbnails alone would hide that these are *ordered*, and the
 * ordering is the argument.
 *
 * The map is the server's PNG, with the same boxes drawn over it as SVG so they
 * can be hovered and clicked. Both come from the same candidate list as the sheet,
 * so the picture, the hit targets and the thumbnails cannot drift apart.
 *
 * **The pick is shown in three places at once**, which is not redundancy. The
 * reader's attention could be on any of them when they click, and a selection
 * visible only where the click landed would leave the other two looking
 * unchanged: the box on the map, the ring on the thumbnail and the line above the
 * sheet naming the rank are one state described three times.
 */

import { densityCandidateUrl, densityPanelUrl } from '@/api/density'
import type { DensityOptions } from '@/api/density'
import type { DensityReport, TileCandidate } from '@/types/density'

import type { TilePick } from './useOpticalDensity'

/**
 * Edge of a candidate thumbnail as the server renders it — `CANDIDATE_SIZE` in
 * `overlay.py`. Here only to give the images an intrinsic size, so the sheet
 * reserves its layout before the twelve PNGs arrive instead of collapsing and
 * then jolting the page down.
 */
const THUMB_PX = 192

interface TileChooserProps {
  report: DensityReport
  options: DensityOptions
  refining: boolean
  onPick: (tile: TilePick | null) => void
}

export function TileChooser({ report, options, refining, onPick }: TileChooserProps) {
  const { candidates, tile, params, white } = report
  const best = candidates[0]

  return (
    <div className="od-tiles">
      <div className="od-tiles__head">
        <span className="eyebrow">which tile was used, and the alternatives</span>
        {tile.requested && (
          <button
            className="od-tiles__reset"
            type="button"
            onClick={() => onPick(null)}
            disabled={refining}
          >
            back to the recommended tile
          </button>
        )}
      </div>

      <div className="od-tiles__body">
        <div className={refining ? 'od-tiles__map od-tiles__map--stale' : 'od-tiles__map'}>
          <img
            src={densityPanelUrl(report.uploadId, 'map', options)}
            alt={`The whole slide, with the tile in use marked and ${candidates.length - 1} alternatives outlined`}
          />
          <svg
            className="od-tiles__hits"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            role="group"
            aria-label="Candidate tiles"
          >
            {candidates.map((candidate) => (
              <rect
                key={`${candidate.col}-${candidate.row}`}
                className={
                  candidate.chosen ? 'od-tiles__hit od-tiles__hit--chosen' : 'od-tiles__hit'
                }
                x={candidate.fx * 100}
                y={candidate.fy * 100}
                width={candidate.fw * 100}
                height={candidate.fh * 100}
                onClick={() => onPick({ x: candidate.x, y: candidate.y })}
                role="button"
                tabIndex={0}
                aria-label={`Tile at ${candidate.x}, ${candidate.y}: stain amount ${candidate.stain.toFixed(3)}, colour mix ${candidate.mixing.toFixed(3)}`}
              />
            ))}
          </svg>
        </div>

        <div className="od-tiles__side">
          <p className="od-tiles__why">
            We need a tile that holds <em>both</em> stains, so every block of the slide
            was scored on two things: how much stain it has, and how mixed the colours
            are. A block with one stain only scores near zero on the second. The two
            scores are multiplied, because a tile needs both.
          </p>

          {/* One line for the whole selection, and the only thing on the screen
              that says what is happening while a pick is in flight. `aria-live`
              rather than a spinner alone, because "the page is re-computing" is
              information and not decoration. */}
          <p
            className={
              refining
                ? 'od-tiles__standing od-tiles__standing--busy'
                : 'od-tiles__standing'
            }
            aria-live="polite"
          >
            {refining ? (
              <>
                <span className="od-tiles__spinner" aria-hidden />
                Reading that tile and measuring it. Every number above describes one
                tile, so they all dim until the new ones are ready.
              </>
            ) : (
              <>
                Showing tile <strong>#{tile.rank}</strong> of{' '}
                {tile.candidatesScored}
                {tile.requested
                  ? ' — your pick. Click another, or go back to the recommended tile.'
                  : ' — the top-scoring one, chosen for you. Click any other to use it instead.'}
              </>
            )}
          </p>

        </div>
      </div>

      {/* An ordered list, because the rank is the content — and it sits outside the
          two-column body on purpose. Twelve photographs need width more than the
          prose does, and inside the body they would be squeezed into one column
          while the map's column ran out of content and left a hole under it. The
          grid tracks flow in document order at every width, so restacking never
          reorders the rank.

          Deliberately not dimmed or disabled while a pick is in flight. Everything
          above is dimmed, because those figures belong to the tile being replaced;
          a thumbnail does not — the twelve blocks look the same whichever one is
          chosen. And `useOpticalDensity` tickets its requests precisely so that a
          reader who changes their mind mid-flight gets the tile they clicked last
          rather than whichever response is slowest. Locking the sheet would throw
          that away. */}
      <ol className="od-tiles__sheet">
        {candidates.map((candidate, index) => (
          <CandidateCell
            key={`${candidate.col}-${candidate.row}`}
            candidate={candidate}
            rank={index + 1}
            share={candidate.score / Math.max(best?.score ?? 1, 1e-9)}
            uploadId={report.uploadId}
            options={options}
            refining={refining}
            onPick={onPick}
          />
        ))}
      </ol>

      <p className="od-tiles__sheetnote">
        These are the top-scoring blocks, shown smaller than the one in use but
        otherwise the same pixels. Look for <em>two</em> colours: a block that is all
        blue has plenty of stain but only one of them.
      </p>

      <p className="od-tiles__gate">
        Two checks ran before the scoring. A block must be at least{' '}
        <strong>{(params.minTissueShare * 100).toFixed(0)}%</strong> tissue and just as
        clear of problem areas. Its stain must also reach{' '}
        <strong>{params.minStain.toFixed(3)}</strong>, which is{' '}
        {params.minStainMultiple.toFixed(0)}× the {white.noiseFloor.toFixed(3)} noise
        level measured on this slide&rsquo;s own glass. Without that second check, a
        faint patch of false colour can beat real stain &mdash; two wrong colours are
        still two colours.
      </p>
    </div>
  )
}

interface CandidateCellProps {
  candidate: TileCandidate
  rank: number
  /** The block's score as a fraction of the winner's — what the bar draws. */
  share: number
  uploadId: string
  options: DensityOptions
  /** A pick is in flight — the cell shows it is no longer the current one. */
  refining: boolean
  onPick: (tile: TilePick | null) => void
}

/**
 * One block of the sheet: its picture, its rank, its two figures and its bar.
 *
 * The button wraps the whole cell rather than sitting beside it, so the picture
 * is the hit target — a reader who has decided from the thumbnail should be able
 * to click the thumbnail. `loading="lazy"` is deliberately not set: the server
 * renders all twelve in one batch behind one slide open, so deferring them would
 * stagger requests into a response that is already sitting in a memo.
 *
 * Only the chosen cell is disabled, and only because clicking it would ask for
 * the tile already on screen. The other eleven stay live even mid-request, so a
 * reader who spots a better block while one is loading can just click it.
 */
function CandidateCell({
  candidate,
  rank,
  share,
  uploadId,
  options,
  refining,
  onPick,
}: CandidateCellProps) {
  const classes = ['od-tiles__cell']
  if (candidate.chosen) classes.push('od-tiles__cell--chosen')
  if (refining) classes.push('od-tiles__cell--waiting')

  return (
    <li className={classes.join(' ')}>
      <button
        type="button"
        onClick={() => onPick({ x: candidate.x, y: candidate.y })}
        disabled={candidate.chosen}
        aria-current={candidate.chosen ? 'true' : undefined}
        title={`Rank ${rank} — stain ${candidate.stain.toFixed(3)} × mixing ${candidate.mixing.toFixed(3)}, ${(candidate.tissueShare * 100).toFixed(0)}% tissue`}
      >
        <span className="od-tiles__thumb">
          <img
            src={densityCandidateUrl(uploadId, candidate, options)}
            alt={`Candidate block ranked ${rank}, at ${candidate.x}, ${candidate.y}`}
            width={THUMB_PX}
            height={THUMB_PX}
          />
          <span className="od-tiles__rank mono">{rank}</span>
          {candidate.chosen && <span className="od-tiles__flag">standing here</span>}
        </span>

        <span className="od-tiles__figures mono">
          {candidate.stain.toFixed(3)} × {candidate.mixing.toFixed(3)}
        </span>

        <span className="od-tiles__bar" aria-hidden>
          <span style={{ width: `${Math.max(0, Math.min(1, share)) * 100}%` }} />
        </span>
      </button>
    </li>
  )
}

export type { TileCandidate }
