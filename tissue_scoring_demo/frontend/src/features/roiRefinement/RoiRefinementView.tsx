/**
 * Step 11's screen: the rough square becoming the real outline, one region at a time.
 *
 *   1. overall progress — one segment per region, so the whole pass reads in one strip
 *   2. the region list — every selected area with its own bar, clickable
 *   3. the comparison for whichever region is open: rough square → the segmentation,
 *      live while it is being made and as a finished overlay once it is
 *   4. the whole slide, before and after, once something has finished
 *   5. the technical detail, folded away
 *
 * **The comparison is the point of the screen, so it is always on screen.** As soon as
 * one region finishes, it opens; a region still running shows the model painting it
 * rather than an empty frame. Nobody waits for the last region to see the first.
 *
 * **Progress is shown three times, and each one answers a different question.** The
 * segmented bar at the top says how far through *everything* the pass is and how many
 * regions there are. Each row's own bar says where *that* region got to — which is what
 * a single shared bar cannot say, since regions differ in size by an order of magnitude
 * and one large one can be ten minutes on its own. And the open region's bar sits under
 * the picture it describes, beside the patch count it is drawn from.
 *
 * **A finished row's bar changes meaning, and says so.** While a region runs, its bar is
 * patches done; once it is complete, the same bar becomes the share of the coarse square
 * that survived refinement — captioned differently and drawn in the success colour. That
 * is deliberate: a list of full bars would say nothing at the end of a pass, and the
 * number that matters then is how much of each square was actually tumour.
 *
 * **The right-hand panel must never look like the left one.** If BEETLE had simply
 * returned the rectangle it was given, the overlay would still look plausible — a
 * boundary drawn on tissue always does. That is what the live paint and the finished
 * mask panel are for: both show the model's answer for every pixel of the box, so a
 * rectangle would be obvious.
 */

import { useEffect, useMemo, useState } from 'react'

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { PaintCanvas, type PaintFrame, type PaintView } from '@/components/paint/PaintCanvas'
import { refinementPanelUrl, refinementRegionUrl } from '@/api/roiRefinement'
import type {
  RefinedRegion,
  RefinementPaint,
  RefinementReport,
  RefinementRun,
  RegionProgress,
} from '@/types/roiRefinement'

import '@/components/paint/paint.css'
import './roiRefinement.css'

interface RoiRefinementViewProps {
  report: RefinementReport
  run: RefinementRun | null
  running: boolean
  /** The region in progress, as flat `row, col, class` triples. Stable identity. */
  painted: number[]
  /** One base64 pixel mask per window in `painted`, same order. Stable identity. */
  paintedMasks: string[]
  /** Windows in `painted`. The number that moves as the region is segmented. */
  paintedCount: number
  onRetry: (roiId: string) => void
  onRestart: () => void
  onCancel: () => void
}

/** The stages a region moves through, in order, with the word the screen shows. */
const STAGES = [
  { key: 'extracting', label: 'Cut out the area' },
  { key: 'segmenting', label: 'Look at every pixel' },
  { key: 'tracing', label: 'Draw the outline' },
] as const

/** How far through `STAGES` a region's state puts it. `complete` is past all of them. */
const STAGE_ORDER: Record<string, number> = {
  pending: 0,
  extracting: 1,
  segmenting: 2,
  tracing: 3,
  complete: 4,
  failed: -1,
  skipped: -1,
}

const STATE_TONE = {
  pending: 'neutral',
  extracting: 'accent',
  segmenting: 'accent',
  tracing: 'accent',
  complete: 'success',
  failed: 'danger',
  skipped: 'neutral',
} as const

const STATE_WORD = {
  pending: 'Waiting',
  extracting: 'Cutting out',
  segmenting: 'Segmenting',
  tracing: 'Tracing',
  complete: 'Done',
  failed: 'Failed',
  skipped: 'Skipped',
} as const

/** Which colour a bar is drawn in. The row's state, reduced to the three that matter. */
type BarTone = 'waiting' | 'working' | 'done' | 'failed'

function toneOf(state: string): BarTone {
  if (state === 'complete') return 'done'
  if (state === 'failed' || state === 'skipped') return 'failed'
  if (state === 'pending') return 'waiting'
  return 'working'
}

/**
 * Patches done of patches priced, clamped.
 *
 * The clamp is not defensive rounding. A region's total is step 9's count until the
 * region starts and the grid's own afterwards — step 9 counts the candidate's windows
 * and step 11 counts every window whose core reaches the *padded* box — so the total
 * can move once, upward, at the moment a region begins. A bar must not be allowed past
 * its own end by a count that was provisional.
 */
function shareOf(row: RegionProgress | undefined): number {
  if (!row || row.windowsTotal <= 0) return 0
  return Math.min(1, row.windowsDone / row.windowsTotal)
}

/** One bar. Width is a share; the tone says what kind of share it is. */
function Bar({
  share,
  tone,
  label,
}: {
  share: number
  tone: BarTone
  label: string
}) {
  return (
    <div
      className={`refine__bar refine__bar--${tone}`}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(share * 100)}
      aria-label={label}
    >
      <span className="refine__barFill" style={{ width: `${share * 100}%` }} />
    </div>
  )
}

export function RoiRefinementView({
  report,
  run,
  running,
  painted,
  paintedMasks,
  paintedCount,
  onRetry,
  onRestart,
  onCancel,
}: RoiRefinementViewProps) {
  const { regions } = report

  /**
   * Which region the comparison is showing. Held here rather than derived, so a viewer
   * who clicked back to region 2 is not dragged forward every time region 7 finishes —
   * but seeded from whatever is worth looking at when the screen has no answer yet.
   */
  const [openId, setOpenId] = useState<string | null>(null)

  useEffect(() => {
    if (openId && regions.some((one) => one.roiId === openId)) return
    const firstDone = regions.find((one) => one.state === 'complete')
    const current = run?.currentRoiId
      ? regions.find((one) => one.roiId === run.currentRoiId)
      : undefined
    setOpenId((firstDone ?? current ?? regions[0])?.roiId ?? null)
  }, [openId, regions, run?.currentRoiId])

  const open = regions.find((one) => one.roiId === openId) ?? null
  const position = open ? regions.indexOf(open) : -1
  // The neighbours as values rather than as indices. `noUncheckedIndexedAccess` is on,
  // so `regions[position - 1]` is `RefinedRegion | undefined` however carefully the
  // bounds were checked a line earlier - and taking them here means the two callbacks
  // below are simply present or absent.
  const previous = position > 0 ? regions[position - 1] : undefined
  const next = position >= 0 ? regions[position + 1] : undefined

  /**
   * Each region's live position, by id.
   *
   * From the run rather than the report, because it exists before the report's row
   * does: a region has a window count and a state from the moment the pass is queued,
   * and rings only once it finishes.
   */
  const progress = useMemo(() => {
    const table = new Map<string, RegionProgress>()
    for (const row of run?.progress ?? []) table.set(row.roiId, row)
    return table
  }, [run?.progress])

  const doneCount = run?.total ? run.done : report.completed
  const totalCount = run?.total || report.selected
  // Narrowed into its own value rather than read off `run` at the point of use: the
  // caption and the aria label both need it, and a nullable field read twice is a
  // nullable field TypeScript has to be argued with twice.
  const windows =
    run && run.windowsTotal > 0
      ? { done: run.windowsDone, total: run.windowsTotal, share: run.windowsDone / run.windowsTotal }
      : null

  /**
   * The segments of the top bar, one per region.
   *
   * Falls back to the report's rows before a run exists, so a finished pass revisited
   * later still shows a strip of completed segments rather than nothing — the run is
   * in-process memory and does not survive a reload, but the report is on disk.
   */
  const segments: { roiId: string; state: string; share: number }[] = useMemo(() => {
    if (run?.progress?.length) {
      return run.progress.map((row) => ({
        roiId: row.roiId,
        state: row.state,
        share: row.state === 'complete' ? 1 : shareOf(row),
      }))
    }
    return regions.map((region) => ({
      roiId: region.roiId,
      state: region.state,
      share: region.state === 'complete' ? 1 : 0,
    }))
  }, [regions, run?.progress])

  return (
    <div className="refine">
      <header className="refine__intro">
        <p className="refine__lede">
          Each area you chose is checked pixel by pixel, so the rough square becomes the
          real edge of the tumour. From here on everything is measured inside these
          outlines. The squares are not used again.
        </p>
      </header>

      {(running || report.state === 'partial' || report.failed > 0) && (
        <section className="refine__progress">
          <div className="refine__progressHead">
            <span className="refine__progressCount">
              {doneCount} / {totalCount} areas
            </span>
            {run?.currentRoiId && running && (
              <span className="refine__progressNow mono">{run.currentRoiId}</span>
            )}
            {running && (
              <Button variant="ghost" onClick={onCancel}>
                Stop after this one
              </Button>
            )}
          </div>

          {/*
            One segment per area rather than one bar for the pass. The strip is the list
            below it in miniature: you can see at a glance that the third of twelve is
            half painted and the rest are waiting, which a single bar at 21% cannot say.
          */}
          <div
            className="refine__segments"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={totalCount || 1}
            aria-valuenow={doneCount}
            aria-label={`${doneCount} of ${totalCount} areas refined`}
          >
            {segments.map((segment) => (
              <span
                key={segment.roiId}
                className={`refine__segment refine__segment--${toneOf(segment.state)}`}
                title={`${segment.roiId}: ${STATE_WORD[segment.state as keyof typeof STATE_WORD] ?? segment.state}`}
              >
                <span
                  className="refine__segmentFill"
                  style={{ width: `${segment.share * 100}%` }}
                />
              </span>
            ))}
          </div>

          {/*
            The finer count, and it earns its place: one large area can be ten minutes on
            its own. This is real — it is windows finished of windows priced — and not a
            timer pretending to be progress.
          */}
          {windows !== null && running && (
            <p className="refine__progressDetail mono">
              {windows.done.toLocaleString()} / {windows.total.toLocaleString()} patches (
              {(windows.share * 100).toFixed(0)}%)
            </p>
          )}
          {run?.message && <p className="refine__progressMessage">{run.message}</p>}
        </section>
      )}

      <div className="refine__body">
        <ol className="refine__list">
          {regions.map((region) => {
            const row = progress.get(region.roiId)
            const complete = region.state === 'complete'
            // A finished row's bar stops being progress and becomes the result: the
            // share of the coarse square that survived. Same bar, different colour,
            // different caption - and at the end of a pass it is the number that
            // matters, where a row of full bars would say nothing.
            const share = complete ? Math.min(1, region.keptShare) : shareOf(row)
            return (
              <li key={region.roiId}>
                <button
                  type="button"
                  className={`refine__row${
                    region.roiId === openId ? ' refine__row--open' : ''
                  }`}
                  onClick={() => setOpenId(region.roiId)}
                  aria-current={region.roiId === openId}
                >
                  <span className="refine__rowHead">
                    <span className="refine__rowId mono">{region.roiId}</span>
                    <Badge tone={STATE_TONE[region.state]}>
                      {STATE_WORD[region.state]}
                    </Badge>
                  </span>

                  <Bar
                    share={share}
                    tone={toneOf(region.state)}
                    label={
                      complete
                        ? `${region.roiId}: ${(share * 100).toFixed(0)}% of the square kept`
                        : `${region.roiId}: ${(share * 100).toFixed(0)}% of its patches done`
                    }
                  />

                  <span className="refine__rowFoot mono">
                    {complete ? (
                      <>
                        {region.areaMm2.toFixed(2)} mm² &middot;{' '}
                        {(region.keptShare * 100).toFixed(0)}% of the square kept
                      </>
                    ) : region.state === 'failed' ? (
                      'could not be segmented'
                    ) : row && row.windowsTotal > 0 && row.windowsDone > 0 ? (
                      <>
                        {row.windowsDone.toLocaleString()} /{' '}
                        {row.windowsTotal.toLocaleString()} patches
                      </>
                    ) : row && row.windowsTotal > 0 ? (
                      <>{row.windowsTotal.toLocaleString()} patches to do</>
                    ) : (
                      'waiting its turn'
                    )}
                  </span>
                </button>
              </li>
            )
          })}
        </ol>

        <div className="refine__stage">
          {open ? (
            <RegionComparison
              uploadId={report.uploadId}
              region={open}
              row={progress.get(open.roiId)}
              paint={run?.paint && run.paint.roiId === open.roiId ? run.paint : null}
              painted={painted}
              paintedMasks={paintedMasks}
              paintedCount={paintedCount}
              position={position}
              count={regions.length}
              onPrevious={previous ? () => setOpenId(previous.roiId) : undefined}
              onNext={next ? () => setOpenId(next.roiId) : undefined}
              onRetry={onRetry}
            />
          ) : (
            <p className="refine__empty">No areas were selected on the previous step.</p>
          )}
        </div>
      </div>

      {report.completed > 0 && (
        <section className="refine__summary">
          <h3 className="refine__summaryTitle">The whole slide, before and after</h3>
          <div className="refine__slides">
            <figure>
              <img
                src={refinementPanelUrl(report.uploadId, 'coarse')}
                alt="The whole slide with the chosen areas drawn as rough squares"
                loading="lazy"
              />
              <figcaption>The areas you chose, as rough squares</figcaption>
            </figure>
            <span className="refine__arrow" aria-hidden>
              →
            </span>
            <figure>
              <img
                src={refinementPanelUrl(report.uploadId, 'refined')}
                alt="The whole slide with only the pixel-level tumour outlines drawn"
                loading="lazy"
              />
              <figcaption>The real tumour outlines inside them</figcaption>
            </figure>
          </div>

          <dl className="refine__totals">
            <div>
              <dt>Rough squares</dt>
              <dd className="mono">{report.tileMm2.toFixed(2)} mm²</dd>
            </div>
            <div>
              <dt>Actual tumour inside them</dt>
              <dd className="mono">{report.refinedMm2.toFixed(2)} mm²</dd>
            </div>
            <div>
              <dt>Kept</dt>
              <dd className="mono">{(report.keptShare * 100).toFixed(0)}%</dd>
            </div>
            <div>
              <dt>Areas done</dt>
              <dd className="mono">
                {report.completed} / {report.selected}
              </dd>
            </div>
          </dl>

          {!running && (
            <Button variant="secondary" onClick={onRestart}>
              Run them all again
            </Button>
          )}
        </section>
      )}

      <details className="refine__details">
        <summary>How this was produced</summary>
        <div className="refine__detailsBody">
          <p>
            <strong>{report.model ?? 'BEETLE'}</strong>, a published segmentation
            network, run at its own fixed 0.5 µm/px over the chosen areas only. Field of
            view {report.fieldOfViewUm} µm, {report.folds} version(s), {report.padUm} µm
            of surrounding tissue added for context, and the answer stored at{' '}
            {report.maskMpp} µm/px.
          </p>
          <p>
            The area per region is counted from the pixels the network produced, not from
            the drawn outline, because the outline is simplified for display.
          </p>
          <p>
            {report.windows.toLocaleString()} patches in {report.seconds.toFixed(0)}s
            across {report.completed} region(s). Running this over the whole slide would
            take far longer, which is why it is limited to the chosen areas.
          </p>
          {report.notes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </div>
      </details>
    </div>
  )
}

interface RegionComparisonProps {
  uploadId: string
  region: RefinedRegion
  /** This region's live position, if the pass knows it. */
  row: RegionProgress | undefined
  /** The paint geometry, but only when the feed belongs to *this* region. */
  paint: RefinementPaint | null
  painted: number[]
  paintedMasks: string[]
  paintedCount: number
  position: number
  count: number
  onPrevious?: () => void
  onNext?: () => void
  onRetry: (roiId: string) => void
}

/**
 * One region, as two rectangles.
 *
 * The left is the coarse square, unchanging. The right is the same crop with the model's
 * answer on it — being painted window by window while the region runs, and the traced
 * boundary once it is done. It is the *same rectangle* throughout, which is what makes
 * the comparison legible: the picture does not move when the region finishes, it simply
 * stops being live.
 *
 * Both panels come from `input.png`, which is written when the region starts rather than
 * when it finishes precisely so that this can exist while the work is happening.
 */
function RegionComparison({
  uploadId,
  region,
  row,
  paint,
  painted,
  paintedMasks,
  paintedCount,
  position,
  count,
  onPrevious,
  onNext,
  onRetry,
}: RegionComparisonProps) {
  const reached = STAGE_ORDER[region.state] ?? 0
  const painting = paint !== null

  // Memoised because they are object props on a component whose effect depends on them:
  // rebuilt every render, the canvas would restart on every poll rather than on every
  // new window.
  const frame = useMemo<PaintFrame | null>(
    () =>
      paint
        ? {
            slideWidth: paint.slideWidth,
            slideHeight: paint.slideHeight,
            span: paint.span,
            stride: paint.stride,
          }
        : null,
    [paint],
  )

  /** Step 11 paints one region, so the view is that region's padded box. */
  const view = useMemo<PaintView | null>(
    () =>
      paint
        ? {
            x: paint.cropX,
            y: paint.cropY,
            width: paint.cropWidth,
            height: paint.cropHeight,
          }
        : null,
    [paint],
  )

  const share = region.state === 'complete' ? 1 : shareOf(row)

  return (
    <article className="compare">
      <header className="compare__head">
        <span className="compare__id mono">{region.roiId}</span>
        <Badge tone={STATE_TONE[region.state]}>{STATE_WORD[region.state]}</Badge>
        <nav className="compare__nav">
          <Button variant="ghost" onClick={onPrevious} disabled={!onPrevious}>
            ← Previous
          </Button>
          <span className="compare__position mono">
            {position + 1} / {count}
          </span>
          <Button variant="ghost" onClick={onNext} disabled={!onNext}>
            Next →
          </Button>
        </nav>
      </header>

      {region.state === 'failed' ? (
        <div className="compare__failed">
          <p className="compare__failedText">
            {region.error ?? 'This area could not be segmented.'}
          </p>
          <p className="compare__failedNote">
            The other areas were not affected and are not being re-run.
          </p>
          <Button variant="secondary" onClick={() => onRetry(region.roiId)}>
            Try this one again
          </Button>
        </div>
      ) : (
        <>
          <div
            className={`compare__panels${
              region.state === 'complete' ? '' : ' compare__panels--pair'
            }`}
          >
            <figure className="compare__panel">
              {painting || region.state === 'complete' ? (
                <img
                  src={refinementRegionUrl(uploadId, region.roiId, 'tile')}
                  alt={`${region.roiId} with the rough square drawn on it`}
                  loading="lazy"
                />
              ) : (
                <div className="compare__placeholder">
                  <span className="mono">waiting its turn</span>
                </div>
              )}
              <figcaption>
                <strong>Rough square</strong>
                <span className="mono">
                  {region.tileAreaMm2.toFixed(2)} mm²
                </span>
              </figcaption>
            </figure>

            <span className="compare__arrow" aria-hidden>
              →
            </span>

            <figure className="compare__panel compare__panel--live">
              {region.state === 'complete' ? (
                <img
                  src={refinementRegionUrl(uploadId, region.roiId, 'beetle_overlay')}
                  alt={`${region.roiId} with the pixel-level tumour outline drawn on it`}
                  loading="lazy"
                />
              ) : painting && frame && view && paint ? (
                <PaintCanvas
                  frame={frame}
                  view={view}
                  colours={paint.colours}
                  perPixel={paint.perPixel}
                  maskPx={paint.maskPx}
                  painted={painted}
                  paintedMasks={paintedMasks}
                  paintedCount={paintedCount}
                  runKey={paint.roiId}
                >
                  <img
                    src={refinementRegionUrl(uploadId, region.roiId, 'input')}
                    alt={`${region.roiId}, being segmented`}
                  />
                </PaintCanvas>
              ) : (
                <div className="compare__placeholder">
                  <span className="mono">waiting its turn</span>
                </div>
              )}
              <figcaption>
                <strong>
                  {region.state === 'complete'
                    ? 'Real outline'
                    : 'What the model is finding, pixel by pixel'}
                </strong>
                <span className="mono">
                  {region.state === 'complete'
                    ? `${region.areaMm2.toFixed(2)} mm²`
                    : row && row.windowsTotal > 0
                      ? `${row.windowsDone.toLocaleString()} / ${row.windowsTotal.toLocaleString()} patches`
                      : 'not started'}
                </span>
              </figcaption>

              {/*
                The bar that belongs to this rectangle, under the rectangle. The paint
                above it is the progress; this is the same progress as a number, for the
                stretch of a large region where a few more windows do not visibly change
                the picture.
              */}
              {region.state !== 'complete' && (
                <Bar
                  share={share}
                  tone={toneOf(region.state)}
                  label={`${region.roiId}: ${(share * 100).toFixed(0)}% of its patches done`}
                />
              )}
            </figure>

            {region.state === 'complete' && (
              <figure className="compare__panel compare__panel--mask">
                <img
                  src={refinementRegionUrl(uploadId, region.roiId, 'beetle_mask')}
                  alt={`What the model called every pixel inside ${region.roiId}`}
                  loading="lazy"
                />
                <figcaption>
                  <strong>What the model saw, pixel by pixel</strong>
                  <span className="mono">
                    {region.focusCount} shape(s), {region.holes} hole(s)
                  </span>
                </figcaption>
              </figure>
            )}
          </div>

          {painting && paint && (
            <div className="compare__legend">
              {paint.labels.map((label, id) => (
                <span key={label} className="compare__key">
                  <span
                    className="compare__swatch"
                    style={{ background: paint.colours[id] }}
                  />
                  {label}
                </span>
              ))}
            </div>
          )}

          {region.state === 'complete' ? (
            <p className="compare__verdict">
              The square covered {region.tileAreaMm2.toFixed(2)} mm². Of that,{' '}
              <strong>{region.areaMm2.toFixed(2)} mm²</strong> is actually tumour &mdash;{' '}
              {(region.keptShare * 100).toFixed(0)}%. The rest was supporting tissue, fat
              or glass that happened to fall inside the square.
            </p>
          ) : (
            <p className="compare__stages">
              {STAGES.map((stage, index) => (
                <span
                  key={stage.key}
                  className={`compare__stageWord${
                    reached > index + 1
                      ? ' compare__stageWord--done'
                      : reached === index + 1
                        ? ' compare__stageWord--now'
                        : ''
                  }`}
                >
                  {stage.label}
                </span>
              ))}
            </p>
          )}
        </>
      )}
    </article>
  )
}
