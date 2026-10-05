/**
 * Step 4's screen, in the order the argument runs.
 *
 *   1. the headline — I₀ itself, and what empty glass measures as under it
 *   2. the panels — slide, sampled glass, the field, the slide flat-fielded
 *   3. the exclusion ladder — what counts as glass, and what does not
 *   4. the percentile ladder — the argument that the 95th is not doing the work
 *   5. the field verdict — flat or varying, and every test it had to pass
 *   6. the citation
 *
 * The headline pairs I₀ with the noise floor on purpose. I₀ alone is a number
 * nobody can check; the noise floor is that number's *consequence* — apply the
 * calibration back to the glass it came from and see what is left — and it is
 * stated in optical density, which is what step 5 produces and what every
 * threshold downstream has to clear.
 *
 * Rule 2 — calibration records, rescaling rewrites — is no longer stated on
 * screen, but it is still what the step does: nothing here changes a pixel.
 */

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import type { CalibrationOptions } from '@/api/calibration'
import { formatCount } from '@/lib/format'
import type { CalibrationReport } from '@/types/calibration'

import { CalibrationPanels } from './CalibrationPanels'
import { ExclusionLadder } from './ExclusionLadder'
import { FieldVerdict } from './FieldVerdict'
import { PercentileLadder } from './PercentileLadder'
import { WhitePointSwatch } from './WhitePointSwatch'

import './calibration.css'

interface CalibrationViewProps {
  report: CalibrationReport
  draft: number
  committed: number
  /** The percentile the server chooses on its own — what "back to" means. */
  defaultPercentile: number
  manual: boolean
  refining: boolean
  onDraft: (value: number) => void
  onResetToDefault: () => void
}

export function CalibrationView({
  report,
  draft,
  committed,
  defaultPercentile,
  manual,
  refining,
  onDraft,
  onResetToDefault,
}: CalibrationViewProps) {
  const { params, white, choice, noise } = report

  // The control is ahead of the server. Everything numeric on this screen belongs
  // to `committed`, so nothing may be phrased as if it described `draft` - the
  // same discipline step 3's screen keeps about its threshold.
  const pending = draft !== committed
  // The figures beside the ladder belong to `committed`, so they are stale from
  // the moment the control moves and stay stale until the run lands. One flag for
  // both halves of that wait: the debounce and the request.
  const busy = pending || refining

  const options: CalibrationOptions = {
    threshold: params.tissueThresholdSource === 'manual' ? params.tissueThreshold : null,
    percentile: manual ? committed : null,
  }

  return (
    <div className="cal">
      {/* --- 1. the headline ---------------------------------------------- */}
      <div className="cal-headline">
        <WhitePointSwatch
          rgb={white.rgb}
          hex={white.hex}
          saturated={white.saturated}
          label="The colour of blank glass on this slide"
        />

        <div className="cal-headline__figures">
          <div className="cal-headline__figure">
            <span className="cal-headline__value mono">{noise.worst.toFixed(3)}</span>
            <span className="cal-headline__caption">stain reading on blank glass</span>
            <span className="cal-headline__sub mono">
              anything weaker than this is noise, not stain
            </span>
          </div>

          <div className="cal-headline__figure cal-headline__figure--muted">
            <span className="cal-headline__value mono">
              {(report.glassShare * 100).toFixed(1)}%
            </span>
            <span className="cal-headline__caption">of the scan was blank glass</span>
            <span className="cal-headline__sub mono">
              {report.glassAreaMm2.toFixed(1)} mm² of glass, {formatCount(white.sampledPixels)} px
            </span>
          </div>
        </div>

        <p className="cal-headline__why">
          To say how much stain a pixel holds, we first need to know what no stain looks
          like. That depends on the scanner lamp, the coverslip and the white balance, so
          it is different on every slide. We measure it from the blank glass, which is the
          one part of the slide we know carries no stain.
        </p>
      </div>

      {/* --- 2. the panels ------------------------------------------------ */}
      <CalibrationPanels
        report={report}
        options={options}
        refining={refining || pending}
      />

      {/* --- 3. what counts as glass -------------------------------------- */}
      <ExclusionLadder exclusions={report.exclusions} />

      {/* --- 4. the percentile ------------------------------------------- */}
      <div className="cal-percentile">
        <PercentileLadder
          white={white}
          draft={draft}
          committed={committed}
          onDraft={onDraft}
        />

        <div
          className={busy ? 'cal-percentile__side is-busy' : 'cal-percentile__side'}
          aria-busy={busy}
        >
          {busy && (
            <span className="is-busy__marker">
              <Spinner size="sm" label={`sampling at ${draft.toFixed(1)}`} />
            </span>
          )}

          <p className="cal-percentile__why">
            We take the <strong>{committed.toFixed(1)}th</strong> brightest reading of the
            glass, not the brightest one. Too high and every stain reading comes out too
            strong; too low and some come out below zero.
          </p>

          <dl className="cal-facts mono is-busy__figures">
            <div>
              <dt>brightest pixel</dt>
              <dd>
                {white.ladder['100']
                  ? `${white.ladder['100'].r.toFixed(0)}, ${white.ladder['100'].g.toFixed(0)}, ${white.ladder['100'].b.toFixed(0)}`
                  : '—'}
                <span className="cal-facts__aside">a single glint decides it</span>
              </dd>
            </div>
            <div>
              <dt>middle pixel</dt>
              <dd>
                {white.ladder['50']
                  ? `${white.ladder['50'].r.toFixed(0)}, ${white.ladder['50'].g.toFixed(0)}, ${white.ladder['50'].b.toFixed(0)}`
                  : '—'}
                <span className="cal-facts__aside">debris drags it down</span>
              </dd>
            </div>
            <div>
              <dt>colour tint</dt>
              <dd>
                {(white.cast * 100).toFixed(2)}%
                <span className="cal-facts__aside">
                  {white.cast < 0.004 ? 'almost neutral' : 'the lamp has a slight tint'}
                </span>
              </dd>
            </div>
            <div>
              <dt>over-bright pixels</dt>
              <dd>
                {(Math.max(white.clipped.r, white.clipped.g, white.clipped.b) * 100).toFixed(2)}%
                <span className="cal-facts__aside">
                  {white.saturated ? 'the scanner ran out of range' : 'none — good'}
                </span>
              </dd>
            </div>
          </dl>

          {busy && (
            <p className="cal-percentile__pending">
              These numbers still belong to the{' '}
              <strong>{committed.toFixed(1)}th</strong> until the{' '}
              <strong>{draft.toFixed(1)}th</strong> has actually been measured.
            </p>
          )}

          {/* Offered for as long as the control is off the default, the wait
              included: a viewer who has just dragged somewhere they did not mean
              to should not have to sit out a run before undoing it. The label
              names the server's own percentile and not the current one, which is
              the whole point of a revert. */}
          {manual && (
            <Button variant="secondary" onClick={onResetToDefault} loading={refining}>
              Back to the automatic value ({defaultPercentile.toFixed(0)}th)
            </Button>
          )}
        </div>
      </div>

      {/* --- 5. the field ------------------------------------------------- */}
      <FieldVerdict choice={choice} surface={report.surface} />

      {/* --- 6. the citation ---------------------------------------------- */}
      <p className="cal-notes__citation">
        <Badge tone="neutral">Ruifrok &amp; Johnston 2001 · Beer 1852</Badge>{' '}
        {report.citation}
      </p>
    </div>
  )
}
