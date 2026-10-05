/**
 * Flat or varying — which field is in force, and every test it had to pass.
 *
 * Built to the same contract as step 3's threshold card: the step is allowed to
 * pick, but not to pick quietly. Both answers are on screen, along with the
 * statistics the decision turned on and the cutoffs they were compared against,
 * whichever way it went.
 *
 * Three tests, and they fail in different ways, so each gets its own row rather
 * than being collapsed into a verdict:
 *
 *   patches    enough samples to determine six coefficients and to trust them.
 *   swing/noise  the fit must bend more than it misses its own samples by. Below
 *                that the bend *is* the scatter, and a white point curving to
 *                follow noise is worse than a flat one that does not.
 *   leverage     the samples must enclose the tissue. Glass rings the section, so
 *                the frame's corners are always extrapolated and never read; what
 *                matters is whether the ring pins the field down where a density
 *                is actually computed.
 *
 * The headline figure is `odError` and not the amplitude, because "1.6% of I₀" is
 * not a number anyone can act on in a pipeline whose output is a density. 0.007 OD
 * is: it is nothing beside a DAB signal of 0.3, and it is worth knowing beside the
 * gap between a 1+ and a 2+ call.
 */

import { Badge } from '@/components/ui/Badge'
import type { CalibrationChoice, CalibrationSurface } from '@/types/calibration'

interface FieldVerdictProps {
  choice: CalibrationChoice
  surface: CalibrationSurface | null
}

interface Test {
  label: string
  value: string
  limit: string
  passed: boolean
  what: string
}

function tests(choice: CalibrationChoice, surface: CalibrationSurface | null): Test[] {
  return [
    {
      label: 'enough sample points',
      value: `${choice.patchesUsed}`,
      limit: `≥ ${choice.minPatches}`,
      passed: choice.patchesUsed >= choice.minPatches,
      what:
        'Too few glass patches and the lighting map would mostly be guesswork.',
    },
    {
      label: 'real variation vs noise',
      value: surface ? `${surface.snr.toFixed(1)}×` : '—',
      limit: `≥ ${choice.snrRequired}×`,
      passed: Boolean(surface) && choice.snr >= choice.snrRequired,
      what:
        'The lighting must vary more than the measurement wobbles. If the two are similar, the pattern is just noise.',
    },
    {
      label: 'samples cover the tissue',
      value: surface ? surface.leverage.toFixed(2) : '—',
      limit: `≤ ${choice.leverageLimit}`,
      passed: Boolean(surface) && choice.leverage <= choice.leverageLimit,
      what:
        'The glass patches must surround the tissue. Otherwise the lighting map is guessing where it matters most.',
    },
  ]
}

export function FieldVerdict({ choice, surface }: FieldVerdictProps) {
  const varying = choice.mode === 'surface'

  return (
    <div className="field">
      <div className="field__head">
        <span className="eyebrow">is the lighting even across this slide?</span>
        <Badge tone={varying ? 'accent' : 'neutral'}>
          {varying ? 'lighting varies' : 'lighting is even'}
        </Badge>
      </div>

      {!choice.fitted ? (
        <p className="field__none">
          Not enough glass patches to check the lighting at all, so one value is used for
          the whole slide and the tests below have nothing to report.
        </p>
      ) : (
        <dl className="field__tests">
          {tests(choice, surface).map((test) => (
            <div
              className={test.passed ? 'field__test field__test--pass' : 'field__test'}
              key={test.label}
            >
              <dt>
                <span className="field__mark" aria-hidden>
                  {test.passed ? '✓' : '✕'}
                </span>
                {test.label}
              </dt>
              <dd>
                <span className="field__value mono">{test.value}</span>
                <span className="field__limit mono">{test.limit}</span>
                <span className="field__what">{test.what}</span>
              </dd>
            </div>
          ))}
        </dl>
      )}

      {/* The cost of the decision, in the units step 5 produces. This is the
          figure that says whether any of it matters. */}
      {surface && (
        <div className="field__cost">
          <div className="field__cost-figure">
            <span className="field__cost-value mono">
              {surface.odError.toFixed(3)}
            </span>
            <span className="field__cost-unit">OD</span>
          </div>
          <p>
            {varying ? (
              <>
                is the error a single value would have left across this slide &mdash; a{' '}
                <strong>{(surface.amplitude * 100).toFixed(1)}%</strong> difference between
                the brightest and dimmest corner. For comparison, a clear brown marker
                reads about 0.3, so this is worth correcting.
              </>
            ) : (
              <>
                is the most that ignoring the lighting pattern could cost, which is very
                little. The correction on offer is smaller than the evidence for it, so we
                use one value for the whole slide.
              </>
            )}
          </p>
        </div>
      )}

      <p className="field__reason">{choice.reason}</p>
    </div>
  )
}
