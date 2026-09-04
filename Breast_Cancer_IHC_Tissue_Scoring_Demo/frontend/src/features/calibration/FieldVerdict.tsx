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
      label: 'usable patches',
      value: `${choice.patchesUsed}`,
      limit: `≥ ${choice.minPatches}`,
      passed: choice.patchesUsed >= choice.minPatches,
      what:
        'Six coefficients can be solved from six points and trusted from far more. Below the bar the fit would mostly be extrapolating into the parts of the slide that had no glass.',
    },
    {
      label: 'swing / residual',
      value: surface ? `${surface.snr.toFixed(1)}×` : '—',
      limit: `≥ ${choice.snrRequired}×`,
      passed: Boolean(surface) && choice.snr >= choice.snrRequired,
      what:
        'How much the field bends across the slide, against how far it typically misses a patch sample by. Comparable numbers mean the bend is the scatter.',
    },
    {
      label: 'leverage over tissue',
      value: surface ? surface.leverage.toFixed(2) : '—',
      limit: `≤ ${choice.leverageLimit}`,
      passed: Boolean(surface) && choice.leverage <= choice.leverageLimit,
      what:
        'How far outside its own samples the fit reaches, measured where a density will actually be computed. Around 1 at the edge of the samples’ spread and unbounded past it.',
    },
  ]
}

export function FieldVerdict({ choice, surface }: FieldVerdictProps) {
  const varying = choice.mode === 'surface'

  return (
    <div className="field">
      <div className="field__head">
        <span className="eyebrow">is one white point enough for this slide?</span>
        <Badge tone={varying ? 'accent' : 'neutral'}>
          {varying ? 'I₀ varies with position' : 'one flat I₀'}
        </Badge>
      </div>

      {!choice.fitted ? (
        <p className="field__none">
          No surface was fitted at all: fewer patches held enough glass than a quadratic
          has coefficients, so there is nothing to fit one through. One flat white point
          stands, and the tests below have nothing to report.
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
                is what a single flat white point would have left in place across this
                slide — a{' '}
                <strong>{(surface.amplitude * 100).toFixed(1)}%</strong>{' '}
                brightest-to-dimmest swing, converted into the quantity step 5 actually
                reports. Stated as a percentage of intensity it is uninterpretable; stated
                as optical density you can weigh it against a DAB signal of 0.3 and
                against the gap between a 1+ and a 2+ call.
              </>
            ) : (
              <>
                is the most that ignoring the fitted surface can cost, so setting it aside
                is cheap. That is the honest reason to prefer the flat value here: not that
                the slide is perfectly lit, but that the correction on offer is smaller than
                the evidence for it.
              </>
            )}
          </p>
        </div>
      )}

      <p className="field__reason">{choice.reason}</p>
    </div>
  )
}
