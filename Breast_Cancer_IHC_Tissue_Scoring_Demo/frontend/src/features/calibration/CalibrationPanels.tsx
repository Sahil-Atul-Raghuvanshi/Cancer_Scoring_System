/**
 * The four panels, as a strip, because the sequence is the explanation.
 *
 *   1  the slide            what went in
 *   2  where I₀ came from   the sampled glass, patches marked in green
 *   3  the illumination     the fitted field, as a heatmap
 *   4  flat-fielded         the slide with that field divided back out
 *
 * Panels 3 and 4 are a pair and only the pair is honest. A heatmap of a fitted
 * field is a picture of a hypothesis; the same field divided back out of the slide
 * is the test of it. If the surface is real the corrected panel is visibly flatter
 * across the glass; if it is not, the two look the same and the reader has caught
 * the step overfitting.
 *
 * When the flat white point is in force, panel 3 is a plain grey and panel 4 is
 * the thumbnail again. That is not a missing feature — it is the step visibly
 * declining to invent a correction, and the captions say so rather than leaving
 * two dead panels on screen.
 */

import { calibrationPanelUrl } from '@/api/calibration'
import type { CalibrationOptions } from '@/api/calibration'
import type { CalibrationPanelName, CalibrationReport } from '@/types/calibration'

interface CalibrationPanelsProps {
  report: CalibrationReport
  options: CalibrationOptions
  /** True while a re-run is in flight, so the stale panels can dim. */
  refining: boolean
}

interface PanelSpec {
  name: CalibrationPanelName
  step: string
  title: string
  caption: string
}

function specs(report: CalibrationReport): PanelSpec[] {
  const usingSurface = report.choice.mode === 'surface'
  const amplitude = report.surface
    ? `${(report.surface.amplitude * 100).toFixed(1)}%`
    : '—'

  return [
    {
      name: 'thumbnail',
      step: '1',
      title: 'The slide',
      caption: 'Read on step 3’s own grid, so the tissue mask needs no resampling.',
    },
    {
      name: 'glass',
      step: '2',
      title: 'Where I₀ came from',
      caption:
        'Tinted glass is what was sampled. Green squares are patches that voted for the illumination fit; faint ones held too little glass to count.',
    },
    {
      name: 'field',
      step: '3',
      title: usingSurface ? 'The illumination, fitted' : 'No field fitted',
      caption: usingSurface
        ? `Contrast stretched to its own range — the real swing is ${amplitude}, which would be invisible on a 0–255 scale.`
        : 'Flat grey, because the fitted surface did not clear its own tests and one white point is in force. The reason is below.',
    },
    {
      name: 'corrected',
      step: '4',
      title: usingSurface ? 'Flat-fielded' : 'Unchanged, deliberately',
      caption: usingSurface
        ? 'The field divided back out. Compare the glass here with panel 1 — if the fit were wrong, nothing would have flattened.'
        : 'Divided by a constant, so only the level changed. This is what declining to invent a correction looks like.',
    },
  ]
}

export function CalibrationPanels({
  report,
  options,
  refining,
}: CalibrationPanelsProps) {
  return (
    <div className={refining ? 'cal-strip cal-strip--stale' : 'cal-strip'}>
      {specs(report).map((panel) => (
        <figure className="cal-strip__item" key={panel.name}>
          <div className="cal-strip__frame">
            {/* Not lazy: the strip is the step's primary content, and the
                sequence only reads as an argument if all four arrive together. */}
            <img
              src={calibrationPanelUrl(report.uploadId, panel.name, options)}
              alt={panel.caption}
            />
          </div>
          <figcaption>
            <span className="cal-strip__step mono">{panel.step}</span>
            <strong>{panel.title}</strong>
            <span className="cal-strip__caption">{panel.caption}</span>
          </figcaption>
        </figure>
      ))}
    </div>
  )
}
