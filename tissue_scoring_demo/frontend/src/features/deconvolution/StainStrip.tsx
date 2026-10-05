/**
 * The step's picture: the mixed tile, then one picture per stain that is
 * actually on the slide, then what is left over.
 *
 * Read left to right it makes the whole argument without a single technical word.
 * The first panel is what the scanner saw — the dyes tangled together in every
 * pixel. The middle ones are the same tile with each colour on its own. The last
 * one should look nearly empty, and "nearly empty" is the good outcome: it means
 * the stains account for what was there.
 *
 * Each stain is drawn in its own dye's colour, so nobody has to consult a legend
 * to know which picture is which. Brighter always means more of that stain — a
 * single progression per panel, never a rainbow, because a rainbow invents bands
 * the data does not have.
 */

import type { ReactNode } from 'react'

import { deconvolutionPanelUrl } from '@/api/deconvolution'
import type { DeconvolutionOptions } from '@/api/deconvolution'
import type {
  DeconvolutionBasisName,
  DeconvolutionPanelName,
  DeconvolutionReport,
} from '@/types/deconvolution'

interface StainStripProps {
  report: DeconvolutionReport
  basis: DeconvolutionBasisName
  options: DeconvolutionOptions
  /**
   * Which slide these pictures came from.
   *
   * On an H&E section there is no brown marker at all, so the DAB panel is not
   * shown there: the arithmetic still produces a channel, but it is separating out
   * a dye that was never applied, and the panel is stretched to its own range — so
   * near-nothing is drawn as a full, convincing brown picture. Showing it invites
   * exactly the reading it should prevent. An unknown slide keeps the panel, since
   * "unknown" is not the same claim as "no marker here".
   */
  slideRole: 'he' | 'ihc' | 'unknown'
}

interface StripPanel {
  key: string
  panel: DeconvolutionPanelName
  alt: string
  title: ReactNode
  caption: ReactNode
}

export function StainStrip({ report, basis, options, slideRole }: StainStripProps) {
  const isHe = slideRole === 'he'
  const isIhc = slideRole === 'ihc'
  const { tile } = report
  const active = basis === 'estimated' ? report.estimated : report.fixed

  const dab = active?.channels.find((channel) => channel.name === 'dab')
  const leftover = active?.residualShare ?? report.fixed.residualShare

  const panels: StripPanel[] = [
    {
      key: 'tile',
      panel: 'tile',
      alt: 'The tile as the scanner recorded it, with both stains mixed together',
      title: <strong>{isHe ? 'The slide as scanned' : 'Both stains, mixed'}</strong>,
      caption: (
        <>
          {tile.tileUm.toFixed(0)} µm of the slide, exactly as scanned.{' '}
          {isHe
            ? 'Blue and pink sit on top of each other in every pixel here.'
            : 'Blue and brown sit on top of each other in every pixel here.'}
        </>
      ),
    },
    {
      key: 'haematoxylin',
      panel: 'haematoxylin',
      alt: 'How much blue stain each pixel carries',
      title: (
        <strong className="cd-strip__label cd-strip__label--h">Blue stain only</strong>
      ),
      caption: (
        <>
          The counterstain, which marks where the cell nuclei are.{' '}
          {isIhc
            ? 'Both slides carry this dye, so cells can be found here without the brown marker affecting where they are.'
            : 'This picture is what the tissue model reads instead of the colour photo.'}
        </>
      ),
    },
  ]

  if (!isHe) {
    panels.push({
      key: 'dab',
      panel: 'dab',
      alt: 'How much brown marker each pixel carries',
      title: (
        <strong className="cd-strip__label cd-strip__label--dab">Brown marker only</strong>
      ),
      caption: isIhc ? (
        <>
          The marker being tested for. This is the picture the score is measured from
          {dab ? `, and it peaks at ${dab.p99.toFixed(2)} here` : ''}.
        </>
      ) : (
        <>
          The marker picture. Until this slide is confirmed as a marker slide, treat a
          strong-looking picture here with care
          {dab ? `; it peaks at ${dab.p99.toFixed(2)}` : ''}.
        </>
      ),
    })
  }

  panels.push({
    key: 'residual',
    panel: 'residual',
    alt: 'What neither stain explains',
    title: <strong>Left over</strong>,
    caption: (
      <>
        Anything the two dyes could not explain &mdash; {(leftover * 100).toFixed(0)}%
        of the colour here. Nearly empty is what you want.
      </>
    ),
  })

  return (
    <div className={`cd-strip${panels.length === 3 ? ' cd-strip--three' : ''}`}>
      {panels.map((item, index) => (
        <figure className="cd-strip__item" key={item.key}>
          <div className="cd-strip__frame">
            <img
              src={deconvolutionPanelUrl(report.uploadId, item.panel, basis, options)}
              alt={item.alt}
            />
          </div>
          <figcaption>
            <span className="cd-strip__step mono">{index + 1}</span>
            {item.title}
            <span className="cd-strip__caption">{item.caption}</span>
          </figcaption>
        </figure>
      ))}
    </div>
  )
}
