/**
 * I₀ as a colour, with its three channels beside it.
 *
 * The most unglamorous component in the app and the one that carries the step's
 * argument, because it is the thing you can put next to another slide's. A number
 * like `(193, 191, 186)` does not read as "this slide's white"; a square of that
 * colour does, and two squares side by side make the case for calibrating per
 * slide without a word of explanation.
 *
 * Drawn in the browser from the reported triple rather than fetched from the
 * server's `swatch.png`. They are the same colour by construction - the swatch
 * endpoint exists for downloading and for embedding elsewhere - and a CSS
 * background costs no request, which matters because the comparison view draws
 * one of these per slide.
 */

import type { Channels } from '@/types/calibration'

interface WhitePointSwatchProps {
  rgb: Channels
  hex: string
  /** Shown under the swatch when this is one of several. */
  label?: string
  size?: 'sm' | 'lg'
  /** A clipped channel means I₀ is an underestimate — worth marking on the chip. */
  saturated?: boolean
}

export function WhitePointSwatch({
  rgb,
  hex,
  label,
  size = 'lg',
  saturated = false,
}: WhitePointSwatchProps) {
  return (
    <figure className={`swatch swatch--${size}`}>
      <div
        className="swatch__chip"
        style={{ background: hex }}
        // The colour is the content, so it needs a text equivalent rather than
        // being left as decoration.
        role="img"
        aria-label={`I₀ is red ${rgb.r.toFixed(0)}, green ${rgb.g.toFixed(0)}, blue ${rgb.b.toFixed(0)}`}
      >
        {saturated && (
          <span className="swatch__warn mono" title="a channel is clipped at 255">
            clipped
          </span>
        )}
      </div>

      <figcaption className="swatch__caption">
        {label && <span className="swatch__label">{label}</span>}
        <span className="swatch__hex mono">{hex}</span>
        <span className="swatch__channels mono">
          <i className="swatch__dot swatch__dot--r" />
          {rgb.r.toFixed(0)}
          <i className="swatch__dot swatch__dot--g" />
          {rgb.g.toFixed(0)}
          <i className="swatch__dot swatch__dot--b" />
          {rgb.b.toFixed(0)}
        </span>
      </figcaption>
    </figure>
  )
}
