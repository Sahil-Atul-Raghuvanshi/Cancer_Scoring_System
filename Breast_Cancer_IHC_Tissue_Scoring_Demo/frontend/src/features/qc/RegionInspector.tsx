/**
 * Click a spot on the overlay; get that region's pixels and its numbers.
 *
 * This is where the explanation becomes concrete. The grid measures every patch
 * at the artefact model's own resolution, which is coarse enough that blur is
 * partly averaged away — so a region picked here is re-measured server-side at
 * the pipeline's working resolution, where sharpness actually means something.
 *
 * The consequence is stated in the panel rather than hidden: a region measured
 * finer than the grid has no like-for-like baseline, so no ratio is quoted for
 * it. Showing one would be comparing two different resolutions.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { fetchQCRegion, qcOverlayUrl, qcRegionImageUrl } from '@/api/qc'
import { Badge } from '@/components/ui/Badge'
import type { QCClassShare, QCRegionExplain } from '@/types/qc'

import './qc.css'

interface RegionInspectorProps {
  uploadId: string
  /** Level-0 dimensions, so a click can be converted into slide coordinates. */
  slideWidth: number
  slideHeight: number
  classes: QCClassShare[]
  /** The resolution step 1 is working at, which is what a region is measured at. */
  targetMpp: number | null
}

/** Region extent in level-0 pixels. About 0.45 mm on a 40x scan. */
const REGION_PX = 2048

export function RegionInspector({
  uploadId,
  slideWidth,
  slideHeight,
  classes,
  targetMpp,
}: RegionInspectorProps) {
  const [point, setPoint] = useState<{ x: number; y: number } | null>(null)
  const [result, setResult] = useState<QCRegionExplain | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const figure = useRef<HTMLDivElement | null>(null)

  const inspect = useCallback(
    (fraction: { x: number; y: number }) => {
      const x = Math.round(fraction.x * slideWidth - REGION_PX / 2)
      const y = Math.round(fraction.y * slideHeight - REGION_PX / 2)
      setPoint({ x: Math.max(0, x), y: Math.max(0, y) })
    },
    [slideHeight, slideWidth],
  )

  useEffect(() => {
    if (!point) return

    const controller = new AbortController()
    setBusy(true)
    setError(null)

    fetchQCRegion(uploadId, { ...point, size: REGION_PX, targetMpp }, controller.signal)
      .then((next) => {
        if (!controller.signal.aborted) setResult(next)
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setError(cause instanceof Error ? cause.message : 'could not inspect that region')
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false)
      })

    return () => controller.abort()
  }, [point, targetMpp, uploadId])

  const onClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const box = figure.current?.getBoundingClientRect()
    if (!box) return
    inspect({
      x: (event.clientX - box.left) / box.width,
      y: (event.clientY - box.top) / box.height,
    })
  }

  const dominant = result?.dominant
    ? classes.find((item) => item.key === result.dominant)
    : null

  return (
    <div className="qc-inspect">
      <div className="qc-inspect__head">
        <span className="eyebrow">inspect a region</span>
        <span className="qc-inspect__hint">
          Click anywhere on the map — the region is re-measured at{' '}
          {targetMpp ? `${targetMpp} µm/px` : 'the working resolution'}, not at the grid&rsquo;s.
        </span>
      </div>

      <div className="qc-inspect__body">
        {/* --- the pickable map ------------------------------------------- */}
        <div
          className="qc-inspect__map"
          ref={figure}
          onClick={onClick}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') inspect({ x: 0.5, y: 0.5 })
          }}
          aria-label="Click a region of the slide to inspect it"
        >
          <img src={qcOverlayUrl(uploadId)} alt="Artefact overlay — click to inspect" />
          {point && (
            <span
              className="qc-inspect__pin"
              style={{
                left: `${((point.x + REGION_PX / 2) / slideWidth) * 100}%`,
                top: `${((point.y + REGION_PX / 2) / slideHeight) * 100}%`,
              }}
              aria-hidden
            />
          )}
        </div>

        {/* --- what was found there --------------------------------------- */}
        <div className="qc-inspect__panel">
          {!point && (
            <p className="qc-inspect__empty">
              Nothing selected. Pick a tinted area to see why it was rejected, or a clean area to
              see what a passing region measures.
            </p>
          )}

          {error && <p className="qc-inspect__error">{error}</p>}

          {point && (
            <>
              <figure className="qc-inspect__tile">
                <img
                  src={qcRegionImageUrl(uploadId, { ...point, size: REGION_PX, out: 512 })}
                  alt="The inspected region at working resolution"
                />
                <figcaption className="mono">
                  {point.x.toLocaleString()}, {point.y.toLocaleString()} ·{' '}
                  {REGION_PX.toLocaleString()} px
                  {result ? ` · measured at ${result.measuredAtMpp} µm/px` : ''}
                </figcaption>
              </figure>

              {busy && !result && <p className="qc-inspect__busy">measuring…</p>}

              {result && (
                <>
                  <div className="qc-inspect__verdict">
                    {dominant && (
                      <Badge tone={dominant.isArtefact ? 'danger' : 'success'}>
                        {dominant.label}
                      </Badge>
                    )}
                    <p>{result.verdict}</p>
                  </div>

                  <ul className="qc-inspect__metrics">
                    {result.metrics.map((metric) => (
                      <li key={metric.key}>
                        <span className="qc-inspect__metric-label" title={metric.description}>
                          {metric.label}
                        </span>
                        <span className="qc-inspect__metric-value mono">
                          {metric.value.toPrecision(3)}
                        </span>
                        <span className="qc-inspect__metric-ratio mono">
                          {metric.ratioToClean === null
                            ? '—'
                            : metric.ratioToClean < 1
                              ? `${(metric.ratioToClean * 100).toFixed(0)}% of clean`
                              : `${metric.ratioToClean.toFixed(2)}× clean`}
                        </span>
                      </li>
                    ))}
                  </ul>

                  {result.metrics.every((metric) => metric.ratioToClean === null) && (
                    <p className="qc-inspect__caveat">
                      No ratios shown: this region was measured finer than the grid, so there is
                      no like-for-like clean-tissue baseline on this slide to divide by.
                      Comparing across resolutions would be meaningless.
                    </p>
                  )}
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
