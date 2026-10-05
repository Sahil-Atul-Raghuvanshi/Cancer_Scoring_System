/**
 * Step 9's screen: two products from one class map.
 *
 *   1. what just happened, and the region every later step measures inside
 *   2. the borders — every class outlined, tissue included — drawn on the slide
 *      itself, with each enclosed region faintly tinted so the tissue inside it still
 *      reads. Click an invasive, DCIS or uncertain patch to fit a box around it and
 *      see it enlarged
 *   3. the largest DCIS and invasive regions, cropped from the slide and ranked by area
 *   4. download the whole border set as a QuPath-importable file
 *   5. how the scoring region itself was built, folded away — it is a different,
 *      already-validated product and does not need to compete with the borders above
 *      for the reader's first look
 *
 * **Stroma never appears in the legend's region counts or the crops.** It is the bulk
 * of every slide and the class nobody circles or clicks through; see `borders.py`'s
 * module docstring for why the three classes a reader is actually looking for stay
 * the only ones with regions on the wire. Its outline exists only to show
 * where the tissue is, not as a fourth clickable class.
 *
 * **The click targets are boxes and are never drawn as boxes.** A region's hit area has
 * to be the axis-aligned box around it — that is what a `<button>` is — but a grid of
 * bright rectangles over the borders hides the borders, which are the picture. So each
 * one is transparent until it is hovered, focused or selected, and only then takes its
 * class colour.
 */

import { useEffect, useState } from 'react'

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { roiPanelUrl, roiQupathUrl, roiRegionCropUrl, roiTopUrl } from '@/api/roi'
import { formatCount } from '@/lib/format'
import type { RoiClassName, RoiRegion, RoiReport } from '@/types/roi'

import './roi.css'

interface RoiViewProps {
  report: RoiReport
  onRebuild: () => Promise<void>
  rebuilding: boolean
}

/**
 * Step 8's own palette for the three border classes, restated here because the
 * report carries areas and rings, not colours — a picture of a chip needs one of its
 * own. See `classes.py: CLASS_COLOURS`; these three values are that table.
 */
const CLASS_INFO: { key: RoiClassName; label: string; colour: string }[] = [
  { key: 'invasive_epithelium', label: 'Invasive tumour', colour: '#ef4444' },
  { key: 'non_invasive_epithelium', label: 'Tumour inside a duct', colour: '#3b82f6' },
  { key: 'uncertain', label: 'Cannot be determined', colour: '#a855f7' },
]

/** Stroma's own colour on the panel — restated for the legend, not for a click target. */
const TISSUE_OUTLINE_COLOUR = '#facc15'

const TOP_RANKS = [1, 2, 3] as const

/** A region's outer ring, reduced to the axis-aligned box that fits around it. */
function ringBounds(region: RoiRegion): { x0: number; y0: number; x1: number; y1: number } | null {
  const outer = region.rings[0]
  if (!outer || outer.length === 0) return null

  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const [x, y] of outer) {
    if (x < x0) x0 = x
    if (y < y0) y0 = y
    if (x > x1) x1 = x
    if (y > y1) y1 = y
  }
  return x1 > x0 && y1 > y0 ? { x0, y0, x1, y1 } : null
}

interface SelectedRegion {
  key: RoiClassName
  index: number
  label: string
  colour: string
  areaMm2: number
}

/**
 * The click targets over the border panel: one box per invasive, DCIS or uncertain
 * region, transparent until pointed at.
 */
function HitLayer({
  report,
  selected,
  onSelect,
}: {
  report: RoiReport
  selected: SelectedRegion | null
  onSelect: (region: SelectedRegion) => void
}) {
  return (
    <div className="roi-map__hitboxes">
      {CLASS_INFO.map((entry) =>
        (report.classRegions[entry.key] ?? []).map((region) => {
          const bounds = ringBounds(region)
          if (!bounds) return null
          const isSelected = selected?.key === entry.key && selected.index === region.index
          return (
            <button
              key={`${entry.key}-${region.index}`}
              type="button"
              className={`roi-map__hit${isSelected ? ' roi-map__hit--selected' : ''}`}
              style={{
                left: `${(bounds.x0 / report.slideWidth) * 100}%`,
                top: `${(bounds.y0 / report.slideHeight) * 100}%`,
                width: `${((bounds.x1 - bounds.x0) / report.slideWidth) * 100}%`,
                height: `${((bounds.y1 - bounds.y0) / report.slideHeight) * 100}%`,
                // Read back by the hover, focus and selected rules in `roi.css` — the
                // colour has to come from here because it is per class, and it has to
                // be a variable because the resting state does not use it.
                ['--hit-colour' as string]: entry.colour,
              }}
              title={`${entry.label} · ${region.areaMm2.toFixed(2)} mm² — click to enlarge it below`}
              onClick={() =>
                onSelect({
                  key: entry.key,
                  index: region.index,
                  label: entry.label,
                  colour: entry.colour,
                  areaMm2: region.areaMm2,
                })
              }
            >
              <span className="roi-visually-hidden">
                {entry.label} region, {region.areaMm2.toFixed(2)} mm&sup2;
              </span>
            </button>
          )
        }),
      )}
    </div>
  )
}

export function RoiView({ report, onRebuild, rebuilding }: RoiViewProps) {
  const { ledger, params } = report

  const [selected, setSelected] = useState<SelectedRegion | null>(null)

  // A rebuild is a different region under the same screen — the previously selected
  // index may no longer exist, so the enlarged panel below is cleared rather than
  // asking the server for a region that might now be a 404 or, worse, someone else's.
  useEffect(() => {
    setSelected(null)
  }, [report.generatedAt])

  return (
    <div className="roi">
      {/* --- 1. the scoring region ------------------------------------------ */}
      <div className="roi-headline">
        <p className="roi-headline__lead">
          The labelled patches have been joined into regions. You get two things: one
          smooth scoring region that every later step measures inside, and a border
          around each separate patch of tumour.
        </p>

        <div className="roi-headline__figures">
          <div className="roi-headline__figure roi-headline__figure--hero">
            <span className="roi-headline__value mono">{report.areaMm2.toFixed(1)} mm²</span>
            <span className="roi-headline__caption">the scoring region</span>
            <span className="roi-headline__sub">
              {(report.roiShare * 100).toFixed(1)}% of the labelled tissue &mdash;{' '}
              {report.regions.length} area{report.regions.length === 1 ? '' : 's'},{' '}
              {report.holes} hole{report.holes === 1 ? '' : 's'}
            </span>
          </div>

          {CLASS_INFO.map((entry) => {
            const found = report.classRegions[entry.key] ?? []
            const areaMm2 = report.classAreaMm2[entry.key] ?? 0
            return (
              <div className="roi-headline__figure" key={entry.key}>
                <span className="roi-headline__value mono">{areaMm2.toFixed(1)} mm²</span>
                <span className="roi-headline__caption">
                  <span className="roi-chip__dot" style={{ background: entry.colour }} />
                  {entry.label}
                </span>
                <span className="roi-headline__sub">
                  {formatCount(found.length)} region{found.length === 1 ? '' : 's'}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* --- 2. the borders, on the slide ----------------------------------- */}
      <figure className="roi-map">
        <div className="roi-map__panel">
          <div className="roi-map__frame">
            <img
              src={roiPanelUrl(report.uploadId, 'borders_on_slide')}
              alt="The region borders drawn on the slide, each one lightly tinted so the tissue inside stays visible"
            />
            <HitLayer report={report} selected={selected} onSelect={setSelected} />
          </div>
          <p className="roi-map__label">
            The regions on the slide, coloured by what they are
          </p>
        </div>

        <div className="roi-legend">
          <span className="roi-chip">
            <span className="roi-chip__dot" style={{ background: TISSUE_OUTLINE_COLOUR }} />
            Tissue
          </span>
          {CLASS_INFO.map((entry) => (
            <span className="roi-chip" key={entry.key}>
              <span className="roi-chip__dot" style={{ background: entry.colour }} />
              {entry.label}
            </span>
          ))}
        </div>
        <figcaption className="roi-map__caption">
          Patches of the same kind that touch each other are merged into one region. The
          tint is light enough to see the tissue through it. Click any region to see it
          enlarged below.
        </figcaption>
      </figure>

      {/* --- 2b. the selected region, enlarged --------------------------------- */}
      {selected && (
        <figure className="roi-selected">
          <div className="roi-selected__frame">
            <img
              key={`${selected.key}-${selected.index}`}
              src={roiRegionCropUrl(report.uploadId, selected.key, selected.index)}
              alt={`The selected ${selected.label.toLowerCase()} region, enlarged from the slide with its border drawn on it`}
            />
          </div>
          <figcaption className="roi-selected__caption">
            <span className="roi-selected__label">
              <span className="roi-chip__dot" style={{ background: selected.colour }} />
              {selected.label} region &middot; {selected.areaMm2.toFixed(2)} mm&sup2;
            </span>
            <span className="roi-selected__hint">
              Enlarged from the slide. Nearby region borders are drawn on it too.
            </span>
            <Button variant="ghost" onClick={() => setSelected(null)}>
              Close
            </Button>
          </figcaption>
        </figure>
      )}

      {/* --- 3. the top-3 crops ------------------------------------------------ */}
      <div className="roi-top">
        {(['invasive_epithelium', 'non_invasive_epithelium'] as const).map((key) => {
          const info = CLASS_INFO.find((entry) => entry.key === key)!
          const className = key === 'invasive_epithelium' ? 'invasive' : 'dcis'
          const found = report.classRegions[key] ?? []

          return (
            <div className="roi-top__column" key={key}>
              <h3 className="roi-top__title">
                <span className="roi-chip__dot" style={{ background: info.colour }} />
                Largest {info.label.toLowerCase()} regions
              </h3>
              {found.length === 0 ? (
                <p className="roi-top__empty">None found on this slide.</p>
              ) : (
                <div className="roi-top__gallery">
                  {found.slice(0, TOP_RANKS.length).map((region, position) => {
                    const rank = position + 1
                    return (
                      <figure className="roi-top__card" key={rank}>
                        <div className="roi-top__frame">
                          <img
                            src={roiTopUrl(report.uploadId, className, rank)}
                            alt={`The rank ${rank} largest ${info.label.toLowerCase()} region, cropped from the slide`}
                          />
                        </div>
                        <figcaption className="roi-top__caption">
                          #{rank} &middot; {region.areaMm2.toFixed(2)} mm²
                        </figcaption>
                      </figure>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* --- 4. the export ------------------------------------------------- */}
      <div className="roi-export">
        <p className="roi-export__text">
          Download every region above as one file that opens directly in QuPath.
        </p>
        <a
          className="btn btn--secondary btn--md roi-export__button"
          href={roiQupathUrl(report.uploadId)}
          download={`${report.uploadId}_qupath.geojson`}
        >
          Download QuPath annotations (.geojson)
        </a>
      </div>

      {/* --- notes ----------------------------------------------------------- */}
      <ul className="roi-notes">
        {report.notes.map((note) => (
          <li key={note.slice(0, 48)}>{note}</li>
        ))}
      </ul>

      {/* --- 5. how the scoring region was built, folded away ---------------- */}
      <details className="roi-detail">
        <summary>How the scoring region was built</summary>
        <div className="roi-detail__body">
          <p className="roi-detail__intro">
            The single scoring region is built differently from the borders above it: it
            is smoothed, gaps are closed, and tumour that is clearly inside a duct is
            cut back out. These five pictures show each move.
          </p>

          <div className="roi-detail__panels">
            {(['seed', 'smoothed', 'binary', 'region', 'outline'] as const).map((name) => (
              <figure className="roi-detail__panel" key={name}>
                <img src={roiPanelUrl(report.uploadId, name)} alt={name} />
                <figcaption>{name}</figcaption>
              </figure>
            ))}
          </div>

          <dl className="roi-detail__ledger">
            <div>
              <dt>Starting area</dt>
              <dd>
                {ledger.seedMm2.toFixed(2)} mm² over {ledger.seedComponents} patch
                {ledger.seedComponents === 1 ? '' : 'es'}, before any tidying
              </dd>
            </div>
            <div>
              <dt>Added by closing gaps</dt>
              <dd>{ledger.mergedMm2.toFixed(2)} mm² added</dd>
            </div>
            <div>
              <dt>Cut back out (tumour inside a duct)</dt>
              <dd>
                {ledger.protectedMm2.toFixed(2)} mm² ({formatCount(ledger.protectedCells)}{' '}
                patches)
              </dd>
            </div>
            <div>
              <dt>Dropped as too small</dt>
              <dd>
                {ledger.droppedMm2.toFixed(2)} mm² over {ledger.droppedComponents}{' '}
                piece{ledger.droppedComponents === 1 ? '' : 's'}
              </dd>
            </div>
          </dl>

          <p className="roi-detail__foot">
            sigma {params.sigma} cells, threshold {params.threshold}, closing radius{' '}
            {params.closeCells} cells, protect in-situ at {params.protectInSitu}, minimum
            area {params.minAreaMm2} mm²
            {params.keepLargest !== null && `, keeping the largest ${params.keepLargest}`}
            .
          </p>

          <Button variant="ghost" onClick={() => void onRebuild()} disabled={rebuilding}>
            {rebuilding ? 'Rebuilding…' : 'Rebuild with the server defaults'}
          </Button>
        </div>
      </details>

      <p className="roi-citation">
        <Badge tone="neutral">Primer items 62-63, 70-72</Badge> Everything after this
        step is measured inside the region above and nowhere else.
      </p>
    </div>
  )
}
