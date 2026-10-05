/**
 * Step 10's screen: every candidate invasive region, as something to choose between.
 *
 *   1. what this choice commits to, in one sentence, and the running total
 *   2. the controls — select all, deselect all, back to the pipeline's own answer
 *   3. the grid: one card per candidate, each the tissue with the model's square
 *      boundary on it, tickable
 *   4. what was not offered, and the technical detail, folded away
 *
 * **The card is the tissue, not the class map.** The judgement being asked for is
 * "is this worth segmenting per pixel", and a coloured tile map cannot be judged — a
 * person needs to see ducts. So each card is the slide's own pixels with step 8's
 * boundary drawn over them, and the boundary is drawn as the staircase it is, because
 * the whole point of the next step is that the staircase is not the tumour's shape.
 *
 * **The whole card is the checkbox.** A 12-pixel box beside a picture is a target that
 * has to be aimed at twenty times; the card is the thing being decided about, so the
 * card is the control. The tick is still drawn, because it is what says which state a
 * card is in at a glance across a grid.
 */

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { roiCandidateCardUrl } from '@/api/roiSelection'
import type { RoiCandidate, RoiSelectionReport } from '@/types/roiSelection'

import './roiSelection.css'

interface RoiSelectionViewProps {
  report: RoiSelectionReport
  selected: string[]
  saving: boolean
  onToggle: (roiId: string) => void
  onSelectAll: () => void
  onDeselectAll: () => void
  onResetToDefault: () => void
}

export function RoiSelectionView({
  report,
  selected,
  saving,
  onToggle,
  onSelectAll,
  onDeselectAll,
  onResetToDefault,
}: RoiSelectionViewProps) {
  const ticked = new Set(selected)
  const { candidates } = report

  const share =
    report.invasiveMm2 > 0 ? report.selectedMm2 / report.invasiveMm2 : 0
  const atDefault =
    selected.length === report.defaultSelection.length &&
    report.defaultSelection.every((one) => ticked.has(one))

  if (candidates.length === 0) {
    return (
      <div className="placeholder placeholder--blocked">
        <span className="placeholder__badge mono">nothing to choose</span>
        <p className="placeholder__text">
          No invasive tumour large enough to outline was found on this slide. Either
          there is none, or every patch found is too small to draw a boundary inside.
        </p>
      </div>
    )
  }

  return (
    <div className="selection">
      <header className="selection__intro">
        <p className="selection__lede">
          <strong>{candidates.length}</strong> separate areas were marked as invasive
          tumour. These are rough squares: they say roughly <em>where</em>, not exactly
          what <em>shape</em>. The next step draws the real outline inside the areas you
          tick here, and it is slow &mdash; so tick the ones that matter.
        </p>
      </header>

      <div className="selection__bar">
        <div className="selection__count">
          <span className="selection__countNumber">
            {selected.length} / {candidates.length}
          </span>
          <span className="selection__countLabel">areas chosen</span>
        </div>

        <dl className="selection__totals">
          <div>
            <dt>Tumour chosen</dt>
            <dd>
              {report.selectedMm2.toFixed(2)} mm²{' '}
              <span className="selection__muted">
                of {report.invasiveMm2.toFixed(2)} mm² ({(share * 100).toFixed(0)}%)
              </span>
            </dd>
          </div>
          <div>
            <dt>Work this will take</dt>
            <dd>
              {report.selectedWindows.toLocaleString()}{' '}
              <span className="selection__muted">patches through the model</span>
            </dd>
          </div>
        </dl>

        <div className="selection__actions">
          <Button variant="secondary" onClick={onSelectAll} disabled={saving}>
            Select all
          </Button>
          <Button variant="secondary" onClick={onDeselectAll} disabled={saving}>
            Deselect all
          </Button>
          {!atDefault && (
            <Button variant="ghost" onClick={onResetToDefault} disabled={saving}>
              Back to suggested
            </Button>
          )}
        </div>
      </div>

      {!report.chosenByPerson && (
        <p className="selection__hint">
          Nothing has been chosen yet, so these are our suggestions: the largest areas,
          enough to cover almost all the tumour. Change whatever you like &mdash; what is
          ticked when you continue is what gets outlined.
        </p>
      )}

      <ul className="selection__grid">
        {candidates.map((candidate) => (
          <CandidateCard
            key={candidate.roiId}
            uploadId={report.uploadId}
            candidate={candidate}
            ticked={ticked.has(candidate.roiId)}
            disabled={saving}
            onToggle={onToggle}
          />
        ))}
      </ul>

      {(report.droppedSmall > 0 || report.droppedCapped > 0) && (
        <p className="selection__hint selection__hint--muted">
          {report.droppedSmall > 0 && (
            <>
              {report.droppedSmall} area(s) were too small to outline and are not listed
              ({report.droppedSmallMm2.toFixed(2)} mm²).{' '}
            </>
          )}
          {report.droppedCapped > 0 && (
            <>
              {report.droppedCapped} more were left off to keep the list short (
              {report.droppedCappedMm2.toFixed(2)} mm²).
            </>
          )}
        </p>
      )}

      <details className="selection__details">
        <summary>What the numbers on each card mean</summary>
        <div className="selection__detailsBody">
          <p>
            Each card is a group of touching squares that the model called invasive
            tumour, ranked by size &mdash; <code>ROI-001</code> is the largest on the
            slide. The numbers are ranks within one run, so labelling the tissue again
            re-assigns them. That is why a selection made against an older run is
            dropped rather than reused.
          </p>
          <p>
            <strong>Confidence</strong> is how sure the model was. It describes the
            model&rsquo;s call, not the tissue &mdash; an area can be 0.97 confident and
            still be a rim of glass &mdash; and nothing is decided on it.
          </p>
          <p>
            <strong>Patches</strong> is how much work the next step will spend on that
            area. It is the real cost, measured against the grid that step will run.
          </p>
          {report.notes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </div>
      </details>
    </div>
  )
}

interface CandidateCardProps {
  uploadId: string
  candidate: RoiCandidate
  ticked: boolean
  disabled: boolean
  onToggle: (roiId: string) => void
}

function CandidateCard({
  uploadId,
  candidate,
  ticked,
  disabled,
  onToggle,
}: CandidateCardProps) {
  return (
    <li className="candidate">
      <button
        type="button"
        role="checkbox"
        aria-checked={ticked}
        aria-label={`${candidate.roiId}, ${candidate.areaMm2.toFixed(2)} square millimetres`}
        className={`candidate__button${ticked ? ' candidate__button--on' : ''}`}
        disabled={disabled}
        onClick={() => onToggle(candidate.roiId)}
      >
        <span className="candidate__head">
          <span className="candidate__tick" aria-hidden>
            {ticked ? '✓' : ''}
          </span>
          <span className="candidate__id mono">{candidate.roiId}</span>
          <Badge tone={ticked ? 'accent' : 'neutral'}>{candidate.areaMm2.toFixed(2)} mm²</Badge>
        </span>

        <img
          className="candidate__image"
          src={roiCandidateCardUrl(uploadId, candidate.roiId)}
          alt={`Tissue around ${candidate.roiId}, with the rough outline drawn on it`}
          loading="lazy"
        />

        <span className="candidate__facts">
          <span>
            <span className="candidate__factLabel">Confidence</span>
            <span className="candidate__factValue mono">
              {candidate.confidence.toFixed(2)}
            </span>
          </span>
          <span>
            <span className="candidate__factLabel">Squares</span>
            <span className="candidate__factValue mono">{candidate.cells}</span>
          </span>
          <span>
            <span className="candidate__factLabel">Patches</span>
            <span className="candidate__factValue mono">
              {candidate.windows.toLocaleString()}
            </span>
          </span>
        </span>
      </button>
    </li>
  )
}
