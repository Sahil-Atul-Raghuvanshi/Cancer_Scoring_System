/**
 * Previous cases: what has been scored on this machine, and a way back in.
 *
 * A case is a patient's block cut into six sections - one H&E to find the
 * tumour on and five stained slides to measure - so the unit a person looks for
 * is the case, and the thing they want to know about it is which of the five
 * antibodies have been done. That is the whole screen: a thumbnail of the H&E to
 * recognise the case by, and five rows saying done, part-way or not started.
 *
 * **Replaying costs nothing and the screen says so.** A finished marker has
 * every step's output on disk, so opening one walks the same seventeen screens
 * without running anything - it is a rename, then a read. That is worth saying
 * out loud, because the alternative reading of "open" is "start this again",
 * which on step 8 is half an hour.
 *
 * **Delete is the only irreversible thing here**, so it is the only control that
 * asks twice, and it says what it will cost to undo - the run, not the slide.
 */

import { useState } from 'react'

import { historyThumbnailUrl } from '@/api/history'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import type { HistoryCase, HistoryMarker } from '@/types/history'

import type { HistoryStateValue } from './useHistory'

import './history.css'

interface PreviousCasesProps {
  history: HistoryStateValue
  /** Open this marker and walk it. `replay` is true when nothing needs running. */
  onOpen: (caseId: string, casePath: string, marker: string, replay: boolean) => void
}

/** The total the walkthrough counts to, for "step 11 of 19". */
const TOTAL_STEPS = 19

function MarkerRow({
  entry,
  casePath,
  busy,
  onOpen,
  onArchive,
  onDelete,
}: {
  entry: HistoryMarker
  casePath: string
  busy: boolean
  onOpen: (replay: boolean) => void
  onArchive: () => void
  onDelete: () => void
}) {
  const [confirming, setConfirming] = useState(false)

  const complete = entry.state === 'complete'
  const partial = entry.state === 'partial'

  return (
    <li className={`hist-marker hist-marker--${entry.state}`}>
      <span className="hist-marker__letter mono">{entry.marker}</span>

      <span className="hist-marker__name">
        {entry.name}
        {entry.location === 'demo' && entry.state !== 'not-started' && (
          <span className="hist-marker__open" title="already open in the working area">
            open
          </span>
        )}
      </span>

      <span className="hist-marker__status">
        {complete ? (
          <Badge tone="success">Scored</Badge>
        ) : partial ? (
          <Badge tone="warn">Step {entry.lastStep} of {TOTAL_STEPS}</Badge>
        ) : (
          <Badge tone="neutral">Not started</Badge>
        )}
      </span>

      <span className="hist-marker__score mono">
        {entry.score?.percent != null ? (
          <>
            {entry.score.percent}%
            {entry.score.intensity != null && (
              <span className="hist-marker__intensity">
                {' '}
                · {entry.score.intensity} {entry.score.intensityLabel}
              </span>
            )}
          </>
        ) : (
          <span className="hist-marker__dash">—</span>
        )}
      </span>

      <span className="hist-marker__actions">
        {busy ? (
          <Spinner />
        ) : confirming ? (
          <>
            <span className="hist-marker__confirm">
              Delete this run? The slide stays; the scoring would have to be done again.
            </span>
            <Button
              variant="secondary"
              onClick={() => {
                setConfirming(false)
                onDelete()
              }}
            >
              Delete
            </Button>
            <Button variant="ghost" onClick={() => setConfirming(false)}>
              Keep
            </Button>
          </>
        ) : (
          <>
            <Button
              variant={complete ? 'primary' : 'secondary'}
              onClick={() => onOpen(complete)}
              disabled={!casePath}
            >
              {complete ? 'Replay' : partial ? 'Resume' : 'Score it'}
            </Button>
            {entry.location === 'demo' && entry.state !== 'not-started' && (
              <Button
                variant="ghost"
                onClick={onArchive}
                title="Move this run out of the working area. You can still reopen it."
              >
                File away
              </Button>
            )}
            {entry.state !== 'not-started' && (
              <Button variant="ghost" onClick={() => setConfirming(true)}>
                Delete
              </Button>
            )}
          </>
        )}
      </span>
    </li>
  )
}

function CaseCard({
  entry,
  history,
  onOpen,
}: {
  entry: HistoryCase
  history: HistoryStateValue
  onOpen: PreviousCasesProps['onOpen']
}) {
  const [expanded, setExpanded] = useState(entry.started > 0)

  return (
    <article className={expanded ? 'hist-case is-open' : 'hist-case'}>
      <button
        type="button"
        className="hist-case__head"
        onClick={() => setExpanded((open) => !open)}
        aria-expanded={expanded}
      >
        <span className="hist-case__thumb">
          {entry.hasThumbnail ? (
            <img
              src={historyThumbnailUrl(entry.caseId)}
              alt={`The H&E slide of case ${entry.caseId}`}
              loading="lazy"
            />
          ) : (
            <span className="hist-case__thumb-empty" aria-hidden />
          )}
        </span>

        <span className="hist-case__id">
          <strong>{entry.caseId}</strong>
          <span className="hist-case__path mono">{entry.casePath || 'folder not found'}</span>
        </span>

        <span className="hist-case__count">
          <strong>{entry.completed}</strong> of {entry.markers.length} scored
        </span>

        <span className="hist-case__chev" aria-hidden>
          {expanded ? '▾' : '▸'}
        </span>
      </button>

      {expanded && (
        <ul className="hist-markers">
          {entry.markers.map((marker) => (
            <MarkerRow
              key={marker.marker}
              entry={marker}
              casePath={entry.casePath}
              busy={history.busy === `${entry.caseId}:${marker.marker}`}
              onOpen={(replay) =>
                onOpen(entry.caseId, entry.casePath, marker.marker, replay)
              }
              onArchive={() => void history.archive(entry.caseId, marker.marker)}
              onDelete={() => void history.remove(entry.caseId, marker.marker)}
            />
          ))}
        </ul>
      )}
    </article>
  )
}

export function PreviousCases({ history, onOpen }: PreviousCasesProps) {
  const { cases, loading, error } = history

  return (
    <section className="container section" id="previous-cases">
      <div className="section__head">
        <h2 className="section__title">Previous cases</h2>
        <p className="section__lede">
          Every case with slides on this machine, and which of its five markers have been
          scored. Open a case to see each marker, its score, and what you can do next:
          reopen a finished one, resume one that stopped part-way, or start a new one.
        </p>
      </div>

      {loading && cases.length === 0 ? (
        <div className="hist-empty">
          <Spinner />
          <p>Looking for saved runs…</p>
        </div>
      ) : error ? (
        <div className="hist-empty hist-empty--error">
          <p>{error}</p>
          <Button variant="secondary" onClick={() => void history.reload()}>
            Try again
          </Button>
        </div>
      ) : cases.length === 0 ? (
        <div className="hist-empty">
          <p>
            Nothing has been scored yet and no case folders were found. Start a run and the
            case will appear here.
          </p>
        </div>
      ) : (
        <div className="hist-cases">
          {cases.map((entry) => (
            <CaseCard
              key={entry.caseId}
              entry={entry}
              history={history}
              onOpen={onOpen}
            />
          ))}
        </div>
      )}
    </section>
  )
}
