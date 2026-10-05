/**
 * Home: start a run, or look after the runs already made.
 *
 * This page used to be a brochure for the pipeline - a grid of all seventeen
 * steps and a row of counters about them. Both are things you read once. What
 * you come back to this screen for is the two jobs it now does and nothing
 * else: put a case through the pipeline, and see what has already been through
 * it. The step catalogue still exists and is still on screen, one step at a
 * time, inside the walkthrough where it is actually being executed.
 */

import { useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import { Button } from '@/components/ui/Button'
import { Spinner } from '@/components/ui/Spinner'
import { PreviousCases } from '@/features/history/PreviousCases'
import { useHistory } from '@/features/history/useHistory'
import { relativeTime } from '@/lib/format'
import type { HistoryCase, HistoryMarker } from '@/types/history'

import './home.css'

/** The total the walkthrough counts to, for "step 11 of 19". */
const TOTAL_STEPS = 19

/** How many unfinished runs get their own card before the list takes over. */
const MAX_RESUME_CARDS = 3

/** A marker that stopped part-way, with the case it belongs to. */
interface UnfinishedRun {
  entry: HistoryCase
  marker: HistoryMarker
}

function count(n: number, singular: string, plural = `${singular}s`): string {
  return `${n} ${n === 1 ? singular : plural}`
}

function ResumeCard({
  run,
  busy,
  onResume,
}: {
  run: UnfinishedRun
  busy: boolean
  onResume: () => void
}) {
  const { entry, marker } = run
  const done = Math.min(marker.lastStep, TOTAL_STEPS)
  const touched = relativeTime(marker.updatedAt)

  return (
    <article className="resume-card">
      <header className="resume-card__head">
        <span className="resume-card__letter mono">{marker.marker}</span>
        <span className="resume-card__names">
          <strong>{marker.name}</strong>
          <span className="resume-card__case mono">{entry.caseId}</span>
        </span>
      </header>

      <div className="resume-card__progress">
        <div className="resume-card__bar" aria-hidden>
          <span style={{ width: `${(done / TOTAL_STEPS) * 100}%` }} />
        </div>
        <span className="resume-card__step">
          Stopped after step {done} of {TOTAL_STEPS}
        </span>
      </div>

      <footer className="resume-card__foot">
        <span className="resume-card__when">{touched ?? 'not dated'}</span>
        <Button
          onClick={onResume}
          loading={busy}
          disabled={!entry.casePath}
          title={
            entry.casePath
              ? 'Continue from the last finished step'
              : 'The case folder for this run was not found'
          }
        >
          Resume →
        </Button>
      </footer>
    </article>
  )
}

export function HomePage() {
  const history = useHistory()
  const navigate = useNavigate()

  /**
   * Bring a stored run back and walk it.
   *
   * The rename happens here, before navigating, so the walkthrough opens onto a
   * case whose files are already in place. Navigating first and restoring after
   * would give the viewer a second or two of a screen reporting that nothing has
   * been run - which is the one thing this list exists to contradict.
   */
  const openRun = async (
    caseId: string,
    casePath: string,
    marker: string,
    replay: boolean,
  ) => {
    try {
      await history.open(caseId, marker)
    } catch {
      // `useHistory` has already put the reason on screen. Going anyway would
      // land on a walkthrough with no files behind it.
      return
    }
    navigate('/demo', { state: { caseId, casePath, marker, replay } })
  }

  const { unfinished, scored } = useMemo(() => {
    const partial: UnfinishedRun[] = []
    let complete = 0

    for (const entry of history.cases) {
      for (const marker of entry.markers) {
        if (marker.state === 'partial') partial.push({ entry, marker })
        if (marker.state === 'complete') complete += 1
      }
    }

    // Most recently touched first: the run abandoned ten minutes ago is the one
    // somebody came back for, not the one from three weeks ago.
    partial.sort((a, b) =>
      (b.marker.updatedAt ?? '').localeCompare(a.marker.updatedAt ?? ''),
    )

    return { unfinished: partial, scored: complete }
  }, [history.cases])

  const known = history.cases.length > 0

  return (
    <>
      <section className="container hero">
        {/* The panel, not one marker of it. This used to name CD44, which is one
            of five antibodies this pipeline scores and not the one a given case is
            loaded with. */}
        <span className="hero__pill">breast cancer · five markers</span>

        <h1 className="hero__title">
          Score a case, <em>end to end</em>
        </h1>

        <p className="hero__lede">
          Choose a case folder and one of the five markers. All {TOTAL_STEPS} steps then run
          on screen against the real slides: an H&amp;E slide to find the tumour on, and one
          marker slide to measure. Finished runs are saved below and reopen instantly.
        </p>

        <div className="hero__actions">
          <Link to="/demo">
            <Button size="lg" attention>
              Start a new run →
            </Button>
          </Link>

          <a href="#previous-cases">
            <Button size="lg" variant="secondary">
              Previous cases
              {known && ` (${history.cases.length})`}
            </Button>
          </a>
        </div>

        <p className="hero__summary">
          {history.loading && history.cases.length === 0 ? (
            <>
              <Spinner /> loading saved cases…
            </>
          ) : history.error && !known ? (
            // Not the same as "nothing has been run". The list below says what
            // went wrong; this line only has to stop claiming the disk is empty.
            'Could not read the saved cases — see below.'
          ) : known ? (
            <>
              {count(history.cases.length, 'case')} on this machine ·{' '}
              {count(scored, 'marker')} scored ·{' '}
              {unfinished.length === 0
                ? 'none left part-way'
                : `${count(unfinished.length, 'run')} unfinished`}
            </>
          ) : (
            'No cases yet — start a run and they appear here.'
          )}
        </p>
      </section>

      {unfinished.length > 0 && (
        <section className="container section" id="in-progress">
          <div className="section__head">
            <h2 className="section__title">Pick up where you left off</h2>
            <p className="section__lede">
              These runs stopped part-way. Resuming reopens the case and continues from the
              last finished step. Steps already done are loaded from disk, not run again.
            </p>
          </div>

          <div className="resume-grid">
            {unfinished.slice(0, MAX_RESUME_CARDS).map((run) => (
              <ResumeCard
                key={`${run.entry.caseId}:${run.marker.marker}`}
                run={run}
                busy={history.busy === `${run.entry.caseId}:${run.marker.marker}`}
                onResume={() =>
                  void openRun(
                    run.entry.caseId,
                    run.entry.casePath,
                    run.marker.marker,
                    false,
                  )
                }
              />
            ))}
          </div>

          {unfinished.length > MAX_RESUME_CARDS && (
            <p className="resume-more">
              and {count(unfinished.length - MAX_RESUME_CARDS, 'more unfinished run')} in the
              list below.
            </p>
          )}
        </section>
      )}

      <PreviousCases history={history} onOpen={(c, p, m, r) => void openRun(c, p, m, r)} />
    </>
  )
}
