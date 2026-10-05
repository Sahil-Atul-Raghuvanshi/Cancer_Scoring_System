import { useEffect, useState } from 'react'

import {
  createDataVersion,
  fetchDataVersions,
  selectDataVersion,
  waitForDataVersion,
  type DataVersions,
} from '@/api/dataVersions'

const NEW_VERSION = '__new__'

/** Set once the viewer has answered the opening question, for this browser tab. */
const CHOSEN_KEY = 'data-version-chosen'

function hasChosen(): boolean {
  try {
    return window.sessionStorage.getItem(CHOSEN_KEY) === '1'
  } catch {
    return false
  }
}

function rememberChosen() {
  try {
    window.sessionStorage.setItem(CHOSEN_KEY, '1')
  } catch {
    /* storage blocked — the question will simply be asked again */
  }
}

/**
 * Which set of results the app is showing — v1, v2, … — and a way to switch.
 *
 * Every version keeps its own results, history and models; the slides themselves are
 * shared. When more than one version exists, the app asks which to show as it opens.
 * Switching restarts the server, so anything running is stopped first, and the page
 * reloads onto the new version once the server is back.
 */
export function DataVersionPicker() {
  const [info, setInfo] = useState<DataVersions | null>(null)
  const [asking, setAsking] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [noticeOpen, setNoticeOpen] = useState(true)

  useEffect(() => {
    fetchDataVersions()
      .then((loaded) => {
        setInfo(loaded)
        const choices = loaded.versions.filter((entry) => entry.openable)
        setAsking(choices.length > 1 && !loaded.pinnedByEnv && !hasChosen())
      })
      .catch(() => setInfo(null))
  }, [])

  if (!info) return null

  async function switchTo(version: string, confirmFirst = true) {
    if (
      confirmFirst &&
      !window.confirm(
        `Switch to ${version}?\n\n` +
          'The server restarts to load that version’s results and history. ' +
          'Anything still running will be stopped. The slides are shared and are not affected.',
      )
    )
      return
    rememberChosen()
    setAsking(false)
    setError(null)
    setBusy(`Switching to ${version}…`)
    try {
      const result = await selectDataVersion(version)
      if (result.restarting && !(await waitForDataVersion(version))) {
        setBusy(null)
        setError(
          `${version} is selected, but the server did not restart on its own. ` +
            'Run stop.bat, then start.bat, to finish switching.',
        )
        return
      }
      window.location.reload()
    } catch (exc) {
      setBusy(null)
      setError(exc instanceof Error ? exc.message : String(exc))
    }
  }

  async function startNew() {
    const ok = window.confirm(
      'Start a new, empty version?\n\n' +
        'It begins with no results or history, uses the latest models, and shares the same ' +
        'slides. The current version is kept exactly as it is.',
    )
    if (!ok) return
    try {
      const created = await createDataVersion()
      await switchTo(created.version, false)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    }
  }

  function choose(version: string) {
    if (version === info?.active) {
      rememberChosen()
      setAsking(false)
    } else {
      void switchTo(version, false)
    }
  }

  const title = info.pinnedByEnv
    ? `Fixed to ${info.active} for this run (CSS_DATA_VERSION is set)`
    : 'Each version keeps its own results, history and models. The slides are shared.'

  return (
    <>
      <div className="version-picker" title={title}>
        <label className="version-picker__label" htmlFor="data-version">
          Version
        </label>
        <select
          id="data-version"
          className="version-picker__select"
          value={info.active}
          disabled={info.pinnedByEnv || busy !== null}
          onChange={(event) => {
            const value = event.target.value
            if (value === NEW_VERSION) void startNew()
            else if (value !== info.active) void switchTo(value)
          }}
        >
          {info.versions.map((entry) => (
            <option
              key={entry.version}
              value={entry.version}
              disabled={!entry.openable}
              title={entry.notOpenableReason ?? undefined}
            >
              {entry.version}
              {entry.version === info.latest ? ' (latest)' : ''}
              {entry.openable ? '' : ' — cannot be opened'}
            </option>
          ))}
          <option value={NEW_VERSION}>+ Start a new version</option>
        </select>
        {busy && <span className="version-picker__status">{busy}</span>}
        {error && (
          <span className="version-picker__status version-picker__status--error" role="alert">
            {error}
          </span>
        )}
      </div>

      {info.fallbackReason && noticeOpen && !asking && (
        <div className="version-notice" role="status">
          <p className="version-notice__text">
            <strong>{info.requested} could not be opened.</strong> {info.fallbackReason}
          </p>
          <button
            type="button"
            className="version-notice__close"
            onClick={() => setNoticeOpen(false)}
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      )}

      {asking && (
        <div className="version-chooser" role="dialog" aria-modal="true" aria-labelledby="version-chooser-title">
          <div className="version-chooser__panel">
            <h2 id="version-chooser-title" className="version-chooser__title">
              Which version do you want to see?
            </h2>
            <p className="version-chooser__lead">
              Each version is a separate set of results from a different version of the
              pipeline. The slides are the same in all of them. You can switch later from
              the Version menu at the top right.
            </p>
            <ul className="version-chooser__list">
              {[...info.versions].reverse().map((entry) => (
                <li key={entry.version}>
                  <button
                    type="button"
                    className="version-chooser__option"
                    aria-current={entry.version === info.active ? 'true' : undefined}
                    disabled={!entry.openable}
                    onClick={() => choose(entry.version)}
                  >
                    <span className="version-chooser__name">
                      {entry.version}
                      {entry.version === info.latest && (
                        <span className="version-chooser__tag">latest</span>
                      )}
                      {entry.version === info.active && (
                        <span className="version-chooser__tag">open now</span>
                      )}
                    </span>
                    {entry.openable ? (
                      entry.summary && (
                        <span className="version-chooser__summary">{entry.summary}</span>
                      )
                    ) : (
                      <span className="version-chooser__summary">
                        Made by an older version of the app, so it cannot be opened here.
                        Its results are kept on disk.
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </>
  )
}
