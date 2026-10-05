import { useCallback, useState } from 'react'

import { Button } from '@/components/ui/Button'
import { useSlideSession } from '@/features/upload/slideSessionContext'
import type { CaseResolution } from '@/types/case'

import { MarkerCard } from './MarkerCard'
import { useMarkerPanel } from './useMarkerPanel'

import './panel.css'

/**
 * Step 0: choose a biomarker, then point at the case folder that holds its
 * IHC slide and the case's H&E slide.
 *
 * Shown wherever the upload drop-zone used to sit - before a slide is loaded,
 * for every step, same as the drop-zone was. Resolving a folder is pure file
 * discovery (no bytes move yet); loading it registers the H&E slide and the
 * chosen marker's slide as the two uploads steps 1 onward already know how to
 * work with (see `app/services/case_service.py`).
 */
export function CaseLoaderPanel() {
  const { markers, error: panelError } = useMarkerPanel()
  const { resolveCase, loadCase } = useSlideSession()

  const [letter, setLetter] = useState<string | null>(null)
  const [casePath, setCasePath] = useState('')
  const [resolution, setResolution] = useState<CaseResolution | null>(null)
  const [resolving, setResolving] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleResolve = useCallback(async () => {
    if (!casePath.trim()) return
    setResolving(true)
    setError(null)
    setResolution(null)
    try {
      setResolution(await resolveCase(casePath.trim()))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'could not read that folder')
    } finally {
      setResolving(false)
    }
  }, [casePath, resolveCase])

  const handleLoad = useCallback(async () => {
    if (!letter || !resolution) return
    setLoading(true)
    setError(null)
    try {
      await loadCase(resolution.casePath, letter)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'could not load the case')
    } finally {
      setLoading(false)
    }
  }, [letter, loadCase, resolution])

  const scoredMarkers = markers?.filter((marker) => marker.scored) ?? []
  const canLoad =
    !!letter && !!resolution && !resolution.missing.includes('HE') && !resolution.missing.includes(letter)

  return (
    <div className="case-loader">
      <div className="case-loader__section">
        <h3 className="case-loader__heading">1. Choose a marker</h3>
        {panelError ? (
          <p className="case-loader__error">{panelError}</p>
        ) : (
          <div className="marker-grid">
            {scoredMarkers.map((marker) => (
              <MarkerCard
                key={marker.letter}
                marker={marker}
                selected={letter === marker.letter}
                available={resolution ? !resolution.missing.includes(marker.letter) : null}
                onSelect={() => setLetter(marker.letter)}
              />
            ))}
          </div>
        )}
      </div>

      <div className="case-loader__section">
        <h3 className="case-loader__heading">2. Choose the case folder</h3>
        <p className="case-loader__body">
          The folder that holds this case's slides. For example:{' '}
          <span className="mono">data/original/oncostem_slides/CAN_00270</span>
        </p>
        <div className="case-loader__row">
          <input
            className="case-loader__input mono"
            type="text"
            value={casePath}
            onChange={(event) => {
              setCasePath(event.target.value)
              setResolution(null)
            }}
            placeholder="C:\...\oncostem_slides\CAN_00270"
          />
          <Button
            variant="secondary"
            onClick={() => void handleResolve()}
            loading={resolving}
            disabled={!casePath.trim()}
          >
            Find slides
          </Button>
        </div>

        {resolution && (
          <div className="case-loader__resolution">
            <span className="case-loader__case-id mono">{resolution.caseId}</span>
            {resolution.missing.length === 0 ? (
              <span className="case-loader__complete">all six slides found</span>
            ) : (
              <span className="case-loader__partial">
                not found: {resolution.missing.join(', ')}
              </span>
            )}
          </div>
        )}
      </div>

      {error && <p className="case-loader__error">{error}</p>}

      <Button onClick={() => void handleLoad()} loading={loading} disabled={!canLoad} attention={canLoad}>
        Load case
      </Button>
    </div>
  )
}
