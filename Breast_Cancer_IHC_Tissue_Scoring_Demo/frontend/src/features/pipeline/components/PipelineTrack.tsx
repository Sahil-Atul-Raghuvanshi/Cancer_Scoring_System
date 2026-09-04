import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { cn } from '@/lib/cn'
import { padIndex } from '@/lib/format'

import './track.css'

/**
 * The three states a pipeline stage can be in, plus a failure state.
 * `active` is the one stage the pipeline is sitting on; everything behind it
 * is `complete`, everything ahead of it is `pending`.
 */
export type TrackStatus = 'pending' | 'active' | 'complete' | 'error'

/** The least a stage has to carry to be drawn on the track. */
export interface TrackItem {
  id: string
  index: number
  title: string
}

interface PipelineTrackProps {
  items: TrackItem[]
  /** Index into `items` of the stage the pipeline is on. */
  activeIndex: number
  statusOf: (index: number) => TrackStatus
  /**
   * How many stages the window shows at once. `'auto'` — the default — measures
   * the space the track has been given and shows as many whole cards as fit,
   * so a tall screen reveals more of the pipeline than a short one does.
   */
  visibleCount?: number | 'auto'
  /** Floor for the auto-fitted window, for screens too short to fit even that. */
  minVisible?: number
  /** Whether the active stage is doing real work this instant. */
  busy?: boolean
  /** Override the status line under a title. */
  labelOf?: (index: number, status: TrackStatus) => string
  onSelect?: (index: number) => void
  canSelect?: (index: number) => boolean
  label?: string
}

const DEFAULT_LABELS: Record<TrackStatus, string> = {
  pending: 'Pending',
  active: 'In progress',
  complete: 'Completed',
  error: 'Failed',
}

function CheckIcon() {
  return (
    <svg className="track__icon" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="9.2" stroke="currentColor" strokeWidth="1.5" opacity="0.5" />
      <path
        d="m8 12.3 2.7 2.7L16.2 9.5"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function AlertIcon() {
  return (
    <svg className="track__icon" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="9.2" stroke="currentColor" strokeWidth="1.5" opacity="0.5" />
      <path
        d="M12 7.6v5.2M12 16.1v.5"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
      />
    </svg>
  )
}

/** A dashed ring: it turns slowly while the stage waits, quickly while it works. */
function RunningIcon({ busy }: { busy: boolean }) {
  return (
    <svg
      className={cn('track__icon', 'track__icon--running', busy && 'track__icon--busy')}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <circle
        cx="12"
        cy="12"
        r="9.2"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeDasharray="4.6 5.2"
      />
    </svg>
  )
}

function PendingIcon() {
  return (
    <svg className="track__icon" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="8.4" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

function ChevronIcon({ up }: { up: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
      style={up ? undefined : { transform: 'rotate(180deg)' }}
    >
      <path
        d="m6 14.5 6-6 6 6"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/**
 * A vertical pipeline: stage cards joined by arrows, with a sliding window
 * over them so a long pipeline still fits beside its own content.
 *
 * The window and the pipeline are separate things. Execution moves the active
 * stage; the window only decides what is on screen. It follows the active
 * stage when that stage moves out of view, and the two nav buttons move it by
 * hand without touching the run at all.
 */
export function PipelineTrack({
  items,
  activeIndex,
  statusOf,
  visibleCount = 'auto',
  minVisible = 1,
  busy = false,
  labelOf,
  onSelect,
  canSelect,
  label = 'Pipeline stages',
}: PipelineTrackProps) {
  const [windowStart, setWindowStart] = useState(0)
  const listRef = useRef<HTMLOListElement>(null)
  const spaceRef = useRef<HTMLDivElement>(null)
  const [offsets, setOffsets] = useState<number[]>([])
  const [listHeight, setListHeight] = useState(0)
  const [space, setSpace] = useState(0)

  const statuses = items.map((_, index) => statusOf(index))
  const statusKey = statuses.join('|')
  const completedCount = statuses.filter((status) => status === 'complete').length

  /* --- measurement: titles may wrap, so the window is measured, not assumed */

  useLayoutEffect(() => {
    const list = listRef.current
    if (!list) return

    const measure = () => {
      const rows = Array.from(list.children) as HTMLElement[]
      setOffsets(rows.map((row) => row.offsetTop))
      setListHeight(list.scrollHeight)
    }

    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(list)
    return () => observer.disconnect()
  }, [items])

  // The space the layout hands the track. It is flex-sized, never content-sized,
  // so reading it here cannot feed back into the window height set below.
  useLayoutEffect(() => {
    const el = spaceRef.current
    if (!el) return

    const measure = () => setSpace(el.clientHeight)
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  /* --- how many cards fit, from the two things actually measured ----------- */

  // Row pitch is card + connector + gap; the last row carries no connector, so
  // n cards need (n - 1) pitches plus one bare card.
  const first = offsets[0]
  const second = offsets[1]
  const last = offsets[offsets.length - 1]

  const pitch = first !== undefined && second !== undefined ? second - first : 0
  const cardHeight = last !== undefined ? listHeight - last : 0

  const fitted =
    pitch > 0 && cardHeight > 0 && space > 0
      ? Math.floor((space - cardHeight) / pitch) + 1
      : minVisible

  const visible =
    visibleCount === 'auto'
      ? Math.max(minVisible, Math.min(items.length, fitted))
      : Math.min(visibleCount, items.length)

  const maxStart = Math.max(0, items.length - visible)
  const start = Math.min(windowStart, maxStart)
  const lastVisible = Math.min(start + visible - 1, items.length - 1)

  const offset = offsets[start] ?? 0
  const nextRowTop = offsets[start + visible]
  const measured = offsets.length > 0 ? (nextRowTop ?? listHeight) - offset : undefined
  // Never taller than the space, even when a caller pins `visibleCount` high.
  const height =
    measured === undefined ? undefined : space > 0 ? Math.min(measured, space) : measured

  /* --- the window follows the active stage, and only the active stage ------ */

  const bringIntoView = useCallback(
    (index: number) => {
      setWindowStart((current) => {
        if (index < current) return index
        if (index > current + visible - 1) return index - visible + 1
        return current
      })
    },
    [visible],
  )

  // Deliberately keyed on the active stage alone: a manual scroll must not be
  // yanked back while the user is reading somewhere else in the pipeline.
  useEffect(() => {
    bringIntoView(activeIndex)
  }, [activeIndex, bringIntoView])

  /* --- fire the arrow once, on the transition that just completed ---------- */

  const previousStatuses = useRef<string | null>(null)
  const [firing, setFiring] = useState<number | null>(null)

  useEffect(() => {
    const previous = previousStatuses.current
    previousStatuses.current = statusKey
    if (previous === null) return

    const before = previous.split('|')
    const now = statusKey.split('|')
    let moved: number | null = null
    now.forEach((status, index) => {
      if (status === 'complete' && before[index] !== 'complete') moved = index
    })
    if (moved === null) return

    setFiring(moved)
    const timer = window.setTimeout(() => setFiring(null), 900)
    return () => window.clearTimeout(timer)
  }, [statusKey])

  const activeAbove = activeIndex < start
  const activeBelow = activeIndex > lastVisible

  return (
    <div className="track">
      <div className="track__head">
        <span className="eyebrow">Pipeline</span>
        <span className="mono track__count">
          {completedCount}/{items.length} done
        </span>
      </div>

      <button
        type="button"
        className={cn('track__nav', activeAbove && 'track__nav--flag')}
        onClick={() => setWindowStart(Math.max(0, start - 1))}
        disabled={start === 0}
        aria-label="Show earlier stages"
      >
        <ChevronIcon up />
      </button>

      <div className="track__space" ref={spaceRef}>
        <div
          className="track__viewport"
          style={height === undefined ? undefined : { height: `${height}px` }}
          // Focusing a card the window has slid past would otherwise scroll the
          // clipped box. The window is what moves, never the scroll position.
          onScroll={(event) => {
            event.currentTarget.scrollTop = 0
          }}
        >
          <ol
            ref={listRef}
            className="track__list"
            style={{ transform: `translateY(${-offset}px)` }}
            aria-label={label}
          >
            {items.map((item, index) => {
              // `statuses` is built by mapping over `items`, so this index is
              // always in range; the fallback satisfies noUncheckedIndexedAccess
              // without pretending the array might be short.
              const status = statuses[index] ?? 'pending'
              const selectable = onSelect !== undefined && (canSelect?.(index) ?? true)
              const inWindow = index >= start && index <= lastVisible

              return (
                <li key={item.id} className="track__row">
                  <button
                    type="button"
                    className={cn('track__card', `track__card--${status}`)}
                    aria-current={index === activeIndex ? 'step' : undefined}
                    aria-hidden={inWindow ? undefined : true}
                    tabIndex={selectable && inWindow ? 0 : -1}
                    disabled={!selectable}
                    onFocus={() => bringIntoView(index)}
                    onClick={() => onSelect?.(index)}
                  >
                    <span className="track__body">
                      <span className="mono track__index">{padIndex(item.index)}</span>
                      <span className="track__title">{item.title}</span>
                      <span className="track__status">
                        {labelOf?.(index, status) ?? DEFAULT_LABELS[status]}
                      </span>
                    </span>

                    <span className="track__mark">
                      {status === 'complete' ? (
                        <CheckIcon />
                      ) : status === 'error' ? (
                        <AlertIcon />
                      ) : status === 'active' ? (
                        <RunningIcon busy={busy} />
                      ) : (
                        <PendingIcon />
                      )}
                    </span>
                  </button>

                  {index < items.length - 1 && (
                    <span
                      className={cn(
                        'track__link',
                        status === 'complete' && 'track__link--complete',
                        index === firing && 'track__link--firing',
                      )}
                      aria-hidden
                    >
                      <span className="track__link-line">
                        {index === firing && <span className="track__link-pulse" />}
                      </span>
                      <svg className="track__link-head" viewBox="0 0 12 8" aria-hidden>
                        <path d="M6 8 0.6 1.6h10.8Z" fill="currentColor" />
                      </svg>
                    </span>
                  )}
                </li>
              )
            })}
          </ol>
        </div>
      </div>

      <button
        type="button"
        className={cn('track__nav', activeBelow && 'track__nav--flag')}
        onClick={() => setWindowStart(Math.min(maxStart, start + 1))}
        disabled={start >= maxStart}
        aria-label="Show later stages"
      >
        <ChevronIcon up={false} />
      </button>
    </div>
  )
}
