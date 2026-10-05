/** Format a whole number with thousands separators. */
export function formatCount(value: number): string {
  return new Intl.NumberFormat('en-GB').format(Math.round(value))
}

/** Turn `1400` into `1.4 s`, `420` into `420 ms`. */
export function formatDuration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`
}

/** `3` -> `03`, so step numbers stay the same width. */
export function padIndex(value: number): string {
  return String(value).padStart(2, '0')
}

/**
 * "4 minutes ago", "yesterday", "12 Mar" - how long since a run was touched.
 *
 * Anything older than a week gets a date instead of a count of days, because
 * "23 days ago" is a number nobody converts back into a day they remember.
 */
export function relativeTime(iso: string | null | undefined): string | null {
  if (!iso) return null
  const then = new Date(iso)
  const ms = then.getTime()
  if (Number.isNaN(ms)) return null

  const minutes = Math.round((Date.now() - ms) / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes} min ago`

  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} h ago`

  const days = Math.round(hours / 24)
  if (days === 1) return 'yesterday'
  if (days < 7) return `${days} days ago`

  return then.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })
}
