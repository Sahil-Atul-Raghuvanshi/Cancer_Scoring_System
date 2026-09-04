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
