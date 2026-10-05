import { cn } from '@/lib/cn'

import './ui.css'

/** A small state indicator: live pulses, offline and idle sit still. */
export function StatusDot({ state }: { state: 'live' | 'offline' | 'idle' }) {
  return <span className={cn('status-dot', `status-dot--${state}`)} aria-hidden />
}
