import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

import './ui.css'

export type BadgeTone = 'neutral' | 'accent' | 'violet' | 'success' | 'warn' | 'danger'

interface BadgeProps {
  tone?: BadgeTone
  children: ReactNode
  className?: string
}

export function Badge({ tone = 'neutral', children, className }: BadgeProps) {
  return <span className={cn('badge', `badge--${tone}`, className)}>{children}</span>
}
