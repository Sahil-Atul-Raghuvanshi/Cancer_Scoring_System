/**
 * NOT USED BY ANY PANEL, as of 2026-09-15. Kept, not deleted.
 *
 * Every feature in this app lays itself out with its own BEM-ish classes in its own
 * stylesheet - `.qc-map__tab`, `.cal-strip__item`, `.align__pair` - rather than with
 * a shared container, and none of them imports this. It is left in place because
 * "there is no Card component" and "there is a Card component nobody uses" are
 * different facts, and only one of them is true; a new panel that wants a plain
 * bordered container should use this rather than inventing a fourth one.
 */
import type { HTMLAttributes, ReactNode } from 'react'

import { cn } from '@/lib/cn'

import './ui.css'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  padded?: boolean
  interactive?: boolean
  accent?: boolean
  children: ReactNode
}

export function Card({
  padded = true,
  interactive = false,
  accent = false,
  className,
  children,
  ...rest
}: CardProps) {
  return (
    <div
      className={cn(
        'card',
        padded && 'card--padded',
        interactive && 'card--interactive',
        accent && 'card--accent',
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  )
}
