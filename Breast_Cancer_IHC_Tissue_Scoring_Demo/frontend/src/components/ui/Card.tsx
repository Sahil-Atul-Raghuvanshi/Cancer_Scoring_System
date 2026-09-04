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
