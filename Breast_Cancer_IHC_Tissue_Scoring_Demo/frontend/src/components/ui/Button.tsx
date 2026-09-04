import type { ButtonHTMLAttributes, ReactNode } from 'react'

import { cn } from '@/lib/cn'

import './ui.css'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'md' | 'lg'
  /** Show a spinner and block interaction while work is in flight. */
  loading?: boolean
  /** Gently pulse the button so the next click target is obvious. */
  attention?: boolean
  children: ReactNode
}

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  attention = false,
  disabled,
  className,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      type="button"
      className={cn(
        'btn',
        `btn--${variant}`,
        `btn--${size}`,
        attention && !loading && !disabled && 'btn--attention',
        className,
      )}
      disabled={disabled || loading}
      {...rest}
    >
      {loading && <span className="btn__spinner" aria-hidden />}
      {children}
    </button>
  )
}
