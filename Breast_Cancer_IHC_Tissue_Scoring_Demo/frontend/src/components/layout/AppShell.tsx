import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'

import { FullscreenToggle } from './FullscreenToggle'

import './layout.css'

interface AppShellProps {
  children: ReactNode
}

function BrandMark() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden>
      <circle cx="12" cy="12" r="8.5" fill="none" stroke="var(--accent)" strokeWidth="1.6" />
      <circle cx="9.6" cy="10.4" r="2.1" fill="var(--stain-hematoxylin)" />
      <circle cx="14.4" cy="13.6" r="1.8" fill="var(--stain-dab)" />
      <circle cx="13.6" cy="8.8" r="1.3" fill="var(--stain-hematoxylin)" />
    </svg>
  )
}

/* Nav glyphs, drawn on the same 24px grid and 1.7 stroke as the fullscreen
   control, so the three sides of the header share one hand. */

function HomeIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M4 10.2 12 4l8 6.2V19a1 1 0 0 1-1 1h-4v-5.4H9V20H5a1 1 0 0 1-1-1z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function WalkthroughIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M12 6.6C10.5 5.2 8.6 4.6 6 4.6a1 1 0 0 0-1 1v11.2a1 1 0 0 0 1 1c2.6 0 4.5.6 6 2 1.5-1.4 3.4-2 6-2a1 1 0 0 0 1-1V5.6a1 1 0 0 0-1-1c-2.6 0-4.5.6-6 2zm0 0v12.2"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="shell">
      <div className="shell__ambient" aria-hidden>
        <div className="shell__grid" />
        <div className="shell__glow shell__glow--one" />
        <div className="shell__glow shell__glow--two" />
        <div className="shell__glow shell__glow--three" />
      </div>

      <header className="header">
        <div className="header__inner">
          <div className="header__brand">
            <span className="header__mark">
              <BrandMark />
            </span>
            <span className="header__title">Breast Cancer IHC Tissue Scoring</span>
          </div>

          <nav className="header__nav" aria-label="Primary">
            <NavLink to="/" className="header__link" end>
              <HomeIcon />
              <span>Home</span>
            </NavLink>
            <NavLink to="/demo" className="header__link">
              <WalkthroughIcon />
              <span>Walkthrough</span>
            </NavLink>
          </nav>

          <div className="header__actions">
            <span className="header__divider" aria-hidden />
            <FullscreenToggle />
          </div>
        </div>
      </header>

      <main className="shell__main">{children}</main>
    </div>
  )
}
