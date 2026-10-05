/** Types for the five-antibody panel. Mirrors `backend/app/schemas/panel.py`. */

export type Compartment = 'membrane' | 'cytoplasm' | 'none'

export interface MarkerInfo {
  letter: string
  name: string
  fullName: string
  compartment: Compartment
  scored: boolean
  compartmentWidthUm: number
  secondMeasure: string
  expectedPercentMin: number
  expectedPercentMax: number
}

export interface PanelResponse {
  markers: MarkerInfo[]
}
