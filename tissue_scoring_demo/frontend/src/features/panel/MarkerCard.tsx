import { Badge } from '@/components/ui/Badge'
import { cn } from '@/lib/cn'
import type { MarkerInfo } from '@/types/panel'

const COMPARTMENT_TONE = { membrane: 'accent', cytoplasm: 'violet' } as const

interface MarkerCardProps {
  marker: MarkerInfo
  selected: boolean
  /** Set once the case folder has been resolved, so a missing slide is visible before loading. */
  available: boolean | null
  onSelect: () => void
}

export function MarkerCard({ marker, selected, available, onSelect }: MarkerCardProps) {
  return (
    <button
      type="button"
      className={cn(
        'marker-card',
        selected && 'marker-card--selected',
        available === false && 'marker-card--missing',
      )}
      onClick={onSelect}
    >
      <span className="marker-card__letter mono">{marker.letter}</span>
      <span className="marker-card__name">{marker.name}</span>
      <Badge tone={COMPARTMENT_TONE[marker.compartment as 'membrane' | 'cytoplasm']}>
        {marker.compartment} · {marker.compartmentWidthUm} µm
      </Badge>
      {available === false && <span className="marker-card__hint">not in this case folder</span>}
    </button>
  )
}
