/**
 * Turning what steps 10 to 13 stored into what the slide viewer draws.
 *
 * Steps 11, 12, 13, 14 and 15 all show the same cells on the same slide and
 * differ only in what colour each one is and what happens when it is clicked.
 * Building the overlay five times would be five chances for one screen's
 * outlines to sit a field away from another's, so the conversion happens once,
 * here, and each step supplies a paint function.
 *
 * **A cell is its field and its id.** Step 11 segments each sampled field on its
 * own, so every field's instance map starts again at 1 and one region holds a
 * dozen nucleus 14s. `cellKey` is the only identifier any screen may use; a bare
 * id already cost this project a 33 % tumour share read as 91 %.
 */

import type {
  OverlayField,
  OverlayPolygon,
  OverlayRegion,
} from '@/components/viewer/SlideOverlayViewer'
import type { RegionCompartmentRings } from '@/types/compartments'
import type { AlignmentReport } from '@/types/ihcAlignment'
import type { NucleusOut, RegionNucleiPayload } from '@/types/nuclei'

/** The only identifier a screen may use for a cell. See the note above. */
export function cellKey(fieldIndex: number, nucleusId: number): string {
  return `${fieldIndex}:${nucleusId}`
}

/** Splits a key back into its parts, for a screen that has to look one up. */
export function parseCellKey(key: string): { fieldIndex: number; id: number } | null {
  const [field, id] = key.split(':')
  if (field === undefined || id === undefined) return null
  const parsed = { fieldIndex: Number(field), id: Number(id) }
  return Number.isFinite(parsed.fieldIndex) && Number.isFinite(parsed.id) ? parsed : null
}

/**
 * Every colour these five screens draw with.
 *
 * `classes` matches `CELL_TYPE_COLOURS` in
 * `backend/app/pipeline/step12_cell_typing/classify.py` exactly, because that
 * module renders the field PNGs and this one draws the slide overlay - two
 * pictures of one classification that must not disagree.
 *
 * `compartments` reuses the same three hues for a different question, which is
 * deliberate rather than careless: on step 13 nothing is classified, so there is
 * no second meaning for red to collide with, and re-using a palette the reader
 * has already learned costs less than teaching a fourth one.
 *
 * **Every shape is filled solid, and `fill` is the identity colour.** A
 * translucent fill over a stained slide is two colours at once: a red cell over
 * dark brown tumour and the same red cell over pale stroma are different reds,
 * so the colour stops meaning the class and starts meaning the class plus
 * whatever happened to be underneath it. Two readers comparing two parts of one
 * slide were comparing the tissue, not the answer. Filled solid, one class is
 * one colour everywhere.
 *
 * So `stroke` is no longer the identity colour - it is a darker shade of the
 * fill, and its only job is to keep two touching cells of the same class from
 * reading as one cell. Anything that shows a colour to the reader - a legend
 * swatch, a layer toggle - uses `fill`.
 */
export const CELL_COLOUR = {
  /** Step 11: a nucleus is a nucleus. Blue, because nothing is decided yet. */
  nucleus: { stroke: '#0c4a6e', fill: '#38bdf8' },
  /**
   * In the border band: drawn, because it is there, but not in the count.
   *
   * The same blue, darkened, rather than a second hue - these are not a
   * different kind of cell, they are the same cell excluded along with the strip
   * of area it sits in. It used to be the hollow one, which is the one thing a
   * solid overlay cannot say.
   */
  uncounted: { stroke: '#082f49', fill: '#075985' },

  /** Step 12. Keyed by `CellType`. */
  classes: {
    0: { label: 'Tumour cell', stroke: '#7f1d1d', fill: '#f87171' },
    1: { label: 'Immune cell', stroke: '#78350f', fill: '#fbbf24' },
    2: { label: 'Support cell', stroke: '#064e3b', fill: '#34d399' },
  } as Record<number, { label: string; stroke: string; fill: string }>,

  /**
   * Step 13.
   *
   * `dim` is the band this antibody is not measured in: the same hue, dark, so
   * "part of the cell" and "where the number came from" stay one glance apart
   * now that lowering the alpha is no longer how that is said.
   */
  compartments: {
    nucleus: { label: 'Nucleus', stroke: '#7f1d1d', fill: '#f87171', dim: '#7f1d1d' },
    cytoplasm: { label: 'Cytoplasm', stroke: '#78350f', fill: '#fbbf24', dim: '#92400e' },
    membrane: { label: 'Membrane', stroke: '#064e3b', fill: '#34d399', dim: '#047857' },
  },

  /**
   * Steps 14 and 15: how much brown is in this cell.
   *
   * One hue getting lighter and stronger, not four unrelated colours, because
   * the levels are ordered and a reader should be able to rank them without
   * consulting the key. Level 0 is deliberately off-hue - a grey - because it is
   * not the bottom of the scale so much as the absence of it.
   *
   * Amber rather than brown: the slide underneath is already brown, and a brown
   * outline on brown tissue is not an outline.
   */
  bins: [
    { label: 'Negative (0)', stroke: '#334155', fill: '#64748b' },
    { label: 'Weak (1+)', stroke: '#7c2d12', fill: '#b45309' },
    { label: 'Moderate (2+)', stroke: '#92400e', fill: '#f59e0b' },
    { label: 'Strong (3+)', stroke: '#a16207', fill: '#fde047' },
  ],
} as const

/**
 * A cell's fill on step 14, from how dark its compartment is.
 *
 * A continuous ramp rather than step 15's four levels, because step 14 has not
 * binned anything yet - putting its cells into bins on this screen would be
 * showing the next step's answer a screen early, and the point of step 14 is
 * that what comes out of it is a number and not yet a verdict.
 *
 * `ceiling` is where the ramp saturates. Passed in from the data rather than
 * fixed, because optical density has no natural maximum and a hard-coded one
 * would make a weakly stained slide look uniformly blank.
 */
export function odColour(od: number, ceiling: number): { stroke: string; fill: string } {
  const t = Math.max(0, Math.min(1, ceiling > 0 ? od / ceiling : 0))
  // Grey-blue at nothing through to bright amber at the ceiling. Interpolated in
  // plain RGB: the two ends are close enough in hue that nothing goes muddy.
  const r = Math.round(100 + (253 - 100) * t)
  const g = Math.round(116 + (224 - 116) * t)
  const b = Math.round(139 + (71 - 139) * t)
  // Solid: the ramp is the whole message here, and an alpha ramp on top of it
  // would have the tissue underneath deciding how far up the ramp a cell looked.
  // The outline is the same colour at 45 %, dark enough to separate two cells
  // that touch without introducing a second hue.
  return {
    stroke: `rgb(${Math.round(r * 0.45)}, ${Math.round(g * 0.45)}, ${Math.round(b * 0.45)})`,
    fill: `rgb(${r}, ${g}, ${b})`,
  }
}

/** The classes a viewer can be shown, in the order the legend lists them. */
export const CELL_TYPE_ORDER = [0, 1, 2] as const

/**
 * The invasive tumour borders step 10 carried onto the stained slide.
 *
 * Drawn on every one of these screens, at every zoom, because they are the
 * answer to "where did these cells come from" - and without them a reader who
 * zooms out sees a slide with a few blue specks on it and no reason for them.
 */
export function regionOutlines(report: AlignmentReport | null | undefined): OverlayRegion[] {
  if (!report) return []
  return report.regions.map((region) => ({ rank: region.rank, rings: region.ihcRings }))
}

/** What a paint function has to return, or null to leave a cell undrawn. */
export interface Paint {
  stroke: string
  fill?: string | null
  layer?: number
}

/**
 * The stored nuclei of every region a viewer has opened, as overlay fields.
 *
 * `paint` gets the nucleus and its field so a screen can colour by anything it
 * has - a class map, a measurement, a bin - without this module knowing what
 * any of those are.
 */
export function nucleiOverlay(
  geometry: Record<number, RegionNucleiPayload>,
  paint: (nucleus: NucleusOut, fieldIndex: number, rank: number) => Paint | null,
): OverlayField[] {
  const out: OverlayField[] = []

  for (const payload of Object.values(geometry)) {
    for (const field of payload.fields) {
      const polygons: OverlayPolygon[] = []
      for (const nucleus of field.nuclei) {
        const colour = paint(nucleus, field.index, payload.rank)
        if (!colour) continue
        polygons.push({
          id: cellKey(field.index, nucleus.id),
          rings: nucleus.rings,
          stroke: colour.stroke,
          fill: colour.fill ?? null,
          layer: colour.layer,
        })
      }
      out.push({
        rank: payload.rank,
        index: field.index,
        x: field.x,
        y: field.y,
        span: field.span,
        polygons,
      })
    }
  }

  return out
}

/** Every stored nucleus, keyed the one legal way, for a screen that inspects one. */
export function nucleiByKey(
  geometry: Record<number, RegionNucleiPayload>,
): Map<string, { nucleus: NucleusOut; fieldIndex: number; rank: number; span: number }> {
  const out = new Map<
    string,
    { nucleus: NucleusOut; fieldIndex: number; rank: number; span: number }
  >()
  for (const payload of Object.values(geometry)) {
    for (const field of payload.fields) {
      for (const nucleus of field.nuclei) {
        out.set(cellKey(field.index, nucleus.id), {
          nucleus,
          fieldIndex: field.index,
          rank: payload.rank,
          span: field.span,
        })
      }
    }
  }
  return out
}

/** The box a viewer should fly to when a screen says "show me this cell". */
export function cellFocus(
  nucleus: NucleusOut,
  /** How many microns across the view should be. A cell is ~10 µm. */
  spanUm = 90,
  mpp: number | null = null,
): { x: number; y: number; span: number } {
  const span = mpp && mpp > 0 ? spanUm / mpp : 400
  return { x: nucleus.x - span / 2, y: nucleus.y - span / 2, span }
}

/** Which of step 13's regions a screen is showing. */
export interface CompartmentLayers {
  nucleus: boolean
  cytoplasm: boolean
  membrane: boolean
  /** What the other fork would have measured, dashed. */
  alternate: boolean
}

/**
 * Step 13's compartments, as overlay fields.
 *
 * **The three regions are disjoint, so all three are filled.** They used to be
 * nested annuli that differed only in width - the "membrane" was the whole cell
 * body - so filling both would have drawn the 4 µm ring inside the 6 µm one and
 * read as "membrane is under cytoplasm", which is backwards. Now the body is
 * split at its outer shell: nucleus, then cytoplasm, then membrane at the rim,
 * none of them overlapping, each one a colour that means what it says.
 *
 * The band this antibody actually measures is drawn in the bright hue and the
 * one it does not in `dim` - the same hue, dark - because the difference between
 * "this is where the number came from" and "this is part of the cell" is the
 * whole content of the screen, and with solid fills it can no longer be said by
 * lowering the alpha. The unmeasured band keeps the bright colour as its
 * outline, so it is still recognisably the cytoplasm or the membrane.
 * `alternate` - the other fork's region - stays a dashed outline with nothing
 * inside it: it is a comparison, not something that was measured, and filling it
 * would hide the band that was.
 */
export function compartmentOverlay(
  geometry: Record<number, RegionCompartmentRings>,
  layers: CompartmentLayers,
): OverlayField[] {
  const out: OverlayField[] = []

  for (const payload of Object.values(geometry)) {
    const measured = payload.measured

    for (const field of payload.fields) {
      const polygons: OverlayPolygon[] = []

      for (const cell of field.cells) {
        const id = cellKey(field.index, cell.id)

        // Widest first: the viewer draws by ascending layer, so the nucleus ends
        // up on top of the bands around it rather than under them.
        if (layers.alternate && cell.alternate.length > 0) {
          polygons.push({
            id,
            rings: cell.alternate,
            stroke:
              measured === 'membrane'
                ? CELL_COLOUR.compartments.cytoplasm.fill
                : CELL_COLOUR.compartments.membrane.fill,
            fill: null,
            dash: [4, 3],
            layer: 0,
            // Never the click target: it is a comparison, and selecting "the
            // compartment this cell does not have" gives the rest of the screen
            // nothing to show.
            hitTarget: false,
          })
        }

        // The unmeasured band is dark but not invisible: it is part of the
        // cell and the reader has to be able to see where it ends, or "measured
        // in the membrane" is a claim about a boundary they cannot find.
        if (layers.cytoplasm && cell.cytoplasm.length > 0) {
          const own = measured === 'cytoplasm'
          const paint = CELL_COLOUR.compartments.cytoplasm
          polygons.push({
            id,
            rings: cell.cytoplasm,
            stroke: own ? paint.stroke : paint.fill,
            fill: own ? paint.fill : paint.dim,
            layer: 1,
            hitTarget: own,
          })
        }

        if (layers.membrane && cell.membrane.length > 0) {
          const own = measured === 'membrane'
          const paint = CELL_COLOUR.compartments.membrane
          polygons.push({
            id,
            rings: cell.membrane,
            stroke: own ? paint.stroke : paint.fill,
            fill: own ? paint.fill : paint.dim,
            layer: 2,
            hitTarget: own,
          })
        }

        if (layers.nucleus && cell.nucleus.length > 0) {
          polygons.push({
            id,
            rings: cell.nucleus,
            stroke: CELL_COLOUR.compartments.nucleus.stroke,
            fill: CELL_COLOUR.compartments.nucleus.fill,
            layer: 3,
            // One hit target per cell, or a click would count the same cell
            // three times in the "in view" figure under the frame.
            hitTarget: false,
          })
        }
      }

      out.push({
        rank: payload.rank,
        index: field.index,
        x: field.x,
        y: field.y,
        span: field.span,
        polygons,
      })
    }
  }

  return out
}

/** Every stored compartment cell, keyed the one legal way. */
export function compartmentsByKey(
  geometry: Record<number, RegionCompartmentRings>,
): Map<
  string,
  { cell: RegionCompartmentRings['fields'][number]['cells'][number]; x: number; y: number; span: number; rank: number; fieldIndex: number }
> {
  const out = new Map<
    string,
    {
      cell: RegionCompartmentRings['fields'][number]['cells'][number]
      x: number
      y: number
      span: number
      rank: number
      fieldIndex: number
    }
  >()
  for (const payload of Object.values(geometry)) {
    for (const field of payload.fields) {
      for (const cell of field.cells) {
        out.set(cellKey(field.index, cell.id), {
          cell,
          x: field.x,
          y: field.y,
          span: field.span,
          rank: payload.rank,
          fieldIndex: field.index,
        })
      }
    }
  }
  return out
}

/** The centre of a traced ring, for flying a viewer to it. */
export function ringCentre(rings: number[][][]): { x: number; y: number } | null {
  const outer = rings[0]
  if (!outer || outer.length === 0) return null
  let x = 0
  let y = 0
  for (const point of outer) {
    x += point[0] ?? 0
    y += point[1] ?? 0
  }
  return { x: x / outer.length, y: y / outer.length }
}

export interface FlyTarget {
  rank: number
  /** How many shapes are drawn in this region, across every sampled square. */
  count: number
  /** The square with the most in it: level-0 pixels. */
  box: { x: number; y: number; span: number }
}

/**
 * Where it is worth flying to, from the overlay itself.
 *
 * Derived rather than passed in, so a screen cannot offer to fly somewhere it is
 * not drawing anything - which is the one way this control can lie.
 */
export function flyTargets(fields: OverlayField[], limit = 8): FlyTarget[] {
  const byRank = new Map<number, { count: number; best: OverlayField | null; most: number }>()

  for (const field of fields) {
    const size = field.polygons.length
    if (size === 0) continue
    const entry = byRank.get(field.rank) ?? { count: 0, best: null, most: 0 }
    entry.count += size
    if (size > entry.most) {
      entry.most = size
      entry.best = field
    }
    byRank.set(field.rank, entry)
  }

  return [...byRank.entries()]
    .filter(([, entry]) => entry.best !== null)
    .map(([rank, entry]) => ({
      rank,
      count: entry.count,
      box: {
        x: entry.best!.x,
        y: entry.best!.y,
        span: entry.best!.span,
      },
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit)
}
