/**
 * Which slide a step is being shown on, resolved from what the step declares.
 *
 * The backend catalogue now carries `runsOn` for every stage - see
 * `backend/app/data/pipeline_steps.py`. This turns that declaration plus the loaded
 * case into the two things a panel needs: which slide it is drawing, and whether
 * there is a second one the viewer may switch to.
 *
 * **Why a step's slide is declared rather than assumed.** The session's `uploadId`
 * is always the H&E, and for a long time every panel simply used it. That is right
 * for steps 7 to 9 and wrong for steps 11 to 16, which read the immunostained slide,
 * and wrong in a subtler way for steps 2 to 6, which are a per-slide prefix that has
 * to run on both. The result on screen was a step captioned "the picture the score
 * is measured from" showing an H&E tile, whose DAB channel is a picture of nothing.
 */

import type { SlideRole } from '@/types/pipeline'
import type { SlideReadout } from '@/types/slide'

/** One slide a step can be shown on. */
export interface SlideOption {
  role: Exclude<SlideRole, 'case'>
  uploadId: string
  /** What to call it on screen: 'H&E', 'CD44'. Falls back to the letter. */
  label: string
  /** The file behind it, so a reader can check it is the slide they meant. */
  filename: string | null
  /**
   * True when the file's name does not look like what this slot is meant to hold -
   * an immunostained slide uploaded on its own lands in the H&E slot, and nothing
   * downstream would notice. A suggestion from the filename, never a verdict.
   */
  suspect: boolean
}

export interface SlideChoice {
  /** Every slide this step can be shown on, H&E first. Empty for a case step. */
  options: SlideOption[]
  /** The one on screen, or null when this step has no single slide. */
  selected: SlideOption | null
  /** Whether to offer the switch: only when the step runs on a slide it is not showing. */
  offersChoice: boolean
  /** True for step 17, which is keyed on the case rather than on any slide. */
  isCaseScoped: boolean
}

interface Session {
  uploadId: string | null
  readout: SlideReadout | null
  ihcUploadId: string | null
  ihcReadout: SlideReadout | null
  marker: string | null
  filename: string | null
}

function option(
  role: 'he' | 'ihc',
  uploadId: string,
  readout: SlideReadout | null,
  fallbackLabel: string,
  fallbackFilename: string | null,
): SlideOption {
  return {
    role,
    uploadId,
    label: readout?.marker ?? fallbackLabel,
    filename: readout?.filename ?? fallbackFilename,
    // 'unknown' is not suspect: a file that does not follow the naming convention
    // is a file we cannot judge, which is different from one we can see is wrong.
    suspect: readout != null && readout.slideRole !== 'unknown' && readout.slideRole !== role,
  }
}

/**
 * Resolve a stage's declared slides against the loaded case.
 *
 * `preferred` is the role the viewer last picked. It is honoured only where the
 * step actually runs on it, so walking from step 5 (both slides) to step 8 (H&E
 * only) shows the H&E rather than an IHC slide the step never reads.
 */
export function slideChoiceFor(
  runsOn: SlideRole[] | undefined,
  session: Session,
  preferred: 'he' | 'ihc',
  readsSlidesTogether = false,
): SlideChoice {
  const roles = runsOn ?? ['he']

  if (roles.includes('case')) {
    return { options: [], selected: null, offersChoice: false, isCaseScoped: true }
  }

  const options: SlideOption[] = []
  if (roles.includes('he') && session.uploadId) {
    options.push(option('he', session.uploadId, session.readout, 'H&E', session.filename))
  }
  if (roles.includes('ihc') && session.ihcUploadId) {
    options.push(
      option('ihc', session.ihcUploadId, session.ihcReadout, session.marker ?? 'IHC', null),
    )
  }

  const selected = options.find((entry) => entry.role === preferred) ?? options[0] ?? null
  return {
    options,
    selected,
    // One option is not a choice, and saying "H&E | IHC" with the second half dead
    // would tell a reader the IHC slide is missing when they simply loaded one file.
    //
    // Nor is a stage that reads both slides at once: step 1 already shows both
    // readouts and step 10 already shows both halves of its registration, so a
    // switch there would be a control that changes nothing - which teaches the
    // reader something false about how the step works.
    offersChoice: options.length > 1 && !readsSlidesTogether,
    isCaseScoped: false,
  }
}

