/**
 * Drive the demo in a real browser and photograph steps 11 to 15.
 *
 *   node scripts/check-walkthrough.mjs [--marker A] [--out ../walkthrough_shots]
 *
 * Written because those five screens were rebuilt around a pan-and-zoom viewer
 * that draws into a canvas, and a canvas drawing the wrong thing - or nothing -
 * type-checks perfectly. `npm run build` says the code compiles; this says the
 * outlines land on the tissue.
 *
 * It walks the walkthrough the way a person does: open the case from the
 * previous-cases list, then press continue until the end, screenshotting the
 * steps that matter. **Console errors are collected throughout and printed at
 * the end**, because a React error inside a canvas effect shows up as an empty
 * frame and nothing else.
 *
 * Needs both servers already running - the API on 8000 and vite on 5173.
 */

import { mkdirSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { chromium } from 'playwright'

const HERE = dirname(fileURLToPath(import.meta.url))
const FRONTEND = 'http://localhost:5173'

/** The steps this run exists to look at, by the number on the rail. */
const WATCHED = {
  11: 'nuclei',
  12: 'cell-typing',
  13: 'compartments',
  14: 'per-cell',
  15: 'binning',
}

function arg(name, fallback) {
  const index = process.argv.indexOf(`--${name}`)
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback
}

const marker = arg('marker', 'A')
const out = resolve(HERE, '..', arg('out', join('..', '..', 'walkthrough_shots')))
mkdirSync(out, { recursive: true })

const problems = []
const shots = []

function shot(page, name, locator) {
  const path = join(out, name)
  const target = locator ?? page
  return target.screenshot({ path, fullPage: locator ? undefined : true }).then(() => {
    shots.push(name)
  })
}

/** The step number on screen, from the "step 11 / 17" eyebrow above the panel. */
async function currentStep(page) {
  const badge = page.locator('.stage__eyebrow .eyebrow').first()
  if ((await badge.count()) === 0) return null
  try {
    const text = await badge.innerText({ timeout: 2000 })
    const match = /step\s+0*(\d+)/i.exec(text)
    return match ? Number(match[1]) : null
  } catch {
    return null
  }
}

/**
 * Give a step long enough to fetch its geometry and paint a frame.
 *
 * Deliberately not `networkidle`: the slide viewer keeps requesting tiles for as
 * long as it is on screen, so the network never goes idle and waiting on it
 * would hang here rather than report anything.
 */
async function settle(page) {
  for (let i = 0; i < 90; i += 1) {
    if ((await page.locator('.viewer__canvas canvas').count()) > 0) break
    await page.waitForTimeout(1000)
  }
  await page.waitForTimeout(4500)
}

/**
 * Go in until one cell is worth drawing, and centre on tissue that has some.
 *
 * Step 11 offers a "fly to" chip per region, which lands on a sampled square -
 * exactly where the outlines are. The later steps have no such control, so they
 * get the viewer's own zoom button pressed until the overlay's threshold is
 * crossed. Either way the screenshot afterwards is of cells, not of a slide.
 */
async function zoomIn(page, step) {
  const viewer = page.locator('.viewer').first()
  if ((await viewer.count()) === 0) return false

  // Every one of these steps now offers the same "fly to" row, and the first
  // chip is the region holding the most shapes - which is where a reader would
  // go, and the only place a screenshot proves anything.
  const fly = page.locator('.nuc-jump .nuc-chip').first()
  if ((await fly.count()) > 0) {
    await fly.click()
    await page.waitForTimeout(7000)
    return true
  }

  const zoom = viewer.locator('.viewer__controls button', { hasText: '+' }).first()
  if ((await zoom.count()) === 0) return false
  for (let i = 0; i < 9; i += 1) {
    await zoom.click()
    await page.waitForTimeout(700)
  }
  await page.waitForTimeout(5000)
  return true
}

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } })

page.on('console', (message) => {
  if (message.type() !== 'error') return
  const [first] = message.text().split(/\r?\n/)
  problems.push(`console.error: ${first}`)
})
page.on('pageerror', (error) => {
  // Printed as it happens as well as collected: the summary at the end gets
  // truncated by whatever is reading this, and the first line is the diagnosis.
  console.log(`  !! pageerror: ${error.message}`)
  problems.push(`pageerror: ${error.message}`)
})

// The URL, not just the status. "409 Conflict" on its own says a step was not
// ready; which step is the whole diagnosis.
page.on('response', (response) => {
  if (response.status() >= 400) {
    problems.push(`HTTP ${response.status()} ${new URL(response.url()).pathname}`)
  }
})

console.log('opening the home page')
await page.goto(FRONTEND, { waitUntil: 'networkidle' })
await page.waitForTimeout(2000)
await shot(page, '00_home.png')

const section = page.locator('#previous-cases')
if ((await section.count()) === 0) {
  problems.push('no previous-cases section on the home page')
} else {
  await section.scrollIntoViewIfNeeded()
  await page.waitForTimeout(1000)
  await shot(page, '01_previous_cases.png', section)

  const row = page.locator('.hist-marker').filter({ hasText: marker }).first()
  let button = row.getByRole('button', { name: 'Replay' })
  if ((await button.count()) === 0) button = row.getByRole('button', { name: 'Resume' })

  if ((await button.count()) === 0) {
    problems.push(`marker ${marker} offers neither Replay nor Resume`)
  } else {
    console.log(`opening marker ${marker}`)
    await button.first().click()
    await page.waitForURL('**/demo', { timeout: 60_000 })
    await page.waitForTimeout(3000)

    const seen = new Set()
    // How many polls this step has been stuck on. A gated step shows its "run it
    // or skip it" offer immediately and only replaces it once the cached report
    // arrives, so skipping the moment a Skip button exists races that fetch - and
    // skipping step 8 on a stored run is how you end up at step 9 being told there
    // is no class map. Skip is a last resort, after the step has had time.
    let stalled = 0
    let previous = null

    for (let i = 0; i < 300; i += 1) {
      const index = await currentStep(page)
      if (index === null) {
        await page.waitForTimeout(1000)
        continue
      }
      if (index !== previous) {
        previous = index
        stalled = 0
      }

      if (WATCHED[index] && !seen.has(index)) {
        console.log(`  step ${index} (${WATCHED[index]}) - settling`)
        await settle(page)
        const stem = `${String(index).padStart(2, '0')}_${WATCHED[index]}`
        await shot(page, `${stem}.png`)

        // The whole point of these screens is what appears when you go in far
        // enough for a cell to be worth drawing. Zoomed out they are correctly
        // blank, so a screenshot of the page as it arrives proves nothing.
        const zoomed = await zoomIn(page, index)
        if (zoomed) {
          await shot(page, `${stem}_zoom.png`, page.locator('.viewer').first())
        } else {
          problems.push(`step ${index}: no viewer to zoom`)
        }
        seen.add(index)
      }

      if (seen.size === Object.keys(WATCHED).length && index >= 15) break

      const advance = page.locator('button', { hasText: 'Continue to step' })
      if ((await advance.count()) > 0 && (await advance.first().isEnabled())) {
        await advance.first().click()
        await page.waitForTimeout(1500)
        continue
      }

      stalled += 1

      // A gated step asks before it runs. Quality control is the one step the
      // pipeline is genuinely designed to work without, and a stored run was
      // scored without it, so walking past it is what a viewer would do - but
      // only after the step has had time to find its own cached result.
      if (stalled >= 8) {
        const skip = page.getByRole('button', { name: /^Skip/ })
        if ((await skip.count()) > 0 && (await skip.first().isEnabled())) {
          console.log(`  step ${index} - no stored result, skipping`)
          await shot(page, `stuck_${String(index).padStart(2, '0')}.png`)
          await skip.first().click()
          await page.waitForTimeout(1500)
          stalled = 0
          continue
        }
      }

      await page.waitForTimeout(2500)
    }

    const missing = Object.keys(WATCHED)
      .map(Number)
      .filter((step) => !seen.has(step))
    if (missing.length) problems.push(`never reached steps ${missing.join(', ')}`)
  }
}

await shot(page, '99_final.png')
await browser.close()

console.log(`\nscreenshots in ${out}`)
for (const name of shots) {
  let size = 0
  try {
    size = statSync(join(out, name)).size
  } catch {
    /* not written */
  }
  console.log(`  ${name.padEnd(28)} ${(size / 1024).toFixed(0).padStart(7)} KB`)
}

if (problems.length) {
  console.log(`\n${problems.length} problem(s):`)
  for (const problem of [...new Set(problems)]) console.log(`  - ${problem}`)
  process.exit(1)
}

console.log('\nno console errors, every watched step reached')
