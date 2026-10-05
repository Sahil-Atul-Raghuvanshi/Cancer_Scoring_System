/** One photograph of the home page, as it will be found. */

import { chromium } from 'playwright'

const out = process.argv[2] ?? '../../walkthrough_shots/morning_home.png'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } })
await page.goto('http://localhost:5173', { waitUntil: 'networkidle' })
await page.waitForTimeout(2500)
await page.locator('#previous-cases').scrollIntoViewIfNeeded()
await page.waitForTimeout(1200)
await page.locator('#previous-cases').screenshot({ path: out })
await browser.close()
console.log(`wrote ${out}`)
