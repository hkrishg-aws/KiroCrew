/**
 * Screenshot runner for capture/capabilities-drift-repair.html.
 *
 * From website/:
 *   npx vite --host 127.0.0.1 --port 6842 --strictPort
 *   node scripts/capture-capabilities-drift-repair.mjs http://127.0.0.1:6842 <outdir>
 *
 * Two frames per theme: the drifted pane (notice names Review changes, button
 * enabled with no edit) and the refusal after Review + Save when the drift sits
 * in a setting the page cannot show (names the setting and the file). Each frame
 * asserts the copy it photographs, so a stale string cannot pass as evidence.
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'

const BASE = process.argv[2] || 'http://127.0.0.1:6842'
const OUT = process.argv[3] || '../temp-screenshots/capabilities-drift-repair'

mkdirSync(OUT, { recursive: true })
const browser = await chromium.launch()
let failed = 0

for (const theme of ['light', 'dark']) {
  const ctx = await browser.newContext({ viewport: { width: 820, height: 700 }, deviceScaleFactor: 2, colorScheme: theme })
  const page = await ctx.newPage()
  const errors = []
  page.on('pageerror', e => errors.push(String(e)))
  try {
    await page.goto(`${BASE}/capture/capabilities-drift-repair.html?theme=${theme}`, { waitUntil: 'networkidle' })
    const root = page.locator('[data-capture-root]')
    const review = page.getByRole('button', { name: 'Review changes', exact: true })
    await review.waitFor({ timeout: 10000 })
    if (await review.isDisabled()) throw new Error('drift scene: Review changes is disabled')
    const hint = await page.getByText(/changed outside this page/).innerText()
    if (!hint.includes('Review changes')) throw new Error(`drift notice does not name the button: ${hint}`)
    if (!hint.includes('what that outside edit changed')) throw new Error(`drift notice does not say what Review shows: ${hint}`)
    if (!hint.includes('If it cannot, Save stops')) throw new Error(`drift notice promises Save unconditionally: ${hint}`)
    await root.screenshot({ path: `${OUT}/01-drift-${theme}.png` })

    await review.click()
    const save = page.getByRole('button', { name: 'Save reviewed changes' })
    await save.waitFor({ timeout: 10000 })
    await save.click()
    const refusal = page.getByText(/a setting this page cannot show/)
    await refusal.waitFor({ timeout: 10000 })
    const text = await refusal.innerText()
    for (const needle of ['toolsSettings', 'crew-3f9a1c2e7b4d.json']) {
      if (!text.includes(needle)) throw new Error(`refusal does not name ${needle}: ${text}`)
    }
    if (text.includes('{{')) throw new Error(`refusal leaked an interpolation placeholder: ${text}`)
    if (await page.getByText(/The saved version changed/).count()) throw new Error('refusal fell back to the stale-version copy')
    await root.screenshot({ path: `${OUT}/02-refused-${theme}.png` })
    if (errors.length) throw new Error(`page errors: ${errors.join(' | ')}`)
    console.log(`ok ${theme}`)
  } catch (e) {
    failed++
    console.error(`FAIL ${theme}: ${e instanceof Error ? e.message : e}`)
  } finally {
    await ctx.close()
  }
}
await browser.close()
process.exit(failed ? 1 : 0)
