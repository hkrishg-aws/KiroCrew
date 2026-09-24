/**
 * Screenshot harness for the trust-and-enable modal's FAILURE states and the
 * detail page's install-refusal error box.
 *
 * Runs the REAL built SPA behind the shared static server and answers every
 * /api/** call from fixtures — including the SSE install stream — so no gateway
 * and no kiro-cli is needed.
 *
 * Three states, one flow each, every one starting from a fresh page:
 *
 *   01 failure            → the modal after the retried install failed with the
 *                           install-time gate's PERMANENT refusal (its sentence
 *                           plus the machine code), the grant rolled back
 *   02 transient-failure  → the modal after the retried install failed with a
 *                           server reason the gateway did NOT mark permanent (a
 *                           build failure, no code): the reason is shown, Try
 *                           again stays
 *   03 trusted-page-banner → the detail page when the app is ALREADY trusted: no
 *                           consent modal opens, the first install attempt is
 *                           refused, and the page's own error box carries the
 *                           refusal
 *
 * BEFORE (`before` prefix, served from a dist built at the PR's base commit)
 * captures 01 only and asserts the modal shows ONLY the generic copy with the
 * server's sentence nowhere in it. AFTER captures all three: 01 asserts the
 * permanent-refusal rendering (the modal's headline naming the app in the
 * dialog's own turn-on words with no retry instruction, the plain-language
 * sentence naming the gateway remedy as an action, the server's own sentence
 * beneath, the footer offering Close only); 02 asserts the generic headline
 * kept as the title with the server's reason beneath it and Try again still
 * offered; 03 asserts no dialog, the page's error box carrying the page's own
 * install-framed headline, the plain sentence and the server's sentence with a
 * dismiss control, and the Install button disabled beneath it. Each assertion
 * runs before its frame is taken, so a screenshot can never show a state the
 * run did not verify.
 *
 * Usage: node scripts/capture-trust-modal-install-reason.mjs [outDir] [prefix] [dist]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { serveDist, DEFAULT_DIST } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi } from './lib/stub-dashboard-api.mjs'

const OUT = process.argv[2] || '../temp-screenshots/trust-modal-install-reason'
const PREFIX = process.argv[3] || 'after'
const DIST = process.argv[4] || DEFAULT_DIST

mkdirSync(OUT, { recursive: true })

const APP = 'agent-dashboard'
const DISPLAY = 'Agent Dashboard'
const REPO = 'https://git.example.test/apps/agent-dashboard.git'

/** The install-time gate's own refusal — the sentence the modal renders beneath
 *  its plain-language copy. */
const REFUSAL =
  'Python apps that require a build step are not supported in the desktop app: ' +
  'its bundled interpreter is inside the signed application bundle and cannot ' +
  'install packages'
/** The code the gate returns beside it: a permanent condition for this gateway. */
const REFUSAL_CODE = 'desktop_build_step_unsupported'
/** A reason the gateway does NOT mark permanent: a build that may well pass next time. */
const TRANSIENT_REASON = 'build failed (exit 1): npm install: ETIMEDOUT registry.npmjs.org'

/** The generic copy: the whole message BEFORE, gone AFTER for a permanent refusal. */
const GENERIC = 'could not be turned on, and nothing was changed'
/** The instruction the generic copy carries and a permanent refusal must not. */
const RETRY_INSTRUCTION = 'Choose Try again'
/** The AFTER copy in the modal: the dialog's own verb and assurance, naming the app. */
const DESKTOP_HEADLINE = `\u201c${DISPLAY}\u201d can't be turned on in the desktop version of Kiro Crew, and nothing was changed.`
/** The AFTER copy on the detail page, whose operation is Install. */
const PAGE_HEADLINE = "This app can't be installed in the desktop version of Kiro Crew."
/** The plain sentence both surfaces share, and the remedy it names. */
const DESKTOP_HELP = "It needs Python packages the desktop build can't add"
const GATEWAY_ACTION = 'install it from a browser connected to your gateway'

const REGISTRY_APP = {
  name: APP, displayName: DISPLAY, version: '3.2.6',
  description: 'A dashboard for your agents, with its own FastAPI backend.',
  author: 'example-apps', repo: REPO, gitUrl: REPO, trustRepository: REPO,
  tags: ['dashboard'], installed: false, enabled: false, origin: 'registry',
  updateAvailable: false,
  manifest: {
    name: APP, version: '3.2.6', displayName: DISPLAY,
    backend: { entryPoint: 'server.py', type: 'asgi' },
  },
}

/** One SSE `done` frame, the shape `installFromRegistryStream` parses. */
const sse = (payload) => `event: done\ndata: ${JSON.stringify(payload)}\n\n`

const DENIED = {
  ok: false, name: APP, code: 'app_execution_denied',
  error: `blocked by execution policy: App ${APP} is not trusted to run its own code.`,
  log: '',
}
const PERMANENT = { ok: false, name: APP, error: REFUSAL, code: REFUSAL_CODE }
const TRANSIENT = { ok: false, name: APP, error: TRANSIENT_REASON }

/**
 * What the install stream answers, per attempt, per scenario. An UNTRUSTED app is
 * refused by the execution gate first (which opens the consent modal) and fails
 * on the retry; an already-TRUSTED app fails on its first and only attempt.
 */
const SCENARIOS = {
  permanent: { attempts: [DENIED, PERMANENT] },
  transient: { attempts: [DENIED, TRANSIENT] },
  trusted: { attempts: [PERMANENT] },
}

/** A fresh page with the API stubbed for one scenario. */
async function openPage(context, base, scenario) {
  const page = await context.newPage()
  logPageProblems(page)
  let installs = 0
  await stubDashboardApi(page, {
    extra: async (path, route) => {
      if (path === '/api/apps') { await route.fulfill({ json: [] }); return true }
      // 404 is how "no app occupies this name" really arrives: it makes the page
      // load from the registry AND is the rollback probe's proof of absence.
      if (path === `/api/apps/${APP}`) {
        await route.fulfill({ status: 404, json: { error: 'app not installed' } })
        return true
      }
      if (path === '/api/apps/registry') {
        await route.fulfill({
          json: { apps: [REGISTRY_APP], serverPlatform: { os: 'darwin', arch: 'arm64' } },
        })
        return true
      }
      if (path === '/api/apps/registries') { await route.fulfill({ json: { registries: [] } }); return true }
      if (path === `/api/apps/${APP}/trust` || path === `/api/apps/${APP}/untrust`) {
        await route.fulfill({ json: { ok: true } })
        return true
      }
      if (path === '/api/apps/registry/install-stream') {
        const { attempts } = SCENARIOS[scenario]
        const payload = attempts[Math.min(installs, attempts.length - 1)]
        installs += 1
        await route.fulfill({ status: 200, contentType: 'text/event-stream', body: sse(payload) })
        return true
      }
      return false
    },
  })
  await page.goto(`${base}/apps/detail/${APP}`, { waitUntil: 'domcontentloaded' })
  await page.getByText(DISPLAY).first().waitFor({ timeout: 15000 })
  await page.getByRole('button', { name: 'Install' }).first().click()
  return page
}

/** Drive the consent modal to its failure state and return the modal locator. */
async function failThroughTheModal(page, settledFooter) {
  const modal = page.locator('[role="dialog"]').filter({ hasText: /to run its own code\?/ })
  await modal.waitFor({ timeout: 15000 })
  await modal.getByRole('button', { name: 'Trust this app and enable' }).click()
  await modal.getByRole('alert').waitFor({ timeout: 15000 })
  // Settle the rollback probe + untrust round trip so the copy is final. The
  // footer is the settle signal — and the dialog's own X is also NAMED Close
  // (aria-label), so the footer button is told apart by its visible text.
  await modal.getByRole('button', { name: settledFooter, exact: true })
    .filter({ hasText: settledFooter }).waitFor({ timeout: 15000 })
  await page.waitForTimeout(600)
  return modal
}

const fail = (msg, text) => { throw new Error(`${PREFIX}: ${msg}: ${JSON.stringify(text)}`) }

async function capturePermanent(context, base) {
  const page = await openPage(context, base, 'permanent')
  // BEFORE offers Try again in every failure state; AFTER offers Close only for
  // a permanent refusal, because retrying cannot change the verdict.
  const modal = await failThroughTheModal(page, PREFIX === 'before' ? 'Try again' : 'Close')
  const text = await modal.getByRole('alert').innerText()
  // The page's OWN error banner (behind the modal) always carries the sentence —
  // `reportInstallFailure` journals and shows it there. The MODAL is the surface
  // the user is looking at and the one whose footer decides the next action, so
  // the assertions are scoped to the modal and the banner is reported as a fact.
  const pageHasReason = (await page.locator('body').innerText()).includes('bundled interpreter')
  if (PREFIX === 'before') {
    if (!text.includes(GENERIC)) fail('the generic copy must be the whole message', text)
    if (text.includes('bundled interpreter')) {
      fail('the modal already shows the server reason, so this is not the BEFORE state', text)
    }
  } else {
    if (!text.includes(DESKTOP_HEADLINE)) fail('the modal must lead with the desktop headline', text)
    if (!text.includes(DESKTOP_HELP)) fail('the modal must explain the refusal in plain words', text)
    if (!text.includes(GATEWAY_ACTION)) fail('the modal must name the gateway remedy as an action', text)
    if (!text.includes(REFUSAL)) fail("the modal must render the server's reason", text)
    if (text.indexOf(DESKTOP_HELP) > text.indexOf(REFUSAL)) {
      fail("the plain sentence must sit above the server's reason", text)
    }
    if (text.includes(RETRY_INSTRUCTION)) fail('a permanent refusal must not carry the retry instruction', text)
    if (await modal.getByRole('button', { name: 'Try again' }).count() !== 0) {
      fail('a permanent refusal must not offer Try again', text)
    }
  }
  await modal.screenshot({ path: `${OUT}/${PREFIX}-01-failure.png` })
  await page.close()
  return pageHasReason
}

async function captureTransient(context, base) {
  const page = await openPage(context, base, 'transient')
  const modal = await failThroughTheModal(page, 'Try again')
  const text = await modal.getByRole('alert').innerText()
  if (!text.includes(GENERIC)) fail('a transient failure keeps the generic headline as its title', text)
  if (!text.includes(TRANSIENT_REASON)) fail("the modal must render the server's reason beneath it", text)
  if (text.indexOf(GENERIC) > text.indexOf(TRANSIENT_REASON)) {
    fail("the headline must sit above the server's reason", text)
  }
  if (text.includes(DESKTOP_HEADLINE)) fail('a reason without the code must not get the desktop copy', text)
  await modal.screenshot({ path: `${OUT}/${PREFIX}-02-transient-failure.png` })
  await page.close()
}

async function captureTrustedPage(context, base) {
  const page = await openPage(context, base, 'trusted')
  // The error box is the alert region that carries the server's sentence; the
  // page has other alert regions (the install log among them).
  const box = page.getByRole('alert').filter({ hasText: 'bundled interpreter' })
  await box.first().waitFor({ timeout: 15000 })
  await page.waitForTimeout(600)
  if (await page.locator('[role="dialog"]').count() !== 0) {
    fail('an already-trusted app must not open the consent modal', await page.locator('body').innerText())
  }
  const text = await box.first().innerText()
  if (!text.includes(PAGE_HEADLINE)) fail("the page box must lead with the page's own headline", text)
  if (text.includes(DESKTOP_HEADLINE)) fail("the page box must not borrow the modal's turn-on headline", text)
  if (!text.includes(DESKTOP_HELP)) fail('the page box must explain the refusal in plain words', text)
  if (!text.includes(GATEWAY_ACTION)) fail('the page box must name the gateway remedy as an action', text)
  if (!text.includes(REFUSAL)) fail("the page box must render the server's reason", text)
  if (await box.first().getByRole('button', { name: 'Dismiss' }).count() !== 1) {
    fail('the page box must offer its dismiss control', text)
  }
  // Install under a banner that says installing is impossible would run the same
  // clone and build into the same refusal, so it is disabled while the banner is
  // up (the banner's dismiss is how a retry is reached).
  const install = page.getByRole('button', { name: 'Install', exact: true })
  if (await install.count() !== 1) fail('the page must show exactly one Install button', text)
  if (!(await install.isDisabled())) fail('Install must be disabled while the permanent banner is up', text)
  await page.screenshot({ path: `${OUT}/${PREFIX}-03-trusted-page-banner.png`, fullPage: false })
  await page.close()
}

async function main() {
  const { srv, base } = await serveDist(DIST)
  const browser = await chromium.launch()
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1, serviceWorkers: 'block',
  })
  const wrote = [`${PREFIX}-01-failure.png`]
  const pageHasReason = await capturePermanent(context, base)
  if (PREFIX !== 'before') {
    await captureTransient(context, base)
    await captureTrustedPage(context, base)
    wrote.push(`${PREFIX}-02-transient-failure.png`, `${PREFIX}-03-trusted-page-banner.png`)
  }
  await browser.close()
  srv.close()
  console.log(
    `Wrote ${wrote.map((f) => `${OUT}/${f}`).join(', ')} (01 asserted: ` +
    `${PREFIX === 'before' ? 'generic copy only, no server reason' : "desktop headline + plain sentence + the server's reason beneath, generic copy gone, Close only"}` +
    `; page banner behind the 01 modal carries the reason: ${pageHasReason})`,
  )
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
