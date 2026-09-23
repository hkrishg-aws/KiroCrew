/**
 * Screenshot harness for the Memory tab's "Summarize now" tally when a target is
 * a Temporary or Incognito session.
 *
 * Runs the REAL built SPA (website/dist) behind the shared in-process static
 * server with every /api/** answered from fixtures (gateway-free).
 *
 * The scenario: `api.sessions` lists three transcripts, one of them a Slack
 * thread the user marked `!incognito`. `POST /api/memory/consolidate` refuses that
 * one with `403 {code: "restricted_target_session"}` -- the refusal the memory
 * modes promise, not a failure of the press. The tally must count it as SKIPPED
 * and keep the success tone; with a fourth session whose request genuinely fails
 * (500), the failed tally names both counts and renders through ErrorNotice,
 * with the failed session named (title and key) under a label; with five failing sessions every
 * one is named, in the notice, in the DOM.
 *
 * Frames:
 *   <prefix>-1-skipped.png         2 ok + 1 refused  -> ok tone, "1 skipped: incognito session" (the mode the route named)
 *   <prefix>-2-failed-skipped.png  1 ok + 1 refused + 1 failed -> ErrorNotice banner: the failed count in the danger text, the skip on its own line in the success tone, "Failed: <key>", stays until dismissed
 *   <prefix>-3-failed-many.png     1 ok + 5 failed -> every failed session listed under the tally as plain text (title, key beside it; an untitled key named by its source -- channel stem or cron job -- the key beside it), "Retry 5 failed" under them, the dismiss control apart
 *   <prefix>-3-failed-many-retried.png  after "Retry 5 failed" (the 5 keys re-posted, nothing else) -> "Summarized 5/5 sessions" replaces the banner
 *   <prefix>-4-list-failed.png     `api.sessions` itself 503s -> ErrorNotice with a localized lead and the server's string on its own line under it, no key list, no per-session retry, never "no sessions to summarize"; "Try again" runs the whole press again (asserted, not shot: the outcome is frame 1's)
 * Every frame carries the helper line under the button (what "Summarize now"
 * does; the chats are left untouched).
 *
 * There is no hover frame: the list is complete in the DOM, so there is no
 * tooltip to hover (a native `title` would not render in a screenshot anyway --
 * it is browser chrome, not page content).
 *
 * Asserts as well as shoots. Pass `--expect-stale` to invert the assertion and
 * capture the BEFORE frame from a base-branch dist (`--dist <dir>`), where the
 * same refusal read "1 failed" in the danger tone.
 *
 * Usage: node scripts/capture-memory-consolidate-skipped.mjs [outDir] [prefix] [--dist <dir>] [--expect-stale]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { serveDist } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi, json } from './lib/stub-dashboard-api.mjs'

const positional = process.argv.slice(2).filter((a, i, all) => !a.startsWith('--') && all[i - 1] !== '--dist')
const OUT = positional[0] || '../temp-screenshots/memory-consolidate-skipped'
const PREFIX = positional[1] || 'after'
const distIdx = process.argv.indexOf('--dist')
const DIST = distIdx > -1 ? process.argv[distIdx + 1] : undefined
const EXPECT_STALE = process.argv.includes('--expect-stale')

mkdirSync(OUT, { recursive: true })

// The transcript stems `list_sessions` hands out. The Slack thread is named by
// its filename stem, exactly as the tab posts it; its `!incognito` flag lives in
// the session map, so no row-side `memory_mode` could have filtered it out.
const OK_KEYS = ['dashboard:chat-release-notes', 'dashboard:chat-perf-triage']
const INCOGNITO_STEM = 'slack_1785861252.833429'
const FAILING_KEY = 'dashboard:chat-flaky-span'
// Five more whose requests fail, for the many-failures frame: two dashboard
// chats, a Telegram thread stem, a Discord thread stem and a cron session --
// the key shapes the list really carries.
const FAILING_KEYS = [
  FAILING_KEY, 'dashboard:chat-mesh-dns', 'telegram_7781120043', 'discord_1418921033711624344', 'cron:nightly-digest',
]

// The route's own refusal body (api_memory_consolidate): the mode rides as a
// field, which is what lets the tally name it.
const REFUSAL = {
  error: 'Consolidation is not allowed for an incognito session: it leaves no durable memory.',
  code: 'restricted_target_session',
  mode: 'incognito',
}

// Titles as the sessions list carries them: a dashboard chat is titled, a channel
// thread or a cron session usually is not, so the banner must cope with both.
const TITLES = {
  'dashboard:chat-release-notes': 'Release notes draft',
  'dashboard:chat-perf-triage': 'Perf triage',
  'dashboard:chat-flaky-span': 'Flaky span investigation',
  'dashboard:chat-mesh-dns': 'Mesh DNS outage notes',
}
const session = (key) => ({ key, title: TITLES[key] ?? '', messages: 12, agent: 'kirocrew', memory_mode: 'persistent' })

// How the banner names an untitled key: the source the reader recognizes,
// from the key's namespace prefix, the raw key beside it. Every namespaced key
// is named -- the cron job as much as the channel stems.
const SOURCE_LABELS = {
  telegram_7781120043: 'Telegram chat 7781120043',
  discord_1418921033711624344: 'Discord conversation 1418921033711624344',
  'cron:nightly-digest': 'Cron job nightly-digest',
}

async function main() {
  const { srv, base } = await serveDist(DIST)
  const { LD_LIBRARY_PATH: _mise, ...browserEnv } = process.env
  const browser = await chromium.launch({ env: browserEnv })
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2 })
  const page = await context.newPage()

  // Mutable fixture state the harness flips between frames.
  let sessions = [...OK_KEYS, INCOGNITO_STEM].map(session)
  // Flipped off for the retry step, so the same keys then succeed.
  let failing = true
  // Flipped on for the last frame: the session list itself rejects.
  let listFailing = false
  const posted = []
  await stubDashboardApi(page, {
    folders: [], slots: [],
    extra: async (path, route) => {
      if (path === '/api/sessions') {
        if (listFailing) { await json(route, { error: 'sessions index unavailable' }, 503); return true }
        await json(route, { sessions }); return true
      }
      if (path === '/api/memory/consolidate') {
        const body = route.request().postDataJSON()
        posted.push(body.key)
        if (body.key === INCOGNITO_STEM) { await json(route, REFUSAL, 403); return true }
        if (failing && FAILING_KEYS.includes(body.key)) { await json(route, { error: 'consolidation retry backoff' }, 500); return true }
        await json(route, { ok: true, key: body.key }); return true
      }
      if (path === '/api/memory/settings') { await json(route, { history_idle_hours: 3, history_max_days: 90, migrated: true }); return true }
      if (path === '/api/memory/preferences' || path === '/api/memory/projects' || path === '/api/memory/history') {
        await json(route, { content: '' }); return true
      }
      if (path === '/api/lessons') { await json(route, { lessons: [] }); return true }
      if (path === '/api/memory/stores') {
        await json(route, { stores: [{ name: 'default', is_default: true, lineage: 'v1', exists: true }], active: 'default' }); return true
      }
      if (path === '/api/memory/retired') { await json(route, { retired: [] }); return true }
      if (path === '/api/memory/backups') { await json(route, { backups: [] }); return true }
      if (path === '/api/memory/carve') { await json(route, { counts: {} }); return true }
      if (path === '/api/memory/stats') { await json(route, { entries: 0, size_bytes: 0, provider: 'local' }); return true }
      if (path === '/api/memory/embedding-status') {
        await json(route, {
          enabled: true, provider: 'local', model_available: true, server_healthy: true,
          setup_step: 'ready', model_id: 'qwen3-embedding:0.6b', model_dim: 1024,
          model_source: 'default', model_path: '', reembed: { step: 'idle', done: 0, total: 0, error: '' },
        })
        return true
      }
      if (path.startsWith('/api/memory/')) { await json(route, {}); return true }
      return false
    },
  })
  logPageProblems(page)

  await page.goto(`${base}/settings/overview?view=memory`, { waitUntil: 'domcontentloaded' })
  const button = page.getByRole('button', { name: 'Summarize now' })
  await button.waitFor({ state: 'visible', timeout: 20_000 })
  // The card that holds the button: the tally renders beside it, and the card
  // is the frame a reader needs to place it.
  const card = button.locator('xpath=ancestor::*[contains(@class,"card-glow")][1]')

  async function press(name, expectText, { expectAlert, forbidText, expectNamed = [], expectNoKeys = false, expectSkipped, retry, retryList }) {
    posted.length = 0
    // The helper line sits under the button in every frame, before and after a
    // press: WHAT "Summarize now" writes into memory, and that the chats are
    // left alone.
    const help = page.getByTestId('summarize-now-help')
    const helpText = (await help.textContent()) ?? ''
    if (!helpText.includes('facts, decisions and lessons') || !helpText.includes('stay untouched')) throw new Error(`${name}: helper line does not say what the summary contains: ${JSON.stringify(helpText)}`)
    await button.click()
    const tally = page.getByText(expectText, { exact: false })
    await tally.first().waitFor({ state: 'visible', timeout: 15_000 })
    const shown = (await tally.first().textContent()) ?? ''
    const alerts = page.getByRole('alert')
    const alertCount = await alerts.count()
    const alertText = alertCount ? ((await alerts.first().textContent()) ?? '') : ''
    console.log(`${name}: posted ${JSON.stringify(posted)}; shows ${JSON.stringify(shown)}; alerts=${alertCount} ${JSON.stringify(alertText)}`)
    if (forbidText && shown.includes(forbidText)) throw new Error(`${name}: tally still reads ${JSON.stringify(forbidText)}`)
    if (expectAlert && alertCount === 0) throw new Error(`${name}: the failed tally did not render through ErrorNotice (no role="alert")`)
    if (!expectAlert && alertCount !== 0) throw new Error(`${name}: an error surface rendered for a refusal the user asked for`)
    if (expectSkipped) {
      // Beside a failure the skip is the tally's other half: its own line inside
      // the banner, in the success tone (the class and the computed colour --
      // the banner's own text is the danger colour), never part of the failed
      // count's danger text.
      const skip = alerts.first().getByTestId('consolidate-skipped')
      const skipText = ((await skip.textContent()) ?? '').trim()
      if (skipText !== expectSkipped) throw new Error(`${name}: the skip line reads ${JSON.stringify(skipText)}, not ${JSON.stringify(expectSkipped)}`)
      if (alertText.replace(skipText, '').includes('skipped')) throw new Error(`${name}: the failed count still names the skip: ${JSON.stringify(alertText)}`)
      const colours = await skip.evaluate(el => {
        const own = getComputedStyle(el).color
        const banner = getComputedStyle(el.closest('[role="alert"]')).color
        return { own, banner, classes: el.className }
      })
      console.log(`${name}: skip line ${JSON.stringify(colours)}`)
      if (!colours.classes.includes('text-ok') || colours.own === colours.banner) throw new Error(`${name}: the skip line renders in the banner's tone: ${JSON.stringify(colours)}`)
    }
    if (expectNoKeys) {
      // A rejected session list: nothing was posted, no session failed, so the
      // notice carries the server's string and no key list or per-session retry
      // -- and never the "no sessions to summarize" claim. Its one action is the
      // whole press again, inside the banner.
      if (posted.length) throw new Error(`${name}: posted ${JSON.stringify(posted)} although the list rejected`)
      if (await alerts.first().getByTestId('consolidate-failed-keys').count()) throw new Error(`${name}: a key list rendered for a rejected session list`)
      if (await alerts.first().getByRole('button', { name: /Retry \d+ failed/ }).count()) throw new Error(`${name}: a per-session retry rendered for a rejected session list`)
      if (!(await alerts.first().getByRole('button', { name: 'Try again' }).count())) throw new Error(`${name}: the rejected session list offers no way to try again`)
      if (await page.getByText('No sessions to summarize', { exact: false }).count()) throw new Error(`${name}: a rejected list read as nothing to summarize`)
      // The server's string is its own clause: on its own line under the bold
      // lead, in the mono font -- its box starts below the lead's box.
      const layout = await alerts.first().evaluate(el => {
        const lead = el.querySelector('strong')
        const server = lead?.nextElementSibling
        const a = lead?.getBoundingClientRect(), b = server?.getBoundingClientRect()
        return { lead: lead?.textContent, server: server?.textContent, serverClasses: server?.className, ownLine: !!(a && b && b.top >= a.bottom - 1) }
      })
      console.log(`${name}: list failure ${JSON.stringify(layout)}`)
      if (!layout.ownLine || !layout.serverClasses?.includes('font-mono')) throw new Error(`${name}: the server string does not sit on its own line under the lead: ${JSON.stringify(layout)}`)
    }
    if (expectNamed.length > 0) {
      // Every failed key in the DOM, under its label, and none behind a tooltip.
      const keys = alerts.first().getByTestId('consolidate-failed-keys')
      const listed = (await keys.textContent()) ?? ''
      for (const key of expectNamed) if (!listed.includes(key)) throw new Error(`${name}: the failed list does not name ${key}: ${JSON.stringify(listed)}`)
      // A titled session is named by its title, with the key beside it; an
      // untitled key by its source, the key beside it -- no row is a bare key.
      for (const key of expectNamed) {
        if (TITLES[key] && !listed.includes(TITLES[key])) throw new Error(`${name}: the failed list does not carry the title of ${key}`)
        if (SOURCE_LABELS[key] && !listed.includes(SOURCE_LABELS[key])) throw new Error(`${name}: the failed list does not name the source of ${key}: ${JSON.stringify(listed)}`)
        if (!TITLES[key] && !SOURCE_LABELS[key]) throw new Error(`${name}: fixture ${key} has neither a title nor an expected source label`)
      }
      const rows = await keys.getByRole('listitem').allTextContents()
      console.log(`${name}: rows ${JSON.stringify(rows)}`)
      for (const key of expectNamed) {
        const row = rows.find(r => r.endsWith(key))
        if (row === undefined || row === key) throw new Error(`${name}: ${key} renders as a bare key: ${JSON.stringify(row)}`)
      }
      // Plain text, not links: the names take the body colour, not the banner's
      // danger accent, nothing is underlined and no row holds a focusable element.
      const plain = await keys.evaluate(el => {
        const bannerColour = getComputedStyle(el.closest('[role="alert"]')).color
        const names = Array.from(el.querySelectorAll('li > span:first-child'))
        return {
          bannerColour,
          nameColours: [...new Set(names.map(n => getComputedStyle(n).color))],
          underlined: names.some(n => getComputedStyle(n).textDecorationLine.includes('underline')),
          pointer: names.some(n => getComputedStyle(n).cursor === 'pointer'),
          focusable: el.querySelectorAll('a, button, [tabindex]').length,
        }
      })
      console.log(`${name}: rows ${JSON.stringify(plain)}`)
      if (plain.nameColours.includes(plain.bannerColour) || plain.underlined || plain.pointer || plain.focusable) throw new Error(`${name}: the failed rows read as links: ${JSON.stringify(plain)}`)
      if (!alertText.includes('Failed:')) throw new Error(`${name}: the key list carries no label`)
      if (await alerts.first().locator('[title]').count()) throw new Error(`${name}: part of the list rides a title tooltip`)
      // The dismiss control is the banner's own, beside the text block -- not
      // the element right after the last key (the retry button IS a sibling of
      // the list, by design; the dismiss control must not be).
      const adjacent = await keys.evaluate(el => {
        const isDismiss = (n) => n?.tagName === 'BUTTON' && n.getAttribute('aria-label') === 'Dismiss'
        return isDismiss(el.nextElementSibling) || Array.from(el.parentElement?.children ?? []).some(isDismiss)
      })
      if (adjacent) throw new Error(`${name}: the dismiss control sits right against the key list`)
    }
    await page.mouse.move(40, 40)
    await card.screenshot({ path: `${OUT}/${PREFIX}-${name}.png` })
    // The success tally clears itself after 4 s; the failure notice stays until
    // dismissed. Wait it out either way so the next press starts clean.
    await page.waitForTimeout(4_500)
    if (expectAlert) {
      if ((await alerts.count()) === 0) throw new Error(`${name}: the failure notice vanished on its own`)
      if (retryList) {
        // "Try again" on a rejected list is the whole press again: the list
        // fetched once more and, with the fixture answering, every listed
        // session posted; the outcome replaces the banner. Asserted, not shot --
        // the outcome is the first frame's.
        posted.length = 0
        listFailing = false
        await alerts.first().getByRole('button', { name: 'Try again' }).click()
        const again = page.getByText(retryList.expectText, { exact: false })
        await again.first().waitFor({ state: 'visible', timeout: 15_000 })
        console.log(`${name}: try again posted ${JSON.stringify(posted)}; shows ${JSON.stringify(await again.first().textContent())}`)
        const expected = [...retryList.expectPosted].sort()
        if (JSON.stringify([...posted].sort()) !== JSON.stringify(expected)) throw new Error(`${name}: try again did not post the listed sessions: ${JSON.stringify(posted)}`)
        if (await alerts.count()) throw new Error(`${name}: the failure notice outlived a successful try again`)
        await page.waitForTimeout(4_500)
        return
      }
      if (retry) {
        // "Try again" re-posts the failed keys and nothing else; with the
        // fixture flipped to succeed, the outcome replaces the banner.
        posted.length = 0
        failing = false
        await alerts.first().getByRole('button', { name: `Retry ${expectNamed.length} failed` }).click()
        const retried = page.getByText(retry.expectText, { exact: false })
        await retried.first().waitFor({ state: 'visible', timeout: 15_000 })
        console.log(`${name}: retry posted ${JSON.stringify(posted)}; shows ${JSON.stringify(await retried.first().textContent())}`)
        const expected = [...retry.expectPosted].sort()
        if (JSON.stringify([...posted].sort()) !== JSON.stringify(expected)) throw new Error(`${name}: the retry did not post exactly the failed keys: ${JSON.stringify(posted)}`)
        if (await alerts.count()) throw new Error(`${name}: the failure notice outlived a successful retry`)
        await page.mouse.move(40, 40)
        await card.screenshot({ path: `${OUT}/${PREFIX}-${name}-retried.png` })
        await page.waitForTimeout(4_500)
        failing = true
        return
      }
      await alerts.first().getByRole('button', { name: 'Dismiss' }).click()
      if (await alerts.count()) throw new Error(`${name}: the failure notice did not dismiss`)
    }
  }

  if (EXPECT_STALE) {
    // The base branch counted the refusal as a failure, in the danger tone, on
    // every press -- the regression this harness exists to disprove.
    await press('1-skipped', '(1 failed)', { expectAlert: false, forbidText: 'skipped' })
  } else {
    await press('1-skipped', '(1 skipped: incognito session)', { expectAlert: false, forbidText: 'failed' })
    sessions = [OK_KEYS[0], INCOGNITO_STEM, FAILING_KEY].map(session)
    await press('2-failed-skipped', '1/3 sessions (1 failed)', { expectAlert: true, expectNamed: [FAILING_KEY], expectSkipped: '1 skipped: incognito session' })
    sessions = [OK_KEYS[0], ...FAILING_KEYS].map(session)
    await press('3-failed-many', '1/6 sessions (5 failed)', {
      expectAlert: true, expectNamed: FAILING_KEYS,
      retry: { expectText: 'Summarized 5/5 sessions', expectPosted: FAILING_KEYS },
    })
    // The list the press would load once it can: frame 1's three sessions, so
    // "Try again" ends in frame 1's tally.
    sessions = [...OK_KEYS, INCOGNITO_STEM].map(session)
    listFailing = true
    await press('4-list-failed', 'Could not list the sessions to summarize', {
      expectAlert: true, expectNoKeys: true,
      retryList: { expectText: '(1 skipped: incognito session)', expectPosted: [...OK_KEYS, INCOGNITO_STEM] },
    })
    listFailing = false
  }

  await browser.close()
  srv.close()
  console.log(`wrote ${OUT}/${PREFIX}-*.png`)
}

main().catch(err => { console.error(err); process.exit(1) })
