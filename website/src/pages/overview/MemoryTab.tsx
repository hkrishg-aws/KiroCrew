import { useState, useEffect, useCallback, useMemo, useRef, type ReactNode } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Trans } from 'react-i18next'
import { XCircle, CheckCircle, RefreshCw, Hourglass, Check, BookOpen, SlidersHorizontal } from 'lucide-react'
import { api } from '../../api/client'
import { Card, CardTitle, Btn, SendBtn, Input, Badge, EmptyState, Skeleton } from '../../components/ui'
import InfoTip from '../../components/InfoTip'
import SimpleSelect from '../../components/SimpleSelect'
import { esc } from '../../api/helpers'
import { channelBrandLabel, isLegacySlackSlotKey } from '../../utils/channelOrigin'
import VectorMemoryCard from './VectorMemoryCard'
import EmbeddingModelCard from './EmbeddingModelCard'
import MemoryStoreCard, {
  MEMORY_QUERY_PREFIXES,
  MemoryScopeNotice,
  useMemoryStores,
} from './MemoryStoreCard'
import MemoryCarveCard from './MemoryCarveCard'
import MemoryRetiredCard from './MemoryRetiredCard'
import MemoryBackupsCard from './MemoryBackupsCard'
import MemberMemoryPanel from './MemberMemoryPanel'
import MemoryRecordsEditor from './MemoryRecordsEditor'
import MemoryDocCard from './MemoryDocCard'
import Modal from '../../components/Modal'
import { useConfirm } from '../../components/ConfirmDialog'
import ErrorNotice from '../../components/ErrorNotice'
import { useSidePanelLeaveGuard } from '../../components/SidePanelLayout'
import { useGuardedLeave } from '../../components/NavigationLeaveGuard'
import type { Lesson, SessionInfo } from '../../types'
import { useSortableTable } from '../../hooks/useSortableTable'
import SortableHeader from '../../components/SortableHeader'

import { i18nT } from '../../i18n/t'
import { compareText, fmtDateTimeNumeric } from '../../i18n/format'

/** `POST /api/memory/consolidate`'s code for a target the memory modes promise
 *  leaves no durable trace (a Temporary or Incognito session). "Summarize now"
 *  posts every stem `api.sessions` lists, those sessions included, and a row's
 *  `memory_mode` cannot pre-filter them all: a channel thread flagged before its
 *  transcript header carried the mode holds the flag in the session map alone,
 *  which the list is not built from. So the tally sorts the refusals after the
 *  fact -- a skip the user asked for by choosing the mode, counted apart from a
 *  request that failed. */
const RESTRICTED_TARGET_CODE = 'restricted_target_session'

/** Machine-readable fields of a rejected consolidate call's JSON body: the
 *  backend `code`, and for a refused target the `mode` it was in. Duck-typed on
 *  `body` rather than `instanceof ApiError` (the `MobileConnectModal` shape), so
 *  it keeps working under a mocked `api/client`. */
const consolidateRefusal = (reason: unknown): { code?: string; mode?: string } => {
  const body = typeof reason === 'object' && reason !== null && 'body' in reason && typeof reason.body === 'string'
    ? reason.body.trim()
    : ''
  if (!body.startsWith('{')) return {}
  try {
    const parsed = JSON.parse(body) as { code?: unknown; mode?: unknown }
    return {
      code: typeof parsed.code === 'string' && parsed.code ? parsed.code : undefined,
      mode: typeof parsed.mode === 'string' && parsed.mode ? parsed.mode : undefined,
    }
  } catch {
    return {}
  }
}

/** The mode a skipped session was in, as the tally names it when every skipped
 *  session shares one ("1 skipped: incognito session"): the reader could not tell
 *  whether "temporary or incognito" was two kinds of private chat or one thing
 *  with two names. A literal map, indexed, so every key stays greppable. */
const SKIPPED_MODE_KEYS = {
  incognito: 'pages.overview.memoryTab.skipped_incognito_session',
  temporary: 'pages.overview.memoryTab.skipped_temporary_session',
} as const

/** One consolidation target as the tally names it: the key the route is posted
 *  for, and the title `api.sessions` returned for it (`""` when it has none). */
type SummarizeTarget = { key: string; title: string }

/** The failure surface's two shapes, told apart by `kind` so the render site
 *  cannot mix their affordances. A failed TALLY: `message` is the failed count
 *  (the danger half), `skipped` the skip fragment when any target was refused
 *  by its mode (rendered in the success tone it has on its own -- the same
 *  words read as a failure inside the banner's danger text), and `failed` the
 *  sessions the reader can act on. A failed session LIST (`api.sessions`
 *  itself rejected, so nothing was posted): `message` is the server's own
 *  string, verbatim -- the journal lookup key ErrorNotice recovers the
 *  endpoint/status/code report by, so it is kept raw rather than localized --
 *  and the one action is the whole press again. */
type ConsolidateFailure =
  | { kind: 'tally'; message: string; skipped?: string; failed: SummarizeTarget[] }
  | { kind: 'list'; message: string }

/** How an untitled failed session is NAMED, from its key's namespace prefix
 *  (`telegram_7781120043`, `cron:nightly-digest`; the live `<ns>:<id>` form and
 *  the transcript stem `<ns>_<id>` fold to the same prefix). A stem is a
 *  filename, not a name: the source the reader recognizes leads ("Telegram
 *  chat 7781120043", "Cron job nightly-digest") and the raw key rides beside
 *  it in the mono font, exactly as a titled row carries its key. EVERY
 *  namespaced key gets a label -- a bare `cron_nightly-digest` in a list of
 *  named rows read as an unexplained chip: the three channels below have their
 *  own copy, every other channel in the shared brand roster
 *  (`utils/channelOrigin.ts`, mirroring the backend's namespace list) is named
 *  by its brand through one translatable pattern, the non-channel families the
 *  gateway mints (`cron:`, `hook:`/`webhook:`, `app:`, `subagent:`,
 *  `dashboard:`) have copy of their own, and a prefix none of those claim is
 *  still named by that prefix rather than left bare. A legacy bare Slack ts
 *  (`1785861252.833429`) is a Slack thread. Only a key with NO namespace --
 *  which the gateway never mints; test fixtures do -- stands alone. Flat
 *  records of full literal keys, indexed inline at the `i18nT()` call, which
 *  is the shape `scripts/check-i18n-keys.mjs` resolves statically. */
const CHANNEL_SOURCE_KEYS = {
  telegram: 'pages.overview.memoryTab.source_telegram_chat',
  slack: 'pages.overview.memoryTab.source_slack_thread',
  discord: 'pages.overview.memoryTab.source_discord_conversation',
} as const

const FAMILY_SOURCE_KEYS = {
  cron: 'pages.overview.memoryTab.source_cron_job',
  hook: 'pages.overview.memoryTab.source_webhook',
  webhook: 'pages.overview.memoryTab.source_webhook',
  app: 'pages.overview.memoryTab.source_app_session',
  subagent: 'pages.overview.memoryTab.source_subagent',
  dashboard: 'pages.overview.memoryTab.source_dashboard_session',
} as const

const hasOwn = (table: object, key: string): boolean => Object.prototype.hasOwnProperty.call(table, key)

const channelSourceLabel = (key: string): string | undefined => {
  if (isLegacySlackSlotKey(key)) return i18nT(CHANNEL_SOURCE_KEYS.slack, { id: key })
  const match = /^([a-z][a-z0-9-]*)[_:](.+)$/.exec(key)
  if (!match) return undefined
  const [, family, id] = match
  if (hasOwn(CHANNEL_SOURCE_KEYS, family)) {
    return i18nT(CHANNEL_SOURCE_KEYS[family as keyof typeof CHANNEL_SOURCE_KEYS], { id })
  }
  if (hasOwn(FAMILY_SOURCE_KEYS, family)) {
    return i18nT(FAMILY_SOURCE_KEYS[family as keyof typeof FAMILY_SOURCE_KEYS], { id })
  }
  const brand = channelBrandLabel(family)
  if (brand) return i18nT('pages.overview.memoryTab.source_channel_chat', { channel: brand, id })
  return i18nT('pages.overview.memoryTab.source_other_session', { source: family, id })
}

/** The failed tally's footer: the skip fragment first, when any target was
 *  refused by its mode -- the other half of the tally, in the success tone and
 *  with the check it has when it stands alone, because the same words in the
 *  banner's danger text read as one more thing gone wrong -- then the failed
 *  sessions, named in full under a label: a UI-font label first, so a bare key
 *  does not read as an unexplained chip, then one line per session -- its
 *  TITLE, the name the user knows it by, with the key beside it in the mono
 *  font for the one who has to find the transcript. A channel stem the list
 *  carries untitled is named by its source instead ("Telegram chat
 *  7781120043"), the stem beside it the same way. Every session is in the
 *  DOM. A truncated list with the tail behind a hover tooltip hid it from the
 *  keyboard, from assistive tech and from a screenshot alike; the lines wrap
 *  instead, inside the banner. The label is one `<Trans>` sentence with the
 *  list as its component, so a translation can place the list where its
 *  grammar wants.
 *
 *  Under the list, the one action a failed key admits: "Retry N failed"
 *  re-posts the failed keys and nothing else, so the sessions that already
 *  summarized are not billed a second turn. The shared `Btn` in its danger
 *  tone, inside the banner where the keys are, not in the button row above.
 *
 *  The rows are PLAIN TEXT and look it: the name in the body color
 *  (`text-text`), the key in the muted mono, nothing underlined, no pointer,
 *  no hover, no element that can take focus. Left to inherit the banner's
 *  danger accent, a list of session names read as a list of links -- an
 *  accent-colored name in a list is the affordance -- and nothing here
 *  navigates: the row names a transcript so the reader can find it, the label
 *  above carries the tone, and the only control is the retry button below. */
function FailedSessions({ failed, skipped, onRetry }: { failed: readonly SummarizeTarget[]; skipped?: string; onRetry: () => void }) {
  return (
    <div>
      {skipped && (
        <div className="text-ok" data-testid="consolidate-skipped">
          <CheckCircle className="lucide-inline" /> {skipped}
        </div>
      )}
      <Trans
        i18nKey="pages.overview.memoryTab.failed_sessions_named"
        components={{
          keys: (
            <ul className="mt-0.5 list-none space-y-0.5 pl-0 text-text" data-testid="consolidate-failed-keys">
              {failed.map(f => {
                const name = f.title || channelSourceLabel(f.key)
                return (
                  <li key={f.key} className="flex flex-wrap items-baseline gap-x-2">
                    {name
                      ? <><span className="text-text">{name}</span><span className="font-mono text-[12px] text-muted">{f.key}</span></>
                      : <span className="font-mono text-text">{f.key}</span>}
                  </li>
                )
              })}
            </ul>
          ),
        }}
      />
      <Btn danger onClick={onRetry} className="mt-1.5 py-0.5 text-[12px]">
        <RefreshCw className="lucide-inline" /> {i18nT('pages.overview.memoryTab.retry_failed_sessions', { count: failed.length })}
      </Btn>
    </div>
  )
}

/** The rejected session list's footer: "Try again" runs the whole press again
 *  -- the list re-fetched, then the summarize -- from inside the banner that
 *  reported the failure. Without it the banner dead-ended: a server string
 *  and a dismiss control, with the way out being the button above, which
 *  nothing in the banner pointed at. The same danger-tone `Btn` as the
 *  per-session retry, so the two failure shapes offer their one action in the
 *  same place. */
function RetrySessionList({ onRetry }: { onRetry: () => void }) {
  return (
    <Btn danger onClick={onRetry} className="mt-1.5 py-0.5 text-[12px]">
      <RefreshCw className="lucide-inline" /> {i18nT('pages.overview.memoryTab.retry_session_list')}
    </Btn>
  )
}

/** The Scope cell. The three values are the three delete selectors the list
 *  reports, and each must read differently: a fragment is that scope's row;
 *  `""` is the global row, labelled rather than left blank so it does not read
 *  as missing data beside a scoped sibling; `null` is a row whose stored scope
 *  the store cannot use, labelled so the reader can see that its Delete is the
 *  one that reaches every scope. */
function scopeCell(lesson: Lesson) {
  const scope = lesson.repo_scope
  const repo = scope === null
    ? <span className="text-muted italic" title={i18nT('pages.overview.memoryTab.scope_unusable_hint')}>{i18nT('pages.overview.memoryTab.scope_unusable')}</span>
    : !scope
      ? <span className="text-muted">{i18nT('pages.overview.memoryTab.scope_global')}</span>
      : <span className="font-mono break-all">{scope}</span>
  // The JSONL tier is the other half of the row's identity: a same-text row in
  // the active workspace's file and one in the global file would otherwise read
  // alike, and their Deletes go to different files.
  if (lesson.scope !== 'workspace' || !lesson.workspace) return repo
  return <>{repo}<span className="block text-[12px] text-muted">{i18nT('pages.overview.memoryTab.scope_workspace', { name: lesson.workspace })}</span></>
}

export default function MemoryTab({ refreshTrigger, selectedStore, onStoreNavigate }: { refreshTrigger: number; selectedStore?: string; onStoreNavigate?: (store: string) => void }) {
  const stores = useMemoryStores()
  const navigate = useNavigate()
  const leave = useGuardedLeave()
  const queryClient = useQueryClient()
  useEffect(() => {
    if (refreshTrigger) for (const prefix of MEMORY_QUERY_PREFIXES) void queryClient.invalidateQueries({ queryKey: prefix })
  }, [refreshTrigger, queryClient])
  const [store, setStore] = useState(() => {
    const selected = new URLSearchParams(window.location.search).get('store') || ''
    return selected === 'default' ? '' : selected
  })
  useEffect(() => {
    if (selectedStore !== undefined) setStore(selectedStore === 'default' ? '' : selectedStore)
  }, [selectedStore])
  const [dirty, setDirty] = useState(false)
  useSidePanelLeaveGuard(() => !dirty || window.confirm(i18nT('memoryV2.leave_discard_explanation')), dirty)
  useEffect(() => {
    if (!dirty) return
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])
  const [pendingStore, setPendingStore] = useState<string | null>(null)
  const selected = stores.data?.stores.find(s => s.name === store)
  // Content may already be cached while its catalog is still loading. Never
  // turn a private directory name into a member identity, or expose actions
  // until the authoritative owner row arrives. React Query retains settled
  // catalog data during refresh, so existing editors keep their drafts mounted.
  const identityReady = !!selected && (
    selected.is_default
    || (!selected.owner_member && (selected.memory_version === 1 || (selected.memory_version == null && selected.lineage === 'v1')))
    || (selected.memory_version === 2 && !!selected.owner_member)
  )
  const applyStore = (next: string) => {
    setStore(next)
    if (onStoreNavigate) { onStoreNavigate(next); return }
    const url = new URL(window.location.href)
    if (next) url.searchParams.set('store', next)
    else url.searchParams.delete('store')
    window.history.replaceState(window.history.state, '', url)
  }
  const choose = (next: string) => {
    if (next === store) return
    if (dirty) setPendingStore(next)
    else applyStore(next)
  }
  return <>
    <MemoryStoreCard store={store} onStoreChange={choose} compact={!!store} />
    {store ? identityReady ? <MemberMemoryPanel key={store} store={store} summary={selected!} onDirtyChange={setDirty} /> : <Card>
      {stores.isPending ? <div role="status" aria-busy="true" className="flex min-w-0 items-center gap-3">
        <Skeleton className="h-12 w-12 shrink-0 motion-reduce:animate-none" />
        <div className="min-w-0 flex-1 space-y-2">
          <p className="text-[13px] text-muted">{i18nT('memoryV2.identity_loading')}</p>
          <Skeleton className="h-4 w-48 max-w-full motion-reduce:animate-none" />
          <Skeleton className="h-3 w-32 max-w-full motion-reduce:animate-none" />
        </div>
      </div> : <>
        <CardTitle>{i18nT('memoryV2.identity_unavailable')}</CardTitle>
        <MemoryScopeNotice error={stores.error} />
        <div className="flex flex-wrap gap-2">
          <Btn className="min-h-11" disabled={stores.isFetching} onClick={() => void stores.refetch()}>{i18nT('memoryV2.retry_identity')}</Btn>
          <Btn className="min-h-11" onClick={() => {
            const destination = selected?.owner_member
              ? `/capabilities?tab=crews&crew=${encodeURIComponent(selected.owner_member)}`
              : '/capabilities?tab=crews'
            leave(() => navigate(destination), destination)
          }}>{i18nT('pages.kiroCrewAgentsPage.open_crew_manager')}</Btn>
        </div>
      </>}
    </Card> : <GlobalMemoryTab refreshTrigger={refreshTrigger} onDirtyChange={setDirty} />}
    {pendingStore !== null && <Modal open title={i18nT('memoryV2.discard_title')} onClose={() => setPendingStore(null)}><div className="flex flex-col gap-3">
      <p className="text-[13px]">{i18nT('memoryV2.discard_explanation')}</p>
      <div className="flex flex-wrap gap-2">
        <Btn onClick={() => setPendingStore(null)}>{i18nT('pages.kiroCrewAgentsPage.keep_editing')}</Btn>
        <Btn danger onClick={() => { applyStore(pendingStore); setPendingStore(null); setDirty(false) }}>{i18nT('memoryV2.discard_title')}</Btn>
      </div>
    </div></Modal>}
  </>
}

function GlobalMemoryTab({ refreshTrigger, onDirtyChange }: { refreshTrigger: number; onDirtyChange?: (dirty: boolean) => void }) {
  const queryClient = useQueryClient()
  const [recordDirty, setRecordDirty] = useState(false)
  const [recordsOpen, setRecordsOpen] = useState(false)
  const [docDirty, setDocDirty] = useState<Record<string, boolean>>({})
  const dirty = recordDirty || Object.values(docDirty).some(Boolean)
  useEffect(() => { onDirtyChange?.(dirty); return () => onDirtyChange?.(false) }, [dirty, onDirtyChange])
  /** The memory store every store-aware card on this page reads, ON THE WIRE.
   *
   *  `''` means no store is NAMED, which the gateway resolves to the global store —
   *  what every one of these routes served before the picker existed. It is
   *  deliberately not spelled `'default'`: the parameter's PRESENCE is what takes
   *  the owner gate, so naming the store the page already reads would gate a read
   *  that needs no gate and refuse the whole page on an install with no configured
   *  owner. `MemoryStoreCard` displays the active store while this stays `''`. */
  const store = ''
  const stores = useMemoryStores()
  /** Look the row up under the store being SHOWN, not the wire value: `''` matches
   *  no row, so keying on it would make every "is this store readable" answer
   *  default to yes for the store the page is actually displaying. */
  const shownStore = store || stores.data?.active || ''
  const selectedStore = stores.data?.stores.find(s => s.name === shownStore)
  /** A store whose file could not be read has nothing to list. Backups are the
   *  exception and stay visible: a missing database is exactly when a restore is
   *  the thing the operator came for. */
  const storeReadable = selectedStore?.exists !== false

  const [lessons, setLessons] = useState<Lesson[]>([]); const [rule, setRule] = useState(''); const [cat, setCat] = useState('knowledge')
  const [lessonFeedback, setLessonFeedback] = useState<{
    tone: 'info' | 'warning' | 'error'
    text: string
  } | null>(null)
  const [idleHours, setIdleHours] = useState(3); const [maxDays, setMaxDays] = useState(90); const [settingsSaved, setSettingsSaved] = useState(false)
  const [migrated, setMigrated] = useState(false)
  const [vectorActive, setVectorActive] = useState(false)
  const [consolidating, setConsolidating] = useState(false)
  const [consolidateMsg, setConsolidateMsg] = useState<ReactNode>('')
  const [consolidateOk, setConsolidateOk] = useState(false)
  // The failure notice, apart from the status message: it reports rejected
  // requests, so it renders through ErrorNotice like every other failure. Its
  // two shapes are `ConsolidateFailure`'s, told apart by `kind` at the render
  // site: the failed tally with its skip fragment and named sessions, or the
  // rejected session list with the server's string and the whole press as its
  // one action.
  const [consolidateError, setConsolidateError] = useState<ConsolidateFailure | null>(null)
  // Track all "Saved" / "consolidate-msg-clear" timeout ids so they can be
  // cleared on unmount — otherwise a pending setTimeout fires after the
  // component is gone and (in vitest) shows up as an unhandled error from
  // "tasks running past test environment teardown".
  const timeoutsRef = useRef<ReturnType<typeof setTimeout>[]>([])
  useEffect(() => () => {
    timeoutsRef.current.forEach(clearTimeout)
    timeoutsRef.current = []
  }, [])
  const scheduleClear = useCallback((fn: () => void, ms: number) => {
    const id = setTimeout(() => {
      timeoutsRef.current = timeoutsRef.current.filter(t => t !== id)
      fn()
    }, ms)
    timeoutsRef.current.push(id)
  }, [])
  const loadLessons = useCallback(async () => { const d = await api.lessons(); setLessons(d.lessons || []) }, [])
  const { confirm, confirmDialog } = useConfirm()
  // Which step of a delete failed decides the banner's title: the request
  // itself (the row is still stored) or the list refresh after it succeeded
  // (the row is gone but may still be shown).
  const [deleteError, setDeleteError] = useState<{ step: 'delete' | 'refresh' | 'nothing'; message?: string } | null>(null)
  // A `null` scope is the one row whose Delete cannot be limited to itself: the
  // route refuses the stored value as a selector, so the client sends none and
  // the unselective delete removes every same-rule row in every scope. That is
  // the collateral this tab otherwise exists to prevent, so it asks first --
  // through the shared themed dialog, whose confirm button restates the act.
  const deleteLesson = async (l: Lesson) => {
    if (l.repo_scope === null && !(await confirm({
      title: i18nT('pages.overview.memoryTab.delete_unusable_scope_title'),
      body: i18nT('pages.overview.memoryTab.delete_unusable_scope_confirm'),
      confirmLabel: i18nT('pages.overview.memoryTab.delete_unusable_scope_button'),
    }))) return
    setDeleteError(null)
    // Both steps are awaited and reported where the row is, rather than letting
    // the click end in silence: a rejected delete leaves the row stored, and a
    // rejected refresh leaves a deleted row on screen.
    // `exact`: this row holds the whole rule, so it names exactly one row; the
    // route's default substring match would also take every longer rule that
    // contains it.
    let result: { ok: boolean }
    try {
      result = await api.deleteLesson(l.rule, l.repo_scope, { scope: l.scope, workspace: l.workspace, exact: true })
    } catch (e) {
      setDeleteError({ step: 'delete', message: e instanceof Error ? e.message : String(e) })
      return
    }
    if (!result?.ok) {
      // The store found no row matching these selectors: the list is stale, or
      // the displayed (redacted) text differs from the stored one.
      setDeleteError({ step: 'nothing' })
    }
    try {
      await loadLessons()
    } catch (e) {
      setDeleteError({ step: 'refresh', message: e instanceof Error ? e.message : String(e) })
    }
  }
  const lessonComparators = useMemo(() => ({
    rule: (a: Lesson, b: Lesson) => a.rule.localeCompare(b.rule),
    category: (a: Lesson, b: Lesson) => a.category.localeCompare(b.category),
    repo_scope: (a: Lesson, b: Lesson) => compareText(a.repo_scope ?? '', b.repo_scope ?? ''),
    ts: (a: Lesson, b: Lesson) => new Date(a.ts).getTime() - new Date(b.ts).getTime(),
  }), [])
  const recentLessons = useMemo(() => lessons.slice(-20), [lessons])
  const { sorted: sortedLessons, sort: lessonSort, toggle: toggleLessonSort } = useSortableTable(recentLessons, 'memory-lessons', lessonComparators, { key: 'ts', dir: 'desc' })
  useEffect(() => {
    api.memorySettings().then(d => { setIdleHours(d.history_idle_hours ?? 3); setMaxDays(d.history_max_days ?? 90); setMigrated(d.migrated ?? false) })
    loadLessons()
  }, [loadLessons])
  // The page's own refresh signal. Invalidated by query-key PREFIX rather than
  // for the selected store only, so the rows cached for a store the user looked
  // at earlier cannot outlive the refresh and reappear on the next switch.
  useEffect(() => {
    loadLessons()
    for (const prefix of MEMORY_QUERY_PREFIXES) {
      queryClient.invalidateQueries({ queryKey: prefix })
    }
  }, [refreshTrigger, loadLessons, queryClient])
  // One pass over `targets`: the button's press runs it over every listed
  // session, the failure banner's "Retry N failed" over the ones that failed
  // alone, so a retry never re-bills the sessions that already summarized.
  // Each target carries the title `api.sessions` returned, so the banner can
  // name a failed session by the name the user knows it by.
  const summarizeTargets = async (targets: SummarizeTarget[]) => {
    setConsolidating(true); setConsolidateMsg(''); setConsolidateOk(false); setConsolidateError(null)
    const results = await Promise.allSettled(targets.map(t => api.consolidateMemory(t.key, true)))
    const succeeded = results.filter(r => r.status === 'fulfilled').length
    // Sort every rejection by the TARGET it was for, so the failed tally can
    // name its sessions: the counts alone leave the user unable to act on a
    // failure. A refusal is sorted by the MODE the route named, so the tally
    // can say which one when every skip shares it.
    const failedTargets: SummarizeTarget[] = []
    const skippedModes = new Set<string>()
    let skipped = 0
    results.forEach((r, i) => {
      if (r.status !== 'rejected') return
      const refusal = consolidateRefusal(r.reason)
      if (refusal.code === RESTRICTED_TARGET_CODE) { skipped += 1; skippedModes.add(refusal.mode ?? '') }
      else failedTargets.push(targets[i])
    })
    const failed = failedTargets.length
    // The one mode every skipped session was in, or nothing: several modes, or
    // a body that names none, keep the either/or wording.
    const [onlyMode] = skippedModes.size === 1 ? skippedModes : []
    const mode = onlyMode === 'incognito' || onlyMode === 'temporary'
      ? i18nT(SKIPPED_MODE_KEYS[onlyMode], { count: skipped })
      : undefined
    const tally = { succeeded, total: targets.length, failed, skipped, mode }
    // The skip fragment as it reads on its own: the mode when every skip shares
    // one, the either/or wording otherwise. Beside a failure it is rendered
    // apart from the failed count, in its own tone -- the two are different
    // outcomes and one string gave them one colour.
    const skippedFragment = skipped > 0
      ? (mode
        ? i18nT('pages.overview.memoryTab.skipped_sessions_mode', tally)
        : i18nT('pages.overview.memoryTab.skipped_sessions', tally))
      : undefined
    if (failed > 0) {
      // Persistent until dismissed: a failure the user has to act on must not
      // vanish on a timer the way the success tally does.
      setConsolidateError({
        kind: 'tally',
        message: i18nT('pages.overview.memoryTab.consolidated_sessions_failed', tally),
        skipped: skippedFragment,
        failed: failedTargets,
      })
    } else if (skipped > 0) {
      setConsolidateMsg(<><CheckCircle className="lucide-inline" /> {mode
        ? i18nT('pages.overview.memoryTab.consolidated_sessions_skipped_mode', tally)
        : i18nT('pages.overview.memoryTab.consolidated_sessions_skipped', tally)}</>); setConsolidateOk(true)
    } else {
      // The same n/m shape as every other tally: "Summarized 5 sessions" beside
      // "Summarized 1/2 sessions" left the reader wondering if some were left out.
      setConsolidateMsg(<><CheckCircle className="lucide-inline" /> {i18nT('pages.overview.memoryTab.consolidated_sessions_all', tally)}</>); setConsolidateOk(true)
    }
    setConsolidating(false)
    if (failed === 0) scheduleClear(() => setConsolidateMsg(''), 4000)
  }
  const consolidate = async () => {
    setConsolidating(true); setConsolidateMsg(''); setConsolidateOk(false); setConsolidateError(null)
    // A rejected session list is a failure, reported as one: nothing was
    // posted, so "no sessions to summarize" would claim a state the request
    // never established. The server's string is the message (the journal key),
    // under a localized lead; the list is empty so the footer has nothing to
    // name, and its one action is this press again, offered in the banner.
    let sessions: { sessions?: SessionInfo[] }
    try {
      sessions = await api.sessions(200)
    } catch (e) {
      setConsolidateError({ kind: 'list', message: e instanceof Error ? e.message : String(e) })
      setConsolidating(false)
      return
    }
    const targets: SummarizeTarget[] = (sessions?.sessions || [])
      .filter((s: SessionInfo) => Boolean(s.key))
      .map((s: SessionInfo) => ({ key: s.key, title: s.title?.trim() || '' }))
    if (targets.length === 0) { setConsolidateMsg(<><XCircle className="lucide-inline" /> {i18nT('pages.overview.memoryTab.no_sessions_to_consolidate_start_a_chat_first')}</>); setConsolidating(false); return }
    await summarizeTargets(targets)
  }
  const addLesson = async () => {
    if (!rule) return
    setLessonFeedback(null)
    const result = await api.createLesson(rule, cat)
    if (result.outcome === 'inserted' || result.outcome === 'enriched') {
      setRule('')
      await loadLessons()
      return
    }
    if (result.outcome === 'unchanged') {
      setRule('')
      setLessonFeedback({
        tone: 'info',
        text: i18nT('pages.overview.memoryTab.lesson_already_stored'),
      })
      return
    }
    if (result.outcome === 'deduped') {
      setLessonFeedback({
        tone: 'warning',
        text: i18nT('pages.overview.memoryTab.lesson_already_covered', {
          reason: result.reason,
        }),
      })
      return
    }
    setLessonFeedback({
      tone: 'error',
      text: i18nT('pages.overview.memoryTab.lesson_not_saved', {
        reason: result.reason,
      }),
    })
  }
  return (<>
    <Card><CardTitle>{i18nT('pages.overview.memoryTab.memory_settings')} <InfoTip text={i18nT('pages.overview.memoryTab.controls_how_conversation_history_is_consolidate')} /></CardTitle>
      <div className="flex gap-3 items-end flex-wrap">
        <label htmlFor="memory-idle-hours" className="flex flex-col gap-1 text-[13px] text-muted">
          <span>{i18nT('pages.overview.memoryTab.consolidation_idle_hours')}</span>
          <input id="memory-idle-hours" aria-label={i18nT('pages.overview.memoryTab.consolidation_idle_hours')} type="number" min={0.5} max={24} step={0.5} className="w-24 bg-bg-elevated border border-border rounded-md px-3 py-2 text-text text-sm font-body outline-hidden transition-colors focus-ring" value={idleHours} onChange={e => setIdleHours(Number(e.target.value))} />
        </label>
        {!migrated && (
          <label htmlFor="memory-max-days" className="flex flex-col gap-1 text-[13px] text-muted">
            <span>{i18nT('pages.overview.memoryTab.history_retention_days')}</span>
            <input id="memory-max-days" aria-label={i18nT('pages.overview.memoryTab.history_retention_days')} type="number" min={7} max={365} step={1} className="w-24 bg-bg-elevated border border-border rounded-md px-3 py-2 text-text text-sm font-body outline-hidden transition-colors focus-ring" value={maxDays} onChange={e => setMaxDays(Number(e.target.value))} />
          </label>
        )}
        <Btn onClick={async () => { await api.saveMemorySettings({ history_idle_hours: idleHours, history_max_days: maxDays }); setSettingsSaved(true); scheduleClear(() => setSettingsSaved(false), 2000) }}>{settingsSaved ? <><Check className="lucide-inline" /> {i18nT('pages.overview.memoryTab.saved')}</> : i18nT('pages.overview.memoryTab.save')}</Btn>
        <Btn onClick={consolidate} disabled={consolidating}>{consolidating ? <><Hourglass className="lucide-inline" /> {i18nT('pages.overview.memoryTab.running')}</> : <><RefreshCw className="lucide-inline" /> {i18nT('pages.overview.memoryTab.summarize_now')}</>}</Btn>
        {/* What the button does, under it on its own line: the reader identified
            "Summarize now" correctly but hesitated to press it, unsure what it
            did to their chats or whether it could be undone. Names the button
            rather than saying "it", since the line sits under the whole row. */}
        <p className="basis-full m-0 text-[12px] text-muted" data-testid="summarize-now-help">{i18nT('pages.overview.memoryTab.summarize_now_help')}</p>
        {/* No hand-off: the tab holds unsaved drafts -- the lesson rule being
            typed below and the store text in the editors -- that the hand-off's
            navigation would discard.

            The block variant, on its own flex line (`basis-full`) so the button
            row stays a row: a failure that persists until dismissed and carries
            a list of session keys is a banner, not a run of text beside the
            buttons -- inline, the keys sat against the dismiss control, where a
            bare identifier read as a chip the ✕ might delete. The keys ride
            INSIDE the notice as its footer, under the tally and away from the
            control (the `SkillsTab` refusal-findings shape): one failure, one
            surface, every key named. Each shape's one action rides in that
            footer too: the failed keys' counted retry, or the whole press again
            for a session list that could not be loaded. */}
        {consolidateError && (
          <ErrorNotice
            className="basis-full"
            title={consolidateError.kind === 'list' ? i18nT('pages.overview.memoryTab.sessions_list_failed') : undefined}
            message={consolidateError.message}
            /* The server's raw string is a separate clause from the bold lead: on
               its own line (`block`), in the mono font, so "Could not list the
               sessions to summarize sessions index unavailable" does not read
               as one run-on sentence. */
            messageClassName={consolidateError.kind === 'list' ? 'block font-mono' : ''}
            footer={consolidateError.kind === 'list'
              ? <RetrySessionList onRetry={() => void consolidate()} />
              : <FailedSessions failed={consolidateError.failed} skipped={consolidateError.skipped} onRetry={() => void summarizeTargets(consolidateError.failed)} />}
            onDismiss={() => setConsolidateError(null)}
            askAgent={false}
            testId="consolidate-failed"
          />
        )}
        {consolidateMsg && <span className={`text-[13px] ${consolidateOk ? 'text-ok' : 'text-danger'}`}>{consolidateMsg}</span>}

        {migrated && <span className="text-[12px] text-muted ml-2">{i18nT('pages.overview.memoryTab.semantic_memory_active_text_files_are_read_only')}</span>}
      </div>
    </Card>
    <VectorMemoryCard onActiveChange={setVectorActive} onMigratedChange={setMigrated} />
    <EmbeddingModelCard />
    <details onToggle={event => setRecordsOpen(event.currentTarget.open)}>
      <summary className="min-h-11 cursor-pointer rounded-lg border border-border p-3 text-[13px] text-muted">
        <SlidersHorizontal className="lucide-inline mr-2" aria-hidden="true" />
        {i18nT('memoryV2.edit_saved_memories')}
      </summary>
      {recordsOpen && <div className="mt-4"><MemoryRecordsEditor store="default" onDirtyChange={setRecordDirty} /></div>}
    </details>
    {/* The graph explorer lives on Developer. This user-facing V1 browser keeps
        its preferences, projects, history, semantic/episodic records, recovery,
        settings and lessons in the normal page flow. */}
    {/* `vectorActive` comes from the vector card, which reads the global store,
        independently of the picked one — so these three text documents are hidden whenever THAT
        store has migrated to semantic memory, whichever store the picker names.
        Pre-existing coupling, kept rather than widened: making the gate per-store
        needs the vector card to take the picker too, and that card is where the
        migration state is actually known. */}
    {!vectorActive && (<>
      {/* `key={store}`: a remount is what drops an unsaved draft when the scope
          changes, so a body typed against one store can never be saved into
          another. */}
      <MemoryDocCard
        key={`preferences-${store}`}
        docKey="preferences"
        onDirtyChange={dirty => setDocDirty(old => old.preferences === dirty ? old : { ...old, preferences: dirty })}
        store={store}
        title={i18nT('pages.overview.memoryTab.preferences')}
        info={i18nT('pages.overview.memoryTab.learned_user_preferences_coding_style_tools_work')}
        rows={8}
        placeholder={i18nT('pages.overview.memoryTab.loading')}
        read={s => api.memoryPreferences(s)}
        write={(c, s) => api.saveMemoryPreferences(c, s)}
      />
      <MemoryDocCard
        key={`projects-${store}`}
        docKey="projects"
        onDirtyChange={dirty => setDocDirty(old => old.projects === dirty ? old : { ...old, projects: dirty })}
        store={store}
        title={i18nT('pages.overview.memoryTab.projects')}
        rows={8}
        placeholder={i18nT('pages.overview.memoryTab.loading')}
        read={s => api.memoryProjects(s)}
        write={(c, s) => api.saveMemoryProjects(c, s)}
      />
      <MemoryDocCard
        key={`history-${store}`}
        docKey="history"
        onDirtyChange={dirty => setDocDirty(old => old.history === dirty ? old : { ...old, history: dirty })}
        store={store}
        title={i18nT('pages.overview.memoryTab.daily_history')}
        rows={10}
        mono
        placeholder={i18nT('pages.overview.memoryTab.no_history_yet')}
        read={s => api.memoryHistory(s)}
        write={(c, s) => api.saveMemoryHistory(c, s)}
      />
    </>)}
    {/* Store-specific keys REMOUNT each card on a store switch, which discards
        its per-store local state. Without it, MemoryBackupsCard keeps an ARMED
        restore across the switch — and a backup's name is not unique across stores
        (one sweep stamps every store's copy identically, and every store's file
        stem is `memory`), so the same-named row of the newly picked store renders
        already-confirmed and one click restores a store the operator never armed.
        Its "Back up now" and "Restored" status lines have the same problem in a
        milder form: they would report a mutation that landed on the store the card
        no longer shows. */}
    {storeReadable && <MemoryCarveCard key={`carve-${store}`} store={store} />}
    {storeReadable && <MemoryRetiredCard key={`retired-${store}`} store={store} />}
    <MemoryBackupsCard key={`backups-${store}`} store={store} />
    {!vectorActive && (
      <Card><CardTitle>{i18nT('pages.overview.memoryTab.lessons')} <InfoTip text={i18nT('pages.overview.memoryTab.persistent_lessons_injected_into_every_session_a')} /></CardTitle>
      <div className="flex gap-2 items-center flex-wrap mb-3">
        <Input placeholder={i18nT('pages.overview.memoryTab.rule_e_g_always_use_tabs_not_spaces')} style={{ flex: 2 }} value={rule} onChange={e => setRule(e.target.value)} />
        <SimpleSelect
          aria-label={i18nT('pages.overview.memoryTab.category')}
          style={{ flex: '0 0 140px' }}
          options={['knowledge', 'tool', 'preference']}
          optionLabels={[i18nT('pages.overview.memoryTab.knowledge'), i18nT('pages.overview.memoryTab.tool'), i18nT('pages.overview.memoryTab.preference')]}
          value={cat}
          onChange={setCat}
        />
        <SendBtn onClick={addLesson}>{i18nT('pages.overview.memoryTab.add')}</SendBtn>
        {lessonFeedback && (
          <span
            role={lessonFeedback.tone === 'error' ? 'alert' : 'status'}
            className={`text-[13px] ${
              lessonFeedback.tone === 'error'
                ? 'text-danger'
                : lessonFeedback.tone === 'warning'
                  ? 'text-warn'
                  : 'text-muted'
            }`}
          >
            {lessonFeedback.text}
          </span>
        )}
      </div>
      {/* No agent hand-off: it navigates away, and the Add row above may hold
          an unsaved rule draft. */}
      <ErrorNotice
        title={i18nT(deleteError?.step === 'refresh' ? 'pages.overview.memoryTab.lessons_refresh_failed' : 'pages.overview.memoryTab.delete_failed')}
        message={deleteError?.step === 'nothing' ? i18nT('pages.overview.memoryTab.delete_matched_nothing') : deleteError?.message}
        onDismiss={() => setDeleteError(null)}
        askAgent={false}
        className="mb-2"
      />
      {/* Scrolls sideways rather than clipping: five columns plus a long path
          fragment overrun a narrow viewport, and the card hides overflow. */}
      <div className="overflow-x-auto"><table className="w-full border-collapse table-striped"><thead><tr><SortableHeader label={i18nT('pages.overview.memoryTab.rule')} sortKey="rule" sort={lessonSort} onToggle={toggleLessonSort} /><SortableHeader label={i18nT('pages.overview.memoryTab.category')} sortKey="category" sort={lessonSort} onToggle={toggleLessonSort} /><SortableHeader label={i18nT('pages.overview.memoryTab.scope')} sortKey="repo_scope" sort={lessonSort} onToggle={toggleLessonSort} /><SortableHeader label={i18nT('pages.overview.memoryTab.when')} sortKey="ts" sort={lessonSort} onToggle={toggleLessonSort} /><th aria-label={i18nT('pages.overview.memoryTab.actions')} className="text-left text-muted text-[12px] uppercase tracking-[.04em] px-2.5 py-2 border-b border-border font-medium"></th></tr></thead>
        <tbody>{lessons.length === 0 ? <tr><td colSpan={5}><EmptyState icon={<BookOpen className="lucide-inline" />} title={i18nT('pages.overview.memoryTab.no_lessons_yet')} subtitle={i18nT('pages.overview.memoryTab.lessons_empty_subtitle')} /></td></tr> : sortedLessons.map((l) => (
          // Scope is part of the key: a scoped and a global row sharing rule text
          // are two lessons, and can share a timestamp. String() keeps the null
          // (unusable-scope) row distinct from the "" (global) one; the JSONL tier
          // keeps a workspace row distinct from a global one.
          <tr key={`${l.rule}-${String(l.repo_scope)}-${l.workspace ?? ''}-${l.ts}`} className="hover:bg-bg-hover transition-colors"><td className="px-2.5 py-2 border-b border-border text-sm">{esc(l.rule)}</td><td className="px-2.5 py-2 border-b border-border text-sm"><Badge variant="ok">{l.category}</Badge></td><td className="px-2.5 py-2 border-b border-border text-sm">{scopeCell(l)}</td><td className="px-2.5 py-2 border-b border-border text-sm">{fmtDateTimeNumeric(l.ts)}</td>
            <td className="px-2.5 py-2 border-b border-border text-sm"><Btn danger onClick={() => deleteLesson(l)}>{i18nT('pages.overview.memoryTab.delete')}</Btn></td></tr>
        ))}</tbody></table></div></Card>
    )}
    {confirmDialog}
  </>)
}
