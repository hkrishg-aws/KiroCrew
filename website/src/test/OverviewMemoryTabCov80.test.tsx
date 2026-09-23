import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { screen, waitFor, act, fireEvent, within } from '@testing-library/react'
import { renderWithProviders } from './helpers'
import userEvent from '@testing-library/user-event'

/**
 * Companion to integration/MemoryTab.integration.test.tsx (which drives the tab
 * against the MSW fixtures). This file stubs the api client directly so the
 * write paths — every Save, the lesson add/delete, and the manual consolidation
 * including its partial-failure branch — are assertable by their calls.
 */
// `vi.hoisted`, not a plain const: `./helpers` imports the Redux store, which
// imports `api/client`, and `vi.mock` is hoisted above a plain declaration -- so
// the factory would run before `api` was initialized and the whole suite would
// fail to load with "Cannot access 'api' before initialization".
const { api } = vi.hoisted(() => ({
  api: {
    lessons: vi.fn(),
    memoryPreferences: vi.fn(),
    memoryProjects: vi.fn(),
    memoryHistory: vi.fn(),
    memorySettings: vi.fn(),
    saveMemorySettings: vi.fn(),
    saveMemoryPreferences: vi.fn(),
    saveMemoryProjects: vi.fn(),
    saveMemoryHistory: vi.fn(),
    createLesson: vi.fn(),
    deleteLesson: vi.fn(),
    sessions: vi.fn(),
    consolidateMemory: vi.fn(),
    // The store picker and the three store-scoped cards the tab now mounts. Stubbed
    // here even though this file asserts none of them: an absent method is called as
    // `undefined` by its queryFn, which surfaces as the picker's refusal notice in
    // every test rather than as a missing-stub error naming the cause.
    memoryStores: vi.fn(),
    memoryRetired: vi.fn(),
    memoryBackups: vi.fn(),
    memoryCarve: vi.fn(),
    memoryBackupNow: vi.fn(),
    memoryRestoreBackup: vi.fn(),
    memoryRestoreRetired: vi.fn(),
  },
}))
vi.mock('../api/client', () => ({ api }))
// Both cards own their own queries and their own tests; here they are seams that
// report the vector/migration state this tab branches on.
vi.mock('../pages/overview/VectorMemoryCard', () => ({
  default: ({ diagnosticsOnly }: { diagnosticsOnly?: boolean }) => <div data-testid="vector-card" data-diagnostics-only={diagnosticsOnly ? 'true' : 'false'} />,
}))
vi.mock('../pages/overview/EmbeddingModelCard', () => ({ default: () => <div data-testid="embed-card" /> }))
vi.mock('../pages/overview/MemoryRecordsEditor', () => ({ default: () => <div data-testid="records-editor" /> }))

const MemoryTab = (await import('../pages/overview/MemoryTab')).default

const LESSONS = [
  { rule: 'zzq-rule-beta', category: 'tool', ts: '2026-01-02T00:00:00Z', repo_scope: '' },
  { rule: 'zzq-rule-alpha', category: 'knowledge', ts: '2026-01-01T00:00:00Z', repo_scope: '' },
]

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  api.lessons.mockResolvedValue({ lessons: LESSONS })
  api.memoryPreferences.mockResolvedValue({ content: 'zzq-prefs-body' })
  api.memoryProjects.mockResolvedValue({ content: 'zzq-projects-body' })
  api.memoryHistory.mockResolvedValue({ content: 'zzq-history-body' })
  api.memorySettings.mockResolvedValue({
    history_idle_hours: 4, history_max_days: 30, migrated: false,
  })
  api.saveMemorySettings.mockResolvedValue({ ok: true })
  api.saveMemoryPreferences.mockResolvedValue({ ok: true })
  api.saveMemoryProjects.mockResolvedValue({ ok: true })
  api.saveMemoryHistory.mockResolvedValue({ ok: true })
  api.createLesson.mockResolvedValue({ ok: true, outcome: 'inserted', reason: '' })
  api.deleteLesson.mockResolvedValue({ ok: true })
  api.sessions.mockResolvedValue({ sessions: [{ key: 'zzq-s1' }, { key: 'zzq-s2' }] })
  api.consolidateMemory.mockResolvedValue({ ok: true })
  // One declared store, the default. That keeps this file's subject the tab's own
  // write paths: the picker has nothing to switch to, so `store` stays at the
  // default and every read below is the storeless one these tests already assert.
  api.memoryStores.mockResolvedValue({
    stores: [{ name: 'default', is_default: true, lineage: 'v1', exists: true }],
  })
  api.memoryRetired.mockResolvedValue({ retired: [] })
  api.memoryBackups.mockResolvedValue({ backups: [] })
  api.memoryCarve.mockResolvedValue({ counts: {} })
})

afterEach(() => {
  vi.useRealTimers()
})

/** The Save button inside the card whose heading contains `heading`.
 *
 * `getAllByText`, not `getByText`: the tab carries a disclosure line naming which
 * cards do NOT follow the store picker, and it says "Memory settings" in prose —
 * so a heading pattern legitimately matches twice and the single-match form throws
 * before reaching the button. The card is identified by CONTAINING a Save button
 * rather than by being the first match, which is the property the caller wants.
 */
function saveIn(heading: RegExp): HTMLButtonElement {
  // A real `.card-glow` ancestor is REQUIRED, with no widening fallback. The prose
  // match has none, so `closest('div').parentElement` resolves to the tab's own
  // container — which holds every card, and therefore holds a Save button. That
  // returns the FIRST Save on the page and the assertion then waits forever on a
  // save the click never reached.
  for (const title of screen.getAllByText(heading)) {
    const card = title.closest('.card-glow')
    if (!card) continue
    const button = Array.from(card.querySelectorAll('button'))
      .find((b) => /save/i.test(b.textContent ?? ''))
    if (button) return button as HTMLButtonElement
  }
  throw new Error(`no card matching ${heading} carries a Save button`)
}

/** The lessons table's body rows.
 *
 * Scoped to the table carrying the "Rule" header rather than to `document`: the
 * store-scoped cards below the lessons card render their own tables, and each
 * mounts an empty-state row immediately — so a document-wide `tbody tr` query
 * silently picks up "Nothing has been retired" as if it were a lesson.
 */
function lessonRows(): HTMLTableRowElement[] {
  const header = screen.getByText(/^Rule$/)
  const table = header.closest('table')
  return Array.from((table as HTMLTableElement).querySelectorAll('tbody tr'))
}

describe('MemoryTab — settings', () => {
  it('keeps bulk management lazy while the legacy browser remains visible', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    expect(screen.queryByTestId('records-editor')).toBeNull()
    expect(await screen.findByText(/^Memory Settings$/i)).toBeInTheDocument()

    const ordered = [
      screen.getByRole('heading', { name: /^Memory Settings \?$/i }),
      screen.getByTestId('vector-card'),
      screen.getByTestId('embed-card'),
      screen.getByText(/^Edit saved memories$/i),
      screen.getByRole('heading', { name: /^Preferences\b/i }),
      screen.getByRole('heading', { name: /^Lessons\b/i }),
    ]
    for (let index = 0; index < ordered.length - 1; index += 1) {
      expect(ordered[index].compareDocumentPosition(ordered[index + 1]) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    }

    await userEvent.click(screen.getByText(/^Edit saved memories$/i))
    expect(screen.getByTestId('records-editor')).toBeInTheDocument()
    expect(screen.getByTestId('vector-card')).toHaveAttribute('data-diagnostics-only', 'false')
  })

  it('loads the saved retention settings and writes both fields back', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    const inputs = await waitFor(() => {
      const found = screen.getAllByRole('spinbutton') as HTMLInputElement[]
      expect(found[0].value).toBe('4')
      return found
    })
    expect(inputs[1].value).toBe('30')

    fireEvent.change(inputs[0], { target: { value: '6' } })
    fireEvent.change(inputs[1], { target: { value: '45' } })
    await userEvent.click(saveIn(/Memory Settings/i))

    await waitFor(() => expect(api.saveMemorySettings).toHaveBeenCalledWith({
      history_idle_hours: 6, history_max_days: 45,
    }))
    expect(await screen.findByText(/Saved/)).toBeInTheDocument()
  })

  it('hides the retention field, and the text-file editors, once memory is migrated', async () => {
    api.memorySettings.mockResolvedValue({
      history_idle_hours: 3, history_max_days: 90, migrated: true,
    })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await waitFor(() => expect(screen.getAllByRole('spinbutton')).toHaveLength(1))
    expect(screen.getByText(/read-only/i)).toBeInTheDocument()
  })

  it('clears the transient Saved marker on its own timer', async () => {
    vi.useFakeTimers()
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await act(async () => {})
    const save = saveIn(/Memory Settings/i)
    fireEvent.click(save)
    await act(async () => {})
    expect(save.textContent).toContain('Saved')

    await act(async () => { vi.advanceTimersByTime(2000) })
    expect(save.textContent).not.toContain('Saved')
  })
})

describe('MemoryTab — the three text stores', () => {
  it.each(['pending', 'failed'] as const)(
    'cannot overwrite a document while its initial read is %s',
    async state => {
      if (state === 'pending') {
        api.memoryPreferences.mockImplementation(() => new Promise(() => {}))
      } else {
        api.memoryPreferences.mockRejectedValue(Object.assign(new Error('zzq-read-failed'), { status: 403 }))
      }
      renderWithProviders(<MemoryTab refreshTrigger={0} />)
      await screen.findByDisplayValue('zzq-projects-body')
      if (state === 'failed') await screen.findByText('zzq-read-failed')

      const save = saveIn(/^Preferences$/)
      expect(save).toBeDisabled()
      fireEvent.click(save)
      expect(api.saveMemoryPreferences).not.toHaveBeenCalled()
    },
  )

  it('can save a genuinely empty document after its read succeeds', async () => {
    api.memoryPreferences.mockResolvedValue({ content: '' })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByDisplayValue('zzq-projects-body')
    const save = saveIn(/^Preferences$/)
    await waitFor(() => expect(save).toBeEnabled())
    fireEvent.click(save)
    await waitFor(() => expect(api.saveMemoryPreferences).toHaveBeenCalledWith('', undefined))
  })

  it('shows a redacted document read-only and never sends its masked body', async () => {
    api.memoryPreferences.mockResolvedValue({
      content: 'keep [REDACTED: credential] hidden',
      content_redacted: true,
    })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)

    const preferences = await screen.findByDisplayValue('keep [REDACTED: credential] hidden')
    expect(preferences).toBeDisabled()
    expect(screen.getByText(/Sensitive values are hidden/)).toBeInTheDocument()
    const save = saveIn(/^Preferences$/)
    expect(save).toBeDisabled()
    fireEvent.click(save)
    expect(api.saveMemoryPreferences).not.toHaveBeenCalled()
  })

  it('preserves a draft but blocks stale cached content during a redacted refetch', async () => {
    const view = renderWithProviders(<MemoryTab refreshTrigger={0} />)
    const preferences = await screen.findByDisplayValue('zzq-prefs-body')
    fireEvent.change(preferences, { target: { value: 'recover this unrelated draft' } })
    let finishRead!: (value: unknown) => void
    api.memoryPreferences.mockImplementation(
      () => new Promise(resolve => { finishRead = resolve }),
    )

    act(() => {
      void view.queryClient.invalidateQueries({ queryKey: ['memory-doc', 'preferences', ''] })
    })
    const save = saveIn(/^Preferences$/)
    await waitFor(() => expect(save).toBeDisabled())
    fireEvent.click(save)
    expect(api.saveMemoryPreferences).not.toHaveBeenCalled()

    finishRead({ content: '[REDACTED: credential]', content_redacted: true })
    expect(await screen.findByText(/Sensitive values are hidden/)).toBeInTheDocument()
    expect(preferences).toHaveValue('recover this unrelated draft')
    expect(preferences).toBeDisabled()
    expect(save).toBeDisabled()
  })

  it('does not save cached clean content after its confirming refetch fails', async () => {
    const view = renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByDisplayValue('zzq-prefs-body')
    api.memoryPreferences.mockRejectedValueOnce(
      Object.assign(new Error('fresh read failed'), { status: 403 }),
    )

    await act(async () => {
      await view.queryClient.refetchQueries({ queryKey: ['memory-doc', 'preferences', ''] })
    })

    expect(await screen.findByText('fresh read failed')).toBeInTheDocument()
    const save = saveIn(/^Preferences$/)
    expect(save).toBeDisabled()
    fireEvent.click(save)
    expect(api.saveMemoryPreferences).not.toHaveBeenCalled()
  })

  // The saves assert `undefined` as the second argument on purpose. A save carries
  // the picked store, and `undefined` is what "no store named" has to look like on
  // the wire: the gateway reads an ABSENT ?store= as the global store and
  // applies the owner gate only to a parameter that is present, so a save that sent
  // `store=default` here would turn a write every session can make into an
  // owner-only one. Asserting the arity is what pins that.
  it('loads each store and saves the edited text back to its own endpoint', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    const prefs = await screen.findByRole('textbox', { name: /preferences/i }) as HTMLTextAreaElement
    await waitFor(() => expect(prefs.value).toBe('zzq-prefs-body'))

    fireEvent.change(prefs, { target: { value: 'zzq-prefs-edited' } })
    await userEvent.click(saveIn(/^Preferences$/))
    await waitFor(() => expect(api.saveMemoryPreferences).toHaveBeenCalledWith('zzq-prefs-edited', undefined))

    const projects = screen.getByRole('textbox', { name: /projects/i }) as HTMLTextAreaElement
    fireEvent.change(projects, { target: { value: 'zzq-projects-edited' } })
    await userEvent.click(saveIn(/^Projects$/))
    await waitFor(() => expect(api.saveMemoryProjects).toHaveBeenCalledWith('zzq-projects-edited', undefined))

    const history = screen.getByRole('textbox', { name: /daily history/i }) as HTMLTextAreaElement
    fireEvent.change(history, { target: { value: 'zzq-history-edited' } })
    await userEvent.click(saveIn(/Daily History/i))
    await waitFor(() => expect(api.saveMemoryHistory).toHaveBeenCalledWith('zzq-history-edited', undefined))
  })

  it('re-reads every store when the parent bumps the refresh trigger', async () => {
    const { rerender } = renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await waitFor(() => expect(api.memoryPreferences).toHaveBeenCalled())
    const before = api.memoryPreferences.mock.calls.length
    const lessonsBefore = api.lessons.mock.calls.length
    rerender(<MemoryTab refreshTrigger={1} />)
    await waitFor(() =>
      expect(api.memoryPreferences.mock.calls.length).toBeGreaterThan(before))
    expect(api.lessons.mock.calls.length).toBeGreaterThan(lessonsBefore)
  })

  it('tolerates an empty payload rather than rendering undefined', async () => {
    api.memoryPreferences.mockResolvedValue({})
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    const prefs = await screen.findByRole('textbox', { name: /preferences/i }) as HTMLTextAreaElement
    await waitFor(() => expect(prefs.value).toBe(''))
  })
})

describe('MemoryTab — lessons', () => {
  it('lists the stored lessons, newest first by default', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    const rules = lessonRows().map((tr) => tr.querySelector('td:first-child'))
      .map((td) => td.textContent)
    expect(rules).toEqual(['zzq-rule-beta', 'zzq-rule-alpha'])
  })

  it('re-sorts on a header click', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    await userEvent.click(screen.getByText(/^Rule$/))
    const rules = lessonRows().map((tr) => tr.querySelector('td:first-child'))
      .map((td) => td.textContent)
    expect(rules).toEqual(['zzq-rule-alpha', 'zzq-rule-beta'])

    await userEvent.click(screen.getByText(/^Category$/))
    const cats = lessonRows().map((tr) => tr.querySelector('td:nth-child(2)'))
      .map((td) => td.textContent)
    expect(cats).toEqual(['knowledge', 'tool'])
  })

  it('adds a lesson with the chosen category, then re-reads the list', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    const reads = api.lessons.mock.calls.length
    const input = screen.getByPlaceholderText(/Rule/) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'zzq-new-rule' } })
    await userEvent.click(screen.getByRole('button', { name: /^Add$/ }))

    await waitFor(() => expect(api.createLesson).toHaveBeenCalledWith('zzq-new-rule', 'knowledge'))
    await waitFor(() => expect(api.lessons.mock.calls.length).toBeGreaterThan(reads))
    expect(input.value).toBe('')
  })

  it('keeps a refused lesson editable and reports the backend reason', async () => {
    api.createLesson.mockResolvedValue({
      ok: false, outcome: 'refused', reason: 'blocked_not_clause',
    })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    const reads = api.lessons.mock.calls.length
    const input = screen.getByPlaceholderText(/Rule/) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'zzq-refused-rule' } })
    await userEvent.click(screen.getByRole('button', { name: /^Add$/ }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /Lesson not saved.*blocked_not_clause.*Edit it and try again/i,
    )
    expect(input.value).toBe('zzq-refused-rule')
    expect(api.lessons).toHaveBeenCalledTimes(reads)
  })

  it('keeps a deduped lesson editable instead of implying it was added', async () => {
    api.createLesson.mockResolvedValue({
      ok: false, outcome: 'deduped', reason: 'substring',
    })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    const reads = api.lessons.mock.calls.length
    const input = screen.getByPlaceholderText(/Rule/) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'zzq-covered-rule' } })
    await userEvent.click(screen.getByRole('button', { name: /^Add$/ }))

    expect(await screen.findByRole('status')).toHaveTextContent(
      /existing lesson already covers this.*substring/i,
    )
    expect(input.value).toBe('zzq-covered-rule')
    expect(api.lessons).toHaveBeenCalledTimes(reads)
  })

  it('clears an unchanged resubmission but says it was already stored', async () => {
    api.createLesson.mockResolvedValue({
      ok: true, outcome: 'unchanged', reason: 'identical',
    })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    const input = screen.getByPlaceholderText(/Rule/) as HTMLInputElement
    fireEvent.change(input, { target: { value: 'zzq-existing-rule' } })
    await userEvent.click(screen.getByRole('button', { name: /^Add$/ }))

    expect(await screen.findByRole('status')).toHaveTextContent(/already stored/i)
    expect(input.value).toBe('')
  })

  it('refuses to add an empty rule', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    await userEvent.click(screen.getByRole('button', { name: /^Add$/ }))
    expect(api.createLesson).not.toHaveBeenCalled()
  })

  it('deletes the lesson its row names, then re-reads the list', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    const reads = api.lessons.mock.calls.length
    const row = screen.getByText('zzq-rule-beta').closest('tr') as HTMLElement
    const del = Array.from(row.querySelectorAll('button'))
      .find((b) => /delete/i.test(b.textContent ?? '')) as HTMLButtonElement
    await userEvent.click(del)

    // The global row's own selector ("") rides along: a lesson's identity is
    // (rule, repo_scope), so a bare rule would delete every scope's row. The
    // fixture rows name no JSONL tier, so none is forwarded.
    await waitFor(() => expect(api.deleteLesson).toHaveBeenCalledWith('zzq-rule-beta', '', { scope: undefined, workspace: undefined, exact: true }))
    await waitFor(() => expect(api.lessons.mock.calls.length).toBeGreaterThan(reads))
  })

  it('tells two same-rule rows apart by scope and deletes only the clicked one (#10651)', async () => {
    api.lessons.mockResolvedValue({
      lessons: [
        { rule: 'zzq-same-rule', category: 'tool', ts: '2026-01-02T00:00:00Z', repo_scope: '' },
        { rule: 'zzq-same-rule', category: 'tool', ts: '2026-01-02T00:00:00Z', repo_scope: 'src/pkg' },
        // Stored scope present but unusable: the list reports null, and the
        // only delete that reaches such a row is the unselective one.
        { rule: 'zzq-broken-rule', category: 'tool', ts: '2026-01-03T00:00:00Z', repo_scope: null },
      ],
    })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('src/pkg')
    // Two rows share the rule; the Scope column is what tells them apart, and
    // each of the three selector values reads differently.
    const sameRule = screen.getAllByText('zzq-same-rule').map((td) => td.closest('tr') as HTMLElement)
    expect(sameRule).toHaveLength(2)
    expect(screen.getByRole('columnheader', { name: /Scope/ })).toBeInTheDocument()
    const scoped = sameRule.find((tr) => tr.textContent?.includes('src/pkg')) as HTMLElement
    const global = sameRule.find((tr) => !tr.textContent?.includes('src/pkg')) as HTMLElement
    expect(global).toHaveTextContent(/Global/)
    const broken = screen.getByText('zzq-broken-rule').closest('tr') as HTMLElement
    expect(broken).toHaveTextContent(/Unusable scope/)
    const deleteIn = (tr: HTMLElement) => Array.from(tr.querySelectorAll('button'))
      .find((b) => /delete/i.test(b.textContent ?? '')) as HTMLButtonElement
    const dialogTitle = /Delete this lesson in every scope\?/

    // A scoped or global row deletes without a prompt: its selector reaches
    // exactly that row.
    await userEvent.click(deleteIn(scoped))
    await waitFor(() => expect(api.deleteLesson).toHaveBeenCalledWith('zzq-same-rule', 'src/pkg', { scope: undefined, workspace: undefined, exact: true }))
    expect(api.deleteLesson).not.toHaveBeenCalledWith('zzq-same-rule', '', { scope: undefined, workspace: undefined, exact: true })
    await userEvent.click(deleteIn(global))
    await waitFor(() => expect(api.deleteLesson).toHaveBeenCalledWith('zzq-same-rule', '', { scope: undefined, workspace: undefined, exact: true }))
    expect(screen.queryByText(dialogTitle)).not.toBeInTheDocument()

    // The null row's delete is the unselective one, so it asks first through
    // the shared dialog, whose confirm button restates the act. Cancel sends
    // nothing; confirming sends the null through (the client drops the key).
    await userEvent.click(deleteIn(broken))
    expect(await screen.findByText(dialogTitle)).toBeInTheDocument()
    expect(screen.getByText(/every lesson with exactly this text will be removed/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /^Cancel$/ }))
    await waitFor(() => expect(screen.queryByText(dialogTitle)).not.toBeInTheDocument())
    expect(api.deleteLesson).not.toHaveBeenCalledWith('zzq-broken-rule', null, { scope: undefined, workspace: undefined, exact: true })

    await userEvent.click(deleteIn(broken))
    await userEvent.click(await screen.findByRole('button', { name: /^Delete in every scope$/ }))
    await waitFor(() => expect(api.deleteLesson).toHaveBeenCalledWith('zzq-broken-rule', null, { scope: undefined, workspace: undefined, exact: true }))
  })

  it('reports a rejected delete beside the table instead of swallowing it (#10651)', async () => {
    api.deleteLesson.mockRejectedValueOnce(new Error('zzq-delete-refused'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    const reads = api.lessons.mock.calls.length
    const row = screen.getByText('zzq-rule-beta').closest('tr') as HTMLElement
    const del = Array.from(row.querySelectorAll('button'))
      .find((b) => /delete/i.test(b.textContent ?? '')) as HTMLButtonElement
    await userEvent.click(del)

    const notice = await screen.findByText('zzq-delete-refused')
    expect(notice).toBeInTheDocument()
    expect(screen.getByText(/Could not delete the lesson/)).toBeInTheDocument()
    // The row is still there and the list was not re-read as if it had gone.
    expect(screen.getByText('zzq-rule-beta')).toBeInTheDocument()
    expect(api.lessons.mock.calls.length).toBe(reads)

    // Dismissable, and a later successful delete clears it on its own.
    await userEvent.click(screen.getByRole('button', { name: /dismiss/i }))
    await waitFor(() => expect(screen.queryByText('zzq-delete-refused')).not.toBeInTheDocument())
  })

  it('reports a failed re-read after a successful delete (#10651)', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    api.lessons.mockRejectedValueOnce(new Error('zzq-refresh-refused'))
    const row = screen.getByText('zzq-rule-beta').closest('tr') as HTMLElement
    const del = Array.from(row.querySelectorAll('button'))
      .find((b) => /delete/i.test(b.textContent ?? '')) as HTMLButtonElement
    await userEvent.click(del)

    await waitFor(() => expect(api.deleteLesson).toHaveBeenCalledWith('zzq-rule-beta', '', { scope: undefined, workspace: undefined, exact: true }))
    expect(await screen.findByText('zzq-refresh-refused')).toBeInTheDocument()
    // Titled for what actually failed: the row is gone, the list is stale.
    expect(screen.getByText(/Lesson deleted, but the list could not be refreshed/)).toBeInTheDocument()
    expect(screen.queryByText(/Could not delete the lesson/)).not.toBeInTheDocument()
  })

  it('sends a workspace-tier row back to its own file on delete (#10651)', async () => {
    api.lessons.mockResolvedValue({
      lessons: [
        { rule: 'zzq-tier-rule', category: 'tool', ts: '2026-01-02T00:00:00Z', repo_scope: '', scope: 'global' },
        { rule: 'zzq-tier-rule', category: 'tool', ts: '2026-01-02T00:00:00Z', repo_scope: '', scope: 'workspace', workspace: 'ws-1' },
      ],
    })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    const rows = (await screen.findAllByText('zzq-tier-rule')).map((td) => td.closest('tr') as HTMLElement)
    expect(rows).toHaveLength(2)
    // The tier shows in the Scope cell, so the two same-text rows read apart.
    expect(rows[0]).not.toHaveTextContent(/Workspace ws-1/)
    expect(rows[1]).toHaveTextContent(/Workspace ws-1/)
    const deleteIn = (tr: HTMLElement) => Array.from(tr.querySelectorAll('button'))
      .find((b) => /delete/i.test(b.textContent ?? '')) as HTMLButtonElement
    await userEvent.click(deleteIn(rows[1]))
    await waitFor(() => expect(api.deleteLesson).toHaveBeenCalledWith('zzq-tier-rule', '', { scope: 'workspace', workspace: 'ws-1', exact: true }))
    await userEvent.click(deleteIn(rows[0]))
    await waitFor(() => expect(api.deleteLesson).toHaveBeenCalledWith('zzq-tier-rule', '', { scope: 'global', workspace: undefined, exact: true }))
  })

  it('says so when the delete matched no stored row (#10651)', async () => {
    api.deleteLesson.mockResolvedValueOnce({ ok: false })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByText('zzq-rule-beta')
    const row = screen.getByText('zzq-rule-beta').closest('tr') as HTMLElement
    const del = Array.from(row.querySelectorAll('button'))
      .find((b) => /delete/i.test(b.textContent ?? '')) as HTMLButtonElement
    await userEvent.click(del)
    expect(await screen.findByText(/No stored lesson matched this row/)).toBeInTheDocument()
    expect(screen.getByText(/Could not delete the lesson/)).toBeInTheDocument()
  })

  it('shows an empty state rather than a bare table', async () => {
    api.lessons.mockResolvedValue({ lessons: [] })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    expect(await screen.findByText(/No lessons yet/i)).toBeInTheDocument()
  })

  it('tolerates a response with no lessons key', async () => {
    api.lessons.mockResolvedValue({})
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    expect(await screen.findByText(/No lessons yet/i)).toBeInTheDocument()
  })
})

describe('MemoryTab — manual consolidation', () => {
  it('consolidates every known session and reports the count', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    await waitFor(() => expect(api.consolidateMemory).toHaveBeenCalledTimes(2))
    expect(api.consolidateMemory).toHaveBeenCalledWith('zzq-s1', true)
    // The tally's verb matches the button that produced it, and the all-success
    // path reads n/n like every other tally: "Summarized 5 sessions" beside
    // "Summarized 1/2 sessions" left the reader wondering if some were left out.
    expect((await screen.findByText(/Summarized/)).textContent).toContain('Summarized 2/2 sessions')
  })

  it('says what the button does, under it, before it is ever pressed', async () => {
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await screen.findByRole('button', { name: /Summarize now/i })
    // The reader identified the button but hesitated: "I don't know what it
    // does to my chats or if I can undo it." The line answers both, and is
    // static text, not part of the button's name. It also says WHAT goes into
    // memory (UX watch on r26: "writes summaries" named no content).
    const help = screen.getByTestId('summarize-now-help')
    expect(help.textContent).toBe(
      'Summarize now writes the facts, decisions and lessons from your conversations into memory; the conversations themselves stay untouched.',
    )
    expect(help.textContent).toMatch(/facts, decisions and lessons/)
    expect(help.closest('button')).toBeNull()
  })

  it('reports a partial failure instead of claiming success', async () => {
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(new Error('zzq-consolidate-failed'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    // A failed tally is an error surface: it renders through ErrorNotice
    // (role="alert", the danger tone), not the success span.
    const notice = await screen.findByRole('alert')
    expect(notice.textContent).toContain('(1 failed)')
    expect(notice.className).toContain('text-danger')
  })

  it('names the sessions that failed, under a label, so the reader can act on them', async () => {
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(new Error('zzq-consolidate-failed'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const notice = await screen.findByRole('alert')
    // The second key is the one that rejected; the first succeeded and is not
    // named. The label says what the key IS -- a bare identifier after the
    // tally read as an unexplained chip.
    expect(notice.textContent).toContain('Failed: zzq-s2')
    expect(notice.textContent).not.toContain('zzq-s1')
  })

  it('names a failed session by the title api.sessions gave it, with the key beside it', async () => {
    api.sessions.mockResolvedValue({ sessions: [{ key: 'zzq-s1', title: 'Release notes draft' }, { key: 'zzq-s2', title: 'Perf triage' }] })
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(new Error('zzq-consolidate-failed'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const notice = await screen.findByRole('alert')
    // The title is what the user knows the session by; the key is for whoever
    // has to find the transcript, so it rides beside the title, not instead.
    expect(notice.textContent).toContain('Perf triage')
    const list = within(notice).getByTestId('consolidate-failed-keys')
    const rows = within(list).getAllByRole('listitem')
    expect(rows).toHaveLength(1)
    expect(rows[0].textContent).toBe('Perf triagezzq-s2')
    expect(within(rows[0]).getByText('Perf triage').className).not.toContain('font-mono')
    expect(within(rows[0]).getByText('zzq-s2').className).toContain('font-mono')
    expect(list.textContent).not.toContain('Release notes draft')
  })

  it('names every failed session in the notice, none hidden behind a tooltip', async () => {
    api.sessions.mockResolvedValue({ sessions: ['a', 'b', 'c', 'd', 'e'].map(k => ({ key: `zzq-${k}` })) })
    api.consolidateMemory.mockRejectedValue(new Error('zzq-consolidate-failed'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const notice = await screen.findByRole('alert')
    expect(notice.textContent).toContain('(5 failed)')
    // All five in the DOM: keyboard and assistive-tech users get the same list
    // a pointer user does, and nothing rides a native title tooltip.
    for (const k of ['a', 'b', 'c', 'd', 'e']) expect(notice.textContent).toContain(`zzq-${k}`)
    expect(notice.querySelector('[title]')).toBeNull()
    const rows = within(screen.getByTestId('consolidate-failed-keys')).getAllByRole('listitem')
    // A session with no title is named by its key alone, once.
    expect(rows.map(r => r.textContent)).toEqual(['zzq-a', 'zzq-b', 'zzq-c', 'zzq-d', 'zzq-e'])
  })

  it('offers a counted retry on the failed sessions, re-posting only the keys that failed', async () => {
    api.sessions.mockResolvedValue({ sessions: [{ key: 'zzq-s1' }, { key: 'zzq-s2' }, { key: 'zzq-s3' }] })
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(new Error('zzq-consolidate-failed'))
      .mockRejectedValueOnce(new Error('zzq-consolidate-failed'))
      .mockResolvedValue({ ok: true })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))
    const notice = await screen.findByRole('alert')
    expect(notice.textContent).toContain('(2 failed)')
    api.consolidateMemory.mockClear()

    // The action lives in the banner beside the keys it acts on, states how
    // many it will re-post, and posts exactly those keys: the session that
    // already summarized is not billed a second turn.
    await userEvent.click(within(notice).getByRole('button', { name: 'Retry 2 failed' }))
    await waitFor(() => expect(api.consolidateMemory).toHaveBeenCalledTimes(2))
    expect(api.consolidateMemory.mock.calls.map(c => c[0])).toEqual(['zzq-s2', 'zzq-s3'])
    // The retry's outcome replaces the banner: both passed, so the success tally.
    expect(screen.queryByRole('alert')).toBeNull()
    expect((await screen.findByText(/Summarized/)).textContent).toContain('2')
  })

  it('names the keys that failed again after a retry', async () => {
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValue(new Error('zzq-consolidate-failed'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))
    const notice = await screen.findByRole('alert')
    await userEvent.click(within(notice).getByRole('button', { name: 'Retry 1 failed' }))

    const again = await screen.findByRole('alert')
    expect(again.textContent).toContain('0/1 sessions (1 failed)')
    expect(again.textContent).toContain('Failed: zzq-s2')
  })

  it('keeps the dismiss control apart from the key list', async () => {
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(new Error('zzq-consolidate-failed'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const notice = await screen.findByRole('alert')
    const keys = screen.getByTestId('consolidate-failed-keys')
    const dismiss = screen.getByRole('button', { name: /dismiss/i })
    // The keys live in the notice's footer, a block under the tally; the ✕ is
    // the banner's own control beside the text block, never the element right
    // after the last key -- where it read as "delete this session".
    expect(keys.nextElementSibling).not.toBe(dismiss)
    expect(keys.parentElement).not.toBe(dismiss.parentElement)
    expect(notice.contains(keys) && notice.contains(dismiss)).toBe(true)
    expect(keys.textContent).toContain('zzq-s2')
  })

  it('keeps the failure notice until it is dismissed', async () => {
    vi.useFakeTimers()
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(new Error('zzq-consolidate-failed'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await act(async () => {})
    fireEvent.click(screen.getByRole('button', { name: /Summarize now/i }))
    await act(async () => {})
    expect(screen.getByRole('alert')).toBeInTheDocument()

    // The success tally clears itself; a failure the user has to act on must not.
    await act(async () => { vi.advanceTimersByTime(4000) })
    expect(screen.queryByRole('alert')).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /dismiss/i }))
    expect(screen.queryByRole('alert')).toBeNull()
  })

  /** The route's refusal for a Temporary or Incognito target, as `j()` rejects it:
   *  an ApiError-shaped rejection whose raw body carries the backend `code`. A body
   *  without `mode` is the shape an older backend answered; the tally then keeps
   *  the either/or wording. */
  const restrictedTarget = (mode?: 'temporary' | 'incognito') => Object.assign(
    new Error(`Consolidation is not allowed for a ${mode ?? 'temporary'} session: it leaves no durable memory.`),
    { status: 403, body: JSON.stringify({ error: `Consolidation is not allowed for a ${mode ?? 'temporary'} session: it leaves no durable memory.`, code: 'restricted_target_session', ...(mode ? { mode } : {}) }) },
  )

  it('names the one mode every skipped session was in', async () => {
    // "skipped as temporary or incognito" left the reader unsure whether that
    // was two kinds of private chat or one thing with two names, and which
    // theirs was. The route's body names the mode; when every skip shares it,
    // the tally says so.
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(restrictedTarget('incognito'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const msg = await screen.findByText(/Summarized/)
    expect(msg.textContent).toContain('1/2 sessions (1 skipped: incognito session)')
    expect(msg.textContent).not.toContain('temporary or incognito')
  })

  it('pluralizes the named mode and keeps it beside a genuine failure', async () => {
    api.sessions.mockResolvedValue({ sessions: [{ key: 'zzq-s1' }, { key: 'zzq-s2' }, { key: 'zzq-s3' }, { key: 'zzq-s4' }] })
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(restrictedTarget('temporary'))
      .mockRejectedValueOnce(restrictedTarget('temporary'))
      .mockRejectedValueOnce(Object.assign(new Error('zzq-consolidate-failed'), { status: 500, body: '{"error": "boom"}' }))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const notice = await screen.findByRole('alert')
    expect(notice.textContent).toContain('1/4 sessions (1 failed)')
    // The skip is the tally's other half, apart from the failed count and in
    // the success tone it has when it stands alone: the same words in the
    // banner's danger text read as one more thing gone wrong.
    const skip = within(notice).getByTestId('consolidate-skipped')
    expect(skip.textContent).toContain('2 skipped: temporary sessions')
    expect(skip.className).toContain('text-ok')
    expect(notice.textContent).toContain('Failed: zzq-s4')
  })

  it('renders the skip beside a failure in the success tone, not the banner\'s danger tone', async () => {
    api.sessions.mockResolvedValue({ sessions: [{ key: 'zzq-s1' }, { key: 'zzq-s2' }, { key: 'zzq-s3' }] })
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(restrictedTarget('incognito'))
      .mockRejectedValueOnce(Object.assign(new Error('zzq-consolidate-failed'), { status: 500, body: '{"error": "boom"}' }))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const notice = await screen.findByRole('alert')
    expect(notice.className).toContain('text-danger')
    const skip = within(notice).getByTestId('consolidate-skipped')
    // Two spans, two tones: the failed count is the banner's own (danger) text
    // and never names the skip; the skip carries its own colour and the check
    // the standalone skip tally carries, and never the failed count.
    expect(skip.className).toContain('text-ok')
    expect(skip.className).not.toContain('text-danger')
    expect((skip.textContent ?? '').trim()).toBe('1 skipped: incognito session')
    expect(skip.querySelector('svg')).not.toBeNull()
    expect((notice.textContent ?? '').replace(skip.textContent ?? '', '')).not.toContain('skipped')
    expect(notice.textContent).toContain('Summarized 1/3 sessions (1 failed)')
    // Under the message, before the failed keys: the tally's second line, not
    // a footnote after the list.
    const keys = within(notice).getByTestId('consolidate-failed-keys')
    expect(skip.compareDocumentPosition(keys) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('keeps the either/or wording when the skipped sessions were in different modes', async () => {
    api.sessions.mockResolvedValue({ sessions: [{ key: 'zzq-s1' }, { key: 'zzq-s2' }, { key: 'zzq-s3' }] })
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(restrictedTarget('temporary'))
      .mockRejectedValueOnce(restrictedTarget('incognito'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const msg = await screen.findByText(/Summarized/)
    expect(msg.textContent).toContain('1/3 sessions (2 skipped as temporary or incognito)')
  })

  it('counts a restricted_target_session refusal as skipped, not failed', async () => {
    // `api.sessions` lists the temporary session too; the route refuses it by
    // design, so the press must read as ok with the skip named, never as a
    // failure on every "Summarize now".
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(restrictedTarget())
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const msg = await screen.findByText(/Summarized/)
    expect(msg.textContent).toContain('1/2 sessions (1 skipped as temporary or incognito)')
    expect(msg.textContent).not.toContain('failed')
    // The success tone, and no error surface for a refusal the user asked for.
    expect((msg.closest('span') as HTMLElement).className).toContain('text-ok')
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('keeps a genuine failure failed beside a skipped refusal', async () => {
    api.sessions.mockResolvedValue({ sessions: [{ key: 'zzq-s1' }, { key: 'zzq-s2' }, { key: 'zzq-s3' }] })
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValueOnce(restrictedTarget())
      .mockRejectedValueOnce(Object.assign(new Error('zzq-consolidate-failed'), { status: 500, body: '{"error": "boom"}' }))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const notice = await screen.findByRole('alert')
    expect(notice.textContent).toContain('1/3 sessions (1 failed)')
    expect(within(notice).getByTestId('consolidate-skipped').textContent).toContain('1 skipped as temporary or incognito')
    // The failed one is named; the refused one is a skip, not a failure.
    expect(notice.textContent).toContain('zzq-s3')
    expect(notice.textContent).not.toContain('zzq-s2')
  })

  it('says there is nothing to summarize when no session exists', async () => {
    api.sessions.mockResolvedValue({ sessions: [] })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))
    expect(await screen.findByText(/No sessions to summarize/i)).toBeInTheDocument()
    expect(api.consolidateMemory).not.toHaveBeenCalled()
  })

  it('reports a rejected session list as a failure, never as nothing to summarize', async () => {
    // Nothing was posted, so "no sessions to summarize" would claim a state the
    // request never established. The server's own string stays the message --
    // the journal key ErrorNotice recovers the endpoint/status/code report by --
    // under a localized lead; no session failed, so the footer names none and
    // offers no per-session retry; its one action is the whole press again.
    api.sessions.mockRejectedValue(Object.assign(new Error('HTTP 503: zzq-sessions-unreachable'), { status: 503, body: 'zzq-sessions-unreachable' }))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const notice = await screen.findByRole('alert')
    expect(notice.textContent).toContain('Could not list the sessions to summarize')
    // The server's string is its own clause: on its own line under the bold
    // lead (`block`), in the mono font -- not run on from the lead as one
    // sentence.
    const serverString = within(notice).getByText('HTTP 503: zzq-sessions-unreachable')
    expect(serverString.className).toContain('font-mono')
    expect(serverString.className).toContain('block')
    expect(serverString.previousElementSibling?.tagName).toBe('STRONG')
    expect(screen.queryByText(/No sessions to summarize/i)).toBeNull()
    expect(screen.queryByTestId('consolidate-failed-keys')).toBeNull()
    expect(within(notice).queryByRole('button', { name: /Retry \d+ failed/ })).toBeNull()
    expect(within(notice).getByRole('button', { name: 'Try again' })).toBeInTheDocument()
    expect(api.consolidateMemory).not.toHaveBeenCalled()
    // Dismissible like every other failure notice.
    await userEvent.click(within(notice).getByRole('button', { name: /dismiss/i }))
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('offers Try again on a rejected session list, which re-fetches the list and runs the summarize', async () => {
    // The banner dead-ended: a server string and a dismiss control, the way out
    // being the button above, which nothing in the banner pointed at. "Try
    // again" is the whole press from inside the banner -- the list fetched
    // again and, once it loads, every listed session posted.
    api.sessions
      .mockRejectedValueOnce(Object.assign(new Error('HTTP 503: zzq-sessions-unreachable'), { status: 503, body: 'zzq-sessions-unreachable' }))
      .mockResolvedValueOnce({ sessions: [{ key: 'zzq-s1' }, { key: 'zzq-s2' }, { key: 'zzq-s3' }] })
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))
    const notice = await screen.findByRole('alert')
    expect(api.sessions).toHaveBeenCalledTimes(1)
    expect(api.consolidateMemory).not.toHaveBeenCalled()

    await userEvent.click(within(notice).getByRole('button', { name: 'Try again' }))
    await waitFor(() => expect(api.sessions).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(api.consolidateMemory).toHaveBeenCalledTimes(3))
    expect(api.consolidateMemory.mock.calls.map(c => c[0])).toEqual(['zzq-s1', 'zzq-s2', 'zzq-s3'])
    // The outcome replaces the banner: the list loaded and every session passed.
    expect(screen.queryByRole('alert')).toBeNull()
    expect((await screen.findByText(/Summarized/)).textContent).toContain('Summarized 3/3 sessions')
  })

  it('names an untitled channel stem by its source, with the stem beside it', async () => {
    // A stem like `telegram_7781120043` is a filename, not a name. The source
    // the reader recognizes leads in the UI font; the stem rides beside it in
    // mono, as a titled row carries its key. EVERY namespaced key is named this
    // way -- the channel with its own copy, any other channel by its brand, the
    // non-channel families the gateway mints (cron, hook, app, subagent,
    // dashboard) by theirs, and a prefix none of them claims by that prefix --
    // so no row is an unexplained mono line. A legacy bare Slack ts is a Slack
    // thread. Only a key with no namespace at all stands alone.
    api.sessions.mockResolvedValue({ sessions: [
      { key: 'zzq-ok' }, { key: 'telegram_7781120043' }, { key: 'slack_1785861252.833429' },
      { key: 'discord_1418921033711624344' }, { key: 'cron:nightly-digest' }, { key: 'telegram_kirocrew_direct_4242', title: 'Deploy questions' },
      { key: 'hook_review-pr-123' }, { key: 'app:ops-mission-control' }, { key: 'subagent_a1b2c3' },
      { key: 'dashboard_chat-flaky-span' }, { key: 'imessage_+15551230000' }, { key: 'whatsapp:kirocrew:direct:44' },
      { key: 'fleet:worker-7' }, { key: '1785861252.833429' }, { key: 'zzq-bare' },
    ] })
    api.consolidateMemory
      .mockResolvedValueOnce({ ok: true })
      .mockRejectedValue(new Error('zzq-consolidate-failed'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const rows = within(await screen.findByTestId('consolidate-failed-keys')).getAllByRole('listitem')
    expect(rows.map(r => r.textContent)).toEqual([
      'Telegram chat 7781120043telegram_7781120043',
      'Slack thread 1785861252.833429slack_1785861252.833429',
      'Discord conversation 1418921033711624344discord_1418921033711624344',
      'Cron job nightly-digestcron:nightly-digest',
      // A title the list carries wins over the source label.
      'Deploy questionstelegram_kirocrew_direct_4242',
      'Webhook review-pr-123hook_review-pr-123',
      'App session ops-mission-controlapp:ops-mission-control',
      'Subagent a1b2c3subagent_a1b2c3',
      'Dashboard session chat-flaky-spandashboard_chat-flaky-span',
      'iMessage chat +15551230000imessage_+15551230000',
      'WhatsApp chat kirocrew:direct:44whatsapp:kirocrew:direct:44',
      'fleet session worker-7fleet:worker-7',
      'Slack thread 1785861252.8334291785861252.833429',
      'zzq-bare',
    ])
    expect(within(rows[0]).getByText('Telegram chat 7781120043').className).not.toContain('font-mono')
    expect(within(rows[0]).getByText('telegram_7781120043').className).toContain('font-mono')
    expect(within(rows[3]).getByText('cron:nightly-digest').className).toContain('font-mono')
    expect(within(rows[13]).getByText('zzq-bare').className).toContain('font-mono')
  })

  it('renders the failed rows as plain text, not as links', async () => {
    // Left to inherit the banner's danger accent, a list of session names read
    // as a list of links; nothing here navigates. The name takes the body
    // color, the key the muted mono, and no row holds an anchor, a button or
    // anything focusable -- the only control in the footer is the retry.
    api.sessions.mockResolvedValue({ sessions: [{ key: 'zzq-s1', title: 'Perf triage' }, { key: 'cron:nightly-digest' }] })
    api.consolidateMemory.mockRejectedValue(new Error('zzq-consolidate-failed'))
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await userEvent.click(await screen.findByRole('button', { name: /Summarize now/i }))

    const list = await screen.findByTestId('consolidate-failed-keys')
    expect(list.className).toContain('text-text')
    const rows = within(list).getAllByRole('listitem')
    expect(rows).toHaveLength(2)
    for (const row of rows) {
      expect(row.querySelector('a, button, [role="link"], [role="button"], [tabindex]')).toBeNull()
      for (const span of Array.from(row.querySelectorAll('span'))) {
        expect(span.className).toMatch(/text-(text|muted)/)
        expect(span.className).not.toMatch(/text-(accent|danger)|underline|cursor-pointer|hover:/)
      }
    }
    expect(within(rows[0]).getByText('Perf triage').className).toContain('text-text')
    expect(within(rows[0]).getByText('zzq-s1').className).toContain('text-muted')
    expect(within(rows[1]).getByText('Cron job nightly-digest').className).toContain('text-text')
  })

  it('clears the outcome message on its own timer', async () => {
    vi.useFakeTimers()
    renderWithProviders(<MemoryTab refreshTrigger={0} />)
    await act(async () => {})
    fireEvent.click(screen.getByRole('button', { name: /Summarize now/i }))
    await act(async () => {})
    expect(screen.getByText(/Summarized/)).toBeInTheDocument()

    await act(async () => { vi.advanceTimersByTime(4000) })
    expect(screen.queryByText(/Summarized/)).toBeNull()
  })
})
