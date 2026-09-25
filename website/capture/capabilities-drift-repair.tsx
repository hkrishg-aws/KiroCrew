/**
 * Capture page for the Capabilities pane's drift repair path (PR #13523).
 *
 * Scenes (`?scene=`):
 *   drift   -- the member's agent file changed outside the page: the notice names
 *              the Review changes button, which is enabled with no edit.
 *   refused -- the same pane after Review + Save on a file whose drift sits in a
 *              setting the page cannot show: the 409 `unreviewable_drift` refusal
 *              names the setting and the file. The runner performs the clicks.
 *
 * A capture page has no gateway behind it, so the pane's three calls are answered
 * here with the shapes the backend returns.
 */
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { initI18n } from '../src/i18n'
import CrewCapabilitiesPane from '../src/components/crew/CrewCapabilitiesPane'
import '../src/index.css'

initI18n('en')

const params = new URLSearchParams(location.search)
const theme = params.get('theme') || 'dark'
document.documentElement.setAttribute('data-theme', theme)

const view = {
  schema_version: 1, member: 'atlas-writer', mode: 'inherited', revision: 'r1',
  template: { name: 'atlas', source: 'custom', scope: 'global', available: true },
  rows: [
    { section: 'tools', id: '@search/read', label: 'Search read', state: 'inherited', present: true, value: true },
    { section: 'allowedTools', id: '@search/read', label: 'Search read', state: 'local', present: true, value: true },
    { section: 'mcpServers', id: 'search', label: 'Search', state: 'local', present: true, value: { command: 'search-cli', args: ['--read'], env: {} } },
    { section: 'skills', id: 'catalog/review', label: 'Review skill', state: 'inherited', present: true, value: null },
  ],
  skills: [{ id: 'catalog/review', label: 'Review skill', shared_reference: true }],
  connections: [],
  parent_changes: [],
  runtime: { status: 'failed', saved_revision: 'r1', sessions: [], error_code: 'materialization_changed' },
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })

const realFetch = window.fetch
window.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = String(typeof input === 'string' ? input : (input as Request).url ?? input)
  const method = (init?.method || 'GET').toUpperCase()
  if (url.includes('/capabilities')) {
    if (method === 'GET') return json(view)
    if (method === 'POST') return json({ ...view, preview_token: 'signed-preview', impact: [] })
    if (method === 'PUT') {
      return json({ error: 'unreviewable_drift', code: 'unreviewable_drift', keys: ['toolsSettings'], file: 'crew-3f9a1c2e7b4d.json' }, 409)
    }
  }
  if (url.includes('/api/')) return json({})
  return realFetch(input, init)
}) as typeof window.fetch

const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={qc}>
    <MemoryRouter initialEntries={['/capabilities']}>
      <div style={{ background: 'var(--bg)', color: 'var(--text)', padding: 24 }} data-capture-root>
        <div style={{ maxWidth: 720, height: 560, display: 'flex' }} className="rounded-lg border border-border">
          <CrewCapabilitiesPane member="atlas-writer" members={['atlas-writer']} hidden={false} onDirtyChange={() => {}} onBusyChange={() => {}} onSaved={() => {}} />
        </div>
      </div>
    </MemoryRouter>
  </QueryClientProvider>,
)
