import { describe, it, expect, vi, afterEach } from 'vitest'
import type { InstanceView } from '../api/client'
import { chainRows, visibleInstanceTabs } from '../components/InstanceTabBar'
import {
  announceChainedCrew,
  CHAIN_REFUSAL_MAX,
  CHAINED_CREW_MESSAGE,
  CHAINED_CREW_REFUSED_MESSAGE,
  CHAINED_HOST_MAX,
  CHAINED_NAME_MAX,
  clearChainRefusal,
  readChainRefusal,
  readChainedCrewNotice,
  subscribeChainRefusal,
} from '../lib/chainAnnounce'

/** Minimal InstanceView; only the fields the chain rules read matter. */
function inst(id: string, extra: Partial<InstanceView> = {}): InstanceView {
  return {
    id,
    name: id,
    ssh_host: `${id}-host`,
    remote_port: 5476,
    local_port: 0,
    ttl: '20h',
    remote_bin: '',
    connection_method: 'ssh',
    ssm_target: '',
    aws_profile: '',
    aws_region: '',
    ssm_run_as: '',
    was_connected: true,
    status: { instance_id: id, state: 'connected' },
    ...extra,
  }
}

function chained(id: string, parent: string, extra: Partial<InstanceView> = {}): InstanceView {
  return inst(id, { via_instance_id: parent, via_remote_port: 53999, ...extra })
}

describe('readChainedCrewNotice', () => {
  const good = { id: 'c-2', name: 'C', sshHost: 'c-host', remotePort: 5476, port: 53999 }

  it('keeps the announcing gateway\u2019s own id for the crew', () => {
    // This id is the whole point of the notice: the parent mints the token and
    // looks the crew up by ITS id, so an id derived on the host side equals it
    // only by luck and the parent then answers 404 for a crew it holds.
    expect(readChainedCrewNotice(good)?.id).toBe('c-2')
  })

  it('drops a notice with no id rather than adopting one we would have to invent', () => {
    const { id: _dropped, ...noId } = good
    expect(readChainedCrewNotice(noId)).toBeNull()
  })

  it('drops an id outside the registry grammar', () => {
    // It ends up in a request path on the announcing gateway, so a path-shaped
    // id would address a different route there.
    expect(readChainedCrewNotice({ ...good, id: 'victim/disconnect?x=' })).toBeNull()
    expect(readChainedCrewNotice({ ...good, id: 'C-2' })).toBeNull()
    expect(readChainedCrewNotice({ ...good, id: '-leading' })).toBeNull()
  })

  it('refuses a hop port outside the range but tolerates an unusable remote port', () => {
    // The hop port is dialled, so a bad one is fatal. The crew's own gateway port
    // is only a record here, so the row is still usable without it.
    expect(readChainedCrewNotice({ ...good, port: 0 })).toBeNull()
    expect(readChainedCrewNotice({ ...good, port: 70000 })).toBeNull()
    expect(readChainedCrewNotice({ ...good, remotePort: 999999 })?.remotePort).toBe(0)
  })

  it('caps the two free-text fields instead of trusting their length', () => {
    const long = readChainedCrewNotice({ ...good, name: 'n'.repeat(500), sshHost: 'h'.repeat(500) })
    expect(long?.name).toHaveLength(CHAINED_NAME_MAX)
    expect(long?.sshHost).toHaveLength(CHAINED_HOST_MAX)
  })

  it('drops a notice that is not an object at all', () => {
    expect(readChainedCrewNotice(null)).toBeNull()
    expect(readChainedCrewNotice('mc-instance-ready')).toBeNull()
  })
})

describe('chainRows', () => {
  it('puts each crew before the crews reached through it, and records the depth', () => {
    const rows = chainRows([inst('b'), chained('c', 'b'), chained('d', 'c'), inst('other')])
    expect(rows.map(r => r.inst.id)).toEqual(['b', 'c', 'd', 'other'])
    expect(rows.map(r => r.depth)).toEqual([0, 1, 2, 0])
    expect(rows.map(r => r.parentName)).toEqual(['', 'b', 'c', ''])
  })

  it('keeps the incoming order among siblings', () => {
    // Tab order is a user-visible preference; the tree must reorder nothing it
    // does not have to.
    const rows = chainRows([inst('b'), inst('a'), chained('c', 'b')])
    expect(rows.map(r => r.inst.id)).toEqual(['b', 'c', 'a'])
  })

  it('treats a crew whose parent has no tab as a root rather than hiding it', () => {
    // The parent was never connected, so it has no tab. Dropping the child would
    // hide a crew that is genuinely connected.
    const rows = chainRows([chained('c', 'never-connected')])
    expect(rows).toHaveLength(1)
    expect(rows[0].depth).toBe(0)
    expect(rows[0].parentName).toBe('')
  })

  it('marks a crew unreachable when an ancestor hop is down', () => {
    // They share the ancestor's tunnel, so a child cannot outlive it however
    // recently its own state said 'connected'.
    const rows = chainRows([
      inst('b', { status: { instance_id: 'b', state: 'error' } }),
      chained('c', 'b'),
      chained('d', 'c'),
    ])
    expect(rows.map(r => [r.inst.id, r.reachable])).toEqual([
      ['b', true],
      ['c', false],
      ['d', false],
    ])
  })

  it('keeps a whole healthy chain reachable', () => {
    const rows = chainRows([inst('b'), chained('c', 'b'), chained('d', 'c')])
    expect(rows.every(r => r.reachable)).toBe(true)
  })

  it('emits every crew exactly once even when the registry holds a loop', () => {
    // A hand-edited registry can name a cycle. A row the user can see is what
    // lets them disconnect it and fix the file.
    const rows = chainRows([chained('a', 'b'), chained('b', 'a')])
    expect(rows.map(r => r.inst.id).sort()).toEqual(['a', 'b'])
    expect(new Set(rows.map(r => r.inst.id)).size).toBe(rows.length)
  })

  it('leaves a list with no chained crew exactly as it was', () => {
    const flat = [inst('a'), inst('b'), inst('c')]
    const rows = chainRows(flat)
    expect(rows.map(r => r.inst.id)).toEqual(['a', 'b', 'c'])
    expect(rows.every(r => r.depth === 0 && r.parentName === '' && r.reachable)).toBe(true)
  })
})

describe('announceChainedCrew', () => {
  const notice = { id: 'c', name: 'C', sshHost: 'c-host', remotePort: 5476, port: 53999 }

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  /** Stand in for a pane: `self !== top`, with a recording parent. */
  function asPane() {
    const posted: unknown[] = []
    const parent = { postMessage: (data: unknown) => posted.push(data) }
    vi.stubGlobal('window', { self: {}, top: {}, parent })
    return posted
  }

  it('does nothing at top level, where there is no host to tell', () => {
    const posted: unknown[] = []
    const shared = {}
    vi.stubGlobal('window', {
      self: shared,
      top: shared,
      parent: { postMessage: (d: unknown) => posted.push(d) },
    })
    expect(announceChainedCrew(notice)).toBe(false)
    expect(posted).toEqual([])
  })

  it('carries the crew and the hop port, and no credential', () => {
    const posted = asPane()
    expect(announceChainedCrew(notice)).toBe(true)
    expect(posted).toHaveLength(1)
    const msg = posted[0] as Record<string, unknown>
    expect(msg.type).toBe(CHAINED_CREW_MESSAGE)
    expect(msg.id).toBe('c')
    expect(msg.port).toBe(53999)
    // The whole reason this notice may travel through frame code: it names a
    // crew and a port, and nothing that grants access to either.
    const keys = Object.keys(msg).join(' ')
    expect(keys).not.toMatch(/token|secret|cookie|credential/i)
  })

  it('refuses a payload with no usable hop port', () => {
    const posted = asPane()
    for (const port of [0, -1, 70000, 1.5, Number.NaN]) {
      expect(announceChainedCrew({ ...notice, port })).toBe(false)
    }
    expect(announceChainedCrew({ ...notice, id: '' })).toBe(false)
    expect(posted).toEqual([])
  })

  it('stays silent when the post itself throws', () => {
    // A pane whose host predates the message, or a browser that refuses the
    // post, must not break the connect that just succeeded.
    vi.stubGlobal('window', {
      self: {},
      top: {},
      parent: {
        postMessage: () => {
          throw new Error('cross-origin')
        },
      },
    })
    expect(announceChainedCrew(notice)).toBe(false)
  })
})

describe('the relayed refusal outlives the panel', () => {
  afterEach(() => clearChainRefusal())

  const refuse = (reason: string, id = 'gpu-box') =>
    window.dispatchEvent(
      new MessageEvent('message', {
        source: window,
        data: { type: CHAINED_CREW_REFUSED_MESSAGE, v: 1, id, reason },
      }),
    )

  it('holds a refusal that arrives while nothing is subscribed', () => {
    // The host answers while the user is still watching the crew connect, which
    // can be long after Settings was closed. A listener mounted with the panel
    // drops it, and the crew then reads as connected-but-missing with no reason.
    expect(readChainRefusal()).toBeNull()
    refuse('that would put 3 machines between this dashboard and the crew')
    expect(readChainRefusal()?.reason).toContain('3 machines')
    expect(readChainRefusal()?.id).toBe('gpu-box')
  })

  it('notifies a subscriber and clears on dismiss', () => {
    let hits = 0
    const stop = subscribeChainRefusal(() => {
      hits += 1
    })
    refuse('loop detected')
    expect(hits).toBe(1)
    clearChainRefusal()
    expect(hits).toBe(2)
    expect(readChainRefusal()).toBeNull()
    stop()
  })

  it('caps the relayed reason and ignores a message of another type', () => {
    refuse('x'.repeat(CHAIN_REFUSAL_MAX + 50))
    expect(readChainRefusal()?.reason.length).toBe(CHAIN_REFUSAL_MAX)
    clearChainRefusal()
    window.dispatchEvent(
      new MessageEvent('message', { source: window, data: { type: 'mc-host-model', v: 1 } }),
    )
    expect(readChainRefusal()).toBeNull()
  })
})

describe('an adopted crew and the tab filter', () => {
  it('gives no tab to a row that was only added', () => {
    // The contract the adoption path turns on. A row written by `POST
    // /api/instances` carries no connect intent, no status and no warm entry, so
    // adopting a crew WITHOUT connecting it leaves the promised top-level tab
    // missing and the crew reachable only from the Remote Crew list.
    const justAdded = chained('c', 'b', { was_connected: false, status: undefined })
    expect(visibleInstanceTabs([inst('b'), justAdded], {}).map(i => i.id)).toEqual(['b'])
  })

  it('gives it a tab once the connect intent is on the row', () => {
    const connected = chained('c', 'b', { was_connected: true, status: undefined })
    expect(visibleInstanceTabs([inst('b'), connected], {}).map(i => i.id)).toEqual(['b', 'c'])
  })
})
