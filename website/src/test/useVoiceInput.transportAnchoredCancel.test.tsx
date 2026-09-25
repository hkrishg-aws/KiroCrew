/**
 * A discard must reach the mechanism the REQUEST IN FLIGHT uses, not the one the
 * saved setting names.
 *
 * The setting describes the next utterance. Flip it while one utterance is open
 * and it names a transport nothing in flight is using, so a discard routed off it
 * lands on the wrong path: it closes a socket that was never opened, or it
 * skips the flag that drops a blob still on its way to the transcriber. Either
 * way the press changes the UI and the utterance survives, which is the one
 * outcome worse than offering no exit at all.
 *
 * So both cases here flip the setting AGAINST the live request. A test where the
 * two agree cannot tell a request-anchored discard from a setting-anchored one,
 * so it is the disagreeing case that carries the whole assertion.
 *
 * The matching visibility half — an exit is offered only where a press really
 * ends the session — is asserted at the UI in
 * `ChatInput.drainCancelAffordance.test.tsx`.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'

/** The streaming hook's state, driven per test. */
const streamState = { recording: false, draining: false }
const streamCancel = vi.fn()
const streamStop = vi.fn()

vi.mock('../hooks/useStreamingStt', () => ({
  streamingSupported: true,
  useStreamingStt: () => ({
    recording: streamState.recording,
    draining: streamState.draining,
    start: vi.fn(),
    stop: streamStop,
    switchDevice: vi.fn(),
    cancel: streamCancel,
  }),
}))

/** Whether the blob reached the transcriber is the batch case's whole verdict. */
const sttTranscribe = vi.fn().mockResolvedValue({ text: 'hello world' })
vi.mock('../api/client', () => ({ api: { sttTranscribe: (...a: unknown[]) => sttTranscribe(...a) } }))

interface FakeTrack { stop: ReturnType<typeof vi.fn>; readyState: string; label: string }
function makeStream() {
  const track: FakeTrack = { stop: vi.fn(), readyState: 'live', label: 'Mock Mic' }
  return { _track: track, getAudioTracks: () => [track], getTracks: () => [track] }
}

const recorders: MockMediaRecorder[] = []
const lastRecorder = () => recorders[recorders.length - 1] ?? null
class MockMediaRecorder {
  static isTypeSupported() { return true }
  state: 'inactive' | 'recording' = 'inactive'
  stream: unknown
  ondataavailable: ((e: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  constructor(stream: unknown) { this.stream = stream; recorders.push(this) }
  start() { this.state = 'recording' }
  stop() { this.state = 'inactive'; this.onstop?.() }
  feed(bytes = 200) { this.ondataavailable?.({ data: new Blob(['x'.repeat(bytes)]) }) }
}

class MockAudioContext {
  createMediaStreamSource() { return { connect() {} } }
  createAnalyser() {
    return { fftSize: 0, frequencyBinCount: 16, getByteTimeDomainData() {}, getByteFrequencyData() {}, connect() {} }
  }
  close() { return Promise.resolve() }
}

let getUserMedia: ReturnType<typeof vi.fn>

beforeEach(() => {
  recorders.length = 0
  streamState.recording = false
  streamState.draining = false
  streamCancel.mockClear()
  streamStop.mockClear()
  sttTranscribe.mockClear()
  getUserMedia = vi.fn().mockResolvedValue(makeStream())
  Object.defineProperty(navigator, 'mediaDevices', {
    value: { getUserMedia, enumerateDevices: vi.fn().mockResolvedValue([]) },
    configurable: true,
    writable: true,
  })
  vi.stubGlobal('MediaRecorder', MockMediaRecorder as unknown as typeof MediaRecorder)
  vi.stubGlobal('AudioContext', MockAudioContext as unknown as typeof AudioContext)
  vi.resetModules()
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

async function loadHook() {
  const mod = await import('../hooks/useVoiceInput')
  return mod.useVoiceInput
}

describe('cancel routes on the live request, not on the setting', () => {
  it('stream drain: ends the socket session even though the setting now says batch', async () => {
    const useVoiceInput = await loadHook()
    // A streaming utterance was released and is draining against its socket.
    streamState.draining = true
    const { result } = renderHook(
      ({ streaming }: { streaming: boolean }) => useVoiceInput(vi.fn(), { streaming }),
      // The user turned streaming OFF while this drain was open.
      { initialProps: { streaming: false } },
    )
    expect(result.current.transcribing).toBe(true)

    act(() => { result.current.cancel() })

    // The socket belongs to the request, so the discard must reach it. Routed on
    // the setting this call takes the batch path and the drain runs to a final.
    expect(streamCancel).toHaveBeenCalledTimes(1)
  })

  it('batch capture: drops the audio even though the setting now says streaming', async () => {
    const useVoiceInput = await loadHook()
    const { result, rerender } = renderHook(
      ({ streaming }: { streaming: boolean }) => useVoiceInput(vi.fn(), { streaming }),
      // Batch, because a batch recorder can only start while streaming is off.
      { initialProps: { streaming: false } },
    )
    await act(async () => { await result.current.start() })
    await waitFor(() => expect(result.current.recording).toBe(true))
    act(() => { lastRecorder()?.feed() })

    // The user turns streaming ON in Settings while this capture is still live.
    rerender({ streaming: true })
    act(() => { result.current.cancel() })

    // Capture really over, and the blob never reaches the transcriber. Routed on
    // the setting this call takes the socket path: the recorder keeps running
    // with a hot mic and the words the user threw away are still pending.
    expect(lastRecorder()?.state).toBe('inactive')
    await act(async () => { await Promise.resolve() })
    expect(sttTranscribe).not.toHaveBeenCalled()
  })

  it('nothing in flight: the setting is what is left to read, and stays honoured', async () => {
    const useVoiceInput = await loadHook()
    // No request at all, so there is no transport to anchor to. The press is
    // disarming a startup or a warm mic, which is the one thing the setting does
    // describe — and the streaming startup needs its own teardown.
    const { result } = renderHook(() => useVoiceInput(vi.fn(), { streaming: true }))

    act(() => { result.current.cancel() })

    expect(streamCancel).toHaveBeenCalledTimes(1)
  })
})

describe('the cancellability the UI gates on', () => {
  it('is true for a draining socket session', async () => {
    const useVoiceInput = await loadHook()
    streamState.draining = true
    const { result } = renderHook(() => useVoiceInput(vi.fn(), { streaming: true }))
    expect(result.current.drainCancellable).toBe(true)
  })

  it('is false for a batch transcription, whose blob is already posted', async () => {
    const useVoiceInput = await loadHook()
    const inbox = await import('../hooks/voiceTranscriptInbox')
    // The finding's own scenario: a batch `/api/stt` request is in flight and the
    // inbox announces it to every mounted instance, so this composer shows the
    // busy state for a request whose audio it cannot recall - while the setting
    // says streaming, because the user changed it during the request.
    const { result } = renderHook(() => useVoiceInput(vi.fn(), {
      streaming: true,
      sessionId: 's1',
      ownsSession: () => true,
      acceptsUnowned: true,
    }))
    act(() => { inbox.beginTranscription('s1') })
    await waitFor(() => expect(result.current.transcribing).toBe(true))

    // Being in flight is not being cancellable. The setting cannot make it so.
    expect(result.current.drainCancellable).toBe(false)
  })

  it('is false with nothing in flight at all', async () => {
    const useVoiceInput = await loadHook()
    const { result } = renderHook(() => useVoiceInput(vi.fn(), { streaming: true }))
    expect(result.current.drainCancellable).toBe(false)
  })
})
