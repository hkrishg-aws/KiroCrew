/**
 * The drain's exit and the drain's cancellability must agree.
 *
 * One rule, asserted in both directions: a discard control on screen means
 * pressing it really ends the session, and a wait that cannot be ended offers no
 * control. The failure this forbids is the middle case - a control that restyles
 * the strip while the work carries on, which reads as a stop that happened.
 *
 * Both directions run over BOTH transports, and each case flips the streaming
 * SETTING against the transport actually in flight. That pairing is what gives
 * the suite teeth: while setting and transport agree, a gate reading either one
 * looks correct, so a suite that exercises only the streaming path with the
 * setting agreeing cannot see a batch window that shows an exit and discards
 * nothing.
 *
 * The capture-phase Escape case is the passing control: it holds whatever the
 * drain window does, so a green suite here cannot come from a broken harness.
 * The "really cancels" half of the rule is asserted at the hooks, in
 * useVoiceInput.transportAnchoredCancel.test.tsx and
 * useStreamingStt.drainCancel.test.tsx.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, fireEvent } from '@testing-library/react'
import { renderWithProviders } from './helpers'
import ChatInput from '../components/ChatInput'
import { ComposerVoiceSliceOverride } from '../chat-core/composer/Composer'
import { createAudioSample } from '../hooks/mic'

vi.mock('../components/Strands', () => ({
  __esModule: true,
  default: () => <div data-testid="strands-stub" />,
  strandsSupported: () => true,
}))

const sampleRef = { current: createAudioSample() }
const base = { value: '', onChange: vi.fn(), onSend: vi.fn() }

/**
 * The released-utterance window, with no transport chosen yet: capture is over,
 * so `voiceRecording` is false, while the composer still holds a dictation.
 */
const DRAIN = {
  voiceRecording: false,
  voiceTranscribing: true,
  voiceDictationPanel: true,
  voiceSampleRef: sampleRef,
  voiceDeviceLabel: 'Mic',
}

/**
 * The two transports an in-flight utterance can be on, each with the streaming
 * setting flipped to the OTHER one - the mid-request Settings change that makes
 * setting and transport disagree.
 */
const TRANSPORTS = [
  {
    /** Socket held open by this composer's own session, so a discard closes it. */
    transport: 'streaming',
    cancellable: true,
    props: { voiceDrainCancellable: true, voiceStreaming: false },
  },
  {
    /** Blob already POSTed to the transcriber, so no press can call it back. */
    transport: 'batch',
    cancellable: false,
    props: { voiceDrainCancellable: false, voiceStreaming: true },
  },
]

/** The stage-announced half of the same window, where a figure is reportable. */
const DOWNLOAD = { done: 400_000_000, total: 1_600_000_000, stage: 'downloading' as const }

beforeEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
  vi.stubGlobal('matchMedia', (q: string) => ({
    matches: false, media: q, addEventListener: vi.fn(), removeEventListener: vi.fn(),
  }))
})

describe.each(TRANSPORTS)('drain on the $transport transport', ({ cancellable, props }) => {
  const render = (extra: Record<string, unknown> = {}) => {
    const onVoiceCancel = vi.fn()
    renderWithProviders(
      <ComposerVoiceSliceOverride inputProps={{ ...DRAIN, ...props, onVoiceCancel, ...extra }}>
        <ChatInput {...base} />
      </ComposerVoiceSliceOverride>,
    )
    return onVoiceCancel
  }

  it('offers an exit exactly when this request can be called off', () => {
    render()
    expect(!!screen.queryByTestId('voice-drain-cancel')).toBe(cancellable)
  })

  it('offers the same exit, and only then, while a model download reports progress', () => {
    render({ voiceDownload: DOWNLOAD })
    expect(screen.getByTestId('voice-status-download')).toBeInTheDocument()
    expect(!!screen.queryByTestId('voice-drain-cancel')).toBe(cancellable)
  })

  it('pressing the exit reaches the discard, which is the path that ends the session', () => {
    const onVoiceCancel = render()
    const button = screen.queryByTestId('voice-drain-cancel')
    if (!cancellable) {
      // Nothing to press is the correct outcome here; the wait is explained by
      // the composer's own transcribing placeholder instead.
      expect(button).toBeNull()
      return
    }
    fireEvent.click(button!)
    expect(onVoiceCancel).toHaveBeenCalledTimes(1)
  })
})

describe('the streaming drain, in full', () => {
  const STREAM = { ...DRAIN, ...TRANSPORTS[0].props }

  it('names what the silent wait is waiting on', () => {
    renderWithProviders(
      <ComposerVoiceSliceOverride inputProps={{ ...STREAM, onVoiceCancel: vi.fn() }}>
        <ChatInput {...base} />
      </ComposerVoiceSliceOverride>,
    )
    const strip = screen.getByTestId('voice-status-draining')
    expect(strip).toHaveTextContent('Waiting for the speech model.')
    expect(strip).toHaveTextContent('Your dictation is kept and will be transcribed.')
  })

  it('reaches the discard by touch, with a label that names what is discarded', () => {
    renderWithProviders(
      <ComposerVoiceSliceOverride inputProps={{ ...STREAM, onVoiceCancel: vi.fn() }}>
        <ChatInput {...base} />
      </ComposerVoiceSliceOverride>,
    )
    const button = screen.getByTestId('voice-drain-cancel')
    expect(button.tagName).toBe('BUTTON')
    expect(button).toHaveAccessibleName('Discard dictation')
    expect(button.textContent).toContain('Discard dictation')
  })
})

describe('drain cancel - nothing to cancel means nothing shown', () => {
  const STREAM = { ...DRAIN, ...TRANSPORTS[0].props }

  it('shows no discard when no dictation is in flight', () => {
    renderWithProviders(
      <ComposerVoiceSliceOverride inputProps={{ ...STREAM, voiceTranscribing: false, voiceDrainCancellable: false, onVoiceCancel: vi.fn() }}>
        <ChatInput {...base} />
      </ComposerVoiceSliceOverride>,
    )
    expect(screen.queryByTestId('voice-drain-cancel')).toBeNull()
    expect(screen.queryByTestId('voice-status-draining')).toBeNull()
  })

  it('shows no discard while capture is still live, where the panel owns the window', () => {
    renderWithProviders(
      <ComposerVoiceSliceOverride inputProps={{ ...STREAM, voiceRecording: true, onVoiceCancel: vi.fn() }}>
        <ChatInput {...base} />
      </ComposerVoiceSliceOverride>,
    )
    expect(screen.queryByTestId('voice-drain-cancel')).toBeNull()
  })

  it('shows no discard when the host wires no discard at all', () => {
    renderWithProviders(
      <ComposerVoiceSliceOverride inputProps={{ ...STREAM, onVoiceCancel: undefined }}>
        <ChatInput {...base} />
      </ComposerVoiceSliceOverride>,
    )
    expect(screen.queryByTestId('voice-drain-cancel')).toBeNull()
  })
})

describe('control - the capture-phase Escape exit is unaffected', () => {
  it('Escape during capture still reaches the discard', () => {
    const onVoiceCancel = vi.fn()
    renderWithProviders(
      <ComposerVoiceSliceOverride inputProps={{ ...DRAIN, ...TRANSPORTS[0].props, voiceRecording: true, onVoiceCancel }}>
        <ChatInput {...base} />
      </ComposerVoiceSliceOverride>,
    )
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onVoiceCancel).toHaveBeenCalledTimes(1)
  })
})
