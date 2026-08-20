'use client'

import { useEffect, useState, type CSSProperties } from 'react'
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useVoiceAssistant,
  useLocalParticipant,
  useConnectionState,
} from '@livekit/components-react'
import { ConnectionState } from 'livekit-client'
import '@livekit/components-styles'
import { X, Loader2, PhoneOff, AlertTriangle, Mic, MicOff, Settings } from 'lucide-react'
import { createClient } from '@/utils/supabase/client'

interface VoiceModalProps {
  chatSessionId: string
  isOpen: boolean
  onClose: () => void
}

const STATE_LABELS: Record<string, string> = {
  connecting: 'Connecting…',
  'pre-connect-buffering': 'Connecting…',
  initializing: 'Starting up…',
  idle: 'Ready',
  listening: "I'm Listening…",
  thinking: 'Thinking…',
  speaking: 'Speaking…',
  disconnected: 'Disconnected',
  failed: 'Connection failed',
}

type OrbState = 'idle' | 'connecting' | 'listening' | 'speaking'

// Collapses the richer set of states useVoiceAssistant() can report (e.g.
// "thinking", "pre-connect-buffering") down to the four visual states the
// orb itself distinguishes -- the status *text* below the orb still shows
// the full label via STATE_LABELS.
function toOrbState(voiceState: string): OrbState {
  if (voiceState === 'listening' || voiceState === 'speaking') return voiceState
  if (
    voiceState === 'connecting' ||
    voiceState === 'pre-connect-buffering' ||
    voiceState === 'initializing'
  ) {
    return 'connecting'
  }
  return 'idle'
}

// Filament rotation speed multiplier per state -- lower is faster.
const FILAMENT_SPEED: Record<OrbState, number> = {
  idle: 1,
  connecting: 1.6,
  listening: 0.68,
  speaking: 0.55,
}

// [min, max] opacity the outer bloom drifts between (orb-bloom-drift reads
// these off --bloom-min/--bloom-max). Unused for "speaking", which swaps to
// the amplitude-style scale pulse instead of the opacity drift.
const BLOOM_RANGE: Record<OrbState, [number, number]> = {
  idle: [0.28, 0.4],
  connecting: [0.16, 0.24],
  listening: [0.5, 0.68],
  speaking: [0.55, 0.55],
}

const NOISE_URL =
  "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")"

// CSS-only stand-in for the default LiveKit <BarVisualizer> -- a translucent
// ball of light (dark core, glowing filaments, soft dissolving edge) built
// as stacked absolutely-positioned layers rather than a solid glossy fill:
//   1. outer bloom (behind everything, no edge)
//   2. body -- inverted radial gradient (dark center, bright 65-85% ring)
//   3. inner filaments -- rotating blurred ellipses, additive blending
//   4. rim -- a partial-arc gradient ring, never a uniform border
//   5. core glow -- small off-center highlight
//   6. grain -- faint tiled noise
function VoiceOrb({ state }: { state: OrbState }) {
  const speed = FILAMENT_SPEED[state]
  const [bloomMin, bloomMax] = BLOOM_RANGE[state]
  const isSpeaking = state === 'speaking'

  return (
    <div className="relative w-56 h-56 shrink-0 flex items-center justify-center">
      {/* 1. Outer bloom -- ~1.8x the orb, no hard edge of its own.
             blur-[40px] rather than the visually-equivalent-looking 60px:
             filter: blur() forces a per-frame rasterize/composite pass
             (worse still once mix-blend-screen is layered on top of it, see
             layer 3 below), and radius drives that cost roughly
             quadratically -- 60->40px cuts it meaningfully without a
             visible difference at this element's size. */}
      <div
        className={`absolute rounded-full pointer-events-none blur-[40px] ${
          isSpeaking ? 'animate-orb-bloom-amplitude' : 'animate-orb-bloom-drift'
        }`}
        style={{
          width: '180%',
          height: '180%',
          background: 'radial-gradient(circle, #E040FB 0%, rgba(224,64,251,0.4) 42%, transparent 72%)',
          ...(isSpeaking
            ? { opacity: 0.55 }
            : ({ '--bloom-min': bloomMin, '--bloom-max': bloomMax } as CSSProperties)),
        }}
      />

      {/* 2. Body -- `isolate` keeps the filaments' mix-blend-screen contained
             to this sphere instead of blending with the page behind it. */}
      <div
        className="relative w-56 h-56 rounded-full overflow-hidden isolate animate-orb-breathe"
        style={{
          background:
            'radial-gradient(circle at 50% 50%, #0A0A0B 0%, rgba(10,10,11,0.88) 30%, rgba(124,58,237,0.22) 58%, rgba(224,64,251,0.55) 74%, rgba(192,38,211,0.3) 86%, rgba(10,10,11,0.1) 100%)',
        }}
      >
        {/* 3. Inner filaments -- each wrapper spins the whole layer; the
               ellipse inside is offset from center so it sweeps an arc
               rather than spinning in place. Blur radii trimmed from their
               original 24/30/20/26px: mix-blend-screen means the browser
               can't cache a precomposited blur result and reuse it across
               frames the way a plain transformed layer would -- the blend
               has to be recomputed against whatever's behind it every
               frame, so these are the most expensive layers in the orb by
               far, and radius is the main cost lever available without
               dropping the blend effect itself. */}
        <div className="absolute inset-0 animate-orb-orbit-a" style={{ animationDuration: `${18 * speed}s` }}>
          <div
            className="absolute w-[70%] h-[26%] rounded-full blur-[16px] mix-blend-screen"
            style={{ top: '50%', left: '50%', background: '#E040FB', opacity: 0.75, transform: 'translate(-42%, -65%) rotate(12deg)' }}
          />
        </div>
        <div className="absolute inset-0 animate-orb-orbit-b" style={{ animationDuration: `${26 * speed}s` }}>
          <div
            className="absolute w-[58%] h-[22%] rounded-full blur-[20px] mix-blend-screen"
            style={{ top: '50%', left: '50%', background: '#C026D3', opacity: 0.7, transform: 'translate(-30%, 20%) rotate(-30deg)' }}
          />
        </div>
        <div className="absolute inset-0 animate-orb-orbit-c" style={{ animationDuration: `${34 * speed}s` }}>
          <div
            className="absolute w-[52%] h-[18%] rounded-full blur-[14px] mix-blend-screen"
            style={{ top: '50%', left: '50%', background: '#7C3AED', opacity: 0.65, transform: 'translate(20%, 15%) rotate(55deg)' }}
          />
        </div>
        <div className="absolute inset-0 animate-orb-orbit-d" style={{ animationDuration: `${22 * speed}s` }}>
          <div
            className="absolute w-[38%] h-[14%] rounded-full blur-[18px] mix-blend-screen"
            style={{ top: '50%', left: '50%', background: '#FF6FD8', opacity: 0.5, transform: 'translate(-70%, -30%) rotate(-15deg)' }}
          />
        </div>

        {/* 4. Rim -- a conic gradient masked down to a ~1.5px ring so it's
               bright on one arc and gone on the rest, never a uniform border. */}
        <div
          className="absolute inset-0 rounded-full pointer-events-none"
          style={{
            background:
              'conic-gradient(from 210deg, rgba(255,255,255,0.45), rgba(255,255,255,0) 35%, rgba(255,255,255,0) 62%, rgba(255,255,255,0.18) 82%, rgba(255,255,255,0) 100%)',
            WebkitMaskImage: 'radial-gradient(farthest-side, transparent calc(100% - 1.5px), #000 calc(100% - 1.5px))',
            maskImage: 'radial-gradient(farthest-side, transparent calc(100% - 1.5px), #000 calc(100% - 1.5px))',
          }}
        />

        {/* 5. Core glow -- small, off-center in the upper-left third. */}
        <div
          className="absolute w-20 h-20 rounded-full pointer-events-none blur-[26px]"
          style={{
            top: '26%',
            left: '28%',
            transform: 'translate(-50%, -50%)',
            background: 'radial-gradient(circle, rgba(255,255,255,0.6) 0%, rgba(255,111,216,0.35) 45%, transparent 75%)',
            opacity: 0.35,
          }}
        />

        {/* 6. Grain -- faint tiled feTurbulence noise. */}
        <div
          className="absolute inset-0 rounded-full pointer-events-none opacity-[0.04] mix-blend-overlay"
          style={{ backgroundImage: NOISE_URL, backgroundSize: '120px 120px' }}
        />
      </div>
    </div>
  )
}

function TopCloseButton({ onClose }: { onClose: () => void }) {
  return (
    <button
      type="button"
      onClick={onClose}
      title="Close"
      className="absolute top-6 right-6 z-10 p-2.5 rounded-full bg-white/5 hover:bg-white/10 text-white/50 hover:text-white/80 transition-colors cursor-pointer"
    >
      <X className="w-5 h-5" />
    </button>
  )
}

// Guards against a silent hang on "Connecting..." -- if the WebRTC handshake
// with LiveKit Cloud hasn't reached Connected within CONNECT_TIMEOUT_MS
// (e.g. a bad/misconfigured NEXT_PUBLIC_LIVEKIT_URL, or the LiveKit project
// unreachable), surface it as a visible error instead of spinning forever.
// Must render as a child of <LiveKitRoom> -- useConnectionState() reads off
// its context, same as every other @livekit/components-react hook.
const CONNECT_TIMEOUT_MS = 15000

function ConnectionWatcher({ onTimeout }: { onTimeout: () => void }) {
  const connectionState = useConnectionState()

  useEffect(() => {
    if (connectionState === ConnectionState.Connected) return
    const timer = setTimeout(onTimeout, CONNECT_TIMEOUT_MS)
    return () => clearTimeout(timer)
  }, [connectionState, onTimeout])

  return null
}

// Split out from VoiceModal because useVoiceAssistant()/useLocalParticipant()
// (and every other @livekit/components-react hook) reads room state from the
// <LiveKitRoom> context provider, so it must render as a *child* of
// <LiveKitRoom> rather than alongside it.
function ActiveVoiceSession({ onClose }: { onClose: () => void }) {
  const { state } = useVoiceAssistant()
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant()

  return (
    <>
      {/* Plays the agent's synthesized speech back through the browser --
          useVoiceAssistant only visualizes track state, it doesn't attach
          the audio to an <audio> element itself. */}
      <RoomAudioRenderer />

      <div className="flex-1 flex flex-col items-center justify-center gap-8 w-full">
        <VoiceOrb state={toOrbState(state)} />
        <p className="text-white/70 text-lg font-light tracking-wide">
          {STATE_LABELS[state] ?? state}
        </p>
      </div>

      <div className="flex items-center justify-center gap-6 pb-2">
        <button
          type="button"
          title="Settings"
          className="bg-white/10 hover:bg-white/20 p-4 rounded-full text-white/70 transition-colors cursor-pointer"
        >
          <Settings className="w-5 h-5" />
        </button>

        <div className="relative">
          {state === 'listening' && (
            <span className="absolute inset-0 rounded-full bg-fuchsia-500/50 animate-orb-ring-pulse pointer-events-none" />
          )}
          <button
            type="button"
            onClick={() => localParticipant.setMicrophoneEnabled(!isMicrophoneEnabled)}
            title={isMicrophoneEnabled ? 'Mute microphone' : 'Unmute microphone'}
            className="relative bg-fuchsia-600 hover:bg-fuchsia-500 p-6 rounded-full shadow-[inset_0_2px_4px_rgba(255,255,255,0.4),_0_0_20px_rgba(217,70,239,0.6)] text-white transition-colors cursor-pointer"
          >
            {isMicrophoneEnabled ? <Mic className="w-7 h-7" /> : <MicOff className="w-7 h-7" />}
          </button>
        </div>

        <button
          type="button"
          onClick={onClose}
          title="End Call"
          className="bg-red-500/20 hover:bg-red-500/40 text-red-500 p-4 rounded-full transition-colors cursor-pointer"
        >
          <PhoneOff className="w-5 h-5" />
        </button>
      </div>
    </>
  )
}

export default function VoiceModal({ chatSessionId, isOpen, onClose }: VoiceModalProps) {
  const [token, setToken] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  // Backgrounded tab -> pause the orb's rotating/blurred/blended layers
  // (toggles .voice-orb-paused, see globals.css) so they stop costing
  // CPU/GPU while nobody can see them -- the call itself keeps running via
  // LiveKitRoom/RoomAudioRenderer, only the visual is paused.
  const [tabHidden, setTabHidden] = useState(false)

  useEffect(() => {
    if (!isOpen) return
    const onVisibilityChange = () => setTabHidden(document.hidden)
    onVisibilityChange()
    document.addEventListener('visibilitychange', onVisibilityChange)
    return () => document.removeEventListener('visibilitychange', onVisibilityChange)
  }, [isOpen])

  // Mints a fresh LiveKit token every time the modal opens, scoped to the
  // active chat session -- voice/router.py's POST /api/voice/token names
  // the LiveKit room after chat_session_id, so a stale token from a
  // previous session would join the wrong room.
  useEffect(() => {
    if (!isOpen) {
      setToken(null)
      setError(null)
      return
    }

    let cancelled = false

    const fetchToken = async () => {
      setLoading(true)
      setError(null)
      try {
        if (!process.env.NEXT_PUBLIC_LIVEKIT_URL) {
          throw new Error('Voice is not configured: NEXT_PUBLIC_LIVEKIT_URL is missing.')
        }

        const supabase = createClient()
        const { data: { session } } = await supabase.auth.getSession()
        const accessToken = session?.access_token || ''

        const response = await fetch('/api/voice/token', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${accessToken}`,
          },
          body: JSON.stringify({ chat_session_id: chatSessionId }),
        })

        if (!response.ok) {
          throw new Error('Failed to start the voice session. Please try again.')
        }

        const data = await response.json()
        if (!cancelled) setToken(data.token)
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to start the voice session.')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    fetchToken()
    return () => {
      cancelled = true
    }
  }, [isOpen, chatSessionId])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-md">
      <div className="fixed inset-6 md:inset-12 z-50 flex flex-col">
        <div
          className={`bg-zinc-950 rounded-[40px] border border-white/10 shadow-2xl overflow-hidden relative flex flex-col items-center justify-between p-8 flex-1 ${
            tabHidden ? 'voice-orb-paused' : ''
          }`}
        >
          <TopCloseButton onClose={onClose} />

          {loading && (
            <div className="flex-1 flex flex-col items-center justify-center gap-8 w-full">
              <VoiceOrb state="connecting" />
              <p className="text-white/70 text-lg font-light tracking-wide flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Connecting…
              </p>
            </div>
          )}

          {error && !loading && (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 px-6 text-center">
              <AlertTriangle className="w-8 h-8 text-red-500" />
              <p className="text-white/70 text-lg font-light tracking-wide">{error}</p>
              <button
                type="button"
                onClick={onClose}
                className="mt-2 px-5 py-2.5 rounded-full bg-white/10 hover:bg-white/20 text-white/80 text-sm font-medium transition-colors cursor-pointer"
              >
                Close
              </button>
            </div>
          )}

          {token && !loading && !error && (
            <LiveKitRoom
              token={token}
              serverUrl={process.env.NEXT_PUBLIC_LIVEKIT_URL}
              connect={true}
              audio={true}
              onError={(err) => setError(err.message || 'Voice connection error.')}
              onDisconnected={onClose}
              className="flex-1 flex flex-col items-center justify-between w-full"
            >
              <ConnectionWatcher
                onTimeout={() => setError('Connection timed out. Please check your voice service configuration and try again.')}
              />
              <ActiveVoiceSession onClose={onClose} />
            </LiveKitRoom>
          )}
        </div>
      </div>
    </div>
  )
}
