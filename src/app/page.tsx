'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/utils/supabase/client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import CanvasWorkspace from '@/components/CanvasWorkspace'
import {
  Plus,
  MessageSquare,
  Trash2,
  LogOut,
  Menu,
  X,
  Send,
  Bot,
  User,
  Copy,
  Check,
  Brain,
  Zap,
  Flame,
  ChevronDown,
  Cpu,
  Sun,
  Moon,
  PanelLeftClose,
  Edit2,
  AlertTriangle,
  Paperclip,
  FileText,
  Database,
  Settings,
  Monitor,
  ChevronRight,
  Bell,
  GitBranch,
  Mail,
  ListChecks,
  Circle,
  CheckCircle2,
  Calendar,
  HardDrive,
  MoreVertical,
  Unlink,
  PanelRightOpen,
  PanelRightClose,
  ExternalLink
} from 'lucide-react'

interface UserProfile {
  id: string
  email?: string
}

interface ChatSession {
  id: string
  title: string
  created_at: string
  user_id: string
}

interface Source {
  title: string
  url: string
  snippet: string
}

interface CanvasDocument {
  title: string
  content: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  provider_used?: string
  sources?: Source[]
  attachedFileName?: string
  // Set when this turn's raw content contained a <canvas>...</canvas> block
  // (see parseCanvasStream) -- the chat bubble renders a placeholder card
  // instead of the raw markdown in that case, see the messages.map render.
  canvasDocument?: CanvasDocument
}

interface MemoryProfile {
  narrative: string
  updated_at: string | null
}

interface Task {
  id: string
  title: string
  description?: string | null
  due_at?: string | null
  status: 'pending' | 'completed'
  created_at: string
  notified_at?: string | null
}

const THEME_OPTIONS = [
  { id: 'system', label: 'System', icon: Monitor },
  { id: 'light', label: 'Light', icon: Sun },
  { id: 'dark', label: 'Dark', icon: Moon }
]

// Connector toggles offered in the chat input's "+" menu (Phase 6/7).
// `provider` is the Supabase Auth OAuth provider id used to obtain this
// connector's per-user access token via supabase.auth.linkIdentity when one
// isn't already on hand -- see toggleConnector. All four google_* entries
// share one Google OAuth token (providerTokens.google) -- Google Workspace
// is a single MCP session/connector server-side -- but are toggled and sent
// to the backend as independent ids (google_gmail/google_docs/
// google_calendar/google_drive) so the backend only offers the model
// whichever of Gmail/Docs/Calendar/Drive's tools the user actually enabled,
// instead of every Workspace tool at once (see
// mcp_manager.filter_workspace_tools).
const CONNECTORS: { id: string; label: string; icon: typeof GitBranch; provider: 'github' | 'google' }[] = [
  { id: 'github', label: 'GitHub', icon: GitBranch, provider: 'github' },
  { id: 'google_gmail', label: 'Gmail', icon: Mail, provider: 'google' },
  { id: 'google_docs', label: 'Google Docs', icon: FileText, provider: 'google' },
  { id: 'google_calendar', label: 'Google Calendar', icon: Calendar, provider: 'google' },
  { id: 'google_drive', label: 'Google Drive', icon: HardDrive, provider: 'google' }
]

const PROVIDERS = [
  { id: 'gemini', name: 'Google Gemini', icon: Flame, color: 'text-violet-600 bg-violet-50 border-violet-200 hover:bg-violet-100', accentColor: 'violet' },
  { id: 'openai', name: 'OpenAI ChatGPT', icon: Zap, color: 'text-emerald-600 bg-emerald-50 border-emerald-200 hover:bg-emerald-100', accentColor: 'emerald' },
  { id: 'claude', name: 'Anthropic Claude', icon: Brain, color: 'text-amber-600 bg-amber-50 border-amber-200 hover:bg-amber-100', accentColor: 'amber' },
  { id: 'groq', name: 'Groq LPU', icon: Cpu, color: 'text-blue-600 bg-blue-50 border-blue-200 hover:bg-blue-100', accentColor: 'blue' },
  { id: 'mock', name: 'Mock AI Provider', icon: Bot, color: 'text-slate-600 bg-slate-50 border-slate-200 hover:bg-slate-100', accentColor: 'slate' }
]

// Canvas (Phase 4): matches the SYSTEM_PROMPT's "## Canvas documents" rule
// (see backend/api/chat/services.py) that instructs the model to wrap an
// entire long-form response in these tags. Module-level and pure (no
// component state) since both the live SSE loop (handleSendMessage) and the
// persisted-history loader (fetchMessages) need to run the exact same
// extraction against a raw content string.
const CANVAS_OPEN_TAG = '<canvas>'
const CANVAS_CLOSE_TAG = '</canvas>'

interface ParsedCanvasStream {
  /** `raw` with the entire <canvas>...</canvas> region (or, mid-stream, everything from <canvas> onward) removed -- this is what the chat bubble renders. */
  chatText: string
  /** The content between the tags, or everything after <canvas> so far if the closing tag hasn't streamed in yet. null if no <canvas> tag is present at all. */
  canvasContent: string | null
}

function parseCanvasStream(raw: string): ParsedCanvasStream {
  const openIdx = raw.indexOf(CANVAS_OPEN_TAG)
  if (openIdx === -1) {
    return { chatText: raw, canvasContent: null }
  }

  const beforeTag = raw.slice(0, openIdx)
  const afterOpen = raw.slice(openIdx + CANVAS_OPEN_TAG.length)
  const closeIdx = afterOpen.indexOf(CANVAS_CLOSE_TAG)

  if (closeIdx === -1) {
    // Still streaming the document -- everything seen after <canvas> so far
    // is document content, not chat text, even though </canvas> hasn't
    // arrived yet.
    return { chatText: beforeTag, canvasContent: afterOpen }
  }

  const canvasContent = afterOpen.slice(0, closeIdx)
  const afterClose = afterOpen.slice(closeIdx + CANVAS_CLOSE_TAG.length)
  return { chatText: `${beforeTag}${afterClose}`, canvasContent }
}

// Best-effort title for the canvas panel/placeholder card -- pulled from the
// document's own content rather than asking the model for a separate title
// out-of-band. Both CanvasWorkspace's header and the chat bubble placeholder
// card key off this same function (see the live SSE loop, the final flush,
// and fetchMessages below), so a better title here improves both instantly.
function deriveCanvasTitle(markdown: string): string {
  const trimmed = markdown.trim()
  if (!trimmed) return 'Untitled Canvas Document'

  // Strongest signal: an explicit H1, then H2 heading -- strip markdown
  // emphasis chars so e.g. "# **Patriotism**" reads as "Patriotism".
  const headingMatch = trimmed.match(/^\s{0,3}#{1,2}\s+(.+)$/m)
  if (headingMatch) {
    const heading = headingMatch[1].replace(/[*_`]/g, '').trim()
    if (heading) return heading.slice(0, 80)
  }

  // No heading yet (still streaming before one has arrived, or the model
  // wrote prose without one) -- fall back to the first non-empty line,
  // stripped of leading markdown punctuation (list/quote markers, stray
  // '#'s) and truncated to ~7 words, so it still reads as a title rather
  // than a mid-sentence fragment.
  const firstLine = trimmed.split(/\r?\n/).find(line => line.trim().length > 0) || ''
  const cleanedLine = firstLine.replace(/^[#>\-*\s]+/, '').replace(/[*_`]/g, '').trim()
  if (!cleanedLine) return 'Untitled Canvas Document'

  const words = cleanedLine.split(/\s+/)
  const title = words.slice(0, 7).join(' ')
  return (words.length > 7 ? `${title}…` : title).slice(0, 80)
}

// Upper bound on how often a still-streaming canvas document is flushed into
// canvasDoc (the CanvasWorkspace panel's state), independent of how often
// chat tokens arrive. CanvasWorkspace calls editor.commands.setContent() on
// every genuinely new `content` value it receives (see its guarded sync
// effect), which fully replaces the ProseMirror doc -- doing that on every
// single streamed token (which can arrive many times a second) would visibly
// thrash the editor. This keeps updates feeling live (~8/sec) while bounding
// the update rate, satisfying Phase 4's "Render Stability" requirement.
const CANVAS_STREAM_FLUSH_INTERVAL_MS = 120

export default function Dashboard() {
  const router = useRouter()
  const supabase = createClient()

  // App state
  const [user, setUser] = useState<UserProfile | null>(null)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])

  // UI states
  const [inputText, setInputText] = useState('')
  const [selectedProvider, setSelectedProvider] = useState('groq')
  const [isStreaming, setIsStreaming] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [providerDropdownOpen, setProviderDropdownOpen] = useState(false)
  const [profileMenuOpen, setProfileMenuOpen] = useState(false)
  const [themeMenuOpen, setThemeMenuOpen] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [theme, setTheme] = useState('system')

  // New Feature States
  const [showLogoutModal, setShowLogoutModal] = useState(false)
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null)
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [editTitleText, setEditTitleText] = useState('')
  const [sessionMenuOpenId, setSessionMenuOpenId] = useState<string | null>(null)
  const [plusMenuOpen, setPlusMenuOpen] = useState(false)
  const [attachedFile, setAttachedFile] = useState<File | null>(null)
  const [showMemory, setShowMemory] = useState(false)
  const [memoryProfile, setMemoryProfile] = useState<MemoryProfile>({ narrative: '', updated_at: null })
  const [loadingMemory, setLoadingMemory] = useState(false)
  const [showClearMemoryConfirm, setShowClearMemoryConfirm] = useState(false)
  const [deletingMemory, setDeletingMemory] = useState(false)

  // Task Manager (Phase 6)
  const [showTasks, setShowTasks] = useState(false)
  const [tasks, setTasks] = useState<Task[]>([])
  const [loadingTasks, setLoadingTasks] = useState(false)
  const [taskStatusFilter, setTaskStatusFilter] = useState<'pending' | 'completed'>('pending')
  const [taskToDelete, setTaskToDelete] = useState<string | null>(null)
  const [deletingTask, setDeletingTask] = useState(false)

  // Connector toggles (Phase 6): which MCP connectors are active this turn,
  // and the per-user OAuth access token backing each one (if the user has
  // signed in with / linked that provider -- see toggleConnector). Neither
  // token is ever persisted anywhere but this in-memory session state; both
  // are sent per-request in handleSendMessage's FormData, never stored.
  const [enabledConnectors, setEnabledConnectors] = useState<string[]>([])
  const [providerTokens, setProviderTokens] = useState<{ google?: string; github?: string }>({})
  // Persistent per-provider link status, derived from user.identities on
  // load -- unlike providerTokens above, this survives reloads even once
  // the OAuth access token Supabase attached right after the exchange has
  // aged out of the session, so the "Connected" badge stays accurate.
  const [linkedProviders, setLinkedProviders] = useState<{ google: boolean; github: boolean }>({ google: false, github: false })
  // Whether the backend actually has a *usable* stored refresh_token per
  // provider (GET /api/connectors/status), as opposed to linkedProviders
  // above merely reflecting that the identity is linked at all. These
  // diverge for any Google identity linked before the offline-access
  // refresh-token flow existed (or via a plain sign-in through Google
  // rather than this app's own connector toggle) -- see toggleConnector,
  // which uses this to tell the user a disconnect+reconnect is actually
  // required instead of silently enabling a connector that can never open
  // a Workspace session (Supabase's linkIdentity() refuses to re-run for
  // an identity already linked to the current user, so simply retrying the
  // normal link flow can't self-heal this).
  const [hasStoredRefreshToken, setHasStoredRefreshToken] = useState<{ google: boolean; github: boolean }>({ google: false, github: false })

  // Toast notification (OAuth redirect outcomes, connector state changes)
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null)

  // Notification Bell (Phase 6): due reminders the background scheduler has
  // already flagged (see backend/core/scheduler.py), polled periodically
  // since nothing pushes these to the client in real time.
  const [dueReminders, setDueReminders] = useState<Task[]>([])
  const [notificationsOpen, setNotificationsOpen] = useState(false)

  // Canvas (Phase 2): split-pane TipTap document panel rendered to the right
  // of the chat thread. `canvasDoc` is only ever replaced wholesale (a new
  // object) when a genuinely new document is opened -- CanvasWorkspace is
  // memoized on it, so streaming chat tokens updating `messages` elsewhere
  // in this component never cause the (expensive) editor tree to re-render.
  const [canvasOpen, setCanvasOpen] = useState(false)
  const [canvasDoc, setCanvasDoc] = useState<{ title: string; content: string }>({
    title: 'Untitled Canvas Document',
    content: '',
  })
  const handleCloseCanvas = useCallback(() => setCanvasOpen(false), [])
  const [loadingReminders, setLoadingReminders] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Auto-dismiss whatever toast is showing after a few seconds.
  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(() => setToast(null), 5000)
    return () => clearTimeout(timer)
  }, [toast])

  const fetchSessions = async () => {
    setLoadingSessions(true)
    try {
      const { data, error } = await supabase
        .from('chat_sessions')
        .select('*')
        .order('created_at', { ascending: false })

      if (error) throw error
      setSessions(data || [])
    } catch (err) {
      console.error('Error fetching sessions:', err)
    } finally {
      setLoadingSessions(false)
    }
  }

  // Fetch user data and chat sessions on mount
  useEffect(() => {
    // Check local storage for a previously saved theme preference
    // ('system' | 'light' | 'dark'). No saved value keeps the 'system'
    // default the theme state was initialized with.
    const savedTheme = localStorage.getItem('aether_theme')
    if (savedTheme === 'light' || savedTheme === 'dark' || savedTheme === 'system') {
      setTheme(savedTheme)
    }

    // A connector's linkIdentity redirect (see toggleConnector) round-trips
    // through /auth/callback/route.ts's server-side PKCE code exchange
    // before landing back here on '/' -- on success with no query params of
    // its own (so genuine success can't be told apart from URL params
    // alone; see the pendingConnectorId handling in initApp below, which
    // uses the freshly-fetched `linked` identities as the real signal
    // instead). Only a real failure -- identity_already_exists (not a true
    // failure, just re-forwarded to be handled below), a different
    // error_code, or a bare `error` -- ever arrives as a query param here,
    // forwarded through by route.ts. Read synchronously, before any async
    // work below, since these params must be captured before a later
    // render strips them. Read first so `pendingLabel` is ready for the
    // toast regardless of how initApp resolves below.
    const params = new URLSearchParams(window.location.search)
    const errorCode = params.get('error_code')
    const errorDescription = params.get('error_description')
    const errorParam = params.get('error')
    const hasOAuthRedirectParams = params.has('error') || params.has('error_code')
    const pendingConnectorId = localStorage.getItem('aether_pending_connector')
    const pendingConnector = CONNECTORS.find(c => c.id === pendingConnectorId)
    const pendingLabel = pendingConnector?.label || 'Connector'

    // A real (non "already linked") failure is decidable immediately from
    // the URL alone. identity_already_exists and genuine success both
    // depend on whether the provider is actually linked afterward, so both
    // are resolved together once `linked` is fetched in initApp below.
    if (errorCode && errorCode !== 'identity_already_exists') {
      if (errorDescription) console.error('OAuth connector error:', decodeURIComponent(errorDescription.replace(/\+/g, ' ')))
      setToast({ type: 'error', message: `Failed to connect ${pendingLabel}. Please try again.` })
    } else if (!errorCode && errorParam) {
      setToast({ type: 'error', message: `Failed to connect ${pendingLabel}. Please try again.` })
    }

    if (hasOAuthRedirectParams) {
      window.history.replaceState({}, '', window.location.pathname)
    }

    const initApp = async () => {
      const { data: { user } } = await supabase.auth.getUser()
      if (user) {
        // Map user properties to avoid type issues
        setUser({ id: user.id, email: user.email })
        await fetchSessions()

        // Persistent link status per provider, straight from the user's own
        // identities list -- unlike provider_token below, this doesn't
        // disappear across reloads once the token ages out of the session.
        const identities = user.identities || []
        const linked = {
          google: identities.some(identity => identity.provider === 'google'),
          github: identities.some(identity => identity.provider === 'github')
        }
        setLinkedProviders(linked)

        // Per-user OAuth access token, if the current session's own sign-in
        // (or most recent linkIdentity) was through an OAuth provider and
        // returned one -- see toggleConnector for how a missing token here
        // is filled in on demand. Supabase only attaches provider_token to
        // the session right after that OAuth exchange, not indefinitely, so
        // this can legitimately be empty even for a provider the user has
        // linked in the past.
        const { data: { session } } = await supabase.auth.getSession()
        const providerToken = session?.provider_token

        // Which provider this session's provider_token (if any) actually
        // belongs to. FIXED BUG: this used to be read straight off
        // session.user.app_metadata.provider, which names the account's
        // ORIGINAL signup provider -- not whichever identity a linkIdentity
        // flow just added. That silently dropped the token for every
        // secondary linked provider (e.g. an email/password account linking
        // Google), since app_metadata.provider never changes to 'google' in
        // that case. Prefer pendingConnector (see toggleConnector -- we know
        // exactly which provider we were just linking, unambiguously) and
        // only fall back to identities/app_metadata for a plain OAuth sign-in
        // that never went through our own connector flow.
        let tokenProvider: 'google' | 'github' | undefined = pendingConnector?.provider
        if (!tokenProvider) {
          const linkedOAuthProviders = (['google', 'github'] as const).filter(p => linked[p])
          if (linkedOAuthProviders.length === 1) {
            tokenProvider = linkedOAuthProviders[0]
          } else {
            // Multiple (or zero) OAuth identities linked and no pending
            // connector -- app_metadata.provider is reliable here since it
            // names whichever provider this session's own sign-in actually
            // went through.
            const signInProvider = session?.user?.app_metadata?.provider
            if (signInProvider === 'google' || signInProvider === 'github') tokenProvider = signInProvider
          }
        }

        if (providerToken && tokenProvider) {
          setProviderTokens(prev => ({ ...prev, [tokenProvider]: providerToken }))
        }

        // Long-lived refresh token (only present when linkIdentity requested
        // offline access -- see toggleConnector's queryParams), persisted
        // server-side so the backend can mint a fresh access token on its
        // own once this session's own copy ages out (mcp_manager
        // .create_workspace_session). Best-effort/fire-and-forget: losing
        // this doesn't block anything the user is doing right now, it just
        // means the connector won't survive a token refresh down the line.
        const providerRefreshToken = session?.provider_refresh_token
        if (providerRefreshToken && tokenProvider && session?.access_token) {
          fetch('/api/connectors/store-token', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${session.access_token}`
            },
            body: JSON.stringify({ provider: tokenProvider, refresh_token: providerRefreshToken })
          }).catch(err => console.error('Failed to persist OAuth refresh token:', err))
        }

        // Whether a stored refresh_token already backs each linked provider
        // (see hasStoredRefreshToken's own comment) -- fetched on every
        // load, not just right after linkIdentity, since a provider can be
        // "linked" without one for reasons unrelated to this page load
        // (linked in a previous session before this endpoint/feature
        // existed, or via a plain provider sign-in). Best-effort: a failed
        // lookup just leaves toggleConnector unable to distinguish the two
        // cases this turn, it doesn't block anything else on the page.
        if (session?.access_token) {
          fetch('/api/connectors/status', {
            headers: { 'Authorization': `Bearer ${session.access_token}` }
          })
            .then(res => res.ok ? res.json() : null)
            .then(status => {
              if (status) setHasStoredRefreshToken({ google: !!status.google, github: !!status.github })
            })
            .catch(err => console.error('Failed to fetch connector status:', err))
        }

        // If this load is the return leg of a linkIdentity redirect and that
        // provider is now linked, flip the specific connector the user asked
        // for straight to "on" rather than leaving them to find and
        // re-toggle it themselves. `linked` (fresh from user.identities) is
        // the authoritative success signal here -- a genuine success lands
        // back on '/' with no query params at all (see the mount effect's
        // top block), and identity_already_exists (deferred from that same
        // block) means the identity was already linked, so `linked` is true
        // in that case too. Only skip this whole block when a real failure
        // already got its own toast up there -- avoid double-toasting.
        if (pendingConnectorId) {
          if (pendingConnector && linked[pendingConnector.provider]) {
            setEnabledConnectors(prev => prev.includes(pendingConnectorId) ? prev : [...prev, pendingConnectorId])
            if (!errorCode || errorCode === 'identity_already_exists') {
              setToast({ type: 'success', message: `Successfully connected ${pendingLabel}!` })
            }
          } else if (!errorCode && !errorParam) {
            // Landed back with a pending connector, no error was reported,
            // yet the provider still isn't linked -- an unexpected outcome
            // (e.g. the user closed the consent screen without an explicit
            // error redirect), but silently leaving the toggle off with no
            // feedback at all would be worse than a possibly-redundant
            // error toast.
            setToast({ type: 'error', message: `Failed to connect ${pendingLabel}. Please try again.` })
          }
          localStorage.removeItem('aether_pending_connector')
        }
      } else {
        router.push('/login')
      }
    }
    initApp()
  }, [])

  const fetchDueReminders = async () => {
    setLoadingReminders(true)
    try {
      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const response = await fetch('/api/tasks?due_only=true', {
        headers: { 'Authorization': `Bearer ${token}` }
      })

      if (!response.ok) {
        // Runs on a background 60s poll (see the effect below), fully
        // decoupled from chat streaming state (isLoading/isStreaming) --
        // a failed poll (expired session, backend hiccup) must never throw,
        // it should just leave the reminder bell empty until the next tick.
        console.warn('fetchDueReminders: request failed with status', response.status)
        setDueReminders([])
        return
      }

      const data = await response.json()
      setDueReminders(data.tasks || [])
    } catch (err) {
      // Network failure, a getSession() rejection, or a malformed JSON body
      // -- same "never interrupt the rest of the app" contract as the
      // !response.ok branch above.
      console.error('Error fetching due reminders:', err)
      setDueReminders([])
    } finally {
      setLoadingReminders(false)
    }
  }

  // Polls for newly-due reminders every 60s -- matches the background
  // scheduler's own poll interval (backend/core/scheduler.py), since a
  // shorter frontend interval couldn't surface anything sooner anyway.
  useEffect(() => {
    if (!user) return
    const poll = () => { fetchDueReminders() }
    poll()
    const interval = setInterval(poll, 60000)
    return () => clearInterval(interval)
  }, [user])

  // Applies the resolved dark/light class for the current `theme`
  // preference and persists it. When `theme` is 'system', this also
  // subscribes to OS-level color-scheme changes so the UI updates live if
  // the user flips their system setting without needing a reload.
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')

    const applyResolvedTheme = () => {
      const isDark = theme === 'dark' || (theme === 'system' && media.matches)
      document.documentElement.classList.toggle('dark', isDark)
    }

    applyResolvedTheme()
    localStorage.setItem('aether_theme', theme)

    if (theme === 'system') {
      media.addEventListener('change', applyResolvedTheme)
      return () => media.removeEventListener('change', applyResolvedTheme)
    }
  }, [theme])


  // Live-refresh sidebar titles once the backend's auto-titling background
  // task finishes writing the generated title, without a manual refetch.
  // Requires Realtime replication to be enabled for `chat_sessions` in the
  // Supabase dashboard (Database > Replication) plus a SELECT policy that
  // covers the authenticated role, since postgres_changes is RLS-gated.
  useEffect(() => {
    if (!user) return

    const channel = supabase
      .channel(`chat_sessions_changes_${user.id}`)
      .on(
        'postgres_changes',
        { event: 'UPDATE', schema: 'public', table: 'chat_sessions', filter: `user_id=eq.${user.id}` },
        (payload) => {
          const updated = payload.new as ChatSession
          setSessions(prev => prev.map(s => s.id === updated.id ? { ...s, title: updated.title } : s))
        }
      )
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [user])

  // Auto-scroll to bottom on message list updates or streaming content updates
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [messages, isLoading, isStreaming])

  // Adjust textarea height dynamically based on input
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [inputText])

  const fetchMessages = async (sessionId: string, showSpinner = true) => {
    if (showSpinner) {
      setLoadingMessages(true)
    }
    try {
      const { data: messagesData, error } = await supabase
        .from('messages')
        .select('*')
        .eq('session_id', sessionId)
        .order('created_at', { ascending: true })

      if (error) throw error
      
      if (messagesData) {
        const parsedMessages = messagesData.map(msg => {
          let content = msg.content || ''
          let sources = undefined
          
          // Extract zero-migration sources tag
          const sourceMatch = content.match(/<aether-sources>([\s\S]*?)<\/aether-sources>/)
          if (sourceMatch) {
            try {
              sources = JSON.parse(sourceMatch[1])
              content = content.replace(sourceMatch[0], '')
            } catch (e) {
              console.error("Error parsing embedded sources:", e)
            }
          }

          // Extract a <canvas> document persisted from a past turn (see
          // parseCanvasStream -- same extraction the live SSE loop uses).
          let canvasDocument: CanvasDocument | undefined
          const canvasParsed = parseCanvasStream(content)
          if (canvasParsed.canvasContent !== null) {
            const trimmedCanvasContent = canvasParsed.canvasContent.trim()
            canvasDocument = { title: deriveCanvasTitle(trimmedCanvasContent), content: trimmedCanvasContent }
            content = canvasParsed.chatText
          }

          return {
            id: msg.id,
            role: msg.role,
            content: content.trim(),
            provider_used: msg.provider_used,
            sources: sources,
            canvasDocument
          }
        })
        setMessages(parsedMessages)
      }
    } catch (err) {
      console.error('Error fetching messages:', err)
    } finally {
      if (showSpinner) {
        setLoadingMessages(false)
      }
    }
  }

  const handleSelectSession = async (sessionId: string) => {
    setActiveSessionId(sessionId)
    setIsLoading(false)
    setIsStreaming(false)
    setAttachedFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
    await fetchMessages(sessionId)
    if (window.innerWidth < 768) {
      setSidebarOpen(false)
    }
  }

  const handleNewChat = () => {
    setActiveSessionId(null)
    setMessages([])
    setIsLoading(false)
    setIsStreaming(false)
    setInputText('')
    setAttachedFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
    setProviderDropdownOpen(false)  // always close any open dropdown
    if (window.innerWidth < 768) {
      setSidebarOpen(false)
    }
    setTimeout(() => textareaRef.current?.focus(), 50)
  }

  const handleAttachButtonClick = () => {
    fileInputRef.current?.click()
    setPlusMenuOpen(false)
  }

  const handleFileSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (file.type === 'application/pdf') {
        setAttachedFile(file)
      } else {
        console.error('Only PDF files are supported.')
      }
    }
    // Reset the input so selecting the same file again still fires onChange
    e.target.value = ''
  }

  const handleRemoveAttachment = () => {
    setAttachedFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleDeleteSessionClick = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    setSessionToDelete(sessionId)
  }

  const confirmDeleteSession = async () => {
    if (!sessionToDelete) return
    try {
      const { error } = await supabase
        .from('chat_sessions')
        .delete()
        .eq('id', sessionToDelete)

      if (error) throw error

      setSessions(prev => prev.filter(s => s.id !== sessionToDelete))
      if (activeSessionId === sessionToDelete) {
        handleNewChat()
      }
    } catch (err) {
      console.error('Error deleting session:', err)
    } finally {
      setSessionToDelete(null)
    }
  }

  const handleRenameSession = async (sessionId: string) => {
    if (!editTitleText.trim()) {
      setEditingSessionId(null)
      return
    }
    try {
      const { error } = await supabase
        .from('chat_sessions')
        .update({ title: editTitleText })
        .eq('id', sessionId)
      
      if (error) throw error

      setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, title: editTitleText } : s))
    } catch (err) {
      console.error('Error renaming session:', err)
    } finally {
      setEditingSessionId(null)
      setEditTitleText('')
    }
  }

  const handleLogoutClick = () => {
    setShowLogoutModal(true)
  }

  const confirmLogout = async () => {
    await supabase.auth.signOut()
    router.refresh()
    router.push('/login')
  }

  const handleCopyText = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const fetchMemory = async () => {
    setLoadingMemory(true)
    try {
      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const response = await fetch('/api/memory', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!response.ok) throw new Error('Failed to fetch memory.')

      const data = await response.json()
      setMemoryProfile({ narrative: data.narrative || '', updated_at: data.updated_at || null })
    } catch (err) {
      console.error('Error fetching memory:', err)
    } finally {
      setLoadingMemory(false)
    }
  }

  const handleOpenMemory = () => {
    setShowMemory(true)
    fetchMemory()
  }

  const confirmDeleteMemory = async () => {
    setDeletingMemory(true)
    try {
      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const response = await fetch('/api/memory', {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!response.ok) throw new Error('Failed to clear memory profile.')

      setMemoryProfile({ narrative: '', updated_at: null })
    } catch (err) {
      console.error('Error clearing memory profile:', err)
    } finally {
      setDeletingMemory(false)
      setShowClearMemoryConfirm(false)
    }
  }

  const fetchTasks = async (status: 'pending' | 'completed') => {
    setLoadingTasks(true)
    try {
      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const response = await fetch(`/api/tasks?status=${status}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!response.ok) throw new Error('Failed to fetch tasks.')

      const data = await response.json()
      setTasks(data.tasks || [])
    } catch (err) {
      console.error('Error fetching tasks:', err)
    } finally {
      setLoadingTasks(false)
    }
  }

  const handleOpenTasks = () => {
    setShowTasks(true)
    fetchTasks(taskStatusFilter)
  }

  const handleSelectTaskFilter = (status: 'pending' | 'completed') => {
    setTaskStatusFilter(status)
    fetchTasks(status)
  }

  const handleCompleteTask = async (taskId: string) => {
    try {
      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const response = await fetch(`/api/tasks/${taskId}/complete`, {
        method: 'PATCH',
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!response.ok) throw new Error('Failed to complete task.')

      setTasks(prev => prev.filter(t => t.id !== taskId))
      setDueReminders(prev => prev.filter(t => t.id !== taskId))
    } catch (err) {
      console.error('Error completing task:', err)
    }
  }

  const confirmDeleteTask = async () => {
    if (!taskToDelete) return
    setDeletingTask(true)
    try {
      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const response = await fetch(`/api/tasks/${taskToDelete}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!response.ok) throw new Error('Failed to delete task.')

      setTasks(prev => prev.filter(t => t.id !== taskToDelete))
      setDueReminders(prev => prev.filter(t => t.id !== taskToDelete))
    } catch (err) {
      console.error('Error deleting task:', err)
    } finally {
      setDeletingTask(false)
      setTaskToDelete(null)
    }
  }

  // Toggles a connector on/off for the next chat turn. Turning one on when
  // its provider identity isn't linked yet doesn't enable it immediately --
  // it starts Supabase's linkIdentity consent flow instead (a full-page
  // redirect through the provider and back to '/'). An already-linked
  // provider (per linkedProviders, set from user.identities on load) flips
  // straight on/off in state without ever re-triggering that OAuth popup,
  // even if this session happens to be missing a live provider_token.
  const toggleConnector = async (connectorId: string, provider: 'google' | 'github') => {
    const isEnabling = !enabledConnectors.includes(connectorId)

    if (isEnabling && !linkedProviders[provider]) {
      try {
        // Remembered across the full-page redirect so the mount effect
        // above knows which connector to flip on and name in its toast.
        localStorage.setItem('aether_pending_connector', connectorId)
        const { error } = await supabase.auth.linkIdentity({
          provider,
          options: {
            // Must land on the server-side PKCE exchange route (matching
            // login/page.tsx's handleGoogleLogin), not '/' directly -- this
            // app's client (src/utils/supabase/client.ts, createBrowserClient
            // from @supabase/ssr) defaults to flowType 'pkce', so the OAuth
            // redirect carries an authorization `?code=` that only
            // /auth/callback/route.ts's server-side exchangeCodeForSession
            // call can redeem into cookies. Landing on '/' directly leaves
            // that code sitting unexchanged in the URL forever -- no session
            // is ever established, so provider_token/provider_refresh_token
            // never get captured (see the mount effect below) and
            // /api/connectors/store-token is never called.
            redirectTo: `${window.location.origin}/auth/callback`,
            // Google only returns a refresh_token (needed for the backend's
            // server-side token-refresh fallback -- see
            // mcp_manager.create_workspace_session) on the FIRST consent for
            // a given client/user, and only when explicitly asked for
            // offline access. access_type=offline requests one at all;
            // prompt=consent forces the consent screen every time so a
            // returning user who silently re-authorizes still gets a fresh
            // refresh_token (Google otherwise skips it on repeat, silent
            // authorizations). select_account (space-combined with consent,
            // per Google's OAuth spec allowing multiple prompt values) makes
            // Google show the account chooser instead of silently
            // re-authorizing whichever Google account is already signed in
            // on this browser -- needed so a user who just disconnected one
            // Google account can actually pick a different one on re-link,
            // rather than being silently signed back into the same one.
            // GitHub has no equivalent concept -- these params are simply
            // meaningless there, not harmful.
            ...(provider === 'google' ? { queryParams: { access_type: 'offline', prompt: 'select_account consent' } } : {})
          }
        })
        if (error) throw error
      } catch (err) {
        console.error(`Failed to connect ${provider}:`, err)
        localStorage.removeItem('aether_pending_connector')
        const label = CONNECTORS.find(c => c.id === connectorId)?.label || 'Connector'
        setToast({ type: 'error', message: `Failed to connect ${label}. Please try again.` })
      }
      return
    }

    // Google is linked, but neither a live in-memory access token nor a
    // stored refresh_token backs it -- e.g. it was linked before the
    // offline-access refresh-token flow existed, or via a plain sign-in
    // through Google rather than this app's own connector toggle. The
    // backend can never open a Workspace session for this user without
    // one (mcp_manager.create_workspace_session), and simply retrying the
    // link flow can't fix it -- Supabase's linkIdentity() refuses to
    // re-run for a provider identity already linked to the current user.
    // Surface that a disconnect+reconnect is required instead of silently
    // flipping the toggle on and leaving Gmail/Docs/Calendar/Drive tool
    // calls to silently fail to reach the model turn after turn.
    if (isEnabling && provider === 'google' && !providerTokens.google && !hasStoredRefreshToken.google) {
      setToast({
        type: 'error',
        message: 'Google is linked without offline access, so this can\'t work yet. Click the unlink icon next to Google below, then reconnect it to grant offline access.'
      })
      return
    }

    setEnabledConnectors(prev =>
      isEnabling ? [...prev, connectorId] : prev.filter(id => id !== connectorId)
    )
  }

  // Fully unlinks a connector's provider identity: removes it from Supabase
  // Auth (unlinkIdentity) so it stops counting as "linked" for linkIdentity's
  // own re-prompt logic, deletes this user's stored refresh_token server-side
  // (so the backend's token-refresh fallback can no longer act on their
  // behalf -- see mcp_manager.create_workspace_session), and clears every
  // piece of local state tied to that provider so the UI reflects "not
  // connected" immediately rather than waiting for a reload.
  const handleDisconnectConnector = async (provider: 'google' | 'github') => {
    const label = provider === 'google' ? 'Google' : 'GitHub'

    try {
      const { data: { user: freshUser } } = await supabase.auth.getUser()
      const identity = freshUser?.identities?.find(i => i.provider === provider)

      if (identity) {
        const { error } = await supabase.auth.unlinkIdentity(identity)
        if (error) throw error
      }

      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''
      const response = await fetch(`/api/connectors/disconnect?provider=${provider}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!response.ok) throw new Error('Failed to disconnect on the server.')

      // Every google_* sub-connector (gmail/docs/calendar/drive) shares this
      // one provider identity, so disconnecting 'google' must drop all of
      // them at once, not just a single connector id.
      setEnabledConnectors(prev =>
        prev.filter(id => CONNECTORS.find(c => c.id === id)?.provider !== provider)
      )
      setLinkedProviders(prev => ({ ...prev, [provider]: false }))
      setProviderTokens(prev => {
        const next = { ...prev }
        delete next[provider]
        return next
      })
      setHasStoredRefreshToken(prev => ({ ...prev, [provider]: false }))

      setToast({ type: 'success', message: `${label} account disconnected.` })
    } catch (err: unknown) {
      console.error(`Failed to disconnect ${provider}:`, err)
      const errorObj = err as { message?: string }
      // Supabase refuses to unlink a user's only remaining identity (they'd
      // be locked out) -- surface that specific reason instead of a generic
      // failure message, since it's an expected, actionable case rather
      // than an actual bug.
      const message = errorObj.message?.toLowerCase().includes('identities')
        ? `Can't disconnect ${label} -- it's your only sign-in method.`
        : `Failed to disconnect ${label}. Please try again.`
      setToast({ type: 'error', message })
    }
  }

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    const messageContent = inputText.trim()
    if ((!messageContent && !attachedFile) || isStreaming) return

    const fileToSend = attachedFile
    setInputText('')
    setAttachedFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    let currentSessionId = activeSessionId

    try {
      // 1. Create a session if none is active
      if (!currentSessionId) {
        const title = messageContent
          ? (messageContent.length > 30 ? messageContent.slice(0, 30) + '...' : messageContent)
          : `📄 ${fileToSend?.name}`
        const { data: newSession, error: sessionErr } = await supabase
          .from('chat_sessions')
          .insert({ title })
          .select()
          .single()

        if (sessionErr) throw sessionErr
        if (!newSession) throw new Error('Failed to create new session.')

        currentSessionId = newSession.id
        setActiveSessionId(currentSessionId)
        setSessions(prev => [newSession, ...prev])
      }

      // 2. Optimistically render the user's turn locally. The backend now
      // persists this message (and the upload marker, if a file was sent)
      // into Supabase itself as part of the multipart /api/chat request
      // below — inserting it here too would create duplicate rows.
      const optimisticUserMsg: Message = {
        id: `temp-user-${Date.now()}`,
        role: 'user',
        content: messageContent || `[Uploaded document: ${fileToSend?.name}]`,
        provider_used: selectedProvider,
        attachedFileName: fileToSend?.name
      }
      setMessages(prev => [...prev, optimisticUserMsg])

      // 3. Trigger Streaming from Backend Proxy (using Supabase Auth JWT header)
      setIsStreaming(true)
      setIsLoading(true)
      let streamedContent = ''
      let streamedSources: Source[] | undefined = undefined
      let firstChunkReceived = false
      // Last time canvasDoc (the CanvasWorkspace panel's state) was flushed
      // during this stream -- see CANVAS_STREAM_FLUSH_INTERVAL_MS. 0 means
      // "not yet flushed", so the very first canvas chunk always flushes
      // immediately rather than waiting out the first interval.
      let lastCanvasFlush = 0
      setMessages(prev => [...prev, { id: 'temp', role: 'assistant', content: '', provider_used: selectedProvider }])

      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const formData = new FormData()
      formData.append('provider', selectedProvider)
      formData.append('sessionId', currentSessionId as string)
      if (messageContent) formData.append('message', messageContent)
      if (fileToSend) formData.append('file', fileToSend)
      enabledConnectors.forEach(connectorId => formData.append('enabled_connectors', connectorId))
      // Only send a provider's OAuth token when at least one of its
      // connectors is actually toggled on -- sending it unconditionally
      // whenever one happened to be cached would open a Workspace/GitHub
      // MCP session server-side on every turn regardless of the toggle
      // state, defeating the point of granular per-connector gating.
      const googleConnectorEnabled = enabledConnectors.some(id => id === 'google_workspace' || id.startsWith('google_'))
      if (googleConnectorEnabled && providerTokens.google) formData.append('google_access_token', providerTokens.google)
      if (enabledConnectors.includes('github') && providerTokens.github) formData.append('github_access_token', providerTokens.github)

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          // Do NOT set Content-Type here — the browser must generate the
          // multipart boundary itself for a FormData body.
          'Authorization': `Bearer ${token}`
        },
        body: formData
      })

      if (!response.ok) {
        let errorMsg = 'Request failed.'
        try {
          const errorData = await response.json()
          if (errorData.detail) {
            errorMsg = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail)
          } else if (errorData.error) {
            errorMsg = errorData.error
          }
        } catch (e) {
          errorMsg = `Request failed with status: ${response.status}`
        }
        throw new Error(errorMsg)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      if (!reader) throw new Error('No stream reader available.')

      let streamError: string | null = null
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim()
            if (dataStr === '[DONE]') break
            try {
              const parsed = JSON.parse(dataStr)
              if (parsed.content) {
                // The loading bubble disappears the instant the first real
                // token arrives, replaced by the growing assistant message.
                if (!firstChunkReceived) {
                  firstChunkReceived = true
                  setIsLoading(false)
                }
                streamedContent += parsed.content

                const canvasParsed = parseCanvasStream(streamedContent)
                let liveCanvasDocument: CanvasDocument | undefined
                if (canvasParsed.canvasContent !== null) {
                  liveCanvasDocument = {
                    title: deriveCanvasTitle(canvasParsed.canvasContent),
                    content: canvasParsed.canvasContent,
                  }
                  // Idempotent -- React bails out on an identical boolean, so
                  // calling this on every chunk once the tag is open is safe.
                  setCanvasOpen(true)
                  const now = Date.now()
                  if (lastCanvasFlush === 0 || now - lastCanvasFlush >= CANVAS_STREAM_FLUSH_INTERVAL_MS) {
                    lastCanvasFlush = now
                    setCanvasDoc({ title: liveCanvasDocument.title, content: canvasParsed.canvasContent })
                  }
                }

                setMessages(prev => {
                  const newMsgs = [...prev]
                  const last = newMsgs[newMsgs.length - 1]
                  if (last && last.role === 'assistant') {
                    last.content = canvasParsed.chatText
                    last.canvasDocument = liveCanvasDocument
                    if (streamedSources) last.sources = streamedSources
                  }
                  return newMsgs
                })
              } else if (parsed.sources) {
                streamedSources = parsed.sources
                setMessages(prev => {
                  const newMsgs = [...prev]
                  const last = newMsgs[newMsgs.length - 1]
                  if (last && last.role === 'assistant') {
                    last.sources = streamedSources
                  }
                  return newMsgs
                })
              } else if (parsed.retract) {
                // The backend already streamed some text this turn, then
                // decided to call a tool after all (e.g. "Let me check the
                // weather..." ahead of the tool call) -- that preamble must
                // never end up merged with the real, post-tool answer. Wipe
                // it from the bubble and show the loading indicator again
                // until the real answer starts arriving.
                streamedContent = ''
                firstChunkReceived = false
                lastCanvasFlush = 0
                setIsLoading(true)
                setMessages(prev => {
                  const newMsgs = [...prev]
                  const last = newMsgs[newMsgs.length - 1]
                  if (last && last.role === 'assistant') {
                    last.content = ''
                    last.canvasDocument = undefined
                  }
                  return newMsgs
                })
              } else if (parsed.error) {
                // Backend sent an error event (quota, model not found, etc.)
                streamError = parsed.error
              }
            } catch {
              // Partial JSON fragment — safe to ignore
            }
          }
        }
        if (streamError) break
      }

      if (buffer.startsWith('data: ') && !streamError) {
        const dataStr = buffer.slice(6).trim()
        if (dataStr !== '[DONE]') {
          try {
            const parsed = JSON.parse(dataStr)
            if (parsed.content) {
              streamedContent += parsed.content
            } else if (parsed.sources) {
              streamedSources = parsed.sources
            } else if (parsed.error) {
              streamError = parsed.error
            }
          } catch {}
        }
      }

      // Parsed against `streamedContent` (not the error-notice-appended text
      // below) so that if a streamError cuts generation off mid-document, the
      // warning notice lands in the chat bubble rather than getting balled
      // up into the (now unclosed) canvas document's content.
      const finalCanvasParsed = parseCanvasStream(streamedContent)
      const finalCanvasDocument: CanvasDocument | undefined =
        finalCanvasParsed.canvasContent !== null
          ? { title: deriveCanvasTitle(finalCanvasParsed.canvasContent), content: finalCanvasParsed.canvasContent.trim() }
          : undefined

      // Final, un-throttled flush -- guarantees canvasDoc ends up with the
      // exact final content even if the last streamed chunk arrived within
      // CANVAS_STREAM_FLUSH_INTERVAL_MS of the previous flush.
      if (finalCanvasDocument) {
        setCanvasDoc(finalCanvasDocument)
        setCanvasOpen(true)
      }

      // A streamError from the backend (e.g. Groq failed to format a tool
      // call) means generation was cut short mid-response, not that nothing
      // happened — `streamedContent` may already hold real, valid text the
      // user has been watching stream in. Append an inline notice instead of
      // throwing, so that partial answer is preserved rather than replaced
      // wholesale by an error bubble.
      const finalChatText = streamError
        ? `${finalCanvasParsed.chatText}${finalCanvasParsed.chatText.trim() ? '\n\n' : ''}⚠️ *The agent encountered an error formatting its response. Please try again.*`
        : finalCanvasParsed.chatText

      // 4. Once streaming is complete, append the assistant response to messages state
      const mockAssistantMsg: Message = {
        id: Math.random().toString(),
        role: 'assistant',
        content: finalChatText,
        provider_used: selectedProvider,
        sources: streamedSources || undefined,
        canvasDocument: finalCanvasDocument
      }

      setMessages(prev => {
        const newMsgs = [...prev]
        newMsgs[newMsgs.length - 1] = mockAssistantMsg
        return newMsgs
      })
      setIsStreaming(false)
      setIsLoading(false)

    } catch (err: unknown) {
      console.error('Failed to complete message cycle:', err)
      const errorObj = err as { message?: string }
      // Extract clean user-facing message from verbose API error responses
      let errMsg: string = errorObj.message || 'Unable to get a response. Please try again.'
      // Trim long API error strings to just the first meaningful sentence
      const msgMatch = errMsg.match(/'message':\s*'([^']+)'/)
      if (msgMatch) errMsg = msgMatch[1]
      // Cap length
      if (errMsg.length > 200) errMsg = errMsg.slice(0, 200) + '...'

      const errorMsg: Message = {
        id: Math.random().toString(),
        role: 'assistant',
        content: `⚠️ ${errMsg}`,
        provider_used: selectedProvider
      }
      setMessages(prev => {
        const newMsgs = [...prev]
        if (newMsgs.length > 0 && newMsgs[newMsgs.length - 1].id === 'temp') {
          newMsgs[newMsgs.length - 1] = errorMsg
        } else {
          newMsgs.push(errorMsg)
        }
        return newMsgs
      })
      setIsStreaming(false)
      setIsLoading(false)
    }
  }

  const handleQuickPrompt = (promptText: string) => {
    setInputText(promptText)
    setTimeout(() => textareaRef.current?.focus(), 50)
  }

  const activeProvider = PROVIDERS.find(p => p.id === selectedProvider) || PROVIDERS[0]
  const ActiveProviderIcon = activeProvider.icon
  // Chat bubbles/input can use the extra reclaimed width once the sidebar is
  // collapsed, but only on large screens where it won't feel overstretched.
  const chatMaxWidth = sidebarOpen ? 'max-w-3xl' : 'max-w-3xl lg:max-w-4xl'

  return (
    <main className="flex h-screen w-screen bg-slate-50 dark:bg-slate-950 text-slate-800 dark:text-slate-200 overflow-hidden font-sans">
      {/* 1. SIDEBAR PANEL */}
      <aside
        className={`fixed md:relative z-20 h-full w-[280px] bg-slate-100/90 dark:bg-slate-900 border-r border-slate-200/80 dark:border-slate-800 flex flex-col transition-all duration-300 ${sidebarOpen ? 'left-0' : '-left-[280px] md:-ml-[280px]'
          }`}
      >
        {/* Sidebar Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 bg-gradient-to-tr from-violet-600 to-cyan-500 rounded-lg flex items-center justify-center shadow-md">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="font-bold text-base tracking-tight text-slate-800 dark:text-slate-100">AetherChat</h2>
              <span className="text-[10px] text-transparent bg-clip-text bg-gradient-to-r from-violet-600 to-indigo-500 font-bold uppercase tracking-wider">Agentic Workspace</span>
            </div>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            title="Collapse sidebar"
            className="p-1.5 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100 cursor-pointer"
          >
            <PanelLeftClose className="w-4.5 h-4.5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-3 space-y-0.5">
          <button
            onClick={handleNewChat}
            className="w-full py-1.5 px-2 text-sm text-slate-600 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-800/60 font-medium rounded-lg transition-colors flex items-center gap-2.5 cursor-pointer"
          >
            <Plus className="w-4 h-4 text-indigo-500" />
            New Chat
          </button>
          <button
            onClick={handleOpenTasks}
            className="w-full py-1.5 px-2 text-sm text-slate-600 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-800/60 font-medium rounded-lg transition-colors flex items-center gap-2.5 cursor-pointer"
          >
            <ListChecks className="w-4 h-4 text-indigo-500" />
            My Tasks
          </button>
          <hr className="my-2 border-gray-200 dark:border-slate-800" />
        </div>

        {/* Sessions History List */}
        <div className="flex-1 overflow-y-auto px-2 space-y-0.5 py-2 custom-scrollbar">
          <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-3 mb-1.5">
            Recent Dialogues
          </div>

          {loadingSessions ? (
            <div className="flex flex-col gap-1.5 p-2">
              <div className="h-7 bg-slate-200/50 dark:bg-slate-800/50 rounded-lg animate-pulse" />
              <div className="h-7 bg-slate-200/50 dark:bg-slate-800/50 rounded-lg animate-pulse" />
              <div className="h-7 bg-slate-200/50 dark:bg-slate-800/50 rounded-lg animate-pulse" />
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-center py-8 px-4 text-xs text-slate-400">
              No chat history yet. Send a message to start!
            </div>
          ) : (
            sessions.map(session => (
              <div
                key={session.id}
                onClick={() => handleSelectSession(session.id)}
                className={`group relative flex items-center justify-between py-1.5 px-2 rounded-xl cursor-pointer transition-all border ${activeSessionId === session.id
                    ? 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-950 dark:text-slate-100 shadow-sm font-semibold'
                    : 'bg-transparent border-transparent hover:bg-slate-200/40 dark:hover:bg-slate-800/40 text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                  }`}
              >
                {editingSessionId === session.id ? (
                  <div className="flex items-center gap-2 w-full mr-2" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="text"
                      value={editTitleText}
                      onChange={(e) => setEditTitleText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleRenameSession(session.id)
                        if (e.key === 'Escape') setEditingSessionId(null)
                      }}
                      autoFocus
                      className="flex-1 bg-white dark:bg-slate-900 border border-violet-400 focus:outline-none focus:ring-1 focus:ring-violet-500 rounded px-2 py-1 text-sm text-slate-800 dark:text-slate-200"
                    />
                    <button onClick={() => handleRenameSession(session.id)} className="p-1 text-emerald-500 hover:bg-emerald-50 dark:hover:bg-emerald-500/10 rounded">
                      <Check className="w-4 h-4" />
                    </button>
                    <button onClick={() => setEditingSessionId(null)} className="p-1 text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 rounded">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="flex-1 flex items-center gap-2.5 truncate">
                      <MessageSquare className={`w-4.5 h-4.5 flex-shrink-0 ${activeSessionId === session.id ? 'text-violet-500' : 'text-indigo-400'
                        }`} />
                      <span className="text-sm truncate">{session.title}</span>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setSessionMenuOpenId(prev => prev === session.id ? null : session.id)
                      }}
                      className={`p-1 flex-shrink-0 rounded hover:bg-slate-200/80 dark:hover:bg-slate-700 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-all cursor-pointer ${sessionMenuOpenId === session.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                        }`}
                      title="More options"
                    >
                      <MoreVertical className="w-4 h-4" />
                    </button>

                    {sessionMenuOpenId === session.id && (
                      <>
                        <div
                          className="fixed inset-0 z-20 cursor-default"
                          onClick={(e) => {
                            e.stopPropagation()
                            setSessionMenuOpenId(null)
                          }}
                        />
                        <div
                          className="absolute right-2 top-full mt-1 w-32 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg z-30 p-1.5 animate-fade-in"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            onClick={() => {
                              setSessionMenuOpenId(null)
                              setEditingSessionId(session.id)
                              setEditTitleText(session.title)
                            }}
                            className="w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-all cursor-pointer"
                          >
                            <Edit2 className="w-3.5 h-3.5 text-slate-400" />
                            <span>Edit</span>
                          </button>
                          <button
                            onClick={(e) => {
                              setSessionMenuOpenId(null)
                              handleDeleteSessionClick(e, session.id)
                            }}
                            className="w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-xs font-medium text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-500/10 transition-all cursor-pointer"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                            <span>Delete</span>
                          </button>
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            ))
          )}
        </div>

        {/* User Card, Profile Settings & Logout */}
        {user && (
          <div className="relative p-4 border-t border-slate-200 dark:border-slate-800 bg-slate-100/50 dark:bg-slate-900/60 flex items-center justify-between gap-3">
            <button
              type="button"
              onClick={() => {
                setProfileMenuOpen(prev => !prev)
                setThemeMenuOpen(false)
              }}
              className="flex items-center gap-2.5 overflow-hidden flex-1 min-w-0 text-left cursor-pointer rounded-lg hover:bg-slate-200/50 dark:hover:bg-slate-800/50 -m-1 p-1 transition-colors"
              title="Profile settings"
            >
              <div className="w-8 h-8 rounded-full bg-violet-600/10 dark:bg-violet-600/20 border border-violet-500/20 dark:border-violet-500/30 flex items-center justify-center text-violet-600 dark:text-violet-400 font-bold uppercase flex-shrink-0 text-sm">
                {user.email?.charAt(0) || 'U'}
              </div>
              <div className="flex flex-col overflow-hidden">
                <span className="text-xs font-semibold text-slate-850 dark:text-slate-200 truncate">{user.email}</span>
                <span className="text-[9px] text-slate-400 uppercase tracking-wider font-bold">Standard Account</span>
              </div>
            </button>
            <button
              onClick={handleLogoutClick}
              title="Sign Out"
              className="p-2 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg text-slate-500 dark:text-slate-400 hover:text-rose-500 dark:hover:text-rose-400 transition-colors cursor-pointer flex-shrink-0"
            >
              <LogOut className="w-4.5 h-4.5" />
            </button>

            {/* Profile Settings Popover */}
            {profileMenuOpen && (
              <>
                <div
                  className="fixed inset-0 z-20 cursor-default"
                  onClick={() => {
                    setProfileMenuOpen(false)
                    setThemeMenuOpen(false)
                  }}
                />
                <div className="absolute left-4 right-4 bottom-full mb-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg z-30 p-2 animate-fade-in">
                  <div className="flex items-center gap-2 px-2.5 py-1.5 mb-1 border-b border-slate-100 dark:border-slate-800">
                    <Settings className="w-3.5 h-3.5 text-slate-400" />
                    <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Profile Settings</span>
                  </div>
                  {/* Memory menu item -> opens the Memory drawer (facts the AI has learned about the user, shared across every chat) */}
                  <button
                    type="button"
                    onClick={() => {
                      setProfileMenuOpen(false)
                      setThemeMenuOpen(false)
                      handleOpenMemory()
                    }}
                    className="w-full flex items-center gap-2 px-2.5 py-2 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-all cursor-pointer"
                  >
                    <Database className="w-3.5 h-3.5 text-slate-400" />
                    <span>Memory</span>
                  </button>

                  {/* Theme menu item -> hover/click flyout submenu, mirroring Gemini's settings menu pattern */}
                  <div
                    className="relative"
                    onMouseEnter={() => setThemeMenuOpen(true)}
                    onMouseLeave={() => setThemeMenuOpen(false)}
                  >
                    <button
                      type="button"
                      onClick={() => setThemeMenuOpen(prev => !prev)}
                      className="w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-xs font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-all cursor-pointer"
                    >
                      <div className="flex items-center gap-2">
                        <Monitor className="w-3.5 h-3.5 text-slate-400" />
                        <span>Theme</span>
                      </div>
                      <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                    </button>

                    {themeMenuOpen && (
                      <div className="absolute left-full top-0 ml-1.5 w-36 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg z-40 p-1.5 animate-fade-in">
                        {THEME_OPTIONS.map(opt => {
                          const Icon = opt.icon
                          const isSelected = theme === opt.id
                          return (
                            <button
                              key={opt.id}
                              type="button"
                              onClick={() => {
                                setTheme(opt.id)
                                setProfileMenuOpen(false)
                                setThemeMenuOpen(false)
                              }}
                              className={`w-full flex items-center justify-between px-2.5 py-2 rounded-lg text-xs font-medium transition-all text-left cursor-pointer ${isSelected
                                  ? 'bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white font-bold'
                                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-slate-800/50'
                                }`}
                            >
                              <div className="flex items-center gap-2">
                                <Icon className="w-3.5 h-3.5" />
                                <span>{opt.label}</span>
                              </div>
                              {isSelected && <Check className="w-3.5 h-3.5" />}
                            </button>
                          )
                        })}
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </aside>

      {/* 2. MAIN CHAT AREA */}
      <section className="flex-1 w-full h-[100dvh] max-h-[100dvh] flex flex-col overflow-hidden bg-slate-50/50 dark:bg-slate-950 relative min-w-0 transition-all duration-300">
        {/* Decorative background glows */}
        <div className="absolute top-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-violet-500/5 blur-[100px] pointer-events-none" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-cyan-500/5 blur-[100px] pointer-events-none" />

        {/* Toast -- OAuth/connector redirect outcomes (see the mount effect
            and toggleConnector). Fixed top-center, above everything else. */}
        {toast && (
          <div className="fixed top-4 left-1/2 -translate-x-1/2 z-[60] animate-fade-in">
            <div
              className={`flex items-center gap-2.5 pl-3.5 pr-2.5 py-2.5 rounded-xl shadow-lg border text-sm font-medium max-w-sm ${toast.type === 'success'
                  ? 'bg-emerald-50 dark:bg-emerald-900/30 border-emerald-200 dark:border-emerald-800/50 text-emerald-700 dark:text-emerald-300'
                  : 'bg-rose-50 dark:bg-rose-900/30 border-rose-200 dark:border-rose-800/50 text-rose-700 dark:text-rose-300'
                }`}
            >
              {toast.type === 'success' ? (
                <CheckCircle2 className="w-4.5 h-4.5 flex-shrink-0" />
              ) : (
                <AlertTriangle className="w-4.5 h-4.5 flex-shrink-0" />
              )}
              <span className="leading-snug">{toast.message}</span>
              <button
                type="button"
                onClick={() => setToast(null)}
                className="p-1 rounded-lg hover:bg-black/5 dark:hover:bg-white/10 flex-shrink-0 cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* Floating sidebar-reopen button, shown on any breakpoint once the
            sidebar is closed -- replaces the old full-width top header's
            left side, given the same floating circular treatment as the
            bell below so removing that header doesn't strand users. */}
        {!sidebarOpen && (
          <button
            onClick={() => setSidebarOpen(true)}
            className="fixed top-4 left-4 z-50 p-2 rounded-full border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm hover:shadow-md text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-100 transition-all cursor-pointer"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}

        {/* Canvas toggle (Phase 2) -- opens/closes the split-pane TipTap
            document panel to the right of the chat thread. Placed just left
            of the bell, same floating circular treatment.

            `absolute` (not `fixed`) is deliberate: this section is already
            `relative`, and CanvasWorkspace mounts as its flex sibling, so
            this section's own right edge automatically moves left to sit at
            the chat/canvas boundary whenever the panel is open -- anchoring
            here (rather than the viewport) means `right-16` always lands in
            the remaining chat space with zero overlap, at any viewport
            width or sidebar state, with no hardcoded panel-width guess. */}
        <button
          type="button"
          onClick={() => setCanvasOpen(prev => !prev)}
          className="absolute top-4 right-16 z-50 p-2 rounded-full border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm hover:shadow-md text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-100 transition-all cursor-pointer"
          title={canvasOpen ? 'Close canvas' : 'Open canvas'}
        >
          {canvasOpen ? <PanelRightClose className="w-5 h-5" /> : <PanelRightOpen className="w-5 h-5" />}
        </button>

        {/* Floating Notification Bell -- anchored to this section (see the
            canvas toggle above for why `absolute` replaces `fixed` here),
            replacing the old full-width top header bar entirely to reclaim
            vertical space. */}
        <div className="absolute top-4 right-4 z-50">
          <button
            type="button"
            onClick={() => setNotificationsOpen(prev => !prev)}
            className="relative p-2 rounded-full border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm hover:shadow-md text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-100 transition-all cursor-pointer"
            title="Reminders"
          >
            <Bell className="w-5 h-5" />
            {dueReminders.length > 0 && (
              <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-rose-500 ring-2 ring-white dark:ring-slate-900" />
            )}
          </button>

          {notificationsOpen && (
            <>
              <div className="fixed inset-0 z-20 cursor-default" onClick={() => setNotificationsOpen(false)} />
              <div className="absolute right-0 top-full mt-2 w-80 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg z-30 animate-fade-in overflow-hidden">
                <div className="flex items-center gap-2 px-3.5 py-2.5 border-b border-slate-100 dark:border-slate-800">
                  <Bell className="w-3.5 h-3.5 text-slate-400" />
                  <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest">Due Reminders</span>
                </div>
                <div className="max-h-80 overflow-y-auto custom-scrollbar">
                  {loadingReminders ? (
                    <div className="p-4 space-y-2">
                      <div className="h-10 bg-slate-100 dark:bg-slate-800/50 rounded-lg animate-pulse" />
                      <div className="h-10 bg-slate-100 dark:bg-slate-800/50 rounded-lg animate-pulse" />
                    </div>
                  ) : dueReminders.length === 0 ? (
                    <div className="flex flex-col items-center justify-center text-center px-6 py-10">
                      <Bell className="w-8 h-8 text-slate-300 dark:text-slate-700 mb-2" />
                      <p className="text-xs font-semibold text-slate-500 dark:text-slate-400">You&apos;re all caught up</p>
                      <p className="text-[11px] text-slate-400 mt-0.5">No due reminders right now.</p>
                    </div>
                  ) : (
                    dueReminders.map(task => (
                      <div
                        key={task.id}
                        className="flex items-start gap-2.5 px-3.5 py-3 border-b border-slate-50 dark:border-slate-800/60 last:border-b-0"
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold text-slate-700 dark:text-slate-200">{task.title}</p>
                          {task.due_at && (
                            <p className="text-[10px] text-slate-400 mt-0.5">
                              Was due {new Date(task.due_at).toLocaleString()}
                            </p>
                          )}
                        </div>
                        <button
                          onClick={() => handleCompleteTask(task.id)}
                          title="Mark completed"
                          className="p-1 text-slate-300 dark:text-slate-600 hover:text-emerald-500 dark:hover:text-emerald-400 transition-colors cursor-pointer flex-shrink-0"
                        >
                          <Circle className="w-4 h-4" />
                        </button>
                      </div>
                    ))
                  )}
                </div>
                <button
                  onClick={() => {
                    setNotificationsOpen(false)
                    handleOpenTasks()
                  }}
                  className="w-full py-2.5 text-xs font-semibold text-violet-600 dark:text-violet-400 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors cursor-pointer border-t border-slate-100 dark:border-slate-800"
                >
                  View all tasks
                </button>
              </div>
            </>
          )}
        </div>

        {/* Chat Thread / Message History */}
        <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 custom-scrollbar min-h-0 relative z-0">
          {loadingMessages ? (
            <div className="flex flex-col items-center justify-center h-full space-y-3">
              <div className="w-8 h-8 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
              <span className="text-xs text-slate-400">Loading history...</span>
            </div>
          ) : messages.length === 0 && !isStreaming ? (
            /* Welcome / Empty Page State */
            <div className="flex flex-col items-center justify-center h-full max-w-xl mx-auto text-center space-y-8 animate-fade-in py-10">
              <div className="w-16 h-16 bg-gradient-to-tr from-violet-600 to-cyan-500 rounded-2xl flex items-center justify-center shadow-lg shadow-violet-500/10">
                <Bot className="w-10 h-10 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-slate-900 via-slate-700 to-slate-500 dark:from-slate-100 dark:via-slate-300 dark:to-slate-500">
                  Welcome to AetherChat
                </h1>
                <p className="text-sm text-slate-500 mt-2.5 max-w-md mx-auto leading-relaxed font-medium">
                  Deploy autonomous tools, schedule background tasks, and orchestrate advanced agentic workflows in real-time.
                </p>
              </div>

              {/* Quick suggestion prompt cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full mt-4">
                <button
                  onClick={() => handleQuickPrompt("Write a clean, documented Python function to calculate Fibonacci sequences.")}
                  className="p-4 bg-white dark:bg-slate-900/40 hover:bg-slate-50 dark:hover:bg-slate-900/80 border border-slate-205 dark:border-slate-800/80 rounded-2xl text-left transition-all hover:scale-[1.01] hover:border-violet-500/30 group cursor-pointer shadow-sm"
                >
                  <span className="block text-xs font-bold text-slate-700 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-white">Write Python Code</span>
                  <span className="block text-[11px] text-slate-400 mt-1">Generate a documented Fibonacci algorithm.</span>
                </button>
                <button
                  onClick={() => handleQuickPrompt("Explain quantum physics principles in three simple bullet points.")}
                  className="p-4 bg-white dark:bg-slate-900/40 hover:bg-slate-50 dark:hover:bg-slate-900/80 border border-slate-205 dark:border-slate-800/80 rounded-2xl text-left transition-all hover:scale-[1.01] hover:border-cyan-500/30 group cursor-pointer shadow-sm"
                >
                  <span className="block text-xs font-bold text-slate-700 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-white">Explain Physics Concepts</span>
                  <span className="block text-[11px] text-slate-400 mt-1">Summarize quantum principles cleanly.</span>
                </button>
              </div>
            </div>
          ) : (
            /* Chat Messages List */
            <div className={`${chatMaxWidth} mx-auto space-y-6 transition-all duration-300`}>
              {messages.map((message) => {
                const isUser = message.role === 'user'
                const MsgIcon = isUser ? User : Bot
                const providerObj = PROVIDERS.find(p => p.id === message.provider_used)
                const uploadMarkerMatch = isUser ? message.content?.match(/^\[Uploaded document: (.+)\]$/) : null

                return (
                  <div
                    key={message.id}
                    className={`flex animate-fade-in ${isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    {isUser ? (
                      /* USER MESSAGE: Solid theme bubble */
                      <div className="bg-violet-600 text-white px-5 py-3.5 rounded-2xl rounded-tr-sm max-w-[85%] sm:max-w-[75%] shadow-sm text-sm whitespace-pre-wrap leading-relaxed">
                        {uploadMarkerMatch ? (
                          /* File-only upload turn: show a clean attachment chip instead of raw marker text */
                          <div className="flex items-center gap-2 text-violet-50">
                            <FileText className="w-4 h-4 flex-shrink-0" />
                            <span className="font-medium">{uploadMarkerMatch[1]}</span>
                          </div>
                        ) : (
                          <>
                            {message.attachedFileName && (
                              <div className="flex items-center gap-1.5 mb-2 pb-2 border-b border-white/20 text-[11px] font-semibold text-violet-100">
                                <FileText className="w-3.5 h-3.5 flex-shrink-0" />
                                {message.attachedFileName}
                              </div>
                            )}
                            {message.content}
                          </>
                        )}
                      </div>
                    ) : (
                      /* AI MESSAGE: Clean text with bot logo */
                      <div className="flex gap-4 max-w-[95%] sm:max-w-[90%] w-full">
                        {/* AetherChat Logo Avatar */}
                        <div className="w-8 h-8 flex-shrink-0 rounded-xl bg-gradient-to-tr from-violet-600 to-cyan-500 flex items-center justify-center shadow-md mt-1">
                          <Bot className="w-5 h-5 text-white" />
                        </div>
                        <div className="flex flex-col w-full min-w-0">
                          {/* Researched Indicator */}
                          {message.sources && message.sources.length > 0 && (
                            <div className="flex items-center gap-1.5 mb-3 text-[11px] font-semibold text-emerald-600 dark:text-emerald-400">
                              <Check className="w-3.5 h-3.5" />
                              Web researched • {message.sources.length} sources
                            </div>
                          )}

                          {/* Content Body — while the placeholder is still empty and a
                              request is in flight, show the loading indicator in its
                              place instead of a second bubble/avatar below it. */}
                          {message.canvasDocument ? (
                            /* A <canvas>...</canvas> block was detected for this turn
                               (see parseCanvasStream) -- its raw markdown lives in
                               canvasDoc/CanvasWorkspace, never duplicated into this
                               bubble. Any real chat text surrounding the tags (there
                               normally isn't any, per the SYSTEM_PROMPT rule) still
                               renders above the placeholder card. */
                            <div className="space-y-3">
                              {message.content.trim() && (
                                <div className="text-[15px] leading-relaxed select-text break-words text-slate-800 dark:text-slate-200">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {message.content}
                                  </ReactMarkdown>
                                </div>
                              )}
                              <div className="group flex items-center gap-3 px-4 py-3 rounded-xl border border-slate-200/80 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-800/40 hover:border-violet-300 dark:hover:border-violet-600 transition-all w-full sm:w-auto sm:max-w-sm">
                                <div className="w-9 h-9 flex-shrink-0 rounded-lg bg-gradient-to-tr from-violet-600 to-cyan-500 flex items-center justify-center shadow-sm">
                                  <FileText className="w-4.5 h-4.5 text-white" />
                                </div>
                                <div className="min-w-0 flex-1">
                                  <p className="text-sm font-semibold text-slate-800 dark:text-slate-100 truncate">
                                    {message.canvasDocument.title}
                                  </p>
                                  <p className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                                    Generated document
                                  </p>
                                </div>
                                {/* Every card gets its own explicit open action -- each message
                                    carries its own canvasDocument (see fetchMessages/parseCanvasStream
                                    above), so this always swaps canvasDoc to *this* card's document,
                                    letting the user toggle between any past document in the thread. */}
                                <button
                                  type="button"
                                  onClick={() => {
                                    setCanvasDoc(message.canvasDocument!)
                                    setCanvasOpen(true)
                                  }}
                                  title={`Open "${message.canvasDocument.title}" in Canvas`}
                                  aria-label={`Open "${message.canvasDocument.title}" in Canvas`}
                                  className="flex-shrink-0 flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-semibold rounded-lg bg-violet-600 text-white hover:bg-violet-700 transition-colors cursor-pointer"
                                >
                                  <ExternalLink className="w-3.5 h-3.5" />
                                  <span>Open in Canvas</span>
                                </button>
                              </div>
                            </div>
                          ) : !message.content && isLoading ? (
                            <div className="flex items-center gap-2 pt-1 text-slate-400 dark:text-slate-500">
                              <div className="flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                                <span className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                                <span className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                              </div>
                            </div>
                          ) : (
                            <div className="text-[15px] leading-relaxed select-text break-words [&>p]:mb-4 [&>ul]:list-disc [&>ul]:ml-5 [&>ol]:list-decimal [&>ol]:ml-5 [&>ul]:mb-4 [&>ol]:mb-4 [&>ul>li]:mb-1 [&>ol>li]:mb-1 [&_code]:bg-slate-200/50 [&_code]:dark:bg-slate-800 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded-md [&_pre]:bg-slate-900 [&_pre]:text-slate-50 [&_pre]:p-4 [&_pre]:rounded-xl [&_pre]:my-4 [&_pre]:overflow-x-auto [&_a]:text-violet-600 [&_a]:dark:text-violet-400 [&_a]:font-medium [&_a]:hover:underline text-slate-800 dark:text-slate-200">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {message.content}
                              </ReactMarkdown>
                            </div>
                          )}

                        {/* Web Research Sources UI */}
                        {message.sources && message.sources.length > 0 && (
                          <div className="mt-2 pt-4 flex flex-col gap-2.5 w-full">
                            {message.sources.map((source, idx) => (
                              <a
                                key={idx}
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="group flex flex-col p-3 rounded-xl border border-slate-200/80 dark:border-slate-800 bg-white dark:bg-slate-900 hover:border-violet-300 dark:hover:border-violet-600 transition-all text-left shadow-sm hover:shadow-md"
                              >
                                <div className="flex items-start gap-3">
                                  <div className="w-5 h-5 flex-shrink-0 flex items-center justify-center rounded bg-slate-100 dark:bg-slate-800 text-[10px] font-bold text-slate-500 dark:text-slate-400 group-hover:bg-violet-100 group-hover:dark:bg-violet-900/30 group-hover:text-violet-600 group-hover:dark:text-violet-400 transition-colors">
                                    {idx + 1}
                                  </div>
                                  <div className="flex-1 min-w-0">
                                    <h4 className="text-xs font-bold text-slate-800 dark:text-slate-200 truncate group-hover:text-violet-600 dark:group-hover:text-violet-400 transition-colors">
                                      {source.title}
                                    </h4>
                                    <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                                      {source.snippet}
                                    </p>
                                    <span className="text-[10px] font-medium text-slate-400 mt-1.5 block truncate opacity-80 group-hover:text-violet-600 dark:group-hover:text-violet-400 transition-colors">
                                      {(() => {
                                        try {
                                          return new URL(source.url).hostname.replace('www.', '')
                                        } catch {
                                          return source.url
                                        }
                                      })()}
                                    </span>
                                  </div>
                                </div>
                              </a>
                            ))}
                          </div>
                        )}
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}

              {/* Anchor for Auto Scroll */}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Text Form Area */}
        <footer className="p-4 bg-gradient-to-t from-slate-50 via-slate-50 to-transparent dark:from-slate-950 dark:via-slate-950 border-t border-slate-200/60 dark:border-slate-900 relative z-10">
          <div className={`${chatMaxWidth} mx-auto transition-all duration-300`}>
            <form onSubmit={handleSendMessage} className="relative flex flex-col gap-2 bg-white dark:bg-slate-900/80 backdrop-blur-md border border-slate-200/80 dark:border-slate-800 focus-within:border-violet-400 dark:focus-within:border-slate-700/80 focus-within:ring-2 focus-within:ring-violet-500/5 dark:focus-within:ring-0 rounded-2xl p-2 transition-all shadow-sm">
              {/* Hidden file input, restricted to PDFs, triggered by the "Attach PDF" menu item */}
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                onChange={handleFileSelected}
                className="hidden"
              />

              {/* Attachment Indicator - shown above the input row once a PDF is selected */}
              {attachedFile && (
                <div className="flex items-center gap-2 px-1">
                  <div className="flex items-center gap-2 bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-800/50 rounded-full pl-2.5 pr-1.5 py-1 text-xs font-medium text-violet-700 dark:text-violet-300 max-w-full">
                    <FileText className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate max-w-[220px]">{attachedFile.name}</span>
                    <button
                      type="button"
                      onClick={handleRemoveAttachment}
                      className="p-0.5 rounded-full hover:bg-violet-200/60 dark:hover:bg-violet-800/40 text-violet-500 dark:text-violet-400 transition-colors cursor-pointer"
                      title="Remove attachment"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              )}

              <div className="flex items-end gap-2">
                <div className="relative flex-shrink-0 self-end mb-0.5 ml-1">
                  <button
                    type="button"
                    onClick={() => setPlusMenuOpen(!plusMenuOpen)}
                    disabled={isStreaming}
                    className={`p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent ${attachedFile ? 'text-violet-500 bg-violet-50 dark:bg-violet-900/20' : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'}`}
                    title="Attachments & Tools"
                  >
                    <Plus className="w-5 h-5" />
                  </button>

                  {plusMenuOpen && (
                    <>
                      <div className="fixed inset-0 z-20 cursor-default" onClick={() => setPlusMenuOpen(false)} />
                      <div className="absolute left-0 bottom-full mb-2 w-64 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg z-30 p-2 animate-fade-in">
                        <button
                          type="button"
                          onClick={handleAttachButtonClick}
                          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-all text-left cursor-pointer ${attachedFile ? 'bg-violet-50 dark:bg-violet-900/30 text-violet-600 dark:text-violet-400' : 'text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800/50'}`}
                        >
                          <div className="flex items-center gap-2">
                            <Paperclip className="w-4 h-4" />
                            <span>Attach PDF</span>
                          </div>
                          {attachedFile && <Check className="w-4 h-4" />}
                        </button>

                        <div className="mt-1.5 pt-1.5 border-t border-slate-100 dark:border-slate-800">
                          <div className="px-3 py-1 text-[9px] font-bold text-slate-400 uppercase tracking-widest">
                            Connectors
                          </div>
                          {CONNECTORS.map(connector => {
                            const Icon = connector.icon
                            const isOn = enabledConnectors.includes(connector.id)
                            // Reflects the persistent identity link (see
                            // linkedProviders), not just whether a live
                            // access token happens to be cached this
                            // session. Only used for the tooltip here -- no
                            // persistent badge text, to keep the row minimal.
                            const isConnected = linkedProviders[connector.provider]
                            return (
                              // A <div role="button"> rather than a real
                              // <button> here -- the subtle Disconnect
                              // icon-button below needs to live inside this
                              // row, and a <button> nested inside another
                              // <button> is invalid HTML (browsers auto-close
                              // the outer one early, breaking the layout and
                              // click handling both).
                              <div
                                key={connector.id}
                                role="button"
                                tabIndex={0}
                                onClick={() => toggleConnector(connector.id, connector.provider)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter' || e.key === ' ') {
                                    e.preventDefault()
                                    toggleConnector(connector.id, connector.provider)
                                  }
                                }}
                                title={isConnected ? `${connector.label} is linked -- toggle to use it this turn` : `Click to link ${connector.label}`}
                                className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium text-left transition-all cursor-pointer text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800/50"
                              >
                                <div className="flex items-center gap-2">
                                  <Icon className="w-4 h-4" />
                                  <span>{connector.label}</span>
                                  {isConnected && (
                                    <button
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        handleDisconnectConnector(connector.provider)
                                      }}
                                      title={`Disconnect ${connector.label}`}
                                      className="p-0.5 rounded text-slate-300 hover:text-rose-500 dark:text-slate-600 dark:hover:text-rose-400 transition-colors cursor-pointer"
                                    >
                                      <Unlink className="w-3 h-3" />
                                    </button>
                                  )}
                                </div>
                                <div
                                  className={`w-9 h-5 rounded-full transition-colors flex-shrink-0 ${isOn ? 'bg-gradient-to-r from-violet-600 to-cyan-500' : 'bg-slate-200 dark:bg-slate-700'}`}
                                >
                                  <div
                                    className={`w-4 h-4 bg-white rounded-full shadow-sm transform transition-transform mt-0.5 ${isOn ? 'translate-x-4' : 'translate-x-0.5'}`}
                                  />
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    </>
                  )}
                </div>

                <textarea
                  ref={textareaRef}
                  rows={1}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSendMessage(e)
                    }
                  }}
                  disabled={isStreaming}
                  placeholder={isStreaming ? "Awaiting assistant response..." : `Message ${activeProvider.name}...`}
                  className="flex-1 bg-transparent resize-none focus:outline-none border-none py-2 px-3 text-sm text-slate-800 dark:text-slate-200 placeholder-slate-400 dark:placeholder-slate-500 max-h-48 custom-scrollbar min-h-[36px] disabled:opacity-50"
                />

                {/* Model Selector Dropdown - Re-located inside input container, on the right side */}
                <div className="relative flex-shrink-0 self-end mb-0.5">
                  <button
                    type="button"
                    onClick={() => setProviderDropdownOpen(!providerDropdownOpen)}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-semibold shadow-sm transition-all cursor-pointer ${activeProvider.color}`}
                  >
                    <ActiveProviderIcon className="w-4 h-4" />
                    <span className="hidden sm:inline">{activeProvider.name}</span>
                    <ChevronDown className="w-3.5 h-3.5 opacity-60" />
                  </button>

                  {providerDropdownOpen && (
                    <>
                      <div
                        className="fixed inset-0 z-20 cursor-default"
                        onClick={() => setProviderDropdownOpen(false)}
                      />

                      <div className="absolute right-0 bottom-full mb-2 w-56 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg z-30 p-1.5 animate-fade-in">
                        <div className="text-[9px] font-bold text-slate-400 uppercase tracking-widest px-2.5 py-1.5 border-b border-slate-100 dark:border-slate-800 mb-1">
                          Select Brain Engine
                        </div>
                        {PROVIDERS.map(p => {
                          const Icon = p.icon
                          const isSelected = p.id === selectedProvider
                          return (
                            <button
                              key={p.id}
                              type="button"
                              onClick={() => {
                                setSelectedProvider(p.id)
                                setProviderDropdownOpen(false)
                              }}
                              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all text-left cursor-pointer ${isSelected
                                  ? 'bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white font-bold'
                                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-white hover:bg-slate-50 dark:hover:bg-slate-800/50'
                                }`}
                            >
                              <div className="flex items-center gap-2.5">
                                <Icon className={`w-4 h-4 ${p.id === 'gemini' ? 'text-violet-500' : p.id === 'openai' ? 'text-emerald-500' : p.id === 'claude' ? 'text-amber-500' : p.id === 'groq' ? 'text-blue-500' : 'text-slate-500'
                                  }`} />
                                <span>{p.name}</span>
                              </div>
                              {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-violet-600" />}
                            </button>
                          )
                        })}
                      </div>
                    </>
                  )}
                </div>

                <button
                  type="submit"
                  disabled={(!inputText.trim() && !attachedFile) || isStreaming}
                  className="p-2.5 bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 disabled:from-slate-200 dark:disabled:from-slate-800 disabled:to-slate-200 dark:disabled:to-slate-800 text-white dark:disabled:text-slate-500 rounded-xl transition-all shadow-sm cursor-pointer disabled:opacity-55 disabled:cursor-not-allowed disabled:shadow-none flex-shrink-0"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </form>
            <p className="text-[10px] text-center text-slate-400 mt-2.5 font-medium">
              Secure Postgres chat storage. Streaming powered by FastAPI backend.
            </p>
          </div>
        </footer>
      </section>

      {/* Canvas Workspace (Phase 2) -- conditionally rendered split-pane
          panel to the right of the chat thread, see CanvasWorkspace.tsx. */}
      {canvasOpen && (
        <CanvasWorkspace
          title={canvasDoc.title}
          content={canvasDoc.content}
          onClose={handleCloseCanvas}
        />
      )}

      {/* MODALS */}

      {/* Memory Drawer */}
      {showMemory && (
        <div className="fixed inset-0 z-[100] flex justify-end bg-slate-900/40 backdrop-blur-sm animate-fade-in">
          <div className="w-full max-w-md h-full bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 shadow-2xl flex flex-col">
            {/* Header */}
            <div className="p-5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 bg-gradient-to-tr from-violet-600 to-cyan-500 rounded-lg flex items-center justify-center shadow-md">
                  <Database className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="font-bold text-base text-slate-800 dark:text-slate-100">Memory</h2>
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Your continuous profile</span>
                </div>
              </div>
              <button
                onClick={() => setShowMemory(false)}
                className="p-1.5 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Narrative Profile */}
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
              {loadingMemory ? (
                <div className="flex flex-col gap-2">
                  <div className="h-4 bg-slate-100 dark:bg-slate-800/50 rounded animate-pulse" />
                  <div className="h-4 bg-slate-100 dark:bg-slate-800/50 rounded animate-pulse" />
                  <div className="h-4 w-2/3 bg-slate-100 dark:bg-slate-800/50 rounded animate-pulse" />
                </div>
              ) : !memoryProfile.narrative ? (
                <div className="flex flex-col items-center justify-center h-full text-center px-6 py-16">
                  <Database className="w-10 h-10 text-slate-300 dark:text-slate-700 mb-3" />
                  <p className="text-sm font-semibold text-slate-500 dark:text-slate-400">Nothing remembered yet</p>
                  <p className="text-xs text-slate-400 mt-1">As you chat, the AI builds an evolving profile of you here (role, projects, tech stack, goals).</p>
                </div>
              ) : (
                <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-800/40">
                  <p className="text-sm leading-relaxed text-slate-700 dark:text-slate-200 whitespace-pre-wrap">{memoryProfile.narrative}</p>
                  {memoryProfile.updated_at && (
                    <p className="text-[10px] text-slate-400 mt-3">Last updated {new Date(memoryProfile.updated_at).toLocaleString()}</p>
                  )}
                </div>
              )}
            </div>

            {memoryProfile.narrative && (
              <div className="p-4 border-t border-slate-200 dark:border-slate-800">
                <button
                  onClick={() => setShowClearMemoryConfirm(true)}
                  className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl text-sm font-semibold text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-500/10 transition-all cursor-pointer"
                >
                  <Trash2 className="w-4 h-4" />
                  Clear Memory
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Clear Memory Confirmation Modal */}
      {showClearMemoryConfirm && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-fade-in">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 max-w-sm w-full shadow-2xl scale-100 transition-all">
            <div className="flex flex-col items-center text-center">
              <div className="w-12 h-12 bg-rose-100 dark:bg-rose-500/20 rounded-full flex items-center justify-center mb-4">
                <AlertTriangle className="w-6 h-6 text-rose-500" />
              </div>
              <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-2">Clear Memory</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
                Erase your entire memory profile? The AI will no longer take it into account in any chat.
              </p>
              <div className="flex w-full gap-3">
                <button
                  onClick={() => setShowClearMemoryConfirm(false)}
                  disabled={deletingMemory}
                  className="flex-1 py-2.5 px-4 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-semibold rounded-xl transition-all cursor-pointer disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmDeleteMemory}
                  disabled={deletingMemory}
                  className="flex-1 py-2.5 px-4 bg-rose-500 hover:bg-rose-600 text-white font-semibold rounded-xl transition-all shadow-md cursor-pointer disabled:opacity-50"
                >
                  {deletingMemory ? 'Clearing...' : 'Clear'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Task Manager Drawer */}
      {showTasks && (
        <div className="fixed inset-0 z-[100] flex justify-end bg-slate-900/40 backdrop-blur-sm animate-fade-in">
          <div className="w-full max-w-md h-full bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 shadow-2xl flex flex-col">
            {/* Header */}
            <div className="p-5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 bg-gradient-to-tr from-violet-600 to-cyan-500 rounded-lg flex items-center justify-center shadow-md">
                  <ListChecks className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="font-bold text-base text-slate-800 dark:text-slate-100">My Tasks</h2>
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Reminders & to-dos</span>
                </div>
              </div>
              <button
                onClick={() => setShowTasks(false)}
                className="p-1.5 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Status Filter Tabs */}
            <div className="p-3 border-b border-slate-100 dark:border-slate-800/80">
              <div className="grid grid-cols-2 bg-slate-100 dark:bg-slate-800/60 p-1 rounded-xl">
                {(['pending', 'completed'] as const).map(status => (
                  <button
                    key={status}
                    type="button"
                    onClick={() => handleSelectTaskFilter(status)}
                    className={`py-1.5 text-xs font-semibold rounded-lg capitalize transition-all cursor-pointer ${taskStatusFilter === status
                        ? 'bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 shadow-sm'
                        : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                      }`}
                  >
                    {status}
                  </button>
                ))}
              </div>
            </div>

            {/* Task List */}
            <div className="flex-1 overflow-y-auto p-4 custom-scrollbar space-y-2.5">
              {loadingTasks ? (
                <div className="flex flex-col gap-2">
                  <div className="h-14 bg-slate-100 dark:bg-slate-800/50 rounded-xl animate-pulse" />
                  <div className="h-14 bg-slate-100 dark:bg-slate-800/50 rounded-xl animate-pulse" />
                  <div className="h-14 bg-slate-100 dark:bg-slate-800/50 rounded-xl animate-pulse" />
                </div>
              ) : tasks.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center px-6 py-16">
                  <ListChecks className="w-10 h-10 text-slate-300 dark:text-slate-700 mb-3" />
                  <p className="text-sm font-semibold text-slate-500 dark:text-slate-400">
                    No {taskStatusFilter} tasks
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Ask the assistant to remind you of something to see it here.
                  </p>
                </div>
              ) : (
                tasks.map(task => (
                  <div
                    key={task.id}
                    className="group flex items-start gap-3 p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-800/40"
                  >
                    {taskStatusFilter === 'pending' ? (
                      <button
                        onClick={() => handleCompleteTask(task.id)}
                        title="Mark completed"
                        className="mt-0.5 text-slate-300 dark:text-slate-600 hover:text-emerald-500 dark:hover:text-emerald-400 transition-colors cursor-pointer flex-shrink-0"
                      >
                        <Circle className="w-5 h-5" />
                      </button>
                    ) : (
                      <CheckCircle2 className="w-5 h-5 mt-0.5 text-emerald-500 flex-shrink-0" />
                    )}
                    <div className="flex-1 min-w-0">
                      <p className={`text-sm font-semibold text-slate-700 dark:text-slate-200 ${taskStatusFilter === 'completed' ? 'line-through opacity-60' : ''}`}>
                        {task.title}
                      </p>
                      {task.description && (
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{task.description}</p>
                      )}
                      {task.due_at && (
                        <p className="text-[10px] text-slate-400 mt-1 font-medium">
                          Due {new Date(task.due_at).toLocaleString()}
                        </p>
                      )}
                    </div>
                    <button
                      onClick={() => setTaskToDelete(task.id)}
                      title="Delete task"
                      className="p-1 opacity-0 group-hover:opacity-100 hover:bg-slate-200/80 dark:hover:bg-slate-700 rounded text-slate-400 hover:text-rose-500 transition-all cursor-pointer flex-shrink-0"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* Delete Task Confirmation Modal */}
      {taskToDelete && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-fade-in">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 max-w-sm w-full shadow-2xl scale-100 transition-all">
            <div className="flex flex-col items-center text-center">
              <div className="w-12 h-12 bg-rose-100 dark:bg-rose-500/20 rounded-full flex items-center justify-center mb-4">
                <AlertTriangle className="w-6 h-6 text-rose-500" />
              </div>
              <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-2">Delete Task</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
                This action cannot be undone. Are you sure you want to permanently delete this task?
              </p>
              <div className="flex w-full gap-3">
                <button
                  onClick={() => setTaskToDelete(null)}
                  disabled={deletingTask}
                  className="flex-1 py-2.5 px-4 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-semibold rounded-xl transition-all cursor-pointer disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmDeleteTask}
                  disabled={deletingTask}
                  className="flex-1 py-2.5 px-4 bg-rose-500 hover:bg-rose-600 text-white font-semibold rounded-xl transition-all shadow-md cursor-pointer disabled:opacity-50"
                >
                  {deletingTask ? 'Deleting...' : 'Delete'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Logout Confirmation Modal */}
      {showLogoutModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-fade-in">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 max-w-sm w-full shadow-2xl scale-100 transition-all">
            <div className="flex flex-col items-center text-center">
              <div className="w-12 h-12 bg-rose-100 dark:bg-rose-500/20 rounded-full flex items-center justify-center mb-4">
                <LogOut className="w-6 h-6 text-rose-500" />
              </div>
              <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-2">Sign Out</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
                Are you sure you want to log out of AetherChat?
              </p>
              <div className="flex w-full gap-3">
                <button
                  onClick={() => setShowLogoutModal(false)}
                  className="flex-1 py-2.5 px-4 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-semibold rounded-xl transition-all cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmLogout}
                  className="flex-1 py-2.5 px-4 bg-rose-500 hover:bg-rose-600 text-white font-semibold rounded-xl transition-all shadow-md cursor-pointer"
                >
                  Sign Out
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {sessionToDelete && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-fade-in">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 max-w-sm w-full shadow-2xl scale-100 transition-all">
            <div className="flex flex-col items-center text-center">
              <div className="w-12 h-12 bg-rose-100 dark:bg-rose-500/20 rounded-full flex items-center justify-center mb-4">
                <AlertTriangle className="w-6 h-6 text-rose-500" />
              </div>
              <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-2">Delete Chat</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
                This action cannot be undone. Are you sure you want to permanently delete this conversation?
              </p>
              <div className="flex w-full gap-3">
                <button
                  onClick={() => setSessionToDelete(null)}
                  className="flex-1 py-2.5 px-4 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-semibold rounded-xl transition-all cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmDeleteSession}
                  className="flex-1 py-2.5 px-4 bg-rose-500 hover:bg-rose-600 text-white font-semibold rounded-xl transition-all shadow-md cursor-pointer"
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

    </main>
  )
}
