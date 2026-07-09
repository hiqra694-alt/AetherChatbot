'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/utils/supabase/client'
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
  Cpu
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

interface Message {
  id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  provider_used: string | null
  created_at: string
}

const PROVIDERS = [
  { id: 'gemini', name: 'Google Gemini', icon: Flame, color: 'text-violet-600 bg-violet-50 border-violet-200 hover:bg-violet-100', accentColor: 'violet' },
  { id: 'openai', name: 'OpenAI ChatGPT', icon: Zap, color: 'text-emerald-600 bg-emerald-50 border-emerald-200 hover:bg-emerald-100', accentColor: 'emerald' },
  { id: 'claude', name: 'Anthropic Claude', icon: Brain, color: 'text-amber-600 bg-amber-50 border-amber-200 hover:bg-amber-100', accentColor: 'amber' },
  { id: 'groq', name: 'Groq LPU', icon: Cpu, color: 'text-blue-600 bg-blue-50 border-blue-200 hover:bg-blue-100', accentColor: 'blue' },
  { id: 'mock', name: 'Mock AI Provider', icon: Bot, color: 'text-slate-600 bg-slate-50 border-slate-200 hover:bg-slate-100', accentColor: 'slate' }
]

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
  const [selectedProvider, setSelectedProvider] = useState('gemini')
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [providerDropdownOpen, setProviderDropdownOpen] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [loadingMessages, setLoadingMessages] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

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
    const initApp = async () => {
      const { data: { user } } = await supabase.auth.getUser()
      if (user) {
        // Map user properties to avoid type issues
        setUser({ id: user.id, email: user.email })
        await fetchSessions()
      } else {
        router.push('/login')
      }
    }
    initApp()
  }, [])

  // Auto-scroll to bottom on message list updates or streaming content updates
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent, isStreaming])

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
      const { data, error } = await supabase
        .from('messages')
        .select('*')
        .eq('session_id', sessionId)
        .order('created_at', { ascending: true })

      if (error) throw error
      setMessages(data || [])
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
    setStreamingContent('')
    setIsStreaming(false)
    await fetchMessages(sessionId)
    if (window.innerWidth < 768) {
      setSidebarOpen(false)
    }
  }

  const handleNewChat = () => {
    setActiveSessionId(null)
    setMessages([])
    setStreamingContent('')
    setIsStreaming(false)
    setInputText('')
    setProviderDropdownOpen(false)  // always close any open dropdown
    if (window.innerWidth < 768) {
      setSidebarOpen(false)
    }
    setTimeout(() => textareaRef.current?.focus(), 50)
  }

  const handleDeleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    try {
      const { error } = await supabase
        .from('chat_sessions')
        .delete()
        .eq('id', sessionId)

      if (error) throw error

      setSessions(prev => prev.filter(s => s.id !== sessionId))
      if (activeSessionId === sessionId) {
        handleNewChat()
      }
    } catch (err) {
      console.error('Error deleting session:', err)
    }
  }

  const handleLogout = async () => {
    await supabase.auth.signOut()
    router.refresh()
    router.push('/login')
  }

  const handleCopyText = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!inputText.trim() || isStreaming) return

    const messageContent = inputText.trim()
    setInputText('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    let currentSessionId = activeSessionId

    try {
      // 1. Create a session if none is active
      if (!currentSessionId) {
        const title = messageContent.length > 30 ? messageContent.slice(0, 30) + '...' : messageContent
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

      // 2. Insert User Message into DB
      const { data: userMsg, error: msgErr } = await supabase
        .from('messages')
        .insert({
          session_id: currentSessionId,
          role: 'user',
          content: messageContent,
          provider_used: selectedProvider
        })
        .select()
        .single()

      if (msgErr) throw msgErr
      if (userMsg) {
        setMessages(prev => [...prev, userMsg])
      }

      // 3. Trigger Streaming from Backend Proxy (using Supabase Auth JWT header)
      setIsStreaming(true)
      setStreamingContent('')

      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          provider: selectedProvider,
          sessionId: currentSessionId
        })
      })

      if (!response.ok) {
        // Safely parse error — backend may return plain text (e.g. "Internal Server Error")
        let errDetail = `HTTP ${response.status}: Request failed.`
        try {
          const errorData = await response.json()
          errDetail = errorData.detail || errorData.error || errDetail
        } catch {
          try { errDetail = await response.text() } catch { /* ignore */ }
        }
        throw new Error(errDetail)
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      if (!reader) throw new Error('No stream reader available.')

      let streamedText = ''
      let streamError: string | null = null

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim()
            if (dataStr === '[DONE]') break
            try {
              const parsed = JSON.parse(dataStr)
              if (parsed.content) {
                streamedText += parsed.content
                setStreamingContent(streamedText)
              } else if (parsed.error) {
                // Backend sent an error event (quota, model not found, etc.)
                streamError = parsed.error
              }
            } catch {
              // Partial JSON fragment — safe to ignore
            }
          }
        }

        // Stop reading as soon as an error was signaled
        if (streamError) break
      }

      if (streamError) {
        throw new Error(streamError)
      }

      // 4. Once streaming is complete, append the assistant response to messages state
      const mockAssistantMsg: Message = {
        id: Math.random().toString(),
        session_id: currentSessionId!,
        role: 'assistant',
        content: streamedText,
        provider_used: selectedProvider,
        created_at: new Date().toISOString()
      }

      setMessages(prev => [...prev, mockAssistantMsg])
      setStreamingContent('')
      setIsStreaming(false)

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
        session_id: currentSessionId || '',
        role: 'assistant',
        content: `⚠️ ${errMsg}`,
        provider_used: selectedProvider,
        created_at: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMsg])
      setIsStreaming(false)
      setStreamingContent('')
    }
  }

  const handleQuickPrompt = (promptText: string) => {
    setInputText(promptText)
    setTimeout(() => textareaRef.current?.focus(), 50)
  }

  const activeProvider = PROVIDERS.find(p => p.id === selectedProvider) || PROVIDERS[0]
  const ActiveProviderIcon = activeProvider.icon

  return (
    <main className="flex h-screen w-screen bg-slate-50 text-slate-800 overflow-hidden font-sans">
      {/* 1. SIDEBAR PANEL */}
      <aside
        className={`fixed md:relative z-20 h-full w-[280px] bg-slate-100/90 border-r border-slate-200/80 flex flex-col transition-all duration-300 ${sidebarOpen ? 'left-0' : '-left-[280px] md:-ml-[280px]'
          }`}
      >
        {/* Sidebar Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 bg-gradient-to-tr from-violet-600 to-cyan-500 rounded-lg flex items-center justify-center shadow-md">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="font-bold text-base tracking-tight text-slate-800">AetherChat</h2>
              <span className="text-[10px] text-violet-600 font-bold uppercase tracking-wider">Multi-AI Portal</span>
            </div>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="md:hidden p-1.5 hover:bg-slate-200 rounded-lg text-slate-500 hover:text-slate-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-3">
          <button
            onClick={handleNewChat}
            className="w-full py-2.5 px-4 bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 text-white font-semibold rounded-xl transition-all shadow-md active:scale-[0.98] flex items-center justify-center gap-2 cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            New Chat Thread
          </button>
        </div>

        {/* Sessions History List */}
        <div className="flex-1 overflow-y-auto px-2 space-y-1 py-2 custom-scrollbar">
          <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-3 mb-2">
            Recent Dialogues
          </div>

          {loadingSessions ? (
            <div className="flex flex-col gap-2 p-3">
              <div className="h-8 bg-slate-200/50 rounded-lg animate-pulse" />
              <div className="h-8 bg-slate-200/50 rounded-lg animate-pulse" />
              <div className="h-8 bg-slate-200/50 rounded-lg animate-pulse" />
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
                className={`group flex items-center justify-between p-3 rounded-xl cursor-pointer transition-all border ${activeSessionId === session.id
                    ? 'bg-white border-slate-200 text-slate-950 shadow-sm font-semibold'
                    : 'bg-transparent border-transparent hover:bg-slate-200/40 text-slate-500 hover:text-slate-800'
                  }`}
              >
                <div className="flex items-center gap-2.5 overflow-hidden w-[80%]">
                  <MessageSquare className={`w-4.5 h-4.5 flex-shrink-0 ${activeSessionId === session.id ? 'text-violet-500' : 'text-slate-400'
                    }`} />
                  <span className="text-sm truncate">{session.title}</span>
                </div>
                <button
                  onClick={(e) => handleDeleteSession(e, session.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-slate-200/80 rounded text-slate-400 hover:text-rose-500 transition-all cursor-pointer"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))
          )}
        </div>

        {/* User Card & Logout */}
        {user && (
          <div className="p-4 border-t border-slate-200 bg-slate-100/50 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5 overflow-hidden">
              <div className="w-8 h-8 rounded-full bg-violet-600/10 border border-violet-500/20 flex items-center justify-center text-violet-600 font-bold uppercase flex-shrink-0 text-sm">
                {user.email?.charAt(0) || 'U'}
              </div>
              <div className="flex flex-col overflow-hidden">
                <span className="text-xs font-semibold text-slate-850 truncate">{user.email}</span>
                <span className="text-[9px] text-slate-400 uppercase tracking-wider font-bold">Standard Account</span>
              </div>
            </div>
            <button
              onClick={handleLogout}
              title="Sign Out"
              className="p-2 hover:bg-slate-200 rounded-lg text-slate-500 hover:text-rose-500 transition-colors cursor-pointer"
            >
              <LogOut className="w-4.5 h-4.5" />
            </button>
          </div>
        )}
      </aside>

      {/* 2. MAIN CHAT AREA — keyed by session ID so React fully remounts on every session change */}
      <section key={activeSessionId ?? 'new-chat'} className="flex-1 h-screen max-h-screen flex flex-col overflow-hidden bg-slate-50/50 relative min-w-0">
        {/* Decorative background glows */}
        <div className="absolute top-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-violet-500/5 blur-[100px] pointer-events-none" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-cyan-500/5 blur-[100px] pointer-events-none" />

        {/* Top Navigation Bar */}
        <header className="h-16 border-b border-slate-200/80 bg-white/80 backdrop-blur-md flex items-center justify-between px-4 z-10">
          {/* Left side: menu toggle + thread title — flex-1 min-w-0 lets it shrink without clipping right side */}
          <div className="flex items-center gap-3 flex-1 min-w-0">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-2 hover:bg-slate-100 rounded-lg text-slate-500 hover:text-slate-800 transition-colors cursor-pointer"
              >
                <Menu className="w-5 h-5" />
              </button>
            )}

            {/* Active Thread Details */}
            <div className="hidden sm:flex flex-col min-w-0">
              <h3 className="text-sm font-bold text-slate-800 truncate max-w-[200px] md:max-w-[400px]">
                {activeSessionId ? sessions.find(s => s.id === activeSessionId)?.title : 'New Workspace'}
              </h3>
              <span className="text-[10px] text-slate-400 font-medium">
                {activeSessionId ? 'Saved to Cloud Database' : 'Drafting message...'}
              </span>
            </div>
          </div>


        </header>

        {/* Chat Thread / Message History */}
        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 custom-scrollbar min-h-0 relative z-0">
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
                <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-slate-900 via-slate-700 to-slate-500">
                  Welcome to AetherChat
                </h1>
                <p className="text-sm text-slate-500 mt-2.5 max-w-md mx-auto leading-relaxed font-medium">
                  Connect securely, switch between LLM providers on the fly, and experience real-time responses.
                </p>
              </div>

              {/* Quick suggestion prompt cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full mt-4">
                <button
                  onClick={() => handleQuickPrompt("Write a clean, documented Python function to calculate Fibonacci sequences.")}
                  className="p-4 bg-white hover:bg-slate-50 border border-slate-205 rounded-2xl text-left transition-all hover:scale-[1.01] hover:border-violet-500/30 group cursor-pointer shadow-sm"
                >
                  <span className="block text-xs font-bold text-slate-700 group-hover:text-slate-900">Write Python Code</span>
                  <span className="block text-[11px] text-slate-400 mt-1">Generate a documented Fibonacci algorithm.</span>
                </button>
                <button
                  onClick={() => handleQuickPrompt("Explain quantum physics principles in three simple bullet points.")}
                  className="p-4 bg-white hover:bg-slate-50 border border-slate-205 rounded-2xl text-left transition-all hover:scale-[1.01] hover:border-cyan-500/30 group cursor-pointer shadow-sm"
                >
                  <span className="block text-xs font-bold text-slate-700 group-hover:text-slate-900">Explain Physics Concepts</span>
                  <span className="block text-[11px] text-slate-400 mt-1">Summarize quantum principles cleanly.</span>
                </button>
              </div>
            </div>
          ) : (
            /* Chat Messages List */
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((message) => {
                const isUser = message.role === 'user'
                const MsgIcon = isUser ? User : Bot
                const providerObj = PROVIDERS.find(p => p.id === message.provider_used)

                return (
                  <div
                    key={message.id}
                    className={`flex gap-4 animate-fade-in ${isUser ? 'justify-end' : 'justify-start'}`}
                  >
                    {/* Message Bubble Container */}
                    <div
                      className={`relative flex flex-col p-4 rounded-2xl border transition-all ${isUser
                          ? 'bg-violet-50 border-violet-100 text-slate-800 max-w-[85%] sm:max-w-[75%]'
                          : 'bg-white border-slate-200/80 text-slate-800 max-w-[85%] sm:max-w-[75%] shadow-sm'
                        }`}
                    >
                      {/* Message Meta Header */}
                      <div className="flex items-center gap-2 mb-2 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                        <MsgIcon className="w-3.5 h-3.5" />
                        <span>{isUser ? 'User Message' : 'AI Assistant'}</span>
                        {!isUser && providerObj && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-650">
                            {providerObj.name}
                          </span>
                        )}
                      </div>

                      {/* Content Body */}
                      <p className="text-sm leading-relaxed whitespace-pre-wrap select-text break-words">
                        {message.content}
                      </p>

                      <div className="flex justify-end mt-2 pt-1 border-t border-slate-100">
                        <button
                          onClick={() => handleCopyText(message.content, message.id)}
                          className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-600 font-semibold transition-colors cursor-pointer"
                        >
                          {copiedId === message.id ? (
                            <>
                              <Check className="w-3 h-3 text-emerald-600" />
                              <span className="text-emerald-600">Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3" />
                              <span>Copy</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                )
              })}

              {/* Real-time Streaming Response Rendering */}
              {isStreaming && streamingContent && (
                <div className="flex gap-4 justify-start animate-fade-in">
                  <div className="flex flex-col p-4 rounded-2xl border bg-white border-slate-200/80 text-slate-800 max-w-[85%] sm:max-w-[75%] shadow-sm">
                    <div className="flex items-center gap-2 mb-2 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                      <Bot className="w-3.5 h-3.5" />
                      <span>AI Assistant</span>
                      <span className="px-1.5 py-0.5 rounded bg-violet-50 border border-violet-100 text-violet-600 animate-pulse">
                        {activeProvider.name} (streaming)
                      </span>
                    </div>

                    <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
                      {streamingContent}
                      <span className="inline-block w-1.5 h-4 ml-1 bg-violet-500 animate-pulse align-middle" />
                    </p>
                  </div>
                </div>
              )}

              {/* Pulsing loader when waiting for API route response */}
              {isStreaming && !streamingContent && (
                <div className="flex gap-4 justify-start animate-fade-in">
                  <div className="flex flex-col p-4 rounded-2xl border bg-white border-slate-200 text-slate-400 w-44 shadow-sm">
                    <div className="flex items-center gap-2 mb-1.5 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                      <Bot className="w-3.5 h-3.5" />
                      <span>Thinking...</span>
                    </div>
                    <div className="flex items-center gap-1.5 py-1">
                      <div className="w-2 h-2 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 rounded-full bg-violet-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              )}

              {/* Anchor for Auto Scroll */}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Text Form Area */}
        <footer className="p-4 bg-gradient-to-t from-slate-50 via-slate-50 to-transparent border-t border-slate-200/60 relative z-10">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSendMessage} className="relative flex items-end gap-2 bg-white border border-slate-200/80 focus-within:border-violet-400 focus-within:ring-2 focus-within:ring-violet-500/5 rounded-2xl p-2 transition-all shadow-sm">
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
                className="flex-1 bg-transparent resize-none focus:outline-none border-none py-2 px-3 text-sm text-slate-800 placeholder-slate-400 max-h-48 custom-scrollbar min-h-[36px] disabled:opacity-50"
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

                    <div className="absolute right-0 bottom-full mb-2 w-56 bg-white border border-slate-200 rounded-xl shadow-lg z-30 p-1.5 animate-fade-in">
                      <div className="text-[9px] font-bold text-slate-400 uppercase tracking-widest px-2.5 py-1.5 border-b border-slate-100 mb-1">
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
                                ? 'bg-slate-100 text-slate-900 font-bold'
                                : 'text-slate-500 hover:text-slate-800 hover:bg-slate-50'
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
                disabled={!inputText.trim() || isStreaming}
                className="p-2.5 bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 disabled:from-slate-200 disabled:to-slate-200 text-white rounded-xl transition-all shadow-sm cursor-pointer disabled:opacity-55 disabled:cursor-not-allowed disabled:shadow-none flex-shrink-0"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
            <p className="text-[10px] text-center text-slate-400 mt-2.5 font-medium">
              Secure Postgres chat storage. Streaming powered by FastAPI backend.
            </p>
          </div>
        </footer>
      </section>
    </main>
  )
}
