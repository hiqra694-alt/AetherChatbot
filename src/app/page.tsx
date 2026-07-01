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
  Sparkles,
  User,
  Copy,
  Check,
  Brain,
  Zap,
  Flame,
  ChevronDown
} from 'lucide-react'

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
  { id: 'gemini', name: 'Google Gemini', icon: Flame, color: 'text-violet-400 bg-violet-950/50 border-violet-800/50 hover:bg-violet-950/80', accentColor: 'violet' },
  { id: 'openai', name: 'OpenAI ChatGPT', icon: Zap, color: 'text-emerald-400 bg-emerald-950/50 border-emerald-800/50 hover:bg-emerald-950/80', accentColor: 'emerald' },
  { id: 'claude', name: 'Anthropic Claude', icon: Brain, color: 'text-amber-400 bg-amber-950/50 border-amber-800/50 hover:bg-amber-950/80', accentColor: 'amber' }
]

export default function Dashboard() {
  const router = useRouter()
  const supabase = createClient()

  // App state
  const [user, setUser] = useState<any>(null)
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

  // Fetch user data and chat sessions on mount
  useEffect(() => {
    const initApp = async () => {
      const { data: { user } } = await supabase.auth.getUser()
      if (user) {
        setUser(user)
        await fetchSessions(user.id)
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

  const fetchSessions = async (userId: string) => {
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
    // Close sidebar on mobile when session is selected
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

      // 3. Trigger Mock Streaming from API
      setIsStreaming(true)
      setStreamingContent('')

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          provider: selectedProvider,
          sessionId: currentSessionId
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Failed to initialize streaming response.')
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      if (!reader) throw new Error('No stream reader available.')

      let streamedText = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6).trim()
            if (dataStr === '[DONE]') {
              break
            }
            try {
              const parsed = JSON.parse(dataStr)
              if (parsed.content) {
                streamedText += parsed.content
                setStreamingContent(streamedText)
              }
            } catch (err) {
              // Partial JSON or stream packet fragment, safe to ignore
            }
          }
        }
      }

      // 4. Once streaming is complete, append the assistant response to messages state
      const mockAssistantMsg: Message = {
        id: Math.random().toString(), // local temporary ID
        session_id: currentSessionId!,
        role: 'assistant',
        content: streamedText,
        provider_used: selectedProvider,
        created_at: new Date().toISOString()
      }

      setMessages(prev => [...prev, mockAssistantMsg])
      setStreamingContent('')
      setIsStreaming(false)

    } catch (err: any) {
      console.error('Failed to complete message cycle:', err)
      // Append a system error message in the chat
      const errorMsg: Message = {
        id: Math.random().toString(),
        session_id: currentSessionId || '',
        role: 'assistant',
        content: `⚠️ Error: ${err.message || 'Unable to get response from mock AI service. Please check database configuration.'}`,
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
    <main className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* 1. SIDEBAR PANEL */}
      <aside
        className={`fixed md:relative z-20 h-full w-[280px] bg-slate-900 border-r border-slate-800 flex flex-col transition-all duration-300 ${
          sidebarOpen ? 'left-0' : '-left-[280px] md:-ml-[280px]'
        }`}
      >
        {/* Sidebar Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 bg-gradient-to-tr from-violet-600 to-cyan-500 rounded-lg flex items-center justify-center shadow-md">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="font-bold text-base tracking-tight text-white">AetherChat</h2>
              <span className="text-[10px] text-cyan-400 font-semibold uppercase tracking-wider">Multi-AI Portal</span>
            </div>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="md:hidden p-1.5 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-3">
          <button
            onClick={handleNewChat}
            className="w-full py-2.5 px-4 bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 text-white font-medium rounded-xl transition-all shadow-md shadow-violet-500/5 hover:shadow-violet-500/10 active:scale-[0.98] flex items-center justify-center gap-2 cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            New Chat Thread
          </button>
        </div>

        {/* Sessions History List */}
        <div className="flex-1 overflow-y-auto px-2 space-y-1 py-2 custom-scrollbar">
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider px-3 mb-2">
            Recent Dialogues
          </div>
          
          {loadingSessions ? (
            <div className="flex flex-col gap-2 p-3">
              <div className="h-8 bg-slate-800/50 rounded-lg animate-pulse" />
              <div className="h-8 bg-slate-800/50 rounded-lg animate-pulse" />
              <div className="h-8 bg-slate-800/50 rounded-lg animate-pulse" />
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-center py-8 px-4 text-xs text-slate-500">
              No chat history yet. Send a message to start!
            </div>
          ) : (
            sessions.map(session => (
              <div
                key={session.id}
                onClick={() => handleSelectSession(session.id)}
                className={`group flex items-center justify-between p-3 rounded-xl cursor-pointer transition-all border ${
                  activeSessionId === session.id
                    ? 'bg-slate-800/80 border-slate-700 text-white shadow-inner'
                    : 'bg-transparent border-transparent hover:bg-slate-800/30 text-slate-400 hover:text-slate-200'
                }`}
              >
                <div className="flex items-center gap-2.5 overflow-hidden w-[80%]">
                  <MessageSquare className={`w-4.5 h-4.5 flex-shrink-0 ${
                    activeSessionId === session.id ? 'text-cyan-400' : 'text-slate-500'
                  }`} />
                  <span className="text-sm truncate font-medium">{session.title}</span>
                </div>
                <button
                  onClick={(e) => handleDeleteSession(e, session.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-slate-700/80 rounded text-slate-500 hover:text-rose-400 transition-all cursor-pointer"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))
          )}
        </div>

        {/* User Card & Logout */}
        {user && (
          <div className="p-4 border-t border-slate-800 bg-slate-900/60 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5 overflow-hidden">
              <div className="w-8 h-8 rounded-full bg-violet-600/20 border border-violet-500/30 flex items-center justify-center text-violet-400 font-bold uppercase flex-shrink-0 text-sm">
                {user.email?.charAt(0) || 'U'}
              </div>
              <div className="flex flex-col overflow-hidden">
                <span className="text-xs font-semibold text-slate-200 truncate">{user.email}</span>
                <span className="text-[9px] text-slate-500 uppercase tracking-wider font-bold">Standard Account</span>
              </div>
            </div>
            <button
              onClick={handleLogout}
              title="Sign Out"
              className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-rose-400 transition-colors cursor-pointer"
            >
              <LogOut className="w-4.5 h-4.5" />
            </button>
          </div>
        )}
      </aside>

      {/* 2. MAIN CHAT AREA */}
      <section className="flex-1 h-screen max-h-screen flex flex-col overflow-hidden bg-slate-950 relative min-w-0">
        {/* Decorative background glows */}
        <div className="absolute top-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-violet-600/5 blur-[100px] pointer-events-none" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-cyan-600/5 blur-[100px] pointer-events-none" />

        {/* Top Navigation Bar */}
        <header className="h-16 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md flex items-center justify-between px-4 z-10">
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-2 hover:bg-slate-900 rounded-lg text-slate-400 hover:text-white transition-colors cursor-pointer"
              >
                <Menu className="w-5 h-5" />
              </button>
            )}
            
            {/* Active Thread Details */}
            <div className="hidden sm:flex flex-col">
              <h3 className="text-sm font-bold text-white truncate max-w-[200px] md:max-w-[400px]">
                {activeSessionId ? sessions.find(s => s.id === activeSessionId)?.title : 'New Workspace'}
              </h3>
              <span className="text-[10px] text-slate-500 font-medium">
                {activeSessionId ? 'Saved to Cloud Database' : 'Drafting message...'}
              </span>
            </div>
          </div>

          {/* Model Selector Dropdown */}
          <div className="relative">
            <button
              onClick={() => setProviderDropdownOpen(!providerDropdownOpen)}
              className={`flex items-center gap-2.5 px-3 py-1.5 rounded-xl border text-xs font-semibold shadow-sm transition-all cursor-pointer ${activeProvider.color}`}
            >
              <ActiveProviderIcon className="w-4 h-4" />
              <span>{activeProvider.name}</span>
              <ChevronDown className="w-3.5 h-3.5 opacity-60" />
            </button>

            {providerDropdownOpen && (
              <>
                {/* Backdrop overlay to close dropdown */}
                <div
                  className="fixed inset-0 z-20 cursor-default"
                  onClick={() => setProviderDropdownOpen(false)}
                />
                
                <div className="absolute right-0 mt-2 w-56 bg-slate-900 border border-slate-800 rounded-xl shadow-xl z-30 p-1.5 animate-fade-in">
                  <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest px-2.5 py-1.5 border-b border-slate-800 mb-1">
                    Select Brain Engine
                  </div>
                  {PROVIDERS.map(p => {
                    const Icon = p.icon
                    const isSelected = p.id === selectedProvider
                    return (
                      <button
                        key={p.id}
                        onClick={() => {
                          setSelectedProvider(p.id)
                          setProviderDropdownOpen(false)
                        }}
                        className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all text-left cursor-pointer ${
                          isSelected
                            ? 'bg-slate-800/80 text-white font-bold'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/30'
                        }`}
                      >
                        <div className="flex items-center gap-2.5">
                          <Icon className={`w-4 h-4 ${
                            p.id === 'gemini' ? 'text-violet-400' : p.id === 'openai' ? 'text-emerald-400' : 'text-amber-400'
                          }`} />
                          <span>{p.name}</span>
                        </div>
                        {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />}
                      </button>
                    )
                  })}
                </div>
              </>
            )}
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
                <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-100 to-slate-400">
                  Welcome to AetherChat
                </h1>
                <p className="text-sm text-slate-400 mt-2.5 max-w-md mx-auto leading-relaxed">
                  Connect securely, switch between LLM providers on the fly, and experience real-time responses.
                </p>
              </div>

              {/* Quick suggestion prompt cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full mt-4">
                <button
                  onClick={() => handleQuickPrompt("Write a clean, documented Python function to calculate Fibonacci sequences.")}
                  className="p-4 bg-slate-900/40 hover:bg-slate-900/80 border border-slate-800/80 rounded-2xl text-left transition-all hover:scale-[1.01] hover:border-violet-500/30 group cursor-pointer"
                >
                  <span className="block text-xs font-bold text-slate-300 group-hover:text-white">Write Python Code</span>
                  <span className="block text-[11px] text-slate-500 mt-1">Generate a documented Fibonacci algorithm.</span>
                </button>
                <button
                  onClick={() => handleQuickPrompt("Explain quantum physics principles in three simple bullet points.")}
                  className="p-4 bg-slate-900/40 hover:bg-slate-900/80 border border-slate-800/80 rounded-2xl text-left transition-all hover:scale-[1.01] hover:border-cyan-500/30 group cursor-pointer"
                >
                  <span className="block text-xs font-bold text-slate-300 group-hover:text-white">Explain Physics Concepts</span>
                  <span className="block text-[11px] text-slate-500 mt-1">Summarize quantum principles cleanly.</span>
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
                      className={`relative flex flex-col p-4 rounded-2xl border transition-all ${
                        isUser
                          ? 'bg-violet-600/10 border-violet-500/20 text-slate-200 max-w-[85%] sm:max-w-[75%]'
                          : 'bg-slate-900/50 border-slate-800/80 text-slate-300 max-w-[85%] sm:max-w-[75%] shadow-md'
                      }`}
                    >
                      {/* Message Meta Header */}
                      <div className="flex items-center gap-2 mb-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                        <MsgIcon className="w-3.5 h-3.5" />
                        <span>{isUser ? 'User Message' : 'AI Assistant'}</span>
                        {!isUser && providerObj && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-400">
                            {providerObj.name}
                          </span>
                        )}
                      </div>

                      {/* Content Body */}
                      <p className="text-sm leading-relaxed whitespace-pre-wrap select-text break-words">
                        {message.content}
                      </p>

                      {/* Floating utilities (e.g. Copy button) */}
                      <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 hover:opacity-100 transition-opacity">
                        {/* Wait, the container needs 'group' class to trigger this. Let's make sure it does. */}
                      </div>
                      <div className="flex justify-end mt-2 pt-1 border-t border-slate-800/40">
                        <button
                          onClick={() => handleCopyText(message.content, message.id)}
                          className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-slate-300 font-semibold transition-colors cursor-pointer"
                        >
                          {copiedId === message.id ? (
                            <>
                              <Check className="w-3 h-3 text-emerald-400" />
                              <span className="text-emerald-400">Copied</span>
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
                  <div className="flex flex-col p-4 rounded-2xl border bg-slate-900/50 border-slate-800/80 text-slate-300 max-w-[85%] sm:max-w-[75%] shadow-md">
                    <div className="flex items-center gap-2 mb-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                      <Bot className="w-3.5 h-3.5" />
                      <span>AI Assistant</span>
                      <span className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-violet-400 animate-pulse">
                        {activeProvider.name} (streaming)
                      </span>
                    </div>

                    <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
                      {streamingContent}
                      <span className="inline-block w-1.5 h-4 ml-1 bg-violet-400 animate-pulse align-middle" />
                    </p>
                  </div>
                </div>
              )}

              {/* Pulsing loader when waiting for API route response */}
              {isStreaming && !streamingContent && (
                <div className="flex gap-4 justify-start animate-fade-in">
                  <div className="flex flex-col p-4 rounded-2xl border bg-slate-900/50 border-slate-800/80 text-slate-400 w-44 shadow-md">
                    <div className="flex items-center gap-2 mb-1.5 text-[10px] font-bold text-slate-500 uppercase tracking-wider">
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
        <footer className="p-4 bg-gradient-to-t from-slate-950 via-slate-950 to-transparent border-t border-slate-900 relative z-10">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={handleSendMessage} className="relative flex items-end gap-2 bg-slate-900/80 backdrop-blur-md border border-slate-850 focus-within:border-slate-700/80 rounded-2xl p-2 transition-all">
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
                className="flex-1 bg-transparent resize-none focus:outline-none border-none py-2 px-3 text-sm text-slate-100 placeholder-slate-500 max-h-48 custom-scrollbar min-h-[36px] disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={!inputText.trim() || isStreaming}
                className="p-2.5 bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 disabled:from-slate-800 disabled:to-slate-800 text-white rounded-xl transition-all shadow-md cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none flex-shrink-0"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
            <p className="text-[10px] text-center text-slate-600 mt-2.5">
              Secure Postgres chat storage. Mock responses stream at 50ms intervals.
            </p>
          </div>
        </footer>
      </section>
    </main>
  )
}
