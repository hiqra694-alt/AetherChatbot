'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/utils/supabase/client'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
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
  Edit2,
  AlertTriangle,
  Paperclip,
  FileText,
  Database
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

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  provider_used?: string
  sources?: Source[]
  attachedFileName?: string
}

interface KnowledgeDocument {
  document_name: string
  chunk_count: number
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
  const [isLoading, setIsLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [providerDropdownOpen, setProviderDropdownOpen] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [theme, setTheme] = useState('light')

  // New Feature States
  const [showLogoutModal, setShowLogoutModal] = useState(false)
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null)
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null)
  const [editTitleText, setEditTitleText] = useState('')
  const [plusMenuOpen, setPlusMenuOpen] = useState(false)
  const [attachedFile, setAttachedFile] = useState<File | null>(null)
  const [showKnowledgeBase, setShowKnowledgeBase] = useState(false)
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([])
  const [loadingDocuments, setLoadingDocuments] = useState(false)
  const [documentToDelete, setDocumentToDelete] = useState<string | null>(null)
  const [deletingDocument, setDeletingDocument] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

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
    // Check local storage for theme
    const savedTheme = localStorage.getItem('aether_theme')
    if (savedTheme) {
      setTheme(savedTheme)
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setTheme('dark')
    }

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

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('aether_theme', theme)
  }, [theme])

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
          
          return {
            id: msg.id,
            role: msg.role,
            content: content.trim(),
            provider_used: msg.provider_used,
            sources: sources
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

  const fetchDocuments = async () => {
    setLoadingDocuments(true)
    try {
      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const response = await fetch('/api/documents', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!response.ok) throw new Error('Failed to fetch documents.')

      const data = await response.json()
      setDocuments(data.documents || [])
    } catch (err) {
      console.error('Error fetching documents:', err)
    } finally {
      setLoadingDocuments(false)
    }
  }

  const handleOpenKnowledgeBase = () => {
    setShowKnowledgeBase(true)
    fetchDocuments()
  }

  const confirmDeleteDocument = async () => {
    if (!documentToDelete) return
    setDeletingDocument(true)
    try {
      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const response = await fetch(`/api/documents/${encodeURIComponent(documentToDelete)}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (!response.ok) throw new Error('Failed to delete document.')

      setDocuments(prev => prev.filter(d => d.document_name !== documentToDelete))
    } catch (err) {
      console.error('Error deleting document:', err)
    } finally {
      setDeletingDocument(false)
      setDocumentToDelete(null)
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
      setMessages(prev => [...prev, { id: 'temp', role: 'assistant', content: '', provider_used: selectedProvider }])

      const { data: { session } } = await supabase.auth.getSession()
      const token = session?.access_token || ''

      const formData = new FormData()
      formData.append('provider', selectedProvider)
      formData.append('sessionId', currentSessionId as string)
      if (messageContent) formData.append('message', messageContent)
      if (fileToSend) formData.append('file', fileToSend)

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
                setMessages(prev => {
                  const newMsgs = [...prev]
                  const last = newMsgs[newMsgs.length - 1]
                  if (last && last.role === 'assistant') {
                    last.content = streamedContent
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

      // A streamError from the backend (e.g. Groq failed to format a tool
      // call) means generation was cut short mid-response, not that nothing
      // happened — `streamedContent` may already hold real, valid text the
      // user has been watching stream in. Append an inline notice instead of
      // throwing, so that partial answer is preserved rather than replaced
      // wholesale by an error bubble.
      const finalContent = streamError
        ? `${streamedContent}${streamedContent.trim() ? '\n\n' : ''}⚠️ *The agent encountered an error formatting its response. Please try again.*`
        : streamedContent

      // 4. Once streaming is complete, append the assistant response to messages state
      const mockAssistantMsg: Message = {
        id: Math.random().toString(),
        role: 'assistant',
        content: finalContent,
        provider_used: selectedProvider,
        sources: streamedSources || undefined
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
              <span className="text-[10px] text-violet-600 font-bold uppercase tracking-wider">Multi-AI Portal</span>
            </div>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="md:hidden p-1.5 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-3 space-y-2">
          <button
            onClick={handleNewChat}
            className="w-full py-2.5 px-4 bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 text-white font-semibold rounded-xl transition-all shadow-md active:scale-[0.98] flex items-center justify-center gap-2 cursor-pointer"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </button>
          <button
            onClick={handleOpenKnowledgeBase}
            className="w-full py-2.5 px-4 bg-white dark:bg-slate-800/60 hover:bg-slate-50 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 font-semibold rounded-xl transition-all active:scale-[0.98] flex items-center justify-center gap-2 cursor-pointer"
          >
            <Database className="w-4 h-4" />
            My Knowledge Base
          </button>
        </div>

        {/* Sessions History List */}
        <div className="flex-1 overflow-y-auto px-2 space-y-1 py-2 custom-scrollbar">
          <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider px-3 mb-2">
            Recent Dialogues
          </div>

          {loadingSessions ? (
            <div className="flex flex-col gap-2 p-3">
              <div className="h-8 bg-slate-200/50 dark:bg-slate-800/50 rounded-lg animate-pulse" />
              <div className="h-8 bg-slate-200/50 dark:bg-slate-800/50 rounded-lg animate-pulse" />
              <div className="h-8 bg-slate-200/50 dark:bg-slate-800/50 rounded-lg animate-pulse" />
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
                    <div className="flex items-center gap-2.5 overflow-hidden w-[70%]">
                      <MessageSquare className={`w-4.5 h-4.5 flex-shrink-0 ${activeSessionId === session.id ? 'text-violet-500' : 'text-slate-400'
                        }`} />
                      <span className="text-sm truncate">{session.title}</span>
                    </div>
                    <div className="flex opacity-0 group-hover:opacity-100 transition-all">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          setEditingSessionId(session.id)
                          setEditTitleText(session.title)
                        }}
                        className="p-1 hover:bg-slate-200/80 dark:hover:bg-slate-700 rounded text-slate-400 hover:text-violet-500 transition-all cursor-pointer"
                        title="Rename"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                      <button
                        onClick={(e) => handleDeleteSessionClick(e, session.id)}
                        className="p-1 hover:bg-slate-200/80 dark:hover:bg-slate-700 rounded text-slate-400 hover:text-rose-500 transition-all cursor-pointer"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))
          )}
        </div>

        {/* User Card & Logout */}
        {user && (
          <div className="p-4 border-t border-slate-200 dark:border-slate-800 bg-slate-100/50 dark:bg-slate-900/60 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5 overflow-hidden">
              <div className="w-8 h-8 rounded-full bg-violet-600/10 dark:bg-violet-600/20 border border-violet-500/20 dark:border-violet-500/30 flex items-center justify-center text-violet-600 dark:text-violet-400 font-bold uppercase flex-shrink-0 text-sm">
                {user.email?.charAt(0) || 'U'}
              </div>
              <div className="flex flex-col overflow-hidden">
                <span className="text-xs font-semibold text-slate-850 dark:text-slate-200 truncate">{user.email}</span>
                <span className="text-[9px] text-slate-400 uppercase tracking-wider font-bold">Standard Account</span>
              </div>
            </div>
            <button
              onClick={handleLogoutClick}
              title="Sign Out"
              className="p-2 hover:bg-slate-200 dark:hover:bg-slate-800 rounded-lg text-slate-500 dark:text-slate-400 hover:text-rose-500 dark:hover:text-rose-400 transition-colors cursor-pointer"
            >
              <LogOut className="w-4.5 h-4.5" />
            </button>
          </div>
        )}
      </aside>

      {/* 2. MAIN CHAT AREA */}
      <section className="flex-1 h-[100dvh] max-h-[100dvh] flex flex-col overflow-hidden bg-slate-50/50 dark:bg-slate-950 relative min-w-0">
        {/* Decorative background glows */}
        <div className="absolute top-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-violet-500/5 blur-[100px] pointer-events-none" />
        <div className="absolute bottom-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-cyan-500/5 blur-[100px] pointer-events-none" />

        {/* Theme Toggle Button */}
        <button
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="absolute top-4 right-4 z-50 p-2.5 rounded-xl bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border border-slate-200/80 dark:border-slate-800/80 text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 shadow-sm transition-all cursor-pointer"
          title="Toggle Theme"
        >
          {theme === 'dark' ? <Sun className="w-4.5 h-4.5" /> : <Moon className="w-4.5 h-4.5" />}
        </button>

        {/* Top Navigation Bar (Only visible when sidebar is closed on mobile) */}
        {!sidebarOpen && (
          <header className="h-16 border-b border-slate-200/80 dark:border-slate-800/80 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md flex items-center justify-between px-4 z-10 shrink-0">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-100 transition-colors cursor-pointer"
              >
                <Menu className="w-5 h-5" />
              </button>
            </div>
          </header>
        )}

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
                  Connect securely, switch between LLM providers on the fly, and experience real-time responses.
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
            <div className="max-w-3xl mx-auto space-y-6">
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
                          {!message.content && isLoading ? (
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
          <div className="max-w-3xl mx-auto">
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
                      <div className="absolute left-0 bottom-full mb-2 w-48 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl shadow-lg z-30 p-2 animate-fade-in">
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

      {/* MODALS */}

      {/* Knowledge Base Drawer */}
      {showKnowledgeBase && (
        <div className="fixed inset-0 z-[100] flex justify-end bg-slate-900/40 backdrop-blur-sm animate-fade-in">
          <div className="w-full max-w-md h-full bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 shadow-2xl flex flex-col">
            {/* Header */}
            <div className="p-5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 bg-gradient-to-tr from-violet-600 to-cyan-500 rounded-lg flex items-center justify-center shadow-md">
                  <Database className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="font-bold text-base text-slate-800 dark:text-slate-100">My Knowledge Base</h2>
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">Documents the AI can access</span>
                </div>
              </div>
              <button
                onClick={() => setShowKnowledgeBase(false)}
                className="p-1.5 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100 cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Document List */}
            <div className="flex-1 overflow-y-auto p-4 space-y-2 custom-scrollbar">
              {loadingDocuments ? (
                <div className="flex flex-col gap-2">
                  <div className="h-14 bg-slate-100 dark:bg-slate-800/50 rounded-xl animate-pulse" />
                  <div className="h-14 bg-slate-100 dark:bg-slate-800/50 rounded-xl animate-pulse" />
                  <div className="h-14 bg-slate-100 dark:bg-slate-800/50 rounded-xl animate-pulse" />
                </div>
              ) : documents.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center px-6 py-16">
                  <FileText className="w-10 h-10 text-slate-300 dark:text-slate-700 mb-3" />
                  <p className="text-sm font-semibold text-slate-500 dark:text-slate-400">No documents yet</p>
                  <p className="text-xs text-slate-400 mt-1">Attach a PDF in chat to add it to your knowledge base.</p>
                </div>
              ) : (
                documents.map(doc => (
                  <div
                    key={doc.document_name}
                    className="group flex items-center justify-between p-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-800/40 hover:bg-slate-100 dark:hover:bg-slate-800/70 transition-all"
                  >
                    <div className="flex items-center gap-2.5 overflow-hidden">
                      <FileText className="w-4.5 h-4.5 text-violet-500 flex-shrink-0" />
                      <div className="flex flex-col overflow-hidden">
                        <span className="text-sm font-medium text-slate-800 dark:text-slate-200 truncate">{doc.document_name}</span>
                        <span className="text-[10px] text-slate-400">{doc.chunk_count} chunk{doc.chunk_count !== 1 ? 's' : ''}</span>
                      </div>
                    </div>
                    <button
                      onClick={() => setDocumentToDelete(doc.document_name)}
                      className="p-1.5 hover:bg-rose-50 dark:hover:bg-rose-500/10 rounded-lg text-slate-400 hover:text-rose-500 transition-all cursor-pointer flex-shrink-0"
                      title="Delete document"
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

      {/* Delete Document Confirmation Modal */}
      {documentToDelete && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-slate-900/40 backdrop-blur-sm animate-fade-in">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 max-w-sm w-full shadow-2xl scale-100 transition-all">
            <div className="flex flex-col items-center text-center">
              <div className="w-12 h-12 bg-rose-100 dark:bg-rose-500/20 rounded-full flex items-center justify-center mb-4">
                <AlertTriangle className="w-6 h-6 text-rose-500" />
              </div>
              <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100 mb-2">Delete Document</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
                Remove <span className="font-semibold text-slate-700 dark:text-slate-300">{documentToDelete}</span> from your knowledge base? The AI will no longer be able to reference it.
              </p>
              <div className="flex w-full gap-3">
                <button
                  onClick={() => setDocumentToDelete(null)}
                  disabled={deletingDocument}
                  className="flex-1 py-2.5 px-4 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-semibold rounded-xl transition-all cursor-pointer disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={confirmDeleteDocument}
                  disabled={deletingDocument}
                  className="flex-1 py-2.5 px-4 bg-rose-500 hover:bg-rose-600 text-white font-semibold rounded-xl transition-all shadow-md cursor-pointer disabled:opacity-50"
                >
                  {deletingDocument ? 'Deleting...' : 'Delete'}
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
