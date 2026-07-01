'use client'

import { useState, useEffect, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { createClient } from '@/utils/supabase/client'
import { Bot, Mail, Lock, Loader2, Sparkles } from 'lucide-react'

function LoginForm() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const supabase = createClient()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isSignUp, setIsSignUp] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  useEffect(() => {
    // Show error from URL parameters if present (e.g. from callback route)
    const errParam = searchParams.get('error')
    if (errParam) {
      setError(errParam)
    }
  }, [searchParams])

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(null)

    if (!email || !password) {
      setError('Please fill in all fields.')
      setLoading(false)
      return
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters.')
      setLoading(false)
      return
    }

    try {
      if (isSignUp) {
        // Sign Up Flow
        const { data, error: signUpError } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: `${window.location.origin}/auth/callback`,
          },
        })

        if (signUpError) throw signUpError

        if (data.session) {
          setSuccess('Account created! Redirecting to dashboard...')
          setTimeout(() => {
            router.refresh()
            router.push('/')
          }, 1500)
        } else {
          setSuccess('Signup successful! Check your email for a confirmation link.')
          setLoading(false)
        }
      } else {
        // Login Flow
        const { error: loginError } = await supabase.auth.signInWithPassword({
          email,
          password,
        })

        if (loginError) throw loginError

        setSuccess('Success! Redirecting...')
        router.refresh()
        setTimeout(() => {
          router.push('/')
        }, 1000)
      }
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.')
      setLoading(false)
    }
  }

  return (
    <main className="relative min-h-screen flex items-center justify-center bg-slate-950 overflow-hidden text-slate-100 font-sans">
      {/* Decorative Background Elements */}
      <div className="absolute top-[-20%] left-[-20%] w-[60%] h-[60%] rounded-full bg-violet-600/10 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-20%] w-[60%] h-[60%] rounded-full bg-cyan-600/10 blur-[120px] pointer-events-none" />

      <div className="w-full max-w-md p-8 bg-slate-900/60 backdrop-blur-xl border border-slate-800/80 rounded-2xl shadow-2xl relative z-10 mx-4">
        {/* Header */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-14 h-14 bg-gradient-to-tr from-violet-600 to-cyan-500 rounded-xl flex items-center justify-center shadow-lg shadow-violet-500/20 mb-4 animate-pulse">
            <Bot className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-slate-400">
            AetherChat
          </h1>
          <p className="text-sm text-slate-400 mt-2 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-violet-400" />
            Your Intelligent Multi-Provider Portal
          </p>
        </div>

        {/* Tab Selection */}
        <div className="grid grid-cols-2 bg-slate-950 p-1 rounded-xl mb-6 border border-slate-800/50">
          <button
            type="button"
            className={`py-2 text-sm font-semibold rounded-lg transition-all duration-300 ${
              !isSignUp
                ? 'bg-slate-900 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            onClick={() => {
              setIsSignUp(false)
              setError(null)
              setSuccess(null)
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`py-2 text-sm font-semibold rounded-lg transition-all duration-300 ${
              isSignUp
                ? 'bg-slate-900 text-white shadow-md'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            onClick={() => {
              setIsSignUp(true)
              setError(null)
              setSuccess(null)
            }}
          >
            Sign Up
          </button>
        </div>

        {/* Status Messages */}
        {error && (
          <div className="mb-4 p-3.5 bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs rounded-lg animate-fade-in font-medium">
            {error}
          </div>
        )}
        {success && (
          <div className="mb-4 p-3.5 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs rounded-lg animate-fade-in font-medium">
            {success}
          </div>
        )}

        {/* Authentication Form */}
        <form onSubmit={handleAuth} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
              Email Address
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Mail className="w-5 h-5 text-slate-500" />
              </span>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full pl-10 pr-4 py-2.5 bg-slate-950 border border-slate-800 focus:border-violet-500 rounded-xl focus:outline-none focus:ring-1 focus:ring-violet-500 text-slate-100 placeholder-slate-500 text-sm transition-all"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
              Password
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="w-5 h-5 text-slate-500" />
              </span>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-10 pr-4 py-2.5 bg-slate-950 border border-slate-800 focus:border-violet-500 rounded-xl focus:outline-none focus:ring-1 focus:ring-violet-500 text-slate-100 placeholder-slate-500 text-sm transition-all"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-2 py-3 bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 text-white font-semibold rounded-xl transition-all shadow-lg shadow-violet-500/10 hover:shadow-violet-500/20 active:scale-[0.99] flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Processing...
              </>
            ) : isSignUp ? (
              'Create Account'
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        {/* Disclaimer / Footer */}
        <p className="text-center text-[11px] text-slate-500 mt-6 leading-relaxed">
          {isSignUp
            ? 'By signing up, you agree to our Terms of Service and Privacy Policy.'
            : 'Secure authentication powered by Supabase Auth with server-side session management.'}
        </p>
      </div>
    </main>
  )
}

export default function LoginPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-400">
        <div className="flex flex-col items-center gap-2">
          <div className="w-8 h-8 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
          <span className="text-xs">Loading authentication portal...</span>
        </div>
      </div>
    }>
      <LoginForm />
    </Suspense>
  )
}
