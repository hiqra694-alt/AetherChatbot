'use client'

import { useState, useEffect, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { createClient } from '@/utils/supabase/client'
import { Mail, Lock, Loader2, Sparkles } from 'lucide-react'
import AetherLogo from '@/components/AetherLogo'

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
    const errParam = searchParams.get('error')
    if (errParam) {
      setError(errParam)
    }
  }, [searchParams])

  const handleGoogleLogin = async () => {
    try {
      const { error: googleError } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/auth/callback`,
        },
      })
      if (googleError) throw googleError
    } catch (err: any) {
      setError(err.message || 'Failed to initialize Google login.')
    }
  }

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
    <main className="relative min-h-screen flex items-center justify-center bg-slate-50 overflow-hidden text-slate-800 font-sans">
      {/* Decorative Background Elements */}
      <div className="absolute top-[-20%] left-[-20%] w-[60%] h-[60%] rounded-full bg-violet-500/5 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-20%] w-[60%] h-[60%] rounded-full bg-cyan-500/5 blur-[120px] pointer-events-none" />

      <div className="w-full max-w-md p-8 bg-white/90 backdrop-blur-xl border border-slate-200/80 rounded-2xl shadow-xl relative z-10 mx-4">
        {/* Header */}
        <div className="flex flex-col items-center mb-8">
          <AetherLogo size={56} className="mb-4" />
          <h1 className="text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-slate-900 via-slate-700 to-slate-500">
            Aether
          </h1>
          <p className="text-sm text-slate-500 mt-2 flex items-center gap-1.5 font-medium">
            <Sparkles className="w-3.5 h-3.5 text-violet-500" />
            Your Intelligent Multi-Provider Portal
          </p>
        </div>

        {/* Tab Selection */}
        <div className="grid grid-cols-2 bg-slate-100 p-1 rounded-xl mb-6 border border-slate-200/60">
          <button
            type="button"
            className={`py-2 text-sm font-semibold rounded-lg transition-all duration-300 ${
              !isSignUp
                ? 'bg-white text-slate-800 shadow-sm border border-slate-200/30'
                : 'text-slate-500 hover:text-slate-800'
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
                ? 'bg-white text-slate-800 shadow-sm border border-slate-200/30'
                : 'text-slate-500 hover:text-slate-800'
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
          <div className="mb-4 p-3.5 bg-rose-50 border border-rose-200 text-rose-600 text-xs rounded-lg animate-fade-in font-semibold">
            {error}
          </div>
        )}
        {success && (
          <div className="mb-4 p-3.5 bg-emerald-50 border border-emerald-200 text-emerald-600 text-xs rounded-lg animate-fade-in font-semibold">
            {success}
          </div>
        )}

        {/* Authentication Form */}
        <form onSubmit={handleAuth} className="space-y-4">
          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-1.5">
              Email Address
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Mail className="w-5 h-5 text-slate-400" />
              </span>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full pl-10 pr-4 py-2.5 bg-white border border-slate-200 focus:border-violet-500 rounded-xl focus:outline-none focus:ring-1 focus:ring-violet-500 text-slate-800 placeholder-slate-400 text-sm transition-all"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-1.5">
              Password
            </label>
            <div className="relative">
              <span className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="w-5 h-5 text-slate-400" />
              </span>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-10 pr-4 py-2.5 bg-white border border-slate-200 focus:border-violet-500 rounded-xl focus:outline-none focus:ring-1 focus:ring-violet-500 text-slate-800 placeholder-slate-400 text-sm transition-all"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-2 py-3 bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 text-white font-semibold rounded-xl transition-all shadow-md active:scale-[0.99] flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100"
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

        <div className="mt-6 mb-4 flex items-center justify-center">
          <div className="border-t border-slate-200/80 w-full"></div>
          <span className="px-3 text-xs text-slate-400 font-bold uppercase tracking-widest bg-white/90">OR</span>
          <div className="border-t border-slate-200/80 w-full"></div>
        </div>

        <button
          onClick={handleGoogleLogin}
          type="button"
          className="w-full py-3 bg-white hover:bg-slate-50 border border-slate-200/80 text-slate-700 font-semibold rounded-xl transition-all shadow-sm active:scale-[0.99] flex items-center justify-center gap-3 cursor-pointer"
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path
              fill="currentColor"
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
            />
            <path
              fill="currentColor"
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
            />
            <path
              fill="currentColor"
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
            />
            <path
              fill="currentColor"
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
            />
          </svg>
          Continue with Google
        </button>

        {/* Disclaimer / Footer */}
        <p className="text-center text-[11px] text-slate-400 mt-6 leading-relaxed font-medium">
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
      <div className="min-h-screen flex items-center justify-center bg-slate-50 text-slate-500">
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
