import { NextResponse } from 'next/server'
import { createClient } from '@/utils/supabase/server'

export async function GET(request: Request) {
  const { searchParams, origin } = new URL(request.url)
  const code = searchParams.get('code')
  // if "next" is in param, use it as the redirect URL
  const next = searchParams.get('next') ?? '/'

  if (code) {
    const supabase = await createClient()
    const { error } = await supabase.auth.exchangeCodeForSession(code)
    if (!error) {
      return NextResponse.redirect(`${origin}${next}`)
    }
  }

  // No code was ever issued (e.g. supabase.auth.linkIdentity() rejects an
  // identity already linked to the current user before minting one -- see
  // the `identity_already_exists` handling in src/app/page.tsx's mount
  // effect, which expects to read error/error_code/error_description off
  // its own URL) or the exchange above failed. Either way, forward
  // whatever Supabase attached to this redirect on to `next` instead of
  // always swallowing it behind a generic /login message -- an
  // already-authenticated user landing back on `next` (e.g. a connector
  // link attempt) needs the real error/error_code to tell an expected
  // outcome apart from a genuine failure, and an unauthenticated one gets
  // bounced to /login with these same params intact anyway (see
  // src/utils/supabase/middleware.ts's route protection, which preserves
  // query params on that redirect).
  const forwardParams = new URLSearchParams()
  for (const key of ['error', 'error_code', 'error_description']) {
    const value = searchParams.get(key)
    if (value) forwardParams.set(key, value)
  }
  if (forwardParams.size > 0) {
    return NextResponse.redirect(`${origin}${next}?${forwardParams.toString()}`)
  }

  // return the user to an error page or home
  return NextResponse.redirect(`${origin}/login?error=Could not authenticate user`)
}
