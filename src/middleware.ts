import { NextResponse, type NextRequest } from 'next/server'
import { updateSession } from '@/utils/supabase/middleware'

export async function middleware(request: NextRequest) {
  const url = request.nextUrl
  if (url.pathname === '/api/chat') {
    let backendBaseUrl = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.BACKEND_API_URL || 'http://127.0.0.1:8000'
    if (!/^https?:\/\//i.test(backendBaseUrl)) {
      backendBaseUrl = `https://${backendBaseUrl}`
    }
    backendBaseUrl = backendBaseUrl.replace(/\/$/, '')
    const targetUrl = `${backendBaseUrl}/api/chat`
    
    return NextResponse.rewrite(new URL(targetUrl))
  }

  return await updateSession(request)
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - all images/files with extensions
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
}
