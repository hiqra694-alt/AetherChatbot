import { NextResponse } from 'next/server'
import { createClient } from '@/utils/supabase/server'

export const dynamic = 'force-dynamic'

export async function POST(request: Request) {
  try {
    const { provider, sessionId } = await request.json()

    if (!sessionId) {
      return NextResponse.json(
        { error: 'Session ID is required to process and store chat history.' },
        { status: 400 }
      )
    }

    // 1. Fetch message history from Supabase to provide conversation context
    const supabase = await createClient()
    const { data: dbMessages, error: dbError } = await supabase
      .from('messages')
      .select('role, content')
      .eq('session_id', sessionId)
      .order('created_at', { ascending: true })

    if (dbError) {
      console.error('Database error fetching messages:', dbError.message)
      return NextResponse.json(
        { error: `Failed to fetch conversation history: ${dbError.message}` },
        { status: 500 }
      )
    }

    const messages = dbMessages?.map(m => ({
      role: m.role,
      content: m.content
    })) || []

    // 2. Map frontend provider selection to active free models on OpenRouter
    // We choose highly stable models first, and fall back to the general free router if rate-limited.
    let modelId = 'meta-llama/llama-3.3-70b-instruct:free'
    if (provider === 'gemini') {
      modelId = 'google/gemma-2-9b-it:free' // Gemma 2 9B is highly stable
    } else if (provider === 'openai') {
      modelId = 'meta-llama/llama-3.3-70b-instruct:free'
    } else if (provider === 'claude') {
      modelId = 'meta-llama/llama-3.2-3b-instruct:free' // Llama 3.2 3B has very high limits
    }

    // 3. Initiate request to OpenRouter Chat Completions endpoint with automatic fallback
    const openRouterApiKey = process.env.OPENROUTER_API_KEY
    if (!openRouterApiKey) {
      return NextResponse.json(
        { error: 'OpenRouter API key is not configured on the server.' },
        { status: 500 }
      )
    }

    const modelsToTry = [modelId, 'openrouter/free']
    let openRouterResponse: Response | null = null
    let lastErrorText = ''

    for (const model of modelsToTry) {
      try {
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${openRouterApiKey}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/google/antigravity',
            'X-Title': 'AetherChat Portal',
          },
          body: JSON.stringify({
            model: model,
            messages: messages,
            stream: true
          })
        })

        if (response.ok) {
          openRouterResponse = response
          break
        } else {
          lastErrorText = await response.text()
          console.warn(`OpenRouter model ${model} failed:`, lastErrorText)
        }
      } catch (fetchErr: any) {
        lastErrorText = fetchErr.message || 'Fetch failed'
        console.warn(`OpenRouter connection for ${model} failed:`, fetchErr)
      }
    }

    if (!openRouterResponse) {
      return NextResponse.json(
        { error: `OpenRouter API error: All fallback routes failed. Last error: ${lastErrorText}` },
        { status: 500 }
      )
    }

    const encoder = new TextEncoder()
    const decoder = new TextDecoder()
    const reader = openRouterResponse.body?.getReader()

    if (!reader) {
      return NextResponse.json(
        { error: 'OpenRouter stream is not readable.' },
        { status: 500 }
      )
    }

    // 4. Create custom ReadableStream to forward chunked SSE data
    const stream = new ReadableStream({
      async start(controller) {
        let fullResponseText = ''
        let buffer = ''

        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''

            for (const line of lines) {
              const cleanedLine = line.trim()
              if (!cleanedLine) continue

              if (cleanedLine === 'data: [DONE]') {
                controller.enqueue(encoder.encode('data: [DONE]\n\n'))
                continue
              }

              if (cleanedLine.startsWith('data: ')) {
                const dataStr = cleanedLine.slice(6)
                try {
                  const parsed = JSON.parse(dataStr)
                  const content = parsed.choices?.[0]?.delta?.content || ''
                  if (content) {
                    fullResponseText += content
                    const ssePacket = `data: ${JSON.stringify({ content })}\n\n`
                    controller.enqueue(encoder.encode(ssePacket))
                  }
                } catch (e) {
                  // Ignore parse error on partial SSE chunk
                }
              }
            }
          }
        } catch (streamErr: any) {
          console.error('Streaming connection error:', streamErr)
        } finally {
          controller.close()

          // 5. Save the complete message content to messages table
          if (fullResponseText.trim()) {
            try {
              const { error: insertError } = await supabase.from('messages').insert({
                session_id: sessionId,
                role: 'assistant',
                content: fullResponseText,
                provider_used: provider,
              })
              if (insertError) {
                console.error('Error inserting assistant message:', insertError.message)
              }
            } catch (dbInsertErr) {
              console.error('Exception inserting assistant message:', dbInsertErr)
            }
          }
        }
      },
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
      },
    })
  } catch (err: any) {
    console.error('Internal handler error in /api/chat:', err)
    return NextResponse.json(
      { error: err.message || 'Internal Server Error' },
      { status: 500 }
    )
  }
}
