'use client'

import { memo, useCallback, useEffect, useRef, useState } from 'react'
import { useEditor, EditorContent, type Editor } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import { Markdown } from 'tiptap-markdown'
import { Heading1, Heading2, Bold, Italic, List, X, FileText, Download, UploadCloud, Loader2 } from 'lucide-react'
import { createClient } from '@/utils/supabase/client'
import { tiptapJsonToDocxBlob } from '@/lib/tiptap-to-docx'

interface CanvasWorkspaceProps {
  /** Raw Markdown string -- parsed into the editor's initial document on mount, and again whenever it changes to a genuinely new value (e.g. a fresh doc streamed in from the LLM). Never used to resync on every parent re-render -- see the guarded effect below. */
  content: string
  title?: string
  onClose: () => void
}

// tiptap-markdown's .d.ts declares the `Markdown` extension's storage shape
// but doesn't module-augment @tiptap/core's `Storage` interface with it, so
// `editor.storage.markdown` doesn't typecheck through the public `Editor`
// type. Narrowing once here (call sites already only run once `editor` is
// non-null) avoids sprinkling an `as` cast at every getMarkdown() call site.
function getMarkdownContent(editor: Editor): string {
  return (editor.storage as unknown as { markdown: { getMarkdown(): string } }).markdown.getMarkdown()
}

function ToolbarButton({
  onClick,
  active,
  label,
  children,
}: {
  onClick: () => void
  active: boolean
  label: string
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
        active
          ? 'bg-violet-100 text-violet-700 dark:bg-violet-500/20 dark:text-violet-300'
          : 'text-slate-500 hover:bg-slate-100 hover:text-slate-800 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100'
      }`}
    >
      {children}
    </button>
  )
}

function Toolbar({ editor }: { editor: Editor }) {
  return (
    <div className="sticky top-0 z-10 flex items-center gap-1 px-3 py-2 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800">
      <ToolbarButton
        label="Heading 1"
        active={editor.isActive('heading', { level: 1 })}
        onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
      >
        <Heading1 className="w-4 h-4" />
      </ToolbarButton>
      <ToolbarButton
        label="Heading 2"
        active={editor.isActive('heading', { level: 2 })}
        onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
      >
        <Heading2 className="w-4 h-4" />
      </ToolbarButton>
      <div className="w-px h-4 bg-slate-200 dark:bg-slate-800 mx-1" />
      <ToolbarButton
        label="Bold"
        active={editor.isActive('bold')}
        onClick={() => editor.chain().focus().toggleBold().run()}
      >
        <Bold className="w-4 h-4" />
      </ToolbarButton>
      <ToolbarButton
        label="Italic"
        active={editor.isActive('italic')}
        onClick={() => editor.chain().focus().toggleItalic().run()}
      >
        <Italic className="w-4 h-4" />
      </ToolbarButton>
      <div className="w-px h-4 bg-slate-200 dark:bg-slate-800 mx-1" />
      <ToolbarButton
        label="Bullet List"
        active={editor.isActive('bulletList')}
        onClick={() => editor.chain().focus().toggleBulletList().run()}
      >
        <List className="w-4 h-4" />
      </ToolbarButton>
    </div>
  )
}

function CanvasWorkspaceImpl({ content, title = 'Untitled Canvas Document', onClose }: CanvasWorkspaceProps) {
  // Tracks the last `content` value actually applied to the editor, so the
  // sync effect below only ever calls setContent when the *prop* genuinely
  // changes to a new document -- never in response to the user's own typing
  // (which changes the editor's internal doc, not this prop) and never on
  // an unrelated parent re-render that happens to pass an equal string.
  const appliedContentRef = useRef<string | null>(null)

  // Own Supabase client, created once -- this component's Drive export/auth
  // calls are entirely separate from the CONNECTORS-array Google Workspace
  // toggle in page.tsx (see canvas/drive_oauth.py's isolation rationale on
  // the backend); it must not read or share that flow's state.
  const supabaseRef = useRef(createClient())

  const [isExporting, setIsExporting] = useState(false)
  const [isDownloading, setIsDownloading] = useState(false)

  const editor = useEditor({
    // Next.js app router renders this component on the server first; TipTap
    // would otherwise try to render its ProseMirror view during SSR and
    // produce a hydration mismatch on mount.
    immediatelyRender: false,
    extensions: [
      StarterKit,
      Markdown.configure({
        html: false,
        transformPastedText: true,
      }),
    ],
    content,
    editorProps: {
      attributes: {
        class: 'canvas-prose focus:outline-none min-h-full px-6 py-5 text-sm leading-relaxed',
      },
    },
  })

  useEffect(() => {
    // Deliberately run-once: useEditor's `content` option already applies
    // the mount-time value, so this just records that it's been applied --
    // it must not re-run on every `content` prop change, which is exactly
    // what the sync effect below is for.
    appliedContentRef.current = content
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!editor) return
    if (content === appliedContentRef.current) return
    editor.commands.setContent(content)
    appliedContentRef.current = content
  }, [editor, content])

  useEffect(() => {
    return () => {
      editor?.destroy()
    }
  }, [editor])

  const handleDownloadLocal = useCallback(async () => {
    if (!editor || isDownloading) return
    setIsDownloading(true)
    try {
      // Convert the editor's structured ProseMirror JSON (not the Markdown
      // serialization) so headings/bold/italic become real Word styling
      // instead of literal `#`/`**` characters when opened in MS Word.
      const blob = await tiptapJsonToDocxBlob(editor.getJSON(), title)
      const objectUrl = URL.createObjectURL(blob)

      const link = document.createElement('a')
      link.href = objectUrl
      link.download = 'document.docx'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(objectUrl)
    } catch (err) {
      console.error('Canvas: failed to generate .docx download:', err)
    } finally {
      setIsDownloading(false)
    }
  }, [editor, isDownloading, title])

  const handleOpenInDrive = useCallback(async () => {
    if (!editor || isExporting) return
    setIsExporting(true)
    try {
      const markdown = getMarkdownContent(editor)
      const { data: { session } } = await supabaseRef.current.auth.getSession()
      const accessToken = session?.access_token || ''

      const response = await fetch('/api/canvas/export', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ content: markdown, title }),
      })

      // 401/403 -- an invalid/expired session, per the directive's own
      // fallback condition. 409 is what canvas/router.py's POST /api/canvas
      // /export actually returns when the session is valid but this user
      // has never completed the Drive OAuth handshake (no stored
      // 'google_drive' refresh token) -- both mean the same thing to the
      // user here ("go connect Drive"), so both send them through the
      // isolated /api/canvas/drive/authorize flow, never the CONNECTORS-
      // array Google Workspace toggle.
      if (response.status === 401 || response.status === 403 || response.status === 409) {
        window.location.href = `/api/canvas/drive/authorize?access_token=${encodeURIComponent(accessToken)}`
        return
      }

      if (!response.ok) {
        throw new Error(`Canvas export failed (HTTP ${response.status}).`)
      }

      const data: { document_id?: string; web_view_link?: string } = await response.json()
      const docUrl = data.web_view_link || (data.document_id ? `https://docs.google.com/document/d/${data.document_id}/edit` : null)
      if (docUrl) {
        window.open(docUrl, '_blank')
      }
    } catch (err) {
      console.error('Canvas: failed to export to Google Drive:', err)
    } finally {
      setIsExporting(false)
    }
  }, [editor, isExporting, title])

  return (
    <section className="flex-1 w-full h-[100dvh] max-h-[100dvh] min-w-0 flex flex-col overflow-hidden bg-white dark:bg-slate-900 border-l border-slate-200/80 dark:border-slate-800">
      <div className="flex items-center justify-between gap-2 px-4 py-3 border-b border-slate-200/80 dark:border-slate-800 flex-shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="w-4 h-4 text-violet-500 flex-shrink-0" />
          <span className="text-sm font-semibold text-slate-800 dark:text-slate-100 truncate">{title}</span>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <button
            type="button"
            onClick={handleDownloadLocal}
            disabled={!editor || isDownloading}
            title="Download as Word document (.docx)"
            aria-label="Download as Word document (.docx)"
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold rounded-lg border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isDownloading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">{isDownloading ? 'Preparing…' : 'Download Local'}</span>
          </button>
          <button
            type="button"
            onClick={handleOpenInDrive}
            disabled={!editor || isExporting}
            title="Export to Google Drive"
            aria-label="Export to Google Drive"
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold rounded-lg bg-violet-600 text-white hover:bg-violet-700 transition-colors cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {isExporting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <UploadCloud className="w-3.5 h-3.5" />}
            <span className="hidden sm:inline">{isExporting ? 'Exporting…' : 'Open in Drive'}</span>
          </button>

          <div className="w-px h-4 bg-slate-200 dark:bg-slate-800 mx-0.5" />

          <button
            type="button"
            onClick={onClose}
            title="Close canvas"
            aria-label="Close canvas"
            className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-100 hover:text-slate-800 dark:hover:bg-slate-800 dark:hover:text-slate-100 cursor-pointer flex-shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {editor && <Toolbar editor={editor} />}

      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <EditorContent editor={editor} />
      </div>
    </section>
  )
}

// Streaming chat tokens update Dashboard's `messages` state on every chunk,
// which re-renders Dashboard itself far more often than the canvas content
// actually changes. Without this memo, every one of those re-renders would
// also re-render (and diff) TipTap's ProseMirror-backed EditorContent tree,
// which is expensive and can visibly stall typing/scrolling in the editor.
// Skipping re-renders whenever `content`/`title`/`onClose` are unchanged
// keeps this panel's cost isolated from the chat panel's render rate.
const CanvasWorkspace = memo(CanvasWorkspaceImpl)
CanvasWorkspace.displayName = 'CanvasWorkspace'

export default CanvasWorkspace
