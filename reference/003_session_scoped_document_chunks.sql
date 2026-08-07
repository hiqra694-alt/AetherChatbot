-- ============================================================================
-- Migration: scope document_chunks to a chat session (Context Isolation)
-- ============================================================================
-- Documents were previously global to a user: any chunk they'd ever uploaded
-- was eligible for retrieval in every chat session. This migration adds a
-- session_id boundary so a document uploaded in one chat is only ever
-- retrievable from that same chat, matching per-session context isolation.
--
-- Run this against a database that already has 001_pgvector_document_chunks.sql
-- applied. Existing rows have no session to attach to -- either backfill them
-- with a real chat_sessions.id per row, or truncate public.document_chunks,
-- before running the final `alter column ... set not null` below.
-- ============================================================================

alter table public.document_chunks
    add column if not exists session_id uuid references public.chat_sessions(id) on delete cascade;

-- Uncomment once existing rows are backfilled/cleared (see note above):
-- alter table public.document_chunks alter column session_id set not null;

-- Composite index: retrieval and listing always filter by (user_id, session_id)
-- together now, so this replaces the standalone user_id index as the hot path.
drop index if exists public.document_chunks_user_id_idx;
create index if not exists document_chunks_user_session_idx
    on public.document_chunks (user_id, session_id);

-- ============================================================================
-- match_document_chunks — add filter_session_id
-- ============================================================================
-- Drop the old 5-arg overload first -- `create or replace` cannot widen a
-- function's argument list in place; without the drop, the old signature
-- would linger as a second overload after this migration runs.
drop function if exists public.match_document_chunks(vector, float, int, uuid, text);

create or replace function public.match_document_chunks (
    query_embedding vector(512),
    match_threshold float,
    match_count int,
    filter_user_id uuid,
    filter_session_id uuid default null,
    filter_document_name text default null
)
returns table (
    document_name text,
    chunk_text text,
    similarity float
)
language sql
stable
security definer
set search_path = public
as $$
    select
        dc.document_name,
        dc.chunk_text,
        1 - (dc.embedding <=> query_embedding) as similarity
    from public.document_chunks dc
    where dc.user_id = filter_user_id
      and (filter_session_id is null or dc.session_id = filter_session_id)
      and (filter_document_name is null or dc.document_name = filter_document_name)
      and 1 - (dc.embedding <=> query_embedding) > match_threshold
    order by dc.embedding <=> query_embedding asc
    limit match_count;
$$;

revoke all on function public.match_document_chunks(vector, float, int, uuid, uuid, text) from public;
grant execute on function public.match_document_chunks(vector, float, int, uuid, uuid, text) to authenticated, service_role;
