-- ============================================================================
-- Migration: pgvector storage + retrieval for multi-tenant RAG pipeline
-- Embeddings: voyage-3-lite (512 dimensions)
-- ============================================================================

-- 1. Enable the pgvector extension (no-op if already enabled)
create extension if not exists vector;

-- ============================================================================
-- 2. document_chunks table
-- ============================================================================
create table if not exists public.document_chunks (
    id            uuid primary key default gen_random_uuid(),

    -- Tenancy boundary: every chunk belongs to exactly one auth.users row.
    -- ON DELETE CASCADE keeps orphaned chunks from surviving a user deletion.
    user_id       uuid not null references auth.users(id) on delete cascade,

    document_name text not null,
    chunk_text    text not null,
    embedding     vector(512) not null,
    created_at    timestamp not null default now()
);

-- Speeds up "give me all chunks for this user" filtering used by both
-- the RLS policies below and the RPC function's WHERE clause.
create index if not exists document_chunks_user_id_idx
    on public.document_chunks (user_id);

-- ANN index for cosine similarity search. HNSW gives better recall/latency
-- than IVFFlat and needs no training step, so it's safe to create up front
-- even on an empty table.
create index if not exists document_chunks_embedding_idx
    on public.document_chunks
    using hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- 3. Row Level Security — strict per-tenant isolation
-- ============================================================================
alter table public.document_chunks enable row level security;

-- Belt-and-braces: even the table owner/service role must opt in explicitly
-- rather than silently bypassing RLS.
alter table public.document_chunks force row level security;

-- SELECT: a user may only read rows whose user_id matches their own JWT
-- subject claim (auth.uid()). Rows belonging to other tenants are invisible,
-- not just unwritable.
-- Dropped first so this script can be re-run safely: unlike CREATE TABLE/INDEX,
-- Postgres has no "CREATE POLICY IF NOT EXISTS", so re-running the bare
-- CREATE POLICY on a database where it already exists throws 42710 ("policy
-- already exists") and rolls back the entire script transaction — including
-- the GRANT statements below — which is why nothing appeared to "save".
drop policy if exists "Users can view their own document chunks" on public.document_chunks;
create policy "Users can view their own document chunks"
    on public.document_chunks
    for select
    to authenticated
    using (auth.uid() = user_id);

-- INSERT: a user may only create rows tagged with their own user_id.
-- The WITH CHECK clause is evaluated against the row being inserted, so a
-- client cannot smuggle in another tenant's user_id even if it fabricates
-- one in the payload.
drop policy if exists "Users can insert their own document chunks" on public.document_chunks;
create policy "Users can insert their own document chunks"
    on public.document_chunks
    for insert
    to authenticated
    with check (auth.uid() = user_id);

-- Note: no UPDATE/DELETE policies are defined per the requirements above,
-- so RLS denies both by default (rows are append-only / read-only for
-- authenticated clients).

-- RLS policies only restrict which ROWS a role can touch — they do nothing
-- until the role already holds the base table privilege. Without this grant,
-- every request as `authenticated` fails with "permission denied for table
-- document_chunks" (Postgres error 42501) before RLS is ever evaluated.
grant usage on schema public to authenticated;
grant select, insert on public.document_chunks to authenticated;

-- ============================================================================
-- 4. match_document_chunks — RPC for similarity search
-- ============================================================================
-- Called from FastAPI via `supabase.rpc("match_document_chunks", {...})`.
--
-- SECURITY DEFINER + explicit p_user_id filtering: this function is intended
-- to be invoked with the service role (bypassing the caller's RLS context),
-- so the p_user_id filter below is what enforces tenant isolation — it is
-- NOT optional and must always be supplied by the backend from a verified
-- JWT, never from unvalidated client input.
create or replace function public.match_document_chunks (
    query_embedding vector(512),
    match_threshold float,
    match_count int,
    p_user_id uuid
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
        -- cosine distance (<=>) ranges [0, 2]; convert to a similarity
        -- score in [-1, 1] so callers can reason about it as "higher is better".
        1 - (dc.embedding <=> query_embedding) as similarity
    from public.document_chunks dc
    where dc.user_id = p_user_id
      and 1 - (dc.embedding <=> query_embedding) > match_threshold
    order by dc.embedding <=> query_embedding asc
    limit match_count;
$$;

-- Restrict execution to authenticated/service roles only.
revoke all on function public.match_document_chunks(vector, float, int, uuid) from public;
grant execute on function public.match_document_chunks(vector, float, int, uuid) to authenticated, service_role;
