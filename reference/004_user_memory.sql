-- ============================================================================
-- Migration: user_memory -- persistent, cross-chat facts about the user
-- ============================================================================
-- Gemini-style global memory: facts the user states about themselves (e.g.
-- "I work at ABC company") are extracted by a background task after a chat
-- turn and stored here, keyed only by user_id (not session_id) so they are
-- available to every chat, unlike the per-session document_chunks table.
-- ============================================================================

create table if not exists public.user_memory (
    id         uuid primary key default gen_random_uuid(),
    user_id    uuid not null references auth.users(id) on delete cascade,
    fact       text not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

create index if not exists user_memory_user_id_idx
    on public.user_memory (user_id);

alter table public.user_memory enable row level security;
alter table public.user_memory force row level security;

create policy "Users can view their own memory"
    on public.user_memory
    for select
    to authenticated
    using (auth.uid() = user_id);

create policy "Users can insert their own memory"
    on public.user_memory
    for insert
    to authenticated
    with check (auth.uid() = user_id);

create policy "Users can delete their own memory"
    on public.user_memory
    for delete
    to authenticated
    using (auth.uid() = user_id);

grant usage on schema public to authenticated;
grant select, insert, delete on public.user_memory to authenticated;
