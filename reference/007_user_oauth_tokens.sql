-- ============================================================================
-- Migration: user_oauth_tokens -- long-lived OAuth refresh tokens per user
-- ============================================================================
-- Backs the server-side token-refresh fallback in
-- mcp_integration/mcp_manager.py (create_workspace_session): Supabase's own
-- session.provider_token is only ever populated right after an OAuth
-- exchange and is dropped on subsequent token refreshes, so it can't be
-- relied on to still be present in the browser once the page reloads. This
-- table stores the durable refresh_token instead (captured client-side once,
-- via POST /api/connectors/store-token -- see api/connectors/routers.py),
-- letting the backend mint a fresh Google access token on demand instead of
-- silently offering zero Workspace tools whenever the cached one has aged
-- out. One row per (user_id, provider); 'google' is the only provider this
-- refresh path currently uses, but the column stays generic for GitHub or
-- future connectors.
--
-- Sensitive by nature -- protected by RLS same as every other per-user table
-- here (tasks, user_memory, document_chunks), not application-level
-- encrypted. Only ever read/written through a caller's own JWT-scoped
-- client, so a row is never visible to anyone but its owner (or a
-- service-role client, granted below for any future background job that
-- needs cross-user access, mirroring core/scheduler.py's use of tasks).
-- ============================================================================

create table if not exists public.user_oauth_tokens (
    user_id       uuid not null references auth.users(id) on delete cascade,
    provider      text not null,
    refresh_token text not null,
    updated_at    timestamp with time zone default timezone('utc'::text, now()),
    primary key (user_id, provider)
);

alter table public.user_oauth_tokens enable row level security;
alter table public.user_oauth_tokens force row level security;

create policy "Users can manage their own oauth tokens"
    on public.user_oauth_tokens
    for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

-- Keeps updated_at current on every upsert-driven UPDATE regardless of what
-- the application sends, since Postgres upsert (INSERT ... ON CONFLICT DO
-- UPDATE) does not re-apply column defaults on the UPDATE branch -- same
-- pattern as 005_user_memory_narrative.sql's set_user_memory_updated_at.
create or replace function public.set_user_oauth_tokens_updated_at()
returns trigger as $$
begin
  new.updated_at = timezone('utc'::text, now());
  return new;
end;
$$ language plpgsql;

drop trigger if exists set_user_oauth_tokens_updated_at on public.user_oauth_tokens;
create trigger set_user_oauth_tokens_updated_at
  before update on public.user_oauth_tokens
  for each row execute procedure public.set_user_oauth_tokens_updated_at();

grant usage on schema public to authenticated;
grant select, insert, update, delete on public.user_oauth_tokens to authenticated, service_role;
