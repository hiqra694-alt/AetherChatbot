-- ============================================================================
-- Migration: tasks -- native task/reminder tracking (Phase 5)
-- ============================================================================
-- Backs api/tasks/services.py's create_task/list_tasks/complete_task (exposed
-- to the LLM as native tools in api/chat/tools.py) and the background
-- scheduler (core/scheduler.py), which polls for pending tasks whose due_at
-- has passed. Per-request access goes through a user-JWT-scoped client (RLS
-- restricts every row to its own user_id, same as user_memory/document_chunks);
-- the scheduler instead uses a service-role client (SUPABASE_SERVICE_ROLE_KEY),
-- which bypasses RLS by default in Supabase, to see due tasks across every user.
-- ============================================================================

create table if not exists public.tasks (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references auth.users(id) on delete cascade,
    title       text not null,
    description text,
    due_at      timestamp with time zone,
    status      text not null default 'pending' check (status in ('pending', 'completed')),
    created_at  timestamp with time zone default timezone('utc'::text, now()) not null,
    notified_at timestamp with time zone
);

create index if not exists tasks_user_id_idx
    on public.tasks (user_id);

-- Used by the scheduler's due-task poll: status = 'pending' and
-- notified_at is null and due_at <= now().
create index if not exists tasks_due_poll_idx
    on public.tasks (status, notified_at, due_at)
    where status = 'pending' and notified_at is null;

alter table public.tasks enable row level security;
alter table public.tasks force row level security;

create policy "Users can view their own tasks"
    on public.tasks
    for select
    to authenticated
    using (auth.uid() = user_id);

create policy "Users can insert their own tasks"
    on public.tasks
    for insert
    to authenticated
    with check (auth.uid() = user_id);

create policy "Users can update their own tasks"
    on public.tasks
    for update
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy "Users can delete their own tasks"
    on public.tasks
    for delete
    to authenticated
    using (auth.uid() = user_id);

grant usage on schema public to authenticated;
grant select, insert, update, delete on public.tasks to authenticated;
