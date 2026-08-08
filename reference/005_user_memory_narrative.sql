-- ============================================================================
-- Migration: user_memory -- collapse per-fact rows into a single continuous
-- narrative profile paragraph per user (Upsert/Rewrite architecture)
-- ============================================================================
-- Supersedes 004_user_memory.sql's one-row-per-fact model. The chat service
-- now maintains exactly ONE evolving paragraph per user_id -- rewritten in
-- place on every turn that reveals a new durable fact, never appended to --
-- so this table must guarantee at most one row per user_id and support
-- upsert-by-user_id from the application.
-- ============================================================================

alter table public.user_memory
    add column if not exists narrative text not null default '',
    add column if not exists updated_at timestamp with time zone not null default timezone('utc'::text, now());

-- Collapse any existing per-fact rows (004's model) into a single narrative
-- row per user, in chronological order, before the old column is dropped.
update public.user_memory t
set narrative = sub.combined,
    updated_at = sub.latest
from (
    select user_id,
           string_agg(fact, ' ' order by created_at) as combined,
           max(created_at) as latest
    from public.user_memory
    where fact is not null
    group by user_id
) sub
where t.user_id = sub.user_id
  and t.id = (
      select id from public.user_memory u2
      where u2.user_id = sub.user_id
      order by u2.created_at desc
      limit 1
  );

-- Drop every row except the single most-recent one per user (the one just
-- populated with the merged narrative above), so the unique constraint
-- below can be added.
delete from public.user_memory t
using public.user_memory u2
where t.user_id = u2.user_id
  and t.created_at < u2.created_at;

alter table public.user_memory drop column if exists fact;

alter table public.user_memory
    add constraint user_memory_user_id_key unique (user_id);

-- Keep updated_at current on every upsert-driven UPDATE regardless of what
-- the application sends, since Postgres upsert (INSERT ... ON CONFLICT DO
-- UPDATE) does not re-apply column defaults on the UPDATE branch.
create or replace function public.set_user_memory_updated_at()
returns trigger as $$
begin
  new.updated_at = timezone('utc'::text, now());
  return new;
end;
$$ language plpgsql;

drop trigger if exists set_user_memory_updated_at on public.user_memory;
create trigger set_user_memory_updated_at
  before update on public.user_memory
  for each row execute procedure public.set_user_memory_updated_at();

-- 004 only granted select/insert/delete; upsert's ON CONFLICT DO UPDATE
-- branch needs UPDATE privilege and an UPDATE RLS policy too.
grant update on public.user_memory to authenticated;

drop policy if exists "Users can update their own memory" on public.user_memory;
create policy "Users can update their own memory"
    on public.user_memory
    for update
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);
