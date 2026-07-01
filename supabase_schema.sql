-- Supabase Database Schema and RLS Policies Migration Script
-- Copy and run this script in your Supabase SQL Editor (https://supabase.com/dashboard/project/_/sql)

-- 1. Create PROFILES Table (linked to auth.users)
create table if not exists public.profiles (
  id uuid references auth.users on delete cascade primary key,
  email text,
  updated_at timestamp with time zone,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable Row Level Security
alter table public.profiles enable row level security;

-- Profiles Policies
create policy "Users can view own profile" on public.profiles
  for select using (auth.uid() = id);

create policy "Users can update own profile" on public.profiles
  for update using (auth.uid() = id);


-- 2. Create CHAT_SESSIONS Table
create table if not exists public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.profiles(id) on delete cascade not null default auth.uid(),
  title text not null,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable Row Level Security
alter table public.chat_sessions enable row level security;

-- Chat Sessions Policies
create policy "Users can view own chat sessions" on public.chat_sessions
  for select using (auth.uid() = user_id);

create policy "Users can create own chat sessions" on public.chat_sessions
  for insert with check (auth.uid() = user_id);

create policy "Users can update own chat sessions" on public.chat_sessions
  for update using (auth.uid() = user_id);

create policy "Users can delete own chat sessions" on public.chat_sessions
  for delete using (auth.uid() = user_id);


-- 3. Create MESSAGES Table
create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references public.chat_sessions(id) on delete cascade not null,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  provider_used text,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable Row Level Security
alter table public.messages enable row level security;

-- Messages Policies
create policy "Users can view messages in own sessions" on public.messages
  for select using (
    exists (
      select 1 from public.chat_sessions
      where public.chat_sessions.id = public.messages.session_id
        and public.chat_sessions.user_id = auth.uid()
    )
  );

create policy "Users can insert messages in own sessions" on public.messages
  for insert with check (
    exists (
      select 1 from public.chat_sessions
      where public.chat_sessions.id = public.messages.session_id
        and public.chat_sessions.user_id = auth.uid()
    )
  );

create policy "Users can delete messages in own sessions" on public.messages
  for delete using (
    exists (
      select 1 from public.chat_sessions
      where public.chat_sessions.id = public.messages.session_id
        and public.chat_sessions.user_id = auth.uid()
    )
  );


-- 4. PROFILE SYNCHRONIZATION TRIGGER (from auth.users to public.profiles)
create or replace function public.handle_new_user()
returns trigger as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email);
  return new;
end;
$$ language plpgsql security definer;

-- Drop trigger if it exists, then create
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();
