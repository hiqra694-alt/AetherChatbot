-- ============================================================================
-- Fix: 42501 "permission denied for table document_chunks"
--
-- Root cause: 001_pgvector_document_chunks.sql created RLS policies scoped
-- `to authenticated`, but RLS policies only restrict which ROWS a role may
-- touch — they never substitute for the base Postgres GRANT. Without an
-- explicit GRANT, every request as the `authenticated` role is rejected at
-- the privilege-check stage, before RLS is ever evaluated. Tables created
-- through the Supabase Studio table editor get this grant automatically;
-- this table was created via raw SQL, so it didn't.
--
-- Run this once against the already-provisioned project. It's idempotent —
-- safe to run again.
-- ============================================================================
grant usage on schema public to authenticated;
grant select, insert on public.document_chunks to authenticated;
