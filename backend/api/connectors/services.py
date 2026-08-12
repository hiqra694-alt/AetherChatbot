import logging

from supabase import Client

logger = logging.getLogger(__name__)


async def store_oauth_refresh_token(supabase: Client, user_id: str, provider: str, refresh_token: str) -> bool:
    """
    Upserts `user_id`'s refresh_token for `provider` into user_oauth_tokens
    (one row per user_id+provider -- see reference/007_user_oauth_tokens.sql).
    Called once, right after a successful linkIdentity redirect, so a later
    server-side refresh (mcp_integration.mcp_manager
    ._refresh_google_access_token) can mint a fresh access token even once
    the browser-cached one has aged out. `supabase` is the caller's own
    JWT-scoped client (see get_authenticated_supabase), so RLS restricts
    this write to the authenticated user's own row regardless of what
    `user_id` claims -- it's only ever passed for logging/clarity here.
    """
    row = {"user_id": user_id, "provider": provider, "refresh_token": refresh_token}
    supabase.table("user_oauth_tokens").upsert(row, on_conflict="user_id,provider").execute()
    return True


async def get_stored_refresh_token_providers(supabase: Client, user_id: str) -> set[str]:
    """
    Every provider `user_id` currently has a stored refresh_token for --
    backs GET /api/connectors/status, which the frontend uses to tell "this
    provider identity is linked" (user.identities, always true once linked)
    apart from "this provider actually has a usable stored token"
    (user_oauth_tokens; only true once linkIdentity's offline-access flow
    has completed for it -- see ConnectorStatusResponse's docstring).
    """
    res = supabase.table("user_oauth_tokens").select("provider").eq("user_id", user_id).execute()
    return {row["provider"] for row in (res.data or [])}


async def delete_oauth_refresh_token(supabase: Client, user_id: str, provider: str) -> bool:
    """
    Deletes `user_id`'s stored refresh_token row for `provider`, if any.
    Backs DELETE /api/connectors/disconnect: once removed, the server-side
    token-refresh fallback (mcp_integration.mcp_manager
    ._refresh_google_access_token) can no longer mint access tokens on this
    user's behalf for that provider. Returns False (not an error) when no
    row existed -- a connector can be disconnected even if it never
    captured an offline refresh_token to begin with (e.g. only ever used
    with a live in-browser token).
    """
    res = (
        supabase.table("user_oauth_tokens")
        .delete()
        .eq("user_id", user_id)
        .eq("provider", provider)
        .execute()
    )
    return len(res.data or []) > 0
