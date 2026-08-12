"""
Google-managed Gmail MCP connector (Phase 1) -- fully isolated from every
other connector in this package.

Deliberately duplicates rather than reuses machinery from mcp_manager.py
(its own Google-token-refresh helper, its own signed-state scheme, its own
service-role client builder) even though some of it looks similar to
_refresh_google_access_token / create_workspace_session. That's intentional,
not an oversight: this connector authenticates against a brand-new,
separate OAuth client (GMAIL_MCP_CLIENT_ID/SECRET/REDIRECT_URI) and stores
its token under a brand-new, distinct user_oauth_tokens provider row
('google_gmail_mcp') -- see GMAIL_MCP_PROVIDER below -- so that linking or
refreshing it can never read, overwrite, or otherwise interact with the
existing 'google' provider row create_workspace_session's refresh fallback
depends on. Keeping the code paths fully separate is what guarantees that;
sharing a helper would risk a future edit to one silently affecting both.

Talks to Google's managed Gmail MCP server (GMAIL_MCP_SERVER_URL) over
Streamable HTTP, not the SSE transport every other connector in
mcp_manager.py uses -- see mcp_manager.MCPServerConnection's `transport`
param and MCPClientManager.create_gmail_mcp_session.
"""

import base64
import hashlib
import hmac
import logging
import time
from typing import Optional
from urllib.parse import urlencode

import httpx
from supabase import Client, create_client

from core.config import get_settings

logger = logging.getLogger(__name__)

# Google's managed Gmail MCP server -- a fixed, Google-operated endpoint,
# not a per-deployment config value like GITHUB_MCP_SERVER_URL/
# GOOGLE_WORKSPACE_MCP_URL are, so it's kept as a constant here rather than
# added to Settings.
GMAIL_MCP_SERVER_URL = "https://gmailmcp.googleapis.com/mcp/v1"

# Distinct user_oauth_tokens.provider value for this connector -- see the
# module docstring for why this must never collide with the existing
# 'google' row.
GMAIL_MCP_PROVIDER = "google_gmail_mcp"

# Broad enough for the MCP server's Gmail tools (read, send, modify labels)
# without granting full account access (gmail.modify stops short of
# https://mail.google.com/, which also allows permanent deletion).
GMAIL_MCP_SCOPE = "https://www.googleapis.com/auth/gmail.modify"

GOOGLE_OAUTH_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"

# How long a signed `state` value (see sign_state/verify_state) stays valid
# for -- comfortably longer than any human takes to complete the Google
# consent screen, short enough that a leaked/logged state can't be replayed
# long after the fact.
STATE_TTL_SECONDS = 600

_HTTP_TIMEOUT_SECONDS = 10.0


def _state_signing_key() -> bytes:
    return get_settings().gmail_mcp_client_secret.encode()


def sign_state(user_id: str) -> str:
    """
    Packs `user_id` and an expiry into a signed, URL-safe token to pass as
    the OAuth `state` param -- this is how gmail_mcp_callback recovers which
    user an incoming Google redirect belongs to, since that redirect is a
    plain browser GET with no Authorization header at all. Signed (HMAC-
    SHA256, keyed on GMAIL_MCP_CLIENT_SECRET) rather than stored
    server-side, so no session store is needed and it survives an app
    restart between authorize and callback.
    """
    expiry = int(time.time()) + STATE_TTL_SECONDS
    payload = f"{user_id}:{expiry}"
    sig = hmac.new(_state_signing_key(), payload.encode(), hashlib.sha256).hexdigest()
    raw = f"{payload}:{sig}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def verify_state(state: str) -> str:
    """
    Inverse of sign_state: returns the embedded user_id if `state` carries a
    valid, unexpired signature, otherwise raises ValueError. Never trusts an
    unsigned or tampered state -- verifying it is what stands in for "this
    callback request is authenticated" in a flow that has no session/JWT to
    check.
    """
    try:
        padded = state + "=" * (-len(state) % 4)
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        user_id, expiry_str, sig = raw.split(":")
    except Exception as exc:
        raise ValueError("Malformed OAuth state.") from exc

    expected_sig = hmac.new(_state_signing_key(), f"{user_id}:{expiry_str}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        raise ValueError("OAuth state signature mismatch.")
    if int(expiry_str) < time.time():
        raise ValueError("OAuth state has expired.")

    return user_id


async def resolve_user_id_from_access_token(access_token: str) -> str:
    """
    Verifies `access_token` (the caller's own live Supabase session token,
    sent as a query param since /authorize is a full-page redirect that
    can't carry a custom Authorization header) against Supabase Auth and
    returns the resolved user id, or raises ValueError.

    Purely read-only against Supabase Auth -- identical in spirit to
    api/connectors/routers.py's get_authenticated_supabase, just accepting
    the token via query param instead of a header. Never modifies, revokes,
    or otherwise touches the session itself.
    """
    settings = get_settings()
    supabase = create_client(settings.supabase_url, settings.supabase_anon_key)
    try:
        user_res = supabase.auth.get_user(access_token)
    except Exception as exc:
        raise ValueError("Invalid or expired access token.") from exc

    if not user_res or not user_res.user:
        raise ValueError("Invalid or expired access token.")

    return user_res.user.id


def build_google_authorize_url(state: str) -> str:
    """The Google OAuth consent screen URL for this connector's own client,
    requesting offline access (a refresh token) and forcing the consent
    prompt every time so a refresh token is issued even on a re-link."""
    settings = get_settings()
    params = {
        "client_id": settings.gmail_mcp_client_id,
        "redirect_uri": settings.gmail_mcp_redirect_uri,
        "response_type": "code",
        "scope": GMAIL_MCP_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{GOOGLE_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code_for_tokens(code: str) -> dict:
    """Exchanges an authorization `code` from gmail_mcp_callback for a
    token response (access_token, refresh_token, ...) via Google's token
    endpoint. Raises on any HTTP/network failure -- the caller (the
    callback route) is responsible for turning that into a user-facing
    error, there's no silent-degrade path for the initial link itself."""
    settings = get_settings()
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            GOOGLE_OAUTH_TOKEN_URL,
            data={
                "client_id": settings.gmail_mcp_client_id,
                "client_secret": settings.gmail_mcp_client_secret,
                "redirect_uri": settings.gmail_mcp_redirect_uri,
                "code": code,
                "grant_type": "authorization_code",
            },
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    return response.json()


def build_service_role_client() -> Optional[Client]:
    """
    A service-role Supabase client, used only because gmail_mcp_callback
    runs as a plain browser redirect from Google with no user JWT attached
    to authenticate a per-request client with -- mirrors core/scheduler.py's
    identical need to write user_oauth_tokens rows across users with RLS
    bypassed. Returns None (never raises) when SUPABASE_SERVICE_ROLE_KEY
    isn't configured; the caller treats that as "storage unavailable".
    """
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


async def store_gmail_mcp_tokens(supabase: Client, user_id: str, refresh_token: str) -> None:
    """Upserts `user_id`'s Gmail MCP refresh token (one row per
    user_id+provider, see reference/007_user_oauth_tokens.sql) under the
    distinct GMAIL_MCP_PROVIDER row -- never the plain 'google' row."""
    row = {"user_id": user_id, "provider": GMAIL_MCP_PROVIDER, "refresh_token": refresh_token}
    supabase.table("user_oauth_tokens").upsert(row, on_conflict="user_id,provider").execute()


async def get_gmail_mcp_refresh_token(supabase: Client, user_id: str) -> Optional[str]:
    """The stored Gmail MCP refresh token for `user_id`, if any."""
    res = (
        supabase.table("user_oauth_tokens")
        .select("refresh_token")
        .eq("user_id", user_id)
        .eq("provider", GMAIL_MCP_PROVIDER)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0].get("refresh_token") if rows else None


async def refresh_gmail_mcp_access_token(supabase: Client, user_id: str) -> Optional[str]:
    """
    Exchanges this user's stored Gmail MCP refresh token for a fresh, short
    -lived access token via Google's token endpoint, using GMAIL_MCP_CLIENT_
    ID/SECRET (never GOOGLE_CLIENT_ID/SECRET -- that pair belongs to
    create_workspace_session's own, separate refresh fallback). Never
    raises: every failure mode (no client credentials configured, no stored
    token, a DB error, Google rejecting the refresh token) is logged and
    degrades to None, which create_gmail_mcp_session treats exactly like
    "no Gmail MCP session available this turn".
    """
    settings = get_settings()
    if not settings.gmail_mcp_client_id or not settings.gmail_mcp_client_secret:
        logger.error(
            "Gmail MCP: refresh skipped for user %s -- GMAIL_MCP_CLIENT_ID/GMAIL_MCP_CLIENT_SECRET not configured.",
            user_id,
        )
        return None

    try:
        refresh_token = await get_gmail_mcp_refresh_token(supabase, user_id)
    except Exception:
        logger.error("Gmail MCP: failed to look up stored refresh token for user %s.", user_id, exc_info=True)
        return None

    if not refresh_token:
        return None

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                GOOGLE_OAUTH_TOKEN_URL,
                data={
                    "client_id": settings.gmail_mcp_client_id,
                    "client_secret": settings.gmail_mcp_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=_HTTP_TIMEOUT_SECONDS,
            )
        response.raise_for_status()
    except Exception:
        logger.error("Gmail MCP: failed to refresh access token for user %s.", user_id, exc_info=True)
        return None

    access_token = response.json().get("access_token")
    if not access_token:
        logger.error("Gmail MCP: Google's token endpoint returned no access_token for user %s.", user_id)
        return None

    return access_token
