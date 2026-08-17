"""
Isolated Google Drive OAuth for the Canvas feature (Phase 1) -- fully
separate from every other connector in this codebase.

Deliberately duplicates rather than reuses machinery from mcp_integration/
gmail_mcp.py (its own state-signing scheme, its own token-refresh helper,
its own service-role client builder) even though the shape is identical.
That's intentional, not an oversight: this module authenticates against a
brand-new, separate OAuth client (CANVAS_DRIVE_CLIENT_ID/SECRET/REDIRECT_URI)
and stores its token under a brand-new, distinct user_oauth_tokens provider
row ('google_drive' -- see CANVAS_DRIVE_PROVIDER below), so that linking or
refreshing it can never read, overwrite, or otherwise interact with the
existing 'google' row (create_workspace_session's refresh fallback) or the
'google_gmail_mcp' row (gmail_mcp.py). Keeping the code paths fully separate
is what guarantees that; sharing a helper would risk a future edit to one
silently affecting the others.
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

# Distinct user_oauth_tokens.provider value for this connector -- see the
# module docstring for why this must never collide with 'google' or
# 'google_gmail_mcp'.
CANVAS_DRIVE_PROVIDER = "google_drive"

# Restricted to files this app itself creates -- not the user's whole Drive.
CANVAS_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"

GOOGLE_OAUTH_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"

# How long a signed `state` value (see sign_state/verify_state) stays valid
# for -- comfortably longer than any human takes to complete the Google
# consent screen, short enough that a leaked/logged state can't be replayed
# long after the fact.
STATE_TTL_SECONDS = 600

_HTTP_TIMEOUT_SECONDS = 10.0


def granted_scope_is_sufficient(token_response: dict) -> bool:
    """
    True iff `token_response` (the JSON body from either the authorization_
    code or refresh_token grant against GOOGLE_OAUTH_TOKEN_URL) actually
    carries CANVAS_DRIVE_SCOPE. Google always echoes back the scope it
    actually granted -- as a space-delimited `scope` string -- on both grant
    types; what's requested in build_drive_authorize_url is only ever a
    request, not a guarantee.

    Missing `scope` entirely (some token responses omit it when the grant is
    unchanged from a prior one) is treated as sufficient -- there's nothing
    to contradict here, and failing closed on an absent field would reject
    perfectly valid tokens.
    """
    granted = token_response.get("scope")
    if not granted:
        return True
    return CANVAS_DRIVE_SCOPE in granted.split()


def _state_signing_key() -> bytes:
    return get_settings().canvas_drive_client_secret.encode()


def sign_state(user_id: str) -> str:
    """
    Packs `user_id` and an expiry into a signed, URL-safe token to pass as
    the OAuth `state` param -- this is how drive_callback recovers which
    user an incoming Google redirect belongs to, since that redirect is a
    plain browser GET with no Authorization header at all. Signed (HMAC-
    SHA256, keyed on CANVAS_DRIVE_CLIENT_SECRET) rather than stored
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


def build_drive_authorize_url(state: str) -> str:
    """The Google OAuth consent screen URL for this connector's own client,
    requesting offline access (a refresh token) and forcing the consent
    prompt every time so a refresh token is issued even on a re-link."""
    settings = get_settings()
    params = {
        "client_id": settings.canvas_drive_client_id,
        "redirect_uri": settings.canvas_drive_redirect_uri,
        "response_type": "code",
        "scope": CANVAS_DRIVE_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{GOOGLE_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_code_for_tokens(code: str) -> dict:
    """Exchanges an authorization `code` from drive_callback for a token
    response (access_token, refresh_token, ...) via Google's token endpoint.
    Raises on any HTTP/network failure -- the caller (the callback route) is
    responsible for turning that into a user-facing error."""
    settings = get_settings()
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            GOOGLE_OAUTH_TOKEN_URL,
            data={
                "client_id": settings.canvas_drive_client_id,
                "client_secret": settings.canvas_drive_client_secret,
                "redirect_uri": settings.canvas_drive_redirect_uri,
                "code": code,
                "grant_type": "authorization_code",
            },
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    response.raise_for_status()
    return response.json()


def build_service_role_client() -> Optional[Client]:
    """
    A service-role Supabase client, used only because drive_callback runs as
    a plain browser redirect from Google with no user JWT attached to
    authenticate a per-request client with. Returns None (never raises)
    when SUPABASE_SERVICE_ROLE_KEY isn't configured; the caller treats that
    as "storage unavailable".
    """
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


async def store_drive_tokens(supabase: Client, user_id: str, refresh_token: str) -> None:
    """Upserts `user_id`'s Drive refresh token (one row per user_id+provider,
    see reference/007_user_oauth_tokens.sql) under the distinct
    CANVAS_DRIVE_PROVIDER row -- never the plain 'google' or
    'google_gmail_mcp' rows."""
    row = {"user_id": user_id, "provider": CANVAS_DRIVE_PROVIDER, "refresh_token": refresh_token}
    supabase.table("user_oauth_tokens").upsert(row, on_conflict="user_id,provider").execute()


async def get_drive_refresh_token(supabase: Client, user_id: str) -> Optional[str]:
    """The stored Drive refresh token for `user_id`, if any."""
    res = (
        supabase.table("user_oauth_tokens")
        .select("refresh_token")
        .eq("user_id", user_id)
        .eq("provider", CANVAS_DRIVE_PROVIDER)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0].get("refresh_token") if rows else None


async def refresh_drive_access_token(supabase: Client, user_id: str) -> Optional[str]:
    """
    Exchanges this user's stored Drive refresh token for a fresh, short-
    lived access token via Google's token endpoint, using CANVAS_DRIVE_
    CLIENT_ID/SECRET (never GOOGLE_CLIENT_ID/SECRET or GMAIL_MCP_CLIENT_ID/
    SECRET -- those pairs belong to other, separate flows). Never raises:
    every failure mode (no client credentials configured, no stored token, a
    DB error, Google rejecting the refresh token) is logged and degrades to
    None, which the export endpoint treats as "no Drive connection
    available".
    """
    settings = get_settings()
    if not settings.canvas_drive_client_id or not settings.canvas_drive_client_secret:
        logger.error(
            "Canvas Drive: refresh skipped for user %s -- CANVAS_DRIVE_CLIENT_ID/CANVAS_DRIVE_CLIENT_SECRET not configured.",
            user_id,
        )
        return None

    try:
        refresh_token = await get_drive_refresh_token(supabase, user_id)
    except Exception:
        logger.error("Canvas Drive: failed to look up stored refresh token for user %s.", user_id, exc_info=True)
        return None

    if not refresh_token:
        return None

    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.post(
                GOOGLE_OAUTH_TOKEN_URL,
                data={
                    "client_id": settings.canvas_drive_client_id,
                    "client_secret": settings.canvas_drive_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=_HTTP_TIMEOUT_SECONDS,
            )
        response.raise_for_status()
    except Exception:
        logger.error("Canvas Drive: failed to refresh access token for user %s.", user_id, exc_info=True)
        return None

    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        logger.error("Canvas Drive: Google's token endpoint returned no access_token for user %s.", user_id)
        return None

    if not granted_scope_is_sufficient(payload):
        logger.error(
            "Canvas Drive: refreshed access token for user %s is missing the required scope (%s) -- "
            "granted scope was %r. Refusing to use it -- have the user disconnect and re-run the "
            "/api/canvas/drive/authorize consent flow.",
            user_id, CANVAS_DRIVE_SCOPE, payload.get("scope"),
        )
        return None

    return access_token
