"""
Unified Google Workspace OAuth flow (Phase 2) -- backs the native
Gmail/Calendar/Drive Python tools in connector_integrations/google_tools.py,
which replace the retired Google Managed MCP Workspace connector (see
connector_integrations/connector_manager.py's module docstring for that
history).

One shared OAuth client and one consent grant covers all three services:
`/authorize` requests all four scopes below up front, so the frontend's
three independent google_gmail/google_calendar/google_drive toggles (see
src/app/page.tsx) never need to trigger a second consent screen -- they
just gate which of google_tools.py's functions the model is offered, once
this single flow has linked the account. The resulting refresh token is
stored under one shared 'google_workspace' user_oauth_tokens row (see
GOOGLE_WORKSPACE_PROVIDER below) -- distinct from every other Google OAuth
client already in this codebase (Supabase's own 'google' sign-in provider,
and canvas/drive_oauth.py's narrower, Canvas-only 'google_drive' client) --
so linking or revoking this one can never read, overwrite, or interact with
those.

Deliberately built on google_auth_oauthlib.flow.Flow / google.oauth2
.credentials.Credentials (the official Google API client libraries) rather
than a hand-rolled httpx token exchange like the now-retired gmail_mcp.py or
canvas/drive_oauth.py used -- google_tools.py already needs
google-api-python-client to build service resources, so standardizing the
OAuth side on google-auth/google-auth-oauthlib too avoids a second, parallel
way of doing the same token exchange.

Reuses api.connectors.services.store_oauth_refresh_token /
get_stored_refresh_token_providers for the actual user_oauth_tokens read/
write -- those are already generic over `provider`, so there's no reason to
duplicate that upsert here. Only the OAuth-specific pieces (signed state,
the Flow, scope verification) are new.
"""

import asyncio
import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from google_auth_oauthlib.flow import Flow
from supabase import Client, create_client

from api.connectors.services import store_oauth_refresh_token
from core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/connectors/google", tags=["connectors-google-workspace"])

# Distinct user_oauth_tokens.provider value for this connector -- see the
# module docstring for why this must never collide with any other Google
# OAuth client's row.
GOOGLE_WORKSPACE_PROVIDER = "google_workspace"

# All four scopes are requested together on every /authorize call so a
# single consent grant covers Gmail, Calendar, and Drive at once -- the
# frontend's three independent toggles only ever gate which native tools
# (connector_integrations/google_tools.py) are offered to the model, never
# which scopes were granted.
GOOGLE_WORKSPACE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

GOOGLE_OAUTH_TOKEN_URI = "https://oauth2.googleapis.com/token"
GOOGLE_OAUTH_AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"

# How long a signed `state` value (see sign_state/verify_state) stays valid
# for -- comfortably longer than any human takes to complete the Google
# consent screen, short enough that a leaked/logged state can't be replayed
# long after the fact.
STATE_TTL_SECONDS = 600


def _state_signing_key() -> bytes:
    return get_settings().google_workspace_client_secret.encode()


def sign_state(user_id: str, code_verifier: str) -> str:
    """
    Packs `user_id`, the PKCE `code_verifier` generated for this
    authorization attempt, and an expiry into a signed, URL-safe token to
    pass as the OAuth `state` param.

    Carrying `code_verifier` here (rather than e.g. an in-memory dict on the
    Flow object) is required because FastAPI/this process is stateless
    across the redirect round-trip: build_workspace_authorize_url's Flow
    instance is gone by the time google_workspace_callback runs (different
    request, possibly a different worker process), so the verifier has no
    other way to survive the trip to Google and back. Google's token
    endpoint rejects the exchange with `invalid_grant: Missing code
    verifier` if the same value that produced the `code_challenge` sent to
    /authorize isn't replayed at /callback.

    This also still doubles as how google_workspace_callback recovers which
    user an incoming Google redirect belongs to, since that redirect is a
    plain browser GET with no Authorization header at all. Signed
    (HMAC-SHA256, keyed on GOOGLE_WORKSPACE_CLIENT_SECRET) rather than
    stored server-side, so no session store is needed and it survives an
    app restart between authorize and callback. Signing (not encrypting)
    is sufficient here: `code_verifier` and the authorization `code` both
    already travel together, in the clear, in the one browser redirect from
    Google to /callback -- anyone able to intercept that URL has both no
    matter how `state` is encoded, and redeeming either still requires this
    app's GOOGLE_WORKSPACE_CLIENT_SECRET, which never leaves the server.
    What signing defends against is a forged/tampered state being accepted
    as if it were ours.
    """
    expiry = int(time.time()) + STATE_TTL_SECONDS
    payload = f"{user_id}:{code_verifier}:{expiry}"
    sig = hmac.new(_state_signing_key(), payload.encode(), hashlib.sha256).hexdigest()
    raw = f"{payload}:{sig}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def verify_state(state: str) -> tuple[str, str]:
    """
    Inverse of sign_state: returns the embedded (user_id, code_verifier) if
    `state` carries a valid, unexpired signature, otherwise raises
    ValueError. Never trusts an unsigned or tampered state -- verifying it
    is what stands in for "this callback request is authenticated" in a
    flow that has no session/JWT to check, and is also what proves the
    recovered code_verifier is the one this app generated rather than one
    an attacker supplied.
    """
    try:
        padded = state + "=" * (-len(state) % 4)
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        user_id, code_verifier, expiry_str, sig = raw.split(":")
    except Exception as exc:
        raise ValueError("Malformed OAuth state.") from exc

    expected_sig = hmac.new(
        _state_signing_key(), f"{user_id}:{code_verifier}:{expiry_str}".encode(), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        raise ValueError("OAuth state signature mismatch.")
    if int(expiry_str) < time.time():
        raise ValueError("OAuth state has expired.")

    return user_id, code_verifier


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


def _client_config() -> dict:
    settings = get_settings()
    return {
        "web": {
            "client_id": settings.google_workspace_client_id,
            "client_secret": settings.google_workspace_client_secret,
            "auth_uri": GOOGLE_OAUTH_AUTH_URI,
            "token_uri": GOOGLE_OAUTH_TOKEN_URI,
            "redirect_uris": [settings.google_workspace_redirect_uri],
        }
    }


def _build_flow() -> Flow:
    settings = get_settings()
    if settings.google_workspace_dev_mode:
        # oauthlib refuses to build/parse any token request whose
        # redirect_uri or token endpoint isn't https -- fatal for local dev,
        # where GOOGLE_WORKSPACE_REDIRECT_URI is necessarily http://localhost.
        # Only ever set under GOOGLE_WORKSPACE_DEV_MODE (never true in a
        # deployed environment, see that setting's docstring in core/config.py).
        # Settings.model_config's env_file is parsed into this Settings model
        # only -- pydantic-settings never mirrors .env into os.environ itself
        # (see core/config.py's module docstring) -- so this has to be set
        # explicitly here, on every Flow build, rather than assumed present.
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    flow = Flow.from_client_config(_client_config(), scopes=GOOGLE_WORKSPACE_SCOPES)
    flow.redirect_uri = settings.google_workspace_redirect_uri
    return flow


def build_workspace_authorize_url(state: str, code_verifier: str) -> str:
    """
    The Google OAuth consent screen URL for this connector's shared client,
    requesting all four GOOGLE_WORKSPACE_SCOPES, offline access (a refresh
    token), and forcing the consent prompt every time so a refresh token is
    issued even on a re-link (Google otherwise only returns one on a
    account/client's very first, non-silent consent).

    `code_verifier` is set on the Flow *before* calling authorization_url so
    google-auth-oauthlib derives this request's `code_challenge` from our
    own verifier instead of silently autogenerating (and then discarding)
    one of its own -- see sign_state's docstring for why the same value has
    to be recoverable again in google_workspace_callback.
    """
    flow = _build_flow()
    flow.code_verifier = code_verifier
    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="false",
        state=state,
    )
    return authorization_url


class GoogleTokenExchangeError(Exception):
    """Raised when Google's token endpoint rejects the authorization code."""


async def exchange_code_for_credentials(code: str, code_verifier: str):
    """
    Exchanges an authorization `code` from google_workspace_callback for
    credentials via google-auth-oauthlib's Flow -- returns a
    google.oauth2.credentials.Credentials instance carrying the access
    token, refresh token, and granted scopes. Flow.fetch_token is a
    blocking (requests-based) call, so it's run off the event loop.
    Raises GoogleTokenExchangeError on any failure; the caller (the
    callback route) is responsible for turning that into a user-facing
    error -- there's no silent-degrade path for the initial link itself.

    `code_verifier` must be the same value build_workspace_authorize_url
    used to derive the `code_challenge` sent to Google's consent screen --
    this is a brand new Flow instance (this process has no memory of the
    one /authorize built), so without it Google's token endpoint rejects
    the exchange with `invalid_grant: Missing code verifier`.
    """
    flow = _build_flow()
    flow.code_verifier = code_verifier
    try:
        await asyncio.to_thread(flow.fetch_token, code=code)
    except Exception as exc:
        # str(exc) alone is often uninformative (e.g. oauthlib's
        # InsecureTransportError has no message at all) -- prefix the
        # exception's class name so the log line identifies which failure
        # mode this is (InsecureTransportError, MismatchingStateError,
        # an OAuth2Error carrying invalid_grant/redirect_uri_mismatch from
        # Google's token endpoint, etc.) without needing the full traceback.
        logger.exception(
            "Google Workspace: flow.fetch_token raised %s: %s", type(exc).__name__, exc
        )
        raise GoogleTokenExchangeError(f"{type(exc).__name__}: {exc}") from exc
    return flow.credentials


def _granted_scope_set(credentials) -> set:
    """Normalizes Credentials.granted_scopes/.scopes -- either can be None,
    a list, or (depending on the token response shape) a single
    space-delimited string -- into a plain set of scope strings."""
    for value in (getattr(credentials, "granted_scopes", None), getattr(credentials, "scopes", None)):
        if not value:
            continue
        if isinstance(value, str):
            return set(value.split())
        return set(value)
    return set()


def granted_scopes_are_sufficient(credentials) -> bool:
    """
    True iff `credentials` carries every scope in GOOGLE_WORKSPACE_SCOPES.
    Google always echoes back the scope(s) it actually granted; what's
    requested in build_workspace_authorize_url is only ever a request, not
    a guarantee -- it can be silently narrowed (no error, token issuance
    still succeeds) if e.g. a Workspace admin hasn't approved this app for
    one of the sensitive scopes under Admin Console > Security > API
    controls. Catching that here, at link time, avoids storing a refresh
    token that will reject every native tool call downstream with a
    confusing "insufficient permission" error two hops from its real cause.
    """
    granted = _granted_scope_set(credentials)
    if not granted:
        # Some token responses omit granted scope entirely when unchanged
        # from a prior grant -- nothing to contradict here, and failing
        # closed on an absent field would reject a perfectly valid token.
        return True
    return set(GOOGLE_WORKSPACE_SCOPES).issubset(granted)


def build_service_role_client() -> Optional[Client]:
    """
    A service-role Supabase client, used only because
    google_workspace_callback runs as a plain browser redirect from Google
    with no user JWT attached to authenticate a per-request client with --
    mirrors canvas/drive_oauth.py's identical need. Returns None (never
    raises) when SUPABASE_SERVICE_ROLE_KEY isn't configured; the caller
    treats that as "storage unavailable".
    """
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


def _require_google_workspace_configured() -> None:
    settings = get_settings()
    if not (
        settings.google_workspace_client_id
        and settings.google_workspace_client_secret
        and settings.google_workspace_redirect_uri
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Google Workspace connector is not configured (GOOGLE_WORKSPACE_CLIENT_ID/"
                "GOOGLE_WORKSPACE_CLIENT_SECRET/GOOGLE_WORKSPACE_REDIRECT_URI)."
            ),
        )


@router.get("/authorize")
async def google_workspace_authorize(
    access_token: Optional[str] = Query(
        None,
        description="The caller's own live Supabase session access token. The frontend builds the URL as "
        "`{BACKEND_URL}/api/connectors/google/authorize?access_token=<session.access_token>` (from "
        "supabase.auth.getSession() client-side) and navigates the browser there (window.location.href, "
        "or a plain <a href>) -- never a fetch/XHR call, since Google's own redirect back to /callback has "
        "to land the browser on this flow. Sent as a query param rather than an Authorization header "
        "because a full-page navigation can't carry custom headers.",
    ),
    dev_user_id: Optional[str] = Query(
        None,
        description="LOCAL DEVELOPMENT ONLY. An unverified user_id, accepted in place of access_token so "
        "the Google consent/redirect/callback flow can be tested by pasting a URL directly into a browser. "
        "Only honored when GOOGLE_WORKSPACE_DEV_MODE=true; ignored (with a 422, same as omitting both "
        "params) otherwise. Must be a real auth.users UUID for the token-storage step at the end of "
        "/callback to succeed (user_oauth_tokens.user_id has a foreign key to auth.users).",
    ),
):
    """
    Redirects the browser to Google's OAuth consent screen for this
    connector's shared client (GOOGLE_WORKSPACE_CLIENT_ID), requesting all
    four GOOGLE_WORKSPACE_SCOPES, offline access, and forcing the consent
    prompt so a refresh token is always returned. The caller's identity is
    embedded in a signed `state` param (see sign_state) rather than a
    server-side session, so google_workspace_callback can recover it later
    with no session store needed.
    """
    _require_google_workspace_configured()
    settings = get_settings()

    if access_token:
        try:
            user_id = await resolve_user_id_from_access_token(access_token)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    elif dev_user_id and settings.google_workspace_dev_mode:
        logger.warning(
            "Google Workspace: /authorize opened via the GOOGLE_WORKSPACE_DEV_MODE dev_user_id shortcut "
            "for unverified user_id=%s -- GOOGLE_WORKSPACE_DEV_MODE must never be set in a deployed "
            "environment.",
            dev_user_id,
        )
        user_id = dev_user_id
    elif dev_user_id and not settings.google_workspace_dev_mode:
        raise HTTPException(
            status_code=403,
            detail="The 'dev_user_id' shortcut requires GOOGLE_WORKSPACE_DEV_MODE=true (local development only).",
        )
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide 'access_token' (the caller's live Supabase session token). For local "
            "development only, set GOOGLE_WORKSPACE_DEV_MODE=true and pass 'dev_user_id' instead to test "
            "the redirect flow without a real session.",
        )

    code_verifier = secrets.token_urlsafe(64)
    state = sign_state(user_id, code_verifier)
    return RedirectResponse(build_workspace_authorize_url(state, code_verifier), status_code=302)


@router.get("/callback")
async def google_workspace_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
):
    """
    Exchanges the authorization code Google redirected back with for
    credentials, and persists the refresh token to user_oauth_tokens under
    the shared 'google_workspace' provider (see
    api.connectors.services.store_oauth_refresh_token) -- never the plain
    'google' row, nor any other connector's row. Written with a
    service-role client since this is a plain browser redirect from Google
    carrying no user JWT to authenticate a per-request client with -- the
    signed `state` param is what proves which user this belongs to instead.
    """
    if error:
        return HTMLResponse(f"<h1>Google Workspace connection failed</h1><p>{error}</p>", status_code=400)

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing 'code' or 'state' parameter.")

    try:
        user_id, code_verifier = verify_state(state)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")

    try:
        credentials = await exchange_code_for_credentials(code, code_verifier)
    except GoogleTokenExchangeError as exc:
        # The traceback/exception type is already logged inside
        # exchange_code_for_credentials -- just tie it to this user here.
        logger.error("Google Workspace: token exchange failed for user %s: %s", user_id, exc)
        raise HTTPException(status_code=502, detail="Failed to exchange the authorization code with Google.")

    refresh_token = credentials.refresh_token
    if not refresh_token:
        logger.error("Google Workspace: Google did not return a refresh_token for user %s.", user_id)
        raise HTTPException(
            status_code=502,
            detail="Google did not return a refresh token. Disconnect any prior grant and try again.",
        )

    # Catch an under-scoped grant here, at link time, rather than storing a
    # refresh token that will mint access tokens Gmail/Calendar/Drive
    # reject on every native tool call with a permission error -- see
    # granted_scopes_are_sufficient for why Google can silently narrow the
    # granted scope below what was requested/consented.
    if not granted_scopes_are_sufficient(credentials):
        logger.error(
            "Google Workspace: token exchange for user %s did not grant every required scope (%s) -- "
            "granted scopes were %r.",
            user_id, GOOGLE_WORKSPACE_SCOPES, sorted(_granted_scope_set(credentials)),
        )
        return HTMLResponse(
            "<h1>Google Workspace connection incomplete</h1>"
            "<p>Google did not grant every permission this app requires "
            f"(<code>{', '.join(GOOGLE_WORKSPACE_SCOPES)}</code>). This usually means either the OAuth "
            "consent screen's scopes were changed after you last approved this app, or -- for a Google "
            "Workspace account -- your organization's admin hasn't approved this app for one of these "
            "scopes under Admin Console &gt; Security &gt; API controls &gt; App access control. Go to "
            "<a href=\"https://myaccount.google.com/permissions\">Google Account &gt; Security &gt; "
            "Third-party access</a>, remove this app's access, then retry the connection.</p>",
            status_code=403,
        )

    supabase = build_service_role_client()
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Google Workspace connector storage is not configured (SUPABASE_SERVICE_ROLE_KEY).",
        )

    try:
        await store_oauth_refresh_token(supabase, user_id, GOOGLE_WORKSPACE_PROVIDER, refresh_token)
    except Exception:
        logger.exception("Google Workspace: failed to store refresh token for user %s.", user_id)
        raise HTTPException(status_code=500, detail="Failed to save the Google Workspace connection.")

    logger.info("Google Workspace: connected successfully for user %s.", user_id)
    settings = get_settings()
    frontend_redirect_url = f"{settings.frontend_url.rstrip('/')}/?connected=google_workspace"
    return RedirectResponse(frontend_redirect_url, status_code=302)
