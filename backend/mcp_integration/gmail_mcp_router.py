"""
Dedicated OAuth routes for the Gmail MCP connector (Phase 1) -- entirely
separate from api/connectors/routers.py, which backs Supabase's own
linkIdentity-based provider linking for the existing 'google'/'github'
rows. This router speaks directly to Google's OAuth endpoints and never
touches Supabase's identity-linking flow at all; see mcp_integration/
gmail_mcp.py's module docstring for the full isolation rationale.

Registered in main.py alongside the app's other routers (one additive
`include_router` call, same as every other router there) -- everything
else about this connector lives in this package.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from core.config import get_settings
from mcp_integration.gmail_mcp import (
    GMAIL_MCP_SCOPE,
    build_google_authorize_url,
    exchange_code_for_tokens,
    build_service_role_client,
    granted_scope_is_sufficient,
    resolve_user_id_from_access_token,
    sign_state,
    store_gmail_mcp_tokens,
    verify_state,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/connectors/gmail-mcp", tags=["mcp-gmail"])


def _require_gmail_mcp_configured() -> None:
    settings = get_settings()
    if not (settings.gmail_mcp_client_id and settings.gmail_mcp_client_secret and settings.gmail_mcp_redirect_uri):
        raise HTTPException(
            status_code=503,
            detail="Gmail MCP connector is not configured (GMAIL_MCP_CLIENT_ID/GMAIL_MCP_CLIENT_SECRET/GMAIL_MCP_REDIRECT_URI).",
        )


@router.get("/authorize")
async def gmail_mcp_authorize(
    access_token: Optional[str] = Query(
        None,
        description="The caller's own live Supabase session access token. This is how the frontend "
        "is expected to call this route: build the URL as "
        "`{BACKEND_URL}/api/connectors/gmail-mcp/authorize?access_token=<session.access_token>` "
        "(from supabase.auth.getSession() client-side) and navigate to it (window.location.href = url, "
        "or a plain <a href>) once the user is signed in -- never a fetch/XHR call, since Google's own "
        "redirect back to /callback has to land the *browser* on this flow. Sent as a query param "
        "rather than an Authorization header because a full-page navigation can't carry custom headers.",
    ),
    dev_user_id: Optional[str] = Query(
        None,
        description="LOCAL DEVELOPMENT ONLY. An unverified user_id, accepted in place of access_token "
        "so the Google consent/redirect/callback flow can be tested by pasting a URL directly into a "
        "browser. Only honored when GMAIL_MCP_DEV_MODE=true; ignored (with a 422, same as omitting "
        "both params) otherwise. Must be a real auth.users UUID for the token-storage step at the end "
        "of /callback to succeed (user_oauth_tokens.user_id has a foreign key to auth.users).",
    ),
):
    """
    Redirects the browser to Google's OAuth consent screen for this
    connector's own client (GMAIL_MCP_CLIENT_ID), requesting offline access
    and forcing the consent prompt so a refresh token is always returned.
    The caller's identity is embedded in a signed `state` param (see
    gmail_mcp.sign_state) rather than a server-side session, so
    gmail_mcp_callback can recover it later with no session store needed.
    """
    _require_gmail_mcp_configured()
    settings = get_settings()

    if access_token:
        try:
            user_id = await resolve_user_id_from_access_token(access_token)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    elif dev_user_id and settings.gmail_mcp_dev_mode:
        logger.warning(
            "Gmail MCP: /authorize opened via the GMAIL_MCP_DEV_MODE dev_user_id shortcut for "
            "unverified user_id=%s -- GMAIL_MCP_DEV_MODE must never be set in a deployed environment.",
            dev_user_id,
        )
        user_id = dev_user_id
    elif dev_user_id and not settings.gmail_mcp_dev_mode:
        raise HTTPException(
            status_code=403,
            detail="The 'dev_user_id' shortcut requires GMAIL_MCP_DEV_MODE=true (local development only).",
        )
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide 'access_token' (the caller's live Supabase session token). For local "
            "development only, set GMAIL_MCP_DEV_MODE=true and pass 'dev_user_id' instead to test the "
            "redirect flow without a real session.",
        )

    state = sign_state(user_id)
    return RedirectResponse(build_google_authorize_url(state), status_code=302)


@router.get("/callback")
async def gmail_mcp_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
):
    """
    Exchanges the authorization code Google redirected back with for
    access/refresh tokens, and persists the refresh token to
    user_oauth_tokens under the distinct 'google_gmail_mcp' provider (see
    gmail_mcp.store_gmail_mcp_tokens) -- never the plain 'google' row the
    existing Workspace connector uses. Written with a service-role client
    (gmail_mcp.build_service_role_client) since this is a plain browser
    redirect from Google carrying no user JWT to authenticate a per-request
    client with -- the signed `state` param is what proves which user this
    belongs to instead.
    """
    if error:
        return HTMLResponse(f"<h1>Gmail MCP connection failed</h1><p>{error}</p>", status_code=400)

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing 'code' or 'state' parameter.")

    try:
        user_id = verify_state(state)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")

    try:
        tokens = await exchange_code_for_tokens(code)
    except Exception:
        logger.exception("Gmail MCP: token exchange failed for user %s.", user_id)
        raise HTTPException(status_code=502, detail="Failed to exchange the authorization code with Google.")

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        logger.error("Gmail MCP: Google did not return a refresh_token for user %s.", user_id)
        raise HTTPException(
            status_code=502,
            detail="Google did not return a refresh token. Disconnect any prior grant and try again.",
        )

    # Catch an under-scoped grant here, at link time, rather than storing a
    # refresh token that will mint access tokens gmailmcp.googleapis.com
    # rejects on every tool call with a "caller does not have permission"
    # error -- see gmail_mcp.granted_scope_is_sufficient for why Google can
    # silently narrow the granted scope below what was requested/consented.
    if not granted_scope_is_sufficient(tokens):
        logger.error(
            "Gmail MCP: token exchange for user %s did not grant the required scope (%s) -- "
            "granted scope was %r.", user_id, GMAIL_MCP_SCOPE, tokens.get("scope"),
        )
        return HTMLResponse(
            "<h1>Gmail MCP connection incomplete</h1>"
            "<p>Google did not grant the Gmail permission this app requires "
            f"(<code>{GMAIL_MCP_SCOPE}</code>). This usually means either the OAuth consent screen's "
            "scope was changed after you last approved this app, or -- for a Google Workspace account -- "
            "your organization's admin hasn't approved this app for Gmail access under "
            "Admin Console &gt; Security &gt; API controls &gt; App access control. "
            "Go to <a href=\"https://myaccount.google.com/permissions\">Google Account &gt; Security &gt; "
            "Third-party access</a>, remove this app's access, then retry the connection.</p>",
            status_code=403,
        )

    supabase = build_service_role_client()
    if supabase is None:
        raise HTTPException(status_code=503, detail="Gmail MCP connector storage is not configured (SUPABASE_SERVICE_ROLE_KEY).")

    try:
        await store_gmail_mcp_tokens(supabase, user_id, refresh_token)
    except Exception:
        logger.exception("Gmail MCP: failed to store refresh token for user %s.", user_id)
        raise HTTPException(status_code=500, detail="Failed to save the Gmail MCP connection.")

    logger.info("Gmail MCP: connected successfully for user %s.", user_id)
    return HTMLResponse("<h1>Gmail MCP connected</h1><p>You can close this window.</p>")
