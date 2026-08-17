"""
Canvas feature (Phase 1): isolated Google Drive OAuth + Markdown export.
Entirely self-contained in the canvas/ package -- never imports from or
modifies mcp_integration/intent_router.py or the Gmail MCP files. See
canvas/drive_oauth.py's module docstring for the isolation rationale.

Registered in main.py alongside the app's other routers (one additive
`include_router` call, same as every other router there).
"""

import logging
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from supabase import Client

from canvas.auth import get_authenticated_supabase
from canvas.drive_export import DriveExportError, export_markdown_as_google_doc
from canvas.drive_oauth import (
    CANVAS_DRIVE_SCOPE,
    build_drive_authorize_url,
    build_service_role_client,
    exchange_code_for_tokens,
    granted_scope_is_sufficient,
    refresh_drive_access_token,
    resolve_user_id_from_access_token,
    sign_state,
    store_drive_tokens,
    verify_state,
)
from canvas.schemas import CanvasExportRequest, CanvasExportResponse
from core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/canvas", tags=["canvas"])


def _require_drive_configured() -> None:
    settings = get_settings()
    if not (settings.canvas_drive_client_id and settings.canvas_drive_client_secret and settings.canvas_drive_redirect_uri):
        raise HTTPException(
            status_code=503,
            detail="Canvas Drive connector is not configured (CANVAS_DRIVE_CLIENT_ID/CANVAS_DRIVE_CLIENT_SECRET/CANVAS_DRIVE_REDIRECT_URI).",
        )


@router.get("/drive/authorize")
async def canvas_drive_authorize(
    access_token: Optional[str] = Query(
        None,
        description="The caller's own live Supabase session access token. The frontend builds the URL as "
        "`{BACKEND_URL}/api/canvas/drive/authorize?access_token=<session.access_token>` and navigates the "
        "browser there (window.location.href, or a plain <a href>) -- never a fetch/XHR call, since Google's "
        "own redirect back to /callback has to land the browser on this flow.",
    ),
    dev_user_id: Optional[str] = Query(
        None,
        description="LOCAL DEVELOPMENT ONLY. An unverified user_id, accepted in place of access_token so the "
        "Google consent/redirect/callback flow can be tested by pasting a URL directly into a browser. Only "
        "honored when CANVAS_DRIVE_DEV_MODE=true.",
    ),
):
    """
    Redirects the browser to Google's OAuth consent screen for this
    connector's own client (CANVAS_DRIVE_CLIENT_ID), requesting offline
    access and forcing the consent prompt so a refresh token is always
    returned.
    """
    _require_drive_configured()
    settings = get_settings()

    if access_token:
        try:
            user_id = await resolve_user_id_from_access_token(access_token)
        except ValueError:
            raise HTTPException(status_code=401, detail="Invalid or expired session token.")
    elif dev_user_id and settings.canvas_drive_dev_mode:
        logger.warning(
            "Canvas Drive: /authorize opened via the CANVAS_DRIVE_DEV_MODE dev_user_id shortcut for "
            "unverified user_id=%s -- CANVAS_DRIVE_DEV_MODE must never be set in a deployed environment.",
            dev_user_id,
        )
        user_id = dev_user_id
    elif dev_user_id and not settings.canvas_drive_dev_mode:
        raise HTTPException(
            status_code=403,
            detail="The 'dev_user_id' shortcut requires CANVAS_DRIVE_DEV_MODE=true (local development only).",
        )
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide 'access_token' (the caller's live Supabase session token). For local development "
            "only, set CANVAS_DRIVE_DEV_MODE=true and pass 'dev_user_id' instead.",
        )

    state = sign_state(user_id)
    return RedirectResponse(build_drive_authorize_url(state), status_code=302)


@router.get("/drive/callback")
async def canvas_drive_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
):
    """
    Exchanges the authorization code Google redirected back with for
    access/refresh tokens, and persists the refresh token to
    user_oauth_tokens under the distinct 'google_drive' provider (see
    drive_oauth.store_drive_tokens) -- never the 'google' or
    'google_gmail_mcp' rows other flows use.
    """
    if error:
        return HTMLResponse(f"<h1>Google Drive connection failed</h1><p>{error}</p>", status_code=400)

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing 'code' or 'state' parameter.")

    try:
        user_id = verify_state(state)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")

    try:
        tokens = await exchange_code_for_tokens(code)
    except Exception:
        logger.exception("Canvas Drive: token exchange failed for user %s.", user_id)
        raise HTTPException(status_code=502, detail="Failed to exchange the authorization code with Google.")

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        logger.error("Canvas Drive: Google did not return a refresh_token for user %s.", user_id)
        raise HTTPException(
            status_code=502,
            detail="Google did not return a refresh token. Disconnect any prior grant and try again.",
        )

    # Catch an under-scoped grant here, at link time, rather than storing a
    # refresh token that will mint access tokens Drive rejects on every
    # export with a permission error.
    if not granted_scope_is_sufficient(tokens):
        logger.error(
            "Canvas Drive: token exchange for user %s did not grant the required scope (%s) -- "
            "granted scope was %r.", user_id, CANVAS_DRIVE_SCOPE, tokens.get("scope"),
        )
        return HTMLResponse(
            "<h1>Google Drive connection incomplete</h1>"
            "<p>Google did not grant the Drive permission this app requires "
            f"(<code>{CANVAS_DRIVE_SCOPE}</code>). Go to "
            "<a href=\"https://myaccount.google.com/permissions\">Google Account &gt; Security &gt; "
            "Third-party access</a>, remove this app's access, then retry the connection.</p>",
            status_code=403,
        )

    supabase = build_service_role_client()
    if supabase is None:
        raise HTTPException(status_code=503, detail="Canvas Drive connector storage is not configured (SUPABASE_SERVICE_ROLE_KEY).")

    try:
        await store_drive_tokens(supabase, user_id, refresh_token)
    except Exception:
        logger.exception("Canvas Drive: failed to store refresh token for user %s.", user_id)
        raise HTTPException(status_code=500, detail="Failed to save the Google Drive connection.")

    logger.info("Canvas Drive: connected successfully for user %s.", user_id)
    return HTMLResponse("<h1>Google Drive connected</h1><p>You can close this window.</p>")


@router.post("/export", response_model=CanvasExportResponse)
async def canvas_export(
    payload: CanvasExportRequest,
    auth: Tuple[Client, str] = Depends(get_authenticated_supabase),
):
    """
    Exports `payload.content` (raw Markdown) as a new Google Doc in the
    authenticated user's Drive, using their stored 'google_drive' refresh
    token (see drive_oauth.refresh_drive_access_token) to mint a fresh
    access token for the upload.
    """
    supabase, user_id = auth

    access_token = await refresh_drive_access_token(supabase, user_id)
    if not access_token:
        raise HTTPException(
            status_code=409,
            detail="Google Drive is not connected. Connect it via /api/canvas/drive/authorize first.",
        )

    try:
        drive_response = await export_markdown_as_google_doc(access_token, payload.content, payload.title)
    except DriveExportError as exc:
        logger.error("Canvas: export failed for user %s: %s", user_id, exc)
        raise HTTPException(status_code=502, detail=str(exc))

    return CanvasExportResponse(
        document_id=drive_response["id"],
        web_view_link=drive_response.get("webViewLink", f"https://docs.google.com/document/d/{drive_response['id']}/edit"),
    )
