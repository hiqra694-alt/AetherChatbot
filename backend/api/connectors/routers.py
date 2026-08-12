import logging
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from supabase import create_client, Client, ClientOptions

from core.config import get_settings
from api.connectors.schemas import (
    ConnectorStatusResponse,
    DisconnectResponse,
    OAuthProvider,
    StoreTokenRequest,
    StoreTokenResponse,
)
from api.connectors.services import (
    delete_oauth_refresh_token,
    get_stored_refresh_token_providers,
    store_oauth_refresh_token,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/connectors", tags=["connectors"])


async def get_authenticated_supabase(authorization: Optional[str] = Header(None)) -> Tuple[Client, str]:
    """
    Dependency mirroring api/tasks/routers.py, api/memory/routers.py, and
    api/chat/routers.py's dependency of the same shape: builds a Supabase
    client bound to the caller's JWT so RLS policies on user_oauth_tokens
    apply, then resolves user_id from that verified token instead of
    trusting a client-supplied value.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required.")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format. Must be Bearer <token>")

    jwt_token = authorization.split(" ")[1]
    settings = get_settings()

    options = ClientOptions(headers={"Authorization": f"Bearer {jwt_token}"})
    supabase = create_client(settings.supabase_url, settings.supabase_anon_key, options=options)

    try:
        user_res = supabase.auth.get_user(jwt_token)
    except Exception as auth_err:
        logger.error(f"Failed to resolve user from token: {auth_err}")
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")

    if not user_res or not user_res.user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")

    return supabase, user_res.user.id


@router.get("/status", response_model=ConnectorStatusResponse)
async def connector_status(
    auth: Tuple[Client, str] = Depends(get_authenticated_supabase),
):
    """
    Reports whether this user has a stored, usable refresh_token per
    provider (see ConnectorStatusResponse) -- called by the frontend
    alongside user.identities so toggleConnector can tell a genuinely
    unlinked provider apart from one that's linked but can never open a
    Workspace/GitHub session because it was linked without offline access,
    and surface that a disconnect+reconnect is required instead of silently
    enabling a connector that will never actually work.
    """
    supabase, user_id = auth

    try:
        providers = await get_stored_refresh_token_providers(supabase, user_id)
    except Exception as db_err:
        logger.error(f"Failed to look up stored OAuth providers for user {user_id}: {db_err}")
        raise HTTPException(status_code=500, detail="Failed to load connector status. Please try again later.")

    return ConnectorStatusResponse(google="google" in providers, github="github" in providers)


@router.post("/store-token", response_model=StoreTokenResponse)
async def store_token(
    payload: StoreTokenRequest,
    auth: Tuple[Client, str] = Depends(get_authenticated_supabase),
):
    """
    Persists the authenticated user's OAuth refresh_token for `payload.provider`
    (upserted, one row per user+provider). Backs the server-side token-refresh
    fallback in mcp_integration.mcp_manager.create_workspace_session, so a
    linked Google connector keeps working across reloads even once the
    access token cached in the browser session has aged out.
    """
    supabase, user_id = auth

    try:
        stored = await store_oauth_refresh_token(supabase, user_id, payload.provider.value, payload.refresh_token)
    except Exception as db_err:
        logger.error(f"Failed to store OAuth refresh token for user {user_id} ({payload.provider.value}): {db_err}")
        raise HTTPException(status_code=500, detail="Failed to store the connector token. Please try again later.")

    return StoreTokenResponse(stored=stored)


@router.delete("/disconnect", response_model=DisconnectResponse)
async def disconnect(
    provider: OAuthProvider = Query(..., description="Which linked provider to disconnect, e.g. 'google' or 'github'"),
    auth: Tuple[Client, str] = Depends(get_authenticated_supabase),
):
    """
    Deletes the authenticated user's stored refresh_token for `provider`
    (see reference/007_user_oauth_tokens.sql), so the server-side
    token-refresh fallback (mcp_manager._refresh_google_access_token) can no
    longer mint fresh access tokens on this user's behalf for that provider.

    Idempotent: succeeds even when no row existed for this provider (e.g.
    it was only ever used with a live in-browser token, never captured an
    offline refresh_token). This endpoint only ever cleans up this app's own
    side-table -- actually unlinking the identity from the user's Supabase
    account is a separate client-side supabase.auth.unlinkIdentity call (see
    src/app/page.tsx's handleDisconnectConnector), since that's a Supabase
    Auth operation this backend has no privileged API for.
    """
    supabase, user_id = auth

    try:
        await delete_oauth_refresh_token(supabase, user_id, provider.value)
    except Exception as db_err:
        logger.error(f"Failed to delete OAuth refresh token for user {user_id} ({provider.value}): {db_err}")
        raise HTTPException(status_code=500, detail="Failed to disconnect the connector. Please try again later.")

    return DisconnectResponse(status="success", message=f"{provider.value.capitalize()} disconnected successfully")
