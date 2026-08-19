"""
Isolated Authorization-header auth dependency for voice/router.py's
POST /api/voice/token -- deliberately duplicates api/chat/routers.py's
get_authenticated_supabase (same shape: verify the caller's Bearer JWT
against Supabase Auth, return a client bound to it plus the resolved user)
rather than importing it, so the voice/ package has no import-time
dependency on anything outside itself (same isolation rationale as
canvas/auth.py).
"""

import logging
from typing import Optional, Tuple

from fastapi import Header, HTTPException
from supabase import Client, ClientOptions, create_client
from supabase_auth.types import User

from core.config import get_settings

logger = logging.getLogger(__name__)


async def get_authenticated_supabase(authorization: Optional[str] = Header(None)) -> Tuple[Client, User]:
    """Builds a Supabase client bound to the caller's JWT and resolves the
    verified user from that token. Raises 401 if the caller is not
    authenticated."""
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
        logger.error(f"Voice: failed to resolve user from token: {auth_err}")
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")

    if not user_res or not user_res.user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")

    return supabase, user_res.user
