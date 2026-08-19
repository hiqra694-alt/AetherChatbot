"""
Voice agent (Phase 1): LiveKit room-token issuance for the realtime voice
pipeline. Entirely self-contained in the voice/ package -- never imports
from or modifies api/chat/routers.py, the RAG pipeline, or any MCP
connector. See voice/auth.py's module docstring for the isolation
rationale.

Registered in main.py alongside the app's other routers (one additive
`include_router` call, same as every other router there).
"""

import logging
from datetime import timedelta
from typing import Tuple

from fastapi import APIRouter, Depends, HTTPException
from livekit import api
from supabase import Client
from supabase_auth.types import User

from core.config import get_settings
from voice.auth import get_authenticated_supabase
from voice.schemas import VoiceTokenRequest, VoiceTokenResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

VOICE_TOKEN_TTL = timedelta(hours=2)


@router.post("/token", response_model=VoiceTokenResponse)
async def issue_voice_token(
    request: VoiceTokenRequest,
    auth: Tuple[Client, User] = Depends(get_authenticated_supabase),
):
    """
    Mints a short-lived LiveKit access token that grants the authenticated
    caller join access to the room named after their chat_session_id, so
    the voice agent and the caller's browser client land in the same room.
    """
    _, user = auth

    settings = get_settings()
    if not (settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret):
        logger.error("Voice: LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET are not fully configured.")
        raise HTTPException(status_code=500, detail="Voice agent is not configured on the server.")

    try:
        token = (
            api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(str(user.id))
            .with_name(user.email or "AetherChat User")
            .with_grants(api.VideoGrants(room_join=True, room=request.chat_session_id))
            .with_ttl(VOICE_TOKEN_TTL)
            .to_jwt()
        )
    except Exception:
        logger.exception("Voice: failed to generate a LiveKit access token for user %s.", user.id)
        raise HTTPException(status_code=500, detail="Failed to generate a voice session token.")

    return VoiceTokenResponse(token=token)
