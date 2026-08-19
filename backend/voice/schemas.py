from pydantic import BaseModel, ConfigDict, Field


class VoiceTokenRequest(BaseModel):
    """Payload for POST /api/voice/token."""
    model_config = ConfigDict(extra="forbid")

    chat_session_id: str = Field(..., min_length=1, description="The chat session id to use as the LiveKit room name.")


class VoiceTokenResponse(BaseModel):
    """Payload for POST /api/voice/token."""
    model_config = ConfigDict(extra="forbid")

    token: str = Field(..., description="A signed LiveKit access token (JWT) scoped to the requested room.")
