from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class UserMemoryProfile(BaseModel):
    """
    The single, continuously-evolving narrative paragraph the AI maintains
    about a user -- one row per user_id in `user_memory`, upserted (never
    appended to) on every turn that reveals a new durable fact, and injected
    into every chat's system prompt regardless of session.
    """
    narrative: str = Field(..., description="The cohesive, evolving profile paragraph")
    updated_at: datetime = Field(..., description="When this profile was last rewritten")


class MemoryProfileResponse(BaseModel):
    """
    Payload for GET /api/memory -- the authenticated user's current memory
    profile. `narrative` is empty and `updated_at` is None when nothing has
    been learned about the user yet.
    """
    narrative: str = ""
    updated_at: Optional[datetime] = None


class MemoryDeleteResponse(BaseModel):
    """
    Payload for DELETE /api/memory -- confirms whether a stored profile
    existed and was cleared.
    """
    deleted: bool = Field(..., description="Whether a profile row existed and was removed")
