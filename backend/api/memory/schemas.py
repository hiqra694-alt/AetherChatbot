from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    """
    A single persistent fact/preference the AI has learned about the user,
    stored in `user_memory` and injected into every chat's system prompt
    regardless of session (unlike per-session document_chunks).
    """
    id: str = Field(..., description="Primary key of the stored fact")
    fact: str = Field(..., description="The extracted fact or preference text")
    created_at: datetime = Field(..., description="When this fact was first stored")


class MemoryListResponse(BaseModel):
    """
    Payload for GET /api/memory — the authenticated user's stored facts.
    """
    facts: List[MemoryFact] = Field(default_factory=list)


class MemoryDeleteResponse(BaseModel):
    """
    Payload for DELETE /api/memory/{memory_id} — confirms the fact was removed.
    """
    id: str = Field(..., description="ID of the deleted fact")
    deleted: bool = Field(..., description="Whether a row was actually removed")
