import logging
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException
from supabase import create_client, Client, ClientOptions

from core.config import get_settings
from api.memory.schemas import MemoryDeleteResponse, MemoryListResponse
from api.memory.services import delete_user_memory, list_user_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])


async def get_authenticated_supabase(authorization: Optional[str] = Header(None)) -> Tuple[Client, str]:
    """
    Dependency mirroring api/documents/routers.py and api/chat/routers.py's
    dependency of the same shape: builds a Supabase client bound to the
    caller's JWT so RLS policies on user_memory apply, then resolves user_id
    from that verified token instead of trusting a client-supplied value.
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


@router.get("", response_model=MemoryListResponse)
async def get_memory(auth: Tuple[Client, str] = Depends(get_authenticated_supabase)):
    """
    Lists every durable fact the AI has stored about the authenticated user,
    across all of their chat sessions.
    """
    supabase, user_id = auth

    try:
        facts = await list_user_memory(supabase, user_id)
    except Exception as db_err:
        logger.error(f"Failed to list memory for user {user_id}: {db_err}")
        raise HTTPException(status_code=500, detail="Failed to fetch memory. Please try again later.")

    return MemoryListResponse(facts=facts)


@router.delete("/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_memory(
    memory_id: str,
    auth: Tuple[Client, str] = Depends(get_authenticated_supabase),
):
    """
    Deletes a single stored fact belonging to the authenticated user.
    """
    supabase, user_id = auth

    try:
        deleted = await delete_user_memory(supabase, user_id, memory_id)
    except Exception as db_err:
        logger.error(f"Failed to delete memory fact '{memory_id}' for user {user_id}: {db_err}")
        raise HTTPException(status_code=500, detail="Failed to delete fact. Please try again later.")

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory fact '{memory_id}' not found.")

    return MemoryDeleteResponse(id=memory_id, deleted=True)
