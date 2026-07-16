import logging
from typing import Optional
from fastapi import APIRouter, Request, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from supabase import create_client, Client, ClientOptions

from core.config import get_settings
from api.chat.schemas import ChatRequest
from api.chat.services import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

async def get_supabase_client(authorization: Optional[str] = Header(None)) -> Client:
    """
    Dependency to instantiate a Supabase client authenticated with the user's JWT.
    Preserves Row-Level Security (RLS) policies defined in the database.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required.")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format. Must be Bearer <token>")
        
    jwt_token = authorization.split(" ")[1]
    settings = get_settings()
    
    options = ClientOptions(headers={"Authorization": f"Bearer {jwt_token}"})
    return create_client(settings.supabase_url, settings.supabase_anon_key, options=options)

@router.post("")
async def chat_endpoint(chat_req: ChatRequest, request: Request, supabase: Client = Depends(get_supabase_client)):
    """
    Handles streaming chat responses using the designated AI provider.
    """
    history = await ChatService.get_history(supabase, chat_req.sessionId)
    provider_instance = ChatService.get_provider(chat_req.provider)
    
    generator = ChatService.stream_chat(
        provider_instance=provider_instance,
        history=history,
        supabase=supabase,
        session_id=chat_req.sessionId,
        provider_name=chat_req.provider.value,
        request_is_disconnected=request.is_disconnected,
        use_web_search=chat_req.useWebSearch
    )

    return StreamingResponse(generator, media_type="text/event-stream")
