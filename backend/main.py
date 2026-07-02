import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import logging
from fastapi import FastAPI, Request, Header, HTTPException, Depends # pyright: ignore [reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware # pyright: ignore [reportMissingImports]
from fastapi.responses import StreamingResponse # pyright: ignore [reportMissingImports]
from pydantic import BaseModel # pyright: ignore [reportMissingImports]
from typing import Optional
from supabase import create_client, Client, ClientOptions # pyright: ignore [reportMissingImports]
from config import settings # pyright: ignore [reportMissingImports]
from providers.factory import ProviderFactory # pyright: ignore [reportMissingImports]

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AetherChat API Portal")

# Allow CORS for dev and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    provider: str
    sessionId: str

async def get_supabase_client(authorization: Optional[str] = Header(None)) -> Client:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header is required.")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format. Must be Bearer <token>")
        
    jwt_token = authorization.split(" ")[1]
    
    # Instantiate client with user JWT to preserve Row-Level Security (RLS)
    options = ClientOptions(headers={"Authorization": f"Bearer {jwt_token}"})
    return create_client(settings.supabase_url, settings.supabase_anon_key, options=options)

@app.post("/api/chat")
async def chat_endpoint(chat_req: ChatRequest, request: Request, supabase: Client = Depends(get_supabase_client)):
    # 1. Fetch conversation history for session
    try:
        db_res = supabase.table("messages").select("role, content").eq("session_id", chat_req.sessionId).order("created_at").execute()
        history = db_res.data
    except Exception as db_err:
        logger.error(f"Database error fetching messages: {db_err}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(db_err)}")
        
    # 2. Instantiate provider via factory
    try:
        provider_instance = ProviderFactory.get_provider(chat_req.provider)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as init_err:
        logger.error(f"Failed to initialize provider {chat_req.provider}: {init_err}")
        raise HTTPException(status_code=500, detail=f"AI Provider Initialization Error: {str(init_err)}")

    # 3. Define the generator to stream response
    async def sse_generator():
        full_text = ""
        try:
            async for chunk in provider_instance.stream_response(history):
                full_text += chunk
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as stream_err:
            logger.error(f"Error during stream generation: {stream_err}")
            yield f"data: {json.dumps({'error': str(stream_err)})}\n\n"
        finally:
            # Save complete response to database in background
            if full_text.strip():
                try:
                    supabase.table("messages").insert({
                        "session_id": chat_req.sessionId,
                        "role": "assistant",
                        "content": full_text,
                        "provider_used": chat_req.provider
                    }).execute()
                    logger.info(f"Successfully saved assistant response for session {chat_req.sessionId}")
                except Exception as save_err:
                    logger.error(f"Failed to save assistant message to DB: {save_err}")

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
