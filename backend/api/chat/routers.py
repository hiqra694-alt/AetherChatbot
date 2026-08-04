import json
import logging
from typing import Optional, Tuple
from fastapi import APIRouter, Request, Header, HTTPException, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from supabase import create_client, Client, ClientOptions

from core.config import get_settings
from api.chat.schemas import Message, ProviderEnum
from api.chat.services import ChatService
from api.documents.services import process_and_store_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

ALLOWED_UPLOAD_CONTENT_TYPE = "application/pdf"

async def get_authenticated_supabase(authorization: Optional[str] = Header(None)) -> Tuple[Client, str]:
    """
    Dependency to instantiate a Supabase client authenticated with the user's JWT
    and resolve the caller's user_id from that verified token (mirrors
    api/documents/routers.py's dependency of the same shape, so document
    retrieval in ChatService can be scoped to the same tenant boundary as
    document_chunks' RLS policies). Preserves Row-Level Security (RLS)
    policies defined in the database.
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

@router.post("")
async def chat_endpoint(
    request: Request,
    provider: ProviderEnum = Form(...),
    sessionId: str = Form(...),
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    auth: Tuple[Client, str] = Depends(get_authenticated_supabase),
):
    """
    Unified multipart chat endpoint. Accepts an optional PDF upload and/or an
    optional text message in a single request:
      - file only  -> ingest the document, return an acknowledgement (no LLM call)
      - message present -> run standard generation; if a file was attached in
        the same request, it has already been embedded and stored, so
        get_relevant_context() (called inside ChatService.stream_chat) picks
        it up immediately.
    """
    supabase, user_id = auth

    has_message = bool(message and message.strip())
    if file is None and not has_message:
        raise HTTPException(status_code=400, detail="Either a file or a message must be provided.")

    chunks_stored = 0
    if file is not None:
        if file.content_type != ALLOWED_UPLOAD_CONTENT_TYPE:
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            # Must finish before generation starts so the newly stored chunks
            # are visible to get_relevant_context() later in this request.
            chunks_stored = await process_and_store_pdf(supabase, file_bytes, file.filename, user_id)
        except Exception as processing_err:
            logger.error(f"Failed to process uploaded document '{file.filename}' for user {user_id}: {processing_err}")
            raise HTTPException(status_code=500, detail="Failed to process the uploaded document. Please try again later.")

    history = await ChatService.get_history(supabase, sessionId)

    if file is not None and not has_message:
        # Scenario A: file-only upload, no question yet — acknowledge and stop, no LLM call.
        if chunks_stored:
            ack_text = f"I have successfully processed your document '{file.filename}'. What would you like to know about it?"
        else:
            ack_text = f"I received '{file.filename}', but couldn't find any readable text in it. Could you try a different file?"

        ChatService.persist_message(supabase, sessionId, "user", f"[Uploaded document: {file.filename}]")
        ChatService.persist_message(supabase, sessionId, "assistant", ack_text, provider.value)

        async def ack_stream():
            yield f"data: {json.dumps({'content': ack_text})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(ack_stream(), media_type="text/event-stream")

    # Scenario B: a message is present (file attached or not) -> standard generation.
    ChatService.persist_message(supabase, sessionId, "user", message)
    history.append(Message(role="user", content=message))

    provider_instance = ChatService.get_provider(provider)

    generator = ChatService.stream_chat(
        provider_instance=provider_instance,
        history=history,
        supabase=supabase,
        session_id=sessionId,
        provider_name=provider.value,
        request_is_disconnected=request.is_disconnected,
        user_id=user_id,
        # Scope RAG retrieval to the file just attached in this request (if
        # any), so a generic prompt like "summarize the pdf" is grounded in
        # the newly uploaded document rather than an older one that happens
        # to score higher on raw embedding similarity.
        scoped_document_name=file.filename if file is not None else None
    )

    return StreamingResponse(generator, media_type="text/event-stream")
