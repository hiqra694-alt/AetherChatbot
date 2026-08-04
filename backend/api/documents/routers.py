import logging
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from supabase import create_client, Client, ClientOptions

from core.config import get_settings
from api.documents.schemas import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentUploadResponse,
)
from api.documents.services import list_user_documents, process_and_store_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_CONTENT_TYPE = "application/pdf"


async def get_authenticated_supabase(authorization: Optional[str] = Header(None)) -> Tuple[Client, str]:
    """
    Dependency mirroring api/chat/routers.py's get_supabase_client: builds a
    Supabase client bound to the caller's JWT so RLS policies on
    document_chunks apply, then resolves user_id from that verified token
    instead of trusting a client-supplied value.
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


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    auth: Tuple[Client, str] = Depends(get_authenticated_supabase),
):
    """
    Accepts a PDF upload and runs it through process_and_store_pdf, storing
    the resulting chunks scoped to the authenticated user.
    """
    supabase, user_id = auth

    if file.content_type != ALLOWED_CONTENT_TYPE:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        chunks_stored = await process_and_store_pdf(supabase, file_bytes, file.filename, user_id)
    except Exception as processing_err:
        logger.error(f"Failed to process document '{file.filename}' for user {user_id}: {processing_err}")
        raise HTTPException(status_code=500, detail="Failed to process document. Please try again later.")

    message = (
        f"Document '{file.filename}' processed and stored successfully."
        if chunks_stored
        else "Document processed but no extractable text was found."
    )
    return DocumentUploadResponse(document_name=file.filename, chunks_stored=chunks_stored, message=message)


@router.get("", response_model=DocumentListResponse)
async def list_documents(auth: Tuple[Client, str] = Depends(get_authenticated_supabase)):
    """
    Lists the authenticated user's processed documents, aggregated from
    their stored chunks. RLS on document_chunks already restricts rows to
    this user, so no additional filtering is required here.
    """
    supabase, user_id = auth

    try:
        documents = await list_user_documents(supabase, user_id)
    except Exception as db_err:
        logger.error(f"Failed to list documents for user {user_id}: {db_err}")
        raise HTTPException(status_code=500, detail="Failed to fetch documents. Please try again later.")

    return DocumentListResponse(documents=documents)


@router.delete("/{document_name}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_name: str,
    auth: Tuple[Client, str] = Depends(get_authenticated_supabase),
):
    """
    Deletes every stored chunk for `document_name` belonging to the
    authenticated user, revoking the AI's access to that document. RLS on
    document_chunks already restricts deletes to the caller's own rows, but
    the user_id filter is kept explicit here to match list_documents/upload.
    """
    supabase, user_id = auth

    try:
        res = (
            supabase.table("document_chunks")
            .delete()
            .eq("user_id", user_id)
            .eq("document_name", document_name)
            .execute()
        )
    except Exception as db_err:
        logger.error(f"Failed to delete document '{document_name}' for user {user_id}: {db_err}")
        raise HTTPException(status_code=500, detail="Failed to delete document. Please try again later.")

    chunks_deleted = len(res.data or [])
    if chunks_deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found.")

    return DocumentDeleteResponse(document_name=document_name, chunks_deleted=chunks_deleted)
