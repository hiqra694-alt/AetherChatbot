from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """
    Response returned after a PDF has been parsed, chunked, embedded, and stored.
    """
    document_name: str = Field(..., description="Original filename of the uploaded PDF")
    chunks_stored: int = Field(..., description="Number of chunks embedded and inserted into document_chunks")
    message: str = Field(..., description="Human-readable status message")


class DocumentMetadata(BaseModel):
    """
    Summary of a single processed document, aggregated from its stored chunks.
    """
    document_name: str = Field(..., description="Original filename of the document")
    chunk_count: int = Field(..., description="Number of chunks stored for this document")
    created_at: datetime = Field(..., description="Timestamp of the earliest stored chunk for this document")


class DocumentListResponse(BaseModel):
    """
    Payload for GET / — the authenticated user's processed documents.
    """
    documents: List[DocumentMetadata] = Field(default_factory=list)


class DocumentDeleteResponse(BaseModel):
    """
    Payload for DELETE /{document_name} — confirms how many chunks were removed.
    """
    document_name: str = Field(..., description="Name of the deleted document")
    chunks_deleted: int = Field(..., description="Number of chunks removed from document_chunks")


class RetrievedChunk(BaseModel):
    """
    A single chunk returned by a similarity search, ready to be injected into
    a RAG prompt by the chat module.
    """
    document_name: str = Field(..., description="Filename of the source document")
    chunk_text: str = Field(..., description="The retrieved chunk text")
    similarity: float = Field(..., description="Cosine similarity score against the query embedding (higher is more relevant)")
