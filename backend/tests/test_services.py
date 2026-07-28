import pytest
import sys
import os
from unittest.mock import MagicMock, AsyncMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.chat.schemas import Message
from api.chat.services import ChatService, SYSTEM_PROMPT
from providers.mock import MockProvider

@pytest.mark.asyncio
async def test_system_prompt_insertion_empty_history():
    history = []
    provider = MockProvider()
    mock_supabase = MagicMock()
    
    async def request_is_disconnected():
        return False

    chunks = []
    async for chunk in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="mock",
        request_is_disconnected=request_is_disconnected,
        use_web_search=False
    ):
        chunks.append(chunk)

    assert len(history) > 0
    assert history[0].role == "system"
    assert history[0].content == SYSTEM_PROMPT.content
    assert "TOOL RESTRICTION" in history[0].content

@pytest.mark.asyncio
async def test_system_prompt_insertion_existing_history():
    history = [Message(role="user", content="Hello")]
    provider = MockProvider()
    mock_supabase = MagicMock()
    
    async def request_is_disconnected():
        return False

    chunks = []
    async for chunk in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="mock",
        request_is_disconnected=request_is_disconnected,
        use_web_search=False
    ):
        chunks.append(chunk)

    assert history[0].role == "system"
    assert history[0].content == SYSTEM_PROMPT.content
    assert history[1].role == "user"

@pytest.mark.asyncio
async def test_stream_chat_scopes_rag_to_uploaded_document(monkeypatch):
    """
    When a file is attached in the same request as the message, retrieval
    must be scoped to that document_name so a generic prompt like "summarize
    the pdf" is grounded in the file just uploaded rather than an older
    document that happens to rank higher on raw embedding similarity.
    """
    history = [Message(role="user", content="summarize the pdf")]
    provider = MockProvider()
    mock_supabase = MagicMock()

    captured_kwargs = {}

    async def fake_get_relevant_context(supabase, query, user_id, top_k=3, document_name=None):
        captured_kwargs["document_name"] = document_name
        return []

    monkeypatch.setattr("api.chat.services.get_relevant_context", fake_get_relevant_context)

    async def request_is_disconnected():
        return False

    async for _ in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="mock",
        request_is_disconnected=request_is_disconnected,
        use_web_search=False,
        user_id="user-123",
        scoped_document_name="IQRA HAMEED_CV.pdf"
    ):
        pass

    assert captured_kwargs["document_name"] == "IQRA HAMEED_CV.pdf"

@pytest.mark.asyncio
async def test_stream_chat_no_file_searches_all_documents(monkeypatch):
    """
    Without a file attached in the request, retrieval must fall back to
    searching across all of the user's documents (document_name=None).
    """
    history = [Message(role="user", content="what did we discuss earlier?")]
    provider = MockProvider()
    mock_supabase = MagicMock()

    captured_kwargs = {}

    async def fake_get_relevant_context(supabase, query, user_id, top_k=3, document_name=None):
        captured_kwargs["document_name"] = document_name
        return []

    monkeypatch.setattr("api.chat.services.get_relevant_context", fake_get_relevant_context)

    async def request_is_disconnected():
        return False

    async for _ in ChatService.stream_chat(
        provider_instance=provider,
        history=history,
        supabase=mock_supabase,
        session_id="test_sess",
        provider_name="mock",
        request_is_disconnected=request_is_disconnected,
        use_web_search=False,
        user_id="user-123"
    ):
        pass

    assert captured_kwargs["document_name"] is None
